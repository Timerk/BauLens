"""Narrative mode check: the model's text must contain no numbers and no findings.

In narrative mode code renders every fact ("Befunde"); the model writes only an assessment,
limitations and recommendations. Those texts are checked, failing closed:

1. Unicode hyphens, minus signs and spaces are normalised and zero-width characters removed
   first, so typographic variants cannot hide anything.
2. **Zero numerals** of any kind (:mod:`baulens.report.numerals`): digits of any script,
   numeric symbols, Roman numerals, number words, ordinals, fractions, multipliers and
   numeric compounds ("zwölfprozentig", "Fünfzehntel", "dreistufig").
3. **No braces** (no placeholders or template syntax).
4. **No restated findings or comparisons** (:data:`FINDINGS`): crack decisions, threshold
   comparisons, training-domain statements, statistical terms, units and pass/fail verdicts
   belong to the code-rendered findings only.

What is *not* checked: whether the narrative's prose is true or sensible.
"""

import re
from collections.abc import Mapping

from baulens.report.numerals import find_numerals, normalize_typography
from baulens.report.schemas import FaithfulnessResult, NumberCheck, SectionName

FINDINGS: tuple[tuple[str, str], ...] = (
    ("Rissbefund", r"\b(riss|risse|risses|rissen)\b|rissfrei\w*|rissbehaftet\w*|gerissen\w*"),
    ("Rissbefund", r"rissbildung\w*|rissbreite\w*|risstiefe\w*|rissverlauf\w*|rissöffnung\w*"),
    ("Befund", r"\b(erkannt|festgestellt|detektiert|nachgewiesen)\b"),
    ("Schwellenvergleich", r"schwelle\w*|grenzwert\w*"),
    (
        "Vergleich",
        r"überschritten|unterschritten|überschreit\w*|unterschreit\w*|\berreicht\w*"
        r"|\bnahe\s+(an|bei)\b|knapp\s+(über|unter)\b|oberhalb|unterhalb"
        r"|\b(höher|niedriger|größer|kleiner|geringer|stärker|schwächer)\s+als\b",
    ),
    (
        "Trainingsbereich",
        r"trainingsbereich\w*|trainingsdaten\w*|wertebereich\w*|datenbereich\w*|innerhalb"
        r"|außerhalb|ausserhalb|extrapol\w*",
    ),
    (
        "Statistik",
        r"intervall\w*|konfidenz\w*|wahrscheinlich\w*|überdeckung\w*|prognosewert\w*"
        r"|schätzwert\w*|signifikan\w*|streuung\w*|standardabweichung\w*",
    ),
    ("Einheit", r"\bmpa\b|n/mm|prozent\w*|promille\w*|%|‰|kg/m|\bpixel\w*|\bpx\b|\bkg\b"),
    ("Klasse", r"festigkeitsklasse\w*|expositionsklasse\w*|betonklasse\w*"),
    (
        "Bewertung eines Befunds",
        r"ausreichend\w*|unzureichend\w*|\bgenügend|ungenügend|bestanden|\bkonform\w*"
        r"|\berfüllt\b",
    ),
)
"""Vocabulary that states or compares a finding. Allowed, because they only *refer* to the
findings: "Rissbefund", "Rissklassifikation", "Druckfestigkeit", "Prognose",
"Modellschätzung", "Befunde"."""

_PATTERNS = tuple((label, re.compile(pattern, re.IGNORECASE)) for label, pattern in FINDINGS)
_BRACES = re.compile(r"[{}]")


def _check(section: str, start: int, end: int, text: str, kind: str, reason: str) -> NumberCheck:
    return NumberCheck(
        section=section,
        start=start,
        end=end,
        text=text,
        kind="numeral" if kind == "numeral" else "brace" if kind == "brace" else "finding",
        value=text,
        status="invented" if kind in {"numeral", "brace"} else "restated",
        reason=reason,
    )


def check_narrative_text(text: str, *, section: str = "text") -> FaithfulnessResult:
    """All violations in one narrative text (positions refer to the normalised text)."""
    text = normalize_typography(text)
    checks = [
        _check(
            section,
            n.start,
            n.end,
            n.text,
            "numeral",
            "Zahl im Freitext; Zahlen stehen nur in den Befunden",
        )
        for n in find_numerals(text)
    ]
    checks += [
        _check(section, m.start(), m.end(), m.group(), "brace", "geschweifte Klammer im Freitext")
        for m in _BRACES.finditer(text)
    ]
    for label, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            reason = f"{label} im Freitext; Befunde und Vergleiche stehen nur in den Befunden"
            checks.append(
                _check(section, match.start(), match.end(), match.group(), "finding", reason)
            )
    checks.sort(key=lambda check: check.start)
    return FaithfulnessResult(checks=checks)


def check_narrative(sections: Mapping[SectionName, str]) -> FaithfulnessResult:
    """Check every narrative section and merge the results."""
    checks: list[NumberCheck] = []
    for name, text in sections.items():
        checks += check_narrative_text(text, section=name).checks
    return FaithfulnessResult(checks=checks)
