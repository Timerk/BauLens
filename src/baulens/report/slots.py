"""Strict slot mode: code renders every quantity, the checker guards how slots are used.

What is enforced, and nothing more (see ``docs/results/report.md``):

1. **Grammar.** A slot is exactly ``{name}`` with a lowercase name from the case's slot list.
   Markdown emphasis directly around a slot (``**{x}**``, ``*{x}*``, ``__{x}__``) is removed
   before all checks and is not delivered. Any other brace or backslash (``{X}``, ``{x:y}``,
   ``{{x}}``, ``\\{x}``) and every unknown name is a violation.
2. **Code owns the quantity.** A slot renders sign, value and unit (``{druckfestigkeit}`` →
   ``61,9 MPa``). A sign directly before a slot or a unit, ``%``, "Prozent", "-Punkte" or
   "Uhr" directly after it is a violation.
3. **No numbers outside slots.** The draft text outside slots must contain no numeral at all
   (:mod:`baulens.report.numerals`): no digit of any script, no number word, ordinal,
   fraction, multiplier or Roman numeral. Identifiers, model names and the two citable
   standards are slots too, so nothing is ever masked.
4. **Role binding.** A numeric slot's clause must not name a different quantity of the same
   unit; if the clause names none, its sentence must contain the slot's own vocabulary
   (e.g. "Alter" for ``{betonalter}``, "Schwelle" for ``{entscheidungsschwelle}``). Slots of
   an ingredient or a feature contribution must be in a clause that names that ingredient or
   feature. A lower bound must not follow the upper bound in the same sentence.
5. **Forbidden contexts.** No slot may share a sentence with deadline vocabulary ("Frist",
   "Termin", …) or a clause with a negation ("nicht", "kein", …). The capture date and time
   must be in a sentence about the image capture.
6. **Mandatory facts.** A strength draft must use the point estimate and both interval bounds;
   a crack draft must state the crack finding explicitly and consistently with the label.

Semantic claims beyond these rules (e.g. a recommendation that makes no sense) are not
verified; the disclaimer and an engineer's review cover them.
"""

import re
from collections.abc import Mapping

from baulens.report.completeness import crack_finding_consistent
from baulens.report.facts import Fact, FactSheet
from baulens.report.faithfulness import ROLE_KEYWORDS, role_name
from baulens.report.lexicon import CAPTURE, DEADLINE, NEGATION, clause_around, sentence_around
from baulens.report.numerals import find_numerals
from baulens.report.schemas import CheckStatus, FaithfulnessResult, NumberCheck, SectionName

SLOT = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_EMPHASIS = re.compile(r"(\*\*|\*|__|_)(\{[a-z][a-z0-9_]*\})\1")
_RANGE = re.compile(
    r"\{untere_grenze\}\s*(?:bis|und|\u2013|-|\u2014)\s*\{obere_grenze\}", re.IGNORECASE
)
_RESIDUE = re.compile(r"[{}\\]")
_SIGN_BEFORE = re.compile(r"[+\-\u2212\u2013±][ \u00a0]?$")
_UNIT_AFTER = re.compile(
    r"^[ \u00a0]?-?[ \u00a0]?(%|‰|°|Prozent\w*|Punkt\w*|MPa|N/mm\S*|kg\S*|t/m\S*|g/\S*|Tag\w*"
    r"|tägig\w*|d\b|Pixel\w*|px\b|Uhr\b|Jahr\w*|Monat\w*|Woche\w*|Stunde\w*|Minute\w*|mm\b|cm\b"
    r"|m\b|m³|m²)"
)
_ROLE_PATTERNS = tuple(
    (role, re.compile(r"(?<![\w])(?:" + pattern + r")", re.IGNORECASE))
    for role, pattern in ROLE_KEYWORDS
)
_OWN_WORDS: dict[str, str] = {
    "point": r"druckfestigkeit|prognos|schätz|festigkeit",
    "lower": r"intervall|grenze|bereich|spanne|zwischen",
    "upper": r"intervall|grenze|bereich|spanne|zwischen",
    "coverage": r"überdeckung|ueberdeckung|abdeckung|konfidenz|vertrauens",
    "probability": r"wahrscheinlich",
    "threshold": r"schwelle|grenzwert",
    "age": r"alter",
    "ratio": r"w/z|wasserzement|wasser-zement",
    "contribution": (
        r"beitr[aä]g|einfluss|erhöh|verringer|minder|steiger|senk|rückgang|zunahme|abnahme"
        r"|anstieg|shap"
    ),
    "pixels": r"bild|aufnahme|auflösung|ausschnitt|format",
}
_SAME_UNIT_ROLES: dict[str, frozenset[str]] = {
    "MPa": frozenset({"point", "lower", "upper", "contribution", "interval"}),
    "%": frozenset({"coverage", "probability", "threshold", "interval"}),
    "d": frozenset({"age"}),
}


def normalize(text: str) -> str:
    """Remove Markdown emphasis that directly wraps a slot."""
    return _EMPHASIS.sub(r"\2", text)


def render_slots(text: str, sheet: FactSheet) -> str:
    """Insert the rendered quantities; unknown slots stay visible (and are violations)."""
    slots = sheet.slots()
    return SLOT.sub(lambda match: slots.get(match[1], match[0]), normalize(text))


def _check(
    section: str, start: int, end: int, text: str, status: CheckStatus, reason: str,
    fields: list[str] | None = None, kind: str = "slot",
) -> NumberCheck:  # fmt: skip
    return NumberCheck(
        section=section,
        start=start,
        end=end,
        text=text,
        kind="slot" if kind == "slot" else "numeral",
        value=text,
        status=status,
        matched_fields=fields or [],
        reason=reason,
    )


def _clause_roles(clause: str, unit: str | None) -> set[str]:
    relevant = _SAME_UNIT_ROLES.get(unit or "", frozenset())
    named: set[str] = set()
    claimed: list[tuple[int, int]] = []
    for role, pattern in _ROLE_PATTERNS:
        for match in pattern.finditer(clause):
            if any(match.start() < end and start < match.end() for start, end in claimed):
                continue
            claimed.append(match.span())
            if role in relevant:
                if role == "interval":
                    named |= {"lower", "upper"} if unit == "MPa" else {"coverage"}
                else:
                    named.add(role)
    return named


def _check_fact_slot(
    fact: Fact, text: str, start: int, end: int, in_range: bool
) -> tuple[CheckStatus, str]:
    clause = clause_around(text, start, end)
    sentence = sentence_around(text, start, end)
    if DEADLINE.search(sentence):
        return "misattributed", "Platzhalter in einem Satz über Fristen, Termine oder Dauer"
    if NEGATION.search(clause) or (in_range and NEGATION.search(sentence)):
        return "misattributed", "Platzhalter in einem verneinten Satzteil"
    if in_range:
        # "{untere_grenze} bis {obere_grenze}": the bounds are bound by their order.
        return "ok", f"Platzhalter für {fact.label} (Intervallangabe)"
    named = _clause_roles(clause, fact.unit)
    if named and fact.role not in named:
        others = " oder ".join(sorted(role_name(role) for role in named))
        return "misattributed", f"Platzhalter für {fact.label} in einem Satzteil über {others}"
    if not named:
        own = _OWN_WORDS.get(fact.role)
        if own is not None and not re.search(own, sentence, re.IGNORECASE):
            return "unbound", f"der Satz nennt die Größe nicht ({fact.label})"
    if fact.terms and not re.search(fact.terms, clause, re.IGNORECASE):
        return "misattributed", f"Platzhalter für {fact.label}, aber der Satzteil nennt sie nicht"
    return "ok", f"Platzhalter für {fact.label}"


def check_slot_text(text: str, sheet: FactSheet, *, section: str = "text") -> FaithfulnessResult:
    """Check one raw slot-mode section (before rendering)."""
    text = normalize(text)
    facts = {fact.slot: fact for fact in sheet.facts}
    texts = {slot.slot for slot in sheet.text_slots}
    checks: list[NumberCheck] = []
    # Context checks must not read words inside slot names ("{beitrag_zement}" is no
    # mention of "Zement"), so every slot is blanked with the same length.
    blanked = SLOT.sub(lambda match: "§" * len(match[0]), text)
    positions: dict[str, list[int]] = {}
    ranged = {
        position
        for match in _RANGE.finditer(text)
        for position in (match.start(), match.end() - 14)
    }
    for match in SLOT.finditer(text):
        name, start, end = match[1], match.start(), match.end()
        if name not in facts and name not in texts:
            checks.append(
                _check(section, start, end, match[0], "invented", "unbekannter Platzhalter")
            )
            continue
        if _SIGN_BEFORE.search(text[:start]) or _UNIT_AFTER.match(text[end:]):
            reason = "Vorzeichen oder Einheit neben dem Platzhalter; der Platzhalter enthält sie"
            checks.append(_check(section, start, end, match[0], "misattributed", reason))
            continue
        positions.setdefault(name, []).append(start)
        if name in facts:
            fact = facts[name]
            verdict, reason = _check_fact_slot(fact, blanked, start, end, start in ranged)
            checks.append(_check(section, start, end, match[0], verdict, reason, [fact.field]))
        elif name in {"aufnahmedatum", "aufnahmezeit"}:
            sentence = sentence_around(blanked, start, end)
            ok = CAPTURE.search(sentence) and not DEADLINE.search(sentence)
            reason = "Aufnahmezeitpunkt" if ok else "Aufnahmezeitpunkt nicht als Aufnahme genannt"
            capture: CheckStatus = "ok" if ok else "misattributed"
            checks.append(
                _check(section, start, end, match[0], capture, reason, ["image.captured_at"])
            )
        else:
            checks.append(_check(section, start, end, match[0], "ok", "Platzhalter"))
    for lower in positions.get("untere_grenze", []):
        for upper in positions.get("obere_grenze", []):
            same = sentence_around(blanked, lower, lower + 1) == sentence_around(
                blanked, upper, upper + 1
            )
            if same and upper < lower:
                reason = "obere Grenze vor der unteren Grenze im selben Satz"
                checks.append(
                    _check(section, upper, upper + 1, "{obere_grenze}", "misattributed", reason)
                )
    for residue in _RESIDUE.finditer(blanked):
        reason = "fehlerhafter Platzhalter (nur {name} in Kleinbuchstaben ist zulässig)"
        checks.append(
            _check(section, residue.start(), residue.end(), residue.group(), "invented", reason)
        )
    for numeral in find_numerals(blanked):
        reason = "Zahl außerhalb eines Platzhalters"
        checks.append(
            _check(
                section,
                numeral.start,
                numeral.end,
                numeral.text,
                "invented",
                reason,
                kind="numeral",
            )
        )
    checks.sort(key=lambda check: check.start)
    return FaithfulnessResult(checks=checks)


def _missing(name: str, reason: str) -> NumberCheck:
    return NumberCheck(
        section="bericht", start=0, end=0, text=name, kind="slot", status="missing", reason=reason
    )


def check_slot_sections(raw: Mapping[SectionName, str], sheet: FactSheet) -> FaithfulnessResult:
    """Check all raw sections and the mandatory facts of a slot-mode draft."""
    checks: list[NumberCheck] = []
    for name, text in raw.items():
        checks += check_slot_text(text, sheet, section=name).checks
    used = {
        check.text.strip("{}") for check in checks if check.kind == "slot" and check.status == "ok"
    }
    case = sheet.case
    if case.strength is not None:
        for slot in ("druckfestigkeit", "untere_grenze", "obere_grenze"):
            if slot not in used:
                checks.append(_missing(f"{{{slot}}}", "Pflichtangabe fehlt"))
    if case.crack is not None:
        finding = render_slots(raw.get("rissbefund", ""), sheet)
        if not crack_finding_consistent(finding, case.crack.label):
            reason = "Rissbefund nicht ausdrücklich oder nicht passend zur Klassifikation genannt"
            checks.append(_missing("Rissbefund", reason))
    return FaithfulnessResult(checks=checks)
