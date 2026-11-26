"""Numeric faithfulness: every number must be a fact, in its exact form, bound to its role.

A numeric token is ``ok`` only if all of the following hold:

1. **Exact form.** Its digits and sign equal ``Fact.number`` of some fact (German format;
   ``\u2212``/``-`` and the thousands grouping ``1040``/``1.040``/``1 040`` are equivalent).
2. **Unit.** The unit written right after it is the fact's unit (MPa, kg/m³, Tage, %, Pixel).
   A missing unit on a unit-bearing fact, or any other unit (``t/m³``, ``kg``, ``Jahre``),
   is a violation. Only the first bound of a range may borrow the unit of the second.
3. **Role.** The words before the number (back to the previous number or the sentence start)
   name the fact's role, see :data:`ROLE_KEYWORDS`. If they name nothing relevant to the
   number's unit, the words after it (up to the next comma, bracket, number or sentence end)
   are used. Ranges ("54,2 bis 69,6 MPa", "zwischen … und …") bind the first number to the
   lower and the second to the upper bound. Roles that the unit alone determines (age in
   days, image pixels) need no keyword. Anything else without a relevant keyword is
   ``unbound``: the checker fails closed on ambiguity.

Verdicts: ``invented`` (no fact has this value; also foreign dates, times, references and
codes such as ``C50/60``), ``misattributed`` (a fact's value with the wrong unit, sign or
role, or reversed bounds), ``inexact`` (a fact's value in another form: rounded, extra or
missing digits, English decimal point) and ``unbound`` (a fact's value whose role the text
does not state).

The checker verifies numbers, not qualitative claims; see ``docs/results/report.md``.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from baulens.report.facts import MIX_LABELS, Fact, FactSheet
from baulens.report.numbers import (
    ALLOWED_REFERENCES,
    Extraction,
    NumberToken,
    SpecialToken,
    extract,
)
from baulens.report.schemas import (
    CheckStatus,
    FaithfulnessResult,
    MaskedText,
    NumberCheck,
    SectionName,
)

CHECKER_VERSION = "3"
"""Bumped whenever a rule changes; resumed runs must match it."""

ROLE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("lower", r"unter\w*\s+grenze\w*|untergrenze\w*|unter(?:e|en|er|es)\b|mindestens"),
    ("upper", r"ober\w*\s+grenze\w*|obergrenze\w*|ober(?:e|en|er|es)\b|höchstens|maximal\w*"),
    (
        "contribution",
        r"\w*beitr(?:a|ä)g\w*|einfluss\w*|erhöh\w*|verringer\w*|minder\w*|steiger\w*|senk\w*|shap"
        r"|rückgang\w*|zunahme\w*|abnahme\w*|anstieg\w*",
    ),
    ("ratio", r"w/z\S*|wasserzement\w*|wasser-zement\w*"),
    ("threshold", r"\w*schwelle\w*|grenzwert\w*"),
    ("coverage", r"\w*überdeckung\w*|\w*ueberdeckung\w*|abdeckung\w*|konfidenz\w*|vertrauens\w*"),
    ("interval", r"\w*intervall\w*|spanne\b|bandbreite\w*"),
    ("probability", r"\w*wahrscheinlich\w*"),
    (
        "point",
        r"druckfestigkeit\w*|prognosewert\w*|schätzwert\w*|punktschätz\w*|prognostiziert\w*"
        r"|geschätzt\w*|prognose\b",
    ),
    ("ingredient:blast_furnace_slag", r"hüttensand\w*|schlacke\w*"),
    ("ingredient:fly_ash", r"flugasche\w*"),
    ("ingredient:superplasticizer", r"fließmittel\w*|fliessmittel\w*"),
    ("ingredient:cement", r"zement\w*"),
    ("ingredient:water", r"wasser\w*"),
    ("ingredient:coarse_aggregate", r"grob\w*|kies\w*"),
    ("ingredient:fine_aggregate", r"fein\w*|sand\w*"),
)
"""Keyword patterns per role in priority order. Within one window the most specific role
wins ("obere Grenze der Druckfestigkeit" names the upper bound); among equally specific
keywords the one closest to the number wins. Earlier patterns claim their span first, so
"Wasserzementwert" is the w/z ratio and "Hüttensand" is not fine aggregate."""

_KEYWORD_PATTERNS = tuple(
    (role, re.compile(r"(?<![\w])(?:" + pattern + r")", re.IGNORECASE))
    for role, pattern in ROLE_KEYWORDS
)
_PRIORITY = {role: index for index, (role, _) in enumerate(ROLE_KEYWORDS)}
UNIT_ROLES = frozenset({"age", "pixels"})
"""Roles that the unit alone determines: only one quantity per case is in days or pixels."""

ROLE_NAMES: dict[str, str] = {
    "point": "Prognosewert der Druckfestigkeit",
    "lower": "untere Grenze",
    "upper": "obere Grenze",
    "coverage": "Überdeckung des Prognoseintervalls",
    "probability": "Risswahrscheinlichkeit",
    "threshold": "Entscheidungsschwelle",
    "contribution": "Beitrag einer Einflussgröße",
    "ratio": "w/z-Wert",
    "age": "Betonalter",
    "pixels": "Bildgröße",
}


def role_name(role: str) -> str:
    if role.startswith("ingredient:"):
        return MIX_LABELS.get(role.split(":", 1)[1], role)
    return ROLE_NAMES.get(role, role)


@dataclass(frozen=True)
class _Keyword:
    role: str
    start: int
    end: int


def _keywords(text: str) -> list[_Keyword]:
    found: list[_Keyword] = []
    for role, pattern in _KEYWORD_PATTERNS:
        for match in pattern.finditer(text):
            if not any(match.start() < kw.end and kw.start < match.end() for kw in found):
                found.append(_Keyword(role, match.start(), match.end()))
    return found


def _relevant(role: str, unit: str | None) -> bool:
    """Whether a keyword can describe a number with this unit."""
    if role == "interval":
        return unit in {"MPa", "%"}
    if role in {"point", "lower", "upper", "contribution"}:
        return unit == "MPa"
    if role in {"coverage", "probability", "threshold"}:
        return unit == "%"
    if role.startswith("ingredient:"):
        return unit == "kg/m³"
    if role == "ratio":
        return unit is None
    return False


def stated_roles(token: NumberToken) -> tuple[set[str], str] | None:
    """Roles the text assigns to a number and the keyword that did it; ``None`` if none."""
    if token.range_role is not None:
        return {token.range_role}, "Intervallangabe von … bis …"
    for window, before in ((token.before, True), (token.after, False), (token.far, True)):
        keywords = [kw for kw in _keywords(window) if _relevant(kw.role, token.unit)]
        if not keywords:
            continue
        best = min(_PRIORITY[kw.role] for kw in keywords)
        winners = [kw for kw in keywords if _PRIORITY[kw.role] == best]
        keyword = (
            max(winners, key=lambda kw: kw.end) if before else min(winners, key=lambda kw: kw.start)
        )
        roles = {keyword.role}
        if keyword.role == "interval":
            roles = {"lower", "upper"} if token.unit == "MPa" else {"coverage"}
        return roles, window[keyword.start : keyword.end]
    return None


def _fact_sign(fact: Fact) -> str | None:
    if fact.number.startswith(("\u2212", "-")):
        return "-"
    return "+" if fact.number.startswith("+") else None


def _fact_digits(fact: Fact) -> str:
    return fact.number.lstrip("+-\u2212")


def _exact(token: NumberToken, fact: Fact) -> bool:
    if token.kind == "word":
        return fact.decimals == 0 and not fact.signed and fact.value == token.value
    return (
        not token.foreign and token.digits == _fact_digits(fact) and token.sign == _fact_sign(fact)
    )


def _decimals(digits: str) -> int:
    return len(digits.partition(",")[2])


def _near(token: NumberToken, fact: Fact) -> bool:
    """The same quantity in another form: rounded, padded or with an English decimal point."""
    finest = max(fact.decimals, _decimals(token.digits))
    coarsest = min(fact.decimals, _decimals(token.digits))
    tolerance = max(Decimal(1).scaleb(-finest), Decimal(5).scaleb(-(coarsest + 1)))
    same_sign = (token.value < 0) == (fact.value < 0) or fact.value == 0
    return same_sign and abs(token.value - fact.value) <= tolerance


def _number_check(
    token: NumberToken,
    status: CheckStatus,
    reason: str,
    section: str,
    fields: list[str] | None = None,
) -> NumberCheck:
    return NumberCheck(
        section=section,
        start=token.start,
        end=token.unit_end,
        text=token.text,
        kind=token.kind,
        value=str(token.value),
        unit=token.unit_text or None,
        status=status,
        matched_fields=fields or [],
        reason=reason,
    )


def _shown(fact: Fact) -> str:
    return f"{fact.label} ({fact.display})"


def _unmatched(token: NumberToken, sheet: FactSheet, section: str) -> NumberCheck:
    if token.kind == "number" and not token.foreign:
        sign_only = [fact for fact in sheet.facts if token.digits == _fact_digits(fact)]
        if sign_only:
            fact = sign_only[0]
            reason = f"Wert von {_shown(fact)}, aber mit falschem oder fehlendem Vorzeichen"
            return _number_check(token, "misattributed", reason, section, [fact.field])
    # Only digits with the fact's own unit count as another form of that fact; a bare "3"
    # next to "Bohrkerne" is not a rounding of 2,5 kg/m³.
    near = [
        fact
        for fact in sheet.facts
        if token.kind == "number" and token.unit == fact.unit and _near(token, fact)
    ]
    if near:
        fact = near[0]
        how = (
            "Englisches Zahlenformat"
            if token.foreign
            else "Nicht exakt in der Form der Faktenliste"
        )
        reason = f"{how}, gemeint ist wohl {_shown(fact)}"
        return _number_check(token, "inexact", reason, section, [fact.field])
    return _number_check(token, "invented", "nicht in der Faktenliste enthalten", section)


def _check_number(token: NumberToken, sheet: FactSheet, section: str) -> NumberCheck:
    if token.code:
        reason = "Zahl in einer Kennung (etwa einer Festigkeitsklasse), nicht in der Faktenliste"
        return _number_check(token, "invented", reason, section)
    exact = [fact for fact in sheet.facts if _exact(token, fact)]
    if not exact:
        return _unmatched(token, sheet, section)
    with_unit = [fact for fact in exact if fact.unit == token.unit]
    if not with_unit:
        fact = exact[0]
        written = token.unit_text or "ohne Einheit"
        reason = f"Wert von {_shown(fact)}, aber {written} statt {fact.unit or 'ohne Einheit'}"
        return _number_check(token, "misattributed", reason, section, [fact.field])
    stated = stated_roles(token)
    if stated is None:
        by_unit = [fact for fact in with_unit if fact.role in UNIT_ROLES]
        if by_unit:
            reason = f"exakt aus {_shown(by_unit[0])} (eindeutig über die Einheit)"
            return _number_check(token, "ok", reason, section, [f.field for f in by_unit])
        reason = f"Wert von {_shown(with_unit[0])}, aber der Text nennt die Größe nicht"
        return _number_check(token, "unbound", reason, section, [f.field for f in with_unit])
    roles, keyword = stated
    bound = [fact for fact in with_unit if fact.role in roles]
    if bound:
        reason = f"exakt aus {_shown(bound[0])} („{keyword.strip()}“)"
        return _number_check(token, "ok", reason, section, [f.field for f in bound])
    fact = with_unit[0]
    named = " oder ".join(sorted(role_name(role) for role in roles))
    reason = f"Wert von {_shown(fact)}, aber im Text als {named} bezeichnet („{keyword.strip()}“)"
    return _number_check(token, "misattributed", reason, section, [fact.field])


def _special_check(
    token: SpecialToken,
    status: CheckStatus,
    reason: str,
    section: str,
    fields: list[str] | None = None,
) -> NumberCheck:
    return NumberCheck(
        section=section,
        start=token.start,
        end=token.end,
        text=token.text,
        kind=token.kind,
        value=token.normalized,
        status=status,
        matched_fields=fields or [],
        reason=reason,
    )


def _check_special(token: SpecialToken, sheet: FactSheet, section: str) -> NumberCheck:
    if token.kind == "slot":
        if token.normalized in sheet.slots():
            return _special_check(token, "ok", "Platzhalter", section, [f"slot:{token.normalized}"])
        return _special_check(token, "invented", "unbekannter Platzhalter", section)
    if token.kind == "reference":
        allowed = ", ".join(sorted(ALLOWED_REFERENCES))
        reason = f"Normverweis nicht zulässig (zulässig: {allowed}; ohne Ausgabejahr)"
        return _special_check(token, "invented", reason, section)
    known = sheet.dates if token.kind == "date" else sheet.times
    if token.normalized in known:
        return _special_check(token, "ok", "Aufnahmezeitpunkt", section, ["image.captured_at"])
    label = "Datum" if token.kind == "date" else "Uhrzeit"
    return _special_check(token, "invented", f"{label} nicht exakt wie in der Faktenliste", section)


def _identifier_items(sheet: FactSheet) -> tuple[tuple[str, bool], ...]:
    return tuple((identifier.text, identifier.labelled) for identifier in sheet.identifiers)


def _masked(extraction: Extraction, section: str) -> list[MaskedText]:
    return [
        MaskedText(section=section, start=span.start, end=span.end, text=span.text, kind=span.kind)
        for span in extraction.masked
    ]


def check_text(text: str, sheet: FactSheet, *, section: str = "text") -> FaithfulnessResult:
    """Free-text mode: check every numeric token against the case facts."""
    extraction = extract(text, identifiers=_identifier_items(sheet))
    checks = [_check_number(token, sheet, section) for token in extraction.numbers]
    checks += [_check_special(token, sheet, section) for token in extraction.specials]
    checks.sort(key=lambda check: check.start)
    return FaithfulnessResult(checks=checks, masked=_masked(extraction, section))


def check_slot_text(text: str, sheet: FactSheet, *, section: str = "text") -> FaithfulnessResult:
    """Slot mode, raw draft: placeholders must exist and nothing numeric may be written."""
    extraction = extract(text, identifiers=_identifier_items(sheet), slots=True)
    outside = "Zahl außerhalb eines Platzhalters"
    checks = [_number_check(token, "invented", outside, section) for token in extraction.numbers]
    for token in extraction.specials:
        if token.kind in {"slot", "reference"}:
            checks.append(_check_special(token, sheet, section))
        else:
            checks.append(_special_check(token, "invented", outside, section))
    checks.sort(key=lambda check: check.start)
    return FaithfulnessResult(checks=checks, masked=_masked(extraction, section))


_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


def render_with_spans(text: str, sheet: FactSheet) -> tuple[str, list[tuple[int, int]]]:
    """Insert known placeholders; return the text and the spans of inserted values."""
    slots = sheet.slots()
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    position = length = 0
    for match in _PLACEHOLDER.finditer(text):
        parts.append(text[position : match.start()])
        length += match.start() - position
        value = slots.get(match[1], match[0])
        if match[1] in slots:
            spans.append((length, length + len(value)))
        parts.append(value)
        length += len(value)
        position = match.end()
    parts.append(text[position:])
    return "".join(parts), spans


def render_slots(text: str, sheet: FactSheet) -> str:
    """Replace known placeholders by their values; unknown ones stay visible."""
    return render_with_spans(text, sheet)[0]


def merge(results: list[FaithfulnessResult]) -> FaithfulnessResult:
    return FaithfulnessResult(
        checks=[check for result in results for check in result.checks],
        masked=[span for result in results for span in result.masked],
    )


def check_sections(sections: Mapping[SectionName, str], sheet: FactSheet) -> FaithfulnessResult:
    """Free-text mode: check each report section and merge the results."""
    return merge([check_text(text, sheet, section=name) for name, text in sections.items()])


def check_slot_sections(raw: Mapping[SectionName, str], sheet: FactSheet) -> FaithfulnessResult:
    """Slot mode: no literal numbers in the raw draft, then role binding on the rendered one.

    Rendered numbers are exact by construction, so the second pass only finds placeholders
    used in the wrong place (``Die Druckfestigkeit beträgt {obere_grenze}``).
    """
    checks: list[NumberCheck] = []
    masked: list[MaskedText] = []
    for name, text in raw.items():
        literal = check_slot_text(text, sheet, section=name)
        checks += [check for check in literal.checks if check.status != "ok"]
        masked += literal.masked
        rendered, spans = render_with_spans(text, sheet)
        # Literal numbers were judged above; here only the inserted values are bound.
        checks += [
            check
            for check in check_text(rendered, sheet, section=name).checks
            if any(start <= check.start < end for start, end in spans)
        ]
    return FaithfulnessResult(checks=checks, masked=masked)


def describe_violations(result: FaithfulnessResult) -> list[str]:
    """One German line per violation, used in the corrective prompt and in the CLI."""
    return [
        f"„{check.text}“ im Abschnitt {check.section}: {check.reason}"
        for check in result.violations
    ]
