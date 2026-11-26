"""Numeric faithfulness: every number in a report must be a permitted rendering of a fact.

Each numeric token from :mod:`baulens.report.numbers` is compared with every
:class:`~baulens.report.facts.Rendering` of the case. The verdict is

``ok``
    the value is a permitted rendering of a fact with a compatible unit and context;
``misattributed``
    the value belongs to a fact but is stated with the wrong unit (``28 MPa`` for an age of
    28 days), the wrong percent quantity (the interval coverage called the crack
    probability), or without the negative sign of a signed fact;
``bad_rounding``
    the value is within one unit of its last written digit of a fact but is not a valid
    rounding (``61`` for ``61,9``), or the rounding moves the crack probability across the
    decision threshold (``50 %`` for ``49,7 %`` at a threshold of ``50 %``);
``invented``
    nothing in the input explains the value, including dates other than the capture date and
    standard references outside the allow-list.

The checker verifies numbers only. It does not verify qualitative claims, such as whether a
strength is called "high", or whether a sentence attributes the right MPa value to the
right bound; see ``docs/results/report.md`` for its limitations.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from typing import Literal

from baulens.report.facts import Fact, FactSheet, Quantity, Rendering
from baulens.report.numbers import (
    ALLOWED_REFERENCES,
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

_PERCENT_QUANTITIES: frozenset[Quantity] = frozenset({"probability", "coverage", "threshold"})
_HINTS: tuple[tuple[Quantity, tuple[str, ...]], ...] = (
    ("threshold", ("schwelle",)),
    ("coverage", ("intervall", "überdeckung", "abdeckung")),
    ("probability", ("risswahrscheinlichkeit", "wahrscheinlichkeit für einen riss")),
)

MatchQuality = Literal["exact", "rounded", "near"]


@dataclass(frozen=True)
class _Candidate:
    fact: Fact
    rendering: Rendering
    quality: MatchQuality
    unit_ok: bool
    context_ok: bool
    sign_error: bool


def _half_unit(decimals: int) -> Decimal:
    return Decimal(5).scaleb(-(decimals + 1))


def _quality(
    token: NumberToken, value: Decimal, decimals: int, rendering: Rendering, target: Decimal
) -> MatchQuality | None:
    """How ``value`` (written with ``decimals`` digits) relates to one rendering.

    Rounding is only accepted for digits with the fact's own unit (or for dimensionless
    facts): a bare "3" next to "Bohrkerne" must not pass as a rounding of 2,5 kg/m³. Number
    words must match exactly.
    """
    if value == target:
        return "exact"
    may_round = token.kind == "number" and (rendering.unit is None or token.unit == rendering.unit)
    within_rounding = abs(value - target) <= _half_unit(decimals)
    if may_round and rendering.min_decimals <= decimals < rendering.decimals and within_rounding:
        crosses = rendering.guard is not None and (value >= rendering.guard) != (
            target >= rendering.guard
        )
        return "near" if crosses else "rounded"
    if abs(value - target) <= Decimal(1).scaleb(-decimals):
        return "near"
    if decimals > rendering.decimals and abs(value - target) <= _half_unit(rendering.decimals):
        return "near"  # more digits than the facts show, e.g. 61,87 for 61,9

    return None


def context_hint(token: NumberToken) -> Quantity | None:
    """Percent quantity named right before the number or glued to its unit, if any."""
    compound = token.compound.lower()
    if "intervall" in compound:
        return "coverage"
    context = token.context.lower()
    best: tuple[int, Quantity] | None = None
    for quantity, keywords in _HINTS:
        position = max(context.rfind(keyword) for keyword in keywords)
        if position >= 0 and (best is None or position > best[0]):
            best = (position, quantity)
    return best[1] if best else None


def _candidates(token: NumberToken, sheet: FactSheet) -> list[_Candidate]:
    hint = context_hint(token)
    enforce_hint = hint is not None and sheet.has_quantity(hint)
    found: list[_Candidate] = []
    for fact in sheet.facts:
        context_ok = (
            not enforce_hint or fact.quantity not in _PERCENT_QUANTITIES or fact.quantity == hint
        )
        for rendering in fact.renderings:
            unit_ok = token.unit is None or token.unit == rendering.unit
            for value, decimals in token.candidates:
                unsigned = fact.allow_unsigned and not token.explicit_sign
                target = abs(rendering.value) if unsigned else rendering.value
                written = abs(value) if unsigned else value
                quality = _quality(token, written, decimals, rendering, target)
                sign_error = False
                if quality is None and target != 0:
                    quality = _quality(token, written, decimals, rendering, -target)
                    sign_error = quality is not None
                if quality is not None:
                    found.append(
                        _Candidate(fact, rendering, quality, unit_ok, context_ok, sign_error)
                    )
    return found


_QUANTITY_NAMES: dict[Quantity, str] = {
    "probability": "Risswahrscheinlichkeit",
    "coverage": "Überdeckung des Prognoseintervalls",
    "threshold": "Entscheidungsschwelle",
}


def _shown(fact: Fact) -> str:
    return f"{fact.label} ({fact.display})"


def _unit_label(unit: str | None) -> str:
    return unit if unit is not None else "ohne Einheit"


def _number_check(
    token: NumberToken,
    section: str,
    status: CheckStatus,
    reason: str,
    fields: list[str] | None = None,
) -> NumberCheck:
    return NumberCheck(
        section=section,
        start=token.start,
        end=token.unit_end,
        text=token.text,
        kind=token.kind,
        value=str(token.candidates[0][0]),
        unit=token.unit,
        status=status,
        matched_fields=fields or [],
        reason=reason,
    )


def _check_number(token: NumberToken, sheet: FactSheet, section: str) -> NumberCheck:
    if token.code:
        reason = "Zahl in einer Kennung (etwa einer Festigkeitsklasse), nicht in der Faktenliste"
        return _number_check(token, section, "invented", reason)
    candidates = _candidates(token, sheet)
    usable = [candidate for candidate in candidates if candidate.quality != "near"]
    ok = [c for c in usable if c.unit_ok and c.context_ok and not c.sign_error]
    if ok:
        ok.sort(key=lambda c: (c.quality != "exact", c.rendering.unit != token.unit))
        best = ok[0]
        how = "exakt" if best.quality == "exact" else "gerundet"
        fields = list(dict.fromkeys(candidate.fact.field for candidate in ok))
        return _number_check(token, section, "ok", f"{how} aus {_shown(best.fact)}", fields)
    # A near miss needs the fact's own unit; a bare "drei" is not a bad rounding of 2,5 kg/m³.
    near = [
        c
        for c in candidates
        if c.quality == "near"
        and c.context_ok
        and not c.sign_error
        and token.unit == c.rendering.unit
    ]
    if usable and not near:
        usable.sort(key=lambda c: (not c.sign_error, c.unit_ok))
        candidate = usable[0]
        if candidate.sign_error:
            reason = f"Wert von {_shown(candidate.fact)}, aber mit falschem Vorzeichen"
        elif not candidate.unit_ok:
            reason = (
                f"Wert von {_shown(candidate.fact)}, aber mit Einheit "
                f"{_unit_label(token.unit)} statt {_unit_label(candidate.rendering.unit)}"
            )
        else:
            hint = context_hint(token)
            named = _QUANTITY_NAMES.get(hint, str(hint)) if hint is not None else "?"
            reason = f"Wert von {_shown(candidate.fact)}, aber im Text als {named} bezeichnet"
        fields = list(dict.fromkeys(c.fact.field for c in usable))
        return _number_check(token, section, "misattributed", reason, fields)
    if near:
        candidate = near[0]
        value, decimals = token.candidates[0]
        rendering = candidate.rendering
        crossing = (
            rendering.guard is not None
            and rendering.min_decimals <= decimals < rendering.decimals
            and abs(value - rendering.value) <= _half_unit(decimals)
        )
        why = (
            "Rundung verschiebt den Wert über die Entscheidungsschwelle"
            if crossing
            else "keine zulässige Rundung"
        )
        reason = f"{why} von {_shown(candidate.fact)}"
        return _number_check(token, section, "bad_rounding", reason, [candidate.fact.field])
    return _number_check(token, section, "invented", "nicht in der Faktenliste enthalten")


def _check_special(token: SpecialToken, sheet: FactSheet, section: str) -> NumberCheck:
    if token.kind == "reference":
        allowed = ", ".join(sorted(ALLOWED_REFERENCES))
        return NumberCheck(
            section=section,
            start=token.start,
            end=token.end,
            text=token.text,
            kind="reference",
            value=str(token.parsed),
            status="invented",
            reason=f"Normverweis nicht zulässig (zulässig: {allowed}; ohne Ausgabejahr)",
        )
    known: frozenset[date] | frozenset[time] = sheet.dates if token.kind == "date" else sheet.times
    parsed = token.parsed if isinstance(token.parsed, date | time) else None
    found = parsed is not None and parsed in known
    label = "Datum" if token.kind == "date" else "Uhrzeit"
    return NumberCheck(
        section=section,
        start=token.start,
        end=token.end,
        text=token.text,
        kind=token.kind,
        value=parsed.isoformat() if parsed is not None else None,
        status="ok" if found else "invented",
        matched_fields=["image.captured_at"] if found else [],
        reason="Aufnahmezeitpunkt des Bildes" if found else f"{label} nicht in der Faktenliste",
    )


def check_text(text: str, sheet: FactSheet, *, section: str = "text") -> FaithfulnessResult:
    """Check every numeric token of one text block against the case facts."""
    extraction = extract(text, identifiers=sheet.identifiers, versions=sheet.versions)
    checks = [_check_number(token, sheet, section) for token in extraction.numbers]
    checks += [_check_special(token, sheet, section) for token in extraction.specials]
    checks.sort(key=lambda check: check.start)
    masked = [
        MaskedText(section=section, start=span.start, end=span.end, text=span.text, kind=span.kind)
        for span in extraction.masked
    ]
    return FaithfulnessResult(checks=checks, masked=masked)


def check_sections(sections: Mapping[SectionName, str], sheet: FactSheet) -> FaithfulnessResult:
    """Check each report section separately and merge the results."""
    checks: list[NumberCheck] = []
    masked: list[MaskedText] = []
    for name, text in sections.items():
        result = check_text(text, sheet, section=name)
        checks += result.checks
        masked += result.masked
    return FaithfulnessResult(checks=checks, masked=masked)


def describe_violations(result: FaithfulnessResult) -> list[str]:
    """One German line per violation, used in the corrective prompt and in the CLI."""
    return [
        f"„{check.text}“ im Abschnitt {check.section}: {check.reason}"
        for check in result.violations
    ]
