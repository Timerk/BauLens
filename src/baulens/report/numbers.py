"""Extract numeric tokens from German report text.

The output locale is German only: decimal comma, ``.`` or a (narrow) space as thousands
separator in groups of three. ``61.9`` is flagged as a foreign format and ``1,040`` is 1,04.

This module serves the free-text mode. Extraction runs in a fixed order on a copy of the
text in which every handled span is blanked, so no digit is read twice and none is skipped:

1. identifiers copied verbatim from the input, only as the complete string and only as a
   whole token; identifiers that look like numbers (``123456``, ``999kg``) only directly
   after a label ("Fall", "Version", …). Parts of an identifier are never masked, so a
   strength class inside an element name ("Wand C30/37") does not excuse "über C30/37";
2. standard references (``DIN EN 12390-3``); allow-listed ones are masked, others become
   ``reference`` tokens;
3. dates (``14.03.2026``, ISO and month-name forms) and clock times (``10:15``, ``10:15:59``)
   as whole tokens;
4. ASCII-digit numbers with sign, grouping, decimals, the unit right after them and the words
   around them;
5. every other numeral (number words in any form, non-ASCII digits, Roman numerals) from
   :mod:`baulens.report.numerals`, reported as ``numeral`` tokens that never match a fact.

There is no exemption for list markers: "1." in prose is a number like any other.
"""

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

from baulens.report.numerals import find_numerals

TokenKind = Literal["number"]
SpecialKind = Literal["date", "time", "reference", "slot", "numeral"]
MaskKind = Literal["identifier", "standard"]

_BLANK = "█"
_SPACES = " \u00a0\u202f\u2009"

ALLOWED_REFERENCES: frozenset[str] = frozenset(
    {
        "EN 12390-1",
        "EN 12390-2",
        "EN 12390-3",
        "EN 206",
        "EN 206-1",
        "EN 13791",
        "EN 12504-1",
        "EN 12504-2",
        "EN 1992-1-1",
        "DIN 1045-2",
        "DIN 1045-3",
    }
)
"""Standards a report may cite, without edition year (the model cannot know the edition)."""

IDENTIFIER_LABELS = (
    "fall",
    "fall-id",
    "fall-nr.",
    "fallnummer",
    "prüffall",
    "prüfkörper",
    "probe",
    "case_id",
    "case",
    "version",
    "modellversion",
    "v",
    "bild",
    "datei",
    "bilddatei",
    "modell",
    "bauteil",
)


@dataclass(frozen=True)
class NumberToken:
    """A number in the text with its exact digits, its unit and the words around it."""

    kind: TokenKind
    start: int
    end: int
    unit_end: int
    text: str
    sign: str | None
    """``"+"`` or ``"-"`` if written, else ``None``."""
    digits: str
    """Canonical German digits without sign or grouping, e.g. ``"1040"`` or ``"61,9"``."""
    value: Decimal
    foreign: bool
    """English decimal point (``61.9``): never accepted."""
    unit: str | None
    """``MPa``, ``kg/m³``, ``d``, ``%``, ``px``, ``other`` (any other unit) or ``None``."""
    unit_text: str
    code: bool
    """Digits glued to a preceding letter (``C50``, ``XC4``): part of a code."""
    before: str
    """Text before the number, back to the previous number or a clause boundary."""
    after: str
    """Text after the unit, up to the next number or a clause boundary; or the content of a
    parenthesis right after the number ("63,1 MPa (Prognosewert)")."""
    far: str = ""
    """Earlier text of the same clause, before the previous number ("Der Beitrag des
    Betonalters von 365 Tagen beträgt +9,5 MPa"); used only if ``before`` and ``after`` name
    nothing relevant."""
    range_role: Literal["lower", "upper"] | None = None


@dataclass(frozen=True)
class SpecialToken:
    kind: SpecialKind
    start: int
    end: int
    text: str
    normalized: str


@dataclass(frozen=True)
class MaskedSpan:
    kind: MaskKind
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Extraction:
    numbers: tuple[NumberToken, ...]
    specials: tuple[SpecialToken, ...]
    masked: tuple[MaskedSpan, ...]
    masked_text: str


_SLOT = re.compile(r"\{([a-z0-9_]+)\}")
_REFERENCE = re.compile(
    r"(?<![\w])(?:(?:DIN|EN|ISO)[ \u00a0]+)+\d{1,5}(?:-\d{1,3})*(?::\d{4})?(?!\d)"
)
_MONTHS = "januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember"
_DATES = (
    re.compile(r"(?<![\d.])\d{1,2}\.[ \u00a0]?\d{1,2}\.[ \u00a0]?\d{2,4}(?![\d])"),
    re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"),
    re.compile(
        r"(?<![\d.])\d{1,2}\.[ \u00a0]?(?:" + _MONTHS + r")[ \u00a0]+\d{4}(?!\d)", re.IGNORECASE
    ),
)
_TIME = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")
_UNIT_EXPONENT = re.compile(r"(?<=/)(m|mm|cm|dm)([23])(?!\d)")

_NUMBER = re.compile(
    r"""
    (?<![0-9])
    (?:
        (?P<grouped>[0-9]{1,3}(?:\.[0-9]{3})+(?:,[0-9]+)?)                  # 1.040 / 1.040,5
      | (?P<spaced>[0-9]{1,3}(?:[ \u00a0\u202f\u2009][0-9]{3})+(?:,[0-9]+)?)  # 1 040 / 1 040,5
      | (?P<english>[0-9]{1,3}(?:,[0-9]{3})+\.[0-9]+|[0-9]+\.[0-9]+)              # 1,040.5 / 61.9
      | (?P<decimal>[0-9]+,[0-9]+)                                      # 61,9 / 1,040
      | (?P<integer>[0-9]+)
    )
    (?![0-9])
    """,
    re.VERBOSE,
)
_UNIT = re.compile(
    r"[ \u00a0\u202f\u2009]?(?:-[ \u00a0]?)?"
    r"(?P<unit>%|[A-Za-zÄÖÜäöüß°µ‰][^\s\d,;:()\[\]{}\"„“”'!?]*)"
)
_KNOWN_UNITS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(MPa|N/mm²)$"), "MPa"),
    (re.compile(r"^kg/m³$"), "kg/m³"),
    (re.compile(r"^(%|Prozent)$"), "%"),
    (re.compile(r"^(Pixel\w*|px)$"), "px"),
    (re.compile(r"^(Tag|Tage|Tagen|Tages|d|tägig\w*|Tage-\w+|Tages\w+|tägige\w*)$"), "d"),
)
_OTHER_UNIT = re.compile(
    r"^(t|kg|g|mg|kN|N|MN|Pa|kPa|GPa|psi|bar|l|ml|m|cm|mm|km|µm|h|min|s|K|°C|°|‰|ppm|"
    r"Jahr\w*|Monat\w*|Woche\w*|Stunde\w*|Minute\w*|Sekunde\w*|Grad|Prozentpunkt\w*)$"
)
_PERCENT_POINTS = re.compile(r"[ \u00a0]?-?[ \u00a0]?Punkt\w*")
_CONNECTOR = re.compile(r"^(x|\u00d7)$")
_RANGE_GAP = re.compile(
    r"^[ \u00a0\u202f\u2009]*(?:[-\u2013\u2014\u2212]|bis)[ \u00a0\u202f\u2009]*$"
)
_AND_GAP = re.compile(r"^[ \u00a0\u202f\u2009]*und[ \u00a0\u202f\u2009]*$")
_TIMES_GAP = re.compile(r"^[ \u00a0\u202f\u2009]*[x\u00d7][ \u00a0\u202f\u2009]*$")
_BEFORE_BOUNDARY = re.compile(r"[.!?](?=\s)|[,;\n]|\bund\b|\bsowie\b")
_PAREN_LABEL = re.compile(r"[ \u00a0]*\(([^()\d]{1,60})\)")
_AFTER_BOUNDARY = re.compile(r"[.!?](?=\s|$)|[,;()\n]|\bund\b|\bsowie\b")
_SIGN_CHARS = "+-\u2212\u2013"
_SIGN_PRECEDERS = set(_SPACES + "\t\n([{:=/;")


def normalize_reference(text: str) -> str:
    return " ".join(text.upper().replace("\u00a0", " ").split())


def is_allowed_reference(text: str) -> bool:
    normalized = normalize_reference(text)
    if normalized.startswith("DIN EN "):
        normalized = normalized.removeprefix("DIN ")
    return normalized in ALLOWED_REFERENCES


def classify_unit(unit_text: str) -> str | None:
    """Kind of the word right after a number; ``None`` if it is not a unit at all."""
    text = unit_text.rstrip(".")
    for pattern, kind in _KNOWN_UNITS:
        if pattern.match(text):
            return kind
    if _OTHER_UNIT.match(text) or "/" in text or "²" in text or "³" in text:
        return "other"
    return None


def _digits(match: re.Match[str]) -> tuple[str, Decimal, bool]:
    raw = match.group()
    group = match.lastgroup
    if group == "english":
        whole, _, fraction = raw.replace(",", "").partition(".")
        return f"{whole},{fraction}", Decimal(f"{whole}.{fraction}"), True
    compact = "".join(char for char in raw if char not in _SPACES + ".")
    whole, _, fraction = compact.partition(",")
    value = Decimal(f"{whole}.{fraction}") if fraction else Decimal(whole)
    return compact, value, False


class _Masker:
    def __init__(self, text: str) -> None:
        self.original = text
        self.text = _UNIT_EXPONENT.sub(lambda m: m[1] + ("³" if m[2] == "3" else "²"), text)
        self.masked: list[MaskedSpan] = []
        self.specials: list[SpecialToken] = []

    def blank(self, start: int, end: int) -> None:
        self.text = self.text[:start] + _BLANK * (end - start) + self.text[end:]

    def mask(self, kind: MaskKind, start: int, end: int) -> None:
        self.masked.append(MaskedSpan(kind, start, end, self.original[start:end]))
        self.blank(start, end)

    def special(self, kind: SpecialKind, start: int, end: int, normalized: str) -> None:
        text = self.original[start:end]
        self.specials.append(SpecialToken(kind, start, end, text, normalized))
        self.blank(start, end)


_LABEL = r"(?:" + "|".join(re.escape(label) for label in IDENTIFIER_LABELS) + r")"


def _mask_identifiers(masker: _Masker, identifiers: tuple[tuple[str, bool], ...]) -> None:
    for text, labelled in sorted(identifiers, key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(text)
        if labelled:
            pattern = re.compile(
                r"(?i)(?<![\w])" + _LABEL + r"[ \u00a0]?[:#]?[ \u00a0]?(" + escaped + r")(?![\w])"
            )
        else:
            pattern = re.compile(r"(?i)(?<![\w])(" + escaped + r")(?![\w])")
        for match in list(pattern.finditer(masker.text)):
            masker.mask("identifier", *match.span(1))


def _mask_structure(
    masker: _Masker, identifiers: tuple[tuple[str, bool], ...], slots: bool
) -> None:
    if slots:
        for match in list(_SLOT.finditer(masker.text)):
            masker.special("slot", *match.span(), match[1])
    _mask_identifiers(masker, identifiers)
    for match in list(_REFERENCE.finditer(masker.text)):
        if is_allowed_reference(match.group()):
            masker.mask("standard", *match.span())
        else:
            masker.special("reference", *match.span(), normalize_reference(match.group()))
    for pattern in _DATES:
        for match in list(pattern.finditer(masker.text)):
            masker.special("date", *match.span(), match.group())
    for match in list(_TIME.finditer(masker.text)):
        masker.special("time", *match.span(), match.group())


def _last_boundary(text: str, position: int) -> int:
    ends = [match.end() for match in _BEFORE_BOUNDARY.finditer(text, 0, position)]
    return ends[-1] if ends else 0


def _next_boundary(text: str, position: int, limit: int) -> int:
    match = _AFTER_BOUNDARY.search(text, position, limit)
    return match.start() if match else limit


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    kind: TokenKind
    digits: str
    value: Decimal
    foreign: bool


def _spans(text: str) -> list[_Span]:
    spans = []
    for match in _NUMBER.finditer(text):
        digits, value, foreign = _digits(match)
        spans.append(_Span(match.start(), match.end(), "number", digits, value, foreign))
    return sorted(spans, key=lambda span: span.start)


def _unit_after(text: str, position: int) -> tuple[str | None, str, int]:
    match = _UNIT.match(text, position)
    if match is None:
        return None, "", position
    unit_text = match["unit"]
    if _CONNECTOR.match(unit_text):
        return None, "", position
    if unit_text.startswith("%"):
        percent_end = match.start("unit") + 1
        points = _PERCENT_POINTS.match(text, percent_end)
        if points:
            # "%-Punkte" are percentage points, a different quantity than percent.
            return "other", text[match.start("unit") : points.end()], points.end()
        return "%", "%", percent_end
    kind = classify_unit(unit_text)
    if kind is None:
        return None, "", position
    return kind, unit_text.rstrip("."), match.start("unit") + len(unit_text.rstrip("."))


def _number_tokens(text: str, original: str) -> list[NumberToken]:
    spans = _spans(text)
    tokens: list[NumberToken] = []
    previous_end = 0
    for index, span in enumerate(spans):
        start, sign = span.start, None
        if span.kind == "number" and start > 0 and text[start - 1] in _SIGN_CHARS:
            position = start - 1
            preceder_ok = position == 0 or text[position - 1] in _SIGN_PRECEDERS
            gap_is_blank = bool(tokens) and text[previous_end:position].strip(_SPACES + "\t") == ""
            if preceder_ok and not gap_is_blank:
                sign = "+" if text[position] == "+" else "-"
                start = position
        unit, unit_text, unit_end = _unit_after(text, span.end)
        next_start = spans[index + 1].start if index + 1 < len(spans) else len(text)
        before_start = max(previous_end, _last_boundary(text, start))
        after_end = _next_boundary(text, unit_end, max(unit_end, next_start))
        label = _PAREN_LABEL.match(text, unit_end)
        after = label[1] if label else text[unit_end:after_end]
        clause_start = _last_boundary(text, start)
        code = (
            sign is None and start > 0 and text[start - 1].isalpha() and text[start - 1] not in "xX"
        )
        tokens.append(
            NumberToken(
                kind=span.kind,
                start=start,
                end=span.end,
                unit_end=unit_end,
                text=original[start:unit_end].strip(),
                sign=sign,
                digits=span.digits,
                value=-span.value if sign == "-" else span.value,
                foreign=span.foreign,
                unit=unit,
                unit_text=unit_text,
                code=code,
                before=text[before_start:start],
                after=after,
                far=text[clause_start:before_start] if clause_start < before_start else "",
            )
        )
        previous_end = unit_end
    return _link_ranges(text, tokens)


def _link_ranges(text: str, tokens: list[NumberToken]) -> list[NumberToken]:
    """A unit after a range ("54,2\u201369,6 MPa") applies to both bounds; MPa ranges get roles."""
    for index in range(len(tokens) - 2, -1, -1):
        first, second = tokens[index], tokens[index + 1]
        gap = text[first.unit_end : second.start]
        between = bool(_AND_GAP.match(gap)) and "zwischen" in first.before[-20:].lower()
        if _TIMES_GAP.match(gap) and first.unit is None:
            tokens[index] = replace(first, unit=second.unit, unit_text=second.unit_text)
            continue
        if not (_RANGE_GAP.match(gap) or between):
            continue
        if first.unit is None:
            first = replace(first, unit=second.unit, unit_text=second.unit_text)
        if first.unit == second.unit == "MPa":
            first = replace(first, range_role="lower")
            tokens[index + 1] = replace(second, range_role="upper")
        tokens[index] = first
    return tokens


def extract(
    text: str,
    *,
    identifiers: tuple[tuple[str, bool], ...] = (),
    slots: bool = False,
) -> Extraction:
    """Find every numeric token in ``text`` after masking identifiers and structure."""
    masker = _Masker(text)
    _mask_structure(masker, identifiers, slots)
    numbers = _number_tokens(masker.text, text)
    # Everything numeric the German parser does not read (number words, non-ASCII digits,
    # Roman numerals) is reported as a numeral and never matches a fact.
    for numeral in find_numerals(masker.text, include_ascii_digits=False):
        masker.special("numeral", numeral.start, numeral.end, numeral.text)
    return Extraction(
        numbers=tuple(numbers),
        specials=tuple(sorted(masker.specials, key=lambda token: token.start)),
        masked=tuple(sorted(masker.masked, key=lambda span: span.start)),
        masked_text=masker.text,
    )
