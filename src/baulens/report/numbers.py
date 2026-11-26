"""Extract numeric tokens from German (and English-formatted) report text.

Extraction runs in a fixed order on a copy of the text in which every handled span is
blanked out, so no digit is read twice and no digit is skipped:

1. identifiers copied verbatim from the input (case id, file name, element, model names);
   identifiers without letters (``1.2.0``) only directly after "Version"/"v";
2. standard references (``DIN EN 12390-3``); allow-listed ones are masked, others become
   ``reference`` tokens and count as violations;
3. dates (``14.03.2026``, ``2026-03-14``, ``14. März 2026``) and clock times (``10:15``);
4. single-digit list markers at the start of a line (``1. ``);
5. numbers with an optional sign, thousands grouping, decimals and a following unit;
6. German number words from null to hundert, including compounds such as "achtundzwanzig".

Larger spelled-out numbers ("dreihundertfünfundsechzig"), ordinals ("zweite"), articles
("ein", "eine") and fractions in words ("halb") are out of scope.
"""

import re
from dataclasses import dataclass, replace
from datetime import date, time
from decimal import Decimal
from typing import Literal

TokenKind = Literal["number", "word"]
SpecialKind = Literal["date", "time", "reference"]
MaskKind = Literal["identifier", "version", "standard", "enumeration"]

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
"""Standards a report may cite. Edition years (``:2019``) are not allowed: the model cannot
know which edition applies, so a year would be an invented number."""


@dataclass(frozen=True)
class NumberToken:
    """A number in the text with its candidate interpretations and its unit."""

    kind: TokenKind
    start: int
    end: int
    unit_end: int
    text: str
    candidates: tuple[tuple[Decimal, int], ...]
    """(value, decimals written) pairs; several when the format is ambiguous (``1.040``)."""
    explicit_sign: bool
    unit: str | None
    context: str
    """Text before the number in the same sentence, back to the previous number."""
    code: bool
    """Digits glued to a preceding letter ("C50", "XC4"): part of a code, never a quantity."""
    compound: str
    """Word glued to the unit with a hyphen, e.g. ``Prognoseintervall`` in ``90-%-Prognose…``."""


@dataclass(frozen=True)
class SpecialToken:
    kind: SpecialKind
    start: int
    end: int
    text: str
    parsed: date | time | str | None


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


_REFERENCE = re.compile(
    r"(?<![\w])(?:(?:DIN|EN|ISO)[ \u00a0]+)+\d{1,5}(?:-\d{1,3})*(?::\d{4})?(?!\d)"
)
_MONTHS = {
    "januar": 1,
    "februar": 2,
    "märz": 3,
    "maerz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}
_DATE_DE = re.compile(r"(?<!\d)(\d{1,2})\.[ \u00a0]?(\d{1,2})\.[ \u00a0]?(\d{4})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DATE_WORD = re.compile(
    r"(?<!\d)(\d{1,2})\.[ \u00a0]?(" + "|".join(_MONTHS) + r")[ \u00a0]+(\d{4})(?!\d)",
    re.IGNORECASE,
)
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?![\d:])")
_UNIT_EXPONENT = re.compile(r"(?<=/)(m|mm|cm|dm)([23])(?!\d)")
_ENUMERATION = re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?(\d)[.)](?=[ \t]+\S)")

_NUMBER = re.compile(
    r"""
    (?<!\d)
    (?:
        (?P<de_grouped>\d{1,3}(?:\.\d{3})+,\d+)          # 1.040,5
      | (?P<en_grouped>\d{1,3}(?:,\d{3})+\.\d+)          # 1,040.5
      | (?P<de_thousands>\d{1,3}(?:\.\d{3}){2,})         # 1.040.000
      | (?P<en_thousands>\d{1,3}(?:,\d{3}){2,})          # 1,040,000
      | (?P<spaced>\d{1,3}(?:[ \u00a0\u202f\u2009]\d{3})+(?:,\d+)?)  # 1 040 / 1 040,5
      | (?P<decimal>\d+[.,]\d+)                          # 61,9 / 61.9 / 1.040 (ambiguous)
      | (?P<integer>\d+)
    )
    (?!\d)
    """,
    re.VERBOSE,
)
_UNIT = re.compile(
    r"""
    [ \u00a0\u202f\u2009]?-?[ \u00a0\u202f\u2009]?
    (?P<unit>
        MPa\b | N/mm² | N/mm2\b | kg/m³ | kg/m3\b | kg\b
      | % | Prozent\w* | v\.[ ]?H\.
      | Pixel\b | px\b
      | Tag\w* | tägig\w* | d\b
      | Woche\w* | Monat\w* | Jahr\w* | Stunde\w* | h\b | min\b
      | mm\b | cm\b | m\b | °C
    )
    (?:-(?P<compound>[\wäöüÄÖÜß]+))?
    """,
    re.VERBOSE,
)
_UNIT_KINDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"MPa|N/mm"), "MPa"),
    (re.compile(r"kg"), "kg/m³"),
    (re.compile(r"%|Prozent|v\."), "%"),
    (re.compile(r"Pixel|px"), "px"),
    (re.compile(r"Tag|tägig|d$"), "d"),
    (re.compile(r"Woche"), "week"),
    (re.compile(r"Monat"), "month"),
    (re.compile(r"Jahr"), "year"),
    (re.compile(r"Stunde|h$|min$"), "hour"),
    (re.compile(r"mm|cm|m$"), "length"),
    (re.compile(r"°C"), "temperature"),
)
_RANGE_GAP = re.compile(
    r"^[ \u00a0\u202f\u2009]*(?:[-\u2013\u2014\u2212]|bis)[ \u00a0\u202f\u2009]*$"
)
_AND_GAP = re.compile(r"^[ \u00a0\u202f\u2009]*und[ \u00a0\u202f\u2009]*$")
_SENTENCE_END = re.compile(r"[.!?;:](?=\s)|\n")
_SIGN_CHARS = "+-\u2212\u2013"
_SIGN_PRECEDERS = set(_SPACES + "\t\n([{:=/;")

_UNITS_WORDS = {
    "null": 0,
    "eins": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "fünfzehn": 15,
    "sechzehn": 16,
    "siebzehn": 17,
    "achtzehn": 18,
    "neunzehn": 19,
    "hundert": 100,
    "einhundert": 100,
}
_TENS_WORDS = {
    "zwanzig": 20,
    "dreißig": 30,
    "dreissig": 30,
    "vierzig": 40,
    "fünfzig": 50,
    "sechzig": 60,
    "siebzig": 70,
    "achtzig": 80,
    "neunzig": 90,
}
_COMPOUND_UNITS = {
    "ein": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
}
_WORD = re.compile(r"(?<![\w])[A-Za-zÄÖÜäöüß]+(?![\w])")


def parse_number_word(word: str) -> int | None:
    """Value of a German cardinal number word between 0 and 100, otherwise ``None``."""
    lowered = word.lower()
    if lowered in _UNITS_WORDS:
        return _UNITS_WORDS[lowered]
    if lowered in _TENS_WORDS:
        return _TENS_WORDS[lowered]
    head, separator, tail = lowered.partition("und")
    if separator and head in _COMPOUND_UNITS and tail in _TENS_WORDS:
        return _COMPOUND_UNITS[head] + _TENS_WORDS[tail]
    return None


def normalize_reference(text: str) -> str:
    return " ".join(text.upper().replace("\u00a0", " ").split())


def is_allowed_reference(text: str) -> bool:
    normalized = normalize_reference(text)
    if normalized.startswith("DIN EN "):
        normalized = normalized.removeprefix("DIN ")
    return normalized in ALLOWED_REFERENCES


def unit_kind(unit_text: str) -> str:
    for pattern, kind in _UNIT_KINDS:
        if pattern.search(unit_text):
            return kind
    return "other"


def _interpretations(match: re.Match[str]) -> tuple[tuple[Decimal, int], ...]:
    group = match.lastgroup
    raw = match.group()
    if group == "de_grouped":
        whole, fraction = raw.replace(".", "").split(",")
        return ((Decimal(f"{whole}.{fraction}"), len(fraction)),)
    if group == "en_grouped":
        whole, fraction = raw.replace(",", "").split(".")
        return ((Decimal(f"{whole}.{fraction}"), len(fraction)),)
    if group in {"de_thousands", "en_thousands"}:
        return ((Decimal(raw.replace(".", "").replace(",", "")), 0),)
    if group == "spaced":
        compact = "".join(char for char in raw if char not in _SPACES)
        whole, _, fraction = compact.partition(",")
        value = Decimal(f"{whole}.{fraction}") if fraction else Decimal(whole)
        return ((value, len(fraction)),)
    if group == "decimal":
        separator = "," if "," in raw else "."
        whole, fraction = raw.split(separator)
        as_decimal = (Decimal(f"{whole}.{fraction}"), len(fraction))
        grouped = len(fraction) == 3 and 1 <= len(whole) <= 3 and not whole.startswith("0")
        if not grouped:
            return (as_decimal,)
        as_thousands = (Decimal(whole + fraction), 0)
        # German reading first: "1.040" is 1040, "1,040" is 1.04.
        return (as_thousands, as_decimal) if separator == "." else (as_decimal, as_thousands)
    return ((Decimal(raw), 0),)


class _Masker:
    def __init__(self, text: str) -> None:
        self.original = text
        self.text = text
        self.masked: list[MaskedSpan] = []
        self.specials: list[SpecialToken] = []

    def blank(self, start: int, end: int) -> None:
        self.text = self.text[:start] + _BLANK * (end - start) + self.text[end:]

    def mask(self, kind: MaskKind, start: int, end: int) -> None:
        self.masked.append(MaskedSpan(kind, start, end, self.original[start:end]))
        self.blank(start, end)

    def special(
        self, kind: SpecialKind, start: int, end: int, parsed: date | time | str | None
    ) -> None:
        self.specials.append(SpecialToken(kind, start, end, self.original[start:end], parsed))
        self.blank(start, end)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _safe_time(hour: int, minute: int) -> time | None:
    try:
        return time(hour, minute)
    except ValueError:
        return None


def _mask_structure(
    masker: _Masker, identifiers: tuple[str, ...], versions: tuple[str, ...]
) -> None:
    # Unit exponents written as digits ("kg/m3", "N/mm2") are not numbers.
    masker.text = _UNIT_EXPONENT.sub(
        lambda match: match[1] + ("³" if match[2] == "3" else "²"), masker.text
    )
    for identifier in sorted(identifiers, key=len, reverse=True):
        pattern = re.compile(r"(?<![\w])" + re.escape(identifier) + r"(?![\w])", re.IGNORECASE)
        for match in list(pattern.finditer(masker.text)):
            masker.mask("identifier", *match.span())
    for version in sorted(versions, key=len, reverse=True):
        pattern = re.compile(
            r"(?i)(?:\bversion|\bv)[ \u00a0]?(" + re.escape(version) + r")(?![\d\w])"
        )
        for match in list(pattern.finditer(masker.text)):
            masker.mask("version", *match.span(1))
    for match in list(_REFERENCE.finditer(masker.text)):
        if is_allowed_reference(match.group()):
            masker.mask("standard", *match.span())
        else:
            masker.special("reference", *match.span(), normalize_reference(match.group()))
    for match in list(_DATE_WORD.finditer(masker.text)):
        day, month, year = match.groups()
        parsed = _safe_date(int(year), _MONTHS[month.lower()], int(day))
        masker.special("date", *match.span(), parsed)
    for match in list(_DATE_DE.finditer(masker.text)):
        day, month, year = (int(part) for part in match.groups())
        masker.special("date", *match.span(), _safe_date(year, month, day))
    for match in list(_DATE_ISO.finditer(masker.text)):
        year, month, day = (int(part) for part in match.groups())
        masker.special("date", *match.span(), _safe_date(year, month, day))
    for match in list(_TIME.finditer(masker.text)):
        masker.special("time", *match.span(), _safe_time(int(match[1]), int(match[2])))
    for match in list(_ENUMERATION.finditer(masker.text)):
        masker.mask("enumeration", *match.span(1))
        masker.blank(match.end(1), match.end(1) + 1)


def _sentence_start(text: str, position: int) -> int:
    starts = [match.end() for match in _SENTENCE_END.finditer(text, 0, position)]
    return starts[-1] if starts else 0


def _unit_after(text: str, position: int) -> tuple[str | None, int, str]:
    match = _UNIT.match(text, position)
    if match is None:
        return None, position, ""
    return unit_kind(match["unit"]), match.end(), match["compound"] or ""


def _is_sentence_initial(text: str, position: int) -> bool:
    before = text[:position].rstrip(_SPACES + "\t")
    return not before or before[-1] in ".!?:\n"


def _number_tokens(text: str, original: str) -> list[NumberToken]:
    tokens: list[NumberToken] = []
    previous_end = 0
    spans: list[tuple[int, int, TokenKind, tuple[tuple[Decimal, int], ...]]] = []
    for match in _NUMBER.finditer(text):
        spans.append((match.start(), match.end(), "number", _interpretations(match)))
    for match in _WORD.finditer(text):
        word = match.group()
        if word[0].isupper() and not _is_sentence_initial(text, match.start()):
            continue
        value = parse_number_word(word)
        if value is not None:
            spans.append((match.start(), match.end(), "word", ((Decimal(value), 0),)))
    spans.sort()

    for start, end, kind, candidates in spans:
        explicit_sign = False
        if kind == "number" and start > 0 and text[start - 1] in _SIGN_CHARS:
            sign_position = start - 1
            preceder_ok = sign_position == 0 or text[sign_position - 1] in _SIGN_PRECEDERS
            gap = text[previous_end:sign_position]
            is_range = bool(tokens) and gap.strip(_SPACES + "\t") == ""
            if preceder_ok and not is_range:
                explicit_sign = True
                if text[sign_position] != "+":
                    candidates = tuple((-value, decimals) for value, decimals in candidates)
                start = sign_position
        unit, unit_end, compound = _unit_after(text, end)
        context_start = max(previous_end, _sentence_start(text, start))
        tokens.append(
            NumberToken(
                kind=kind,
                start=start,
                end=end,
                unit_end=unit_end,
                text=original[start:unit_end].strip(),
                candidates=candidates,
                explicit_sign=explicit_sign,
                unit=unit,
                context=text[context_start:start],
                code=not explicit_sign
                and start > 0
                and text[start - 1].isalpha()
                and text[start - 1] not in "xX",
                compound=compound,
            )
        )
        previous_end = unit_end

    # A unit written once after a range ("54,2-69,6 MPa", "zwischen 54 und 70 MPa") applies to
    # both bounds.
    for index in range(len(tokens) - 2, -1, -1):
        first, second = tokens[index], tokens[index + 1]
        if first.unit is not None or second.unit is None:
            continue
        gap = text[first.unit_end : second.start]
        between = _AND_GAP.match(gap) and "zwischen" in first.context[-20:].lower()
        if _RANGE_GAP.match(gap) or between:
            tokens[index] = replace(first, unit=second.unit, compound=second.compound)
    return tokens


def extract(
    text: str, *, identifiers: tuple[str, ...] = (), versions: tuple[str, ...] = ()
) -> Extraction:
    """Find every numeric token in ``text`` after masking identifiers and structure."""
    masker = _Masker(text)
    _mask_structure(masker, identifiers, versions)
    numbers = _number_tokens(masker.text, text)
    return Extraction(
        numbers=tuple(numbers),
        specials=tuple(sorted(masker.specials, key=lambda token: token.start)),
        masked=tuple(sorted(masker.masked, key=lambda span: span.start)),
    )
