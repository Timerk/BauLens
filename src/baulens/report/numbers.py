"""Extract numeric tokens from German report text.

The output locale is German only: decimal comma, ``.`` or a (narrow) space as thousands
separator in groups of three. ``61.9`` is flagged as a foreign format and ``1,040`` is 1,04.

Extraction runs in a fixed order on a copy of the text in which every handled span is
blanked, so no digit is read twice and none is skipped:

1. slot placeholders ``{name}`` (slot mode);
2. identifiers copied verbatim from the input, only as whole tokens; identifiers that look
   like numbers (``123456``, ``999kg``) only directly after a label ("Fall", "Version", …);
3. standard references (``DIN EN 12390-3``); allow-listed ones are masked, others become
   ``reference`` tokens;
4. dates (``14.03.2026``, ISO and month-name forms) and clock times (``10:15``, ``10:15:59``)
   as whole tokens;
5. single-digit list markers at the start of a line;
6. numbers with sign, grouping, decimals, the unit right after them and the words around them;
7. German cardinal number words up to the hundreds of thousands ("dreihundertfünfundsechzig").

Out of scope (documented): ordinals ("zweite"), fractions in words ("halb"), the article
"ein/eine" as the number one, and vague plurals ("Hunderte").
"""

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

TokenKind = Literal["number", "word"]
SpecialKind = Literal["date", "time", "reference", "slot"]
MaskKind = Literal["identifier", "standard", "enumeration"]

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
    """Text after the unit, up to the next number or a clause boundary."""
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
_ENUMERATION = re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?(\d)[.)](?=[ \t]+\D)")
_UNIT_EXPONENT = re.compile(r"(?<=/)(m|mm|cm|dm)([23])(?!\d)")

_NUMBER = re.compile(
    r"""
    (?<!\d)
    (?:
        (?P<grouped>\d{1,3}(?:\.\d{3})+(?:,\d+)?)                  # 1.040 / 1.040,5
      | (?P<spaced>\d{1,3}(?:[ \u00a0\u202f\u2009]\d{3})+(?:,\d+)?)  # 1 040 / 1 040,5
      | (?P<english>\d{1,3}(?:,\d{3})+\.\d+|\d+\.\d+)              # 1,040.5 / 61.9
      | (?P<decimal>\d+,\d+)                                      # 61,9 / 1,040
      | (?P<integer>\d+)
    )
    (?![\d])
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
    (re.compile(r"^(Pixel|px)$"), "px"),
    (re.compile(r"^(Tag|Tage|Tagen|Tages|d|tägig\w*|Tage-\w+|Tages\w+|tägige\w*)$"), "d"),
)
_OTHER_UNIT = re.compile(
    r"^(t|kg|g|mg|kN|N|MN|Pa|kPa|GPa|psi|bar|l|ml|m|cm|mm|km|µm|h|min|s|K|°C|°|‰|ppm|"
    r"Jahr\w*|Monat\w*|Woche\w*|Stunde\w*|Minute\w*|Sekunde\w*|Grad|Prozentpunkt\w*)$"
)
_CONNECTOR = re.compile(r"^(x|\u00d7)$")
_RANGE_GAP = re.compile(
    r"^[ \u00a0\u202f\u2009]*(?:[-\u2013\u2014\u2212]|bis)[ \u00a0\u202f\u2009]*$"
)
_AND_GAP = re.compile(r"^[ \u00a0\u202f\u2009]*und[ \u00a0\u202f\u2009]*$")
_TIMES_GAP = re.compile(r"^[ \u00a0\u202f\u2009]*[x\u00d7][ \u00a0\u202f\u2009]*$")
_BEFORE_BOUNDARY = re.compile(r"[.!?](?=\s)|[,;\n]|\bund\b|\bsowie\b")
_AFTER_BOUNDARY = re.compile(r"[.!?](?=\s|$)|[,;()\n]|\bund\b|\bsowie\b")
_SIGN_CHARS = "+-\u2212\u2013"
_SIGN_PRECEDERS = set(_SPACES + "\t\n([{:=/;")

_SMALL = {
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
}
_TENS = {
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
_UNIT_PREFIX = {
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


def _below_hundred(word: str) -> int | None:
    if word in _SMALL:
        return _SMALL[word]
    if word in _TENS:
        return _TENS[word]
    head, separator, tail = word.partition("und")
    if separator and head in _UNIT_PREFIX and tail in _TENS:
        return _UNIT_PREFIX[head] + _TENS[tail]
    return None


def _below_thousand(word: str) -> int | None:
    if "hundert" not in word:
        return _below_hundred(word)
    head, _, tail = word.partition("hundert")
    hundreds = 1 if head in {"", "ein"} else _UNIT_PREFIX.get(head)
    rest = 0 if tail == "" else _below_hundred(tail)
    if hundreds is None or rest is None:
        return None
    return hundreds * 100 + rest


def parse_number_word(word: str) -> int | None:
    """Value of a German cardinal number word below one million, otherwise ``None``."""
    lowered = word.lower()
    if lowered in {"ein", "eine", "einer", "einem", "einen", "eines"}:
        return None
    if "tausend" not in lowered:
        return _below_thousand(lowered)
    head, _, tail = lowered.partition("tausend")
    thousands = 1 if head in {"", "ein"} else _below_thousand(head)
    rest = 0 if tail == "" else _below_thousand(tail)
    if thousands is None or rest is None:
        return None
    return thousands * 1000 + rest


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
    for match in list(_ENUMERATION.finditer(masker.text)):
        masker.mask("enumeration", *match.span(1))
        masker.blank(match.end(1), match.end(1) + 1)


def _last_boundary(text: str, position: int) -> int:
    ends = [match.end() for match in _BEFORE_BOUNDARY.finditer(text, 0, position)]
    return ends[-1] if ends else 0


def _next_boundary(text: str, position: int, limit: int) -> int:
    match = _AFTER_BOUNDARY.search(text, position, limit)
    return match.start() if match else limit


def _is_sentence_initial(text: str, position: int) -> bool:
    before = text[:position].rstrip(_SPACES + "\t")
    return not before or before[-1] in ".!?:\n"


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
    for match in _WORD.finditer(text):
        word = match.group()
        if word[0].isupper() and not _is_sentence_initial(text, match.start()):
            continue
        number = parse_number_word(word)
        if number is not None:
            spans.append(
                _Span(match.start(), match.end(), "word", str(number), Decimal(number), False)
            )
    return sorted(spans, key=lambda span: span.start)


def _unit_after(text: str, position: int) -> tuple[str | None, str, int]:
    match = _UNIT.match(text, position)
    if match is None:
        return None, "", position
    unit_text = match["unit"]
    if _CONNECTOR.match(unit_text):
        return None, "", position
    if unit_text.startswith("%-") or unit_text.startswith("%"):
        return "%", "%", match.start("unit") + 1
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
                after=text[unit_end:after_end],
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
    return Extraction(
        numbers=tuple(numbers),
        specials=tuple(sorted(masker.specials, key=lambda token: token.start)),
        masked=tuple(sorted(masker.masked, key=lambda span: span.start)),
        masked_text=masker.text,
    )
