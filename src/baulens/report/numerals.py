"""One broad numeral detector shared by both generation modes.

Anything that can express a quantity outside the facts is a numeral here, so both modes fail
closed on it:

* any Unicode numeric character (ASCII digits are reported separately so that free-text mode
  can parse them): other decimal digits (Arabic-Indic ``٩``, fullwidth digits), superscripts
  and subscripts (``⁹``), circled and enclosed numbers (``⑨``), Roman numeral characters
  (``Ⅻ``) and vulgar fractions (``½``). Superscript ``²``/``³`` directly after a length unit
  (``m³``, ``mm²``) is a unit exponent, not a numeral;
* Roman numerals written with letters as standalone tokens of two or more characters
  (``II``, ``XII``) and the single tokens ``I``, ``V``, ``X`` after a word such as "CEM";
* German number words in any capitalisation: cardinals below one million ("drei",
  "Drei", "dreihundertfünfundsechzig", "eins"), ordinals ("erste", "dritten",
  "achtundzwanzigste", "zweitens"), fractions ("halb", "Hälfte", "Drittel", "anderthalb"),
  multipliers ("doppelt", "dreifach", "zweimal") and quantity nouns ("Dutzend", "Paar",
  "Hunderte", "Tausende", "Million").

Deliberately *not* numerals: the indefinite article "ein/eine/einer/einem/einen/eines",
lowercase "paar" (vague "a few"), "einmal" (an idiom more often than a count), "mehrfach",
"einzeln", "einzig" and "beide". These are documented gaps, not oversights.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

NumeralKind = Literal["digit", "symbol", "roman", "word"]

_SMALL = {
    "null": 0,
    "eins": 1,
    "zwei": 2,
    "zwo": 2,
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
ARTICLES = frozenset({"ein", "eine", "einer", "einem", "einen", "eines"})
_ORDINAL_STEMS = {
    "erst": 1,
    "zweit": 2,
    "dritt": 3,
    "viert": 4,
    "fünft": 5,
    "sechst": 6,
    "siebt": 7,
    "siebent": 7,
    "acht": 8,
    "neunt": 9,
    "zehnt": 10,
    "elft": 11,
    "zwölft": 12,
}
_ORDINAL_ENDINGS = ("ens", "e", "en", "er", "es", "em")
_FRACTION = re.compile(
    r"^(halb\w*|hälfte\w*|anderthalb\w*|eineinhalb\w*|"
    r"(dritt|viert|fünft|sechst|siebt|acht|neunt|zehnt|zwanzigst|hundertst|tausendst)el\w*)$"
)
_MULTIPLIER = re.compile(r"^(doppelt\w*|zweierlei|dreierlei)$")
_QUANTITY_NOUN = re.compile(r"^(dutzend\w*|hunderte\w*|tausende\w*|million\w*|milliard\w*)$")
_PAAR = re.compile(r"^Paar(e|en)?$")
_ROMAN = re.compile(r"^(?=[IVXLCDM]{2,}$)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_SINGLE_AFTER = re.compile(r"(?:CEM|Teil|Klasse|Typ|Stufe|Kategorie)\s+$")
_WORD = re.compile(r"(?<![\w])[A-Za-zÄÖÜäöüß]+(?![\w])")
_UNIT_EXPONENT_BEFORE = re.compile(r"(?:^|[^A-Za-z])(?:m|mm|cm|dm|km)$")


@dataclass(frozen=True)
class Numeral:
    kind: NumeralKind
    start: int
    end: int
    text: str


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


def parse_cardinal(word: str) -> int | None:
    """Value of a German cardinal number word below one million, otherwise ``None``."""
    lowered = word.lower()
    if lowered in ARTICLES:
        return None
    if "tausend" not in lowered:
        return _below_thousand(lowered)
    head, _, tail = lowered.partition("tausend")
    thousands = 1 if head in {"", "ein"} else _below_thousand(head)
    rest = 0 if tail == "" else _below_thousand(tail)
    if thousands is None or rest is None:
        return None
    return thousands * 1000 + rest


def _is_ordinal(word: str) -> bool:
    for ending in _ORDINAL_ENDINGS:
        if not word.endswith(ending):
            continue
        stem = word[: -len(ending)]
        if stem in _ORDINAL_STEMS:
            return True
        # 13th and above: cardinal + "t" (dreizehnte) or + "st" (zwanzigste, hundertste)
        for suffix in ("st", "t"):
            if stem.endswith(suffix) and parse_cardinal(stem[: -len(suffix)]) is not None:
                return True
    return False


def _is_multiplier(word: str) -> bool:
    if _MULTIPLIER.match(word):
        return True
    for suffix in ("fach", "fache", "fachen", "facher", "faches", "mal", "malig", "maligen"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if stem and stem != "ein" and parse_cardinal(stem) is not None:
                return True
    return False


def is_number_word(word: str) -> bool:
    """Whether a German word expresses a number (see the module docstring for the scope)."""
    if _PAAR.match(word):
        return True
    lowered = word.lower()
    if lowered in ARTICLES:
        return False
    return bool(
        parse_cardinal(lowered) is not None
        or _is_ordinal(lowered)
        or _FRACTION.match(lowered)
        or _is_multiplier(lowered)
        or _QUANTITY_NOUN.match(lowered)
    )


def find_numerals(text: str, *, include_ascii_digits: bool = True) -> list[Numeral]:
    """Every numeral in ``text``: digits, numeric symbols, Roman numerals and number words."""
    found: list[Numeral] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isascii() and char.isdigit():
            end = index
            while end < len(text) and text[end].isascii() and text[end].isdigit():
                end += 1
            if include_ascii_digits:
                found.append(Numeral("digit", index, end, text[index:end]))
            index = end
            continue
        if unicodedata.category(char) in {"Nd", "Nl", "No"}:
            exponent = char in "²³" and _UNIT_EXPONENT_BEFORE.search(text[:index])
            if not exponent:
                found.append(Numeral("symbol", index, index + 1, char))
        index += 1
    for match in _WORD.finditer(text):
        word = match.group()
        if _ROMAN.match(word) or (
            word in {"I", "V", "X"} and _ROMAN_SINGLE_AFTER.search(text[: match.start()])
        ):
            found.append(Numeral("roman", match.start(), match.end(), word))
        elif is_number_word(word):
            found.append(Numeral("word", match.start(), match.end(), word))
    return sorted(found, key=lambda numeral: numeral.start)
