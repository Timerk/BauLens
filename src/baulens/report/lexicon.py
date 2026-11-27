"""Word lists that bind a number to its meaning, shared by the free-text and the slot checks.

All patterns are case-insensitive regular expressions on German text. They are deliberately
small and listed in one place so that a reviewer can see exactly what "the sentence names the
quantity" means.
"""

import re

DEADLINE = re.compile(
    r"frist|termin|deadline|fällig|spätestens|wartezeit|nach\s+ablauf|bis\s+zum|innerhalb\s+von"
    r"|dauer|zeitraum|zeitspanne|nachprüfung|wiederhol\w*\s+(nach|in)",
    re.IGNORECASE,
)
"""A number used as a deadline, duration or schedule is never a fact of the input."""

CAPTURE = re.compile(r"aufgenommen|aufnahme|erfasst|fotografiert|bildaufnahme", re.IGNORECASE)
"""The capture date and time may only be stated as the time the image was taken."""

NEGATION = re.compile(
    r"\b(nicht|nie|niemals|weder|statt|anstatt)\b|\bkein\w*\b(?!\s+(?:\w+\s+)?Riss)",
    re.IGNORECASE,
)
"""Slot mode: a placeholder must not share a clause with a negation ("beträgt nicht …").
"kein Riss erkannt" is the crack finding itself and not a negation of a number."""

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")
_CLAUSE_END = re.compile(r"[.!?](?=\s|$)|[;\n,]|\bund\b|\bsowie\b|\bbzw\.|\boder\b")


def sentence_around(text: str, start: int, end: int) -> str:
    """The sentence of ``text`` that contains ``text[start:end]``."""
    begin = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    stop = _SENTENCE_END.search(text, end)
    return text[begin : stop.start() if stop else len(text)]


def clause_around(text: str, start: int, end: int) -> str:
    """The clause (sentence part between commas, "und", "sowie", …) around a span."""
    begin = max((m.end() for m in _CLAUSE_END.finditer(text, 0, start)), default=0)
    stop = _CLAUSE_END.search(text, end)
    return text[begin : stop.start() if stop else len(text)]
