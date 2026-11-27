"""Output contracts of the report generator: the draft, the checks and the run result."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SectionName = Literal[
    "zusammenfassung",
    "druckfestigkeit",
    "rissbefund",
    "bewertung",
    "einschraenkungen",
    "empfehlung",
]
SECTION_ORDER: tuple[SectionName, ...] = (
    "zusammenfassung",
    "druckfestigkeit",
    "rissbefund",
    "bewertung",
    "einschraenkungen",
    "empfehlung",
)
SECTION_TITLES: dict[SectionName, str] = {
    "zusammenfassung": "Zusammenfassung",
    "druckfestigkeit": "Druckfestigkeit",
    "rissbefund": "Rissbefund",
    "bewertung": "Bewertung",
    "einschraenkungen": "Einschränkungen",
    "empfehlung": "Empfehlung",
}
FACT_SECTIONS: tuple[SectionName, ...] = ("zusammenfassung", "druckfestigkeit", "rissbefund")
"""Narrative mode: rendered by code from the input ("Befunde")."""
NARRATIVE_SECTIONS: tuple[SectionName, ...] = ("bewertung", "einschraenkungen", "empfehlung")
"""Narrative mode: written by the model (or the template fallback), checked to contain no
numbers and no findings."""
MAX_NARRATIVE_CHARS = 900
MAX_SECTION_CHARS = 2000

DISCLAIMER = (
    "Hinweis: Dieser Prüfberichtsentwurf wurde automatisiert erstellt und enthält "
    "Modellschätzungen. Er ist kein Prüfbericht und muss vor jeder Verwendung von einer "
    "fachkundigen Ingenieurin oder einem fachkundigen Ingenieur geprüft werden. Er ersetzt "
    "weder eine normgerechte Druckfestigkeitsprüfung noch eine Begutachtung vor Ort."
)
"""Inserted by code, never generated. Deliberately free of numbers."""

Source = Literal["live", "replay", "fake", "template"]
Status = Literal["accepted", "rejected", "invalid", "error", "replay_miss", "template"]
Mode = Literal["free", "narrative"]
CheckStatus = Literal["ok", "invented", "misattributed", "inexact", "unbound", "restated"]
ViolationKind = Literal["invented", "misattributed", "inexact", "unbound", "restated"]
VIOLATION_KINDS: tuple[ViolationKind, ...] = (
    "invented",
    "misattributed",
    "inexact",
    "unbound",
    "restated",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReportDraft(_Model):
    """The part of the report written by the model (or by the template)."""

    zusammenfassung: str = Field(min_length=1, max_length=MAX_SECTION_CHARS)
    druckfestigkeit: str | None = Field(default=None, min_length=1, max_length=MAX_SECTION_CHARS)
    rissbefund: str | None = Field(default=None, min_length=1, max_length=MAX_SECTION_CHARS)
    bewertung: str | None = Field(default=None, min_length=1, max_length=MAX_SECTION_CHARS)
    einschraenkungen: str = Field(min_length=1, max_length=MAX_SECTION_CHARS)
    empfehlung: str = Field(min_length=1, max_length=MAX_SECTION_CHARS)

    def sections(self) -> dict[SectionName, str]:
        """Non-empty sections in report order."""
        values: dict[SectionName, str | None] = {
            name: getattr(self, name) for name in SECTION_ORDER
        }
        return {name: text for name, text in values.items() if text is not None}


class InspectionReport(_Model):
    """German inspection report draft ("Prüfberichtsentwurf")."""

    case_id: str
    draft: ReportDraft
    disclaimer: str = DISCLAIMER
    generator: str = Field(description="Model name, or 'template' for the deterministic text")
    prompt_version: str | None = None
    provenance: str = Field(
        default="Vorlagenbericht ohne Sprachmodell",
        description="Shown with the report: live model output, recorded replay or template",
    )

    def to_markdown(self) -> str:
        lines = [f"# Prüfberichtsentwurf {self.case_id}", ""]
        for name, text in self.draft.sections().items():
            lines += [f"## {SECTION_TITLES[name]}", "", text.strip(), ""]
        lines += [f"> {self.disclaimer}", "", f"_Herkunft: {self.provenance}_"]
        return "\n".join(lines) + "\n"


class NumberCheck(_Model):
    """Verdict on one numeric token in the generated text."""

    section: str
    start: int
    end: int
    text: str
    kind: Literal["number", "numeral", "date", "time", "reference", "brace", "finding"]
    value: str | None = None
    unit: str | None = None
    status: CheckStatus
    matched_fields: list[str] = Field(default_factory=list)
    reason: str


class MaskedText(_Model):
    """A span excluded from number extraction, listed so the exclusion stays auditable."""

    section: str
    start: int
    end: int
    text: str
    kind: Literal["identifier", "standard"]


class FaithfulnessResult(_Model):
    checks: list[NumberCheck]
    masked: list[MaskedText] = Field(default_factory=list)

    @property
    def violations(self) -> list[NumberCheck]:
        return [check for check in self.checks if check.status != "ok"]

    @property
    def faithful(self) -> bool:
        return not self.violations

    def violation_counts(self) -> dict[ViolationKind, int]:
        counts = dict.fromkeys(VIOLATION_KINDS, 0)
        for check in self.violations:
            if check.status != "ok":
                counts[check.status] += 1
        return counts

    def matched_fields(self) -> set[str]:
        return {
            field for check in self.checks if check.status == "ok" for field in check.matched_fields
        }


class CompletenessItem(_Model):
    name: str
    satisfied: bool
    detail: str


class CompletenessResult(_Model):
    items: list[CompletenessItem]

    @property
    def score(self) -> float:
        return sum(item.satisfied for item in self.items) / len(self.items) if self.items else 1.0

    @property
    def missing(self) -> list[str]:
        return [item.name for item in self.items if not item.satisfied]


class LLMResponse(_Model):
    """One model reply with Ollama's timing counters (nanoseconds) when available."""

    text: str
    model: str
    source: Source
    done_reason: str | None = None
    latency_s: float = Field(ge=0)
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    load_duration_ns: int | None = None
    total_duration_ns: int | None = None


AttemptKind = Literal["initial", "schema_retry", "correction"]


class Attempt(_Model):
    kind: AttemptKind
    response: LLMResponse
    schema_error: str | None = None
    violations: int | None = None
    raw_draft: dict[str, str] | None = Field(
        default=None, description="Slot mode: the draft with placeholders, before rendering"
    )


class ReportResult(_Model):
    """Outcome of one guarded generation.

    ``delivered`` is what a caller may show: the model's report when it passed the
    faithfulness check, otherwise the deterministic template report. ``candidate`` keeps the
    last schema-valid model output for inspection, including rejected ones.
    """

    case_id: str
    input_sha256: str
    source: Source
    model: str | None
    mode: Mode | None = None
    prompt_version: str
    status: Status
    delivered: InspectionReport
    candidate: InspectionReport | None = None
    initial_faithfulness: FaithfulnessResult | None = None
    faithfulness: FaithfulnessResult | None = None
    completeness: CompletenessResult | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    latency_s: float = Field(default=0.0, description="Sum over completed model calls")
    failed_call_s: float = Field(default=0.0, description="Wall time of calls that failed")
    fallback_used: bool = False
    note: str | None = None
