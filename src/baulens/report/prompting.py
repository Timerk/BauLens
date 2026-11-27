"""Versioned prompt template, chat messages and the per-case JSON schema."""

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from string import Template
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from baulens.report.facts import FactSheet
from baulens.report.faithfulness import describe_violations
from baulens.report.schemas import (
    MAX_NARRATIVE_CHARS,
    MAX_SECTION_CHARS,
    NARRATIVE_SECTIONS,
    SECTION_ORDER,
    SECTION_TITLES,
    FaithfulnessResult,
    Mode,
    ReportDraft,
    SectionName,
)
from baulens.report.template import template_facts

DEFAULT_TEMPLATE = "report_v2"
DEFAULT_NARRATIVE_TEMPLATE = "report_narrative_v1"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class DraftError(ValueError):
    """The model output is not a valid draft for this case."""


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    mode: Mode
    digest: str
    system: str
    user: str
    sections: dict[str, str]
    notes: dict[str, str]
    schema_retry: str
    correction: str

    @property
    def prompt_version(self) -> str:
        """Version plus content digest, used to key recorded replays."""
        return f"{self.version}+{self.digest[:8]}"


@cache
def load_prompt(name: str = DEFAULT_TEMPLATE) -> PromptTemplate:
    """Load ``prompts/<name>.yaml`` shipped with the package."""
    raw = resources.files("baulens.report").joinpath("prompts", f"{name}.yaml").read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    return PromptTemplate(
        name=name,
        version=str(data["version"]),
        mode="narrative" if data.get("mode") == "narrative" else "free",
        digest=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        system=str(data["system"]).strip(),
        user=str(data["user"]).strip(),
        sections={str(key): str(value) for key, value in data["sections"].items()},
        notes={str(key): str(value) for key, value in data["notes"].items()},
        schema_retry=str(data["schema_retry"]).strip(),
        correction=str(data["correction"]).strip(),
    )


def required_sections(sheet: FactSheet, mode: Mode = "free") -> tuple[SectionName, ...]:
    """Sections the model must write for this case, in report order."""
    if mode == "narrative":
        return NARRATIVE_SECTIONS
    case = sheet.case
    skip: set[SectionName] = {"bewertung"}
    if case.strength is None:
        skip.add("druckfestigkeit")
    if case.crack is None:
        skip.add("rissbefund")
    return tuple(name for name in SECTION_ORDER if name not in skip)


def draft_schema(sheet: FactSheet, mode: Mode = "free") -> dict[str, Any]:
    """JSON schema passed to Ollama's ``format`` so that decoding follows the sections."""
    names = required_sections(sheet, mode)
    limit = MAX_NARRATIVE_CHARS if mode == "narrative" else MAX_SECTION_CHARS
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "minLength": 1, "maxLength": limit} for name in names
        },
        "required": list(names),
        "additionalProperties": False,
    }


def _findings_text(sheet: FactSheet) -> str:
    return "\n\n".join(
        f"{SECTION_TITLES[name]}: {text}" for name, text in template_facts(sheet).items()
    )


def initial_messages(sheet: FactSheet, prompt: PromptTemplate) -> list[ChatMessage]:
    names = required_sections(sheet, prompt.mode)
    sections = "\n".join(f"- {name}: {prompt.sections[name]}" for name in names)
    if sheet.case.strength is not None and prompt.mode == "free":
        note = "in_domain" if sheet.case.strength.in_training_domain else "out_of_domain"
        sections += "\n\n" + prompt.notes[note]
    values = {"facts": sheet.render(), "sections": sections}
    if prompt.mode == "narrative":
        values["befunde"] = _findings_text(sheet)
    user = Template(prompt.user).substitute(values)
    return [
        ChatMessage(role="system", content=prompt.system),
        ChatMessage(role="user", content=user),
    ]


def schema_retry_message(error: DraftError, prompt: PromptTemplate) -> ChatMessage:
    return ChatMessage(
        role="user", content=Template(prompt.schema_retry).substitute(error=str(error))
    )


def correction_message(result: FaithfulnessResult, prompt: PromptTemplate) -> ChatMessage:
    violations = "\n".join(f"- {line}" for line in describe_violations(result))
    return ChatMessage(
        role="user", content=Template(prompt.correction).substitute(violations=violations)
    )


def _load_object(text: str) -> dict[str, Any]:
    """Parse a JSON object; every decoder failure becomes a ``DraftError``."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise DraftError(f"kein gültiges JSON ({error.msg} an Position {error.pos})") from error
    except (ValueError, RecursionError) as error:
        # e.g. integers beyond Python's digit limit, or pathological nesting
        raise DraftError(f"JSON nicht verarbeitbar ({error.__class__.__name__})") from error
    if not isinstance(data, dict):
        raise DraftError("JSON-Objekt erwartet")
    return data


def _sections(data: dict[str, Any], required: tuple[SectionName, ...]) -> None:
    missing = sorted(set(required) - data.keys())
    extra = sorted(set(data) - set(required))
    if missing or extra:
        raise DraftError(f"fehlende Felder: {missing or '-'}; unzulässige Felder: {extra or '-'}")


def _details(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )


def parse_draft(text: str, sheet: FactSheet) -> ReportDraft:
    """Free-text mode: validate a reply against the draft schema of this case."""
    data = _load_object(text)
    _sections(data, required_sections(sheet, "free"))
    try:
        return ReportDraft.model_validate(
            {key: value.strip() if isinstance(value, str) else value for key, value in data.items()}
        )
    except ValidationError as error:
        raise DraftError(_details(error)) from error


def parse_narrative(text: str) -> dict[SectionName, str]:
    """Narrative mode: validate the three narrative fields."""
    data = _load_object(text)
    _sections(data, NARRATIVE_SECTIONS)
    narrative: dict[SectionName, str] = {}
    for name in NARRATIVE_SECTIONS:
        value = data[name]
        if not isinstance(value, str) or not value.strip():
            raise DraftError(f"{name}: nicht leerer Text erwartet")
        if len(value) > MAX_NARRATIVE_CHARS:
            raise DraftError(f"{name}: höchstens {MAX_NARRATIVE_CHARS} Zeichen")
        narrative[name] = value.strip()
    return narrative
