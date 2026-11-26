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
    MAX_SECTION_CHARS,
    SECTION_ORDER,
    FaithfulnessResult,
    ReportDraft,
    SectionName,
)

DEFAULT_TEMPLATE = "report_v1"


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
        digest=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        system=str(data["system"]).strip(),
        user=str(data["user"]).strip(),
        sections={str(key): str(value) for key, value in data["sections"].items()},
        notes={str(key): str(value) for key, value in data["notes"].items()},
        schema_retry=str(data["schema_retry"]).strip(),
        correction=str(data["correction"]).strip(),
    )


def required_sections(sheet: FactSheet) -> tuple[SectionName, ...]:
    """Sections a draft must contain for this case, in report order."""
    case = sheet.case
    skip: set[SectionName] = set()
    if case.strength is None:
        skip.add("druckfestigkeit")
    if case.crack is None:
        skip.add("rissbefund")
    return tuple(name for name in SECTION_ORDER if name not in skip)


def draft_schema(sheet: FactSheet) -> dict[str, Any]:
    """JSON schema passed to Ollama's ``format`` so that decoding follows the sections."""
    names = required_sections(sheet)
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "minLength": 1, "maxLength": MAX_SECTION_CHARS}
            for name in names
        },
        "required": list(names),
        "additionalProperties": False,
    }


def initial_messages(sheet: FactSheet, prompt: PromptTemplate) -> list[ChatMessage]:
    lines = [f"- {name}: {prompt.sections[name]}" for name in required_sections(sheet)]
    sections = "\n".join(lines)
    if sheet.case.strength is not None and not sheet.case.strength.in_training_domain:
        sections += "\n\n" + prompt.notes["out_of_domain"]
    user = Template(prompt.user).substitute(facts=sheet.render(), sections=sections)
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


def parse_draft(text: str, sheet: FactSheet) -> ReportDraft:
    """Validate a model reply against the draft schema and the sections this case needs."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise DraftError(f"kein gültiges JSON ({error.msg} an Position {error.pos})") from error
    if not isinstance(data, dict):
        raise DraftError("JSON-Objekt erwartet")
    required = set(required_sections(sheet))
    missing = sorted(required - data.keys())
    extra = sorted(set(data) - required)
    if missing or extra:
        raise DraftError(f"fehlende Felder: {missing or '-'}; unzulässige Felder: {extra or '-'}")
    try:
        return ReportDraft.model_validate(
            {key: value.strip() if isinstance(value, str) else value for key, value in data.items()}
        )
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        raise DraftError(details) from error
