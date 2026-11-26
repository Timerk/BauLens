"""Record model replies and serve them again without a model (public demo, CI).

A replay file ``data/report_replays/<model>.jsonl`` holds one :class:`ReplayRecord` per model
call, for both generation modes. Records are keyed by the SHA-256 of the model, the full
input digest (including fields the prompt does not show, such as the image hash), the prompt
version, the exact chat messages and the JSON schema. ``case_id``, ``input_sha256`` and
``prompt_version`` are stored in clear as well so the file stays auditable. Because
decoding runs with temperature 0 and a fixed seed, replaying the recorded replies walks the
same path through :func:`baulens.report.generate.generate_report` as the live run, including
corrective retries.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from baulens.report.clients import GenerationOptions, LLMClient
from baulens.report.prompting import ChatMessage
from baulens.report.schemas import LLMResponse, Source


class ReplayMissError(LookupError):
    """No recorded reply exists for these messages."""


class ReplayRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    model: str
    case_id: str
    input_sha256: str
    prompt_version: str
    call: int
    response: LLMResponse


def request_key(
    model: str,
    messages: Sequence[ChatMessage],
    schema: dict[str, Any],
    *,
    input_sha256: str,
    prompt_version: str,
) -> str:
    """Key of one model call: the exact input (incl. image hash), prompt, messages and schema."""
    payload = {
        "model": model,
        "input_sha256": input_sha256,
        "prompt_version": prompt_version,
        "messages": [message.model_dump() for message in messages],
        "schema": schema,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def replay_path(directory: Path, model: str) -> Path:
    """File name for a model tag; ``:`` and ``/`` are not portable in file names."""
    return directory / f"{model.replace(':', '_').replace('/', '_')}.jsonl"


def load_records(path: Path) -> list[ReplayRecord]:
    with path.open(encoding="utf-8") as handle:
        return [ReplayRecord.model_validate_json(line) for line in handle if line.strip()]


def write_records(path: Path, records: Sequence[ReplayRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: (record.case_id, record.call))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(record.model_dump_json() + "\n")


class ReplayClient:
    """Serves recorded replies; raises :class:`ReplayMissError` for anything unrecorded."""

    def __init__(self, model: str, records: Sequence[ReplayRecord]) -> None:
        self._case: tuple[str, str, str] | None = None
        self._model = model
        self._records = {record.key: record for record in records if record.model == model}

    @classmethod
    def from_directory(cls, directory: Path, model: str) -> "ReplayClient":
        path = replay_path(directory, model)
        return cls(model, load_records(path) if path.exists() else [])

    @property
    def model(self) -> str:
        return self._model

    @property
    def source(self) -> Source:
        return "replay"

    def begin_case(self, case_id: str, input_sha256: str, prompt_version: str) -> None:
        self._case = (case_id, input_sha256, prompt_version)

    def __len__(self) -> int:
        return len(self._records)

    def generate(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        options: GenerationOptions,
    ) -> LLMResponse:
        if self._case is None:
            raise ReplayMissError("no case context; call begin_case() first")
        _, input_sha256, prompt_version = self._case
        key = request_key(
            self._model,
            messages,
            schema,
            input_sha256=input_sha256,
            prompt_version=prompt_version,
        )
        record = self._records.get(key)
        if record is None:
            raise ReplayMissError(f"no recorded reply of {self._model} for this input")
        return record.response.model_copy(update={"source": "replay"})


class RecordingClient:
    """Wraps a client and keeps every reply so it can be written as a replay file."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.records: list[ReplayRecord] = []
        self._case: tuple[str, str, str] | None = None
        self._calls = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def source(self) -> Source:
        return self._inner.source

    def begin_case(self, case_id: str, input_sha256: str, prompt_version: str) -> None:
        self._case = (case_id, input_sha256, prompt_version)
        self._calls = 0

    def generate(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        options: GenerationOptions,
    ) -> LLMResponse:
        if self._case is None:
            raise RuntimeError("call begin_case() before recording")
        case_id, input_sha256, prompt_version = self._case
        response = self._inner.generate(messages, schema, options)
        self.records.append(
            ReplayRecord(
                key=request_key(
                    self.model,
                    messages,
                    schema,
                    input_sha256=input_sha256,
                    prompt_version=prompt_version,
                ),
                model=self.model,
                case_id=case_id,
                input_sha256=input_sha256,
                prompt_version=prompt_version,
                call=self._calls,
                response=response,
            )
        )
        self._calls += 1
        return response
