"""LLM clients behind one small protocol: Ollama over HTTP, a scripted fake for tests."""

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from baulens.report.prompting import ChatMessage
from baulens.report.schemas import LLMResponse, Source


class LLMError(RuntimeError):
    """The model service failed or returned an unusable HTTP response."""


class GenerationOptions(BaseModel):
    """Decoding options; identical for every model in a comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = Field(ge=0, le=2)
    seed: int
    max_tokens: int = Field(gt=0)
    num_ctx: int = Field(gt=0)


class LLMClient(Protocol):
    """Anything that turns chat messages and a JSON schema into one reply."""

    @property
    def model(self) -> str: ...

    @property
    def source(self) -> Source: ...

    def generate(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        options: GenerationOptions,
    ) -> LLMResponse: ...


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


class OllamaClient:
    """Minimal client for Ollama's ``/api/chat`` with structured outputs."""

    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout_s: float = 600.0,
        keep_alive: str = "10m",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._keep_alive = keep_alive
        self._http = httpx.Client(
            base_url=host.rstrip("/"), timeout=httpx.Timeout(timeout_s), transport=transport
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def source(self) -> Source:
        return "live"

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(path, json=payload)
        except httpx.HTTPError as error:
            raise LLMError(f"Ollama request to {path} failed: {error}") from error
        if response.status_code != httpx.codes.OK:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
        except ValueError as error:
            raise LLMError("Ollama returned a non-JSON response") from error
        if not isinstance(data, dict):
            raise LLMError("Ollama returned an unexpected JSON payload")
        if "error" in data:
            raise LLMError(f"Ollama error: {data['error']}")
        return data

    def generate(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        options: GenerationOptions,
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "format": schema,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": options.temperature,
                "seed": options.seed,
                "num_predict": options.max_tokens,
                "num_ctx": options.num_ctx,
            },
        }
        started = time.perf_counter()
        data = self._post("/api/chat", payload)
        latency = time.perf_counter() - started
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError("Ollama response has no message content")
        return LLMResponse(
            text=message["content"],
            model=str(data.get("model", self._model)),
            source="live",
            done_reason=data.get("done_reason"),
            latency_s=latency,
            eval_count=_int_or_none(data.get("eval_count")),
            eval_duration_ns=_int_or_none(data.get("eval_duration")),
            prompt_eval_count=_int_or_none(data.get("prompt_eval_count")),
            prompt_eval_duration_ns=_int_or_none(data.get("prompt_eval_duration")),
            load_duration_ns=_int_or_none(data.get("load_duration")),
            total_duration_ns=_int_or_none(data.get("total_duration")),
        )

    def load(self) -> float:
        """Load the model into memory and return the reported load time in seconds."""
        data = self._post("/api/generate", {"model": self._model, "keep_alive": self._keep_alive})
        return (_int_or_none(data.get("load_duration")) or 0) / 1e9

    def unload(self) -> None:
        self._post("/api/generate", {"model": self._model, "keep_alive": 0})

    def loaded_size_bytes(self) -> int | None:
        """Memory Ollama reports for this model in ``/api/ps`` (weights, KV cache, buffers)."""
        try:
            response = self._http.get("/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        for entry in models:
            if isinstance(entry, dict) and entry.get("name") == self._model:
                return _int_or_none(entry.get("size"))
        return None

    def version(self) -> str | None:
        try:
            response = self._http.get("/api/version")
            response.raise_for_status()
            return str(response.json().get("version"))
        except (httpx.HTTPError, ValueError, AttributeError):
            return None


Script = Callable[[Sequence[ChatMessage]], str]


class FakeClient:
    """Test double that replies from a list of texts or a function of the messages."""

    def __init__(
        self, replies: Sequence[str] | Script, *, model: str = "fake", latency_s: float = 0.01
    ) -> None:
        self._replies = replies
        self._model = model
        self._latency_s = latency_s
        self.calls: list[list[ChatMessage]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def source(self) -> Source:
        return "fake"

    def generate(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        options: GenerationOptions,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if callable(self._replies):
            text = self._replies(messages)
        else:
            index = len(self.calls) - 1
            if index >= len(self._replies):
                raise LLMError("fake client has no scripted reply left")
            text = self._replies[index]
        return LLMResponse(
            text=text,
            model=self._model,
            source="fake",
            done_reason="stop",
            latency_s=self._latency_s,
            eval_count=max(1, len(text) // 4),
            eval_duration_ns=int(self._latency_s * 1e9),
        )
