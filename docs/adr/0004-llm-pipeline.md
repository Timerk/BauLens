# 0004 Hand-written LLM pipeline on httpx with a numeric faithfulness gate

Accepted on 2026-11-26.

## Context

The report module drafts German inspection reports from the strength and crack predictions
(`InspectionInput`). An inspection report that states a wrong number is worse than no report,
so the pipeline must guarantee that every number in a delivered draft comes from the input.
It has to run on a CPU-only laptop with small local models, be testable in CI without network
access or a running model, and be replayable in the public demo without a model server.

The pipeline needs only four things: one chat call with a JSON schema, validation of the
reply, a corrective second call, and recording of replies. There is no retrieval, no tool
use and no agent loop.

## Options

| Option | For | Against |
| --- | --- | --- |
| **Hand-written pipeline on `httpx`** (chosen) | About 150 lines for the client; full control over the request (`format`, `seed`, `num_ctx`, `keep_alive`) and over timing counters; `httpx.MockTransport` makes HTTP handling testable; no new dependency | We maintain the retry and replay logic ourselves |
| LangChain (`langchain-ollama`) | Prompt templates, output parsers and retry helpers ready-made; familiar to many readers | Large dependency tree and frequent API changes for a two-call pipeline; abstractions hide the raw request and Ollama's `eval_count`/`eval_duration`; structured-output and retry helpers would still need our own faithfulness check; not in the lockfile |
| Ollama Python client (`ollama`) | Official, thin, typed | Adds a dependency for what one `POST /api/chat` does; testing still needs a transport mock; no benefit for replay |
| llama.cpp server (OpenAI-compatible API) | Fewer layers, GBNF grammars, very good CPU performance | Model management (GGUF downloads, chat templates) by hand; Ollama already wraps llama.cpp with the same grammar-constrained decoding |
| vLLM | High throughput, batching, production serving | Needs a CUDA or ROCm GPU; this machine has neither; overkill for 28 cases |

## Decision

- **Serving:** Ollama (v0.40) on the developer machine. Models are addressed by tag
  (`qwen2.5:3b`, `llama3.2:3b`, `gemma3:4b`), so swapping models is configuration.
- **Client:** a `LLMClient` protocol (`generate(messages, schema, options) -> LLMResponse`) with
  three implementations: `OllamaClient` (httpx, `/api/chat`, structured outputs via `format`),
  `FakeClient` (scripted replies for tests, including deliberately unfaithful ones) and
  `ReplayClient` (recorded replies keyed by the SHA-256 of messages and schema).
- **Decoding:** temperature 0 and a fixed seed for every model, and a per-case JSON schema that
  contains exactly the report sections the case needs.
- **Facts:** one module renders the input as a German fact list. The prompt shows these
  strings and the checker accepts exactly their documented renderings, so prompt and checker
  cannot disagree about what a number may look like.
- **Gate:** schema validation with one bounded retry; numeric faithfulness check; one corrective
  retry that lists the offending numbers; otherwise the draft is rejected. Only accepted drafts
  are delivered; every other outcome (rejected, invalid, service error, replay miss) delivers a
  deterministic template report and says so in `ReportResult`.
- **Disclaimer:** inserted by code, never generated.

## Consequences

- The pipeline has no dependency beyond the locked stack (`httpx`, `pydantic`, `pyyaml`,
  `psutil`), and CI tests the whole generate → check → reject path offline.
- Replays make the public demo (#5) deterministic and free of model hosting, but any change to
  the prompt template, the fact rendering or the case inputs invalidates recorded replies;
  `baulens report compare` re-records them.
- The faithfulness check is the safety property, not the model: a model that ignores the rules
  produces rejected drafts and template reports, never unfaithful deliveries. The check
  verifies numbers only; qualitative claims stay the reviewer's job (see
  `docs/results/report.md`).
- Moving to another server (llama.cpp server, vLLM on a GPU machine, a hosted API) means one
  new `LLMClient` implementation; prompt, checker, replays and metrics stay unchanged.
