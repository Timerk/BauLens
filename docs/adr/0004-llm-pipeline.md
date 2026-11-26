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
  `ReplayClient` (recorded replies keyed by the SHA-256 of model, full input digest, prompt
  version, messages and schema).
- **Decoding:** temperature 0 and a fixed seed for every model, and a per-case JSON schema that
  contains exactly the report sections the case needs.
- **Facts:** one module renders every input number once, in German format and with enough
  precision to keep the decisions the input encodes (a crack probability of 0.4999 against a
  threshold of 0.5 renders as 49,99 % against 50 %). Each fact has a role (point estimate,
  lower and upper bound, coverage, crack probability, threshold, ingredient, age, …) and a
  slot name.
- **Slot generation as the production default** (added after the review of PR #11). The model
  writes prose with placeholders such as `{druckfestigkeit}`; code inserts the numbers. Any
  digit, number word, date or time the model writes itself is a violation, and the rendered
  text is checked again so that a placeholder in the wrong place ("Die Druckfestigkeit beträgt
  {obere_grenze} MPa") or with the wrong unit is caught.
- **Free-text generation as the evaluated baseline.** The model writes numbers itself and the
  checker accepts a number only in exactly the rendered form, with the fact's unit right after
  it, and with its role named by the surrounding words; it fails closed on ambiguity.
- **Gate (both modes):** schema validation with one bounded retry; numeric faithfulness check;
  one corrective retry that lists the violations; otherwise the draft is rejected. Only accepted drafts
  are delivered; every other outcome (rejected, invalid, service error, replay miss) delivers a
  deterministic template report and says so in `ReportResult`.
- **Disclaimer:** inserted by code, never generated.

## Consequences

- The pipeline has no dependency beyond the locked stack (`httpx`, `pydantic`, `pyyaml`,
  `psutil`), and CI tests the whole generate → check → reject path offline.
- Replays make the public demo (#5) deterministic and free of model hosting, but any change to
  the prompt template, the fact rendering or the case inputs invalidates recorded replies;
  `baulens report compare` re-records them.
- Why slots: checking free text is bounded by the parser's coverage. The review of PR #11
  found numbers that passed because they were bound to no quantity, rounded, written in
  another locale or hidden by an identifier mask; each fix narrows the gap, but a parser can
  never enumerate every way prose can state a number. Rendering the numbers in code makes the
  number channel faithful by construction, and the checker is left to guard misuse (literal
  numbers, wrong placeholder, wrong unit, missing role), a much smaller surface. The cost is a
  stricter prompt that small models follow less reliably; the comparison reports both modes.
- The faithfulness check is the safety property, not the model: a model that ignores the rules
  produces rejected drafts and template reports, never deliveries with detected violations.
  The check verifies numbers only; qualitative claims stay the reviewer's job (see
  `docs/results/report.md`).
- Moving to another server (llama.cpp server, vLLM on a GPU machine, a hosted API) means one
  new `LLMClient` implementation; prompt, checker, replays and metrics stay unchanged.
