"""Guarded report generation: schema validation, faithfulness gate and template fallback.

Two modes share one policy:

``slots`` (production default)
    The model writes prose with placeholders such as ``{druckfestigkeit}``; code inserts the
    numbers. Any digit, number word, date or time the model writes itself is a violation,
    and the rendered draft is checked again so that a placeholder in the wrong place
    ("Die Druckfestigkeit beträgt {obere_grenze}") is caught as ``misattributed``.
``free``
    The model writes every number itself; the checker verifies form, unit and role. Kept as
    the evaluated unconstrained baseline.

Policy:

1. Ask the model for a JSON draft constrained by the per-case schema.
2. If the reply is not a valid draft, retry up to ``max_schema_retries`` times with the
   validation error; then give up with status ``invalid``.
3. Check every number. If any violates the facts, send one corrective prompt that lists
   the violations (``max_correction_retries``); if the new draft still violates, the report
   is ``rejected``.
4. Only an ``accepted`` model report is delivered. Every other outcome (rejected, invalid,
   service error, replay miss) delivers the deterministic template report, labelled as such.
"""

import hashlib
import time

from pydantic import ValidationError

from baulens.config import ReportConfig
from baulens.report.clients import GenerationOptions, LLMClient, LLMError
from baulens.report.completeness import check_completeness
from baulens.report.facts import FactSheet, build_facts
from baulens.report.faithfulness import check_sections
from baulens.report.prompting import (
    ChatMessage,
    DraftError,
    PromptTemplate,
    correction_message,
    draft_schema,
    initial_messages,
    load_prompt,
    parse_draft,
    schema_retry_message,
)
from baulens.report.replay import RecordingClient, ReplayClient, ReplayMissError
from baulens.report.schemas import (
    Attempt,
    AttemptKind,
    FaithfulnessResult,
    InspectionReport,
    Mode,
    ReportDraft,
    ReportResult,
    Status,
)
from baulens.report.slots import check_slot_sections, render_slots
from baulens.report.template import TEMPLATE_PROVENANCE, template_report
from baulens.schemas import InspectionInput


def input_sha256(case: InspectionInput) -> str:
    """Stable digest of the full input (including image hash); keys replays and resumes."""
    return hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()


def generation_options(config: ReportConfig) -> GenerationOptions:
    return GenerationOptions(
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
        num_ctx=config.num_ctx,
    )


def prompt_for(config: ReportConfig, mode: Mode) -> PromptTemplate:
    return load_prompt(config.slot_prompt_template if mode == "slots" else config.prompt_template)


def _provenance(client: LLMClient, mode: Mode, prompt: PromptTemplate) -> str:
    kind = {
        "live": "Modellausgabe",
        "replay": "Aufgezeichnete Modellausgabe (Replay)",
        "fake": "Testausgabe",
        "template": "Vorlage",
    }[client.source]
    return f"{kind} von {client.model}, Modus {mode}, Prompt {prompt.prompt_version}"


def _fallback(sheet: FactSheet, reason: str) -> InspectionReport:
    report = template_report(sheet)
    return report.model_copy(update={"provenance": f"{TEMPLATE_PROVENANCE} (Ersatz: {reason})"})


def generate_template(
    case: InspectionInput,
    *,
    note: str | None = None,
    config: ReportConfig | None = None,
) -> ReportResult:
    """Template report with its checks, in the same result shape as a model report."""
    sheet = build_facts(case)
    prompt = load_prompt(config.prompt_template) if config is not None else load_prompt()
    report = template_report(sheet)
    faithfulness = check_sections(report.draft.sections(), sheet)
    return ReportResult(
        case_id=case.case_id,
        input_sha256=input_sha256(case),
        source="template",
        model=None,
        prompt_version=prompt.prompt_version,
        status="template",
        delivered=report,
        candidate=report,
        initial_faithfulness=faithfulness,
        faithfulness=faithfulness,
        completeness=check_completeness(report, faithfulness, sheet),
        note=note,
    )


def _check(
    raw: ReportDraft, sheet: FactSheet, mode: Mode
) -> tuple[ReportDraft, FaithfulnessResult]:
    """The draft to show and its check. Slot drafts are rendered before display."""
    if mode == "free":
        return raw, check_sections(raw.sections(), sheet)
    result = check_slot_sections(raw.sections(), sheet)
    rendered = {name: render_slots(text, sheet) for name, text in raw.sections().items()}
    try:
        return ReportDraft.model_validate(rendered), result
    except ValidationError as error:
        # e.g. a long element name repeated through {bauteil} exceeds a section limit
        raise DraftError(
            f"nach dem Einsetzen der Platzhalter ungültig: {error.errors()[0]['msg']}"
        ) from error


def generate_report(
    case: InspectionInput,
    client: LLMClient,
    config: ReportConfig,
    *,
    mode: Mode | None = None,
) -> ReportResult:
    """Draft a report with ``client`` and deliver it only if every number is faithful."""
    mode = mode or config.generation_mode
    sheet = build_facts(case)
    prompt = prompt_for(config, mode)
    digest = input_sha256(case)
    if isinstance(client, RecordingClient | ReplayClient):
        client.begin_case(case.case_id, digest, prompt.prompt_version)
    schema = draft_schema(sheet)
    options = generation_options(config)
    messages: list[ChatMessage] = initial_messages(sheet, prompt)

    attempts: list[Attempt] = []
    candidate: InspectionReport | None = None
    initial: FaithfulnessResult | None = None
    final: FaithfulnessResult | None = None
    schema_retries = corrections = 0
    failed_call_s = 0.0
    status: Status
    note: str | None = None
    kind: AttemptKind = "initial"

    while True:
        started = time.perf_counter()
        try:
            response = client.generate(messages, schema, options)
        except ReplayMissError as error:
            status, note = "replay_miss", f"Keine aufgezeichnete Antwort ({error})"
            break
        except LLMError as error:
            failed_call_s += time.perf_counter() - started
            status, note = "error", f"Modellaufruf fehlgeschlagen: {error}"
            break
        try:
            raw = parse_draft(response.text, sheet)
            draft, final = _check(raw, sheet, mode)
        except DraftError as error:
            attempts.append(Attempt(kind=kind, response=response, schema_error=str(error)))
            if schema_retries >= config.max_schema_retries:
                status = "rejected" if candidate is not None else "invalid"
                break
            schema_retries += 1
            messages += [
                ChatMessage(role="assistant", content=response.text),
                schema_retry_message(error, prompt),
            ]
            kind = "schema_retry"
            continue
        if initial is None:
            initial = final
        candidate = InspectionReport(
            case_id=case.case_id,
            draft=draft,
            generator=client.model,
            prompt_version=prompt.prompt_version,
            provenance=_provenance(client, mode, prompt),
        )
        attempts.append(
            Attempt(
                kind=kind,
                response=response,
                violations=len(final.violations),
                raw_draft=(
                    {str(name): text for name, text in raw.sections().items()}
                    if mode == "slots"
                    else None
                ),
            )
        )
        if final.faithful:
            status = "accepted"
            break
        if corrections >= config.max_correction_retries:
            status = "rejected"
            break
        corrections += 1
        messages += [
            ChatMessage(role="assistant", content=response.text),
            correction_message(final, prompt),
        ]
        kind = "correction"

    reasons = {
        "rejected": "Zahlen nicht durch die Eingabe gedeckt",
        "invalid": "keine schemagültige Antwort",
        "error": "Modellaufruf fehlgeschlagen",
        "replay_miss": "keine aufgezeichnete Modellausgabe für diese Eingabe",
    }
    delivered = candidate if status == "accepted" and candidate is not None else None
    if delivered is None:
        reason = reasons.get(status, status)
        note = f"{note + '; ' if note else ''}{reason}; Vorlagenbericht geliefert"
    completeness = (
        check_completeness(candidate, final, sheet)
        if candidate is not None and final is not None
        else None
    )
    return ReportResult(
        case_id=case.case_id,
        input_sha256=digest,
        source=client.source,
        model=client.model,
        mode=mode,
        prompt_version=prompt.prompt_version,
        status=status,
        delivered=delivered or _fallback(sheet, reasons.get(status, status)),
        candidate=candidate,
        initial_faithfulness=initial,
        faithfulness=final,
        completeness=completeness,
        attempts=attempts,
        latency_s=sum(attempt.response.latency_s for attempt in attempts),
        failed_call_s=failed_call_s,
        fallback_used=delivered is None,
        note=note,
    )
