"""Guarded report generation: schema validation, faithfulness gate and template fallback.

Policy:

1. Ask the model for a JSON draft constrained by the per-case schema.
2. If the reply is not a valid draft, retry up to ``max_schema_retries`` times with the
   validation error; then give up with status ``invalid``.
3. Check every number. If any violates the facts, send one corrective prompt that lists the
   offending numbers (``max_correction_retries``); if the new draft still violates, the
   report is ``rejected``.
4. Only an ``accepted`` model report is delivered. Every other outcome, including transport
   errors and replay misses, delivers the deterministic template report and says so.
"""

import hashlib

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
from baulens.report.replay import RecordingClient, ReplayMissError
from baulens.report.schemas import (
    Attempt,
    AttemptKind,
    FaithfulnessResult,
    InspectionReport,
    ReportDraft,
    ReportResult,
    Status,
)
from baulens.report.template import template_report
from baulens.schemas import InspectionInput


def input_sha256(case: InspectionInput) -> str:
    """Stable digest of an input; key of recorded replays and of run outputs."""
    return hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()


def generation_options(config: ReportConfig) -> GenerationOptions:
    return GenerationOptions(
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
        num_ctx=config.num_ctx,
    )


def _report(
    sheet: FactSheet, draft: ReportDraft, model: str, prompt: PromptTemplate
) -> InspectionReport:
    return InspectionReport(
        case_id=sheet.case.case_id,
        draft=draft,
        generator=model,
        prompt_version=prompt.prompt_version,
    )


def generate_template(
    case: InspectionInput, *, note: str | None = None, config: ReportConfig | None = None
) -> ReportResult:
    """Template report with its checks, in the same result shape as a model report."""
    sheet = build_facts(case)
    prompt = load_prompt(config.prompt_template if config else "report_v1")
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


def generate_report(case: InspectionInput, client: LLMClient, config: ReportConfig) -> ReportResult:
    """Draft a report with ``client`` and deliver it only if every number is faithful."""
    sheet = build_facts(case)
    prompt = load_prompt(config.prompt_template)
    digest = input_sha256(case)
    if isinstance(client, RecordingClient):
        client.begin_case(case.case_id, digest, prompt.prompt_version)
    schema = draft_schema(sheet)
    options = generation_options(config)
    messages: list[ChatMessage] = initial_messages(sheet, prompt)

    attempts: list[Attempt] = []
    candidate: InspectionReport | None = None
    initial: FaithfulnessResult | None = None
    final: FaithfulnessResult | None = None
    schema_retries = corrections = 0
    status: Status
    note: str | None = None
    kind: AttemptKind = "initial"

    while True:
        try:
            response = client.generate(messages, schema, options)
        except ReplayMissError as error:
            note = f"Keine aufgezeichnete Antwort ({error}); Vorlagenbericht geliefert"
            return generate_template(case, config=config, note=note)
        except LLMError as error:
            status, note = "error", f"Modellaufruf fehlgeschlagen: {error}"
            break
        try:
            draft = parse_draft(response.text, sheet)
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
        candidate = _report(sheet, draft, client.model, prompt)
        final = check_sections(draft.sections(), sheet)
        if initial is None:
            initial = final
        attempts.append(Attempt(kind=kind, response=response, violations=len(final.violations)))
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

    if status == "rejected":
        note = "Zahlen nicht durch die Eingabe gedeckt; Vorlagenbericht geliefert"
    elif status == "invalid":
        note = "Keine schemagültige Antwort; Vorlagenbericht geliefert"
    delivered = candidate if status == "accepted" and candidate is not None else None
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
        prompt_version=prompt.prompt_version,
        status=status,
        delivered=delivered or template_report(sheet),
        candidate=candidate,
        initial_faithfulness=initial,
        faithfulness=final,
        completeness=completeness,
        attempts=attempts,
        latency_s=sum(attempt.response.latency_s for attempt in attempts),
        fallback_used=delivered is None,
        note=note,
    )
