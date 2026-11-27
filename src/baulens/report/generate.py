"""Guarded report generation: schema validation, faithfulness gate and template fallback.

Two modes share one policy:

``narrative`` (the delivery mode)
    Code renders every finding (summary, strength, crack: all numbers, units, intervals,
    comparisons, decisions, identifiers and dates). The model writes only an assessment,
    limitations and recommendations; :mod:`baulens.report.narrative` rejects any numeral,
    brace or restated finding in them. A rejected narrative is replaced by the template
    narrative, so the delivered report is always code findings plus a checked or a template
    narrative, labelled as such.
``free`` (experimental, not a delivery mode)
    The model writes every number itself and the checker verifies form, unit and role on a
    best-effort basis. Kept as the evaluated unconstrained baseline; its known false
    negatives are listed as ``xfail`` tests and in ``docs/results/report.md``.

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
from baulens.report.narrative import check_narrative
from baulens.report.prompting import (
    ChatMessage,
    DraftError,
    PromptTemplate,
    correction_message,
    draft_schema,
    initial_messages,
    load_prompt,
    parse_draft,
    parse_narrative,
    schema_retry_message,
)
from baulens.report.replay import RecordingClient, ReplayClient, ReplayMissError
from baulens.report.schemas import (
    FACT_SECTIONS,
    Attempt,
    AttemptKind,
    FaithfulnessResult,
    InspectionReport,
    Mode,
    ReportDraft,
    ReportResult,
    Status,
)
from baulens.report.template import TEMPLATE_PROVENANCE, template_facts, template_report
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


FACTS_BY_CODE = "Befunde aus den Eingabedaten, von Code erzeugt"
EXPERIMENTAL = "Experimentell (Freitext, Zahlenprüfung nach bestem Bemühen)"


def prompt_for(config: ReportConfig, mode: Mode) -> PromptTemplate:
    name = config.narrative_prompt_template if mode == "narrative" else config.prompt_template
    return load_prompt(name)


def _origin(client: LLMClient, prompt: PromptTemplate) -> str:
    kind = {
        "live": "Modellausgabe",
        "replay": "Aufgezeichnete Modellausgabe (Replay)",
        "fake": "Testausgabe",
        "template": "Vorlage",
    }[client.source]
    return f"{kind} von {client.model}, Prompt {prompt.prompt_version}"


def _provenance(client: LLMClient, mode: Mode, prompt: PromptTemplate) -> str:
    if mode == "narrative":
        return (
            f"{FACTS_BY_CODE}; Bewertung, Einschränkungen und Empfehlung: "
            f"{_origin(client, prompt)}, geprüft auf Zahlen und Befunde"
        )
    return f"{EXPERIMENTAL}: {_origin(client, prompt)}"


def _fallback(sheet: FactSheet, mode: Mode, reason: str) -> InspectionReport:
    report = template_report(sheet)
    if mode == "narrative":
        provenance = (
            f"{FACTS_BY_CODE}; Bewertung, Einschränkungen und Empfehlung: Vorlage "
            f"(Ersatz: {reason})"
        )
    else:
        provenance = f"{TEMPLATE_PROVENANCE} (Ersatz: {reason})"
    return report.model_copy(update={"provenance": provenance})


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
    text: str, sheet: FactSheet, mode: Mode
) -> tuple[ReportDraft, FaithfulnessResult, dict[str, str] | None]:
    """Parse a reply, build the draft to show and check it.

    Free text: the whole draft is the model's and is checked number by number. Narrative: the
    findings come from code, the three narrative fields from the model and are checked for
    numbers and findings. Every parsing or validation failure is a ``DraftError``.
    """
    if mode == "free":
        draft = parse_draft(text, sheet)
        return draft, check_sections(draft.sections(), sheet), None
    narrative = parse_narrative(text)
    combined = {**template_facts(sheet), **narrative}
    try:
        draft = ReportDraft.model_validate(combined)
    except ValidationError as error:
        raise DraftError(f"Bericht ungültig: {error.errors()[0]['msg']}") from error
    return draft, check_narrative(narrative), {str(k): v for k, v in narrative.items()}


def facts_check(
    draft: ReportDraft, final: FaithfulnessResult, sheet: FactSheet, mode: Mode
) -> FaithfulnessResult:
    """Which facts the report states; in narrative mode they are in the code findings."""
    if mode == "free":
        return final
    facts = {name: text for name, text in draft.sections().items() if name in FACT_SECTIONS}
    return check_sections(facts, sheet)


def generate_report(
    case: InspectionInput,
    client: LLMClient,
    config: ReportConfig,
    *,
    mode: Mode | None = None,
) -> ReportResult:
    """Draft a report with ``client``; deliver model text only if it passes its check."""
    mode = mode or config.generation_mode
    sheet = build_facts(case)
    prompt = prompt_for(config, mode)
    digest = input_sha256(case)
    if isinstance(client, RecordingClient | ReplayClient):
        client.begin_case(case.case_id, digest, prompt.prompt_version)
    schema = draft_schema(sheet, mode)
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
            draft, final, raw = _check(response.text, sheet, mode)
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
            Attempt(kind=kind, response=response, violations=len(final.violations), raw_draft=raw)
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
        "rejected": "Text nicht durch die Prüfung gedeckt",
        "invalid": "keine schemagültige Antwort",
        "error": "Modellaufruf fehlgeschlagen",
        "replay_miss": "keine aufgezeichnete Modellausgabe für diese Eingabe",
    }
    delivered = candidate if status == "accepted" and candidate is not None else None
    if delivered is None:
        reason = reasons.get(status, status)
        note = f"{note + '; ' if note else ''}{reason}; Vorlagentext geliefert"
    completeness = None
    if candidate is not None and final is not None:
        facts = facts_check(candidate.draft, final, sheet, mode)
        completeness = check_completeness(candidate, facts, sheet)
    return ReportResult(
        case_id=case.case_id,
        input_sha256=digest,
        source=client.source,
        model=client.model,
        mode=mode,
        prompt_version=prompt.prompt_version,
        status=status,
        delivered=delivered or _fallback(sheet, mode, reasons.get(status, status)),
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
