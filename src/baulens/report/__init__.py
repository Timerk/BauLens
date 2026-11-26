"""German inspection report drafting and numeric verification.

Typical use (API, dashboard)::

    from baulens.report import ReplayClient, generate_report

    client = ReplayClient.from_directory(project_path(config.replay_dir), "llama3.2:3b")
    result = generate_report(case, client, config)
    markdown = result.delivered.to_markdown()   # never an unfaithful model draft
"""

from baulens.report.clients import FakeClient, LLMClient, LLMError, OllamaClient
from baulens.report.facts import build_facts
from baulens.report.faithfulness import check_text
from baulens.report.generate import generate_report, generate_template, input_sha256
from baulens.report.replay import RecordingClient, ReplayClient
from baulens.report.schemas import DISCLAIMER, InspectionReport, ReportResult

__all__ = [
    "DISCLAIMER",
    "FakeClient",
    "InspectionReport",
    "LLMClient",
    "LLMError",
    "OllamaClient",
    "RecordingClient",
    "ReplayClient",
    "ReportResult",
    "build_facts",
    "check_text",
    "generate_report",
    "generate_template",
    "input_sha256",
]
