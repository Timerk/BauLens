"""Adapter between the API and the report generator (``baulens.report``, Milestone 4).

The routes only know :class:`ReportService` and the API-owned envelopes in
:mod:`baulens.api.models`. Until the report module is merged, :func:`build_report_service`
returns :class:`UnavailableReportService`, so ``POST /report`` answers 503 and the dashboard
shows that state. Wiring the generator in means adding one implementation of the protocol
that maps ``ReportResult`` onto :class:`~baulens.api.models.ReportResponse`; neither the
routes nor the dashboard change.
"""

from typing import Protocol

from baulens.api.models import ComponentStatus, RecordedCase, ReportRequest, ReportResponse
from baulens.config import ApiConfig, Settings


class ReportUnavailableError(RuntimeError):
    """The report generator (or the requested source) cannot serve this request."""


class ReportService(Protocol):
    def status(self) -> ComponentStatus: ...

    def recorded_cases(self) -> list[RecordedCase]:
        """Cases with recorded model output, the only inputs ``source="replay"`` can serve."""
        ...

    def generate(self, request: ReportRequest) -> ReportResponse:
        """Blocking; live generation takes tens of seconds on a CPU."""
        ...


class UnavailableReportService:
    def __init__(self, detail: str) -> None:
        self._detail = detail

    def status(self) -> ComponentStatus:
        return ComponentStatus(ready=False, detail=self._detail)

    def recorded_cases(self) -> list[RecordedCase]:
        raise ReportUnavailableError(self._detail)

    def generate(self, request: ReportRequest) -> ReportResponse:
        raise ReportUnavailableError(self._detail)


REPORT_MODULE_MISSING = "report module not available (Milestone 4 is not merged yet)"


def build_report_service(config: ApiConfig, settings: Settings) -> ReportService:
    """Return the report adapter for this process."""
    del config, settings  # used once the generator is wired in
    return UnavailableReportService(REPORT_MODULE_MISSING)
