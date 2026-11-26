"""Background sampling of system CPU load and Ollama process memory during a run."""

import statistics
import threading
from types import TracebackType
from typing import Self

import psutil  # type: ignore[import-untyped]  # no stubs in the locked dev group
from pydantic import BaseModel, ConfigDict


class LoadSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    samples: int
    cpu_mean_percent: float | None
    cpu_max_percent: float | None
    ollama_rss_peak_bytes: int | None
    logical_cpus: int | None
    total_memory_bytes: int


_RUNNER_NAMES = ("ollama", "llama-server", "llama_server")


def _ollama_rss() -> int | None:
    """RSS of the largest Ollama process.

    The weights live in the model runner, which Ollama 0.40 starts as its bundled
    ``llama-server`` (llama.cpp) rather than as an ``ollama`` process.
    """
    peak: int | None = None
    for process in psutil.process_iter(["name", "memory_info"]):
        name = (process.info.get("name") or "").lower()
        memory = process.info.get("memory_info")
        if any(runner in name for runner in _RUNNER_NAMES) and memory is not None:
            peak = max(peak or 0, int(memory.rss))
    return peak


class LoadMonitor:
    """Context manager that samples once per ``interval_s`` on a daemon thread."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self._interval_s = interval_s
        self._cpu: list[float] = []
        self._rss: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self._interval_s):
            self._cpu.append(psutil.cpu_percent(interval=None))
            rss = _ollama_rss()
            if rss is not None:
                self._rss.append(rss)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> LoadSummary:
        return LoadSummary(
            samples=len(self._cpu),
            cpu_mean_percent=round(statistics.fmean(self._cpu), 1) if self._cpu else None,
            cpu_max_percent=max(self._cpu) if self._cpu else None,
            ollama_rss_peak_bytes=max(self._rss) if self._rss else None,
            logical_cpus=psutil.cpu_count(),
            total_memory_bytes=psutil.virtual_memory().total,
        )
