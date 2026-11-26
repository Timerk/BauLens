"""CPU inference benchmark: PyTorch eager vs ONNX Runtime, each in a fresh subprocess.

Separate processes keep peak RSS and thread pools of the two runtimes apart. Run as a module
(``python -m baulens.cracks.benchmark``) only through :func:`benchmark`.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil  # type: ignore[import-untyped]
from numpy.typing import NDArray

from baulens.config import CracksConfig
from baulens.cracks.metadata import load_published, run_paths, verify_training_data
from baulens.cracks.splits import ensure_splits
from baulens.paths import PROJECT_ROOT

BACKENDS = ("torch", "onnx")
BATCH_SIZES = (1, 32)


@dataclass(frozen=True)
class BenchmarkResult:
    cpu: str
    logical_cpus: int
    physical_cpus: int
    threads: int
    system_cpu_percent_before: float
    shared_load_note: str
    model_pt_bytes: int
    model_onnx_bytes: int
    runs: list[dict[str, Any]]


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def peak_rss_bytes() -> int:
    info = psutil.Process().memory_info()
    if sys.platform == "win32":
        return int(info.peak_wset)
    import resource

    # ru_maxrss is in KiB on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _worker(model_dir: Path, backend: str, threads: int, warmup: int, iterations: int) -> None:
    from baulens.cracks.inference import Backend, CrackClassifier

    started = time.perf_counter()
    classifier = CrackClassifier.load(model_dir, cast(Backend, backend), threads=threads)
    load_seconds = time.perf_counter() - started
    size = classifier.metadata.image_size
    rng = np.random.default_rng(0)
    runs = []
    for batch_size in BATCH_SIZES:
        batch = rng.standard_normal((batch_size, 3, size, size)).astype(np.float32)

        def run(batch: NDArray[np.float32] = batch) -> None:
            classifier.predict_batch(batch)

        for _ in range(warmup):
            run()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            run()
            timings.append(time.perf_counter() - start)
        median = float(np.median(timings))
        runs.append(
            {
                "backend": backend,
                "batch_size": batch_size,
                "median_ms": round(median * 1000, 3),
                "p95_ms": round(float(np.percentile(timings, 95)) * 1000, 3),
                "throughput_images_per_s": round(batch_size / median, 1),
                "iterations": iterations,
                "load_seconds": round(load_seconds, 3),
            }
        )
    for row in runs:
        row["peak_rss_mb"] = round(peak_rss_bytes() / 2**20, 1)
    print(json.dumps(runs))


def benchmark(config: CracksConfig) -> BenchmarkResult:
    """Benchmark both backends at batch 1 and 32 with warm-up; write ``benchmark.json``."""
    paths = run_paths(config)
    published = load_published(paths.model_dir, require_onnx=True)
    verify_training_data(published.metadata, ensure_splits(config), config)
    threads = config.num_threads or psutil.cpu_count(logical=False) or 1
    load = float(psutil.cpu_percent(interval=1.0))
    runs: list[dict[str, Any]] = []
    for backend in BACKENDS:
        command = [
            sys.executable,
            "-m",
            "baulens.cracks.benchmark",
            "--model-dir",
            str(paths.model_dir),
            "--backend",
            backend,
            "--threads",
            str(threads),
            "--warmup",
            str(config.benchmark_warmup),
            "--iterations",
            str(config.benchmark_iterations),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
        )
        runs.extend(json.loads(completed.stdout.strip().splitlines()[-1]))
    result = BenchmarkResult(
        cpu=cpu_name(),
        logical_cpus=psutil.cpu_count(logical=True) or 0,
        physical_cpus=psutil.cpu_count(logical=False) or 0,
        threads=threads,
        system_cpu_percent_before=load,
        shared_load_note=(
            "Measured on a shared workstation; other CPU jobs may have been running "
            f"(system CPU {load:.0f} % in the second before the run)."
        ),
        model_pt_bytes=len(published.weights),
        model_onnx_bytes=len(published.onnx or b""),
        runs=runs,
    )
    (paths.root / "benchmark.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    _worker(args.model_dir, args.backend, args.threads, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
