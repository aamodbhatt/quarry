"""Small numeric helpers shared by the evaluation reports."""

from __future__ import annotations

import math
import os
import platform
import statistics
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); NaN for an empty sequence."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def latency_summary(values_ms: Sequence[float]) -> dict[str, float | int]:
    if not values_ms:
        return {"n": 0}
    return {
        "n": len(values_ms),
        "mean_ms": round(statistics.fmean(values_ms), 3),
        "p50_ms": round(percentile(values_ms, 50), 3),
        "p95_ms": round(percentile(values_ms, 95), 3),
        "max_ms": round(max(values_ms), 3),
    }


def pct(part: int, whole: int) -> float | None:
    """A percentage with two decimals, or ``None`` (JSON ``null``) when there is no whole."""
    return round(100.0 * part / whole, 2) if whole else None


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value}%"


def hardware() -> dict[str, Any]:
    """CPU model, core count and OS, for the 'measured on' line of every report."""
    cpu = platform.processor() or "unknown CPU"
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            cpu = out.stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
