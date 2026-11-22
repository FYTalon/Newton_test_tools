"""NVML polling in the runner process, outside simulation timing."""

from __future__ import annotations

import time
from pathlib import Path
from subprocess import Popen
from typing import Any

import pynvml as nvml


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def sample_gpu_memory(
    process: Popen[Any], *, selector: str, interval_ms: float, deadline: float, log: Path
) -> dict[str, Any]:
    """Measure the peak resident memory of the adapter process on one GPU."""

    nvml.nvmlInit()
    try:
        handle = (
            nvml.nvmlDeviceGetHandleByIndex(int(selector))
            if selector.isdecimal()
            else nvml.nvmlDeviceGetHandleByUUID(selector)
        )
        uuid = _text(nvml.nvmlDeviceGetUUID(handle))
        peak = 0
        samples: list[tuple[int, int]] = []
        last_progress = time.monotonic()
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"adapter exceeded its timeout; see {log}")
            processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            sampled_at = time.monotonic_ns()
            for row in processes:
                if row.pid == process.pid and row.usedGpuMemory is not None and 0 <= row.usedGpuMemory < 2**64 - 1:
                    peak = max(peak, int(row.usedGpuMemory))
                    samples.append((sampled_at, int(row.usedGpuMemory)))
            if now - last_progress >= 30:
                print(f"  GPU peak so far: {peak / (1024**2):.1f} MiB", flush=True)
                last_progress = now
            time.sleep(min(interval_ms / 1000, max(0.0, deadline - time.monotonic())))
        return {
            "device_uuid": uuid,
            "device_name": _text(nvml.nvmlDeviceGetName(handle)),
            "driver_version": _text(nvml.nvmlSystemGetDriverVersion()),
            "peak_bytes": peak,
            "sample_count": len(samples),
            "sample_interval_ms": interval_ms,
            "scope": "process",
            "samples": samples,
        }
    finally:
        nvml.nvmlShutdown()


def select_timed_memory(memory: dict[str, Any], window: dict[str, int] | None) -> dict[str, Any]:
    """Keep samples inside the adapter's synchronized wall-clock window."""
    if window is None:
        return memory
    start, end = window["start_ns"], window["end_ns"]
    if type(start) is not int or type(end) is not int or not 0 < start < end:
        raise ValueError("invalid GPU memory timing window")
    samples = [(t, size) for t, size in memory["samples"] if start <= t <= end]
    result = {**memory, "timing_window": window, "process_peak_bytes": memory["peak_bytes"]}
    if not samples:
        return {**result, "fallback_reason": "no NVML sample inside the timed window"}
    return {
        **result,
        "scope": "timed",
        "samples": samples,
        "sample_count": len(samples),
        "peak_bytes": max(size for _, size in samples),
    }
