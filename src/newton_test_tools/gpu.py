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
        peak = samples = 0
        last_progress = time.monotonic()
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"adapter exceeded its timeout; see {log}")
            for row in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if row.pid == process.pid and row.usedGpuMemory is not None and 0 <= row.usedGpuMemory < 2**64 - 1:
                    peak = max(peak, int(row.usedGpuMemory))
                    samples += 1
            if now - last_progress >= 30:
                print(f"  GPU peak so far: {peak / (1024**2):.1f} MiB", flush=True)
                last_progress = now
            time.sleep(min(interval_ms / 1000, max(0.0, deadline - time.monotonic())))
        return {
            "device_uuid": uuid,
            "device_name": _text(nvml.nvmlDeviceGetName(handle)),
            "driver_version": _text(nvml.nvmlSystemGetDriverVersion()),
            "peak_bytes": peak,
            "sample_count": samples,
            "sample_interval_ms": interval_ms,
        }
    finally:
        nvml.nvmlShutdown()
