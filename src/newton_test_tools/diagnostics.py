"""Stream benchmark samples without retaining a full trajectory in memory."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager

import numpy as np

Sample = dict[str, Any]


@dataclass(frozen=True)
class Diagnostic:
    """Read model and initial state, then observe complete frames or solver substeps."""

    model: Callable[[], Sample]
    initial: Callable[[], Sample]
    frame: Callable[[], Sample] | None = None
    substeps: Callable[[Callable[[Sample], None]], ContextManager[None]] | None = None


def tgv_sample_frames(frames: int, frame_dt: float, sample_dt: float) -> tuple[int, ...]:
    if sample_dt <= frame_dt:
        return tuple(range(frames + 1))
    m = math.ceil(frames * frame_dt / sample_dt)
    targets = {min(frames, round(i * sample_dt / frame_dt)) for i in range(m)}
    targets.update((0, frames))
    if frames >= 2 and len(targets) == 2:
        targets.add(frames // 2)
    return tuple(sorted(targets))


def record_samples(
    output: Path,
    capture: Diagnostic,
    *,
    frames: int,
    substeps: int,
    step: Callable[[], None],
    synchronize: Callable[[], None] | None,
    sample_frames: tuple[int, ...] | None = None,
) -> None:
    if (capture.frame is None) == (capture.substeps is None):
        raise ValueError("a diagnostic must provide exactly one frame reader or substep observer")
    if sample_frames is not None and capture.substeps is not None:
        raise ValueError("selected frames require a frame reader")
    directory = output / "samples"
    directory.mkdir()
    if synchronize is not None:
        synchronize()
    np.savez(output / "model.npz", **capture.model())
    index = 0

    def record(sample: Sample) -> None:
        nonlocal index
        np.savez(directory / f"{index:06d}.npz", **sample)
        index += 1

    record(capture.initial())
    if capture.substeps is not None:
        with capture.substeps(record):
            for frame in range(1, frames + 1):
                step()
                if index != 1 + frame * substeps:
                    raise ValueError(f"expected {substeps} completed substeps in frame {frame}")
    else:
        assert capture.frame is not None
        selected = frozenset(sample_frames) if sample_frames is not None else None
        for frame in range(1, frames + 1):
            step()
            if selected is not None and frame not in selected:
                continue
            if synchronize is not None:
                synchronize()
            record(capture.frame())


def load_sample(directory: Path, index: int | None = None) -> Sample:
    path = directory / ("model.npz" if index is None else f"samples/{index:06d}.npz")
    with np.load(path, allow_pickle=False) as data:
        return dict(data)
