"""Record one engine's run with a SiMonitor-style Python interface."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import (
    RESULT_FORMAT,
    TRACE_FORMAT,
    check_runtime,
    positive_int,
    positive_number,
    validate_trace,
    write_json,
)

BodyRows = list[dict[str, Any]]
ContactPairs = list[list[Any]]
Observation = Callable[[], tuple[BodyRows, ContactPairs]]


@dataclass(frozen=True)
class MonitorConfig:
    engine: str
    gpu_devices: tuple[str, ...] = ("0",)


class Monitor:
    """Publish one trajectory or one timed run from a dedicated adapter process."""

    def __init__(
        self,
        output: str | Path,
        config: MonitorConfig,
        *,
        request: dict[str, Any],
        runtime: dict[str, Any],
        synchronize: Callable[[], None] | None = None,
        observe: Observation | None = None,
    ) -> None:
        if not config.engine.strip() or tuple(request["gpu_devices"]) != config.gpu_devices:
            raise ValueError("MonitorConfig must name the engine and use the requested GPU")
        check_runtime(request, runtime)
        self.output: Path = Path(output)
        self.config: MonitorConfig = config
        self.request: dict[str, Any] = request
        self.runtime: dict[str, Any] = runtime
        self.report: dict[str, Any] | None = None
        self._sync: Callable[[], None] | None = synchronize
        self._observe: Observation | None = observe
        self._trace: dict[str, Any] | None = None
        self._metrics: dict[str, Any] | None = None
        self._sim_time: float = 0.0

    def __enter__(self) -> Monitor:
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "report.json").exists() or (self.output / "trajectory.json").exists():
            raise FileExistsError(f"output already contains a result: {self.output}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None:
            return
        case, protocol = self.request["case"], self.request["protocol"]
        if self.request["mode"] == "accuracy":
            if self._trace is None:
                raise RuntimeError("call start before recording a trajectory")
            validate_trace(self._trace, case, protocol["num_frames"])
            write_json(self.output / "trajectory.json", self._trace, compact=True)
            self._metrics = {
                "completed_frames": protocol["num_frames"],
                "simulation_seconds": protocol["num_frames"] * case["frame_dt_seconds"],
            }
        if self._metrics is None:
            raise RuntimeError("the adapter did not report a performance measurement")
        self.report = {
            "format": RESULT_FORMAT,
            "engine": self.config.engine,
            "case": case["id"],
            "mode": self.request["mode"],
            "runtime": self.runtime,
            "metrics": self._metrics,
        }
        write_json(self.output / "report.json", self.report)

    def start(
        self,
        sim_time_s: float = 0.0,
        *,
        model: BodyRows,
        shapes: BodyRows,
        initial_state: BodyRows,
        velocity_convention: str = "com_world",
    ) -> None:
        """Record the model, shape table and frame 0 of an accuracy run."""

        if self.request["mode"] != "accuracy" or self._trace is not None:
            raise RuntimeError("start is used once per accuracy run")
        if not math.isfinite(sim_time_s):
            raise ValueError("the starting simulation clock must be finite")
        case = self.request["case"]
        self._sim_time = sim_time_s
        self._trace = {
            "format": TRACE_FORMAT,
            "case": case["id"],
            "solver": case["solver"],
            "frame_dt_seconds": case["frame_dt_seconds"],
            "num_frames": self.request["protocol"]["num_frames"],
            "velocity_convention": velocity_convention,
            "model": deepcopy(model),
            "shapes": deepcopy(shapes),
            "states": [{"frame": 0, "bodies": deepcopy(initial_state)}],
            "pairs": {},
        }

    def advance(
        self, dt_s: float, *, state: BodyRows | None = None, contact_pairs: ContactPairs | None = None, steps: int = 1
    ) -> None:
        """Record exactly one completed frame and its effective contact pairs."""

        if self._trace is None:
            raise RuntimeError("start the accuracy run before advancing it")
        if type(steps) is not int or steps != 1:
            raise ValueError("advance records exactly one completed frame")
        expected = self.request["case"]["frame_dt_seconds"]
        if not math.isclose(positive_number(dt_s, "dt_s"), expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("dt_s differs from the case timestep")
        if state is None and contact_pairs is None and self._observe is not None:
            if self._sync is not None:
                self._sync()
            state, contact_pairs = self._observe()
        if state is None or contact_pairs is None:
            raise ValueError("advance requires a state and contact pairs, using [] for an empty frame")
        frame = len(self._trace["states"])
        if frame > self._trace["num_frames"]:
            raise ValueError("accuracy run exceeded its configured frame count")
        self._trace["states"].append({"frame": frame, "bodies": deepcopy(state)})
        self._trace["pairs"][str(frame)] = deepcopy(contact_pairs)
        self._sim_time += dt_s

    @contextmanager
    def step(self, dt_s: float) -> Iterator[None]:
        """Run a user step and read its completed state through observe()."""

        yield
        self.advance(dt_s)

    def record_sim_time(self, sim_time_s: float) -> None:
        """Record one frame using an absolute simulation clock."""

        self.advance(sim_time_s - self._sim_time)
        self._sim_time = sim_time_s

    def measure(
        self, step: Callable[[], None], *, first_warmup_step: Callable[[], None] | None = None
    ) -> dict[str, Any]:
        """Run the requested warmup and time complete synchronized frame batches."""

        if self.request["mode"] != "performance" or self._metrics is not None:
            raise RuntimeError("measure is used once per performance run")
        protocol = self.request["protocol"]
        synchronize = self._sync
        for frame in range(protocol["warmup_frames"]):
            (first_warmup_step if frame == 0 and first_warmup_step is not None else step)()
        if synchronize is not None:
            synchronize()
        timed_frames, elapsed = 0, 0.0
        batch_frames = protocol["sync_interval_frames"]
        start = time.perf_counter()
        while elapsed < protocol["duration_seconds"]:
            for _ in range(batch_frames):
                step()
            if synchronize is not None:
                synchronize()
            timed_frames += batch_frames
            elapsed = time.perf_counter() - start
        return self.record_performance(timed_frames, elapsed)

    def record_performance(self, timed_frames: int, elapsed_seconds: float) -> dict[str, Any]:
        """Publish raw counters from an existing timing loop using this protocol."""

        if self.request["mode"] != "performance" or self._metrics is not None:
            raise RuntimeError("record_performance is used once per performance run")
        self._metrics = performance_metrics(self.request, timed_frames, elapsed_seconds)
        return self._metrics


def run_trial(
    output: str | Path,
    config: MonitorConfig,
    *,
    request: dict[str, Any],
    runtime: dict[str, Any],
    step: Callable[[], None],
    synchronize: Callable[[], None] | None = None,
    read_initial: Callable[[], dict[str, Any]] | None = None,
    observe: Observation | None = None,
    first_warmup_step: Callable[[], None] | None = None,
    check_final: Callable[[], None] | None = None,
) -> None:
    """Drive a configured engine for one accuracy or performance trial.

    The engine provides a full-frame step, synchronization, and optional
    accuracy readers and validation hooks. Monitor owns sampling and timing.
    """
    accuracy = request["mode"] == "accuracy"
    with Monitor(
        output,
        config,
        request=request,
        runtime=runtime,
        synchronize=synchronize,
        observe=observe if accuracy else None,
    ) as monitor:
        if accuracy:
            if read_initial is None or observe is None:
                raise ValueError("accuracy requires initial-state and frame readers")
            if synchronize is not None:
                synchronize()
            monitor.start(**read_initial())
            frame_dt = runtime["frame_dt_seconds"]
            for _ in range(request["protocol"]["num_frames"]):
                with monitor.step(frame_dt):
                    step()
        else:
            monitor.measure(step, first_warmup_step=first_warmup_step)
        if check_final is not None:
            check_final()


def performance_metrics(request: dict[str, Any], timed_frames: int, elapsed_seconds: float) -> dict[str, Any]:
    timed_frames = positive_int(timed_frames, "timed_frames")
    elapsed_seconds = positive_number(elapsed_seconds, "elapsed_seconds")
    protocol = request["protocol"]
    if elapsed_seconds < protocol["duration_seconds"] or timed_frames % protocol["sync_interval_frames"]:
        raise ValueError("incomplete timed duration or incomplete synchronization batch")
    simulation_seconds = timed_frames * request["case"]["frame_dt_seconds"]
    return {
        "timed_frames": timed_frames,
        "elapsed_seconds": elapsed_seconds,
        "simulation_seconds": simulation_seconds,
        "fps": timed_frames / elapsed_seconds,
        "rtf": elapsed_seconds / simulation_seconds,
    }
