"""Load an engine adapter and run one requested trial in this process."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from .monitor import MonitorConfig, Observation, run_trial
from .protocol import read_request


@dataclass(frozen=True)
class Trial:
    """An initialized engine's hooks and observed settings for one trial."""

    engine: str
    runtime: dict[str, Any]
    step: Callable[[], None]
    synchronize: Callable[[], None]
    read_initial: Callable[[], dict[str, Any]] | None = None
    observe: Observation | None = None
    warp_graphs: Callable[[], tuple[Any, ...]] | None = None
    check_final: Callable[[], None] | None = None


def _load_adapter(reference: str) -> ModuleType:
    if not reference.endswith(".py"):
        return import_module(reference)
    path = Path(reference).resolve()
    # Treat a file adapter as a package so it can import sibling helpers.
    spec = spec_from_file_location("_benchmark_engine_adapter", path, submodule_search_locations=[str(path.parent)])
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load adapter {path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _performance_hooks(
    trial: Trial, request: dict[str, Any]
) -> tuple[Callable[[], None] | None, Callable[[], None] | None]:
    protocol = request["protocol"]
    read_graphs = trial.warp_graphs
    if request["mode"] != "performance" or not protocol["graph"] or read_graphs is None:
        return None, trial.check_final

    import warp as wp

    graphs = read_graphs()
    if not graphs or protocol["warmup_frames"] < 1:
        raise ValueError("checking Warp graph replay requires a captured graph and a warmup frame")
    replays = 0

    def first_warmup_step() -> None:
        launch = wp.capture_launch

        def counted_launch(graph: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal replays
            replays += any(graph is known for known in graphs)
            return launch(graph, *args, **kwargs)

        with patch.object(wp, "capture_launch", counted_launch):
            trial.step()
        if replays == 0:
            raise ValueError(f"{trial.engine} did not replay its physics graph during warmup")

    def check_final() -> None:
        if read_graphs() != graphs:
            raise ValueError(f"{trial.engine} physics graph changed during the timed run")
        trial.runtime["warmup_physics_graph_replays"] = replays
        if trial.check_final is not None:
            trial.check_final()

    return first_warmup_step, check_final


def run_worker(adapter: str, request_path: Path, output: Path) -> None:
    """Drive the trial supplied by an adapter's prepare_trial() context."""
    request = read_request(request_path)
    module = _load_adapter(adapter)
    prepare = getattr(module, "prepare_trial", None)
    if not callable(prepare):
        raise ValueError(f"adapter {adapter} must export prepare_trial(request, *, output)")
    with prepare(request, output=output) as trial:
        if not isinstance(trial, Trial):
            raise TypeError("prepare_trial must yield a newton_test_tools.Trial")
        first_warmup_step, check_final = _performance_hooks(trial, request)
        run_trial(
            output,
            MonitorConfig(trial.engine, tuple(request["gpu_devices"])),
            request=request,
            runtime=trial.runtime,
            step=trial.step,
            synchronize=trial.synchronize,
            read_initial=trial.read_initial,
            observe=trial.observe,
            first_warmup_step=first_warmup_step,
            check_final=check_final,
        )
