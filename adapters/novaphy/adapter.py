# SPDX-License-Identifier: Apache-2.0
"""Run the NovaPhy demos selected by the regular-six benchmark."""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import UUID

import novaphy
import numpy as np
from newton_test_tools import Trial
from novaphy.demos import data_path
from novaphy.viewer import ViewerNull

from .robots import create_robot
from .snapshot import RigidReader, _array


_SCENES = Path(__file__).with_name("scenes.json")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_device_uuid(ordinal: int) -> str:
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    raw_uuid = (ctypes.c_ubyte * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), ordinal):
        raise RuntimeError(f"Could not query CUDA device {ordinal}")
    if driver.cuDeviceGetUuid(ctypes.byref(raw_uuid), device):
        raise RuntimeError(f"Could not read UUID for CUDA device {ordinal}")
    return f"GPU-{UUID(bytes=bytes(raw_uuid))}"


def _check_first_native_replay(scene: Any, runtime: dict[str, Any]) -> Callable[[], None]:
    graphs = scene.frame_graphs
    runtime["first_step_physics_graph_replays"] = 0

    class FirstReplay:
        def __init__(self, graph: Any) -> None:
            self.graph = graph

        def launch(self) -> None:
            self.graph.launch()
            runtime["first_step_physics_graph_replays"] += 1
            scene.frame_graphs = graphs

    scene.frame_graphs = {state: FirstReplay(graph) for state, graph in graphs.items()}

    def check_final() -> None:
        if scene.frame_graphs is not graphs or runtime["first_step_physics_graph_replays"] != 1:
            raise ValueError("NovaPhy did not replay its native physics graph on the first step")

    return check_final


@contextmanager
def _xpbd_scene(case: dict[str, Any], settings: dict[str, Any], frames: int, spec: dict[str, Any]):
    example = import_module(spec["module"])
    argv = [
        "--headless", "--device", "cuda:0", "--frames", str(frames),
        "--deterministic" if settings["deterministic"] else "--no-deterministic",
        "--graph" if settings["graph"] else "--no-graph",
    ]
    if case["id"] == "nut_bolt_hydro":
        argv += ["--world-count", str(case["world_count"]), "--num-per-world", "1"]
    args = getattr(example, spec["parser"])().parse_args(argv)
    viewer = ViewerNull(num_frames=frames)
    try:
        yield getattr(example, spec["class"]).create_example(viewer, args), example, argv
    finally:
        viewer.close()


@contextmanager
def prepare_trial(request: dict[str, Any], *, output: Path) -> Iterator[Trial]:
    case, settings = request["case"], request["protocol"]
    case_id, solver_name = case["id"], case["solver"]
    manifest = _SCENES.read_bytes()
    spec = json.loads(manifest).get(case_id)
    if spec is None or spec["solver"] != solver_name or case["collision"] != "external":
        raise ValueError(f"unsupported NovaPhy case and solver: {case_id}, {solver_name}")
    is_robot = solver_name == "mujoco"
    accuracy = request["mode"] == "accuracy"
    if accuracy and request.get("reference_identity") is None:
        raise ValueError(f"NovaPhy {case_id} accuracy requires Newton reference_identity")
    frames = settings["num_frames"] if accuracy else round(20 / case["frame_dt_seconds"])
    context = (
        nullcontext(create_robot(case, settings, frames, spec))
        if is_robot else _xpbd_scene(case, settings, frames, spec)
    )
    with context as (scene, example, arguments):
        model = scene.model
        expected_solver = novaphy.solvers.SolverMuJoCo if is_robot else novaphy.solvers.SolverXPBD
        if model.device != novaphy.Device.cuda(0) or type(scene.solver) is not expected_solver:
            raise ValueError(f"Expected NovaPhy {solver_name} on cuda:0")
        deterministic = bool(scene.solver.settings.deterministic)
        collision_deterministic = bool(scene.collision_pipeline.deterministic)
        graph = bool(scene.frame_graphs)
        if (deterministic, collision_deterministic, graph) != (
            settings["deterministic"], settings["deterministic"], settings["graph"],
        ):
            raise ValueError("NovaPhy solver, collision or native graph settings differ from the request")
        shape_flags = _array(model, "shape_flags", (int(model.shape_count),))
        runtime = {
            "engine_version": novaphy.version(),
            "python_version": sys.version.split()[0],
            "example": example.__name__,
            "example_sha256": _sha256(example.__file__),
            "adapter_sha256": _sha256(__file__),
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "extension_sha256": _sha256(novaphy._core.__file__),
            "args": arguments,
            "device_uuid": _cuda_device_uuid(model.device.ordinal),
            "frame_dt_seconds": float(scene.frame_dt),
            "substeps": int(scene.config.sim_substeps if is_robot else scene.config.substeps),
            "world_count": int(model.world_count),
            "solver": solver_name,
            "collision": "external",
            "deterministic": deterministic,
            "collision_deterministic": collision_deterministic,
            "graph": graph,
            "model_counts": {
                **{field: int(getattr(model, field)) for field in (
                    "world_count", "body_count", "joint_count", "joint_dof_count", "particle_count"
                )},
                "collidable_shape_count": int(np.count_nonzero(shape_flags & int(novaphy.ShapeFlags.COLLIDE_SHAPES))),
            },
        }
        if case_id == "basic_shapes":
            runtime["bunny_sha256"] = _sha256(next(Path(data_path()).rglob("bunny.usd")))
        shape_aliases = {}
        if case_id == "robot_g1":
            runtime["asset_sha256"] = _sha256(example.G1_USD)
            aliases_path = Path(__file__).with_name("g1_shapes.json")
            shape_aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
            runtime["shape_aliases_sha256"] = _sha256(aliases_path)
        if case_id == "robot_anymal_c_walk":
            runtime["policy_sha256"] = _sha256(scene.runtime.asset_paths.policy_path)
        reader = RigidReader(
            scene, request["reference_identity"], shape_flags, mujoco=is_robot, shape_aliases=shape_aliases,
        ) if accuracy else None
        if accuracy:
            runtime["contact_stage"] = "end_of_frame"
        yield Trial(
            engine="NovaPhy",
            runtime=runtime,
            step=scene.step,
            synchronize=novaphy.synchronize_current_cuda_stream,
            read_initial=reader.read_initial if reader is not None else None,
            observe=reader.observe if reader is not None else None,
            check_final=_check_first_native_replay(scene, runtime) if graph else None,
        )
