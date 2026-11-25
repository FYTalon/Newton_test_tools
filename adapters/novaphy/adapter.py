# SPDX-License-Identifier: Apache-2.0
"""Run NovaPhy's eleven benchmark scenarios (twelve implementation groups)."""

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import novaphy
import numpy as np
from novaphy.viewer import ViewerNull

from newton_test_tools import Trial

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
    original = getattr(example, spec["class"])

    class ConfiguredScene(original):
        def __init__(self, options):
            super().__init__(options)
            self.collision_deterministic = settings["deterministic"]

    viewer = ViewerNull(num_frames=frames)
    try:
        yield ConfiguredScene.create_example(viewer, args), example, argv
    finally:
        viewer.close()


class _RobotScene:
    def __init__(self, runtime: Any, config: Any, example: Any, step, *, graph: bool) -> None:
        self.runtime = runtime
        self.config = config
        self.example = example
        self.model = runtime.model
        self.contacts = runtime.contacts
        self.solver = runtime.solver
        self.collision_pipeline = runtime.collision_pipeline
        self.frame_dt = config.frame_dt
        self.frame_graphs = {}
        self._eager_step = partial(step, runtime, config)
        self._policy = getattr(runtime, "policy", None)
        if self._policy is not None:
            if runtime.policy_io is None:
                raise RuntimeError("ANYmal requires the demo's CUDA policy path")
            if not graph:
                runtime.policy_io.disable_graph()
        if graph:
            self._capture()

    @property
    def state_0(self):
        return self.runtime.state_0

    def _capture(self) -> None:
        runtime = self.runtime
        if self.config.sim_substeps % 2:
            raise ValueError("The robot scenes require an even substep count for one frame graph")
        first, second = runtime.state_0, runtime.state_1
        self.collision_pipeline.collide(first, self.contacts)
        self.solver.prepare_contacts(self.contacts)
        novaphy.synchronize_current_cuda_stream()
        sim_time = runtime.sim_time
        if self._policy is not None:
            runtime.policy = None
        try:
            with novaphy.CudaGraphCapture(self.model.device.ordinal) as capture:
                self._eager_step()
        finally:
            runtime.state_0, runtime.state_1 = first, second
            runtime.sim_time = sim_time
            if self._policy is not None:
                runtime.policy = self._policy
        self.frame_graphs[id(first)] = capture

    def step(self) -> None:
        if not self.frame_graphs:
            self._eager_step()
            return
        runtime = self.runtime
        if self._policy is not None:
            runtime.actions, targets = runtime.policy_io.evaluate(
                self._policy, runtime.state_0, runtime.actions, runtime.command, self.config.action_scale,
            )
            self.example._assign_array(runtime.control.joint_target_pos, targets)
        self.frame_graphs[id(runtime.state_0)].launch()
        runtime.sim_time += self.frame_dt


def _create_robot(
    case: dict[str, Any], settings: dict[str, Any], frames: int, spec: dict[str, Any],
) -> tuple[_RobotScene, Any, dict]:
    if not novaphy.has_mujoco_cuda():
        raise RuntimeError("This NovaPhy build has no native CUDA MuJoCo backend")
    example = import_module(spec["module"])
    deterministic = settings["deterministic"]
    options = {
        "backend": "cuda", "cuda_device": 0, "frames": frames,
        "world_count": case["world_count"], "deterministic": deterministic,
    }
    config = getattr(example, spec["config"])(**options)
    make_config = example.make_mujoco_solver_config
    make_pipeline = novaphy.CollisionPipeline

    def solver_config(**kwargs):
        return make_config(**{**kwargs, "disable_sensors": True, "use_mujoco_contacts": False})

    def collision_pipeline(*args, **kwargs):
        return make_pipeline(*args, **{**kwargs, "deterministic": deterministic})

    with (
        patch.object(example, "make_mujoco_solver_config", solver_config),
        patch.object(novaphy, "CollisionPipeline", collision_pipeline),
    ):
        runtime = getattr(example, spec["factory"])(config)

    if not runtime.solver.settings.disable_sensors or runtime.solver.settings.use_mujoco_contacts:
        raise RuntimeError("The robot trial requires external collisions and disabled sensors")
    scene = _RobotScene(runtime, config, example, getattr(example, spec["step"]), graph=settings["graph"])
    return scene, example, options


@contextmanager
def prepare_trial(request: dict[str, Any], *, output: Path) -> Iterator[Trial]:
    case, settings = request["case"], request["protocol"]
    case_id, solver_name = case["id"], case["solver"]
    manifest = _SCENES.read_bytes()
    spec = json.loads(manifest).get(case_id)
    if spec is None or spec["solver"] != solver_name:
        raise ValueError(f"unsupported NovaPhy case and solver: {case_id}, {solver_name}")
    if spec.get("metric") in ("particles", "pressure_cube", "volume", "tgv"):
        from .fluids import prepare_trial as prepare_fluid

        with prepare_fluid(request, spec=spec, output=output) as trial:
            yield trial
        return
    if case["collision"] != "external":
        raise ValueError("Rigid NovaPhy scenes require external collision detection")
    if case.get("metric") == "conservation":
        from .large import prepare_trial as prepare_large

        with prepare_large(request, spec=spec, output=output) as trial:
            yield trial
        return
    is_robot = solver_name == "mujoco"
    accuracy = request["mode"] == "accuracy"
    if accuracy and request.get("reference_identity") is None:
        raise ValueError(f"NovaPhy {case_id} accuracy requires Newton reference_identity")
    frames = settings["num_frames"] + (0 if accuracy else settings["warmup_frames"])
    context = (
        nullcontext(_create_robot(case, settings, frames, spec))
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
        reader = None
        if accuracy:
            runtime["contact_stage"] = "end_of_frame"
            reader = RigidReader(
                scene, request["reference_identity"], shape_flags, mujoco=is_robot,
                shape_aliases={},
            )
        yield Trial(
            engine="NovaPhy",
            runtime=runtime,
            step=scene.step,
            synchronize=novaphy.synchronize_current_cuda_stream,
            read_initial=reader.read_initial if reader is not None else None,
            observe=reader.observe if reader is not None else None,
            check_final=_check_first_native_replay(scene, runtime) if graph else None,
        )
