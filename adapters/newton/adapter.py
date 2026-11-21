# SPDX-License-Identifier: Apache-2.0
"""Reference adapter for the official Newton 1.6 examples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import runpy
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mujoco
import newton
import newton.examples
import warp as wp

from newton_test_tools import Trial

from .snapshot import RigidReader

_MODULES: dict[str, str] = {
    "basic_shapes": "basic.example_basic_shapes",
    "basic_conveyor": "basic.example_basic_conveyor",
    "nut_bolt_hydro": "contacts.example_nut_bolt_hydro",
    "basic_pendulum": "basic.example_basic_pendulum",
    "robot_g1": "robot.example_robot_g1",
    "robot_anymal_c_walk": "robot.example_robot_anymal_c_walk",
}


def _example_argv(request: dict[str, Any]) -> list[str]:
    case, protocol = request["case"], request["protocol"]
    frames = protocol["num_frames"] if request["mode"] == "accuracy" else round(20 / case["frame_dt_seconds"])
    argv = ["--viewer", "null", "--num-frames", str(frames), "--device", "cuda:0", "--quiet"]
    case_id = case["id"]
    if case_id in ("basic_shapes", "basic_conveyor", "nut_bolt_hydro"):
        argv.extend(("--solver", case["solver"]))
    if case_id in ("nut_bolt_hydro", "robot_g1"):
        argv.extend(("--world-count", str(case["world_count"])))
    if case_id == "nut_bolt_hydro":
        argv.extend(("--num-per-world", "1"))
        if protocol["deterministic"]:
            argv.append("--deterministic")
    if case_id in ("robot_g1", "robot_anymal_c_walk"):
        argv.append("--no-use-mujoco-contacts")
    return argv


def _hydro_capacity(request: dict[str, Any]) -> int | None:
    if request["case"]["id"] != "nut_bolt_hydro":
        return None
    from newton._src.geometry import contact_reduction_hydroelastic, sdf_hydroelastic

    capacity = int(sdf_hydroelastic._MAX_DETERMINISTIC_ISO_VOXELS)
    key_source = Path(contact_reduction_hydroelastic.__file__).read_text(encoding="utf-8")
    if request["protocol"]["deterministic"] and (
        capacity != 838_835 or "contact_data.sort_sub_key = 0x7FFF00 | bin_id" not in key_source
    ):
        raise ValueError(
            "20-world deterministic nut_bolt_hydro needs the reference capacity and sort keys. "
            "See adapters/newton/README.md and hydro-v1.6.0.patch."
        )
    return capacity


def _physics_graphs(scene: Any) -> tuple[Any, ...]:
    return tuple(graph for name in ("graph", "graph_alt") if (graph := getattr(scene, name, None)) is not None)


def _adapt_scene(
    scene: Any,
    parsed: argparse.Namespace,
    request: dict[str, Any],
    provenance: dict[str, Any],
    argv: list[str],
    capacity: int | None,
) -> Trial:
    protocol = request["protocol"]
    model, solver = scene.model, scene.solver
    device = wp.get_device(parsed.device)
    if parsed.viewer != "null" or not device.is_cuda or wp.get_device(model.device) != device:
        raise ValueError("Newton must run headless on the selected GPU")
    if isinstance(solver, newton.solvers.SolverXPBD):
        solver_name = "xpbd"
        solver_modes = {option["deterministic"] for option in solver._module_options.values()}
    elif isinstance(solver, newton.solvers.SolverMuJoCo):
        solver_name = "mujoco"
        solver_modes = {solver._deterministic}
        if not int(solver.mj_model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_SENSOR):
            raise ValueError("Newton MuJoCo sensors are enabled")
        if solver.mjw_model.opt.run_collision_detection:
            raise ValueError("Newton MuJoCo must use the external collision pipeline")
    else:
        raise ValueError(f"unsupported Newton solver {type(solver).__name__}")
    required_mode = (
        wp.DeterministicMode.GPU_TO_GPU if protocol["deterministic"] else wp.DeterministicMode.NOT_GUARANTEED
    )
    pipeline = scene.collision_pipeline
    if (
        wp.config.deterministic != wp.DeterministicMode.NOT_GUARANTEED
        or solver_modes != {required_mode}
        or pipeline.deterministic is not protocol["deterministic"]
        or scene.contacts is None
    ):
        raise ValueError("Newton solver or external contact configuration differs from the request")

    graph_enabled = bool(_physics_graphs(scene)) or bool(getattr(scene, "use_graph", False))
    runtime = {
        "frame_dt_seconds": float(scene.frame_dt),
        "substeps": int(scene.sim_substeps),
        "world_count": int(model.world_count),
        "solver": solver_name,
        "collision": "external",
        "deterministic": protocol["deterministic"],
        "graph": graph_enabled,
        "device_uuid": str(device.uuid),
        "device_name": device.name,
        "engine_version": newton.__version__,
        "python_version": sys.version.split()[0],
        "warp": wp.__version__,
        "mujoco": mujoco.__version__,
        "args": argv,
        **provenance,
        "deterministic_hydro_capacity": capacity,
    }

    def synchronize() -> None:
        wp.synchronize_device(device)

    reader = None
    if request["mode"] == "accuracy":
        runtime["contact_stage"] = "end_of_frame"
        reader = RigidReader(scene, mujoco=solver_name == "mujoco")
    else:
        counts = {
            field: int(getattr(model, field))
            for field in ("world_count", "body_count", "joint_count", "joint_dof_count", "particle_count")
        }
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        counts["collidable_shape_count"] = int(((model.shape_flags.numpy() & collide) != 0).sum())
        runtime["model_counts"] = counts

    return Trial(
        engine="Newton",
        runtime=runtime,
        step=scene.step,
        synchronize=synchronize,
        read_initial=reader.read_initial if reader is not None else None,
        observe=reader.observe if reader is not None else None,
        warp_graphs=lambda: _physics_graphs(scene),
    )


@contextmanager
def prepare_trial(request: dict[str, Any], *, output: Path) -> Iterator[Trial]:
    if newton.__version__ != "1.6.0" or wp.__version__ != "1.17.0":
        raise RuntimeError("this reference adapter expects Newton 1.6.0 and Warp 1.17.0")
    module = f"newton.examples.{_MODULES[request['case']['id']]}"
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Newton example module missing: {module}")
    script = Path(spec.origin).resolve()
    argv = _example_argv(request)
    protocol = request["protocol"]
    capacity = _hydro_capacity(request)
    provenance = {"example": script.name, "example_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    original_init = newton.examples.init
    original_pipeline = newton.CollisionPipeline

    with ExitStack() as stack:
        initialized: list[tuple[Any, argparse.Namespace]] = []

        def retain_scene(scene: Any, parsed: argparse.Namespace) -> None:
            stack.callback(scene.viewer.close)
            initialized.append((scene, parsed))

        # The official examples use an init/run lifecycle, with capture performed
        # inside Example.__init__. Scope the configuration to this process.
        def init_without_capture(*args: Any, **kwargs: Any) -> Any:
            scene_type = sys._getframe(1).f_globals["Example"]

            def skip_capture(scene: Any) -> None:
                scene.graph = None
                scene.use_graph = False

            stack.enter_context(patch.object(scene_type, "capture", skip_capture))
            return original_init(*args, **kwargs)

        def configure_solver(solver_type: type, *, mujoco_solver: bool) -> None:
            original = solver_type.__init__

            def constructor(instance: Any, *args: Any, **kwargs: Any) -> None:
                if protocol["deterministic"]:
                    kwargs["deterministic"] = wp.DeterministicMode.GPU_TO_GPU
                if mujoco_solver:
                    kwargs["disable_sensors"] = True
                original(instance, *args, **kwargs)

            stack.enter_context(patch.object(solver_type, "__init__", constructor))

        def sorted_pipeline(*args: Any, **kwargs: Any) -> Any:
            kwargs["deterministic"] = True
            return original_pipeline(*args, **kwargs)

        stack.enter_context(patch.object(sys, "argv", [str(script), *argv]))
        stack.enter_context(patch.object(newton.examples, "run", retain_scene))
        if not protocol["graph"]:
            stack.enter_context(patch.object(newton.examples, "init", init_without_capture))
        if protocol["deterministic"]:
            stack.enter_context(patch.object(newton, "CollisionPipeline", sorted_pipeline))
            configure_solver(newton.solvers.SolverXPBD, mujoco_solver=False)
        configure_solver(newton.solvers.SolverMuJoCo, mujoco_solver=True)
        runpy.run_module(module, run_name="__main__")

        if len(initialized) != 1:
            raise RuntimeError("the Newton example must initialize exactly one scene")
        scene, parsed = initialized[0]
        yield _adapt_scene(scene, parsed, request, provenance, argv, capacity)
