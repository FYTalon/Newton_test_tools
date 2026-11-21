#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
"""Candidate adapter that drives the six WanPhys examples directly."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mujoco
import wanphys
import warp as wp
from newton_test_tools import MonitorConfig, read_request, run_trial
from newton_test_tools.protocol import read_json
from wanphys.rigid import SolverXPBD

from snapshot import COLLIDE_SHAPES, RigidReader

_SCENES = Path(__file__).with_name("scenes.json")
_OWNED_MUJOCO = "wanphys._src.rigid.mujoco_owned.solver_mujoco.WanPhysMujocoSolver"


def _example_argv(spec: dict[str, Any], request: dict[str, Any]) -> list[str]:
    settings = request["protocol"]
    frames = (
        settings["num_frames"] if request["mode"] == "accuracy" else round(20 / request["case"]["frame_dt_seconds"])
    )
    return [
        "--viewer",
        "null",
        "--device",
        "cuda:0",
        "--num-frames",
        str(frames),
        "--quiet",
        *spec["args"],
        "--deterministic" if settings["deterministic"] else "--no-deterministic",
        "--graph" if settings["graph"] else "--no-graph",
    ]


def _solver_settings(scene: Any, solver_name: str, deterministic: bool) -> dict[str, Any]:
    solver = scene.solver
    solver_class = f"{type(solver).__module__}.{type(solver).__qualname__}"
    sensors = None
    if solver_name == "mujoco":
        if solver_class != _OWNED_MUJOCO:
            raise ValueError(f"expected owned MuJoCo, loaded {solver_class}")
        model = solver.native_model
        disabled = int(model.opt.disableflags)
        if (
            scene.use_mujoco_contacts
            or solver.config.use_mujoco_contacts
            or model.opt.run_collision_detection
            or disabled & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        ):
            raise ValueError("MuJoCo must use external contacts")
        if not disabled & int(mujoco.mjtDisableBit.mjDSBL_SENSOR):
            raise ValueError("MuJoCo sensors must be disabled")
        sensors = int(model.nsensor)
        mode = solver._determinism.mode
    elif solver_name == "xpbd" and type(solver) is SolverXPBD:
        mode = solver._deterministic_mode
    else:
        raise ValueError(f"unexpected solver {solver_class} for {solver_name}")

    if scene.contacts is None:
        raise ValueError("the scene did not expose external contacts")
    global_mode = wp.config.deterministic
    collision = getattr(scene, "collision", None)
    if collision is None:
        collision = scene.collision_pipeline
    collision_sorted = collision.deterministic
    expected = wp.DeterministicMode.GPU_TO_GPU if deterministic else wp.DeterministicMode.NOT_GUARANTEED
    if global_mode != wp.DeterministicMode.NOT_GUARANTEED or mode != expected or collision_sorted is not deterministic:
        raise ValueError(f"expected solver mode {expected.name} and external collision sorting={deterministic}")
    return {
        "solver_class": solver_class,
        "mujoco_sensor_count": sensors,
        "determinism": {
            "global_warp_mode": global_mode.name,
            "solver_mode": mode.name,
            "collision_sorted": collision_sorted,
        },
    }


def _physics_graphs(scene: Any) -> tuple[Any, ...]:
    return tuple(graph for name in ("graph", "graph_alt") if (graph := getattr(scene, name, None)) is not None)


def _graph_hooks(scene: Any, runtime: dict[str, Any]) -> tuple[Callable[[], None] | None, Callable[[], None]]:
    graphs = _physics_graphs(scene)
    launches = 0

    def first_warmup_step() -> None:
        original_launch = wp.capture_launch

        def counted_launch(graph: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal launches
            if any(graph is known for known in graphs):
                launches += 1
            return original_launch(graph, *args, **kwargs)

        with patch.object(wp, "capture_launch", counted_launch):
            scene.step()
        if launches == 0:
            raise ValueError("WanPhys did not replay the physics graph during warmup")

    def check_graphs() -> None:
        if _physics_graphs(scene) != graphs:
            raise ValueError("WanPhys physics graph changed during the timed run")
        if graphs:
            runtime["warmup_physics_graph_replays"] = launches

    return (first_warmup_step if graphs else None), check_graphs


def _run_trial(
    scene: Any,
    args: argparse.Namespace,
    spec: dict[str, Any],
    request: dict[str, Any],
    output: Path,
    provenance: dict[str, Any],
) -> None:
    case, settings = request["case"], request["protocol"]
    device = wp.get_device(args.device)
    if args.viewer != "null" or not device.is_cuda or wp.get_device(scene.model.device) != device:
        raise ValueError("WanPhys must run headless on the selected GPU")
    settings_observed = _solver_settings(scene, case["solver"], settings["deterministic"])
    graphs = _physics_graphs(scene)
    policy_graph = getattr(scene, "policy_graph", None)
    if policy_graph is not None and any(policy_graph is graph for graph in graphs):
        raise ValueError("policy and physics graphs must be distinct")
    if not settings["graph"] and (graphs or policy_graph is not None or getattr(scene, "use_graph", False)):
        raise ValueError("the request disabled graph but the example captured one")
    runtime = {
        **provenance,
        **settings_observed,
        "engine_version": getattr(wanphys, "__version__", None),
        "warp": wp.__version__,
        "frame_dt_seconds": float(scene.frame_dt),
        "substeps": int(scene.sim_substeps),
        "world_count": int(scene.model.world_count),
        "solver": case["solver"],
        "collision": "external",
        "deterministic": settings["deterministic"],
        "graph": bool(graphs),
        "device_uuid": str(device.uuid),
        "device_name": device.name,
    }

    def synchronize() -> None:
        wp.synchronize_device(device)

    reader = None
    first_warmup_step = check_graphs = None
    if request["mode"] == "accuracy":
        runtime["contact_stage"] = "end_of_frame"
        reader = RigidReader(scene, spec["state"], spec.get("labels", {}), mujoco=case["solver"] == "mujoco")
    else:
        model = scene.model
        counts = {
            key: int(getattr(model, key))
            for key in ("world_count", "body_count", "joint_count", "joint_dof_count", "particle_count", "shape_count")
        }
        counts["collidable_shape_count"] = int(((model.shape_flags.numpy() & COLLIDE_SHAPES) != 0).sum())
        runtime["model_counts"] = counts
        first_warmup_step, check_graphs = _graph_hooks(scene, runtime)

    def check_final() -> None:
        if check_graphs is not None:
            check_graphs()
        _solver_settings(scene, case["solver"], settings["deterministic"])

    config = MonitorConfig(engine="WanPhys", gpu_devices=tuple(request["gpu_devices"]))
    run_trial(
        output,
        config,
        request=request,
        runtime=runtime,
        step=scene.step,
        synchronize=synchronize,
        read_initial=reader.read_initial if reader is not None else None,
        observe=reader.observe if reader is not None else None,
        first_warmup_step=first_warmup_step,
        check_final=check_final,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = read_request(args.request)
    case, settings = request["case"], request["protocol"]
    if request["mode"] == "performance" and settings["graph"] and settings["warmup_frames"] < 1:
        raise ValueError("performance needs a warmup step to verify graph replay")

    spec = read_json(_SCENES).get(case["id"])
    if spec is None or spec["solver"] != case["solver"]:
        raise ValueError(f"unsupported WanPhys case and solver: {case['id']}, {case['solver']}")
    argv = _example_argv(spec, request)
    example = import_module(spec["module"])
    create_scene = getattr(example, "create_scene", None)
    if not callable(create_scene):
        raise ValueError(f"{spec['module']} must provide create_scene(argv) in the installed WanPhys version")
    script = Path(example.__file__)
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    provenance = {
        "example": spec["module"],
        "example_sha256": script_sha,
        "scene_config_sha256": hashlib.sha256(_SCENES.read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0],
        "args": argv,
    }
    scene, parsed = create_scene(argv)
    try:
        _run_trial(scene, parsed, spec, request, args.output, provenance)
    finally:
        scene.viewer.close()
    if hashlib.sha256(script.read_bytes()).hexdigest() != script_sha:
        raise ValueError("example changed during the trial")

    sys.stdout.flush()
    sys.stderr.flush()
    log = args.output / "worker.log"
    if case["id"] == "nut_bolt_hydro" and log.is_file():
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "Hydroelastic buffers overflowed" in line or ("[hydroelastic]" in line and "overflow:" in line):
                    raise RuntimeError("hydroelastic capacity overflow, see worker.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
