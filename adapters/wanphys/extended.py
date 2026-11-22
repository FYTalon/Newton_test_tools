# SPDX-License-Identifier: Apache-2.0
"""Bind WanPhys's extended benchmark scenes to the trial worker."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import wanphys
import warp as wp

from newton_test_tools import Diagnostic, Trial
from .snapshot import COLLIDE_SHAPES
from .substeps import diagnostics as rigid_diagnostics

_MODULES = {
    "basic_shapes_100k": "newton1_6.basic.example_basic_shapes_100k",
    "anymal_c_walk_100k": "newton1_6.robot.example_anymal_c_walk_5560",
    "implicit_mpm_dam_break": "newton1_6.mpm.example_mpm_water_dam_break_implicit",
    "implicit_mpm_water_dam_single_box": "fluids.implicit_mpm_water_dam_single_box",
    "fluid_grid_mpm_single_box": "fluids.fluid_grid_mpm_single_box",
    "fluid_grid_taylor_green": "fluids.fluid_grid_taylor_green",
}
_PRESSURE_METHOD = "solved_floor_contact_plus_project_outside_static_floor_velocity_impulse_plus_tank_floor_projection_per_area_per_frame"


def _particle_diagnostic(scene: Any) -> Diagnostic:
    def sample() -> dict[str, Any]:
        return {"particle_q": scene.state_in.particle_q.numpy(), "particle_qd": scene.state_in.particle_qd.numpy()}

    return Diagnostic(lambda: {"mass": scene.model.particle_mass.numpy()}, sample, frame=sample)


def _coupled_diagnostic(scene: Any, *, volume: bool, stack: ExitStack | None) -> Diagnostic:
    recorder = scene._metric_recorder
    solved_floor = np.zeros((32, 32), dtype=np.float64)
    if stack is not None:
        read_solved = scene.coupling._diagnostic_solved_impulses

        def record_solved() -> dict[str, Any]:
            result = read_solved()
            positions = np.asarray(result["_floor_contact_positions"], dtype=np.float64)
            impulses = result["_floor_contact_impulses"][:, 2]
            cell = np.floor(positions[:, :2] / 0.1).astype(np.int64)
            inside = np.all((cell >= 0) & (cell < 32), axis=1)
            solved_floor.fill(0)
            np.add.at(solved_floor, (cell[inside, 1], cell[inside, 0]), impulses[inside])
            return result

        stack.enter_context(patch.object(scene.coupling, "_diagnostic_solved_impulses", record_solved))

    def model() -> dict[str, Any]:
        keys = ["initial_state_sha256", "total_mass_kg", "diagnostic_h_m", "diagnostic_origin_m"]
        keys += ["volume_method"] if volume else ["initial_cube_center_m"]
        return {
            **{key: recorder.metadata[key] for key in keys},
            **({} if volume else {"pressure_method": _PRESSURE_METHOD}),
        }

    def sample() -> dict[str, Any]:
        row = recorder.samples[-1]
        sample = {"frame": row["frame"], "time_end_s": row["time_end_s"]}
        if volume:
            sample.update({key: row[key] for key in ("volume_ratio", "initial_effective_volume_m3", "nonfinite_count")})
        else:
            if row["pressure_method"] != _PRESSURE_METHOD:
                raise ValueError("WanPhys bottom pressure method changed")
            pressure = np.asarray(row["floor_pressure_pa"], dtype=float)
            sample.update(cube_center_m=row["cube_center_m"], floor_pressure_pa=pressure)
            if stack is not None:
                total = np.nan_to_num(pressure) * (0.01 * scene.frame_dt)
                sample.update(
                    floor_contact_impulse_ns=solved_floor,
                    floor_boundary_correction_impulse_ns=total - solved_floor,
                )
        return sample

    return Diagnostic(model, dict, frame=sample)


def _configure_mls() -> None:
    for name, module in tuple(sys.modules.items()):
        if module is not None and (name.startswith("wanphys._src.fluid.") or name == "wanphys.examples.fluids.metrics"):
            wp.set_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU}, module=module)


def _counts(model: Any, particle_count: int) -> dict[str, int]:
    counts = {field: int(getattr(model, field, 0)) for field in ("body_count", "joint_count", "joint_dof_count")}
    flags = getattr(model, "shape_flags", None)
    return {
        **counts,
        "world_count": int(getattr(model, "world_count", 1)),
        "particle_count": particle_count,
        "collidable_shape_count": int(((flags.numpy() & COLLIDE_SHAPES) != 0).sum()) if flags is not None else 0,
    }


@contextmanager
def prepare_trial(request: dict[str, Any], *, output: Path) -> Iterator[Trial]:
    case, protocol = request["case"], request["protocol"]
    kind = case["metric"]
    accuracy = request["mode"] == "accuracy"
    module_name = f"wanphys.examples.{_MODULES[case['id']]}"
    module = import_module(module_name)
    frames = protocol.get("num_frames", round(20 / case["frame_dt_seconds"]))
    argv = ["--viewer", "null", "--device", "cuda:0", "--quiet", "--num-frames", str(frames)]
    snapshot = None
    if kind == "conservation":
        if case["solver"] == "xpbd":
            argv.extend(("--solver", "xpbd"))
            if accuracy:
                argv.append("--contact-force")
        else:
            argv.extend(("--world-count", str(case["world_count"])))
    elif kind == "particles":
        argv.extend(("--substeps", "2", "--stress-records", "128"))
    elif kind == "pressure_cube":
        argv.extend(("--voxel-size", "0.05", "--particles-per-cell", "1"))
        if accuracy:
            argv.extend(("--floor-pressure", "--cube-metrics"))
    elif kind == "volume":
        snapshot = Path(request["reference_snapshot"])
        argv.extend(("--grid-resolution", "64", "--newton-initial-state", str(snapshot)))
        if accuracy:
            argv.extend(("--incompressibility", "--wanphys-deterministic"))

    if accuracy and kind in ("volume", "pressure_cube"):
        argv.extend(("--metrics-output", str(output / "scene-metrics.json")))
    capture = None
    if kind == "tgv":
        if protocol["deterministic"]:
            raise ValueError("Taylor–Green uses the default Warp mode")
        options = module.TgvRunOptions(
            num_frames=frames, benchmark_only=not accuracy, cuda_graph=protocol["graph"], quiet=True
        )
        scene = module.create_tgv_scene(options)
        device, dt, substeps = scene.device, scene.frame_interval, scene.steps_per_frame
        counts = _counts(None, 0)
        if accuracy:
            capture = Diagnostic(
                lambda: {"grid_size": options.grid_size, "reynolds": options.reynolds},
                lambda: {"energy": scene.read_energy()},
                frame=lambda: {"energy": scene.read_energy()},
            )
    else:
        if kind == "volume":
            if protocol["graph"]:
                raise ValueError("the MLS-MPM scene has no full-frame graph")
            if protocol["deterministic"]:
                _configure_mls()
            scenarios = import_module("wanphys.examples")
            viewer, args = scenarios.init(module.GridMpmSingleBoxScenario.create_parser(), argv=argv)
            scene = module.GridMpmSingleBoxScenario(viewer, args)
        else:
            if case["id"] != "basic_shapes_100k":
                argv.append("--deterministic" if protocol["deterministic"] else "--no-deterministic")
            elif protocol["deterministic"]:
                raise ValueError("the 100k XPBD case uses ordinary solver mode")
            argv.append("--graph" if protocol["graph"] else "--no-graph")
            scene, args = module.create_scene(argv)
        device, dt, substeps = wp.get_device(args.device), float(scene.frame_dt), int(scene.sim_substeps)
        model = scene.render_model if kind == "volume" else scene.model
        fluid = scene.model if kind == "volume" else getattr(scene, "water_model", model)
        counts = _counts(model, fluid.particle_count)
        if accuracy:
            if kind == "conservation":
                capture = rigid_diagnostics(scene, case["solver"])
            elif kind == "particles":
                capture = _particle_diagnostic(scene)

    try:
        graph = getattr(scene, "graph", None)
        runtime = {
            "engine_version": wanphys.__version__,
            "warp": wp.__version__,
            "example": module_name,
            "example_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "args": argv,
            "device_uuid": str(device.uuid),
            "device_name": device.name,
            "frame_dt_seconds": dt,
            "substeps": substeps,
            "world_count": counts["world_count"],
            "solver": case["solver"],
            "collision": case["collision"],
            "deterministic": protocol["deterministic"],
            "graph": graph is not None,
            "model_counts": counts,
            "contact_stage": protocol.get("contact_stage"),
        }
        if kind == "tgv":
            runtime["scene_settings"] = scene.settings
        with ExitStack() as stack:
            if accuracy and kind in ("pressure_cube", "volume"):
                capture = _coupled_diagnostic(
                    scene,
                    volume=kind == "volume",
                    stack=stack if kind == "pressure_cube" and case.get("coupled_summary") else None,
                )
            yield Trial(
                "WanPhys",
                runtime,
                scene.step,
                lambda: wp.synchronize_device(device),
                warp_graphs=(lambda: (scene.graph,)) if graph is not None else None,
                diagnostic=capture,
            )
    finally:
        if kind != "tgv":
            if accuracy and kind in ("volume", "pressure_cube"):
                scene.finish_metrics()
            scene.viewer.close()
