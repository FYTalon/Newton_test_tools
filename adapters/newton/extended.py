# SPDX-License-Identifier: Apache-2.0
"""Bind Newton's extended benchmark scenes to the trial worker."""

from __future__ import annotations

import hashlib
import runpy
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest.mock import patch

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_test_tools import Diagnostic, Trial
from .substeps import diagnostics as rigid_diagnostics

_MODULES = {
    "basic_shapes_100k": "basic.example_basic_shapes_100k",
    "anymal_c_walk_100k": "robot.example_anymal_c_walk_5560",
    "implicit_mpm_dam_break": "mpm.example_mpm_water_dam_break",
    "implicit_mpm_water_dam_single_box": "mpm.example_mpm_water_dam_single_box",
}
_PRESSURE_METHOD = "solved_floor_contact_plus_project_outside_static_floor_velocity_impulse_plus_tank_floor_projection_per_area_per_frame"


def _particle_diagnostic(scene: Any) -> Diagnostic:
    def sample() -> dict[str, Any]:
        return {"particle_q": scene.state_0.particle_q.numpy(), "particle_qd": scene.state_0.particle_qd.numpy()}

    return Diagnostic(lambda: {"mass": scene.model.particle_mass.numpy()}, sample, frame=sample)


def _coupled_diagnostic(scene: Any, solved_floor: dict[str, np.ndarray] | None = None) -> Diagnostic:
    recorder = scene._metric_recorder

    def model() -> dict[str, Any]:
        return {
            **{
                key: recorder.metadata[key]
                for key in (
                    "initial_state_sha256",
                    "total_mass_kg",
                    "diagnostic_h_m",
                    "diagnostic_origin_m",
                    "initial_cube_center_m",
                    "volume_method",
                )
            },
            "pressure_method": _PRESSURE_METHOD,
        }

    def sample() -> dict[str, Any]:
        row = recorder.samples[-1]
        if row["pressure_method"] != _PRESSURE_METHOD:
            raise ValueError("Newton bottom pressure method changed")
        result = {
            **{
                key: row[key]
                for key in (
                    "frame",
                    "time_end_s",
                    "cube_center_m",
                    "volume_ratio",
                    "initial_effective_volume_m3",
                    "nonfinite_count",
                )
            },
            "floor_pressure_pa": np.asarray(row["floor_pressure_pa"], dtype=float),
        }
        if solved_floor is not None:
            total = np.nan_to_num(result["floor_pressure_pa"]) * (0.01 * scene.frame_dt)
            result.update(
                floor_contact_impulse_ns=solved_floor["impulse_ns"],
                floor_boundary_correction_impulse_ns=total - solved_floor["impulse_ns"],
            )
        return result

    return Diagnostic(model, dict, frame=sample)


def _skip_surface(scene: Any) -> None:
    scene.surface_graph = None
    scene.surface_mesh = None


@contextmanager
def prepare_trial(request: dict[str, Any], *, output: Path) -> Iterator[Trial]:
    case, protocol = request["case"], request["protocol"]
    kind = case["metric"]
    accuracy = request["mode"] == "accuracy"
    module = f"newton.examples.{_MODULES[case['id']]}"
    spec = find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError(f"reference scene missing: {module}. Install the sealed Newton scenario files.")
    script = Path(spec.origin)
    frames = protocol.get("num_frames", round(20 / case["frame_dt_seconds"]))
    argv = ["--viewer", "null", "--device", "cuda:0", "--quiet", "--num-frames", str(frames)]
    if kind == "conservation":
        if case["solver"] == "xpbd":
            argv.extend(("--solver", "xpbd"))
            if accuracy:
                argv.append("--contact-force")
        else:
            argv.extend(("--world-count", str(case["world_count"])))
        if not protocol["graph"]:
            argv.append("--disable-graph")
    elif kind == "particles":
        argv.extend(("--world-count", "1", "--substeps", "2"))
    else:
        argv.extend(
            (
                "--tank-planes",
                "--voxel-size",
                "0.05",
                "--particles-per-cell",
                "1",
                "--initial-snapshot",
                str(output / "initial_snapshot.npz"),
            )
        )
        if accuracy:
            argv.extend(
                ("--floor-pressure", "--incompressibility", "--metrics-output", str(output / "scene-metrics.json"))
            )
        else:
            argv.append("--no-floor-pressure")

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", [str(script), *argv]))
        if kind == "conservation":
            if protocol["deterministic"]:
                raise ValueError("these sealed large-scene configurations use ordinary solver mode")
            retained: list[tuple[Any, Any]] = []
            with patch.object(newton.examples, "run", lambda scene, args: retained.append((scene, args))):
                runpy.run_module(module, run_name="__main__")
            ((scene, args),) = retained
            capture = rigid_diagnostics(scene, case["solver"]) if accuracy else None
        else:
            if protocol["graph"]:
                raise ValueError("the implicit MPM reference runs without scene-level graph capture")
            mode = wp.DeterministicMode.GPU_TO_GPU if protocol["deterministic"] else wp.DeterministicMode.NOT_GUARANTEED
            stack.enter_context(patch.object(wp.config, "deterministic", mode))
            rheology = import_module("newton._src.solvers.implicit_mpm.rheology_solver_kernels")
            wp.set_module_options({"deterministic_max_records": 128}, module=rheology.apply_stress_gs.module)
            example = import_module(module).Example
            if kind == "particles":
                stack.enter_context(patch.object(example, "_capture_surface_extraction", _skip_surface))
            else:
                solver_type = newton.solvers.SolverMuJoCo
                original_init = solver_type.__init__

                def sensor_free_init(instance: Any, *a: Any, **kw: Any) -> None:
                    original_init(instance, *a, **{**kw, "disable_sensors": True})

                stack.enter_context(patch.object(solver_type, "__init__", sensor_free_init))
            viewer, args = newton.examples.init(example.create_parser())
            scene = example(viewer, args)
            capture = None
            if accuracy:
                if kind == "particles":
                    capture = _particle_diagnostic(scene)
                else:
                    solved_floor = None
                    if case.get("coupled_summary"):
                        solved_floor = {}
                        pressure_module = import_module(module)
                        estimate = pressure_module.floor_pressure_estimate

                        def record_pressure(*a: Any, **kw: Any) -> dict[str, Any]:
                            contact = np.asarray(estimate(*a[:10])["floor_pressure_pa"], dtype=float)
                            solved_floor["impulse_ns"] = np.nan_to_num(contact) * (0.01 * scene.frame_dt)
                            return estimate(*a, **kw)

                        stack.enter_context(patch.object(pressure_module, "floor_pressure_estimate", record_pressure))
                    capture = _coupled_diagnostic(scene, solved_floor)
                    stack.callback(scene.finish_metrics)

        stack.callback(scene.viewer.close)
        model = scene.model
        fluid = getattr(scene, "water_model", model)
        device = wp.get_device(args.device)
        graph = getattr(scene, "graph", None)
        counts = {
            field: int(getattr(model, field, 0))
            for field in ("world_count", "body_count", "joint_count", "joint_dof_count")
        }
        counts["particle_count"] = int(fluid.particle_count)
        counts["collidable_shape_count"] = int(
            ((model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0).sum()
        )
        runtime = {
            "engine_version": newton.__version__,
            "warp": wp.__version__,
            "example": module,
            "example_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "args": argv,
            "device_uuid": str(device.uuid),
            "device_name": device.name,
            "frame_dt_seconds": float(scene.frame_dt),
            "substeps": int(scene.sim_substeps),
            "world_count": counts["world_count"],
            "solver": case["solver"],
            "collision": case["collision"],
            "deterministic": protocol["deterministic"],
            "graph": graph is not None,
            "model_counts": counts,
            "contact_stage": protocol.get("contact_stage"),
        }
        yield Trial(
            "Newton",
            runtime,
            scene.step,
            lambda: wp.synchronize_device(device),
            warp_graphs=(lambda: (scene.graph,)) if graph is not None else None,
            diagnostic=capture,
        )
