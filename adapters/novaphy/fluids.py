# SPDX-License-Identifier: Apache-2.0
"""Native fluid scene drivers and observations for the public diagnostic protocol."""

import ctypes
import hashlib
import math
import os
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from unittest.mock import patch

import novaphy
import numpy as np
from novaphy.viewer import ViewerNull

from newton_test_tools import Diagnostic, Trial

from .adapter import _cuda_device_uuid, _sha256
from .substeps import array

PRESSURE_METHOD = (
    "solved_floor_contact_plus_project_outside_static_floor_velocity_impulse_"
    "plus_tank_floor_projection_per_area_per_frame"
)
VOLUME_METHOD = "CIC_saturated_rest_volume_sparse_float64_v2"


@lru_cache(maxsize=1)
def _driver():
    return ctypes.CDLL("libcuda.so.1")


def synchronize():
    if code := _driver().cuCtxSynchronize():
        raise RuntimeError(f"Native fluid CUDA context synchronization failed: {code}")


def particle_state(scene, rotation=None):
    q = array(scene.state.particle_q_array).astype(np.float32, copy=False)
    v = array(scene.state.particle_qd_array).astype(np.float32, copy=False)
    if rotation is not None:
        q, v = q @ rotation, v @ rotation
    if q.ndim != 2 or q.shape[1] != 3 or v.shape != q.shape or not all(np.isfinite(x).all() for x in (q, v)):
        raise ValueError("Invalid native fluid particle state")
    return np.ascontiguousarray(q), np.ascontiguousarray(v)


def coupled_state(scene, rotation):
    q, v = particle_state(scene, rotation)
    body_q = array(scene.state.body_q_array).astype(np.float32, copy=False)
    body_qd = array(scene.state.body_qd_array).astype(np.float32, copy=False)
    if body_q.shape != (1, 7) or body_qd.shape != (1, 6) or not all(
        np.isfinite(value).all() for value in (body_q, body_qd)
    ):
        raise ValueError("Invalid native coupled rigid state")
    return dict(particle_q=q, particle_qd=v, body_q_native=body_q, body_qd_native=body_qd)


def initial_metadata(scene, rotation):
    q, v = particle_state(scene, rotation)
    mass = array(scene.model.particle_mass).astype(np.float32, copy=False)
    active = (array(scene.model.particle_flags) & 1) != 0
    digest = hashlib.sha256()
    for value in (q, mass, v):
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    digest.update(active.astype(np.uint8).tobytes())
    args = scene.config.args
    return dict(
        initial_state_sha256=digest.hexdigest(), total_mass_kg=float(mass.sum(dtype=np.float64)),
        diagnostic_h_m=float(args.voxel_size if args.diagnostic_h is None else args.diagnostic_h),
        diagnostic_origin_m=args.diagnostic_origin, volume_method=VOLUME_METHOD,
        particle_mass=mass, particle_active=active, native_to_reference_rotation=rotation,
        particle_state_frame="reference_z_up", rigid_state_frame="native_y_up",
        body_q_layout="px,py,pz,qx,qy,qz,qw", body_qd_layout="vx,vy,vz,wx,wy,wz",
    )


def check_reference_initial(scene, rotation, snapshot):
    if snapshot is None:
        raise ValueError("The volume trial requires the same-round Newton reference_snapshot")
    q, v = particle_state(scene, rotation)
    with np.load(snapshot, allow_pickle=False) as expected:
        for name, actual in (
            ("positions", q), ("velocities", v), ("masses", array(scene.model.particle_mass)),
            ("active", (array(scene.model.particle_flags) & 1) != 0),
        ):
            if not np.array_equal(actual, expected[name]):
                raise ValueError(f"Native V2 initial {name} differs from the sealed reference snapshot")


def pressure_sample(scene, collector, helpers, rotation, dt):
    row = collector.sample()
    if row["pressure_method"] != PRESSURE_METHOD:
        raise ValueError("Native pressure observer changed its method")
    data = collector.observer.data
    state = coupled_state(scene, rotation)
    q = state["particle_q"]
    # Use the same supplied observation function as the reference adapter. The
    # contact-only pass separates solved impulses from SDF and tank corrections.
    solved = helpers.floor_pressure_estimate(
        data["contact_impulses_on_collider"], data["contact_positions"], data["contact_ids"],
        data["collider_body_ids"], data["contact_normals"], q, collector.active,
        array(scene.model.particle_radius), float(scene.config.dx), dt,
    )
    pressure = np.asarray(row["floor_pressure_pa"], dtype=np.float64)
    contact = np.nan_to_num(np.asarray(solved["floor_pressure_pa"], dtype=np.float64)) * (0.01 * dt)
    if pressure.shape != (32, 32) or np.isinf(pressure).any():
        raise ValueError("Invalid native floor pressure grid")
    return dict(
        **state,
        frame=scene.frame, time_end_s=scene.frame * dt, floor_pressure_pa=pressure,
        cube_center_m=np.asarray(row["cube_center_m"]), floor_contact_impulse_ns=contact,
        floor_boundary_correction_impulse_ns=np.nan_to_num(pressure) * (0.01 * dt) - contact,
    )


def volume_sample(scene, volume, rotation, dt):
    state = coupled_state(scene, rotation)
    result = dict(volume.measure(state["particle_q"]))
    result.update(
        **state,
        frame=scene.frame, time_end_s=scene.frame * dt,
        initial_effective_volume_m3=float(volume.initial_volume_m3),
    )
    return result


def disable_solver_graphs(scene):
    for config in (scene.mpm.implicit_mpm.config(), scene.mpm.params.mpm):
        for name in (
            "newton_fem_cuda_graph", "newton_fem_graph_cond_while", "newton_fem_gs_color_ring_graph",
        ):
            setattr(config, name, False)


def counts(scene):
    model = getattr(scene, "rigid_model", scene.model)
    return {
        **{key: int(getattr(model, key, 0)) for key in ("body_count", "joint_count", "joint_dof_count")},
        "world_count": 1, "particle_count": int(scene.model.particle_count),
        "collidable_shape_count": int(np.count_nonzero(array(model.shape_flags) & 2)),
    }


@contextmanager
def _mpm_example(request, spec):
    protocol = request["protocol"]
    frames = protocol["num_frames"] + (
        protocol["warmup_frames"] if request["mode"] == "performance" else 0
    )
    module = import_module(spec["module"])
    coupled = spec["metric"] != "particles"
    argv = ["--headless", "--frames", str(frames)]
    if coupled:
        argv += ["--mode", "accuracy" if protocol["deterministic"] else "normal"]
    args = module.create_parser(module.__doc__, mode=coupled).parse_args(argv)
    viewer = ViewerNull(num_frames=frames)
    try:
        if spec["metric"] == "volume":
            make_scene = module.make_scene

            def without_graphs(options):
                # Example initializes its backend before this factory; that
                # setup enables prewarming again. Override it at construction.
                options.graph_solve = False
                os.environ["NOVAPHY_V2_PREWARM_GRAPHS"] = "0"
                return make_scene(options)

            with patch.object(module, "make_scene", without_graphs):
                example = module.Example(viewer, args)
        else:
            example = module.Example(viewer, args)
        disable_solver_graphs(example.scene)
        yield example, module, argv
    finally:
        viewer.close()


@contextmanager
def mpm_scene(request, spec, output):
    kind, protocol = spec["metric"], request["protocol"]
    accuracy = request["mode"] == "accuracy"
    deterministic = protocol["deterministic"]
    if kind == "particles" and deterministic:
        raise ValueError("The dam-break demo has no selectable deterministic switch; its algorithm is fixed ordered")
    os.environ.update(
        NOVAPHY_CUDA_ORDERED_ACCUMULATION="1" if deterministic else "0",
        NOVAPHY_MPM_DETERMINISTIC="1" if deterministic else "0",
        NOVAPHY_MPM_PERSISTENT_WATER_RING="0", NOVAPHY_WATER_GS_COLOR_RING="0",
    )
    if kind == "particles":
        # This demo has no mode selector. Its fixed ordered algorithm is
        # reported separately from the optional deterministic runtime switch.
        os.environ.pop("NOVAPHY_MPM_ACCUMULATION_MODE", None)
    else:
        os.environ["NOVAPHY_MPM_ACCUMULATION_MODE"] = "accuracy" if deterministic else "normal"
    with _mpm_example(request, spec) as (example, module, argv):
        scene, config = example.scene, example.scene.config
        backend = module.advanced_density_cuda if kind == "volume" else module.mpm_backend
        collector = None
        capture = None
        details = dict(deterministic=deterministic, determinism_configurable=True)
        try:
            if kind == "particles":
                capture = Diagnostic(
                    lambda: {"mass": array(scene.model.particle_mass)},
                    lambda: dict(zip(("particle_q", "particle_qd"), particle_state(scene), strict=True)),
                    frame=lambda: dict(zip(("particle_q", "particle_qd"), particle_state(scene), strict=True)),
                ) if accuracy else None
                details.update(deterministic=False, determinism_configurable=False, determinism_policy="fixed_ordered")
            elif kind == "pressure_cube" and accuracy:
                rotation = module.ROTATION
                metrics = import_module("novaphy.tools.mpm.water_cube_metrics")
                collector = metrics.NativeMetrics(scene, output / "scene-metrics.json", defer_volume=True)
                helpers = import_module("novaphy.tools.mpm.water_cube_metrics_cpu")
                metadata = initial_metadata(scene, rotation)
                metadata.update(
                    pressure_method=PRESSURE_METHOD,
                    initial_cube_center_m=array(scene.state.body_q_array)[0, :3] @ rotation,
                )
                capture = Diagnostic(
                    lambda: metadata, lambda: coupled_state(scene, rotation),
                    frame=lambda: pressure_sample(scene, collector, helpers, rotation, 1 / config.args.fps),
                )
            elif kind == "volume" and accuracy:
                rotation = module.ROTATION
                check_reference_initial(scene, rotation, request.get("reference_snapshot"))
                helpers = import_module("novaphy.tools.mpm.water_cube_metrics_cpu")
                mass = array(scene.model.particle_mass)
                active = (array(scene.model.particle_flags) & 1) != 0
                volume = helpers.CICEffectiveVolume(
                    particle_state(scene, rotation)[0], mass.astype(np.float64) / config.args.density,
                    config.args.voxel_size if config.args.diagnostic_h is None else config.args.diagnostic_h,
                    config.args.diagnostic_origin, active,
                )

                def sample_volume():
                    return volume_sample(scene, volume, rotation, 1 / config.args.fps)

                capture = Diagnostic(lambda: initial_metadata(scene, rotation), sample_volume, frame=sample_volume)
            if kind == "volume":
                details["density_projection"] = vars(example.options)
            if int(scene.model.particle_count) != spec["particles"]:
                raise ValueError("Native fluid particle count differs from the sealed scene")
            details.update(
                backend_numerics=dict(backend.mpm_accumulation_info()), backend_sha256=_sha256(backend.__file__),
                scene_settings=config.report(), internal_solver_graph=False, demo_args=argv,
            )
            yield scene, module, example.step, capture, details
        finally:
            if collector is not None:
                collector.close()


class TGVScene:
    def __init__(self, spec, case):
        self.module = import_module(spec["module"])
        dt = case["frame_dt_seconds"] / case["substeps"]
        self.model, self.solver, self.meta = self.module.make_case(
            "tgv", spec["cells"], spec["reynolds"], collision_model=spec["implementation"],
            population_storage=spec["storage"], timestep_star=dt,
        )
        self.state, self.out = self.model.state(), self.model.state()
        self.case, self.frames = case, 0

    def step(self):
        self.solver.step(self.state, self.out, None, None, self.case["substeps"] * self.solver.internal_dt)
        if self.solver.last_substeps != self.case["substeps"]:
            raise ValueError("Incomplete native TGV lattice steps")
        self.state, self.out = self.out, self.state
        self.frames += 1

    def energy(self):
        grid = self.solver.export_dense_grid(max_cells=self.solver.cell_count)
        velocity, density = grid["velocity"], grid["density"]
        if not np.isfinite(velocity).all() or not np.isfinite(density).all() or density.min() <= 0:
            raise ValueError("Nonfinite native TGV field or nonpositive density")
        measure = import_module("novaphy.demos.lbm_validation.tgv_diagnostics").kinetic_energy
        return float(measure(velocity, density, 1.0, self.meta["velocity_scale_m_s"]))


@contextmanager
def prepare_trial(request, *, spec, output):
    case, protocol = request["case"], request["protocol"]
    kind, accuracy = spec["metric"], request["mode"] == "accuracy"
    if case.get("metric") != kind or case["world_count"] != 1 or protocol["graph"]:
        raise ValueError("Native fluid trials require the sealed metric, one world and scene graph OFF")
    dt = 0.01 if kind == "tgv" else 1 / 60
    expected_substeps = 6 if kind == "tgv" else 2 if kind == "particles" else 1
    if case["substeps"] != expected_substeps or not math.isclose(case["frame_dt_seconds"], dt, abs_tol=1e-12):
        raise ValueError("Native fluid frame interval or substep count differs from the suite")
    if kind == "tgv":
        if protocol["deterministic"]:
            raise ValueError("Native TGV uses ordinary LBM mode")
        scene = TGVScene(spec, case)
        def sample():
            return {"energy": scene.energy()}
        capture = Diagnostic(lambda: dict(grid_size=spec["cells"], reynolds=spec["reynolds"]), sample, frame=sample)
        info = dict(
            deterministic=False, native_internal_dt=float(scene.solver.internal_dt), scene_settings=scene.meta,
            model_counts=dict(world_count=1, body_count=0, joint_count=0, joint_dof_count=0,
                              particle_count=0, collidable_shape_count=0),
        )
        def finish():
            check_frames(request, scene.frames)
            scene.energy()

        yield _trial(request, spec, scene.module, scene.step, capture if accuracy else None, info, finish)
    else:
        with mpm_scene(request, spec, output) as (scene, module, advance, capture, info):
            info["model_counts"] = counts(scene)

            def finish():
                check_frames(request, scene.frame)
                particle_state(scene)

            yield _trial(request, spec, module, advance, capture, info, finish)


def check_frames(request, completed):
    protocol = request["protocol"]
    expected = protocol["num_frames"] + (
        protocol["warmup_frames"] if request["mode"] == "performance" else 0
    )
    if completed != expected:
        raise ValueError(f"Native fluid completed {completed} frames, expected {expected}")


def _trial(request, spec, module, advance, capture, info, finish):
    case = request["case"]
    runtime = dict(
        engine_version=novaphy.version(), example=module.__name__, example_sha256=_sha256(module.__file__),
        adapter_sha256=_sha256(__file__), extension_sha256=_sha256(novaphy._core.__file__),
        implementation=spec["implementation"], device_uuid=_cuda_device_uuid(0),
        frame_dt_seconds=case["frame_dt_seconds"], substeps=case["substeps"], world_count=1,
        solver=spec["solver"], collision=case["collision"], graph=False,
        contact_stage=request["protocol"].get("contact_stage"), **info,
    )
    return Trial(
        "NovaPhy", runtime, advance, synchronize,
        diagnostic=capture, check_final=finish,
    )
