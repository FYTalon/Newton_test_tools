# SPDX-License-Identifier: Apache-2.0
"""Bind the two sealed task 2 scenes to native NovaPhy diagnostics."""

import math
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import novaphy
import numpy as np

from newton_test_tools import Trial

from .adapter import _check_first_native_replay, _cuda_device_uuid, _sha256, _xpbd_scene
from .substeps import array, check_overflow, diagnostics


class SwarmRuntime:
    """Expose the public swarm demo's model, policy and complete frame to the worker."""

    def __init__(self, module, worlds):
        self.frame_dt, self.substeps = module.FRAME_DT, module.SUBSTEPS
        self.assets = module.resolve_anymal_c_asset_paths()
        self.model = module.build_model(novaphy, self.assets.asset_dir, worlds)
        self.robots = worlds * module.ROBOTS_PER_WORLD
        if self.model.world_count != worlds or self.model.joint_dof_count != self.robots * 18:
            raise ValueError("Public swarm model has unexpected world or joint counts")
        config = module.make_mujoco_solver_config(
            deterministic=False, solver="newton", integrator="implicitfast", cone="pyramidal",
            impratio=1.0, disable_sensors=True, use_mujoco_contacts=False,
            iterations=100, ls_iterations=50, tolerance=1.0e-8, ls_tolerance=0.01, nconmax=4096,
        )
        config.njmax = 1024
        self.solver = novaphy.solvers.SolverMuJoCo(self.model, config)
        self.state_in, self.state_out = self.model.state(), self.model.state()
        for state in (self.state_in, self.state_out):
            novaphy.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.control = self.model.control(coordinate_targets=True)
        self.pipeline = novaphy.CollisionPipeline(
            self.model, broad_phase="explicit", max_triangle_pairs=8_000_000, deterministic=False,
        )
        self.contacts = self.pipeline.contacts()
        self.routes = module.make_route_commands(worlds)
        device = self.model.device
        self.policy = module.load_onnx_policy(self.assets.policy_path, device, obs_width=48, action_width=12)
        self.policy_io = module.make_policy_io(self.state_in, self.robots, device)
        self.observations = novaphy.DeviceArrayFloat.zeros(self.robots * 48, device)
        self.previous_action = novaphy.DeviceArrayFloat.zeros(self.robots * 12, device)
        self.commands = novaphy.DeviceArrayFloat.zeros(self.robots * 3, device)
        self.policy.evaluate_device(self.observations)
        self.frames = 0

    def update_policy(self):
        self.commands.assign(self.routes.update(
            self.state_in.joint_q.numpy().reshape(self.robots, 19),
            self.state_in.joint_qd.numpy().reshape(self.robots, 18), self.frames,
        ).ravel())
        self.policy_io.observe(
            self.state_in.joint_q, self.state_in.joint_qd, self.commands,
            self.previous_action, self.observations,
        )
        self.previous_action = self.policy.evaluate_device(self.observations)
        self.policy_io.apply(self.previous_action, 0.5)
        self.control.joint_target_q.assign_device(self.policy_io.targets)

    def physics_frame(self):
        for _ in range(self.substeps):
            self.state_in.clear_forces()
            self.pipeline.collide(self.state_in, self.contacts)
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, self.frame_dt / self.substeps)
            self.state_in, self.state_out = self.state_out, self.state_in

    def step(self):
        self.update_policy()
        self.physics_frame()
        self.frames += 1


class LargeRobotScene:
    """Retain the group scene's policy, four physics substeps and frame clock."""

    def __init__(self, simulation, graph):
        self.simulation = simulation
        self.model, self.solver = simulation.model, simulation.solver
        self.contacts, self.collision_pipeline = simulation.contacts, simulation.pipeline
        self.frame_dt, self.substeps = simulation.frame_dt, simulation.substeps
        self.frame_graphs = {}
        self.graph_enabled = graph
        self.captures = self.replays = 0

    def capture(self):
        runtime = self.simulation
        self.solver.prepare_contacts(self.contacts)
        novaphy.synchronize_current_cuda_stream()
        first, second = runtime.state_in, runtime.state_out
        try:
            with novaphy.CudaGraphCapture(self.model.device.ordinal) as capture:
                runtime.physics_frame()
            if not capture.captured:
                raise RuntimeError("Large robot physics capture failed")
        finally:
            runtime.state_in, runtime.state_out = first, second
        self.frame_graphs[id(first)] = capture
        self.captures += 1

    @property
    def state_0(self):
        return self.simulation.state_in

    def step(self):
        if not self.frame_graphs:
            self.simulation.step()
            if self.graph_enabled:
                # The first requested warmup frame allocates lazy solver
                # buffers. Capture records the next frame without executing it.
                self.capture()
            return
        runtime = self.simulation
        runtime.update_policy()
        self.frame_graphs[id(runtime.state_in)].launch()
        runtime.frames += 1
        self.replays += 1

    def check_graph(self, total_frames):
        if self.captures != 1 or self.replays != total_frames - 1 or self.simulation.frames != total_frames:
            raise ValueError("Group graph did not execute every requested frame")
        return dict(bootstrap_warmup_frames=1, physics_graph_captures=self.captures, physics_graph_replays=self.replays)


def validate_request(request, spec):
    case, protocol = request["case"], request["protocol"]
    accuracy = request["mode"] == "accuracy"
    expected_substeps, expected_dt = (5, 0.01) if case["solver"] == "xpbd" else (4, 0.02)
    if (
        request["mode"] not in ("accuracy", "performance")
        or case["world_count"] != spec["world_count"]
        or case["substeps"] != expected_substeps
        or not math.isclose(case["frame_dt_seconds"], expected_dt, rel_tol=0, abs_tol=1e-12)
        or protocol["deterministic"] is not False
        or protocol["graph"] is not (not accuracy)
        or (accuracy and protocol["contact_stage"] != "end_of_substep")
        or (not accuracy and protocol.get("warmup_frames", 0) < 1)
    ):
        raise ValueError("Task 2 requires the sealed scale, ordinary mode and substep protocol")


@contextmanager
def large_scene(request, spec):
    case, protocol = request["case"], request["protocol"]
    frames = protocol["num_frames"] + (
        protocol.get("warmup_frames", 0) if request["mode"] == "performance" else 0
    )
    if case["solver"] == "xpbd":
        with _xpbd_scene(case, protocol, frames, spec) as (scene, module, options):
            if request["mode"] == "accuracy":
                scene.model.request_contact_attributes("force")
                scene.contacts = scene.collision_pipeline.contacts()
            yield scene, module, options, {"manifest_sha256": _sha256(Path(__file__).with_name("scenes.json"))}
    else:
        if not novaphy.has_mujoco_cuda():
            raise RuntimeError("Task 2 robots require NovaPhy's CUDA MuJoCo backend")
        module = import_module(spec["module"])
        authored = import_module(spec["scene_module"])
        if (module.FRAME_DT, module.SUBSTEPS) != (case["frame_dt_seconds"], case["substeps"]):
            raise ValueError("Public swarm timing differs from the standard suite")
        simulation = SwarmRuntime(module, case["world_count"])
        scene = LargeRobotScene(simulation, protocol["graph"])
        sources = {
            "scene": Path(authored.__file__),
            "route_commands": Path(module.make_route_commands.__code__.co_filename),
            "robot_model": simulation.assets.urdf_path,
            "policy": simulation.assets.policy_path,
            "asset_revision": simulation.assets.asset_dir / "ASSET_REVISION.json",
            **{f"scene_assets/{path.name}": path for path in authored.SCENE_ASSETS.iterdir()},
        }
        provenance = {name: _sha256(path) for name, path in sorted(sources.items()) if path.is_file()}
        options = dict(world_count=case["world_count"], robots=simulation.robots, frames=frames)
        yield scene, module, options, {"scene_sources": provenance}


@contextmanager
def prepare_trial(request, *, spec, output):
    validate_request(request, spec)
    case, protocol = request["case"], request["protocol"]
    accuracy = request["mode"] == "accuracy"
    with large_scene(request, spec) as (scene, module, options, provenance):
        model = scene.model
        mass, inverse, flags = array(model.body_mass), array(model.body_inv_mass), array(model.body_flags)
        dynamic = (mass > 0) & (inverse > 0) & ((flags & int(novaphy.BodyFlags.KINEMATIC)) == 0)
        dynamic_count = int(np.count_nonzero(dynamic))
        dof = 6 * dynamic_count if case["solver"] == "xpbd" else int(model.joint_dof_count)
        if (dynamic_count, dof, int(model.world_count)) != (
            spec["dynamic_bodies"], spec["effective_dof"], case["world_count"]
        ):
            raise ValueError("Task 2 model differs in body/world count or effective DOF")
        if (
            scene.solver.settings.deterministic or scene.collision_pipeline.deterministic
            or (scene.graph_enabled if isinstance(scene, LargeRobotScene) else bool(scene.frame_graphs))
            != protocol["graph"]
        ):
            raise ValueError("Task 2 runtime settings differ from the request")
        if not math.isclose(scene.frame_dt, case["frame_dt_seconds"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Task 2 runtime frame interval differs from the suite")
        runtime = dict(
            engine_version=novaphy.version(),
            example=module.__name__, example_sha256=_sha256(module.__file__),
            adapter_sha256=_sha256(__file__), extension_sha256=_sha256(novaphy._core.__file__),
            **provenance, args=options, device_uuid=_cuda_device_uuid(model.device.ordinal),
            frame_dt_seconds=float(scene.frame_dt), substeps=case["substeps"],
            world_count=int(model.world_count), solver=case["solver"], collision="external",
            deterministic=False, collision_deterministic=False, graph=protocol["graph"],
            effective_dof=dof,
            model_counts={
                **{field: int(getattr(model, field)) for field in (
                    "world_count", "body_count", "joint_count", "joint_dof_count", "particle_count"
                )},
                "collidable_shape_count": int(np.count_nonzero(
                    array(model.shape_flags) & int(novaphy.ShapeFlags.COLLIDE_SHAPES)
                )),
            },
        )
        capture = None
        check_graph = (
            _check_first_native_replay(scene, runtime)
            if protocol["graph"] and not isinstance(scene, LargeRobotScene) else None
        )
        if accuracy:
            runtime["contact_stage"] = "end_of_substep"
            capture = diagnostics(scene, case["solver"], runtime)

        def check_final():
            check_overflow(scene.collision_pipeline)
            if isinstance(scene, LargeRobotScene):
                if scene.simulation.frames != protocol["num_frames"] + (0 if accuracy else protocol["warmup_frames"]):
                    raise ValueError("Incomplete public swarm frame count")
                if not all(np.isfinite(array(getattr(scene.state_0, field))).all() for field in ("joint_q", "joint_qd")):
                    raise ValueError("Nonfinite public swarm final state")
            if accuracy and runtime["observed_substeps"] != protocol["num_frames"] * case["substeps"]:
                raise ValueError("Task 2 capture missed completed physics substeps")
            if check_graph is not None:
                check_graph()
            if protocol["graph"] and isinstance(scene, LargeRobotScene):
                runtime["graph_audit"] = scene.check_graph(protocol["warmup_frames"] + protocol["num_frames"])

        yield Trial(
            engine="NovaPhy", runtime=runtime, step=scene.step,
            synchronize=novaphy.synchronize_current_cuda_stream, diagnostic=capture, check_final=check_final,
        )
