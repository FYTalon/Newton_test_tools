# SPDX-License-Identifier: Apache-2.0
"""Use the NovaPhy G1 and ANYmal demos with the benchmark's execution modes."""

from __future__ import annotations

from functools import partial
from importlib import import_module
from typing import Any
from unittest.mock import patch

import novaphy


class RobotScene:
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
                disable_graph = getattr(runtime.policy_io, "disable_graph", None)
                if disable_graph is None:
                    runtime.policy_io._is_cuda = False
                else:
                    disable_graph()
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


def create_robot(
    case: dict[str, Any], settings: dict[str, Any], frames: int, spec: dict[str, Any],
) -> tuple[RobotScene, Any, dict]:
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
        return make_config(**{**kwargs, "deterministic": deterministic, "disable_sensors": True, "use_mujoco_contacts": False})

    def collision_pipeline(*args, **kwargs):
        return make_pipeline(*args, **{**kwargs, "deterministic": deterministic})

    with (
        patch.object(example, "make_mujoco_solver_config", solver_config),
        patch.object(novaphy, "CollisionPipeline", collision_pipeline),
    ):
        runtime = getattr(example, spec["factory"])(config)

    if not runtime.solver.settings.disable_sensors or runtime.solver.settings.use_mujoco_contacts:
        raise RuntimeError("The robot trial requires external collisions and disabled sensors")
    scene = RobotScene(runtime, config, example, getattr(example, spec["step"]), graph=settings["graph"])
    return scene, example, options
