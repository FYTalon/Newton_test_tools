# SPDX-License-Identifier: Apache-2.0
"""Read completed substeps from WanPhys's native rigid scene."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

from newton_test_tools import Diagnostic
from newton_test_tools.conservation import pyramidal_forces, xpbd_contact_geometry
from newton_test_tools.diagnostics import Sample


def _slice(array: Any, count: int) -> np.ndarray:
    return array[:count].numpy()


def diagnostics(scene: Any, solver_name: str) -> Diagnostic:
    model = scene.model
    mass = model.body_mass.numpy()
    flags = model.body_flags.numpy()
    dynamic = (mass > 0) & (model.body_inv_mass.numpy() > 0) & ((flags & 1) != 0) & ((flags & 2) == 0)
    shape_body = model.shape_body.numpy()
    shape_world = model.shape_world.numpy()

    def read_model() -> Sample:
        return {
            "mass": mass[dynamic],
            "com_body": model.body_com.numpy()[dynamic],
            "inertia_body": model.body_inertia.numpy()[dynamic],
            "body_world": model.body_world.numpy()[dynamic],
            "gravity": model.gravity.numpy().reshape(-1, 3)[0],
            "velocity_convention": "com_world",
            "effective_dof": 6 * np.count_nonzero(dynamic) if solver_name == "xpbd" else int(model.joint_dof_count),
        }

    def state_sample(state: Any, **contacts: Any) -> Sample:
        return {
            "body_q": state.body_q.numpy()[dynamic],
            "body_qd": state.body_qd.numpy()[dynamic],
            **contacts,
        }

    def initial() -> Sample:
        return state_sample(
            (scene.state_0 if solver_name == "xpbd" else scene.state),
            contact_distance_m=np.empty(0),
            normal_compression_N=np.empty(0),
        )

    @contextmanager
    def observe_xpbd(record: Callable[[Sample], None]) -> Iterator[None]:
        class Observer:
            def before(self, example: Any, state: Any, contacts: Any) -> None:
                count = int(contacts.rigid_contact_count.numpy()[0])
                if not 0 <= count <= contacts.rigid_contact_max:
                    raise ValueError("rigid contact buffer overflow")
                self.shape0 = _slice(contacts.rigid_contact_shape0, count)
                self.shape1 = _slice(contacts.rigid_contact_shape1, count)
                self.normal = _slice(contacts.rigid_contact_normal, count)
                self.distance, self.point = xpbd_contact_geometry(
                    state.body_q.numpy(),
                    shape_body,
                    self.shape0,
                    self.shape1,
                    _slice(contacts.rigid_contact_point0, count),
                    _slice(contacts.rigid_contact_point1, count),
                    self.normal,
                    _slice(contacts.rigid_contact_margin0, count),
                    _slice(contacts.rigid_contact_margin1, count),
                )

            def after(self, example: Any, state: Any, contacts: Any) -> None:
                if contacts.force is None:
                    raise ValueError("XPBD contact force output is unavailable")
                force = _slice(contacts.force, len(self.distance))[:, :3]
                if not np.isfinite(self.distance).all() or not np.isfinite(force).all():
                    raise ValueError("nonfinite XPBD contact data")
                active = (self.distance < 0.0) | (np.linalg.norm(force, axis=1) > 1e-12)
                normal_force = -np.einsum("ni,ni->n", force, self.normal)
                record(
                    state_sample(
                        state,
                        contact_distance_m=self.distance[active],
                        normal_compression_N=normal_force[active],
                        contact_normal=self.normal[active],
                        contact_point_m=self.point[active],
                        contact_force_N=force[active],
                        shape0=self.shape0[active],
                        shape1=self.shape1[active],
                        contact_world=np.maximum(shape_world[self.shape0[active]], shape_world[self.shape1[active]]),
                    )
                )

        if not scene.contact_force_enabled:
            raise ValueError("the XPBD scene must enable substep contact forces")
        with patch.object(scene, "substep_observer", Observer()):
            yield

    @contextmanager
    def observe_mujoco(record: Callable[[Sample], None]) -> Iterator[None]:
        solver = scene.solver
        data = solver.native_data
        step = solver.step

        def monitored(state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
            step(state_in, state_out, control, contacts, dt)
            count = int(data.nacon.numpy()[0])
            if not 0 <= count <= data.naconmax:
                raise ValueError("MuJoCo contact buffer overflow")
            if int(contacts.rigid_contact_count.numpy()[0]) != count:
                raise ValueError("external contacts were truncated during MuJoCo assembly")
            contact = data.contact
            world = _slice(contact.worldid, count)
            distance = _slice(contact.dist, count)
            frame = _slice(contact.frame, count)
            shape0, shape1 = (_slice(getattr(contacts, f"rigid_contact_shape{index}"), count) for index in (0, 1))
            constraints = data.efc.force.numpy()
            if np.any(data.nefc.numpy() >= constraints.shape[1]) or np.any((world < 0) | (world >= model.world_count)):
                raise ValueError("invalid MuJoCo constraint capacity or contact world")
            local = pyramidal_forces(
                world,
                _slice(contact.dim, count),
                _slice(contact.efc_address, count),
                _slice(contact.friction, count),
                constraints,
            )
            force = np.einsum("nij,ni->nj", frame, local[:, :3])
            if not np.isfinite(distance).all() or not np.isfinite(force).all():
                raise ValueError("nonfinite MuJoCo contact data")
            active = (distance < 0.0) | (np.linalg.norm(force, axis=1) > 1e-12)
            record(
                state_sample(
                    state_out,
                    contact_distance_m=distance[active],
                    normal_compression_N=local[active, 0],
                    contact_normal=frame[active, 0],
                    contact_point_m=_slice(contact.pos, count)[active],
                    contact_force_N=force[active],
                    shape0=shape0[active],
                    shape1=shape1[active],
                    contact_world=world[active],
                )
            )

        with patch.object(solver, "step", monitored):
            yield

    observe = observe_xpbd if solver_name == "xpbd" else observe_mujoco
    return Diagnostic(read_model, initial, substeps=observe)
