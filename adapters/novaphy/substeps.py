# SPDX-License-Identifier: Apache-2.0
"""Export native rigid state and retained contacts after every solver substep."""

from contextlib import contextmanager
from unittest.mock import patch

import novaphy
import numpy as np

from newton_test_tools import Diagnostic
from newton_test_tools.conservation import xpbd_contact_geometry


def array(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def check_overflow(pipeline):
    pipeline.refresh_diagnostics()
    for kind in ("candidate_pair", "rigid_contact", "triangle_pair", "soft_contact"):
        if getattr(pipeline, f"last_{kind}_overflow_count"):
            raise ValueError(f"NovaPhy collision {kind} buffer overflow")


def diagnostics(scene, solver_name, runtime):
    model, contacts, solver = scene.model, scene.contacts, scene.solver
    mass, flags = array(model.body_mass), array(model.body_flags)
    dynamic = (
        (mass > 0)
        & (array(model.body_inv_mass) > 0)
        & ((flags & int(novaphy.BodyFlags.DYNAMIC)) != 0)
        & ((flags & int(novaphy.BodyFlags.KINEMATIC)) == 0)
    )
    shape_body, shape_world = array(model.shape_body), array(model.shape_world)
    margins = np.maximum(0, array(model.shape_margin))
    dof = 6 * int(np.count_nonzero(dynamic)) if solver_name == "xpbd" else int(model.joint_dof_count)
    runtime.update(effective_dof=dof, observed_substeps=0)

    def read_model():
        return dict(
            mass=mass[dynamic], com_body=array(model.body_com)[dynamic],
            inertia_body=array(model.body_inertia)[dynamic], body_world=array(model.body_world)[dynamic],
            gravity=array(model.gravity).reshape(-1, 3)[0],
            velocity_convention="com_world", effective_dof=dof,
        )

    def state_sample(state, **geometry):
        q, qd = array(state.body_q_array), array(state.body_qd_array)
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Nonfinite NovaPhy substep state")
        return dict(body_q=q[dynamic], body_qd=qd[dynamic], **geometry)

    def initial():
        return state_sample(
            scene.state_0, contact_distance_m=np.empty(0, np.float32), normal_compression_N=np.empty(0, np.float32)
        )

    @contextmanager
    def observe(record):
        original = solver.step
        force_buffer = (
            novaphy.DeviceArrayFloat6.zeros(contacts.rigid_contact_max + contacts.soft_contact_max, model.device)
            if solver_name == "mujoco" else None
        )

        def monitored(state_in, state_out, control, contact_buffer, dt):
            if contact_buffer is not contacts:
                raise ValueError("Substep used an unexpected contact buffer")
            check_overflow(scene.collision_pipeline)
            count = int(contacts.rigid_count())
            if not 0 <= count <= contacts.rigid_contact_max:
                raise ValueError("NovaPhy rigid contact count exceeds capacity")

            def prefix(field):
                return array(getattr(contacts, f"rigid_contact_{field}").prefix(count))

            first, second = prefix("shape0"), prefix("shape1")
            if np.any((first < 0) | (first >= len(shape_body))) or np.any(
                (second < 0) | (second >= len(shape_body))
            ):
                raise ValueError("Invalid substep collision shape identity")
            w0, w1 = shape_world[first], shape_world[second]
            if np.any((w0 >= 0) & (w1 >= 0) & (w0 != w1)):
                raise ValueError("Cross-world NovaPhy contact")
            world = np.maximum(w0, w1)
            if np.any(world < 0):
                raise ValueError("Contact has no owning world")
            normal = prefix("normal")
            radius0, radius1 = prefix("margin0"), prefix("margin1")
            if solver_name == "mujoco":
                # Match the distances assembled by the native MuJoCo solver:
                # geometric radii exclude the authored contact margins.
                radius0, radius1 = radius0 - margins[first], radius1 - margins[second]
            distance, point = xpbd_contact_geometry(
                array(state_in.body_q_array), shape_body, first, second,
                prefix("point0"), prefix("point1"), normal, radius0, radius1,
            )
            original(state_in, state_out, control, contact_buffer, dt)
            if solver_name == "xpbd":
                solver.update_contacts(contacts)
                force = array(contacts.force.prefix(count)).reshape(-1, 6)[:, :3]
                compression = -np.einsum("ni,ni->n", force, normal)
            else:
                if count:
                    solver.export_contact_forces(contacts, force_buffer)
                force = -array(force_buffer.prefix(count)).reshape(-1, 6)[:, :3]
                compression = np.einsum("ni,ni->n", force, normal)
            if not all(np.isfinite(x).all() for x in (distance, force, compression)):
                raise ValueError("Nonfinite NovaPhy substep contact data")
            active = (distance < 0) | (np.linalg.norm(force, axis=1) > 1e-12)
            record(state_sample(
                state_out, contact_distance_m=distance[active], normal_compression_N=compression[active],
                contact_world=world[active], shape0=first[active], shape1=second[active],
                contact_point_m=point[active], contact_normal=normal[active], contact_force_N=force[active],
            ))
            runtime["observed_substeps"] += 1

        with patch.object(solver, "step", monitored):
            yield

    return Diagnostic(read_model, initial, substeps=observe)
