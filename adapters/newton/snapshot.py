# SPDX-License-Identifier: Apache-2.0
"""Export Newton rigid bodies and contacts in benchmark coordinates."""

from __future__ import annotations

from typing import Any

import newton
import warp as wp


class RigidReader:
    """Read the last collision buffer and a detached COM-velocity state."""

    def __init__(self, scene: Any, *, mujoco: bool) -> None:
        self.scene: Any = scene
        model = scene.model
        worlds = model.body_world.numpy().tolist()
        self._bodies: list[tuple[int, str]] = [
            (0 if world == -1 and model.world_count == 1 else world, name)
            for world, name in zip(worlds, model.body_label, strict=True)
        ]
        self._shapes: list[tuple[int, str, bool]] = [
            (world, label, bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)))
            for world, label, flags in zip(
                model.shape_world.numpy().tolist(), model.shape_label, model.shape_flags.numpy().tolist(), strict=True
            )
        ]
        self._fk_state: Any = model.state() if mujoco else None
        self.model_rows: list[dict[str, Any]] = [
            {
                "world": world,
                "name": name,
                "mass": mass,
                "inv_mass": inv_mass,
                "com_body": com,
                "inertia_body": inertia,
                "flags": flags,
            }
            for (world, name), mass, inv_mass, com, inertia, flags in zip(
                self._bodies,
                model.body_mass.numpy().tolist(),
                model.body_inv_mass.numpy().tolist(),
                model.body_com.numpy().tolist(),
                model.body_inertia.numpy().tolist(),
                model.body_flags.numpy().tolist(),
                strict=True,
            )
        ]
        self.shape_rows: list[dict[str, Any]] = sorted(
            (
                {
                    "world": world,
                    "name": label,
                    "body": self._bodies[owner][1] if owner >= 0 else None,
                    "type": shape_type,
                }
                for (local_world, label, collidable), owner, shape_type in zip(
                    self._shapes, model.shape_body.numpy().tolist(), model.shape_type.numpy().tolist(), strict=True
                )
                if collidable
                for world in (range(model.world_count) if local_world == -1 else (local_world,))
            ),
            key=lambda row: (row["world"], row["name"]),
        )

    def read_initial(self) -> dict[str, Any]:
        return {"model": self.model_rows, "shapes": self.shape_rows, "initial_state": self.bodies()}

    def bodies(self) -> list[dict[str, Any]]:
        state = self.scene.state_0
        if self._fk_state is not None:
            # Reconstruct world COM twists from a copy of the observed joint state.
            # Reading the trajectory does not update the actual solver state.
            scratch = self._fk_state
            wp.copy(scratch.joint_q, state.joint_q)
            wp.copy(scratch.joint_qd, state.joint_qd)
            newton.eval_fk(self.scene.model, scratch.joint_q, scratch.joint_qd, scratch)
            state = scratch
        return [
            {"world": world, "name": name, "q": q, "qd": qd}
            for (world, name), q, qd in zip(
                self._bodies, state.body_q.numpy().tolist(), state.body_qd.numpy().tolist(), strict=True
            )
        ]

    def contact_pairs(self) -> list[list[Any]]:
        contacts = self.scene.contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        if not 0 <= count <= contacts.rigid_contact_max:
            raise ValueError(f"contact count {count} is outside the collision buffer")
        pairs = []
        for a, b in zip(
            contacts.rigid_contact_shape0.numpy()[:count].tolist(),
            contacts.rigid_contact_shape1.numpy()[:count].tolist(),
            strict=True,
        ):
            if not (0 <= a < len(self._shapes) and 0 <= b < len(self._shapes)) or a == b:
                raise ValueError(f"invalid contact shape indices {a}, {b}")
            world_a, name_a, valid_a = self._shapes[a]
            world_b, name_b, valid_b = self._shapes[b]
            if not (valid_a and valid_b) or (world_a >= 0 and world_b >= 0 and world_a != world_b):
                raise ValueError("contact refers to noncollidable or cross-world shapes")
            world = max(world_a, world_b)
            if world == -1:
                if self.scene.model.world_count != 1:
                    raise ValueError("contact between two shared shapes has no unique world")
                world = 0
            pairs.append([world, name_a, name_b])
        return pairs

    def observe(self) -> tuple[list[dict[str, Any]], list[list[Any]]]:
        return self.bodies(), self.contact_pairs()
