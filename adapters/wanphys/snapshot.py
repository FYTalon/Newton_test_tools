# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
"""Export WanPhys rigid states and retained contact pairs for comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

COLLIDE_SHAPES = 1 << 1


def _array(owner: Any, field: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(getattr(owner, field).numpy())
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field} has invalid shape or nonfinite values")
    return result


def _labels(labels: list[str], overrides: dict[str, str]) -> list[str]:
    for index in overrides:
        if not 0 <= int(index) < len(labels):
            raise ValueError(f"label override {index} is outside the scene")
    names = [overrides.get(str(index), label) for index, label in enumerate(labels)]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every body and shape must have a nonempty label")
    return names


class RigidReader:
    """Read one frame without modifying the state used by the solver."""

    def __init__(self, scene: Any, state_name: str, labels: dict[str, Any], *, mujoco: bool) -> None:
        self._scene = scene
        self._state_name = state_name
        self._model = scene.model
        self._world_count = int(self._model.world_count)
        self._fk_state = self._model.state() if mujoco else None
        self.model_rows = self._read_model(labels.get("bodies", {}))
        self.shape_rows = self._read_shapes(labels.get("shapes", {}))

    def _read_model(self, aliases: dict[str, str]) -> list[dict[str, Any]]:
        model = self._model
        count = int(model.body_count)
        names = _labels(model.body_label, aliases)
        worlds = _array(model, "body_world", (count,)).tolist()
        if any(not -1 <= world < self._world_count for world in worlds):
            raise ValueError("body world index is outside the scene")
        self._body_ids = [
            (0 if world == -1 and self._world_count == 1 else world, name)
            for world, name in zip(worlds, names, strict=True)
        ]
        return [
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
                self._body_ids,
                _array(model, "body_mass", (count,)).tolist(),
                _array(model, "body_inv_mass", (count,)).tolist(),
                _array(model, "body_com", (count, 3)).tolist(),
                _array(model, "body_inertia", (count, 3, 3)).tolist(),
                _array(model, "body_flags", (count,)).tolist(),
                strict=True,
            )
        ]

    def _read_shapes(self, aliases: dict[str, str]) -> list[dict[str, Any]]:
        model = self._model
        count = int(model.shape_count)
        names = _labels(model.shape_label, aliases)
        worlds = _array(model, "shape_world", (count,)).tolist()
        owners = _array(model, "shape_body", (count,)).tolist()
        types = _array(model, "shape_type", (count,)).tolist()
        flags = _array(model, "shape_flags", (count,)).tolist()
        self._shapes = [
            (world, name, bool(flag & COLLIDE_SHAPES)) for world, name, flag in zip(worlds, names, flags, strict=True)
        ]
        rows = []
        for (local_world, name, collidable), owner, kind in zip(self._shapes, owners, types, strict=True):
            if not -1 <= local_world < self._world_count or not -1 <= owner < len(self._body_ids):
                raise ValueError("shape refers to an invalid world or body")
            if not collidable:
                continue
            for world in range(self._world_count) if local_world == -1 else (local_world,):
                body = None
                if owner >= 0:
                    body_world, body = self._body_ids[owner]
                    if body_world not in (-1, world):
                        raise ValueError("collision shape belongs to a body in another world")
                rows.append({"world": world, "name": name, "body": body, "type": kind})
        return sorted(rows, key=lambda row: (row["world"], row["name"]))

    def read_initial(self) -> dict[str, Any]:
        return {"model": self.model_rows, "shapes": self.shape_rows, "initial_state": self.bodies()}

    def bodies(self) -> list[dict[str, Any]]:
        live_state = getattr(self._scene, self._state_name)
        state = live_state
        if self._fk_state is not None:
            # The owned MuJoCo joint arrays already use the public FK convention.
            # Reconstruct world COM twists in an independent state.
            state = self._fk_state
            wp.copy(state.joint_q, live_state.joint_q)
            wp.copy(state.joint_qd, live_state.joint_qd)
            self._model.eval_forward_kinematics(state)
        count = len(self._body_ids)
        poses = _array(state, "body_q", (count, 7))
        twists = _array(state, "body_qd", (count, 6))
        if state is not live_state:
            actual = _array(live_state, "body_q", (count, 7)).astype(np.float64)
            reconstructed = poses.astype(np.float64)
            position_errors = np.linalg.norm(reconstructed[:, :3] - actual[:, :3], axis=1)
            rotation_a, rotation_b = reconstructed[:, 3:], actual[:, 3:]
            orientation_errors = np.minimum(
                np.linalg.norm(rotation_a - rotation_b, axis=1), np.linalg.norm(rotation_a + rotation_b, axis=1)
            )
            if np.any(position_errors > 1.0e-4) or np.any(orientation_errors > 1.0e-4):
                raise ValueError("detached MuJoCo FK pose differs from the live body pose")
        return [
            {"world": world, "name": name, "q": q, "qd": qd}
            for (world, name), q, qd in zip(self._body_ids, poses.tolist(), twists.tolist(), strict=True)
        ]

    def contact_pairs(self) -> list[list[Any]]:
        contacts = self._scene.contacts
        if contacts is None:
            raise ValueError("the scene did not retain a collision buffer")
        count = int(contacts.rigid_contact_count.numpy()[0])
        capacity = int(contacts.rigid_contact_max)
        if not 0 <= count <= capacity:
            raise ValueError(f"contact count {count} is outside the buffer of {capacity}")
        pairs = []
        for a, b in zip(
            _array(contacts, "rigid_contact_shape0", (capacity,))[:count].tolist(),
            _array(contacts, "rigid_contact_shape1", (capacity,))[:count].tolist(),
            strict=True,
        ):
            if a == b or not (0 <= a < len(self._shapes) and 0 <= b < len(self._shapes)):
                raise ValueError(f"invalid contact shape indices {a}, {b}")
            world_a, name_a, valid_a = self._shapes[a]
            world_b, name_b, valid_b = self._shapes[b]
            if not (valid_a and valid_b) or (world_a >= 0 and world_b >= 0 and world_a != world_b):
                raise ValueError("contact refers to noncollidable or cross-world shapes")
            world = max(world_a, world_b)
            if world == -1:
                if self._world_count != 1:
                    raise ValueError("contact between two shared shapes has no unique world")
                world = 0
            pairs.append([world, name_a, name_b])
        # Monitor canonicalizes unordered shape pairs and removes duplicates.
        return pairs

    def observe(self) -> tuple[list[dict[str, Any]], list[list[Any]]]:
        return self.bodies(), self.contact_pairs()
