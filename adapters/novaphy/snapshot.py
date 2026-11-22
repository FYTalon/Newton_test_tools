# SPDX-License-Identifier: Apache-2.0
"""Read NovaPhy rigid-body states and effective contacts in the benchmark format."""

from __future__ import annotations

from typing import Any

import novaphy
import numpy as np


def _array(owner: Any, field: str, shape: tuple[int, ...]) -> np.ndarray:
    data = getattr(owner, field)
    array = np.asarray(data.numpy() if hasattr(data, "numpy") else data)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite with shape {shape}, got {array.shape}")
    return array


def _body_world(world: int, world_count: int) -> int:
    if not -1 <= world < world_count:
        raise ValueError(f"Unexpected body world {world} in scene with {world_count} worlds")
    return 0 if world == -1 and world_count == 1 else world


class RigidReader:
    def __init__(
        self, scene: Any, reference: dict[str, Any], shape_flags: np.ndarray, *,
        mujoco: bool, shape_aliases: dict[str, str],
    ) -> None:
        self.scene = scene
        model = scene.model
        self.fk_state = model.state() if mujoco else None
        self.world_count = int(model.world_count)
        self.body_count = int(model.body_count)
        count = self.body_count
        self.body_keys = [
            (_body_world(world, self.world_count), label)
            for world, label in zip(_array(model, "body_world", (count,)).tolist(), model.body_label, strict=True)
        ]
        expected = {(row["world"], row["name"]) for row in reference["bodies"]}
        if len(set(self.body_keys)) != count or set(self.body_keys) != expected:
            raise ValueError(f"NovaPhy body identities differ from Newton: {self.body_keys} vs {sorted(expected)}")
        self.model_rows = [
            dict(world=world, name=name, mass=mass, inv_mass=inv_mass, com_body=com, inertia_body=inertia, flags=flags)
            for (world, name), mass, inv_mass, com, inertia, flags in zip(
                self.body_keys,
                _array(model, "body_mass", (count,)).tolist(),
                _array(model, "body_inv_mass", (count,)).tolist(),
                _array(model, "body_com", (count, 3)).tolist(),
                _array(model, "body_inertia", (count, 3, 3)).tolist(),
                _array(model, "body_flags", (count,)).tolist(),
                strict=True,
            )
        ]

        self.shape_rows, self.shape_names, self.shape_worlds = self._map_shapes(
            model, reference["shapes"], shape_flags, shape_aliases,
        )

    def _map_shapes(self, model: Any, expected_rows: list, shape_flags: np.ndarray, shape_aliases: dict[str, str]):
        expected_shapes = {(row["world"], row["name"]): row for row in expected_rows}
        count = int(model.shape_count)
        worlds = _array(model, "shape_world", (count,)).tolist()
        local_shapes = []
        for index, (local_world, owner, kind, flags, label) in enumerate(zip(
            worlds,
            _array(model, "shape_body", (count,)).tolist(),
            _array(model, "shape_type", (count,)).tolist(),
            shape_flags.tolist(),
            model.shape_label,
            strict=True,
        )):
            if flags & int(novaphy.ShapeFlags.COLLIDE_SHAPES) == 0:
                continue
            if not -1 <= owner < self.body_count or not -1 <= local_world < self.world_count:
                raise ValueError(f"Invalid NovaPhy world or owner for shape {index}: {local_world}, {owner}")
            body = self.body_keys[owner][1] if owner >= 0 else None
            alias = shape_aliases.get(str(index))
            if alias is not None:
                expected = expected_shapes.get((local_world, alias))
                if label != f"shape_{index}" or expected is None or (expected["body"], expected["type"]) != (body, kind):
                    raise ValueError(f"Stale collision shape alias for NovaPhy shape {index}")
                label = alias
            for world in range(self.world_count) if local_world == -1 else (local_world,):
                local_shapes.append((index, world, label, body, kind))

        rows = []
        names = {}
        unresolved = []
        for index, world, label, body, kind in local_shapes:
            key = (world, label)
            candidate = expected_shapes.get(key)
            if candidate is not None and (candidate["body"], candidate["type"]) == (body, kind):
                rows.append(dict(world=world, name=label, body=body, type=kind))
                names[(world, index)] = label
                del expected_shapes[key]
            else:
                unresolved.append((index, world, label, body, kind))
        for index, world, label, body, kind in unresolved:
            matches = [key for key, row in expected_shapes.items()
                       if (row["world"], row["body"], row["type"]) == (world, body, kind)]
            if len(matches) != 1:
                raise ValueError(f"Unmatched or ambiguous NovaPhy shape {(index, world, label, body, kind)}: {matches}")
            key, = matches
            name = key[1]
            rows.append(dict(world=world, name=name, body=body, type=kind))
            names[(world, index)] = name
            del expected_shapes[key]
        if expected_shapes:
            raise ValueError(f"NovaPhy did not provide Newton shapes: {expected_shapes}")
        return sorted(rows, key=lambda row: (row["world"], row["name"])), names, worlds

    def bodies(self) -> list[dict[str, Any]]:
        state = self.scene.state_0
        if self.fk_state is not None:
            novaphy.eval_fk(self.scene.model, state.joint_q, state.joint_qd, self.fk_state)
            state = self.fk_state
        return [
            dict(world=world, name=name, q=q, qd=qd)
            for (world, name), q, qd in zip(
                self.body_keys,
                _array(state, "body_q", (self.body_count, 7)).tolist(),
                _array(state, "body_qd", (self.body_count, 6)).tolist(),
                strict=True,
            )
        ]

    def read_initial(self) -> dict[str, Any]:
        return dict(
            model=self.model_rows,
            shapes=self.shape_rows,
            initial_state=self.bodies(),
            velocity_convention="com_world",
        )

    def observe(self) -> tuple[list[dict[str, Any]], list[list[Any]]]:
        contacts = self.scene.contacts
        count, capacity = int(contacts.rigid_count()), int(contacts.rigid_contact_max)
        if not 0 <= count <= capacity:
            raise ValueError(f"NovaPhy contact count {count} is outside capacity {capacity}")
        pairs = []
        for left, right in zip(
            _array(contacts, "rigid_contact_shape0", (capacity,))[:count].tolist(),
            _array(contacts, "rigid_contact_shape1", (capacity,))[:count].tolist(),
            strict=True,
        ):
            if left == right or not 0 <= min(left, right) <= max(left, right) < len(self.shape_worlds):
                raise ValueError(f"Invalid NovaPhy contact shape pair {left}, {right}")
            worlds = self.shape_worlds[left], self.shape_worlds[right]
            world = max(worlds)
            if min(worlds) >= 0 and min(worlds) != world:
                raise ValueError(f"NovaPhy contact crosses worlds: {left}, {right}")
            if world == -1 and self.world_count == 1:
                world = 0
            a, b = self.shape_names.get((world, left)), self.shape_names.get((world, right))
            if a is None or b is None:
                raise ValueError(f"NovaPhy contact contains unmapped shapes in world {world}: {left}, {right}")
            pairs.append([world, a, b])
        return self.bodies(), pairs
