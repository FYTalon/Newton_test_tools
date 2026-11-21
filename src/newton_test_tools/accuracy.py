# SPDX-License-Identifier: Apache-2.0
"""Evaluate the rigid-body trajectory metrics in the public adapter protocol."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_QUANTITIES = (
    "force_N",
    "torque_com_Nm",
    "position_m",
    "linear_velocity_mps",
    "angular_velocity_radps",
)
_ERROR_KEYS = (
    "force_relative_max",
    "torque_relative_max",
    "position_relative_max",
    "linear_velocity_relative_max",
    "angular_velocity_relative_max",
)
_FLOOR_KEYS = (
    "force_floor_N",
    "torque_floor_Nm",
    "position_floor_m",
    "linear_velocity_floor_mps",
    "angular_velocity_floor_radps",
)
_SCALE_KEYS = (
    "force_scale_N",
    "torque_scale_Nm",
    "position_scale_m",
    "linear_velocity_scale_mps",
    "angular_velocity_scale_radps",
)


def _index(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    indexed = {(row["world"], row["name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate identity in trajectory")
    return indexed


def _aligned_bodies(candidate: dict, reference: dict) -> tuple[list, list]:
    ref, actual = _index(reference["model"]), _index(candidate["model"])
    if ref.keys() != actual.keys():
        raise ValueError("rigid-body identities differ")
    names = sorted(ref)
    for key in names:
        a, b = actual[key], ref[key]
        if a["flags"] != b["flags"]:
            raise ValueError(f"mobility of body {key!r} differs")
        for field, tolerance in (("mass", 1e-8), ("com_body", 2e-7), ("inv_mass", 1e-8)):
            if field == "inv_mass" and b["flags"] == 2:
                continue
            if not np.allclose(a[field], b[field], rtol=1e-5, atol=tolerance):
                raise ValueError(f"{field} of body {key!r} differs")
        ref_inertia = np.asarray(b["inertia_body"], dtype=np.float64)
        diff = np.asarray(a["inertia_body"], dtype=np.float64) - ref_inertia
        if np.linalg.norm(diff) > 1e-8 + 1e-5 * np.linalg.norm(ref_inertia):
            raise ValueError(f"center-of-mass inertia of body {key!r} differs")
    return [actual[key] for key in names], [ref[key] for key in names]


def _frames(trace: dict, model: list) -> tuple[np.ndarray, np.ndarray]:
    ids = [(row["world"], row["name"]) for row in model]
    required = set(ids)
    positions, velocities = [], []
    for number, frame in enumerate(sorted(trace["states"], key=lambda row: row["frame"])):
        if frame["frame"] != number:
            raise ValueError("trajectory frames must start at zero and be consecutive")
        by_id = _index(frame["bodies"])
        if by_id.keys() != required:
            raise ValueError(f"rigid-body identities differ at frame {number}")
        positions.append([by_id[key]["q"] for key in ids])
        velocities.append([by_id[key]["qd"] for key in ids])
    q, qd = np.asarray(positions, dtype=np.float64), np.asarray(velocities, dtype=np.float64)
    shape = (trace["num_frames"] + 1, len(ids))
    if q.shape != (*shape, 7) or qd.shape != (*shape, 6) or not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError("trajectory is missing pose or velocity data or contains a non-finite value")
    return q, qd


def _com_motion(q: np.ndarray, qd: np.ndarray, centers: np.ndarray, origin: bool) -> tuple[np.ndarray, ...]:
    positions = np.empty_like(q[:, :3])
    velocities = np.empty_like(positions)
    orientations = np.empty_like(q[:, 3:])
    for body, (pose, twist, center) in enumerate(zip(q, qd, centers, strict=True)):
        unit = pose[3:] / np.linalg.norm(pose[3:])
        twice_cross = 2 * np.cross(unit[:3], center)
        offset = center + unit[3] * twice_cross + np.cross(unit[:3], twice_cross)
        positions[body] = pose[:3] + offset
        velocities[body] = twist[:3] + np.cross(twist[3:], offset) if origin else twist[:3]
        orientations[body] = unit
    return positions, orientations, velocities, qd[:, 3:]


def _initial_agreement(candidate: tuple, reference: tuple) -> dict:
    labels = ("com_position_m", "orientation_quaternion", "com_velocity_m_s", "angular_velocity_rad_s")
    errors = {}
    for index, (label, actual, baseline) in enumerate(zip(labels, candidate, reference, strict=True)):
        values = [np.linalg.norm(a - b) for a, b in zip(actual, baseline, strict=True)]
        if index == 1:
            values = [min(value, np.linalg.norm(a + b)) for value, a, b in zip(values, actual, baseline, strict=True)]
        errors[label] = float(max(values))
    tolerance = 1e-5
    if any(not math.isfinite(error) or error > tolerance for error in errors.values()):
        raise ValueError(f"initial center-of-mass states differ: {errors}")
    return {
        "status": "within_tolerance",
        "absolute_tolerance": tolerance,
        "position_absolute_tolerance_m": tolerance,
        "max_error": errors,
    }


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(np.abs(norm - 1) > 1e-3):
        raise ValueError("expected a unit xyzw quaternion in every body pose")
    unit = quaternion / norm
    vector, scalar = unit[..., :3], unit[..., 3]
    matrix = np.empty((*norm.shape[:-1], 3, 3), dtype=quaternion.dtype)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        a, b = vector[..., j], vector[..., k]
        matrix[..., i, i] = 1 - 2 * (a * a + b * b)
        dot = vector[..., i] * a
        cross = b * scalar
        matrix[..., i, j] = 2 * (dot - cross)
        matrix[..., j, i] = 2 * (dot + cross)
    return matrix


def _measure(trace: dict, model: list) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    poses, twists = _frames(trace, model)
    q, qd = poses.astype(np.float32), twists.astype(np.float32)
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError("pose or velocity cannot be represented in float32")
    mass = np.asarray([row["mass"] for row in model], dtype=np.float32)
    inv_mass = np.asarray([row["inv_mass"] for row in model], dtype=np.float32)
    flags = np.asarray([row["flags"] for row in model])
    model_centers = np.asarray([row["com_body"] for row in model], dtype=np.float64)
    centers = model_centers.astype(np.float32)
    inertia = np.asarray([row["inertia_body"] for row in model], dtype=np.float32)
    count = len(model)
    if mass.shape != (count,) or inv_mass.shape != (count,) or centers.shape != (count, 3):
        raise ValueError("invalid rigid-body mass or center-of-mass array")
    if inertia.shape != (count, 3, 3) or any(not np.isfinite(a).all() for a in (mass, inv_mass, centers, inertia)):
        raise ValueError("invalid rigid-body inertia or non-finite model data")
    if np.any(mass < 0) or np.any(inv_mass < 0) or flags.dtype.kind not in "iu" or not np.isin(flags, (0, 1, 2)).all():
        raise ValueError("invalid rigid-body mass, inverse mass, or mobility flag")
    dynamic = (flags == 1) & (mass > 0) & (inv_mass > 0)
    tensors = inertia[dynamic]
    transpose = tensors.swapaxes(-2, -1)
    symmetric = (tensors + transpose) * np.float32(0.5)
    if np.any(np.linalg.norm(tensors - transpose, axis=(1, 2)) > 1e-7 * np.linalg.norm(symmetric, axis=(1, 2)) + 1e-12):
        raise ValueError("dynamic body inertia must be symmetric")
    if np.any(np.linalg.eigvalsh(symmetric) < -1e-10):
        raise ValueError("dynamic body inertia must be positive semidefinite")

    convention = trace["velocity_convention"]
    if convention not in ("origin_world", "com_world"):
        raise ValueError("unknown linear velocity convention")
    origin = convention == "origin_world"
    initial = _com_motion(poses[0], twists[0], model_centers, origin)

    rotation = _rotation(q[..., 3:])
    selected = rotation[:, dynamic]
    world_inertia = selected @ symmetric @ selected.swapaxes(-2, -1)
    spin = qd[:, dynamic, 3:]
    moment = np.einsum("...ij,...j->...i", world_inertia, spin)
    linear = qd[:, dynamic, :3]
    if origin:
        offset = np.einsum("...ij,...j->...i", selected, centers[dynamic])
        linear = linear + np.cross(spin, offset)
    dt = np.float32(trace["frame_dt_seconds"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("frame interval is outside the positive float32 range")
    force = mass[dynamic, None] * np.diff(linear, axis=0) / dt
    torque = np.diff(moment, axis=0) / dt

    world_pose, _, com_velocity, _ = zip(
        *(_com_motion(pose, twist, centers, origin) for pose, twist in zip(q, qd, strict=True)),
        strict=True,
    )
    metrics = force, torque, np.stack(world_pose)[:, dynamic], np.stack(com_velocity)[:, dynamic], spin
    return dynamic, initial, metrics


def jaccard_error(actual: list[list[Any]], reference: list[list[Any]]) -> float:
    actual_keys, reference_keys = set(map(tuple, actual)), set(map(tuple, reference))
    size = len(actual_keys | reference_keys)
    return float(np.float32(len(actual_keys ^ reference_keys)) / np.float32(size)) if size else 0.0


def _frame_errors(errors: dict[str, np.ndarray], frame: int) -> dict:
    kinematics = {key: float(errors[key][frame]) for key in _ERROR_KEYS[2:]}
    return {**kinematics, "x_K": max(kinematics.values())}


def compare_accuracy(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    force_floor_N: float,
    torque_floor_Nm: float,
    position_floor_m: float,
    linear_velocity_floor_mps: float,
    angular_velocity_floor_radps: float,
) -> dict[str, Any]:
    """Compare complete traces using the reference's per-body peak scales."""

    floors = (force_floor_N, torque_floor_Nm, position_floor_m, linear_velocity_floor_mps, angular_velocity_floor_radps)
    if any(not math.isfinite(value) or value <= 0 for value in floors):
        raise ValueError("all normalization floors must be finite and positive")
    for key in ("case", "solver", "num_frames"):
        if candidate[key] != reference[key]:
            raise ValueError(f"trajectory {key} differs")
    if not math.isclose(candidate["frame_dt_seconds"], reference["frame_dt_seconds"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("frame intervals differ")

    models = _aligned_bodies(candidate, reference)

    def shapes(trace: dict) -> dict:
        return {key: (row["body"], row["type"]) for key, row in _index(trace["shapes"]).items()}

    reference_shapes = shapes(reference)
    if shapes(candidate) != reference_shapes:
        raise ValueError("collision shape identities, owners, or types differ")
    (dynamic_a, initial_a, series_a), (dynamic_r, initial_r, series_r) = (
        _measure(trace, model) for trace, model in zip((candidate, reference), models, strict=True)
    )
    if not dynamic_r.any() or not np.array_equal(dynamic_a, dynamic_r):
        raise ValueError("comparison requires the same nonempty set of dynamic bodies")
    initial_alignment = _initial_agreement(initial_a, initial_r)

    scales = {}
    errors = {}
    for label, actual, baseline, floor, error_key in zip(
        _QUANTITIES, series_a, series_r, floors, _ERROR_KEYS, strict=True
    ):
        magnitude = baseline - baseline[0] if label == "position_m" else baseline
        peak = np.nan_to_num(np.linalg.norm(magnitude, axis=-1), nan=0.0).max(axis=0)
        scale = np.maximum(np.float32(floor), peak)
        scales[label] = scale
        errors[error_key] = (np.linalg.norm(actual - baseline, axis=-1) / scale).max(axis=1)

    initial = {"frame": 0, **_frame_errors(errors, 0)}
    frames = []
    for frame in range(1, reference["num_frames"] + 1):
        f, t = (float(errors[key][frame - 1]) for key in _ERROR_KEYS[:2])
        frames.append(
            {
                "frame": frame,
                "x_C": jaccard_error(candidate["pairs"][str(frame)], reference["pairs"][str(frame)]),
                "force_relative_max": f,
                "torque_relative_max": t,
                "x_D": max(f, t),
                **_frame_errors(errors, frame),
            }
        )

    bodies = []
    dynamic_index = 0
    for body, dynamic in zip(models[1], dynamic_r, strict=True):
        bodies.append(
            {
                "name": body["name"],
                "world": body["world"],
                "dynamic": bool(dynamic),
                **{
                    key: float(scales[label][dynamic_index]) if dynamic else None
                    for label, key in zip(_QUANTITIES, _SCALE_KEYS, strict=True)
                },
            }
        )
        dynamic_index += int(dynamic)

    normalizer = dict(zip(_FLOOR_KEYS, floors, strict=True))
    return {
        "format": "sim-bench-accuracy-v2",
        "case": reference["case"],
        "solver": reference["solver"],
        "x_C": max(row["x_C"] for row in frames),
        "x_D": max(row["x_D"] for row in frames),
        "x_K": max(row["x_K"] for row in [initial, *frames]),
        "frames": frames,
        "kinematic_initial": initial,
        "normalization": {"calculation_dtype": "float32", **normalizer, "bodies": bodies},
        "initial_alignment": initial_alignment,
        "shape_identity": {
            "status": "world_name_owner_type_checked",
            "shape_count": len(reference_shapes),
            "geometry_parameters_checked": False,
        },
    }
