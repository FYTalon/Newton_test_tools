"""Compute hundred-thousand-DOF physics metrics from substep captures."""

from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .accuracy import _rotation
from .diagnostics import Sample, load_sample


def _totals(model: Sample, sample: Sample) -> dict[str, Any]:
    q, qd = (np.asarray(sample[key], dtype=np.float64) for key in ("body_q", "body_qd"))
    mass = model["mass"].astype(np.float64)
    if q.shape != (len(mass), 7) or qd.shape != (len(mass), 6):
        raise ValueError("substep rigid state has an unexpected shape")
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError("substep state contains a nonfinite value")
    rotation = _rotation(q[:, 3:])
    offset = np.einsum("nij,nj->ni", rotation, model["com_body"])
    position = q[:, :3] + offset
    omega = qd[:, 3:]
    velocity = qd[:, :3]
    if str(model["velocity_convention"]) == "origin_world":
        velocity = velocity + np.cross(omega, offset)
    center = np.sum(mass[:, None] * position, axis=0) / mass.sum()
    particle_momentum = mass[:, None] * velocity
    momentum = particle_momentum.sum(axis=0)
    spin = np.einsum("nij,njk,nlk,nl->ni", rotation, model["inertia_body"], rotation, omega, optimize=True)
    angular = np.sum(np.cross(position - center, particle_momentum) + spin, axis=0)
    kinetic = 0.5 * float(np.sum(velocity * particle_momentum) + np.sum(omega * spin))
    potential = -float(np.sum(mass * np.einsum("ni,i->n", position, model["gravity"])))
    return {
        "P": momentum,
        "H_com": angular,
        "K": kinetic,
        "U": potential,
        "E": kinetic + potential,
        "initial_momentum_norm_sum": float(np.linalg.norm(particle_momentum, axis=1).sum()),
    }


def _setup(folder: Path) -> tuple[Sample, dict[str, Any]]:
    model = load_sample(folder)
    mass = model["mass"]
    for key, shape in (
        ("mass", (len(mass),)),
        ("com_body", (len(mass), 3)),
        ("inertia_body", (len(mass), 3, 3)),
        ("gravity", (3,)),
    ):
        if model[key].shape != shape or not np.isfinite(model[key]).all():
            raise ValueError(f"invalid conservation model field {key}")
    if not len(mass) or np.any(mass <= 0) or str(model["velocity_convention"]) not in ("origin_world", "com_world"):
        raise ValueError("conservation requires dynamic bodies and a known velocity convention")
    return model, _totals(model, load_sample(folder, 0))


def _write_event(
    stream: TextIO,
    pair: tuple[int, int, int],
    event: tuple[int, float, float],
    rows: list[dict[str, Any]],
    end: int | None,
    scales: dict[str, float],
) -> None:
    start, depth, negative_force = event
    world, shape0, shape1 = pair
    result = {
        "world": world,
        "shape0": shape0,
        "shape1": shape1,
        "start_substep": start,
        "before_substep": start - 1,
        "after_substep": end,
        "status": "ended" if end is not None else "sustained_at_trajectory_end",
        "penetration_max_m": depth,
        "negative_normal_force_max_N": negative_force,
    }
    if end is not None:
        before, after = rows[start - 1], rows[end]
        delta_p = np.subtract(after["P"], before["P"])
        delta_e = after["E"] - before["E"]
        result.update(
            delta_P=delta_p.tolist(),
            delta_H_com=np.subtract(after["H_com"], before["H_com"]).tolist(),
            delta_E_J=delta_e,
            normalized_delta_P=float(np.linalg.norm(delta_p)) / scales["P_ref"],
            normalized_delta_E=delta_e / scales["E_ref"],
        )
    stream.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


def _observe_events(
    stream: TextIO,
    active: dict[tuple[int, int, int], tuple[int, float, float]],
    sample: Sample,
    rows: list[dict[str, Any]],
    scales: dict[str, float],
) -> None:
    distance = sample["contact_distance_m"]
    index = rows[-1]["step"]
    current = set()
    if len(distance):
        world, a, b = (sample[key] for key in ("contact_world", "shape0", "shape1"))
        if any(array.shape != distance.shape or np.any(array < 0) for array in (world, a, b)):
            raise ValueError(f"invalid contact identities at substep {index}")
        pairs = np.column_stack((world, np.minimum(a, b), np.maximum(a, b)))
        unique, indices = np.unique(pairs, axis=0, return_inverse=True)
        peak_depth, peak_negative = np.zeros(len(unique)), np.zeros(len(unique))
        np.maximum.at(peak_depth, indices, -distance)
        np.maximum.at(peak_negative, indices, -sample["normal_compression_N"])
        for triple, depth, negative in zip(unique, peak_depth, peak_negative, strict=True):
            key = tuple(int(value) for value in triple)
            current.add(key)
            start, old_depth, old_negative = active.get(key, (index, 0.0, 0.0))
            active[key] = (start, max(old_depth, float(depth)), max(old_negative, float(negative)))
    for key in sorted(active.keys() - current):
        _write_event(stream, key, active.pop(key), rows, index, scales)


def _summarize(
    folder: Path,
    model: Sample,
    initial: dict[str, Any],
    case: dict[str, Any],
    frames: int,
    scales: dict[str, float],
    event_stream: TextIO | None,
) -> dict[str, Any]:
    substeps = case["substeps"]
    dt = case["frame_dt_seconds"] / substeps
    peak_p = peak_e = peak_negative = 0.0
    peak_free_fall = None
    penetration_sum, penetrating_count = 0.0, 0
    free_fall = case["free_fall"]
    rows = []
    active_events: dict[tuple[int, int, int], tuple[int, float, float]] = {}
    for index in range(frames * substeps + 1):
        sample = load_sample(folder, index)
        totals = initial if index == 0 else _totals(model, sample)
        distance, normal_force = sample["contact_distance_m"], sample["normal_compression_N"]
        if (
            distance.ndim != 1
            or distance.shape != normal_force.shape
            or not all(np.isfinite(x).all() for x in (distance, normal_force))
        ):
            raise ValueError(f"invalid effective contacts at substep {index}")
        penetrating = distance[distance < 0]
        penetration_sum -= float(np.sum(penetrating, dtype=np.float64))
        penetrating_count += len(penetrating)
        negative = float(np.maximum(0.0, -normal_force).max(initial=0))
        peak_negative = max(peak_negative, negative)
        p = float(np.linalg.norm(totals["P"] - initial["P"])) / scales["P_ref"]
        e = abs(totals["E"] - initial["E"]) / scales["E_ref"]
        peak_p, peak_e = max(peak_p, p), max(peak_e, e)
        if len(distance):
            free_fall = False
        if free_fall and index:
            peak_free_fall = e if peak_free_fall is None else max(peak_free_fall, e)
        rows.append(
            {
                "step": index,
                "time_s": index * dt,
                "P": totals["P"].tolist(),
                "H_com": totals["H_com"].tolist(),
                "K": totals["K"],
                "U": totals["U"],
                "E": totals["E"],
                "R_P": p,
                "R_E": e,
                "free_fall": free_fall,
                "penetrating_contacts": len(penetrating),
                "negative_normal_peak_N": negative,
            }
        )
        if index and event_stream is not None:
            _observe_events(event_stream, active_events, sample, rows, scales)
    if event_stream is not None:
        for key, event in sorted(active_events.items()):
            _write_event(event_stream, key, event, rows, None, scales)
    return {
        "effective_dof": int(model["effective_dof"]),
        "frames_completed": frames,
        "substeps_completed": frames * substeps,
        "R_P_max": peak_p,
        "R_E_max": peak_e,
        "free_fall_R_E_max": peak_free_fall,
        "mean_penetration_over_length": penetration_sum / (penetrating_count * scales["L"])
        if penetrating_count
        else 0.0,
        "penetrating_contact_count": penetrating_count,
        "negative_normal_peak_over_force": peak_negative / scales["F_ref"],
        "substeps": rows,
    }


def compare_conservation(
    candidate: Path, reference: Path, case: dict[str, Any], frames: int, *, save_event_details: bool = False
) -> dict[str, Any]:
    baseline, initial = _setup(reference)
    actual, actual_initial = _setup(candidate)
    if int(actual["effective_dof"]) != int(baseline["effective_dof"]) or not np.array_equal(
        actual["gravity"], baseline["gravity"]
    ):
        raise ValueError("conservation workload or gravity differs")
    length, speed = case["length_m"], case["velocity_mps"]
    mass = float(baseline["mass"].astype(np.float64).sum())
    if not math.isclose(float(actual["mass"].astype(np.float64).sum()), mass, rel_tol=1e-6):
        raise ValueError("conservation total dynamic mass differs")
    gravity_force = mass * float(np.linalg.norm(baseline["gravity"]))
    if gravity_force <= 0:
        raise ValueError("conservation reference gravity must be nonzero")
    scales = {
        "L": length,
        "F_ref": gravity_force,
        "P_ref": max(initial["initial_momentum_norm_sum"], mass * speed),
        "E_ref": max(initial["K"], gravity_force * length, 0.5 * mass * speed**2),
    }
    with ExitStack() as stack:
        reports = []
        for folder, model, start in ((candidate, actual, actual_initial), (reference, baseline, initial)):
            stream = (
                stack.enter_context((folder / "events.jsonl").open("w", encoding="utf-8"))
                if save_event_details
                else None
            )
            reports.append(_summarize(folder, model, start, case, frames, scales, stream))
        a, b = reports
    metrics = ("R_P_max", "free_fall_R_E_max", "mean_penetration_over_length", "negative_normal_peak_over_force")
    ratios = {}
    for name in metrics:
        n, w = b[name], a[name]
        ratios[name] = w / n if n is not None and abs(n) > 0.001 and w is not None else None
    return {
        "reference_scales": scales,
        "minimum_effective_dof_met": a["effective_dof"] >= 100_000,
        "newton_denominator_floor": 0.001,
        "candidate": a,
        "reference": b,
        "ratios": ratios,
    }


def xpbd_contact_geometry(
    q: np.ndarray,
    shape_body: np.ndarray,
    shape0: np.ndarray,
    shape1: np.ndarray,
    point0: np.ndarray,
    point1: np.ndarray,
    normal: np.ndarray,
    margin0: np.ndarray,
    margin1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct pre-solve signed distances from body-local contact points."""
    points = []
    for ids, local in ((shape0, point0), (shape1, point1)):
        owners = shape_body[ids]
        world = local.astype(np.float64)
        moving = owners >= 0
        pose = q[owners[moving]].astype(np.float64)
        world[moving] = pose[:, :3] + np.einsum("nij,nj->ni", _rotation(pose[:, 3:]), world[moving])
        points.append(world)
    p0, p1 = points
    distance = np.einsum("ni,ni->n", normal, p1 - p0) - margin0 - margin1
    return distance, (p0 + p1) * 0.5


def pyramidal_forces(
    world: np.ndarray, dimension: np.ndarray, address: np.ndarray, friction: np.ndarray, constraint_force: np.ndarray
) -> np.ndarray:
    """Decode MuJoCo's pyramidal constraints into local contact wrench components."""
    valid = address[:, 0] >= 0
    result = np.zeros((len(address), 6), dtype=np.float64)
    if np.any((dimension < 1) | (dimension > 6)):
        raise ValueError("invalid MuJoCo contact dimension")
    normal_only = valid & (dimension == 1)
    result[normal_only, 0] = constraint_force[world[normal_only], address[normal_only, 0]]
    for axis in range(5):
        mask = valid & (dimension > axis + 1)
        if not np.any(mask):
            continue
        if 2 * axis + 1 >= address.shape[1]:
            raise ValueError("pyramidal contacts require more constraint addresses than the model provides")
        a, b = address[mask, 2 * axis], address[mask, 2 * axis + 1]
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("invalid pyramidal contact constraint address")
        fa, fb = constraint_force[world[mask], a], constraint_force[world[mask], b]
        result[mask, 0] += fa + fb
        result[mask, axis + 1] = (fa - fb) * friction[mask, axis]
    return result
