"""Compare frame samples for the fluid scenarios without an engine dependency."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import Sample, load_sample

SPECTRUM: Path = Path(__file__).with_name("data") / "spectral_Re1600_512.gdiag"


def _particle_state(folder: Path, index: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    sample = load_sample(folder, index)
    positions, velocities = sample["particle_q"], sample["particle_qd"]
    for array in (positions, velocities):
        if array.dtype != np.float32 or array.shape != (count, 3) or not np.isfinite(array).all():
            raise ValueError(f"invalid FP32 particle state at {folder}, frame {index}")
    return positions, velocities


def _relative_rms(actual: np.ndarray, expected: np.ndarray, weights: np.ndarray) -> tuple[float, float | None]:
    delta = actual - expected
    numerator = np.sum(weights * np.sum(delta * delta, axis=1, dtype=np.float32), dtype=np.float32)
    denominator = np.sum(weights * np.sum(expected * expected, axis=1, dtype=np.float32), dtype=np.float32)
    relative = float(np.sqrt(numerator) / np.sqrt(denominator)) if denominator > 0 else None
    return float(np.sqrt(numerator)), relative


def compare_particles(
    candidate: Path, reference: Path, frames: int, *, scales: dict[str, float] | None = None
) -> dict[str, Any]:
    """Compare every FP32 particle position and velocity at matching frames."""
    mass = load_sample(reference)["mass"]
    if mass.ndim != 1 or mass.dtype != np.float32 or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("invalid reference particle masses")
    if not np.array_equal(load_sample(candidate)["mass"], mass):
        raise ValueError("particle masses differ")
    weights = mass / np.sum(mass, dtype=np.float32)
    initial, initial_v = _particle_state(reference, 0, len(mass))
    actual_initial, actual_initial_v = _particle_state(candidate, 0, len(mass))
    if not np.array_equal(initial, actual_initial) or not np.array_equal(initial_v, actual_initial_v):
        raise ValueError("initial particle state differs")
    maximum, unequal = (
        dict.fromkeys(("particle_q", "particle_qd"), 0.0),
        dict.fromkeys(("particle_q", "particle_qd"), 0),
    )
    rms_max = dict.fromkeys(("displacement", "linear_velocity"), 0.0)
    vector_fields = {"particle_q": ("x_p", "position_m"), "particle_qd": ("x_v", "linear_velocity_mps")}
    peaks: dict[str, dict[str, Any]] = {}
    rows = []
    for index in range(frames + 1):
        pos, vel = _particle_state(candidate, index, len(mass))
        ref_pos, ref_vel = _particle_state(reference, index, len(mass))
        row: dict[str, Any] = {"frame": index}
        for name, info in (("particle_q", (pos, ref_pos)), ("particle_qd", (vel, ref_vel))):
            delta = np.abs(info[0] - info[1])
            value, count = float(delta.max()), int(np.count_nonzero(delta))
            maximum[name] = max(maximum[name], value)
            unequal[name] += count
            row[name] = {"max_abs": value, "unequal_components": count}
            if scales is not None:
                metric, field = vector_fields[name]
                squared_norm = np.sum(delta * delta, axis=1, dtype=np.float32)
                particle = int(np.argmax(squared_norm))
                distance = float(np.sqrt(squared_norm[particle]))
                error = distance / scales[field]
                if not math.isfinite(error):
                    raise ValueError(f"nonfinite particle error in {name} at frame {index}")
                row[name]["max_normalized_l2"] = error
                if metric not in peaks or error > peaks[metric]["error"]:
                    peaks[metric] = {"error": error, "absolute_l2": distance, "frame": index, "particle": particle}
        for name, a, b in (
            ("displacement", pos - actual_initial, ref_pos - initial),
            ("linear_velocity", vel, ref_vel),
        ):
            absolute, relative = _relative_rms(a, b, weights)
            row[name] = {"absolute_rms": absolute, "relative_rms": relative}
            if relative is not None:
                rms_max[name] = max(rms_max[name], relative)
        rows.append(row)
    normalized = {}
    if scales is not None:
        errors = {metric: peak["error"] for metric, peak in peaks.items()}
        normalized = {**errors, "x_F": max(errors.values()), "scales": scales, "peak_samples": peaks}
    return {
        **normalized,
        "particles": len(mass),
        "frames_inclusive": [0, frames],
        "worst_max_abs": maximum,
        "max_relative_rms": rms_max,
        "unequal_components_total": unequal,
        "exact_motion": not any(unequal.values()),
        "frames": rows,
    }


def _checked_metadata(a: Sample, b: Sample) -> None:
    for field in ("initial_state_sha256", "total_mass_kg", "diagnostic_h_m", "diagnostic_origin_m"):
        if not np.array_equal(a[field], b[field]):
            raise ValueError(f"single-box reference differs in {field}")


def _frame_time(sample: Sample, index: int, dt: float) -> None:
    if index != int(sample["frame"]) or not math.isclose(float(sample["time_end_s"]), index * dt, abs_tol=1e-9):
        raise ValueError(f"single-box diagnostic frame or sample time differs at {index}")


def compare_box(
    candidate: Path,
    reference: Path,
    frames: int,
    dt: float,
    *,
    volume: bool,
    pressure_grid: bool = False,
    scoring_metrics: bool = False,
) -> dict[str, Any]:
    a_model, b_model = load_sample(candidate), load_sample(reference)
    _checked_metadata(a_model, b_model)
    method_field = "volume_method" if volume else "pressure_method"
    if str(a_model[method_field]) != str(b_model[method_field]):
        raise ValueError(f"single-box diagnostic method differs in {method_field}")
    center = None
    if not volume:
        center = b_model["initial_cube_center_m"]
        if not np.allclose(a_model["initial_cube_center_m"], center, atol=1e-6, rtol=0.0):
            raise ValueError("single-box initial cube centers differ")

    rows = []
    for index in range(1, frames + 1):
        a, b = load_sample(candidate, index), load_sample(reference, index)
        _frame_time(a, index, dt)
        _frame_time(b, index, dt)
        row: dict[str, Any] = {"frame": index, "time_end_s": float(b["time_end_s"])}
        if volume:
            for sample in (a, b):
                if not np.isfinite(sample["volume_ratio"]) or int(sample["nonfinite_count"]) != 0:
                    raise ValueError(f"invalid volume sample at frame {index}")
            if not math.isclose(
                float(a["initial_effective_volume_m3"]),
                float(b["initial_effective_volume_m3"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError("single-box initial effective volumes differ")
            a_ratio, b_ratio = float(a["volume_ratio"]), float(b["volume_ratio"])
            row.update(
                volume_ratio_W=a_ratio, volume_ratio_N=b_ratio, abs_volume_ratio_difference=abs(a_ratio - b_ratio)
            )
            if pressure_grid:
                row.update(
                    effective_volume_W_m3=a_ratio * float(a["initial_effective_volume_m3"]),
                    effective_volume_N_m3=b_ratio * float(b["initial_effective_volume_m3"]),
                )
        else:
            pressures = []
            for sample in (a, b):
                pressure = sample["floor_pressure_pa"]
                if pressure.shape != (32, 32) or np.isinf(pressure).any():
                    raise ValueError(f"invalid pressure grid at frame {index}")
                pressures.append(float(np.nan_to_num(pressure, nan=0.0).mean()))
                if pressure_grid:
                    contact = sample["floor_contact_impulse_ns"]
                    correction = sample["floor_boundary_correction_impulse_ns"]
                    if any(grid.shape != (32, 32) or not np.isfinite(grid).all() for grid in (contact, correction)):
                        raise ValueError(f"invalid floor impulse grid at frame {index}")
                    if not np.allclose(contact + correction, np.nan_to_num(pressure) * (0.01 * dt), atol=1e-10):
                        raise ValueError(f"floor impulse and pressure observations differ at frame {index}")
            if pressure_grid:
                a_pressure, b_pressure = a["floor_pressure_pa"], b["floor_pressure_pa"]
                a_wet, b_wet = np.isfinite(a_pressure), np.isfinite(b_pressure)
                common = a_wet & b_wet
                if scoring_metrics:
                    actual_pressure = np.nan_to_num(a_pressure, nan=0.0)
                    expected_pressure = np.nan_to_num(b_pressure, nan=0.0)
                    pressure_error = float(np.abs(actual_pressure - expected_pressure).sum())
                    reference_pressure = float(np.abs(expected_pressure).sum())
                else:
                    pressure_error = float(np.abs(a_pressure[common] - b_pressure[common]).sum())
                    reference_pressure = float(np.abs(b_pressure[common]).sum())
                row.update(
                    floor_pressure_grid_error_sum_pa=pressure_error,
                    floor_pressure_grid_reference_sum_pa=reference_pressure,
                    floor_pressure_grid_relative_error=pressure_error / reference_pressure
                    if reference_pressure
                    else 0.0
                    if scoring_metrics and pressure_error == 0
                    else None,
                    common_wet_cells=int(common.sum()),
                    wet_mask_difference_cells=int(np.count_nonzero(a_wet ^ b_wet)),
                )
            a_center, b_center = a["cube_center_m"], b["cube_center_m"]
            if any(value.shape != (3,) or not np.isfinite(value).all() for value in (a_center, b_center)):
                raise ValueError(f"invalid cube center at frame {index}")
            absolute = float(np.linalg.norm(a_center - b_center))
            displacement = float(np.linalg.norm(b_center - center))
            row.update(
                mean_pressure_W_pa=pressures[0],
                mean_pressure_N_pa=pressures[1],
                absolute_pressure_error_pa=abs(pressures[0] - pressures[1]),
                cube_center_error_m=absolute,
                cube_relative_error=absolute / displacement
                if displacement > 1e-12
                else 0.0
                if scoring_metrics and absolute == 0
                else None,
            )
        rows.append(row)

    if volume:
        a_mean = math.fsum(row["volume_ratio_W"] for row in rows) / frames
        b_mean = math.fsum(row["volume_ratio_N"] for row in rows) / frames
        differences = [row["abs_volume_ratio_difference"] for row in rows]
        return {
            "candidate_mean_volume_loss_fraction": 1.0 - a_mean,
            "reference_mean_volume_loss_fraction": 1.0 - b_mean,
            "mean_abs_ratio_difference": math.fsum(differences) / frames,
            "max_abs_ratio_difference": max(differences),
            **(
                {
                    "volume_loss_error": max(max(0.0, 1.0 - row["volume_ratio_W"]) for row in rows),
                    "reference_volume_loss_max": max(max(0.0, 1.0 - row["volume_ratio_N"]) for row in rows),
                }
                if scoring_metrics
                else {}
            ),
            "frames": rows,
        }
    denominator = math.fsum(abs(row["mean_pressure_N_pa"]) for row in rows)
    old_pressure_error = (
        math.fsum(row["absolute_pressure_error_pa"] for row in rows) / denominator if denominator > 0 else None
    )
    pressure_result = {"floor_pressure_relative_error": old_pressure_error}
    if pressure_grid:
        error = math.fsum(row["floor_pressure_grid_error_sum_pa"] for row in rows)
        reference_load = math.fsum(row["floor_pressure_grid_reference_sum_pa"] for row in rows)
        pressure_result = {
            "floor_pressure_relative_error": error / reference_load if reference_load > 0 else None,
            "floor_pressure_mean_load_relative_error": old_pressure_error,
            "floor_pressure_comparison": "relative_l1_full_grid_dry_cells_zero"
            if scoring_metrics
            else "relative_l1_common_wet_cells",
        }
    relative = [row["cube_relative_error"] for row in rows if row["cube_relative_error"] is not None]
    return {
        **pressure_result,
        "cube_center_mean_relative_error": math.fsum(relative) / len(relative) if relative else None,
        **(
            {
                "pressure_error": max(row["floor_pressure_grid_relative_error"] for row in rows)
                if all(row.get("floor_pressure_grid_relative_error") is not None for row in rows)
                else None,
                "cube_state_error": max(row["cube_relative_error"] for row in rows)
                if all(row["cube_relative_error"] is not None for row in rows)
                else None,
                "scoring_error_reduction": "trajectory_maximum",
                "pressure_error_method": "relative_l1_full_grid_dry_cells_zero",
                "cube_state_error_method": "center_position_relative_to_Newton_displacement",
            }
            if scoring_metrics
            else {}
        ),
        "frames": rows,
    }


def compare_tgv(candidate: Path, frames: int, dt: float, spectrum: Path) -> dict[str, Any]:
    """Compare both curves at the prescribed times before averaging their errors."""
    energy = np.asarray([float(load_sample(candidate, i)["energy"]) for i in range(frames + 1)])
    if not np.isfinite(energy).all():
        raise ValueError("non-finite Taylor–Green energy")
    times = np.arange(frames) * dt
    reference = np.loadtxt(spectrum, usecols=(0, 1, 2))
    if not np.isfinite(reference).all() or not np.all(np.diff(reference[:, 0]) > 0):
        raise ValueError("invalid Taylor–Green reference curve")
    if times[0] < reference[0, 0] or times[-1] > reference[-1, 0] + 1e-10:
        raise ValueError("Taylor–Green reference does not cover the requested frames")
    dissipation = -np.gradient(energy, dt, edge_order=2 if frames >= 2 else 1)[:-1]
    expected_energy = np.interp(times, reference[:, 0], reference[:, 1])
    expected_dissipation = np.interp(times, reference[:, 0], reference[:, 2])

    def error(actual: np.ndarray, expected: np.ndarray) -> float:
        norm = float(np.linalg.norm(expected))
        if norm <= 0:
            raise ValueError("Taylor–Green reference norm is zero")
        return float(np.linalg.norm(actual - expected)) / norm

    energy_error = error(energy[:-1], expected_energy)
    dissipation_error = error(dissipation, expected_dissipation)
    return {
        "energy_relative_l2": energy_error,
        "dissipation_relative_l2": dissipation_error,
        "mean_relative_l2": (energy_error + dissipation_error) / 2.0,
        "frames": [
            {
                "time": float(t),
                "energy": float(e),
                "dissipation": float(d),
                "reference_energy": float(ref_e),
                "reference_dissipation": float(ref_d),
            }
            for t, e, d, ref_e, ref_d in zip(
                times, energy[:-1], dissipation, expected_energy, expected_dissipation, strict=True
            )
        ],
    }
