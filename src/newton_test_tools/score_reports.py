"""Read benchmark reports into the task scoring interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import read_json

CASE_SCENES = {
    "basic_shapes": "1-1",
    "basic_conveyor": "1-2",
    "nut_bolt_hydro": "1-3",
    "basic_shapes_100k": "1-4",
    "basic_pendulum": "1-5",
    "robot_g1": "1-6",
    "robot_anymal_c_walk": "1-7",
    "anymal_c_walk_100k": "1-8",
    "implicit_mpm_dam_break": "1-9",
    "implicit_mpm_water_dam_single_box": "1-10",
    "fluid_grid_mpm_single_box": "1-10",
    "fluid_grid_taylor_green": "1-11",
}


def _formal_protocol(row: dict, mode: str, scene_id: str) -> bool:
    protocol = row.get("protocol", {}).get(mode, {})
    frames = 2000 if scene_id == "1-11" else 600 if scene_id in {"1-9", "1-10"} else 500
    repetitions = 3 if mode == "performance" or scene_id in {"1-4", "1-8"} else 1
    deterministic = mode == "accuracy" and scene_id not in {"1-4", "1-8", "1-9", "1-11"}
    if (
        protocol.get("num_frames") != frames
        or protocol.get("repetitions") != repetitions
        or protocol.get("deterministic") is not deterministic
    ):
        return False
    if scene_id not in {"1-9", "1-10", "1-11"} and protocol.get("graph") is not (mode == "performance"):
        return False
    return True


def _coupled_trials(row: dict, *, volume: bool) -> list[dict[str, Any]]:
    accuracy = row.get("accuracy", {})
    trials = accuracy.get("trials") or ([accuracy] if accuracy else [])
    result = []
    for number, trial in enumerate(trials, 1):
        frames = trial.get("frames", [])
        summary: dict[str, Any] = {"run": number, "frame_count": len(frames)}
        if volume:
            summary["max_abs_volume_ratio_difference"] = trial.get("max_abs_ratio_difference")
            summary["candidate_mean_volume_loss_fraction"] = trial.get("candidate_mean_volume_loss_fraction")
            summary["reference_mean_volume_loss_fraction"] = trial.get("reference_mean_volume_loss_fraction")
        else:
            relative = [
                frame["cube_relative_error"] for frame in frames if frame.get("cube_relative_error") is not None
            ]
            pressure = [
                frame["floor_pressure_grid_relative_error"]
                for frame in frames
                if frame.get("floor_pressure_grid_relative_error") is not None
            ]
            summary.update(
                max_cube_center_relative_error=max(relative) if relative else None,
                missing_cube_error_frames=len(frames) - len(relative),
                max_floor_pressure_grid_relative_error=max(pressure) if pressure else None,
                missing_pressure_error_frames=len(frames) - len(pressure),
                wet_mask_difference_frames=sum(bool(frame.get("wet_mask_difference_cells")) for frame in frames),
            )
        result.append(summary)
    return result


def read_team_reports(team_id: str, directories: list[Path]) -> dict[str, Any]:
    """Extract available metrics while keeping source reports and trial records."""
    if not team_id or not directories:
        raise ValueError("a team needs an identifier and at least one report directory")
    cases: dict[str, dict] = {}
    sources = []
    for directory in directories:
        directory = directory.expanduser().resolve()
        report = read_json(directory / "report.json")
        if report.get("format") not in {"sim-bench-comparison-v2", "sim-bench-comparison-v3"}:
            raise ValueError(f"unsupported comparison format: {directory}")
        sources.append(str(directory))
        for row in report["cases"]:
            case_id = row["case"]
            if case_id not in CASE_SCENES:
                raise ValueError(f"unknown benchmark case: {case_id}")
            if case_id in cases:
                raise ValueError(f"duplicate benchmark case for {team_id}: {case_id}")
            cases[case_id] = {**row, "_source_report": str(directory / "report.json")}
    scenes: dict[str, dict] = {}
    for case_id, row in cases.items():
        scene_id = CASE_SCENES[case_id]
        scene = scenes.setdefault(scene_id, {"source_cases": {}})
        scene["source_cases"][case_id] = {
            "report": row["_source_report"],
            "protocol": row.get("protocol", {}),
            "performance_trials": {
                role: details.get("trials", [details] if "metrics" in details else [])
                for role, details in row.get("performance", {}).get("engines", {}).items()
            },
            "performance_reasons": row.get("performance", {}).get("reasons", []),
            "memory_reasons": row.get("performance", {}).get("gpu_memory_reasons", []),
            "accuracy_trial_count": len(row.get("accuracy", {}).get("trials", [])) or int(bool(row.get("accuracy"))),
        }
        if scene_id == "1-10":
            continue
        performance = row.get("performance", {})
        scene["fps_ratio"] = performance.get("fps_ratio")
        scene["gpu_peak_ratio"] = performance.get("gpu_peak_ratio")
        accuracy = row.get("accuracy", {})
        if scene_id in {"1-4", "1-8"}:
            candidate = accuracy.get("candidate", {})
            if accuracy.get("trial_means"):
                means = accuracy["trial_means"]
                candidate = {**candidate, **means.get("candidate", {})}
                reference = {**accuracy.get("reference", {}), **means.get("reference", {})}
            else:
                reference = accuracy.get("reference", {})
            scene.update(
                effective_dof=candidate.get("effective_dof"),
                completed_frames=candidate.get("frames_completed"),
                completed_substeps=candidate.get("substeps_completed"),
                accuracy={"candidate": candidate, "reference": reference},
            )
            trials = accuracy.get("trials", [])
            if trials:
                for target, field in (
                    ("effective_dof", "effective_dof"),
                    ("completed_frames", "frames_completed"),
                    ("completed_substeps", "substeps_completed"),
                ):
                    values = [trial.get("candidate", {}).get(field) for trial in trials]
                    scene[target] = min(values) if all(isinstance(value, int) for value in values) else None
        elif scene_id == "1-9":
            scene["accuracy"] = {"x_F": accuracy.get("x_F")}
        elif scene_id == "1-11":
            scene["accuracy"] = {
                "energy_relative_l2": accuracy.get("energy_relative_l2"),
                "dissipation_relative_l2": accuracy.get("dissipation_relative_l2"),
            }
        else:
            scene["accuracy"] = {key: accuracy.get(key) for key in ("x_C", "x_D", "x_K")}
        pending = []
        if not _formal_protocol(row, "performance", scene_id):
            pending.extend(("performance", "memory"))
        if not _formal_protocol(row, "accuracy", scene_id):
            pending.append("accuracy")
        scene["pending_observations"] = pending
    coupled = scenes.get("1-10")
    if coupled is not None:
        version_1 = cases.get("implicit_mpm_water_dam_single_box", {})
        version_2 = cases.get("fluid_grid_mpm_single_box", {})
        coupled["source_cases"].get("implicit_mpm_water_dam_single_box", {}).update(
            accuracy_trials=_coupled_trials(version_1, volume=False)
        )
        coupled["source_cases"].get("fluid_grid_mpm_single_box", {}).update(
            accuracy_trials=_coupled_trials(version_2, volume=True)
        )
        coupled["fps_ratio"] = version_2.get("coupled_performance", {}).get("fps_ratio")
        coupled["gpu_peak_ratio"] = version_2.get("coupled_performance", {}).get("gpu_peak_ratio")
        coupled["accuracy"] = {
            "pressure_error": version_1.get("accuracy", {}).get("pressure_error"),
            "cube_state_error": version_1.get("accuracy", {}).get("cube_state_error"),
            "volume_loss_error": version_2.get("accuracy", {}).get("volume_loss_error"),
        }
        coupled["pending_observations"] = []
        if not _formal_protocol(version_1, "accuracy", "1-10") or not _formal_protocol(version_2, "accuracy", "1-10"):
            coupled["pending_observations"].append("accuracy")
        if not _formal_protocol(version_1, "performance", "1-10") or not _formal_protocol(
            version_2, "performance", "1-10"
        ):
            coupled["pending_observations"].extend(("performance", "memory"))
    for scene_id in set(CASE_SCENES.values()) - scenes.keys():
        scenes[scene_id] = {
            "awaiting_results": True,
            "pending_observations": ["performance", "accuracy", "memory"],
        }
    return {"id": team_id, "scenes": scenes, "sources": sources}
