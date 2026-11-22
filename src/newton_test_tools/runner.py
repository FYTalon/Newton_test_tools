"""Run independent adapters and recompute comparison reports from their files."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from .accuracy import compare_accuracy
from .conservation import compare_conservation
from .extended_accuracy import SPECTRUM, compare_box, compare_particles, compare_tgv
from .gpu import sample_gpu_memory, select_timed_memory
from .monitor import performance_metrics
from .protocol import (
    IDENTITY_FORMAT,
    REQUEST_FORMAT,
    RESULT_FORMAT,
    case_protocol,
    check_runtime,
    load_suite,
    metric_kind,
    particle_error_scales,
    positive_int,
    read_json,
    reference_case,
    validate_trace,
    write_json,
)

_COMPARISON_FORMAT: str = "sim-bench-comparison-v2"
_REPEATABILITY_IDENTITY: tuple[str, ...] = (
    "device_uuid",
    "engine_version",
    "checkout_commit",
    "example_sha256",
    "manifest_sha256",
)
_PHYSICAL_COUNTS: tuple[str, ...] = (
    "world_count",
    "body_count",
    "joint_count",
    "joint_dof_count",
    "particle_count",
    "collidable_shape_count",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "hashes.json"
    }


def compare_trajectories(
    case: dict[str, Any], frames: int, candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    validate_trace(reference, case, frames)
    validate_trace(candidate, case, frames)
    floors = case["normalization_floor"]
    return compare_accuracy(
        candidate,
        reference,
        force_floor_N=floors["force_N"],
        torque_floor_Nm=floors["torque_com_Nm"],
        position_floor_m=floors["position_m"],
        linear_velocity_floor_mps=floors["linear_velocity_mps"],
        angular_velocity_floor_radps=floors["angular_velocity_radps"],
    )


def _read_report(directory: Path, case: dict[str, Any], mode: str, protocol: dict[str, Any]) -> dict[str, Any]:
    result = read_json(directory / "report.json")
    if result["format"] != RESULT_FORMAT or result["case"] != case["id"] or result["mode"] != mode:
        raise ValueError(f"adapter result does not match the request in {directory}")
    if not isinstance(result["engine"], str) or not result["engine"].strip():
        raise ValueError(f"adapter did not identify its engine in {directory}")
    check_runtime({"case": case, "mode": mode, "protocol": protocol}, result["runtime"])
    return result


def _mode_directory(root: Path, case: dict[str, Any], mode: str) -> Path:
    return root / "cases" / case["id"] / mode


def _roles(case: dict[str, Any]) -> tuple[str, ...]:
    return ("candidate",) if metric_kind(case) == "tgv" else ("reference", "candidate")


def _trial_directory(root: Path, case: dict[str, Any], mode: str, role: str, number: int, repetitions: int) -> Path:
    trial_case = reference_case(case) if role == "reference" else case
    path = _mode_directory(root, trial_case, mode) / role
    return path / f"run-{number}" if mode == "accuracy" or repetitions > 1 else path


def _read_accuracy(root: Path, case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    captures = {}
    sides = {}
    for role in ("reference", "candidate"):
        repetition_hashes: list[str] = []
        for i in range(1, protocol["repetitions"] + 1):
            directory = root / role / f"run-{i}"
            report = _read_report(directory, case, "accuracy", protocol)
            path = directory / "trajectory.json"
            capture = read_json(path)
            validate_trace(capture, case, protocol["num_frames"])
            if report["metrics"]["completed_frames"] != protocol["num_frames"]:
                raise ValueError(f"adapter did not complete the declared frames in {directory}")
            repetition_hashes.append(_sha256(path))
            if i == 1:
                sides[role] = report
                captures[role] = capture
            elif report["engine"] != sides[role]["engine"] or any(
                report["runtime"].get(field) != sides[role]["runtime"].get(field) for field in _REPEATABILITY_IDENTITY
            ):
                raise ValueError(f"engine or runtime changed between repetitions in {directory}")
        sides[role]["repeatability"] = {
            "runs": len(repetition_hashes),
            "identical_trajectory": len(set(repetition_hashes)) == 1 if len(repetition_hashes) > 1 else None,
        }
    if sides["reference"]["runtime"]["device_uuid"] != sides["candidate"]["runtime"]["device_uuid"]:
        raise ValueError("accuracy results came from different devices")
    result = compare_trajectories(case, protocol["num_frames"], captures["candidate"], captures["reference"])
    return {**result, "engines": sides}


def _read_diagnostic_accuracy(
    root: Path, case: dict[str, Any], protocol: dict[str, Any], *, save_event_details: bool = False
) -> dict[str, Any]:
    kind = metric_kind(case)
    frames, repetitions = protocol["num_frames"], protocol["repetitions"]
    sides = {}
    expected = frames * (case["substeps"] if kind == "conservation" else 1) + 1
    gpu_seen = None
    for role in _roles(case):
        runs = []
        for number in range(1, repetitions + 1):
            directory = _trial_directory(root, case, "accuracy", role, number, repetitions)
            trial_case = reference_case(case) if role == "reference" else case
            report = _read_report(directory, trial_case, "accuracy", protocol)
            count = len(list((directory / "samples").glob("*.npz")))
            if report["metrics"]["completed_frames"] != frames or count != expected:
                raise ValueError(f"incomplete {kind} accuracy capture in {directory}")
            device = report["runtime"]["device_uuid"]
            if gpu_seen is not None and device != gpu_seen:
                raise ValueError("accuracy results came from different devices")
            gpu_seen = device
            runs.append(report)
        sides[role] = runs
    dt = case["frame_dt_seconds"]
    coupled = case.get("coupled_summary", False)
    count = repetitions if case.get("paired_passes") or coupled else 1
    results = []
    for number in range(1, count + 1):
        candidate = _trial_directory(root, case, "accuracy", "candidate", number, repetitions)
        if kind == "tgv":
            spectrum = _mode_directory(root, case, "accuracy") / "spectrum.gdiag"
            results.append(compare_tgv(candidate, frames, dt, spectrum))
            continue
        reference = _trial_directory(root, case, "accuracy", "reference", number, repetitions)
        if kind == "conservation":
            result = compare_conservation(candidate, reference, case, frames, save_event_details=save_event_details)
        elif kind == "particles":
            result = compare_particles(candidate, reference, frames, scales=particle_error_scales(case))
        else:
            result = compare_box(candidate, reference, frames, dt, volume=kind == "volume", pressure_grid=coupled)
        results.append(result)
    summary = 0
    summary_field = {"particles": "x_F", "tgv": "mean_relative_l2"}.get(kind)
    if summary_field is not None and summary_field in results[0]:
        summary = max(range(len(results)), key=lambda index: results[index][summary_field])
    trials = {"trials": results, "summary_trial": summary + 1} if count > 1 else {}
    return {"metric": kind, **results[summary], **trials, "engines": sides}


def _read_performance_trial(directory: Path, case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    report = _read_report(directory, case, "performance", protocol)
    counters = report["metrics"]
    metrics = performance_metrics(
        {"case": case, "protocol": protocol}, counters["timed_frames"], counters["elapsed_seconds"]
    )
    memory = read_json(directory / "gpu-memory.json")
    positive_int(memory["peak_bytes"], "peak_bytes")
    positive_int(memory["sample_count"], "sample_count")
    if memory["sample_interval_ms"] != protocol["memory_sample_interval_ms"]:
        raise ValueError("GPU memory polling interval differs from the requested interval")
    for field in _PHYSICAL_COUNTS:
        positive_int(report["runtime"]["model_counts"][field], field, minimum=0)
    return {**report, "metrics": metrics, "gpu_memory": memory}


def _performance_means(side: dict[str, Any]) -> tuple[float, float]:
    if "mean" in side:
        return side["mean"]["fps"], side["mean"]["gpu_peak_bytes"]
    return side["metrics"]["fps"], side["gpu_memory"]["peak_bytes"]


def _read_performance(
    root: Path,
    case: dict[str, Any],
    protocol: dict[str, Any],
    *,
    legacy_fps: bool = False,
    legacy_frame_summary: bool = False,
) -> dict[str, Any]:
    sides = {}
    reasons: list[str] = []
    repetitions = protocol.get("repetitions", 1)
    first_trials = {}
    kind = metric_kind(case)
    for role in _roles(case):
        trial_case = reference_case(case) if role == "reference" else case
        trials = [
            _read_performance_trial(
                _trial_directory(root, case, "performance", role, i, repetitions), trial_case, protocol
            )
            for i in range(1, repetitions + 1)
        ]
        first = trials[0]
        first_trials[role] = first
        for i, trial in enumerate(trials, 1):
            runtime, memory = trial["runtime"], trial["gpu_memory"]
            label = role if repetitions == 1 else f"{role} trial {i}"
            if runtime["device_uuid"] != memory["device_uuid"]:
                reasons.append(f"{label} used a different GPU from the one monitored")
            if runtime["model_counts"]["world_count"] != case["world_count"]:
                reasons.append(f"{label} model world count differs from the case")
            if i > 1 and (
                trial["engine"] != first["engine"]
                or any(runtime.get(field) != first["runtime"].get(field) for field in _REPEATABILITY_IDENTITY)
                or any(
                    runtime["model_counts"][field] != first["runtime"]["model_counts"][field]
                    for field in _PHYSICAL_COUNTS
                )
                or any(memory[field] != first["gpu_memory"][field] for field in ("device_uuid", "driver_version"))
            ):
                reasons.append(f"{label} engine, GPU or workload differs from its first trial")
        if repetitions == 1:
            sides[role] = first
        else:
            mean = {"gpu_peak_bytes": fmean(trial["gpu_memory"]["peak_bytes"] for trial in trials)}
            if legacy_frame_summary:
                elapsed = fmean(trial["metrics"]["elapsed_seconds"] for trial in trials)
                mean.update(elapsed_seconds=elapsed, fps=protocol["num_frames"] / elapsed)
            elif legacy_fps:
                mean["fps"] = fmean(trial["metrics"]["fps"] for trial in trials)
            else:
                frames = sum(trial["metrics"]["timed_frames"] for trial in trials)
                elapsed = sum(trial["metrics"]["elapsed_seconds"] for trial in trials)
                mean["fps"] = frames / elapsed
            sides[role] = {
                "engine": first["engine"],
                "mean": mean,
                "trials": trials,
            }
    candidate_fps, candidate_peak = _performance_means(sides["candidate"])
    if kind == "tgv":
        reference_fps, reference_peak = case["baseline"]["fps"], case["baseline"]["gpu_peak_bytes"]
        sides["reference"] = {
            "engine": "fixed baseline",
            "mean": {"fps": reference_fps, "gpu_peak_bytes": reference_peak},
        }
    else:
        a, b = first_trials["candidate"], first_trials["reference"]
        if (a["gpu_memory"]["device_uuid"], a["gpu_memory"]["driver_version"]) != (
            b["gpu_memory"]["device_uuid"],
            b["gpu_memory"]["driver_version"],
        ):
            reasons.append("GPU or driver differs between the two runs")
        fields = _PHYSICAL_COUNTS if kind in ("rigid", "conservation") else ("world_count", "particle_count")
        for field in fields:
            if a["runtime"]["model_counts"][field] != b["runtime"]["model_counts"][field]:
                reasons.append(f"runtime model {field} differs")
        reference_fps, reference_peak = _performance_means(sides["reference"])
    return {
        "case": case["id"],
        "engines": sides,
        "fps_ratio": None if reasons else candidate_fps / reference_fps,
        "gpu_peak_ratio": None if reasons else candidate_peak / reference_peak,
        "reasons": reasons,
    }


def _read_comparison(
    root: Path, plan: dict[str, Any], *, save_event_details: bool = False, legacy_frame_summary: bool = False
) -> dict[str, Any]:
    rows = []
    legacy_fps = plan.get("comparison_version", 1) == 1
    for case in plan["cases"]:
        row: dict[str, Any] = {"case": case["id"]}
        if not legacy_fps:
            row["protocol"] = {mode: case_protocol(plan, case, mode) for mode in plan["modes"]}
        for mode in plan["modes"]:
            protocol = case_protocol(plan, case, mode)
            if mode == "performance":
                row[mode] = _read_performance(
                    root,
                    case,
                    protocol,
                    legacy_fps=legacy_fps,
                    legacy_frame_summary=legacy_frame_summary,
                )
            elif metric_kind(case) == "rigid":
                row[mode] = _read_accuracy(_mode_directory(root, case, mode), case, protocol)
            else:
                row[mode] = _read_diagnostic_accuracy(root, case, protocol, save_event_details=save_event_details)
        rows.append(row)
    by_case = {row["case"]: row for row in rows}
    for case in plan["cases"]:
        if metric_kind(case) != "volume" or not case.get("coupled_summary"):
            continue
        second = by_case[case["id"]]
        first = by_case.get(reference_case(case)["id"], {})
        if "performance" not in first or "performance" not in second:
            continue
        versions = (first["performance"], second["performance"])
        fps = fmean(_performance_means(version["engines"]["candidate"])[0] for version in versions)
        reference_fps, _ = _performance_means(versions[0]["engines"]["reference"])
        second["coupled_performance"] = {
            "candidate_fps": fps,
            "reference_fps": reference_fps,
            "fps_ratio": None if any(version["reasons"] for version in versions) else fps / reference_fps,
        }
    return {
        "format": "sim-bench-comparison-v1" if legacy_fps else _COMPARISON_FORMAT,
        "protocol": plan["protocol"],
        "cases": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    def number(value: float | None) -> str:
        return "NA" if value is None else f"{value:.8g}"

    lines = ["# 仿真评测结果", ""]
    first = report["cases"][0]
    protocols = first.get("protocol", {})
    uniform = all(row.get("protocol", {}) == protocols for row in report["cases"])
    if uniform and "accuracy" in first:
        accuracy = protocols.get("accuracy", report["protocol"]["accuracy"])
        lines.append(f"精度每端采集 {accuracy['num_frames']} 帧，独立运行 {accuracy['repetitions']} 次。")
    if uniform and "performance" in first:
        performance = protocols.get("performance", report["protocol"]["performance"])
        trials = performance.get("repetitions", 1)
        if "num_frames" in performance:
            length = f"每次预热后严格计时 {performance['num_frames']} 帧"
        else:
            length = f"每次目标计时 {performance['duration_seconds']:g} 秒"
        lines.append(f"性能每端独立运行 {trials} 次，{length}。")
    if "performance" in first:
        lines.append("FPS 为各轮总帧数除以总耗时。显存为各轮进程峰值的算术平均，单位为 MiB。")
    lines.extend(
        [
            "",
            "| 场景 | x_C | x_D | x_K | 候选 FPS | 参考 FPS | FPS 比 | 候选显存 | 参考显存 | 显存比 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    notes = []
    for row in report["cases"]:
        accuracy, performance = row.get("accuracy", {}), row.get("performance")
        cells = [row["case"], *(number(accuracy.get(key)) for key in ("x_C", "x_D", "x_K"))]
        if performance is None:
            cells.extend(["NA"] * 6)
        else:
            (a_fps, a_peak), (b_fps, b_peak) = (
                _performance_means(performance["engines"][role]) for role in ("candidate", "reference")
            )
            cells.extend(
                [
                    number(a_fps),
                    number(b_fps),
                    number(performance["fps_ratio"]),
                    number(a_peak / 1024**2),
                    number(b_peak / 1024**2),
                    number(performance["gpu_peak_ratio"]),
                ]
            )
            notes.extend(f"- {row['case']} · {reason}" for reason in performance["reasons"])
        if combined := row.get("coupled_performance"):
            notes.append(
                f"- 流固耦合两版本平均 {number(combined['candidate_fps'])} FPS，"
                f"Newton {number(combined['reference_fps'])} FPS，FPS 比 {number(combined['fps_ratio'])}。"
                "各版本显存见表。"
            )
        lines.append("| " + " | ".join(cells) + " |")
    extended = [row for row in report["cases"] if "metric" in row.get("accuracy", {})]
    if extended:
        lines.extend(["", "| 扩展场景 | 精度指标 |", "| --- | --- |"])
        for row in extended:
            result = row["accuracy"]
            fields = {
                "conservation": result.get("ratios"),
                "particles": {key: result[key] for key in ("x_F", "x_p", "x_v")}
                if "x_F" in result
                else result.get("max_relative_rms"),
                "pressure_cube": {
                    key: result[key]
                    for key in ("floor_pressure_relative_error", "cube_center_mean_relative_error")
                    if key in result
                },
                "volume": {
                    key: result[key]
                    for key in ("candidate_mean_volume_loss_fraction", "reference_mean_volume_loss_fraction")
                    if key in result
                },
                "tgv": {
                    key: result[key]
                    for key in ("energy_relative_l2", "dissipation_relative_l2", "mean_relative_l2")
                    if key in result
                },
            }[result["metric"]]
            metrics = "，".join(f"{key}={number(value)}" for key, value in fields.items())
            lines.append(f"| {row['case']} | {metrics} |")
    if notes:
        lines.extend(["", *notes])
    return "\n".join(lines) + "\n"


def _run_adapter(command: list[str], request: dict[str, Any], directory: Path, timeout: float) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    request_path = directory / "request.json"
    write_json(request_path, request)
    log = directory / "worker.log"
    argv = [*command, "--request", str(request_path), "--output", str(directory)]
    deadline = time.monotonic() + timeout
    memory = None
    with (
        log.open("wb") as output,
        subprocess.Popen(
            argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix")
        ) as process,
    ):
        try:
            if request["mode"] == "performance":
                memory = sample_gpu_memory(
                    process,
                    selector=request["gpu_devices"][0],
                    interval_ms=request["protocol"]["memory_sample_interval_ms"],
                    deadline=deadline,
                    log=log,
                )
            code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            if process.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
            raise
    if code != 0:
        raise RuntimeError(f"adapter exited with code {code}; see {log}")
    if memory is not None:
        if memory["sample_count"] == 0 or memory["peak_bytes"] <= 0:
            raise RuntimeError(f"NVML did not observe GPU memory for the adapter process; see {log}")
        metrics = read_json(directory / "report.json")["metrics"]
        write_json(directory / "gpu-memory.json", select_timed_memory(memory, metrics.get("timing_window")))


def run(output: Path, plan: dict[str, Any], commands: dict[str, list[str]], gpu: str, timeout: float) -> dict[str, Any]:
    """Run the selected case/mode matrix on one GPU, one adapter at a time."""

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {**plan, "comparison_version": 2}
    write_json(output / "run.json", {**plan, "commands": commands, "gpu": gpu})
    references: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for case in plan["cases"]:
        kind = metric_kind(case)
        ref = reference_case(case)
        roles = _roles(case)
        protocols = {mode: case_protocol(plan, case, mode) for mode in plan["modes"]}
        reused_modes: set[str] = set()
        trials: list[tuple[str, str, int]] = []
        for mode, protocol in protocols.items():
            ref_key = (ref["id"], mode)
            ref_run = (ref, protocol)
            if ref_key in references:
                if references[ref_key] != ref_run:
                    raise ValueError(f"inconsistent configuration for shared reference {ref['id']}")
                reused_modes.add(mode)
            if "reference" in roles:
                references[ref_key] = ref_run
            if kind == "tgv" and mode == "accuracy":
                root = _mode_directory(output, case, mode)
                root.mkdir(parents=True)
                shutil.copyfile(SPECTRUM, root / "spectrum.gdiag")
            repetitions = protocol.get("repetitions", 1)
            if mode == "accuracy":
                trials.extend((mode, role, i) for role in roles for i in range(1, repetitions + 1))
            else:
                for i in range(1, repetitions + 1):
                    trials.extend((mode, role, i) for role in (roles if i % 2 else roles[::-1]))
        if case.get("paired_passes"):
            trials.sort(key=lambda trial: trial[2])
        identity = None
        for mode, role, run_number in trials:
            if role == "reference" and mode in reused_modes:
                continue
            protocol = protocols[mode]
            repetitions = protocol.get("repetitions", 1)
            trial_case = ref if role == "reference" else case
            directory = _trial_directory(output, case, mode, role, run_number, repetitions)
            request = {
                "format": REQUEST_FORMAT,
                "case": trial_case,
                "mode": mode,
                "protocol": protocol,
                "gpu_devices": [gpu],
                "reference_identity": identity if mode == "accuracy" else None,
            }
            if role == "candidate" and kind == "volume":
                reference_dir = _trial_directory(output, case, mode, "reference", run_number, repetitions)
                snapshot = reference_dir / "initial_snapshot.npz"
                if not snapshot.is_file():
                    raise ValueError("reference did not supply the coupled single-box initial snapshot")
                request["reference_snapshot"] = str(snapshot)
            print(f"{case['id']} {mode} {role} ({run_number}/{repetitions})", flush=True)
            _run_adapter(commands[role], request, directory, timeout)
            _read_report(directory, trial_case, mode, protocol)
            if mode == "accuracy" and role == "reference" and run_number == 1 and kind == "rigid":
                trace = read_json(directory / "trajectory.json")
                validate_trace(trace, case, protocol["num_frames"])
                identity = {
                    "format": IDENTITY_FORMAT,
                    "bodies": [{key: row[key] for key in ("world", "name")} for row in trace["model"]],
                    "shapes": trace["shapes"],
                }
                write_json(_mode_directory(output, case, mode) / "identity.json", identity)

    report = _read_comparison(output, plan, save_event_details=True)
    write_json(output / "report.json", report)
    with (output / "report.md").open("x", encoding="utf-8") as file:
        file.write(_markdown(report))
    write_json(output / "hashes.json", _file_hashes(output))
    return report


def verify(output: Path) -> None:
    """Check saved files and reproduce the report without running an engine."""

    if read_json(output / "hashes.json") != _file_hashes(output):
        raise ValueError("run artifacts differ from hashes.json")
    plan = load_suite(output / "run.json")
    if not plan["modes"] or any(mode not in ("accuracy", "performance") for mode in plan["modes"]):
        raise ValueError("saved run has invalid modes")
    expected = read_json(output / "report.json")
    # The original fixed-frame release used v1 with a per-trial mean duration.
    legacy_frame_summary = (
        expected["format"] == "sim-bench-comparison-v1"
        and "num_frames" in plan["protocol"]["performance"]
        and any(
            "elapsed_seconds" in side.get("mean", {})
            for row in expected["cases"]
            for side in row.get("performance", {}).get("engines", {}).values()
        )
    )
    report = _read_comparison(output, plan, legacy_frame_summary=legacy_frame_summary)
    if report != expected:
        raise ValueError("recomputed comparison differs from the saved report")
