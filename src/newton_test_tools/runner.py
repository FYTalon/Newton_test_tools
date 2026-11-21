"""Run independent adapters and recompute comparison reports from their files."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from .accuracy import compare_accuracy
from .gpu import sample_gpu_memory
from .monitor import performance_metrics
from .protocol import (
    IDENTITY_FORMAT,
    REQUEST_FORMAT,
    RESULT_FORMAT,
    check_runtime,
    load_suite,
    positive_int,
    read_json,
    validate_trace,
    write_json,
)

_COMPARISON_FORMAT: str = "sim-bench-comparison-v1"
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


def _read_performance(root: Path, case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    sides = {}
    reasons: list[str] = []
    repetitions = protocol.get("repetitions", 1)
    first_trials = {}
    for role in ("reference", "candidate"):
        directory = root / role
        trials = [
            _read_performance_trial(directory / f"run-{i}" if repetitions > 1 else directory, case, protocol)
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
            sides[role] = {
                "engine": first["engine"],
                "mean": {
                    "fps": fmean(trial["metrics"]["fps"] for trial in trials),
                    "gpu_peak_bytes": fmean(trial["gpu_memory"]["peak_bytes"] for trial in trials),
                },
                "trials": trials,
            }
    a, b = first_trials["candidate"], first_trials["reference"]
    if (a["gpu_memory"]["device_uuid"], a["gpu_memory"]["driver_version"]) != (
        b["gpu_memory"]["device_uuid"],
        b["gpu_memory"]["driver_version"],
    ):
        reasons.append("GPU or driver differs between the two runs")
    for field in _PHYSICAL_COUNTS:
        if a["runtime"]["model_counts"][field] != b["runtime"]["model_counts"][field]:
            reasons.append(f"runtime model {field} differs")
    candidate_fps, candidate_peak = _performance_means(sides["candidate"])
    reference_fps, reference_peak = _performance_means(sides["reference"])
    return {
        "case": case["id"],
        "engines": sides,
        "fps_ratio": None if reasons else candidate_fps / reference_fps,
        "gpu_peak_ratio": None if reasons else candidate_peak / reference_peak,
        "reasons": reasons,
    }


def _read_comparison(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in plan["cases"]:
        row: dict[str, Any] = {"case": case["id"]}
        for mode in plan["modes"]:
            reader = _read_accuracy if mode == "accuracy" else _read_performance
            row[mode] = reader(_mode_directory(root, case, mode), case, plan["protocol"][mode])
        rows.append(row)
    return {"format": _COMPARISON_FORMAT, "protocol": plan["protocol"], "cases": rows}


def _markdown(report: dict[str, Any]) -> str:
    def number(value: float | None) -> str:
        return "NA" if value is None else f"{value:.8g}"

    lines = ["# 仿真评测结果", ""]
    if any("accuracy" in row for row in report["cases"]):
        accuracy = report["protocol"]["accuracy"]
        lines.append(f"精度每端采集 {accuracy['num_frames']} 帧，独立运行 {accuracy['repetitions']} 次。")
    if any("performance" in row for row in report["cases"]):
        performance = report["protocol"]["performance"]
        duration, trials = performance["duration_seconds"], performance.get("repetitions", 1)
        lines.append(f"性能每端独立运行 {trials} 次，每次目标计时 {duration:g} 秒。")
        lines.append("FPS 与每次显存峰值分别取算术平均，比值为候选引擎 / 参考引擎。显存单位为 MiB。")
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
        lines.append("| " + " | ".join(cells) + " |")
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
        write_json(directory / "gpu-memory.json", memory)


def run(output: Path, plan: dict[str, Any], commands: dict[str, list[str]], gpu: str, timeout: float) -> dict[str, Any]:
    """Run the selected case/mode matrix on one GPU, one adapter at a time."""

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {**plan, "commands": commands, "gpu": gpu})
    for case in plan["cases"]:
        for mode in plan["modes"]:
            protocol = plan["protocol"][mode]
            root = _mode_directory(output, case, mode)
            request = {
                "format": REQUEST_FORMAT,
                "case": case,
                "mode": mode,
                "protocol": protocol,
                "gpu_devices": [gpu],
            }
            identity = None
            roles = ("reference", "candidate")
            repetitions = protocol.get("repetitions", 1)
            if mode == "accuracy":
                runs = [(role, i) for role in roles for i in range(1, repetitions + 1)]
            else:
                runs = [(role, i) for i in range(1, repetitions + 1) for role in (roles if i % 2 else roles[::-1])]
            for role, run_number in runs:
                directory = root / role
                if mode == "accuracy" or repetitions > 1:
                    directory /= f"run-{run_number}"
                print(f"{case['id']} {mode} {role} ({run_number}/{repetitions})", flush=True)
                _run_adapter(commands[role], {**request, "reference_identity": identity}, directory, timeout)
                _read_report(directory, case, mode, protocol)
                if mode == "accuracy" and role == "reference" and run_number == 1:
                    trace = read_json(directory / "trajectory.json")
                    validate_trace(trace, case, protocol["num_frames"])
                    identity = {
                        "format": IDENTITY_FORMAT,
                        "bodies": [{key: row[key] for key in ("world", "name")} for row in trace["model"]],
                        "shapes": trace["shapes"],
                    }
                    write_json(root / "identity.json", identity)

    report = _read_comparison(output, plan)
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
    report = _read_comparison(output, plan)
    if report != read_json(output / "report.json"):
        raise ValueError("recomputed comparison differs from the saved report")
