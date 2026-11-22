"""Versioned file interface shared by the runner and engine adapters."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SUITE_FORMAT: str = "sim-bench-suite-v1"
REQUEST_FORMAT: str = "sim-bench-request-v1"
RESULT_FORMAT: str = "sim-bench-result-v1"
TRACE_FORMAT: str = "sim-bench-rigid-trace-v1"
IDENTITY_FORMAT: str = "sim-bench-identity-v1"
FLOOR_FIELDS: tuple[str, ...] = (
    "force_N",
    "torque_com_Nm",
    "position_m",
    "linear_velocity_mps",
    "angular_velocity_radps",
)


def read_json(path: Path) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"{path}: invalid JSON number {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any], *, compact: bool = False) -> None:
    options = {"separators": (",", ":")} if compact else {"indent": 2}
    content = json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, **options)
    with path.open("x", encoding="utf-8") as output:
        output.write(content + "\n")


def positive_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def positive_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _check_case(case: dict[str, Any]) -> str:
    case_id = _name(case["id"], "case id")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is None:
        raise ValueError(f"invalid case id {case_id!r}")
    positive_number(case["frame_dt_seconds"], "frame_dt_seconds")
    positive_int(case["substeps"], "substeps")
    positive_int(case["world_count"], "world_count")
    for field in ("solver", "collision"):
        _name(case[field], field)
    for field in FLOOR_FIELDS:
        positive_number(case["normalization_floor"][field], field)
    return case_id


def _check_protocol(mode: str, config: dict[str, Any]) -> None:
    for field in ("deterministic", "graph"):
        if type(config[field]) is not bool:
            raise ValueError(f"{mode}.{field} must be a boolean")
    if mode == "accuracy":
        positive_int(config["num_frames"], "num_frames")
        positive_int(config["repetitions"], "repetitions")
        if config["contact_stage"] != "end_of_frame":
            raise ValueError("v1 requires end_of_frame contacts")
    elif mode == "performance":
        positive_int(config["num_frames"], "performance.num_frames")
        positive_int(config.get("repetitions", 1), "performance.repetitions")
        positive_number(config["memory_sample_interval_ms"], "memory_sample_interval_ms")
        positive_int(config["warmup_frames"], "warmup_frames", minimum=0)
        positive_int(config["sync_interval_frames"], "sync_interval_frames")
    else:
        raise ValueError(f"unknown benchmark mode {mode!r}")


def load_suite(path: Path) -> dict[str, Any]:
    suite = read_json(path)
    if suite["format"] != SUITE_FORMAT:
        raise ValueError("unsupported benchmark suite version")
    for mode in ("accuracy", "performance"):
        _check_protocol(mode, suite["protocol"][mode])
    case_ids = [_check_case(case) for case in suite["cases"]]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("suite must contain distinct case IDs")
    return suite


def read_request(path: Path) -> dict[str, Any]:
    """Load an adapter request issued by newton-test run."""

    request = read_json(path)
    if request["format"] != REQUEST_FORMAT:
        raise ValueError("unsupported adapter request version")
    _check_case(request["case"])
    _check_protocol(request["mode"], request["protocol"])
    if not isinstance(request["gpu_devices"], list) or len(request["gpu_devices"]) != 1:
        raise ValueError("v1 requests select one GPU")
    _name(request["gpu_devices"][0], "GPU selector")
    return request


def check_runtime(request: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Require measured runtime settings to agree with the case and mode."""

    case, protocol = request["case"], request["protocol"]
    if not math.isclose(
        positive_number(runtime["frame_dt_seconds"], "actual frame dt"),
        case["frame_dt_seconds"],
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("runtime frame step differs from case")
    expected = {
        **{field: case[field] for field in ("substeps", "world_count", "solver", "collision")},
        **{field: protocol[field] for field in ("deterministic", "graph")},
    }
    if request["mode"] == "accuracy":
        expected["contact_stage"] = protocol["contact_stage"]
    for field, value in expected.items():
        if field == "graph" and runtime[field] is None:
            # Engines without a graph execution mode report that explicitly.
            continue
        if type(runtime[field]) is not type(value) or runtime[field] != value:
            raise ValueError(f"runtime {field}={runtime[field]!r}, expected {value!r}")
    _name(runtime["device_uuid"], "runtime device UUID")


def _identity(row: dict[str, Any], world_count: int, *, allow_shared: bool) -> tuple[int, str]:
    world = row["world"]
    if type(world) is not int or not (-int(allow_shared) <= world < world_count):
        raise ValueError(f"invalid world ID {world!r}")
    return world, _name(row["name"], "body or shape ID")


def _identities(rows: list[dict[str, Any]], world_count: int, *, allow_shared: bool) -> set[tuple[int, str]]:
    keys = {_identity(row, world_count, allow_shared=allow_shared) for row in rows}
    if len(rows) != len(keys):
        raise ValueError("duplicate body or shape ID")
    return keys


def canonical_pairs(rows: list[list[Any]], shape_ids: set[tuple[int, str]]) -> list[list[Any]]:
    """Resolve unordered, unique pairs and reject IDs outside the shape table."""

    if not isinstance(rows, list):
        raise ValueError("contact pairs must be a list, including [] when there is no contact")
    pairs: set[tuple[int, str, str]] = set()
    for row in rows:
        world, a, b = row
        if (
            type(world) is not int
            or not isinstance(a, str)
            or not isinstance(b, str)
            or a == b
            or (world, a) not in shape_ids
            or (world, b) not in shape_ids
        ):
            raise ValueError(f"invalid or unknown contact shape pair {row!r}")
        left, right = sorted((a, b))
        pairs.add((world, left, right))
    return [list(pair) for pair in sorted(pairs)]


def validate_trace(trace: dict[str, Any], case: dict[str, Any], frames: int) -> None:
    """Validate and canonicalize one continuous trajectory in place."""

    expected = {"format": TRACE_FORMAT, "case": case["id"], "solver": case["solver"], "num_frames": frames}
    for field, value in expected.items():
        if type(trace[field]) is not type(value) or trace[field] != value:
            raise ValueError(f"trajectory {field} differs from case: {trace[field]!r}")
    if not math.isclose(
        positive_number(trace["frame_dt_seconds"], "trajectory frame dt"),
        case["frame_dt_seconds"],
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("trajectory frame step differs from case")
    if trace["velocity_convention"] not in ("com_world", "origin_world"):
        raise ValueError("invalid trajectory velocity convention")
    count = case["world_count"]
    body_ids = _identities(trace["model"], count, allow_shared=True)
    shape_ids = _identities(trace["shapes"], count, allow_shared=False)
    if not body_ids:
        raise ValueError("trajectory must include a body model")
    for shape in trace["shapes"]:
        world, owner = shape["world"], shape["body"]
        if owner is not None and (world, owner) not in body_ids and (-1, owner) not in body_ids:
            raise ValueError(f"shape {shape['name']!r} has an unknown body ID")
        if type(shape["type"]) is not int or not 0 <= shape["type"] <= 11:
            raise ValueError("unknown v1 shape type")
    states, pairs = trace["states"], trace["pairs"]
    if len(states) != frames + 1 or set(pairs) != {str(i) for i in range(1, frames + 1)}:
        raise ValueError("incomplete state or contact trajectory")
    for frame, state in enumerate(states):
        if type(state["frame"]) is not int or state["frame"] != frame:
            raise ValueError(f"expected state frame {frame}")
        if _identities(state["bodies"], count, allow_shared=True) != body_ids:
            raise ValueError(f"body IDs differ from the model at frame {frame}")
        if frame:
            pairs[str(frame)] = canonical_pairs(pairs[str(frame)], shape_ids)
