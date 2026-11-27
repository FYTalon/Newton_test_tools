"""Map scene measurements to observation, task and total scores."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def score_curve(value: float, anchors: list[list[float]]) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("a score input must be finite and nonnegative")
    if value <= anchors[0][0]:
        return float(anchors[0][1])
    for (left, left_score), (right, right_score) in zip(anchors, anchors[1:]):
        if value <= right:
            return left_score + (right_score - left_score) * (value - left) / (right - left)
    return float(anchors[-1][1])


def _observation(value: float | None, points: float, curve: list[list[float]]) -> dict[str, Any]:
    return {
        "raw": value,
        "anchors": curve,
        "score": score_curve(value, curve) if value is not None else 0.0,
        "maximum_points": points,
        "status": "ok" if value is not None else "missing",
    }


def _eligible(scene: dict, task: dict) -> bool:
    if scene.get("awaiting_results"):
        return True
    if scene.get("status", "ok") != "ok":
        return False
    if "minimum_effective_dof" in task:
        dof = _number(scene.get("effective_dof"))
        if dof is None and scene.get("pending_observations"):
            return True
        return dof is not None and dof >= task["minimum_effective_dof"]
    return True


def _conservation(scene: dict, task: dict, scene_id: str, rules: dict) -> dict:
    points = task["observations"]["accuracy"]
    result = {"score": 0.0, "maximum_points": points, "status": "missing", "components": {}}
    frames = _number(scene.get("completed_frames"))
    substeps = _number(scene.get("completed_substeps"))
    required_frames = task["physics_frames"]
    if frames is None or substeps is None or frames < required_frames:
        return result
    if substeps < required_frames * task["substeps"][scene_id]:
        return result
    accuracy = scene.get("accuracy", {})
    candidate, reference = accuracy.get("candidate", {}), accuracy.get("reference", {})
    configuration = rules["conservation"]
    components = result["components"]
    scores, ratios = [], []
    for field in configuration["fields"]:
        actual, expected = _number(candidate.get(field)), _number(reference.get(field))
        if expected is None:
            return result
        if expected <= configuration["exclude_reference_abs_at_most"]:
            components[field] = {"reference": expected, "status": "excluded"}
            continue
        if actual is None:
            return result
        ratio = actual / expected
        mapped = score_curve(ratio, rules["curves"]["conservation"])
        components[field] = {"candidate": actual, "reference": expected, "ratio": ratio, "score": mapped}
        scores.append(mapped)
        ratios.append(ratio)
    if scores:
        result.update(
            score=math.fsum(scores) / len(scores),
            status="ok",
            qualified=all(ratio <= configuration["qualification_ratio_at_most"] for ratio in ratios),
        )
    return result


def _accuracy(scene: dict, task: dict, scene_id: str, rules: dict) -> dict:
    points = task["observations"]["accuracy"]
    accuracy = scene.get("accuracy", {})
    curves = rules["curves"]
    if task["id"] == 1:
        values = [_number(accuracy.get(field)) for field in ("x_C", "x_D", "x_K")]
        value = max(values) if all(item is not None for item in values) else None
        result = _observation(value, points, curves["trajectory"])
        if value is not None:
            result["qualified"] = value <= 0.1
        return result
    if task["id"] == 2:
        return _conservation(scene, task, scene_id, rules)
    if scene_id == "1-9":
        value = _number(accuracy.get("x_F"))
        result = _observation(value, points, curves["trajectory"])
        if value is not None:
            result["qualified"] = value <= 0.1
        return result
    if scene_id == "1-11":
        energy = _number(accuracy.get("energy_relative_l2"))
        dissipation = _number(accuracy.get("dissipation_relative_l2"))
        value = energy / 2 + dissipation / 2 if energy is not None and dissipation is not None else None
        return _observation(value, points, curves["tgv"])
    components = {}
    for field, specification in rules["coupled"]["components"].items():
        raw = accuracy.get(field)
        if field == "volume_loss_error" and type(raw) in (int, float) and math.isfinite(raw):
            raw = max(0.0, raw)
        value = _number(raw)
        anchors = [[specification["full_score_error"], 100], [1.0, 0]]
        components[field] = {**_observation(value, points, anchors), "weight": specification["weight"]}
    complete = all(component["status"] == "ok" for component in components.values())
    return {
        "score": math.fsum(component["score"] * component["weight"] for component in components.values())
        if complete
        else 0.0,
        "maximum_points": points,
        "status": "ok" if complete else "missing",
        "components": components,
    }


def _round(value: float, places: int) -> float:
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def score(data: dict, rules: dict) -> dict:
    teams = data.get("teams", [])
    if len(teams) != 2 or any(not isinstance(team.get("id"), str) or not team["id"] for team in teams):
        raise ValueError("provide two teams with nonempty string identifiers")
    if teams[0]["id"] == teams[1]["id"]:
        raise ValueError("team identifiers must differ")
    tasks = rules["tasks"]
    maximum = rules["maximum_points"]
    if sum(task["maximum_points"] for task in tasks) != maximum:
        raise ValueError("task points do not add up to the maximum")
    known_scenes = {scene for task in tasks for scene in task["scenes"]}
    if len(known_scenes) != sum(len(task["scenes"]) for task in tasks):
        raise ValueError("each scene must belong to exactly one task")
    for task in tasks:
        if sum(task["observations"].values()) != task["maximum_points"]:
            raise ValueError("observation points do not add up to the task maximum")
    for team in teams:
        if set(team.get("scenes", {})) - known_scenes:
            raise ValueError("unknown scene identifier")
    for anchors in rules["curves"].values():
        if len(anchors) < 2 or any(right[0] <= left[0] for left, right in zip(anchors, anchors[1:])):
            raise ValueError("score anchors must be strictly increasing")
    output = {"format": "sim-bench-scores-v1", "maximum_points": maximum, "teams": []}
    places = rules["aggregation"]["display_decimal_places"]
    for team in teams:
        rows, task_rows, unrounded_tasks = [], [], []
        for task in tasks:
            task_points = []
            for scene_id in task["scenes"]:
                scene = team.get("scenes", {}).get(scene_id, {})
                ratio = _number(scene.get("fps_ratio"))
                peer_ratios = []
                for peer in teams:
                    peer_scene = peer.get("scenes", {}).get(scene_id, {})
                    peer_ratio = _number(peer_scene.get("fps_ratio"))
                    failed = peer_scene.get("failed_observations", [])
                    pending = peer_scene.get("pending_observations", [])
                    if (
                        _eligible(peer_scene, task)
                        and "performance" not in failed
                        and "performance" not in pending
                        and peer_ratio is not None
                    ):
                        peer_ratios.append(peer_ratio)
                highest = max(peer_ratios) if len(peer_ratios) == len(teams) else None
                performance_score = 0.0
                performance_status = "ok" if ratio is not None else "missing"
                if ratio is not None and ratio <= 1:
                    performance_score = 80 * ratio
                elif ratio is not None and highest is None:
                    performance_score = None
                    performance_status = "pending_peer"
                elif ratio is not None and highest is not None and highest > 1:
                    performance_score = 80 + 20 * ((ratio - 1) / (highest - 1))
                observations = {
                    "performance": {
                        "raw": ratio,
                        "H": highest,
                        "score": performance_score,
                        "maximum_points": task["observations"]["performance"],
                        "status": performance_status,
                    },
                    "accuracy": _accuracy(scene, task, scene_id, rules),
                    "memory": _observation(
                        _number(scene.get("gpu_peak_ratio")),
                        task["observations"]["memory"],
                        rules["curves"]["memory"],
                    ),
                }
                for name in scene.get("pending_observations", []):
                    if name in observations:
                        observations[name].update(score=None, status="pending_definition")
                eligible = _eligible(scene, task)
                for name, observation in observations.items():
                    if not eligible or name in scene.get("failed_observations", []):
                        observation.update(score=0.0, status="failed")
                    observation["points"] = (
                        observation["maximum_points"] * observation["score"] / 100
                        if observation["score"] is not None
                        else None
                    )
                values = [observation["points"] for observation in observations.values()]
                points = math.fsum(values) if all(value is not None for value in values) else None
                task_points.append(points)
                rows.append({"scene": scene_id, "task": task["id"], "observations": observations, "points": points})
            points = (
                math.fsum(task_points) / len(task_points) if all(value is not None for value in task_points) else None
            )
            unrounded_tasks.append(points)
            task_rows.append(
                {
                    "task": task["id"],
                    "points": _round(points, places) if points is not None else None,
                    "maximum_points": task["maximum_points"],
                }
            )
        output["teams"].append(
            {
                "id": team["id"],
                "scenes": rows,
                "tasks": task_rows,
                "total_points": (
                    _round(math.fsum(unrounded_tasks), places)
                    if all(value is not None for value in unrounded_tasks)
                    else None
                ),
            }
        )
    return output
