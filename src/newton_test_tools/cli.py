"""Command line entry points for running or reviewing a benchmark."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .protocol import case_protocol, load_suite, metric_kind, positive_int, positive_number, read_json, write_json
from .runner import compare_trajectories, run, verify
from .score_reports import read_team_reports
from .scoring import score
from .worker import run_worker


def _add_case_arguments(parser: argparse.ArgumentParser, *, repeatable: bool = False) -> None:
    parser.add_argument("--suite", type=Path, required=True, help="Path to a versioned suite JSON.")
    parser.add_argument("--case", action="append" if repeatable else "store", required=not repeatable)
    parser.add_argument("--frames", type=int, help="Override the accuracy frame count for a diagnostic run.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newton-test", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="Run reference and candidate adapters.")
    _add_case_arguments(execute, repeatable=True)
    execute.add_argument("--mode", choices=("all", "accuracy", "performance"), default="all")
    execute.add_argument("--reference", help="Reference adapter command as a single quoted string.")
    execute.add_argument("--candidate", required=True, help="Candidate adapter command as a single quoted string.")
    execute.add_argument("--gpu", default="0", help="NVML device index or UUID used by both engines.")
    performance_length = execute.add_mutually_exclusive_group()
    performance_length.add_argument("--duration", type=float, help="Set the wall-clock duration per performance trial.")
    performance_length.add_argument(
        "--performance-frames", type=int, help="Time exactly this many frames per performance trial, after warmup."
    )
    execute.add_argument("--repetitions", type=int, help="Override the number of accuracy runs per engine.")
    execute.add_argument(
        "--performance-repetitions", type=int, help="Override the number of performance trials per engine."
    )
    execute.add_argument(
        "--timeout", type=float, default=1800, help="Process timeout per adapter invocation, in seconds."
    )
    execute.add_argument("--output", type=Path, required=True, help="New run directory.")

    worker = commands.add_parser("worker", help="Run one trial from an engine adapter module or file.")
    worker.add_argument("--adapter", required=True, help="Importable module or Python file exporting prepare_trial().")
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare", help="Compare two previously exported trajectories.")
    _add_case_arguments(compare)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, help="Optional new JSON file for the full accuracy report.")

    report = commands.add_parser("report", help="View the results from a completed run.")
    report.add_argument("directory", type=Path)
    report.add_argument("--json", action="store_true")
    check = commands.add_parser("verify", help="Check hashes and recompute a completed report.")
    check.add_argument("directory", type=Path)
    scoring = commands.add_parser("score", help="Calculate observation, task and total scores.")
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--rules", type=Path, default=Path("cases/scoring.json"))
    scoring.add_argument("--output", type=Path, required=True)
    score_reports = commands.add_parser("score-reports", help="Calculate scores from benchmark report directories.")
    score_reports.add_argument("--team", action="append", nargs=2, metavar=("ID", "RUN_DIR"), required=True)
    score_reports.add_argument("--rules", type=Path, default=Path("cases/scoring.json"))
    score_reports.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("score", "score-reports"):
            if args.command == "score":
                data = read_json(args.input)
            else:
                teams: dict[str, list[Path]] = {}
                for team_id, directory in args.team:
                    teams.setdefault(team_id, []).append(Path(directory))
                if len(teams) == 1:
                    placeholder = "pending"
                    while placeholder in teams:
                        placeholder += "_"
                    data = {
                        "teams": [read_team_reports(team_id, directories) for team_id, directories in teams.items()]
                    }
                    scene_ids = {scene for task in read_json(args.rules)["tasks"] for scene in task["scenes"]}
                    data["teams"].append(
                        {
                            "id": placeholder,
                            "scenes": {
                                scene_id: {
                                    "awaiting_results": True,
                                    "pending_observations": ["performance", "accuracy", "memory"],
                                }
                                for scene_id in scene_ids
                            },
                        }
                    )
                else:
                    data = {
                        "teams": [read_team_reports(team_id, directories) for team_id, directories in teams.items()]
                    }
            result = score(data, read_json(args.rules))
            if args.command == "score-reports":
                result["sources"] = {team["id"]: team.get("sources", []) for team in data["teams"]}
                by_id = {team["id"]: team for team in data["teams"]}
                for team in result["teams"]:
                    for scene in team["scenes"]:
                        scene["evidence"] = (
                            by_id[team["id"]].get("scenes", {}).get(scene["scene"], {}).get("source_cases", {})
                        )
            write_json(args.output, result)
            for team in result["teams"]:
                points = team["total_points"]
                display = f"{points:.2f}" if points is not None else "pending"
                print(f"{team['id']}: {display} / {result['maximum_points']}")
            return 0
        if args.command == "worker":
            run_worker(args.adapter, args.request, args.output)
            return 0
        if args.command == "verify":
            verify(args.directory)
            print("Artifacts verified and comparison recomputed.")
            return 0
        if args.command == "report":
            path = args.directory / ("report.json" if args.json else "report.md")
            print(path.read_text(encoding="utf-8"), end="")
            return 0

        suite = load_suite(args.suite)
        cases = {case["id"]: case for case in suite["cases"]}
        overrides: dict = {"accuracy": {}, "performance": {}}
        if args.frames is not None:
            overrides["accuracy"]["num_frames"] = positive_int(args.frames, "--frames")
        if args.command == "compare":
            case = cases[args.case]
            if metric_kind(case) != "rigid":
                raise ValueError("use verify to recompute an extended-case run")
            frames = case_protocol({**suite, "overrides": overrides}, case, "accuracy")["num_frames"]
            report = compare_trajectories(case, frames, read_json(args.candidate), read_json(args.reference))
            if args.output is not None:
                write_json(args.output, report)
            fields = ("x_C", "x_D", "x_K", "x", "orientation_relative_max")
            print(json.dumps({key: report[key] for key in fields if key in report}, allow_nan=False, indent=2))
            return 0

        performance = overrides["performance"]
        if args.duration is not None:
            performance["duration_seconds"] = positive_number(args.duration, "--duration")
        elif args.performance_frames is not None:
            performance["num_frames"] = positive_int(args.performance_frames, "--performance-frames")
        if args.repetitions is not None:
            overrides["accuracy"]["repetitions"] = positive_int(args.repetitions, "--repetitions")
        if args.performance_repetitions is not None:
            performance["repetitions"] = positive_int(args.performance_repetitions, "--performance-repetitions")
        modes = ["accuracy", "performance"] if args.mode == "all" else [args.mode]
        requested_cases = args.case or list(cases)
        if len(requested_cases) != len(set(requested_cases)):
            raise ValueError("each --case can appear only once")
        selected = [cases[name] for name in requested_cases]
        plan = {**suite, "modes": modes, "cases": selected, "overrides": overrides}
        commands = {"candidate": shlex.split(args.candidate)}
        if any(metric_kind(case) != "tgv" for case in selected):
            if args.reference is None:
                raise ValueError("a Newton reference adapter command is required for the selected cases")
            commands["reference"] = shlex.split(args.reference)
        if not all(commands.values()):
            raise ValueError("adapter command must not be empty")
        result = run(args.output, plan, commands, args.gpu, positive_number(args.timeout, "--timeout"))
        print(f"Report: {args.output.resolve() / 'report.md'}")
        return int(any(case.get("performance", {}).get("reasons") for case in result["cases"]))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"newton-test: {error}", file=sys.stderr)
        return 1
