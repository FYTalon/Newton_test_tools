"""Command line entry points for running or reviewing a benchmark."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .protocol import load_suite, positive_int, positive_number, read_json, write_json
from .runner import compare_trajectories, run, verify
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
    execute.add_argument("--reference", required=True, help="Reference adapter command as a single quoted string.")
    execute.add_argument("--candidate", required=True, help="Candidate adapter command as a single quoted string.")
    execute.add_argument("--gpu", default="0", help="NVML device index or UUID used by both engines.")
    execute.add_argument("--duration", type=float, help="Override wall-clock duration per performance trial.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
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
        if args.frames is not None:
            suite["protocol"]["accuracy"]["num_frames"] = positive_int(args.frames, "--frames")
        frames = suite["protocol"]["accuracy"]["num_frames"]
        cases = {case["id"]: case for case in suite["cases"]}
        if args.command == "compare":
            report = compare_trajectories(
                cases[args.case], frames, read_json(args.candidate), read_json(args.reference)
            )
            if args.output is not None:
                write_json(args.output, report)
            print(json.dumps({key: report[key] for key in ("x_C", "x_D", "x_K")}, allow_nan=False, indent=2))
            return 0

        if args.duration is not None:
            suite["protocol"]["performance"]["duration_seconds"] = positive_number(args.duration, "--duration")
        if args.repetitions is not None:
            suite["protocol"]["accuracy"]["repetitions"] = positive_int(args.repetitions, "--repetitions")
        if args.performance_repetitions is not None:
            suite["protocol"]["performance"]["repetitions"] = positive_int(
                args.performance_repetitions, "--performance-repetitions"
            )
        modes = ["accuracy", "performance"] if args.mode == "all" else [args.mode]
        requested_cases = args.case or list(cases)
        if len(requested_cases) != len(set(requested_cases)):
            raise ValueError("each --case can appear only once")
        plan = {**suite, "modes": modes, "cases": [cases[name] for name in requested_cases]}
        commands = {role: shlex.split(getattr(args, role)) for role in ("reference", "candidate")}
        if not all(commands.values()):
            raise ValueError("adapter command must not be empty")
        result = run(args.output, plan, commands, args.gpu, positive_number(args.timeout, "--timeout"))
        print(f"Report: {args.output.resolve() / 'report.md'}")
        return int(any(case.get("performance", {}).get("reasons") for case in result["cases"]))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"newton-test: {error}", file=sys.stderr)
        return 1
