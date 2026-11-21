# SPDX-License-Identifier: Apache-2.0
"""Example adapter for an engine that already has runnable scenes."""

import argparse
from importlib import import_module
from pathlib import Path

from newton_test_tools import MonitorConfig, read_request, run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-module", required=True, help="Import path of your scene factory.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = read_request(args.request)
    settings = request["protocol"]
    accuracy = request["mode"] == "accuracy"
    config = MonitorConfig(engine="Candidate", gpu_devices=tuple(request["gpu_devices"]))
    scene_module = import_module(args.scene_module)
    with scene_module.create_scene(
        request["case"],
        device=request["gpu_devices"][0],
        graph=settings["graph"],
        deterministic=settings["deterministic"],
        reference_identity=request.get("reference_identity"),
    ) as scene:
        run_trial(
            args.output,
            config,
            request=request,
            runtime=scene.runtime(),
            step=scene.step,
            synchronize=scene.synchronize,
            read_initial=scene.read_initial if accuracy else None,
            observe=scene.observe if accuracy else None,
        )


if __name__ == "__main__":
    main()
