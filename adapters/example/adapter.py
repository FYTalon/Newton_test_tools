# SPDX-License-Identifier: Apache-2.0
"""Copy this file and import your own scene factory."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from newton_test_tools import Trial
from your_engine.benchmark_scenes import create_scene


@contextmanager
def prepare_trial(request: dict[str, Any], *, output: Path) -> Iterator[Trial]:
    settings = request["protocol"]
    accuracy = request["mode"] == "accuracy"
    with create_scene(
        request["case"],
        device=request["gpu_devices"][0],
        graph=settings["graph"],
        deterministic=settings["deterministic"],
        reference_identity=request.get("reference_identity"),
    ) as scene:
        yield Trial(
            engine="YourEngine",
            runtime=scene.runtime(),
            step=scene.step,
            synchronize=scene.synchronize,
            read_initial=scene.read_initial if accuracy else None,
            observe=scene.observe if accuracy else None,
        )
