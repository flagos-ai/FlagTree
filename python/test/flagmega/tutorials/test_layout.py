# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import subprocess
import sys

import pytest


TUTORIALS = Path(__file__).resolve().parents[3] / "tutorials" / "flagmega"
MODELS = ("01-qwen3-1.7b-bf16", "02-qwen3.5-35b-a3b-bf16")


@pytest.mark.parametrize("model", MODELS)
def test_hardware_scoped_tutorial(model):
    root = TUTORIALS / model
    hardware = root / "nvidia-h800"
    assert hardware.is_dir()
    assert not (TUTORIALS / f"{model}-h800").exists()
    assert "nvidia-h800/readme.md" in (root / "readme.md").read_text()
    assert (hardware / ".gitignore").read_text().splitlines().count("/.local/") == 1
    for name in ("optimize.py", "accuracy.py", "generated_kernels.py", "local_optimizations"):
        assert (hardware / name).exists()
    for name in ("decode_latency.svg", "decode_throughput.svg"):
        assert (hardware / "figures" / name).is_file()
    readme = (hardware / "readme.md").read_text()
    assert f"python/tutorials/flagmega/{model}/nvidia-h800" in readme
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if "://" not in target and not target.startswith("#"):
            assert (hardware / target.split("#", 1)[0]).exists(), target


@pytest.mark.parametrize("model,script", [
    (MODELS[0], "optimize.py"),
    (MODELS[0], "render_results.py"),
    (MODELS[0], "reproduce.py"),
    (MODELS[1], "optimize.py"),
    (MODELS[1], "optimize_agent.py"),
    (MODELS[1], "accuracy.py"),
    (MODELS[1], "render_results.py"),
    (MODELS[1], "prepare_checkpoint.py"),
])
def test_moved_entrypoint_help(model, script, tmp_path):
    path = TUTORIALS / model / "nvidia-h800" / script
    result = subprocess.run([sys.executable, str(path), "--help"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
