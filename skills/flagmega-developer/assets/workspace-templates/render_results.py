#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Use the skill's fixed figure renderers from a tracked workspace script."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=["graph", "kernel"], required=True)
    parser.add_argument("--skill-dir", type=Path)
    args, forwarded = parser.parse_known_args()
    skill_dir = args.skill_dir
    if skill_dir is None:
        skill_dir = next((parent / "skills/flagmega-developer"
                          for parent in Path(__file__).resolve().parents
                          if (parent / "skills/flagmega-developer/SKILL.md").is_file()), None)
    if skill_dir is None:
        parser.error("cannot locate the skill; pass --skill-dir")
    filename = "render_workspace_figures.py" if args.renderer == "graph" else "render_kernel_swimlane.py"
    script = skill_dir / "scripts" / filename
    if not script.is_file():
        parser.error(f"missing renderer: {script}")
    return subprocess.run([sys.executable, str(script), *forwarded], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
