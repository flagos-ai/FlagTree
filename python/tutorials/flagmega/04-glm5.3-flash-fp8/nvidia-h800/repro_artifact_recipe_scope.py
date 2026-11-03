# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU reproducer for stale frozen recipes during sequential full-model loads."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--unscoped", action="store_true", help="Reproduce the original worker's recipe lifetime")
    args = parser.parse_args()
    from local_optimizations.artifact_recipe import activated_artifact_recipe
    from triton.flagmega.runtime import load
    import torch

    assert not torch.cuda.is_initialized()
    if args.unscoped:
        for name in tuple(sys.modules):
            if name == "local_optimizations" or name.startswith("local_optimizations."):
                del sys.modules[name]
        sys.path.insert(0, str(args.first.resolve()))
        from local_optimizations.kernel_recipe import activate
        activate()
    for trial in (args.first, args.second):
        with nullcontext() if args.unscoped else activated_artifact_recipe(trial):
            runtime = load(trial / "artifact")
            try:
                # This exercises lazy descriptor generation, unlike import-only preflight.
                specs = runtime._kernel_launch_specs()
                assert len(specs) == 228
                assert len({spec["entry_symbol"] for spec in specs}) == 26
                print(json.dumps(dict(trial=str(trial.resolve()), semantic_hash=runtime.ir_module.semantic_hash,
                                      invocations=len(specs), GPU_initialized=torch.cuda.is_initialized())), flush=True)
            finally:
                runtime.close()
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
