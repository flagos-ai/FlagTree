"""Build a fresh native GGUF decode artifact with FlagMega's C++ backend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from download_checkpoint import FILENAME, REVISION, SHA256

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def source_identity():
    # Include recipes, typed plan companions, ownership contracts and licenses.
    paths = {p for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".cpp", ".h")} | {REPO / "LICENSE"}
    suffixes = {".py", ".h", ".cpp", ".jinja", ".json", ".md", ".txt"}
    for root in (REPO / "python/triton/flagmega", HERE / "local_optimizations",
                 HERE / "reproduction"):
        paths.update(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes
                     and "__pycache__" not in p.parts and ".local" not in p.relative_to(REPO).parts)
    files = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--rdata-cache-dir", type=Path, default=HERE / ".local/rdata-cache")
    parser.add_argument("--input", type=Path, help="Trusted complete Python IR or dump directory")
    parser.add_argument("--layer", type=int, help="Support probe only; default is all 64 layers")
    parser.add_argument("--recipe", type=Path, help="JSON arguments for local_optimizations.install")
    parser.add_argument("--placement", help="Optional physical placement, e.g. yx=12x8/bb")
    parser.add_argument("--cpp-max-registers", type=int)
    parser.add_argument("--cpp-operator-profile", choices=("cta-globaltimer-v1", "cta-globaltimer-v2"))
    parser.add_argument("--schedule", choices=("baseline275", "candidate49"),
                        help="Reconstruct the accepted complete ordinary schedule from original GGUF")
    args = parser.parse_args()
    if args.schedule:
        from normal_compile import validate_options
        try:
            validate_options(args)
        except ValueError as error:
            parser.error(str(error))
    checkpoint, trial = args.checkpoint.resolve(), args.trial.resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".gguf":
        parser.error("--checkpoint requires an original GGUF file")
    if args.input and args.layer is not None:
        parser.error("Resumed IR already fixes layer scope")
    if trial.exists():
        parser.error("Use a fresh trial directory; previous attempts are immutable")
    recipe = json.loads(args.recipe.read_text()) if args.recipe else None
    if recipe is not None and not isinstance(recipe, dict):
        parser.error("Recipe must be a JSON object")
    trial.mkdir(parents=True)
    (trial / "logs").mkdir()
    identity, sources = source_identity()
    (trial / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    for source in sources:
        target = trial / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / source, target)
    if recipe is not None:
        (trial / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "python"), str(HERE)])
    env.pop("U4_PRIVATE_FLAGMEGA_ROOT", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    bootstrap = ("import json; from local_optimizations import install; "
                 f"install(**json.loads({json.dumps(recipe)!r})); "
                 "from triton.flagmega.cli import main; raise SystemExit(main())")
    if args.schedule:
        bootstrap = ("from normal_compile import register_recipe; "
                     f"register_recipe({str(args.recipe.resolve())!r}); "
                     "from triton.flagmega.cli import main; raise SystemExit(main())")
    prefix = [sys.executable, "-c", bootstrap] if recipe is not None else [sys.executable, "-m", "triton.flagmega"]
    started = time.perf_counter()
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
              "source_identity": identity, "target": "nvidia-sm120", "codegen_backend": "cpp",
              "recipe": recipe, "placement": args.placement, "layer": args.layer,
              "resumed_from": str(args.input) if args.input else None, "schedule": args.schedule, "commands": []}

    def save():
        (trial / "build.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, argv, command_prefix=None):
        command = (prefix if command_prefix is None else command_prefix) + argv
        print(f"[{name}] {trial / 'logs' / (name + '.log')}", flush=True)
        before = time.perf_counter()
        with (trial / "logs" / (name + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        report["commands"].append({"name": name, "argv": command, "exit": result.returncode,
                                   "seconds": time.perf_counter() - before})
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed: {trial / 'logs' / (name + '.log')}")

    save()
    try:
        if args.schedule:
            before = checkpoint.stat()
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
            after = checkpoint.stat()
            identity_fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                              value.st_mtime_ns, value.st_ctime_ns)
            if identity_fields(before) != identity_fields(after) or digest.hexdigest() != SHA256:
                raise ValueError("Original GGUF checksum or file identity differs from the pinned checkpoint")
            report["checkpoint_sha256"] = digest.hexdigest()
            report["checkpoint_identity"] = identity_fields(after)
            save()
        imported = trial / "imported.py"
        if args.input:
            from triton.flagmega.ir import emit_module, load_module
            if recipe is not None:
                from local_optimizations import install
                install(**recipe)
            emit_module(load_module(args.input), imported)
        else:
            scope = ["--full-model"] if args.layer is None else ["--layer", str(args.layer)]
            revision = ["--revision", REVISION] if args.schedule or checkpoint.name == FILENAME else []
            run("import", ["import", "--model", str(checkpoint), "--output", str(imported),
                           "--mode", "decode-1", "--numerical-profile", "nncase", *scope, *revision, "--json"])
        run("verify-ir", ["verify", str(imported), "--json"])
        run("inspect-ir", ["inspect", str(imported), "--json"])
        options = []
        for flag, value in (("--placement", args.placement), ("--cpp-max-registers", args.cpp_max_registers),
                            ("--cpp-operator-profile", args.cpp_operator_profile)):
            if value is not None:
                options.extend((flag, str(value)))
        artifact = trial / "artifact"
        if args.schedule:
            run("compile", ["--imported", str(imported), "--checkpoint", str(checkpoint),
                            "--output", str(artifact), "--work-dir", str(trial / "dumps"),
                            "--rdata-cache-dir", str(args.rdata_cache_dir.resolve()),
                            "--recipe", str(args.recipe.resolve()), "--schedule", args.schedule],
                command_prefix=[sys.executable, "-m", "normal_compile"])
        else:
            run("compile", ["compile", "--input", str(imported), "--checkpoint", str(checkpoint),
                            "--target", "nvidia-sm120", "--codegen-backend", "cpp", "--emit-executable",
                            "--bufferize-opt-level", "fast", "--dump-flags", "compile,pass-ir,egraph-cost",
                            "--work-dir", str(trial / "dumps"), "--output", str(artifact),
                            "--rdata-cache-dir", str(args.rdata_cache_dir.resolve()), *options, "--json"])
        run("verify-artifact", ["artifact", "verify", str(artifact), "--json"])
        if source_identity()[0] != identity:
            raise RuntimeError("Build sources changed during compilation; retain this attempt and rebuild")
        report["artifact_pins"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact.iterdir()
                                   if p.suffix in (".json", ".cpp", ".so")}
        if args.schedule and tuple(report["checkpoint_identity"]) != identity_fields(checkpoint.stat()):
            raise RuntimeError("Original GGUF changed during reconstruction")
        report["status"] = "verified"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        save()


if __name__ == "__main__":
    main()
