"""Build an immutable default Triton decode-1 trial from the pinned original GGUF.

All command logs and receipts go through workspace.py run. The default scope is
the complete 64-layer text decoder; --layer is an explicitly labelled probe.
No optimization recipes, saved physical IR, or backend substitutions are used.
"""
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
WORKSPACE = REPO / "skills/flagmega-developer/scripts/workspace.py"
CHECKPOINT_SHA256 = "ea5a3c45d407f9b9e5d2c0d647f0ea600f486f6b86b92b56d0823ba073dae675"
CHECKPOINT_BYTES = 14534384640
REVISION = "5a285a6217024a96be7064d6caf0a06121630ee1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_identity():
    suffixes = {".py", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".c", ".inc", ".cmake",
                ".jinja", ".json", ".md", ".txt", ".td"}
    paths = {REPO / "LICENSE", WORKSPACE, Path(__file__).resolve(),
             HERE / "benchmark/build_artifacts.py", REPO / "python/triton/_C/libtriton.so",
             REPO / "CMakeLists.txt", REPO / "setup.py", REPO / "python/build_helpers.py"}
    for root in (REPO / "python/flagtree", REPO / "python/triton/flagmega", REPO / "python/triton/backends",
                 REPO / "python/triton/compiler", REPO / "python/triton/runtime",
                 REPO / "python/triton/language", REPO / "python/triton/experimental/tle",
                 REPO / "python/src", REPO / "include", REPO / "lib", REPO / "cmake",
                 REPO / "third_party/tle", REPO / "third_party/nvidia"):
        paths.update(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes
                     and "__pycache__" not in p.parts)
    paths.update(p for p in (REPO / "python/triton").glob("*.py"))
    files = {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--rdata-cache-dir", type=Path, default=HERE / ".local/cache/weights")
    parser.add_argument("--layer", type=int, help="Backend support probe only; default: full64")
    parser.add_argument("--candidate", choices=["r001-v2.1"], help="Install the signed E32 IQ4 configuration (r000 uses its frozen source snapshot)")
    parser.add_argument("--row-tile", type=int, choices=[4], default=4)
    parser.add_argument("--kv-row-tile", type=int, choices=[4], default=4)
    parser.add_argument("--resume-packing", type=Path, help="Complete unused_functions_removed dump directory")
    parser.add_argument("--graph-only", action="store_true", help="Verify graph stages; no executable or native fallback")
    args = parser.parse_args(argv)
    if args.candidate:
        from agent_optimizations.iq4_streaming.build import run
        return run(args)
    if args.resume_packing or args.graph_only or args.row_tile != 4 or args.kv_row_tile != 4:
        parser.error("Candidate configuration requires --candidate r001-v2.1")
    checkpoint, trial = args.checkpoint.resolve(), args.trial.resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".gguf":
        parser.error("--checkpoint requires the pinned original GGUF")
    if trial.exists():
        parser.error("Use a fresh trial directory; previous attempts are immutable")
    if not trial.is_relative_to(HERE / ".local"):
        parser.error("Trial weights, caches and binaries must remain under WORK_DIR/.local")
    if args.layer is not None and not 0 <= args.layer < 64:
        parser.error("Only text decoder layers 0..63 may be probed")
    trial.mkdir(parents=True, exist_ok=False)
    (trial / "jobs").mkdir()
    start = time.perf_counter()
    identity, sources = source_identity()
    write_json(trial / "sources.json", {"source_identity": identity, "files": sources})
    for relative in sources:
        # The native library is pinned by bytes, not copied or rebuilt.
        if relative.endswith(".so"):
            continue
        destination = trial / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    report = dict(schema="flagmega.baseline-build/v1", status="running",
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                  target="nvidia-sm120", codegen_backend="triton", mode="decode-1",
                  scope="full64-text-decoder" if args.layer is None else "layer-probe",
                  layer=args.layer, source_identity=identity, compiler_options="defaults",
                  rdata_cache_dir=str(args.rdata_cache_dir.resolve()), commands=[])
    prefix = [sys.executable, "-m", "triton.flagmega"]
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), PYTHONDONTWRITEBYTECODE="1")
    env.pop("U4_PRIVATE_FLAGMEGA_ROOT", None)
    actor = os.environ.get("FLAGMEGA_AGENT_ID", "codex:/root:graph_optimizer:build_baseline")
    job_prefix = "baseline-" + trial.name

    def save():
        write_json(trial / "build.json", report)

    def run(name, arguments):
        job_id = job_prefix + "-" + name
        job = dict(schema="flagmega.job/v1", job_id=job_id, agent_name="graph optimizer",
                   agent_id=actor, cwd=".", argv=["/usr/bin/env", "PYTHONPATH=" + env["PYTHONPATH"],
                        "PYTHONDONTWRITEBYTECODE=1", *prefix, *arguments],
                   input_identities={"source_identity": identity, "checkpoint_sha256": CHECKPOINT_SHA256})
        job_path = trial / "jobs" / (name + ".json")
        write_json(job_path, job)
        print(json.dumps({"command": job["argv"], "job": str(job_path)}), flush=True)
        before = time.perf_counter()
        result = subprocess.run([sys.executable, str(WORKSPACE), "run", "--work-dir", str(HERE),
                                 "--agent", "graph_optimizer", "--job", str(job_path)], env=env)
        receipt_path = HERE / ".local/jobs" / job_id / "receipt.json"
        command = dict(name=name, argv=job["argv"], seconds=time.perf_counter() - before,
                       wrapper_exit=result.returncode, receipt=str(receipt_path))
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            command.update(exit=receipt["exit_code"], receipt_sha256=sha(receipt_path))
        report["commands"].append(command)
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; retained stdout/stderr: {receipt_path.parent}")

    save()
    try:
        before = checkpoint.stat()
        actual = sha(checkpoint)
        after = checkpoint.stat()
        fields = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if fields(before) != fields(after) or actual != CHECKPOINT_SHA256 or after.st_size != CHECKPOINT_BYTES:
            raise ValueError("Original GGUF identity differs from the frozen checkpoint")
        report.update(checkpoint_sha256=actual, checkpoint_file_identity=fields(after))
        save()
        imported = trial / "imported.py"
        scope = ["--full-model"] if args.layer is None else ["--layer", str(args.layer)]
        run("import", ["import", "--model", str(checkpoint), "--revision", REVISION,
                       "--mode", "decode-1", "--numerical-profile", "nncase", *scope,
                       "--output", str(imported), "--json"])
        run("verify-ir", ["verify", str(imported), "--expect-stage", "imported", "--json"])
        run("inspect-ir", ["inspect", str(imported), "--json"])
        artifact = trial / "artifact"
        run("compile", ["compile", "--input", str(imported), "--checkpoint", str(checkpoint),
                        "--target", "nvidia-sm120", "--codegen-backend", "triton", "--emit-executable",
                        "--dump-flags", "compile,pass-ir", "--work-dir", str(trial / "dumps"),
                        "--output", str(artifact), "--rdata-cache-dir", str(args.rdata_cache_dir.resolve()), "--json"])
        run("verify-artifact", ["artifact", "verify", str(artifact), "--json"])
        if source_identity()[0] != identity:
            raise RuntimeError("Build sources changed during compilation; preserve and rebuild this attempt")
        if fields(checkpoint.stat()) != fields(after):
            raise RuntimeError("Checkpoint changed during compilation")
        report["status"] = "built-and-verified"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        # Pin actual outputs, including partial dumps from a failed build.
        outputs = {}
        for root in (trial / "dumps", trial / "artifact"):
            if root.exists():
                outputs.update({str(p.relative_to(trial)): dict(sha256=sha(p), bytes=p.stat().st_size)
                                for p in sorted(root.rglob("*")) if p.is_file()})
        for p in (trial / "imported.py", trial / "imported.il"):
            if p.is_file():
                outputs[str(p.relative_to(trial))] = dict(sha256=sha(p), bytes=p.stat().st_size)
        write_json(trial / "outputs.json", outputs)
        report.update(seconds=time.perf_counter() - start,
                      ended_utc=datetime.now(timezone.utc).isoformat(), outputs_sha256=sha(trial / "outputs.json"))
        save()


if __name__ == "__main__":
    main()
