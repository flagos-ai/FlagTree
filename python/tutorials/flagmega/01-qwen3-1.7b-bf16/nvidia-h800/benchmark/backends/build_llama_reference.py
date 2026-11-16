#!/usr/bin/env python3
"""Build an unmodified pinned official llama.cpp SM90 reference and BF16 GGUF.

All configure/build/conversion commands are CPU-only and retain logs. The CUDA
backend is compiled for the actual H800 architecture, with native CUDA graphs.
No quantization or upstream-source edits are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_model import sha, validate_checkpoint, write_new


COMMIT = "61dbf49228c8a96b72e40c89d93cb7c003411f8c"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--cuda-compiler", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--resume-interrupted-job", type=Path,
                        help="Retained prior job JSON; resumes its identical configured object tree")
    parser.add_argument("--interruption-audit", type=Path,
                        help="Owning-context proof that the prior process/handle is missing")
    args = parser.parse_args(argv)
    if args.receipt.exists() or args.gguf.exists():
        raise ValueError("Use fresh GGUF and receipt paths")
    if args.build.exists() and args.resume_interrupted_job is None:
        raise ValueError("Existing build requires explicit interrupted-job admission")
    source = args.source.resolve()
    def git(*values):
        return subprocess.check_output(["git", "-C", str(source), *values], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain"):
        raise ValueError("Require a clean unmodified official pinned source checkout")
    checkpoint = validate_checkpoint(args.checkpoint)
    resume = None
    if args.resume_interrupted_job is not None:
        if args.interruption_audit is None:
            raise ValueError("Incremental continuation requires actual owner-context interruption audit")
        old = json.loads(args.resume_interrupted_job.read_text())
        audit = json.loads(args.interruption_audit.read_text())
        if audit.get("prior_job") != old.get("job_id") or not audit.get("not_completed"):
            raise ValueError("Interruption audit does not cover the pinned prior job")
        for flag, expected in [("--source", args.source), ("--build", args.build),
                               ("--checkpoint", args.checkpoint), ("--gguf", args.gguf),
                               ("--cuda-compiler", args.cuda_compiler)]:
            actual = old["argv"][old["argv"].index(flag) + 1]
            if Path(actual).resolve() != expected.resolve():
                raise ValueError(f"Continuation changed the admitted {flag}")
        cache = {}
        for line in (args.build / "CMakeCache.txt").read_text().splitlines():
            if "=" in line and not line.startswith(("#", "//")):
                key, value = line.split("=", 1)
                cache[key.split(":", 1)[0]] = value
        required = {"CMAKE_HOME_DIRECTORY": str(source), "CMAKE_CUDA_ARCHITECTURES": "90",
                    "CMAKE_CUDA_COMPILER": str(args.cuda_compiler), "GGML_CUDA": "ON",
                    "GGML_CUDA_GRAPHS": "ON", "CMAKE_BUILD_TYPE": "Release"}
        for key, expected in required.items():
            if cache.get(key) != expected:
                raise ValueError(f"Continuation CMake identity mismatch {key}")
        resume = {"prior_job": {"path": str(args.resume_interrupted_job), "sha256": sha(args.resume_interrupted_job)},
                  "interruption_audit": {"path": str(args.interruption_audit), "sha256": sha(args.interruption_audit)},
                  "same_object_tree": str(args.build.resolve()), "cache_sha256_before": sha(args.build / "CMakeCache.txt"),
                  "mode": "incremental continuation; no clean-first or source/config downgrade"}
    if not 1 <= args.jobs <= 4:
        raise ValueError("Reference builds must use1..4 jobs for resource isolation")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    logs = args.receipt.with_suffix(".logs")
    logs.mkdir()
    report = {"schema": "flagmega.llama-reference-build/v1", "status": "building",
              "source": {"path": str(source), "commit": COMMIT, "dirty": False},
              "checkpoint": checkpoint, "script_sha256": sha(__file__),
              "resume": resume,
              "started_utc": datetime.now(timezone.utc).isoformat(), "commands": []}
    config = [str(args.cmake), "-S", str(source), "-B", str(args.build.resolve()), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DGGML_CUDA_GRAPHS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=90", f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler}",
        "-DLLAMA_BUILD_COMMON=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_MTMD=OFF"]
    build = [str(args.cmake), "--build", str(args.build.resolve()), "--target", "llama", "--parallel", str(args.jobs)]
    conversion = [sys.executable, str(source / "convert_hf_to_gguf.py"), str(args.checkpoint.resolve()),
                  "--outfile", str(args.gguf.resolve()), "--outtype", "bf16"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
    try:
        for name, command in [("configure", config), ("build", build), ("convert", conversion)]:
            started = datetime.now(timezone.utc).isoformat()
            with (logs / (name + ".log")).open("wb") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
            report["commands"].append({"step": name, "argv": command, "exit_code": result.returncode,
                "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "log": {"path": str((logs / (name + ".log")).resolve()), "sha256": sha(logs / (name + ".log"))}})
            if result.returncode:
                raise RuntimeError(f"Reference {name} failed, see retained exact log")
        if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain"):
            raise RuntimeError("Pinned reference source changed during build")
        report.update(status="built_cpu_only", ended_utc=datetime.now(timezone.utc).isoformat(),
            config={"gpu_architecture": "sm90", "cuda_graphs": True, "dtype": "BF16",
                    "reference_intermediates": "native llama.cpp", "quantization": None},
            binaries=[{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)}
                      for p in sorted(args.build.rglob("*.so*")) if p.is_file() and not p.is_symlink()],
            gguf={"path": str(args.gguf.resolve()), "bytes": args.gguf.stat().st_size, "sha256": sha(args.gguf)},
            performance="unmeasured; GPU runner and matched scope/state review pending")
    except BaseException as error:
        report.update(status="failed", error=str(error), ended_utc=datetime.now(timezone.utc).isoformat())
        write_new(args.receipt, report)
        raise
    write_new(args.receipt, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
