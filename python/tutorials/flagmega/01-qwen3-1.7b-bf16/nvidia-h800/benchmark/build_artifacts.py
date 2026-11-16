#!/usr/bin/env python3
"""Build full Qwen3-1.7B with the current FlagMega Triton/TLE compiler.

Executable assets stay under .local. The exported artifact view contains only
IR and generated source. Identity records stay under .local outside artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parents[4]
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PROFILE = "vllm-493bd8323-inductor-level3"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_identity():
    roots = (REPO / "python/triton/flagmega", REPO / "python/triton/experimental/tle",
             REPO / "python/flagtree", REPO / "third_party/tle",
             REPO / "third_party/nvidia/backend")
    paths = [Path(__file__), REPO / "python/triton/_C/libtriton.so"]
    for root in roots:
        paths.extend(path for path in root.rglob("*") if path.is_file()
                     and path.suffix in (".py", ".jinja", ".cpp", ".h", ".td")
                     and "__pycache__" not in path.parts)
    files = {str(path.relative_to(REPO)): sha(path) for path in sorted(set(paths))}
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": files}


def export_sources(source_root, destination, records):
    """Copy genuine emitted sources to the tracked IR/kernel view.

    ``.script`` is the compiler's readable distributed/TIR companion. The
    workspace display uses ``.il`` for that same byte content. Runtime manifests,
    binary payloads and weights remain in the executable build directory.
    """
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".il", ".script"):
            continue
        if "assets" in path.relative_to(source_root).parts:
            continue
        relative = path.relative_to(source_root)
        if destination.name == 'generated' and (path.suffix != '.py' or 'ir' in relative.parts):
            continue
        if relative.suffix == ".script":
            relative = relative.with_suffix(".il")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append({"source": str(path), "destination": str(target), "sha256": sha(path),
                        "content_transformation": "none"})


CHECKPOINT_STAGES = (
    ("01-after-target-independent", "00_TargetIndependentPass/After"),
    ("02-vectorization-candidates", "01_AutoVectorizePass/00_AutoVectorize/After"),
    ("03-after-vectorize", "01_AutoVectorizePass/After"),
    ("04-after-packing", "02_AutoPackingPass/After"),
    ("05-after-distribution", "03_AutoDistributedPass/After"),
    ("06-fused-norm", "04_TIRPass/00_FuseNormStatsApply/After"),
    ("07-selected-tir", "04_TIRPass/07_LowerSelectedTIR/After"),
    ("08-selected-microkernels", "04_TIRPass/12_SelectTIRMicroKernels/After"),
    ("09-memory-placed", "04_TIRPass/14_PlanFunctionMemory/After"),
    ("10-bufferized", "04_TIRPass/15_Bufferize/After"),
    ("11-scheduled", "04_TIRPass/16_MaterializeExecutionFunctions/After"),
    ("12-synchronized", "04_TIRPass/17_PlanMemorySynchronization/After"),
    ("13-after-tir", "04_TIRPass/After"),
)


def export_checkpoint_stages(build, destination):
    """Export byte-identical real compiler dumps and their recorded identities."""
    index_path = build / "dumps/dumps.json"
    index = json.loads(index_path.read_text())
    records = []
    for display, directory in CHECKPOINT_STAGES:
        entries = [row for row in index if Path(row["relative_path"]).parent == Path(directory)]
        if not entries or {row["function"] for row in entries} != {"main", "decode_layer"}:
            raise ValueError(f"Missing complete actual function set at {directory}")
        for row in entries:
            for name in (row["relative_path"], row["companion_path"]):
                source = build / "dumps" / name
                target = destination / "ir" / display / Path(name).name
                if target.suffix == ".script":
                    target = target.with_suffix(".il")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                records.append({"path": str(target.relative_to(destination)), "sha256": sha(source),
                                "size_bytes": source.stat().st_size, "stage_info": {
                                    **row, "pass_boundary": directory, "chronological_index": display.split("-", 1)[0],
                                    "dump_source": str(source.relative_to(WORK)), "dump_index_sha256": sha(index_path),
                                    "content_transformation": "none", "display_suffix": target.suffix}})
    return records


def export_frozen_build(build, base, destination):
    """Packaging-only export: preserve the existing full-model build identity."""
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((base / "manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.parts[:2] == ('generated', 'ir'):
            continue  # The same genuine final IR is exported under ir.
        if relative.parts[0] == 'native-ir':
            relative = Path('ir/native') / Path(*relative.parts[1:])
        if relative.parts[0] not in ('generated', 'ir'):
            raise ValueError(f"Not generated code or IR: {row['path']}")
        source, target = base / row["path"], destination / relative
        if sha(source) != row["sha256"]:
            raise ValueError(f"Frozen source-view changed: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({**row, 'path': str(relative)})
    rows.extend(export_checkpoint_stages(build, destination))
    manifest.update({"directory": str(destination.relative_to(WORK)), "files": rows,
                     "export_created_at_utc": datetime.now(timezone.utc).isoformat(),
                     "export_operation": "package existing real compiler dumps; no compilation or semantic transformation",
                     "exporter_sha256": sha(Path(__file__)),
                     "base_source_view": {"path": str((base / "manifest.json").relative_to(WORK)), "sha256": sha(base / "manifest.json")},
                     "dump_index": {"path": str((build / "dumps/dumps.json").relative_to(WORK)), "sha256": sha(build / "dumps/dumps.json")}})
    output = destination.parent / (destination.name + "-manifest.json")
    write_json(output, manifest)
    return output


def checkpoint_path(requested, download):
    if requested is not None:
        result = requested.expanduser().resolve()
    else:
        result = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B"
                  / "snapshots" / REVISION)
    if not (result / "config.json").is_file() and download:
        from huggingface_hub import snapshot_download
        result = Path(snapshot_download("Qwen/Qwen3-1.7B", revision=REVISION,
                                       allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    if not (result / "model.safetensors.index.json").is_file():
        raise ValueError(f"Complete Qwen checkpoint missing: {result}; pass --download or --checkpoint")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=WORK / ".local/baseline/build/b000")
    parser.add_argument("--artifacts-dir", type=Path, default=WORK / ".local/baseline/artifacts")
    parser.add_argument("--cache-dir", type=Path, default=WORK / ".local/cache/weights")
    parser.add_argument("--stop-after", help="Explicit compiler boundary for diagnosis; no executable emitted")
    parser.add_argument("--export-only-build", type=Path, help="Package actual existing dumps without compiling")
    parser.add_argument("--base-source-view", type=Path, help="Immutable source view for packaging-only export")
    args = parser.parse_args(argv)
    if args.export_only_build:
        if not args.base_source_view:
            parser.error("--export-only-build requires --base-source-view")
        for path in (args.export_only_build, args.base_source_view, args.artifacts_dir):
            if not path.resolve().is_relative_to(WORK / ".local"):
                parser.error("Frozen build/export paths must remain under workspace .local")
        print(export_frozen_build(args.export_only_build.resolve(), args.base_source_view.resolve(), args.artifacts_dir.resolve()))
        return 0
    # A complete source checkout is the binding, including the FlagTree host
    # namespace absent from older editable-install package maps.
    sys.path.insert(0, str(REPO / "python"))
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.diagnostics import DumpFlags
    from triton.flagmega.importer import DirectoryCheckpoint, import_model
    from triton.flagmega.ir import emit_module, verify_module
    from triton.flagmega.options import CompileOptions

    checkpoint = checkpoint_path(args.checkpoint, args.download)
    build, artifacts, cache = (path.expanduser().resolve()
                               for path in (args.build_dir, args.artifacts_dir, args.cache_dir))
    for path in (build, artifacts, cache):
        if not path.is_relative_to(WORK / ".local"):
            raise ValueError("Build, provisional artifacts and weights must remain under this workspace .local")
    build.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = source_identity()
    write_json(build / "source-before.json", sources)
    # Hash the acquired checkpoint once and retain the exact local acquisition
    # identity. Packing reuses immutable readonly caches on subsequent builds.
    weights = {path.name: {"sha256": sha(path), "size_bytes": path.stat().st_size}
               for path in sorted(checkpoint.iterdir()) if path.is_file()
               and (path.suffix == ".safetensors" or path.name.endswith(".json"))}
    write_json(cache / "checkpoint.json", {"model": "Qwen/Qwen3-1.7B", "revision": REVISION,
                                          "path": str(checkpoint), "files": weights})
    source = DirectoryCheckpoint(checkpoint)
    imported = import_model(source, revision=REVISION, numerical_profile=PROFILE)
    verify_module(imported)
    emit_module(imported, build / "ir/imported.py")
    print(f"Imported all layers: {imported.metadata}; compiling normal target pipeline", flush=True)
    compiler = Compiler(CompileOptions(target="nvidia-sm90", codegen_backend="triton",
                        work_dir=build / "dumps", dump_flags=DumpFlags.COMPILE | DumpFlags.PASS_IR))
    compiled = compiler.compile(imported, stop_after=args.stop_after).module
    emit_module(compiled, build / "ir/final.py")
    exported = []
    export_sources(build / "ir", artifacts / "ir", exported)
    checkpoint_records = export_checkpoint_stages(build, artifacts) if not args.stop_after else []
    if args.stop_after:
        write_json(build / "paused.json", {"stage": compiled.stage, "boundary": args.stop_after})
        return 0
    runtime_artifact = write_artifact(compiled, build / "runtime-artifact", target=compiler.target.name,
                                     checkpoint=source, emit_executable=True, codegen_backend="triton",
                                     rdata_cache_dir=cache / "packed")
    export_sources(runtime_artifact, artifacts / "generated", exported)
    write_json(build / "source-view-map.json", {"files": exported, "compiler_checkpoints": checkpoint_records})
    after = source_identity()
    write_json(build / "source-after.json", after)
    if sources["sha256"] != after["sha256"]:
        raise RuntimeError("Compiler sources changed during build; artifact remains diagnostic only")
    record = {"schema": "flagmega.baseline-build/v1", "agent_name": "graph optimizer",
              "agent_id": "/root/graph_optimizer", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "target": compiler.target.name, "codegen_backend": "triton+tle",
              "numerical_contract": PROFILE, "checkpoint": str(checkpoint), "revision": REVISION,
              "source_sha256": sources["sha256"], "semantic_hash": compiled.semantic_hash,
              "runtime_artifact": str(runtime_artifact), "artifacts": str(artifacts),
              "compile_wall_seconds": time.perf_counter() - started,
              "argv": sys.argv, "python": sys.executable}
    write_json(build / "build.json", record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
