"""Normal full64 compiler binding, with immutable attempts and explicit pins."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from collections.abc import Mapping

from .graph import PLAN_HASH, StreamingConfig, configure_distribution_target, configure_target

WORK = Path(__file__).resolve().parents[2]
REPO = WORK.parents[4]
REV = WORK / ".local/milestones/m000/revisions/r001"
BASELINE = WORK / ".local/milestones/m000/revisions/r000/attempts/candidate-r4-fresh-20261116T1023"
PEER_RECEIPTS = {
    "graph_expert": REV / "agents/graph_expert/plan_r001_v2/retry-completion/handoff-v2.1.json",
    "kernel_expert": WORK / ".local/agents/kernel_expert/sign_r001_v21/handoff.json",
    "graph_optimizer": WORK / ".local/agents/graph_optimizer/sign_r001_v2_1/handoff.json",
    "kernel_optimizer": WORK / ".local/agents/kernel_optimizer/sign_r001_v21/handoff.json",
    "reviewer": WORK / ".local/agents/reviewer/sign_r001_v21/handoff.json",
}
CHECKPOINT_SHA256 = "ea5a3c45d407f9b9e5d2c0d647f0ea600f486f6b86b92b56d0823ba073dae675"
CHECKPOINT_BYTES = 14534384640
MODEL_REVISION = "5a285a6217024a96be7064d6caf0a06121630ee1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "nbytes": path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def source_identity():
    # Reproduction includes local installer/templates and the actual harness.
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("baseline_build_binding", WORK / "compile.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, files = module.source_identity()
    for root in (WORK / "agent_optimizations", WORK / "benchmark"):
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".jinja", ".h", ".cpp", ".json", ".txt"}:
                files[str(p.relative_to(REPO))] = sha(p)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return identity, files


def verify_gate():
    plan = (REV / "plan.md").read_bytes()
    body = plan.split(b"## 4. Agent signatures", 1)[0]
    if hashlib.sha256(body).hexdigest() != PLAN_HASH:
        raise RuntimeError("r001 v2.1 frozen plan body differs")
    release_path = WORK / ".local/agents/orchestrator/sign_r001_v21/handoff.json"
    release = json.loads(release_path.read_text())
    assert release["decision"] == "approve" and release["flow_release"]["released"]
    assert release["signed_subject"]["body_sha256"] == PLAN_HASH
    peers = []
    assert {p["agent"] for p in release["verified_peer_approvals"]} == set(PEER_RECEIPTS)
    for peer in release["verified_peer_approvals"]:
        receipt_path = PEER_RECEIPTS[peer["agent"]]
        receipt = json.loads(receipt_path.read_text())
        signatures = receipt.get("signoffs", ())
        if not any(s.get("plan_hash") == PLAN_HASH and s.get("decision") == "approve"
                   for s in signatures):
            raise RuntimeError(f"Peer signature does not approve r001 v2.1: {receipt_path}")
        peers.append({"agent": peer["agent"], **pin(receipt_path)})
    assert len(peers) == 5
    return {"plan_hash": PLAN_HASH, "release": pin(release_path), "peers": peers}


def verify_packing_input(directory):
    """Only reuse the exact accepted full-packing function set, never later IR."""
    directory = Path(directory).resolve()
    expected = BASELINE / "dumps/Compile/02_AutoPackingPass/After"
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if {p.relative_to(directory) for p in files} != {p.relative_to(expected) for p in expected.rglob("*") if p.is_file()}:
        raise ValueError("Resume requires the complete accepted full-packing function set")
    frozen = json.loads((BASELINE / "outputs.json").read_text())
    pins = []
    for path in files:
        relative = path.relative_to(directory)
        recorded = frozen[str(Path("dumps/Compile/02_AutoPackingPass/After") / relative)]
        digest = recorded if isinstance(recorded, str) else recorded["sha256"]
        if sha(path) != digest:
            raise ValueError(f"Packing fragment differs from accepted r000: {relative}")
        pins.append(pin(path))
    return pins


def graph_checks(module):
    from triton.flagmega.passes.constants import ConstnessAnalysis
    from triton.flagmega.passes.functions import static_function_invocation_counts, static_node_invocation_counts
    from .graph import is_iq4, scalar_rows
    constant_ids = ConstnessAnalysis.analyze(module).constants
    ops = {"tensors.pack", "tensors.slice", "tensors.cast", "ntt.vectorized_cast",
           "tensors.concat", "tensors.unpack"}
    node_counts = static_node_invocation_counts(module)
    adapters = [{"id": n.id, "op": n.op, "invocations": node_counts[n.id]}
                for n in module.nodes if node_counts.get(n.id, 0) and n.id not in constant_ids and n.op in ops]
    projections = [{"id": n.id, "op": n.op, "rows": scalar_rows(n), "type": str(n.type),
                    "metadata": dict(n.metadata), "invocations": node_counts.get(n.id, 0)}
                   for n in module.nodes if node_counts.get(n.id, 0) and is_iq4(n)]
    invocations = static_function_invocation_counts(module)
    return {"stage": module.stage, "module_semantic_hash": module.semantic_hash,
            "functions": [f.name for f in module.functions], "invocations": dict(invocations),
            "runtime_adapters": adapters, "G3_passed": not adapters, "projections": projections,
            "projection_sites": len(projections),
            "projection_invocations": sum(p["invocations"] for p in projections),
            "tuple_inventory": [{"id": n.id, "type": str(n.type)} for n in module.nodes
                                if node_counts.get(n.id, 0) and n.op in {"builtin.tuple", "builtin.get_item"}]}


def run(args):
    from triton.flagmega import Compiler, CompileOptions
    from triton.flagmega.diagnostics import DumpFlags
    from triton.flagmega.importer import import_model, checkpoint_from_path
    from triton.flagmega.ir import load_module, emit_module, verify_module
    from triton.flagmega.artifacts import write_artifact, load_artifact
    import ortools
    if ortools.__version__ != "9.10.4067":
        raise RuntimeError("Use the repository ortools pin 9.10.4067")
    if args.layer is not None:
        raise ValueError("r001 candidate always builds all64 layers")
    trial = args.trial.resolve()
    if trial.exists() or not trial.is_relative_to(WORK / ".local"):
        raise ValueError("Candidate needs a fresh WORK_DIR/.local attempt")
    trial.mkdir(parents=True)
    start = time.perf_counter()
    report = {"schema": "flagmega.candidate-build/v1", "status": "running", "runnable": False,
              "plan_hash": PLAN_HASH, "source_identity": None, "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": StreamingConfig(args.row_tile, args.kv_row_tile).to_data(), "scope": "full64-text-decoder",
              "target": "nvidia-sm120", "codegen_backend": "triton", "NUM": "pending", "review": "pending",
              "H001-static": "pending", "H001-preserve": "pending", "measurement": "pending",
              "performance_accepted": False, "milestone_achieved": False}
    try:
        report["gate"] = verify_gate()
        identity, files = source_identity()
        report["source_identity"] = identity
        save(trial / "sources.json", {"source_identity": identity, "files": files})
        for relative in files:
            if relative.endswith(".so"):
                continue
            destination = trial / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, destination)
        save(trial / "build.json", report)
        checkpoint = args.checkpoint.resolve()
        if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Original GGUF differs from the frozen checkpoint")
        report["checkpoint"] = pin(checkpoint)
        options = CompileOptions(target="nvidia-sm120", codegen_backend="triton", work_dir=trial / "dumps",
                                 dump_flags=DumpFlags.COMPILE | DumpFlags.PASS_IR)
        compiler = Compiler(options)
        config = StreamingConfig(args.row_tile, args.kv_row_tile)
        compiler.target = (configure_distribution_target(compiler.target, config) if args.graph_only
                           else configure_target(compiler.target, config))
        if args.resume_packing is None:
            source = import_model(checkpoint, revision=MODEL_REVISION, mode="decode-1", num_tokens=1,
                                  numerical_profile="nncase")
            emit_module(source, trial / "imported.py")
            packed = compiler.compile(source, stop_after="unused_functions_removed").module
            emit_module(packed, trial / "packing.py")
            report["lineage"] = "fresh-original-GGUF"
        else:
            # Directory loading merges every genuine function fragment.
            report["packing_input"] = verify_packing_input(args.resume_packing)
            packed = load_module(args.resume_packing)
            verify_module(packed, expected_stage="unused_functions_removed")
            report["lineage"] = "verified-complete-packing-resume"
            emit_module(packed, trial / "packing.py")
        assert packed.metadata["num_hidden_layers"] == 64
        assert tuple(packed.metadata["imported_layer_indices"]) == tuple(range(64))
        report["packing_checks"] = graph_checks(packed)
        assert (report["packing_checks"]["G3_passed"] and report["packing_checks"]["projection_sites"] == 12
                and report["packing_checks"]["projection_invocations"] == 337)
        result = compiler.compile(packed, stop_after="distributed_boundary_layout_propagated").module
        emit_module(result, trial / "distribution.py")
        report["distribution_checks"] = graph_checks(result)
        assert (report["distribution_checks"]["G3_passed"] and report["distribution_checks"]["projection_sites"] == 12
                and report["distribution_checks"]["projection_invocations"] == 337)
        if not args.graph_only:
            result = compiler.compile(result).module
            artifact = write_artifact(result, trial / "artifact", target="nvidia-sm120",
                checkpoint=checkpoint_from_path(checkpoint), emit_executable=True, codegen_backend="triton",
                dumper=compiler.diagnostics.dumper, rdata_cache_dir=args.rdata_cache_dir)
            manifest, loaded = load_artifact(artifact)
            assert loaded.semantic_hash == result.semantic_hash and manifest["codegen_backend"] == "triton"
            report.update(artifact=pin(artifact / "artifact.json"), artifact_semantic_hash=loaded.semantic_hash,
                          executable_package=True, artifact_verified=True,
                          runnable=False, status="exported-native-unqualified")
            save(trial / "native-interface.json", {
                "plan_hash": PLAN_HASH, "config": config.to_data(),
                "implementation_model": plain(compiler.target.triton_implementation_model.snapshot()),
                "implementations": [{"id": i.id, "family": i.family, "variant": i.variant,
                    "parameters": plain(i.parameters), "facts": plain(i.facts)}
                    for i in compiler.target.triton_implementation_model.implementations
                    if i.family.startswith("iq4_streaming")],
                "static_lowering_qualification": "pending; interface admission alone is not evidence"})
            save(trial / "operator-map.json", {
                "source_identity": identity, "artifact": report["artifact"],
                "calls": manifest["codegen"]["kernels"],
                "program_execution": manifest["codegen"]["program_execution"],
                "program_sources": manifest["codegen"]["program_sources"],
                "resources": None, "NUM": None})
        else:
            report.update(status="graph-stages-verified", native="pending")
        if source_identity()[0] != identity:
            raise RuntimeError("Sources changed during build; preserve this attempt and rebuild")
    except BaseException as error:
        report.update(status="failed", runnable=False, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.perf_counter() - start, ended_utc=datetime.now(timezone.utc).isoformat())
        save(trial / "build.json", report)
        outputs = {str(p.relative_to(trial)): pin(p) for root in (trial / "dumps", trial / "artifact")
                   if root.exists() for p in sorted(root.rglob("*")) if p.is_file()}
        outputs.update({p.name: pin(p) for p in trial.iterdir() if p.is_file()
                        and p.suffix in {".py", ".il"}})
        save(trial / "outputs.json", outputs)
        print(json.dumps(plain(report), indent=2), flush=True)
