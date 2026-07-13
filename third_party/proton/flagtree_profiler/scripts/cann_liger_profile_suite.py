#!/usr/bin/env python3
"""Profile LinkedIn Liger-Kernel operators with Proton CANN.

This is the public-library full evaluation companion to
``cann_operator_profile_suite.py``. It intentionally uses Proton's direct
start/finalize flow: no user-visible external ``msprof python ...`` wrapper is
required.

Example:

    python third_party/proton/flagtree_profiler/scripts/cann_liger_profile_suite.py \
      --out /tmp/proton_cann_liger_full --clean
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import triton.profiler as proton
from triton._C.libproton import proton as libproton


@dataclass(frozen=True)
class LigerCase:
    name: str
    category: str
    description: str
    make: Callable


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/proton_cann_liger_full")
    parser.add_argument(
        "--liger-source",
        help="Path to a Liger-Kernel checkout. The script adds <path>/src to PYTHONPATH.",
    )
    parser.add_argument(
        "--clone-liger",
        action="store_true",
        help=
        "Clone Liger-Kernel into --liger-source, or <out>/Liger-Kernel when omitted. This is the default when --liger-source is not set.",
    )
    parser.add_argument(
        "--no-clone-liger",
        action="store_true",
        help=
        "Do not auto-clone Liger-Kernel when --liger-source is omitted; use an installed liger_kernel package instead.",
    )
    parser.add_argument(
        "--liger-repo",
        default="https://github.com/linkedin/Liger-Kernel.git",
        help="Repository used by --clone-liger.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case name to run. May be repeated. Default: all cases.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--allow-case-failures",
        action="store_true",
        help="Finalize and return success even if one or more Liger cases fail.",
    )
    return parser


def _run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _prepare_liger_import(args: argparse.Namespace, out: pathlib.Path) -> pathlib.Path | None:
    liger_source = pathlib.Path(args.liger_source) if args.liger_source else None
    should_clone = args.clone_liger or liger_source is None
    if args.no_clone_liger:
        should_clone = False
    if should_clone:
        if liger_source is None:
            liger_source = out / "Liger-Kernel"
        if not liger_source.exists():
            liger_source.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--depth", "1", args.liger_repo, str(liger_source)])

    if liger_source is None:
        return None

    src_dir = liger_source / "src"
    import_dir = src_dir if src_dir.exists() else liger_source
    if not import_dir.exists():
        raise RuntimeError(f"Liger source import path does not exist: {import_dir}")
    sys.path.insert(0, str(import_dir))
    return liger_source


def _load_torch_npu():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the Liger profile suite.") from exc

    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch_npu is required for NPU tensor allocation.") from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is unavailable after importing torch_npu.")
    if not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed, but torch.npu.is_available() is false.")
    return torch


def _load_liger(torch):
    # Liger currently uses transformers.utils.is_torch_npu_available() in its
    # device inference path. The low-level operator APIs do not otherwise need
    # transformers, so use torch.npu directly when transformers is not installed.
    os.environ.setdefault("LIGER_KERNEL_IMPL", "ascend")
    try:
        liger_utils = importlib.import_module("liger_kernel.utils")
    except ModuleNotFoundError as exc:
        raise RuntimeError("liger_kernel is not importable. Install liger-kernel or pass "
                           "--liger-source /path/to/Liger-Kernel.") from exc

    liger_utils.is_npu_available = lambda: hasattr(torch, "npu") and torch.npu.is_available()

    cross_entropy = importlib.import_module("liger_kernel.transformers.cross_entropy")
    dyt = importlib.import_module("liger_kernel.transformers.dyt")
    fused_add_rms_norm = importlib.import_module("liger_kernel.transformers.fused_add_rms_norm")
    fused_linear_cross_entropy = importlib.import_module("liger_kernel.transformers.fused_linear_cross_entropy")
    fused_linear_jsd = importlib.import_module("liger_kernel.transformers.fused_linear_jsd")
    geglu = importlib.import_module("liger_kernel.transformers.geglu")
    group_norm = importlib.import_module("liger_kernel.transformers.group_norm")
    jsd = importlib.import_module("liger_kernel.transformers.jsd")
    kl_div = importlib.import_module("liger_kernel.transformers.kl_div")
    layer_norm = importlib.import_module("liger_kernel.transformers.layer_norm")
    modulated_rms_norm = importlib.import_module("liger_kernel.transformers.modulated_rms_norm")
    poly_norm = importlib.import_module("liger_kernel.transformers.poly_norm")
    relu_squared = importlib.import_module("liger_kernel.transformers.relu_squared")
    rms_norm = importlib.import_module("liger_kernel.transformers.rms_norm")
    rope = importlib.import_module("liger_kernel.transformers.rope")
    softmax = importlib.import_module("liger_kernel.transformers.softmax")
    sparsemax = importlib.import_module("liger_kernel.transformers.sparsemax")
    swiglu = importlib.import_module("liger_kernel.transformers.swiglu")
    tvd = importlib.import_module("liger_kernel.transformers.tvd")

    return SimpleNamespace(
        LigerCrossEntropyLoss=cross_entropy.LigerCrossEntropyLoss,
        LigerDyT=dyt.LigerDyT,
        LigerFusedAddRMSNorm=fused_add_rms_norm.LigerFusedAddRMSNorm,
        LigerFusedLinearCrossEntropyLoss=fused_linear_cross_entropy.LigerFusedLinearCrossEntropyLoss,
        LigerFusedLinearJSD=fused_linear_jsd.LigerFusedLinearJSD,
        LigerGEGLUMLP=geglu.LigerGEGLUMLP,
        LigerGroupNorm=group_norm.LigerGroupNorm,
        LigerJSD=jsd.LigerJSD,
        LigerKLDIVLoss=kl_div.LigerKLDIVLoss,
        LigerLayerNorm=layer_norm.LigerLayerNorm,
        LigerModulatedRMSNorm=modulated_rms_norm.LigerModulatedRMSNorm,
        LigerPolyNorm=poly_norm.LigerPolyNorm,
        LigerRMSNorm=rms_norm.LigerRMSNorm,
        LigerReLUSquared=relu_squared.LigerReLUSquared,
        LigerSoftmax=softmax.LigerSoftmax,
        LigerSparsemax=sparsemax.LigerSparsemax,
        LigerSwiGLUMLP=swiglu.LigerSwiGLUMLP,
        LigerTVDLoss=tvd.LigerTVDLoss,
        liger_rotary_pos_emb=rope.liger_rotary_pos_emb,
    )


def _checksum(torch, out) -> float:
    if isinstance(out, (tuple, list)):
        return sum(_checksum(torch, item) for item in out)
    if hasattr(out, "loss"):
        out = out.loss
    if not hasattr(out, "float"):
        return 0.0
    return float(out.float().mean().cpu().item())


def _measure_op(torch, op: Callable, iters: int) -> tuple[float, object]:
    out = None
    start = time.perf_counter()
    for _ in range(iters):
        out = op()
    torch.npu.synchronize()
    return time.perf_counter() - start, out


def _timing_fields(baseline_elapsed_s: float, profiled_elapsed_s: float, iters: int) -> dict:
    overhead_s = profiled_elapsed_s - baseline_elapsed_s
    overhead_ratio = None
    overhead_percent = None
    if baseline_elapsed_s > 0:
        overhead_ratio = overhead_s / baseline_elapsed_s
        overhead_percent = overhead_ratio * 100.0
    return {
        "baseline_elapsed_s": baseline_elapsed_s,
        "profiled_elapsed_s": profiled_elapsed_s,
        "elapsed_s": profiled_elapsed_s,
        "overhead_s": overhead_s,
        "overhead_ratio": overhead_ratio,
        "overhead_percent": overhead_percent,
        "baseline_per_iter_s": baseline_elapsed_s / iters if iters else None,
        "profiled_per_iter_s": profiled_elapsed_s / iters if iters else None,
        "overhead_per_iter_s": overhead_s / iters if iters else None,
    }


def _summarize_timing(results: list[dict]) -> dict:
    timed = [
        result for result in results if result.get("status") == "ok" and result.get("baseline_elapsed_s") is not None
        and result.get("profiled_elapsed_s") is not None
    ]
    ratios = [result["overhead_ratio"] for result in timed if result.get("overhead_ratio") is not None]
    baseline_total = sum(result["baseline_elapsed_s"] for result in timed)
    profiled_total = sum(result["profiled_elapsed_s"] for result in timed)
    weighted_ratio = None
    if baseline_total > 0:
        weighted_ratio = (profiled_total - baseline_total) / baseline_total
    return {
        "timed_case_count": len(timed),
        "baseline_total_s": baseline_total,
        "profiled_total_s": profiled_total,
        "overhead_total_s": profiled_total - baseline_total,
        "average_overhead_ratio": sum(ratios) / len(ratios) if ratios else None,
        "average_overhead_percent": (sum(ratios) / len(ratios) * 100.0) if ratios else None,
        "weighted_overhead_ratio": weighted_ratio,
        "weighted_overhead_percent": weighted_ratio * 100.0 if weighted_ratio is not None else None,
    }


def _cases(torch, device, liger) -> list[LigerCase]:
    hidden = 128
    inter = 256
    vocab = 512
    tokens = 32
    cfg_silu = SimpleNamespace(hidden_size=hidden, intermediate_size=inter, hidden_act="silu")
    cfg_gelu = SimpleNamespace(hidden_size=hidden, intermediate_size=inter, hidden_act="gelu_pytorch_tanh")

    def rand(shape, dtype=torch.float32):
        return torch.randn(shape, device=device, dtype=dtype)

    def labels(n: int, v: int):
        return torch.randint(0, v, (n, ), device=device, dtype=torch.long)

    def prob(shape):
        return torch.softmax(rand(shape), dim=-1)

    def logprob(shape):
        return torch.log_softmax(rand(shape), dim=-1)

    def rms_norm():
        module = liger.LigerRMSNorm(hidden).to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def layer_norm():
        module = liger.LigerLayerNorm(hidden).to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def fused_add_rms_norm():
        module = liger.LigerFusedAddRMSNorm(hidden).to(device)
        x = rand((tokens, hidden))
        residual = rand((tokens, hidden))
        return lambda: module(x, residual)

    def modulated_rms_norm():
        module = liger.LigerModulatedRMSNorm(hidden).to(device)
        x = rand((tokens, hidden))
        scale = rand((tokens, hidden))
        shift = rand((tokens, hidden))
        return lambda: module(x, scale, shift)

    def poly_norm():
        module = liger.LigerPolyNorm().to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def dyt():
        module = liger.LigerDyT(hidden).to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def relu_squared():
        module = liger.LigerReLUSquared().to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def swiglu_mlp():
        module = liger.LigerSwiGLUMLP(cfg_silu).to(device)
        x = rand((2, 16, hidden))
        return lambda: module(x)

    def geglu_mlp():
        module = liger.LigerGEGLUMLP(cfg_gelu).to(device)
        x = rand((2, 16, hidden))
        return lambda: module(x)

    def softmax():
        module = liger.LigerSoftmax().to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def sparsemax():
        module = liger.LigerSparsemax(dim=-1).to(device)
        x = rand((tokens, hidden))
        return lambda: module(x)

    def rope():
        bsz = 2
        heads = 4
        seq = 16
        head_dim = 64
        q = rand((bsz, heads, seq, head_dim))
        k = rand((bsz, heads, seq, head_dim))
        angles = rand((1, seq, head_dim))
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return lambda: liger.liger_rotary_pos_emb(q, k, cos, sin)

    def cross_entropy():
        module = liger.LigerCrossEntropyLoss().to(device)
        x = rand((tokens, vocab))
        target = labels(tokens, vocab)
        return lambda: module(x, target)

    def fused_linear_cross_entropy():
        module = liger.LigerFusedLinearCrossEntropyLoss().to(device)
        x = rand((tokens, hidden))
        weight = rand((vocab, hidden))
        target = labels(tokens, vocab)
        return lambda: module(weight, x, target)

    def kl_div():
        module = liger.LigerKLDIVLoss(reduction="batchmean").to(device)
        y_pred = logprob((tokens, vocab))
        y_true = prob((tokens, vocab))
        return lambda: module(y_pred, y_true)

    def jsd():
        module = liger.LigerJSD(beta=0.5).to(device)
        log_q = logprob((tokens, vocab))
        log_p = logprob((tokens, vocab))
        target = labels(tokens, vocab)
        return lambda: module(log_q, log_p, target)

    def tvd():
        module = liger.LigerTVDLoss(reduction="batchmean").to(device)
        p = prob((tokens, vocab))
        q = prob((tokens, vocab))
        target = labels(tokens, vocab)
        return lambda: module(p, q, target)

    def group_norm():
        module = liger.LigerGroupNorm(num_channels=16, num_groups=4).to(device)
        x = rand((4, 16, 8, 8))
        return lambda: module(x)

    def fused_linear_jsd():
        module = liger.LigerFusedLinearJSD(jsd_beta=0.5).to(device)
        student_input = rand((tokens, hidden))
        student_weight = rand((vocab, hidden))
        teacher_input = rand((tokens, hidden))
        teacher_weight = rand((vocab, hidden))
        target = labels(tokens, vocab)
        return lambda: module(student_input, student_weight, teacher_input, teacher_weight, target)

    return [
        LigerCase("liger_rms_norm", "normalization", "LigerRMSNorm", rms_norm),
        LigerCase("liger_layer_norm", "normalization", "LigerLayerNorm", layer_norm),
        LigerCase("liger_fused_add_rms_norm", "normalization", "LigerFusedAddRMSNorm", fused_add_rms_norm),
        LigerCase("liger_modulated_rms_norm", "normalization", "LigerModulatedRMSNorm", modulated_rms_norm),
        LigerCase("liger_poly_norm", "normalization", "LigerPolyNorm", poly_norm),
        LigerCase("liger_dyt", "activation", "LigerDyT", dyt),
        LigerCase("liger_relu_squared", "activation", "LigerReLUSquared", relu_squared),
        LigerCase("liger_swiglu_mlp", "mlp", "LigerSwiGLUMLP", swiglu_mlp),
        LigerCase("liger_geglu_mlp", "mlp", "LigerGEGLUMLP", geglu_mlp),
        LigerCase("liger_softmax", "probability", "LigerSoftmax", softmax),
        LigerCase("liger_sparsemax", "probability", "LigerSparsemax", sparsemax),
        LigerCase("liger_rope", "position_embedding", "liger_rotary_pos_emb", rope),
        LigerCase("liger_cross_entropy", "loss", "LigerCrossEntropyLoss", cross_entropy),
        LigerCase(
            "liger_fused_linear_cross_entropy",
            "loss",
            "LigerFusedLinearCrossEntropyLoss",
            fused_linear_cross_entropy,
        ),
        LigerCase("liger_kl_div", "loss", "LigerKLDIVLoss", kl_div),
        LigerCase("liger_jsd", "loss", "LigerJSD", jsd),
        LigerCase("liger_tvd", "loss", "LigerTVDLoss", tvd),
        LigerCase("liger_group_norm", "normalization", "LigerGroupNorm", group_norm),
        LigerCase("liger_fused_linear_jsd", "loss", "LigerFusedLinearJSD", fused_linear_jsd),
    ]


def _summarize_artifacts(base: pathlib.Path, out: pathlib.Path, results: list[dict], failures: list[dict]) -> dict:
    meta_path = base.with_suffix(".meta.json")
    vendor_path = base.with_suffix(".vendor.json")
    timeline_path = base.with_suffix(".timeline.json")

    meta = json.loads(meta_path.read_text())
    vendor = json.loads(vendor_path.read_text())
    timeline = json.loads(timeline_path.read_text().splitlines()[0])

    source_counts = Counter(assoc.get("source", "") for assoc in vendor.get("associations", []))
    op_types = Counter(
        assoc.get("metrics", {}).get("op_type")
        for assoc in vendor.get("associations", [])
        if assoc.get("metrics", {}).get("op_type"))
    bandwidth_count = sum(1 for assoc in vendor.get("associations", []) if "bandwidth_gb_s" in assoc.get("metrics", {}))
    mstx_messages = sorted({
        assoc.get("metrics", {}).get("message")
        for assoc in vendor.get("associations", [])
        if assoc.get("source") == "msprof_mstx" and assoc.get("metrics", {}).get("message")
    })

    return {
        "profile_base": str(base),
        "meta_json": str(meta_path),
        "vendor_json": str(vendor_path),
        "timeline_json": str(timeline_path),
        "summary_json": str(out / "summary.json"),
        "backend": meta.get("backend"),
        "config": {
            key: meta.get("config", {}).get(key)
            for key in [
                "aclprof_runtime_enabled",
                "aclprof_auto_export",
                "aclprof_output_path",
                "mstx_enabled",
                "mstx_domain",
            ]
        },
        "case_count": len(results),
        "ok_count": sum(1 for r in results if r.get("status") == "ok"),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
        "raw_input_count": len(vendor.get("raw_inputs", [])),
        "association_count": len(vendor.get("associations", [])),
        "association_sources": dict(source_counts),
        "bandwidth_association_count": bandwidth_count,
        "mstx_range_count": len(mstx_messages),
        "mstx_ranges": mstx_messages,
        "top_op_types": dict(op_types.most_common(40)),
        "timeline_event_count": len(timeline.get("traceEvents", [])),
        "degrade_reasons": meta.get("degrade_reasons", []),
    }


def main() -> int:
    args = _make_arg_parser().parse_args()
    out = pathlib.Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    msprof_out = out / "msprof"
    msprof_out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    os.chmod(msprof_out, 0o700)

    liger_source = _prepare_liger_import(args, out)
    torch = _load_torch_npu()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    liger = _load_liger(torch)

    cases = _cases(torch, device, liger)
    case_by_name = {case.name: case for case in cases}
    selected_names = args.cases or [case.name for case in cases]
    unknown = sorted(set(selected_names) - set(case_by_name))
    if unknown:
        raise RuntimeError(f"Unknown Liger case(s): {', '.join(unknown)}")

    prepared = []
    results = []
    failures = []
    for name in selected_names:
        case = case_by_name[name]
        try:
            op = case.make()
            for _ in range(args.warmup):
                op()
            torch.npu.synchronize()
            baseline_elapsed_s, _ = _measure_op(torch, op, args.iters)
            prepared.append((case, op, baseline_elapsed_s))
        except Exception as exc:
            failure = {"name": case.name, "phase": "baseline", "error": repr(exc)}
            failures.append(failure)
            results.append({
                "name": case.name,
                "category": case.category,
                "description": case.description,
                "status": "failed",
                "phase": "baseline",
                "error": repr(exc),
            })
            if not args.allow_case_failures:
                break

    base = out / "liger_full_profile"
    mode = ("runtime_base:"
            "vendor_metrics=aicore,bandwidth:"
            f"aclprof_output_path={msprof_out}:"
            "runtime_host_timing_fallback=true:"
            "aclprof_runtime_enabled=true:"
            "aclprof_auto_export=true:"
            "mstx_enabled=true:"
            "mstx_domain=proton")
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=mode,
    )

    try:
        for case, op, baseline_elapsed_s in prepared:
            scope_name = f"proton_cann_liger::{case.name}"
            try:
                scope_id = libproton.record_scope()
                out_value = None
                libproton.enter_op(scope_id, scope_name)
                try:
                    profiled_elapsed_s, out_value = _measure_op(torch, op, args.iters)
                finally:
                    libproton.exit_op(scope_id, scope_name)

                result = {
                    "name": case.name,
                    "scope": scope_name,
                    "category": case.category,
                    "description": case.description,
                    "iters": args.iters,
                    "checksum": _checksum(torch, out_value),
                    "status": "ok",
                }
                result.update(_timing_fields(baseline_elapsed_s, profiled_elapsed_s, args.iters))
                results.append(result)
            except Exception as exc:
                failure = {"name": case.name, "phase": "profiled", "error": repr(exc)}
                failures.append(failure)
                results.append({
                    "name": case.name,
                    "category": case.category,
                    "description": case.description,
                    "status": "failed",
                    "phase": "profiled",
                    "error": repr(exc),
                })
                if not args.allow_case_failures:
                    break
    finally:
        proton.finalize(session_id)

    summary = _summarize_artifacts(base, out, results, failures)
    summary["liger_source"] = str(liger_source) if liger_source else "installed-package"
    summary["timing"] = _summarize_timing(results)
    summary["overhead_method"] = "same_process_pre_session_no_profiler_baseline"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))

    if failures and not args.allow_case_failures:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
