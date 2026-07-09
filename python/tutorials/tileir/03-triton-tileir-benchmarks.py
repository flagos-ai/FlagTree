"""
Triton TileIR Benchmarks
========================

Run the selected ported benchmark kernels on both native NVIDIA and TileIR
paths.  The benchmark is self-contained: the original Triton operator
implementations and autotune configs used by these cases are bundled as the
local ``tilegym`` package next to this script.  Timings use CUPTI CUDA kernel
self time through ``torch.profiler``.
"""

import argparse
import gc
import importlib
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import traceback
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import torch  # noqa: E402

CASES = ("bmm", "fmha", "linear_bias_act", "mla", "mla_decoding", "matmul", "rope")
DEFAULT_CACHE_DIR = pathlib.Path("/tmp/flagtree_tileir_tutorial_cache/03")


@dataclass(frozen=True)
class PairSpec:
    case: str
    test_class: str
    dtype: str
    pair_id: str
    function_name: str

    @property
    def signature(self) -> str:
        return f"{self.test_class} / {self.function_name}"


@dataclass
class BuiltCase:
    run: Callable[[], object]
    ref: Optional[Callable[[], object]]
    meta: dict
    check: bool = True
    atol: float = 1e-2
    rtol: float = 1e-2


def _add_pair(out: List[PairSpec], case: str, test_class: str, dtype: str, function_name: str):
    idx = sum(1 for item in out if item.case == case)
    out.append(PairSpec(case, test_class, dtype, f"{case}-{idx:02d}", function_name))


def _build_pair_specs() -> Tuple[PairSpec, ...]:
    specs: List[PairSpec] = []

    for batch in (2, 8):
        for dim in (2048, 4096, 8192):
            for trans_a in (False, True):
                for trans_b in (False, True):
                    _add_pair(
                        specs,
                        "bmm",
                        "Test_BMM_FWD",
                        "float16",
                        f"test_perf[FRAMEWORK-{batch}-{dim}-{dim}-{dim}-{trans_a}-{trans_b}-torch.float16]",
                    )

    for is_causal in (False, True):
        for seq_len in (1024, 1025, 2048, 2049, 4096, 4097, 512, 8192):
            _add_pair(
                specs,
                "fmha",
                "Test_FMHA",
                "float8_e5m2",
                f"test_perf[FRAMEWORK-{is_causal}-4-32-{seq_len}-128-torch.float8_e5m2]",
            )
    for seq_len in (31072, 9):
        _add_pair(
            specs,
            "fmha",
            "Test_FMHA",
            "float8_e5m2",
            f"test_perf_llm[FRAMEWORK-torch.float8_e5m2-llama-1-32-{seq_len}-128]",
        )

    for dim in (1024, 2048, 4096, 8192):
        for act in ("None", "gelu", "relu"):
            _add_pair(
                specs,
                "linear_bias_act",
                "Test_LinearBiasAct",
                "float16",
                f"test_perf[FRAMEWORK-{dim}-{dim}-{dim}-{act}-torch.float16-True]",
            )

    for seq_len in (1024, 2048, 4096, 512):
        _add_pair(
            specs,
            "mla",
            "Test_MLA",
            "float8_e5m2",
            f"test_perf[FRAMEWORK-True-4-32-{seq_len}-128-64-torch.float8_e5m2]",
        )

    for kv_len in (1024, 2048, 8192):
        for batch in (1, 1024, 128):
            for num_heads, transpose in ((128, False), (16, True), (32, True), (64, False)):
                _add_pair(
                    specs,
                    "mla_decoding",
                    "Test_MLADecoding",
                    "float16",
                    f"test_perf[FRAMEWORK-{kv_len}-{batch}-512-64-float16-{num_heads}-{transpose}]",
                )

    matmul_sizes = (16384, 2048, 256, 4096, 64, 8192)
    for trans_b in (False, True):
        for trans_a in (False, True):
            for dim in matmul_sizes:
                _add_pair(
                    specs,
                    "matmul",
                    "Test_Matmul",
                    "float32",
                    f"test_perf[FRAMEWORK-True-False-{trans_b}-{trans_a}-{dim}-{dim}-{dim}-0-0-torch.float32]",
                )
    for trans_b in (False, True):
        for dim in matmul_sizes:
            _add_pair(
                specs,
                "matmul",
                "Test_Matmul",
                "float32",
                f"test_perf[FRAMEWORK-True-True-{trans_b}-False-{dim}-{dim}-{dim}-0-0-torch.float32]",
            )

    rope_configs = ((32, 32, 128, 1.0), (32, 8, 128, 0.5), (32, 8, 128, 1.0), (32, 8, 256, 0.25))
    for seq_len in (1, 128, 65536):
        for num_q_heads, num_kv_heads, head_dim, partial in rope_configs:
            _add_pair(
                specs,
                "rope",
                "Test_RoPE",
                "float16",
                f"test_perf[FRAMEWORK-torch.float16-8-{seq_len}-{num_q_heads}-{num_kv_heads}-{head_dim}-{partial}]",
            )

    return tuple(specs)


PAIR_SPECS = _build_pair_specs()
PAIR_BY_ID: Dict[str, PairSpec] = {spec.pair_id: spec for spec in PAIR_SPECS}
PAIRS_BY_CASE = {case: tuple(spec for spec in PAIR_SPECS if spec.case == case) for case in CASES}
CI_PAIR_IDS = {
    "bmm": ("bmm-00", "bmm-04", "bmm-08"),
    "fmha": ("fmha-06", "fmha-00", "fmha-08"),
    "linear_bias_act": ("linear_bias_act-00", "linear_bias_act-04", "linear_bias_act-08"),
    "mla": ("mla-03", "mla-00", "mla-01"),
    "mla_decoding": ("mla_decoding-00", "mla_decoding-10", "mla_decoding-20"),
    "matmul": ("matmul-04", "matmul-20", "matmul-32"),
    "rope": ("rope-01", "rope-05", "rope-09"),
}


def _representative_specs(cases: Sequence[str]) -> List[PairSpec]:
    return [PAIR_BY_ID[pair_id] for case in cases for pair_id in CI_PAIR_IDS[case]]


def _set_path(path: str) -> None:
    if path == "tileir":
        os.environ["FLAGTREE_USE_TILEIR"] = "1"
    else:
        os.environ.pop("FLAGTREE_USE_TILEIR", None)
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")


def _path_name() -> str:
    return "tileir" if os.environ.get("FLAGTREE_USE_TILEIR", "0") == "1" else "native"


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this tutorial requires a CUDA GPU")


def _dtype(token: str) -> torch.dtype:
    name = token.replace("torch.", "")
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    float8_e5m2 = getattr(torch, "float8_e5m2", None)
    float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    if float8_e5m2 is not None:
        mapping["float8_e5m2"] = float8_e5m2
    if float8_e4m3fn is not None:
        mapping["float8_e4m3fn"] = float8_e4m3fn
    if name not in mapping:
        raise RuntimeError(f"unsupported dtype token: {token}")
    return mapping[name]


def _bool(token: str) -> bool:
    if token == "True":
        return True
    if token == "False":
        return False
    raise ValueError(f"not a bool token: {token}")


def _maybe_none(token: str):
    return None if token == "None" else token


def _params(function_name: str) -> List[str]:
    """Parse bracketed pytest-style parameters.

    Tokens are separated by top-level ``-`` characters. The selected benchmark
    names only use non-negative numeric values, booleans, dtypes, and simple
    strings, so parameter values themselves must not contain ``-``.
    """
    start = function_name.index("[") + 1
    end = function_name.rindex("]")
    body = function_name[start:end]
    if body.startswith("FRAMEWORK-"):
        body = body[len("FRAMEWORK-"):]
    out: List[str] = []
    token: List[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "-" and depth == 0:
            out.append("".join(token))
            token = []
        else:
            token.append(ch)
    out.append("".join(token))
    return out


def _meta(x):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""), "device": str(x.device)}
    if isinstance(x, (list, tuple)):
        return [_meta(v) for v in x]
    if isinstance(x, dict):
        return {k: _meta(v) for k, v in x.items()}
    return repr(x)


def _seed_from_name(name: str) -> None:
    seed = zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF
    torch.manual_seed(seed)
    random.seed(seed)


def _randn(shape, dtype, scale=1.0):
    if dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return (torch.randn(shape, device="cuda", dtype=torch.float16) * scale).to(dtype)
    return torch.randn(shape, device="cuda", dtype=dtype) * scale


def _rand(shape, dtype):
    if dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return torch.rand(shape, device="cuda", dtype=torch.float16).to(dtype)
    return torch.rand(shape, device="cuda", dtype=dtype)


def _make_tensor_descriptor_hashable() -> None:
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except Exception:
        return
    if getattr(TensorDescriptor, "__hash__", None) is None:
        TensorDescriptor.__hash__ = object.__hash__


def _import_triton_op(module: str):
    mod = importlib.import_module(f"tilegym.ops.triton.{module}")
    _make_tensor_descriptor_hashable()
    return mod


def _patch_empty_prune_fallback(autotuner) -> None:
    if getattr(autotuner, "_flagtree_empty_prune_fallback", False):
        return
    original = getattr(autotuner, "early_config_prune", None)
    if original is None:
        return

    def prune(configs, named_args, **kwargs):
        pruned = original(configs, named_args, **kwargs)
        if pruned:
            return pruned
        return configs[:1]

    autotuner.early_config_prune = prune
    autotuner._flagtree_empty_prune_fallback = True


def _patch_matmul_empty_prune_fallback(mod) -> None:
    if _path_name() != "tileir":
        return
    _patch_empty_prune_fallback(mod._matmul_kernel)
    _patch_empty_prune_fallback(mod._static_persistent_matmul_kernel)


def _assert_close(name: str, actual, expected, atol: float, rtol: float) -> float:
    actual_items = [actual] if isinstance(actual, torch.Tensor) else list(actual)
    expected_items = [expected] if isinstance(expected, torch.Tensor) else list(expected)
    if len(actual_items) != len(expected_items):
        raise AssertionError(f"{name}: output arity mismatch {len(actual_items)} != {len(expected_items)}")
    max_abs = 0.0
    for idx, (a, e) in enumerate(zip(actual_items, expected_items)):
        diff = (a.float() - e.float()).abs()
        item_max = float(diff.max().item())
        max_abs = max(max_abs, item_max)
        if not torch.allclose(a.float(), e.float(), atol=atol, rtol=rtol, equal_nan=True):
            raise AssertionError(f"{name}: output {idx} mismatch max_abs={item_max:.6g}")
    return max_abs


def _estimate_ms(fn: Callable[[], object], initial_rep: int) -> float:
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(initial_rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return max(start_event.elapsed_time(end_event) / initial_rep, 1.0e-6)


def _bench_ms_cupti(fn: Callable[[], object], warmup: float, rep: float, min_rep: int, initial_rep: int) -> float:
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    estimate_ms = _estimate_ms(fn, initial_rep)
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(min_rep, int(rep / estimate_ms))

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    run_times_us = []
    for _ in range(n_repeat):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        total_us = sum(evt.self_device_time_total for evt in prof.key_averages() if evt.self_device_time_total > 0)
        if total_us == 0:
            raise RuntimeError("CUPTI returned 0 device time")
        run_times_us.append(total_us)

    times = torch.tensor(run_times_us, dtype=torch.float64) / 1000.0
    return times.mean().item()


def _bench_ms(
    fn: Callable[[], object],
    warmup: float,
    rep: float,
    min_rep: int,
    initial_rep: int,
) -> Tuple[float, str]:
    return _bench_ms_cupti(fn, warmup, rep, min_rep, initial_rep), "cupti"


def _reference_bmm(a, b, transpose_a=False, transpose_b=False):
    aa = a.transpose(-1, -2) if transpose_a else a
    bb = b.transpose(-1, -2) if transpose_b else b
    return torch.bmm(aa, bb)


def _build_bmm(fn: str) -> BuiltCase:
    q, m, n, k, ta, tb, dtype_s = _params(fn)
    q, m, n, k = int(q), int(m), int(n), int(k)
    ta, tb = _bool(ta), _bool(tb)
    dtype = _dtype(dtype_s)
    mod = _import_triton_op("bmm")
    a_shape = (q, k, m) if ta else (q, m, k)
    b_shape = (q, n, k) if tb else (q, k, n)
    a = _rand(a_shape, dtype)
    b = _rand(b_shape, dtype)
    return BuiltCase(
        run=lambda: mod.bmm_memref(a, b, transpose_a=ta, transpose_b=tb, static_persistent=True),
        ref=lambda: _reference_bmm(a, b, ta, tb),
        meta={"a": _meta(a), "b": _meta(b), "transpose_a": ta, "transpose_b": tb},
        atol=5e-2,
        rtol=5e-2,
    )


def _build_fmha(fn: str) -> BuiltCase:
    params = _params(fn)
    if fn.startswith("test_perf_llm["):
        dtype_s, _model, batch, heads, seq_len, head_dim = params
        is_causal = True
    else:
        is_causal_s, batch, heads, seq_len, head_dim, dtype_s = params
        is_causal = _bool(is_causal_s)
    batch, heads, seq_len, head_dim = int(batch), int(heads), int(seq_len), int(head_dim)
    dtype = _dtype(dtype_s)
    mod = _import_triton_op("attention")
    q = _randn((batch, heads, seq_len, head_dim), dtype)
    k = _randn((batch, heads, seq_len, head_dim), dtype)
    v = _randn((batch, heads, seq_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)
    check = dtype not in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    ref = None if not check else lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=scale)
    return BuiltCase(
        run=lambda: mod.triton_fmha(q, k, v, scaling=scale, is_causal=is_causal),
        ref=ref,
        meta={"q": _meta(q), "is_causal": is_causal},
        check=check,
        atol=8e-2,
        rtol=8e-2,
    )


def _build_linear_bias_act(fn: str) -> BuiltCase:
    m, n, k, act_type, dtype_s, is_bias = _params(fn)
    m, n, k = int(m), int(n), int(k)
    act_type = _maybe_none(act_type)
    dtype = _dtype(dtype_s)
    is_bias = _bool(is_bias)
    mod = _import_triton_op("linear_bias_activation")
    x = _rand((m, k), dtype)
    weight = _rand((n, k), dtype)
    bias = _rand((n, ), dtype) if is_bias else None

    def ref():
        out = torch.nn.functional.linear(x, weight, bias)
        if act_type == "relu":
            return torch.relu(out)
        if act_type == "gelu":
            return torch.nn.functional.gelu(out, approximate="tanh")
        return out

    return BuiltCase(
        run=lambda: mod.linear_bias_act_dropout(x, weight, bias, act_type),
        ref=ref,
        meta={"x": _meta(x), "weight": _meta(weight), "act_type": act_type, "is_bias": is_bias},
        atol=8e-2,
        rtol=8e-2,
    )


def _build_mla(fn: str) -> BuiltCase:
    is_causal, batch, heads, seq_len, d_model, d_pe, dtype_s = _params(fn)
    is_causal = _bool(is_causal)
    batch, heads, seq_len, d_model, d_pe = map(int, (batch, heads, seq_len, d_model, d_pe))
    dtype = _dtype(dtype_s)
    mod = _import_triton_op("mla")
    q = _randn((batch, heads, seq_len, d_model), dtype, scale=0.3)
    k = _randn((batch, heads, seq_len, d_model), dtype, scale=0.3)
    v = _randn((batch, heads, seq_len, d_model), dtype, scale=0.3)
    qpe = _randn((batch, heads, seq_len, d_pe), dtype, scale=0.3)
    kpe = _randn((batch, 1, seq_len, d_pe), dtype, scale=0.3)
    scale = 1.0 / math.sqrt(d_model + d_pe)
    return BuiltCase(
        run=lambda: mod.triton_mla(q, k, v, qpe, kpe, is_causal, scale),
        ref=None,
        meta={"q": _meta(q), "qpe": _meta(qpe), "is_causal": is_causal},
        check=False,
    )


def _build_mla_decoding(fn: str) -> BuiltCase:
    s_kv, num_batch, block_d, block_kpe, dtype_s, num_heads, transpose = _params(fn)
    s_kv, num_batch, block_d, block_kpe, num_heads = map(int, (s_kv, num_batch, block_d, block_kpe, num_heads))
    dtype = _dtype(dtype_s)
    transpose = _bool(transpose)
    mod = _import_triton_op("mla_decoding")
    q = torch.ones((num_batch, num_heads, block_d), device="cuda", dtype=torch.float32).to(dtype)
    qpe = torch.ones((num_batch, num_heads, block_kpe), device="cuda", dtype=torch.float32).to(dtype)
    kv = torch.ones((num_batch, s_kv, block_d), device="cuda", dtype=torch.float32).to(dtype)
    kpe = torch.ones((num_batch, s_kv, block_kpe), device="cuda", dtype=torch.float32).to(dtype)
    scale = 1.0 / math.sqrt(block_d + block_kpe)
    return BuiltCase(
        run=lambda: mod.mla_decoding(q, qpe, kv, kpe, scale, transpose=transpose),
        ref=None,
        meta={"q": _meta(q), "kv": _meta(kv), "transpose": transpose},
        check=False,
    )


def _build_matmul(fn: str) -> BuiltCase:
    use_tma, static_persistent, tb, ta, m, n, k, _offset_a, _offset_b, dtype_s = _params(fn)
    use_tma, static_persistent, ta, tb = map(_bool, (use_tma, static_persistent, ta, tb))
    m, n, k = int(m), int(n), int(k)
    dtype = _dtype(dtype_s)
    mod = _import_triton_op("matmul")
    _patch_matmul_empty_prune_fallback(mod)
    a = _rand((k, m) if ta else (m, k), dtype)
    b = _rand((n, k) if tb else (k, n), dtype)
    check = dtype not in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    return BuiltCase(
        run=lambda: mod.matmul(a, b, trans_a=ta, trans_b=tb, static_persistent=static_persistent, use_tma=use_tma),
        ref=None if not check else lambda: (a.t() if ta else a) @ (b.t() if tb else b),
        meta={"a": _meta(a), "b": _meta(b), "use_tma": use_tma, "static_persistent": static_persistent},
        check=check,
        atol=8e-2,
        rtol=8e-2,
    )


def _build_rope(fn: str) -> BuiltCase:
    dtype_s, bsz, seq_len, num_q_heads, num_kv_heads, head_dim, partial = _params(fn)
    dtype = _dtype(dtype_s)
    bsz, seq_len, num_q_heads, num_kv_heads, head_dim = map(int, (bsz, seq_len, num_q_heads, num_kv_heads, head_dim))
    partial = float(partial)
    mod = _import_triton_op("rope")
    q = _randn((bsz, seq_len, num_q_heads, head_dim), dtype).transpose(1, 2)
    k = _randn((bsz, seq_len, num_kv_heads, head_dim), dtype).transpose(1, 2)
    rope_dim = int(head_dim * partial)
    inv_freq = 1.0 / (10000.0**(torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32) / rope_dim))
    freqs = torch.outer(torch.arange(seq_len, device="cuda", dtype=torch.float32), inv_freq)
    cos_half = torch.cos(freqs).to(dtype)
    sin_half = torch.sin(freqs).to(dtype)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(0).expand(bsz, -1, -1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(0).expand(bsz, -1, -1)
    return BuiltCase(
        run=lambda: mod.apply_rope_base(q, k, cos, sin, partial_rotary_factor=partial),
        ref=None,
        meta={"q": _meta(q), "k": _meta(k), "partial": partial},
        check=False,
    )


BUILDERS = {
    "Test_BMM_FWD": _build_bmm,
    "Test_FMHA": _build_fmha,
    "Test_LinearBiasAct": _build_linear_bias_act,
    "Test_MLA": _build_mla,
    "Test_MLADecoding": _build_mla_decoding,
    "Test_Matmul": _build_matmul,
    "Test_RoPE": _build_rope,
}


def _build_case(spec: PairSpec) -> BuiltCase:
    return BUILDERS[spec.test_class](spec.function_name)


def _cache_route(cache_dir: pathlib.Path) -> str:
    files = [path.name for path in cache_dir.glob("**/*") if path.is_file()]
    if any(name.endswith(".tileir") for name in files):
        return "tileir"
    if any(name.endswith((".ttgir", ".ptx", ".cubin")) for name in files):
        return "native"
    return "unknown"


def _group_specs_by_case(specs: Sequence[PairSpec]) -> List[Tuple[str, List[PairSpec]]]:
    grouped: Dict[str, List[PairSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.case, []).append(spec)
    return list(grouped.items())


def run_child(
    pair_id: str,
    path: str,
    warmup: float,
    rep: float,
    min_rep: int,
    initial_rep: int,
    check: bool,
) -> int:
    _set_path(path)
    os.environ.setdefault("TILEGYM_DISABLE_AUTOTUNE", "1")
    _require_cuda()
    spec = PAIR_BY_ID[pair_id]
    row = {
        "pair_id": spec.pair_id,
        "case": spec.case,
        "path": _path_name(),
        "signature": spec.signature,
        "status": "FAIL",
    }
    try:
        _seed_from_name(spec.function_name)
        built = _build_case(spec)
        out = built.run()
        torch.cuda.synchronize()
        validation = "runtime"
        max_abs = None
        if check and built.check and built.ref is not None:
            expected = built.ref()
            torch.cuda.synchronize()
            max_abs = _assert_close(spec.function_name, out, expected, built.atol, built.rtol)
            validation = "torch"
        bench_ms, timing = _bench_ms(
            built.run,
            warmup,
            rep,
            min_rep,
            initial_rep,
        )
        row.update({
            "status": "PASS",
            "ms": bench_ms,
            "timing": timing,
            "validation": validation,
            "max_abs": max_abs,
            "meta": built.meta,
        })
    except Exception as exc:
        row["message"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=8)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps(row, sort_keys=True))
    return 0 if row["status"] == "PASS" else 1


def _run_subprocess(
    script: pathlib.Path,
    spec: PairSpec,
    path: str,
    warmup: float,
    rep: float,
    min_rep: int,
    initial_rep: int,
    check: bool,
    cache_root: pathlib.Path,
) -> dict:
    env = os.environ.copy()
    env.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    env.setdefault("TILEGYM_DISABLE_AUTOTUNE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    cache_dir = cache_root / spec.case / spec.pair_id / path
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    if path == "tileir":
        env["FLAGTREE_USE_TILEIR"] = "1"
    else:
        env.pop("FLAGTREE_USE_TILEIR", None)
    cmd = [
        sys.executable,
        str(script),
        "--child",
        "--pair",
        spec.pair_id,
        "--path",
        path,
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--min-rep",
        str(min_rep),
        "--initial-rep",
        str(initial_rep),
    ]
    if not check:
        cmd.append("--no-check")
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    row = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            row = json.loads(line)
            break
    if row is None:
        row = {
            "pair_id": spec.pair_id,
            "case": spec.case,
            "path": path,
            "status": "FAIL",
            "message": "child did not print JSON",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    if proc.returncode != 0 and row.get("status") == "PASS":
        row["status"] = "FAIL"
        row["message"] = f"child exited with {proc.returncode}"
    row["route"] = _cache_route(cache_dir)
    row["cache_dir"] = str(cache_dir)
    if proc.stderr:
        row.setdefault("stderr", proc.stderr)
    return row


def _summarize(rows: Sequence[dict], specs: Sequence[PairSpec], paths: Sequence[str], cache_root: pathlib.Path) -> dict:
    passed = sum(1 for row in rows if row.get("status") == "PASS")
    failed = len(rows) - passed
    by_pair = {spec.pair_id: {} for spec in specs}
    for row in rows:
        by_pair[row["pair_id"]][row["path"]] = row
    summary = {
        "pairs": len(specs),
        "paths": list(paths),
        "runs": len(rows),
        "passed": passed,
        "failed": failed,
        "cache_root": str(cache_root),
        "cases": {},
    }
    for case, case_specs in _group_specs_by_case(specs):
        case_rows = [row for row in rows if row["pair_id"] in {spec.pair_id for spec in case_specs}]
        case_summary = {
            "pairs":
            len(case_specs),
            "runs":
            len(case_rows),
            "passed":
            sum(1 for row in case_rows if row.get("status") == "PASS"),
            "failed":
            sum(1 for row in case_rows if row.get("status") != "PASS"),
            "passed_pairs":
            sum(1 for spec in case_specs
                if by_pair[spec.pair_id] and all(row.get("status") == "PASS"
                                                 for row in by_pair[spec.pair_id].values())),
        }
        if set(paths) == {"native", "tileir"}:
            case_ratios = []
            for spec in case_specs:
                native = by_pair[spec.pair_id].get("native")
                tileir = by_pair[spec.pair_id].get("tileir")
                if native and tileir and native.get("status") == "PASS" and tileir.get("status") == "PASS":
                    case_ratios.append(native["ms"] / tileir["ms"])
            if case_ratios:
                case_summary.update({
                    "mean_native_over_tileir": sum(case_ratios) / len(case_ratios),
                    "min_native_over_tileir": min(case_ratios),
                    "max_native_over_tileir": max(case_ratios),
                    "tileir_faster": sum(1 for value in case_ratios if value > 1.0),
                    "speedup_pairs": len(case_ratios),
                })
        summary["cases"][case] = case_summary
    if set(paths) == {"native", "tileir"}:
        ratios = []
        for pair_rows in by_pair.values():
            native = pair_rows.get("native")
            tileir = pair_rows.get("tileir")
            if native and tileir and native.get("status") == "PASS" and tileir.get("status") == "PASS":
                ratios.append(native["ms"] / tileir["ms"])
        if ratios:
            summary.update({
                "mean_native_over_tileir": sum(ratios) / len(ratios),
                "min_native_over_tileir": min(ratios),
                "max_native_over_tileir": max(ratios),
                "tileir_faster": sum(1 for value in ratios if value > 1.0),
                "speedup_pairs": len(ratios),
            })
    return summary


def _print_case_table(rows: Sequence[dict], specs: Sequence[PairSpec], paths: Sequence[str]) -> None:
    by_pair = {spec.pair_id: {} for spec in specs}
    for row in rows:
        if row["pair_id"] in by_pair:
            by_pair[row["pair_id"]][row["path"]] = row

    pair_col_width = 20
    if set(paths) == {"native", "tileir"}:
        print(f"{'pair':<{pair_col_width}} {'native ms':>10} {'TileIR ms':>10} "
              f"{'native/TileIR':>13} {'check':>8} {'route':>13} {'timing':>14}  signature")
        print("-" * 144)
        for spec in specs:
            native = by_pair[spec.pair_id].get("native")
            tileir = by_pair[spec.pair_id].get("tileir")
            if native and tileir and native.get("status") == "PASS" and tileir.get("status") == "PASS":
                ratio = native["ms"] / tileir["ms"] if tileir["ms"] else float("nan")
                validation = native.get("validation") if native.get("validation") == tileir.get(
                    "validation") else "mixed"
                route_s = f"{native.get('route')}/{tileir.get('route')}"
                timing_s = native.get("timing") if native.get("timing") == tileir.get("timing") else "mixed"
                print(f"{spec.pair_id:<{pair_col_width}} {native['ms']:10.3f} {tileir['ms']:10.3f} "
                      f"{ratio:13.3f} {validation:>8} {route_s:>13} {timing_s:>14}  {spec.function_name}")
            else:
                for row in (native, tileir):
                    if row:
                        print(f"{spec.pair_id:<{pair_col_width}} {row.get('path', ''):<10} "
                              f"{row.get('status', 'FAIL'):<8} route={row.get('route')} "
                              f"message={row.get('message', '')}")
        return

    print(f"{'pair':<{pair_col_width}} {'path':<8} {'ms':>10} "
          f"{'check':>8} {'route':>8} {'timing':>14}  signature")
    print("-" * 128)
    for spec in specs:
        for path in paths:
            row = by_pair[spec.pair_id].get(path)
            if row and row.get("status") == "PASS":
                print(f"{spec.pair_id:<{pair_col_width}} {path:<8} {row['ms']:10.3f} "
                      f"{row.get('validation'):>8} {row.get('route'):>8} {row.get('timing'):>14}  {spec.function_name}")
            elif row:
                print(f"{spec.pair_id:<{pair_col_width}} {path:<8} {row.get('status', 'FAIL'):<8} "
                      f"route={row.get('route')} message={row.get('message', '')}")


def _print_case_summary(summary: dict) -> None:
    print(f"passed runs: {summary['passed']}/{summary['runs']}  "
          f"passed pairs: {summary['passed_pairs']}/{summary['pairs']}")
    if "mean_native_over_tileir" in summary:
        print("native/TileIR ratio: "
              f"mean={summary['mean_native_over_tileir']:.3f}, "
              f"min={summary['min_native_over_tileir']:.3f}, "
              f"max={summary['max_native_over_tileir']:.3f}, "
              f"TileIR faster={summary['tileir_faster']}/{summary['speedup_pairs']}")


def _print_tables(rows: Sequence[dict], specs: Sequence[PairSpec], paths: Sequence[str], summary: dict) -> None:
    for idx, (case, case_specs) in enumerate(_group_specs_by_case(specs)):
        if idx:
            print("")
        print(f"case: {case} ({len(case_specs)} pairs)")
        _print_case_table(rows, case_specs, paths)
        _print_case_summary(summary["cases"][case])


def _print_pair_progress(spec: PairSpec, pair_rows: Sequence[dict], stream) -> None:
    by_path = {row.get("path"): row for row in pair_rows}
    failures = [row for row in pair_rows if row.get("status") != "PASS"]
    if failures:
        details = "; ".join(f"{row.get('path', '?')}: {row.get('message', 'failed')}" for row in failures)
        print(f"[FAIL] {spec.pair_id} {spec.case}: {details}", file=stream, flush=True)
        return

    native = by_path.get("native")
    tileir = by_path.get("tileir")
    if native and tileir:
        ratio = native["ms"] / tileir["ms"] if tileir["ms"] else float("nan")
        print(
            f"[OK] {spec.pair_id} {spec.case}: "
            f"native={native['ms']:.3f} ms, TileIR={tileir['ms']:.3f} ms, speedup={ratio:.3f}x",
            file=stream,
            flush=True,
        )
        return

    details = ", ".join(f"{row.get('path', '?')}={row['ms']:.3f} ms route={row.get('route')}" for row in pair_rows)
    print(f"[OK] {spec.pair_id} {spec.case}: {details}", file=stream, flush=True)


def run_parent(
    specs: Sequence[PairSpec],
    paths: Sequence[str],
    warmup: float,
    rep: float,
    min_rep: int,
    initial_rep: int,
    check: bool,
    cache_dir: Optional[pathlib.Path],
    clear_cache: bool,
    json_output: bool,
    require_speedup_fraction: Optional[float],
) -> int:
    script = pathlib.Path(__file__).resolve()
    if cache_dir is None:
        cache_root = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    else:
        cache_root = cache_dir
    if clear_cache and cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = []
    progress_stream = sys.stderr if json_output else sys.stdout
    for spec in specs:
        pair_rows = []
        for path in paths:
            row = _run_subprocess(script, spec, path, warmup, rep, min_rep, initial_rep, check, cache_root)
            rows.append(row)
            pair_rows.append(row)
        _print_pair_progress(spec, pair_rows, progress_stream)

    summary = _summarize(rows, specs, paths, cache_root)
    if json_output:
        print(json.dumps({"summary": summary, "rows": rows}, sort_keys=True, indent=2))
    else:
        _print_tables(rows, specs, paths, summary)
        print("")
        print(f"cache root: {summary['cache_root']}")

    if summary["failed"]:
        return 1
    if require_speedup_fraction is not None:
        speedup_pairs = summary.get("speedup_pairs", 0)
        if speedup_pairs == 0:
            return 2
        faster_fraction = summary.get("tileir_faster", 0) / speedup_pairs
        if faster_fraction < require_speedup_fraction:
            return 2
    return 0


def _parse_cases(value: str) -> List[str]:
    if value == "all":
        return list(CASES)
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in out if item not in CASES]
    if bad:
        raise SystemExit(f"unknown case(s): {', '.join(bad)}")
    return out


def _parse_pair_filter(value: str, cases: Sequence[str]) -> List[PairSpec]:

    def sort_specs(specs: Sequence[PairSpec]) -> List[PairSpec]:
        case_index = {case: idx for idx, case in enumerate(cases)}
        pair_index = {spec.pair_id: idx for idx, spec in enumerate(PAIR_SPECS)}
        return sorted(specs, key=lambda spec: (case_index[spec.case], pair_index[spec.pair_id]))

    if value == "all":
        return sort_specs([spec for spec in PAIR_SPECS if spec.case in cases])
    requested = [item.strip() for item in value.split(",") if item.strip()]
    out: List[PairSpec] = []
    for item in requested:
        if item in PAIR_BY_ID:
            spec = PAIR_BY_ID[item]
            if spec.case not in cases:
                raise SystemExit(f"pair {item} belongs to case {spec.case}, not selected cases {','.join(cases)}")
            out.append(spec)
            continue
        if item.isdigit():
            suffix = f"-{int(item):02d}"
            matches = [spec for spec in PAIR_SPECS if spec.case in cases and spec.pair_id.endswith(suffix)]
            if len(matches) == 1:
                out.append(matches[0])
                continue
            if len(matches) > 1:
                raise SystemExit(f"pair index {item} matches multiple cases; use full pair id")
        raise SystemExit(f"unknown pair id: {item}")
    return sort_specs(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="comma-separated case list or 'all'")
    parser.add_argument("--pair", default="all",
                        help="pair id, comma-separated pair ids, numeric case-local index, or 'all'")
    parser.add_argument("--path", choices=("both", "native", "tileir"), default="both")
    parser.add_argument("--warmup", type=float, default=100.0, help="benchmark warmup duration in ms")
    parser.add_argument("--rep", type=float, default=50.0, help="benchmark measured duration in ms")
    parser.add_argument("--min-rep", type=int, default=2, help="minimum measured benchmark iterations")
    parser.add_argument("--initial-rep", type=int, default=5, help="initial iterations for benchmark runtime estimate")
    parser.add_argument("--no-check", dest="check", action="store_false",
                        help="skip torch-reference checks where available")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--ci-subset",
        action="store_true",
        help="run three curated representative pairs per selected case for CI",
    )
    parser.add_argument("--require-speedup", action="store_true", help="require every paired TileIR run to be faster")
    parser.add_argument(
        "--require-speedup-fraction",
        type=float,
        default=None,
        help="require at least this fraction of paired TileIR runs to be faster",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.require_speedup_fraction is not None and not 0.0 <= args.require_speedup_fraction <= 1.0:
        raise SystemExit("--require-speedup-fraction must be between 0.0 and 1.0")
    require_speedup_fraction = args.require_speedup_fraction
    if args.require_speedup:
        require_speedup_fraction = 1.0

    cases = _parse_cases(args.case)
    if args.ci_subset:
        if args.pair != "all":
            raise SystemExit("--ci-subset cannot be combined with --pair")
        specs = _representative_specs(cases)
    else:
        specs = _parse_pair_filter(args.pair, cases)
    if args.child:
        if len(specs) != 1 or args.path == "both":
            raise SystemExit("--child requires exactly one pair and one path")
        return run_child(
            specs[0].pair_id,
            args.path,
            args.warmup,
            args.rep,
            args.min_rep,
            args.initial_rep,
            args.check,
        )

    paths = ["native", "tileir"] if args.path == "both" else [args.path]
    return run_parent(
        specs=specs,
        paths=paths,
        warmup=args.warmup,
        rep=args.rep,
        min_rep=args.min_rep,
        initial_rep=args.initial_rep,
        check=args.check,
        cache_dir=args.cache_dir,
        clear_cache=args.clear_cache,
        json_output=args.json,
        require_speedup_fraction=require_speedup_fraction,
    )


if __name__ == "__main__":
    raise SystemExit(main())
