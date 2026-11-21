# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import os
import subprocess

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from triton.backends.ascend.program_grid import (
    DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK,
    INDEPENDENT_AXIS_TENSORIZE_RULE_BIT,
    PERSISTENT_TASK_STRIP_MINING_RULE_BIT,
)
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource
from triton.errors import TritonError

pytestmark = pytest.mark.backend("none")


@triton.jit
def graph_optimize_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offsets)
    tl.store(y_ptr + offsets, value)


@triton.jit
def graph_optimize_legacy_memory_isolation_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    # This is deliberately a strided AddPtr load/store shape.  It is eligible
    # for the legacy T2L StridedLoadStoreRewrite, but must remain untouched by
    # the early generic graph-optimize pass even when its default mask carries
    # all seven rule identities.
    offsets = tl.arange(0, BLOCK) * 3
    value = tl.load(x_ptr + offsets)
    tl.store(y_ptr + offsets, value)


@triton.jit
def diagonal_shift_forward_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    # Shift a cumulative sum left by one the way the segmented-cumsum kernels
    # spell it: a sub-diagonal mask over a BLOCK x BLOCK matrix, reduced back
    # to one dimension.  DiagonalMaskRemoval must collapse it to a subtraction.
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    cumulative = tl.cumsum(x, axis=0)
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    matrix = tl.where(rows == cols + 1, cumulative[None, :], 0.0)
    tl.store(y_ptr + offsets, tl.sum(matrix, axis=1))


@triton.jit
def diagonal_shift_reverse_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    # The mirrored form: a reverse scan shifted right by a super-diagonal mask.
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    cumulative = tl.cumsum(x, axis=0, reverse=True)
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    matrix = tl.where(rows == cols - 1, cumulative[None, :], 0.0)
    tl.store(y_ptr + offsets, tl.sum(matrix, axis=1))


@triton.jit
def wrapped_tile_copy_kernel(src_ptr, dst_ptr, n, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # The fused-MoE weight-tile shape: the column offset wraps on a runtime
    # dimension, and the store mask already discards the lanes past it.
    # ConvertModuloToMask must drop the wrap and mask the load instead.
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    wrapped = offs_n % n
    tile = tl.load(src_ptr + (offs_k[:, None] * n + wrapped[None, :]))
    tl.store(
        dst_ptr + (offs_k[:, None] * n + offs_n[None, :]),
        tile,
        mask=offs_n[None, :] < n,
    )


@triton.jit
def constexpr_wrapped_tile_copy_kernel(src_ptr, dst_ptr, N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # Same shape with a compile-time bound.  TritonToStructured can keep this
    # wrap and re-express it as a strided access, which is exactly equivalent,
    # so ConvertModuloToMask must leave it alone.
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    wrapped = offs_n % N
    tile = tl.load(src_ptr + (offs_k[:, None] * N + wrapped[None, :]))
    tl.store(
        dst_ptr + (offs_k[:, None] * N + offs_n[None, :]),
        tile,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def overflow_assert_provenance_kernel(output_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    checked_offset = offsets * n
    # Deliberately reuse the automatic message.  This must remain a user
    # assertion without the private auto-overflow provenance marker.
    tl.device_assert(
        checked_offset >= 0,
        "int32 overflow detected for operation mul",
    )
    tl.store(output_ptr + checked_offset, 0.0)


@triton.jit
def fused_swiglu_fast_sigmoid(x):
    return tl.fdiv(1.0, 1.0 + tl.exp(-x))


@triton.jit
def fused_swiglu_fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return tl.fdiv(x, 1.0 + tl.exp(-x)).to(dtype)


@triton.jit
def fused_swiglu_fast_silu_bwd(dy, x):
    dtype = x.type.element_ty
    dy = dy.to(tl.float32)
    x = x.to(tl.float32)
    sigmoid = fused_swiglu_fast_sigmoid(x)
    return (dy * sigmoid * (1 + x * (1 - sigmoid))).to(dtype)


@triton.jit
def fused_swiglu_bwd_b_graph_optimize_kernel(
    dy_ptr,
    g_ptr,
    fc_ptr,
    dg_ptr,
    dfc_ptr,
    db_g_ptr,
    db_fc_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Test-only fixed-config clone of fused_swiglu_bwd_b_kernel.

    Keep the original [column, row] pointer layout deliberately: it is the
    implicit transpose that LoadStoreTranspose rewrites.  The body retains the
    three loads, two loop stores, binary fanout, mask/other, loop-carried
    reductions, and two final reduction stores of the task kernel.
    """
    dtype = dy_ptr.type.element_ty
    col_idx = tl.program_id(axis=0)
    col_off = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_off = tl.arange(0, BLOCK_SIZE_M)
    dy_ptrs = dy_ptr + (row_off[None, :] * N + col_off[:, None])
    g_ptrs = g_ptr + (row_off[None, :] * N + col_off[:, None])
    fc_ptrs = fc_ptr + (row_off[None, :] * N + col_off[:, None])
    dg_ptrs = dg_ptr + (row_off[None, :] * N + col_off[:, None])
    dfc_ptrs = dfc_ptr + (row_off[None, :] * N + col_off[:, None])
    sum_b_g = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    sum_b_fc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for row_idx in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask = (row_off[None, :] < M - row_idx * BLOCK_SIZE_M) & (col_off[:, None] < N)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=mask, other=0.0)
        fc = tl.load(fc_ptrs, mask=mask, other=0.0).to(tl.float32)
        silu_g = fused_swiglu_fast_silu(g)
        dg = (dy * fc).to(dtype)
        dg = fused_swiglu_fast_silu_bwd(dg, g)
        dfc = (dy * silu_g.to(tl.float32)).to(dtype)
        sum_b_g += dg.to(tl.float32)
        sum_b_fc += dfc.to(tl.float32)
        tl.store(dg_ptrs, dg, mask=mask)
        tl.store(dfc_ptrs, dfc, mask=mask)
        dy_ptrs += BLOCK_SIZE_M * N
        g_ptrs += BLOCK_SIZE_M * N
        fc_ptrs += BLOCK_SIZE_M * N
        dg_ptrs += BLOCK_SIZE_M * N
        dfc_ptrs += BLOCK_SIZE_M * N

    tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 1), mask=col_off < N)
    tl.store(db_fc_ptr + col_off, tl.sum(sum_b_fc, 1), mask=col_off < N)


def make_ast_ttir(options):
    source = ASTSource(
        graph_optimize_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32"},
        {"BLOCK": 16},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return ast_to_ttir(graph_optimize_kernel, source, context, options, {}, {})


def make_legacy_memory_isolation_ast_ttir(options):
    source = ASTSource(
        graph_optimize_legacy_memory_isolation_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32"},
        {"BLOCK": 16},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return ast_to_ttir(
        graph_optimize_legacy_memory_isolation_kernel,
        source,
        context,
        options,
        {},
        {},
    )


def make_fused_swiglu_ttir(options, block_m=256, block_n=32):
    """Compile the original-layout fused-SwiGLU clone only through TTIR.

    This deliberately avoids the backend binary pipeline, so the structural
    256x32 regression remains runnable on a host where that production tile is
    too large for the local NPU UB.  The separate small-tiling test exercises
    the complete on-device pipeline.
    """
    source = ASTSource(
        fused_swiglu_bwd_b_graph_optimize_kernel,
        {
            "dy_ptr": "*bf16",
            "g_ptr": "*bf16",
            "fc_ptr": "*bf16",
            "dg_ptr": "*bf16",
            "dfc_ptr": "*bf16",
            "db_g_ptr": "*bf16",
            "db_fc_ptr": "*bf16",
            "M": "i32",
            "N": "i32",
        },
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(
        fused_swiglu_bwd_b_graph_optimize_kernel,
        source,
        context,
        options,
        {},
        {},
    )
    return make_ttir(module, {}, options)


def assert_ttir_text_reparseable(ttir, tmp_path, name):
    path = tmp_path / f"{name}.ttir.mlir"
    path.write_text(ttir)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    reparsed = ir.parse_mlir_module(str(path), context)
    assert "tt.func" in str(reparsed)


def assert_reparseable(module, tmp_path, name):
    assert_ttir_text_reparseable(str(module), tmp_path, name)


def test_graph_optimize_pass_accepts_zero_rule_mask(tmp_path):
    module = make_ast_ttir(NPUOptions(arch="Ascend910_95"))
    pm = ir.pass_manager(module.context)
    ascend.passes.ttir.add_graph_optimize(pm, rule_mask=0)
    pm.run(module, "")

    assert "tt.func" in str(module)
    assert_reparseable(module, tmp_path, "zero-rule-mask")


def test_default_graph_mask_enables_iat_and_ptsm():
    expected_mask = 511 | INDEPENDENT_AXIS_TENSORIZE_RULE_BIT | PERSISTENT_TASK_STRIP_MINING_RULE_BIT

    assert DEFAULT_GRAPH_OPTIMIZATION_RULE_MASK == expected_mask
    assert NPUOptions(arch="Ascend910_95").rule_mask == expected_mask


def test_default_generic_graph_mask_excludes_legacy_memory_compatibility(tmp_path, ):
    """The 8/16/32/64 identities are enabled by default, not generic rules.

    The generic GraphOptimizePass runs at early TTIR.  Row, Axis, Chunk, and
    StridedLoadStoreRewrite retain their original compatibility-pass slots, so
    this strided memory shape must not acquire either coalescing metadata or
    an indirect-memory op merely because the default mask is 3071.
    """
    options = NPUOptions(arch="Ascend910_95", enable_graph_optimize=True)
    default_result = make_ttir(
        make_legacy_memory_isolation_ast_ttir(options),
        {},
        options,
    )
    text = str(default_result)

    assert "hacc.coalesce_factor" not in text
    assert "hacc.coalesce_axis" not in text
    assert "hacc.coalesce_grid_ceil_div" not in text
    assert "tt.indirect_load" not in text
    assert "tt.indirect_store" not in text
    assert_reparseable(default_result, tmp_path, "generic-legacy-memory-isolation")


def test_default_graph_mask_preserves_native_graph_rule_bundle(monkeypatch, tmp_path):
    """Default 3071 retains native 1|2|4 behavior, including StoreCoalescing,
    without running legacy rules.

    This input has the LoadStoreTranspose (bit 1) structural signature.  The
    default all-identity mask must retain that native rewrite.  Together with
    the legacy isolation test above, this catches either failure mode:
    accidentally dropping native rules, or scheduling compatibility passes
    from the early generic graph pass.
    """
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path / "dump"))

    default_options = NPUOptions(
        arch="Ascend910_95",
        enable_graph_optimize=True,
        debug=True,
        sanitize_overflow=True,
    )

    default_ttir = str(make_fused_swiglu_ttir(default_options))

    # The default must perform the native transpose rewrite to [M, N].
    assert "tensor<256x32x!tt.ptr<bf16>>" in default_ttir
    assert "tensor<32x256x!tt.ptr<bf16>>" not in default_ttir
    assert "hacc.coalesce_factor" not in default_ttir
    assert "tt.indirect_load" not in default_ttir
    assert "tt.indirect_store" not in default_ttir
    assert_ttir_text_reparseable(
        default_ttir,
        tmp_path,
        "default-native-graph-rule-bundle",
    )


def test_make_ttir_supports_graph_optimize_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    for enabled in (False, True):
        options = NPUOptions(
            arch="Ascend910_95",
            enable_graph_optimize=enabled,
        )
        for attempt in range(2):
            module = make_ast_ttir(options)
            result = make_ttir(module, {}, options)

            assert result is module
            assert "tt.func" in str(result)
            assert_reparseable(
                result,
                tmp_path,
                f"graph-optimize-{int(enabled)}-{attempt}",
            )


def test_auto_overflow_assert_has_frontend_provenance_marker():
    source = ASTSource(
        overflow_assert_provenance_kernel,
        {"output_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 16},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(
        overflow_assert_provenance_kernel,
        source,
        context,
        NPUOptions(
            arch="Ascend910_95",
            debug=True,
            sanitize_overflow=True,
        ),
        {},
        {},
    )
    overflow_asserts = [
        line for line in str(module).splitlines() if '"int32 overflow detected for operation mul"' in line
    ]
    assert any("tt.auto_overflow_assert" in line for line in overflow_asserts)
    assert any("tt.auto_overflow_assert" not in line for line in overflow_asserts)


def test_fused_swiglu_bwd_b_256x32_graph_optimize_structure(monkeypatch, tmp_path):
    """Keep the original production tile as a TTIR-only structural gate."""
    # make_ttir() writes its debug dump through Triton's standard dump manager;
    # keep test artifacts scoped to pytest's disposable directory.
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path / "dump"))
    ttirs = {}
    for enabled in (False, True):
        options = NPUOptions(
            arch="Ascend910_95",
            enable_graph_optimize=enabled,
            debug=True,
            sanitize_overflow=True,
        )
        module = make_fused_swiglu_ttir(options)
        ttir = str(module)
        assert_ttir_text_reparseable(ttir, tmp_path, f"fused-swiglu-256x32-{int(enabled)}")
        (tmp_path / f"fused-swiglu-256x32-{int(enabled)}.ttir").write_text(ttir)
        ttirs[enabled] = ttir

    off_ttir = ttirs[False]
    on_ttir = ttirs[True]
    assert off_ttir.count("tt.assert") == on_ttir.count("tt.assert") == 20
    assert on_ttir.count("tt.auto_overflow_assert") == 20
    assert "tensor<32x256x!tt.ptr<bf16>>" in off_ttir
    assert "tensor<32x256x!tt.ptr<bf16>>" not in on_ttir
    assert "tensor<256x32x!tt.ptr<bf16>>" in on_ttir
    for old_guard_shape in (
            "tensor<1x256xi64>",
            "tensor<32x256xi64>",
            "tensor<1x256xi1>",
            "tensor<32x256xi1>",
    ):
        assert old_guard_shape not in on_ttir

    # This is the original 256x32 target from the task's 189-line comparison:
    # the rule must emit the natural-layout structure itself, without relying
    # on a post graph-optimize canonicalize/CSE/DCE cleanup.
    assert on_ttir.count("tt.expand_dims") == 2
    assert on_ttir.count("tt.splat") == 15
    assert on_ttir.count("arith.muli") == 8


def _require_npu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu", exc_type=ImportError)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    return torch


def test_graph_optimize_numerical_equivalence(monkeypatch, tmp_path):
    """Exercise the complete JIT pipeline on hardware with cache bypassed."""
    torch = _require_npu()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    source = torch.arange(16, dtype=torch.float32, device="npu")
    outputs = {}

    for enabled in (False, True):
        repeated_outputs = []
        for attempt in range(2):
            # JITFunction has its own in-memory cache, so clear it before each
            # launch to ensure TRITON_ALWAYS_COMPILE reaches triton.compile().
            graph_optimize_kernel.device_caches.clear()
            output = torch.empty_like(source)
            compiled = graph_optimize_kernel[(1, )](
                source,
                output,
                BLOCK=16,
                enable_graph_optimize=enabled,
            )
            torch.npu.synchronize()
            assert_ttir_text_reparseable(
                compiled.asm["ttir"],
                tmp_path,
                f"hardware-{int(enabled)}-{attempt}",
            )
            repeated_outputs.append(output.cpu())

        torch.testing.assert_close(repeated_outputs[0], source.cpu())
        torch.testing.assert_close(repeated_outputs[1], repeated_outputs[0])
        outputs[enabled] = repeated_outputs[-1]

    torch.testing.assert_close(outputs[True], outputs[False])


def test_fused_swiglu_bwd_b_small_tiling_graph_optimize_equivalence(monkeypatch, tmp_path):
    """Run the task kernel shape with a CANN-friendly validation tile.

    The production kernel's 256x32 autotune configuration is intentionally not
    changed.  The current environment's frozen validation tile is 64x8 (1/16
    of that tile); 32x8 remains an off-only fallback for smaller devices.
    """
    torch = _require_npu()
    torch_npu = pytest.importorskip("torch_npu")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    torch.manual_seed(20260724)

    m, n = 128, 64
    dtype = torch.bfloat16
    dy = torch.randn((m, n), dtype=dtype, device="npu")
    g = torch.randn((m, n), dtype=dtype, device="npu")
    fc = torch.randn((m, n), dtype=dtype, device="npu")

    def eager_reference():
        dy_f32 = dy.float()
        g_f32 = g.float()
        fc_f32 = fc.float()
        sigmoid = 1.0 / (1.0 + torch.exp(-g_f32))
        # The kernel explicitly rounds this intermediate to bf16 before the
        # SiLU backward expression, so reproduce that rounding in the oracle.
        dg = (dy_f32 * fc_f32).to(dtype).float()
        dg = (dg * sigmoid * (1.0 + g_f32 * (1.0 - sigmoid))).to(dtype)
        silu_g = (g_f32 / (1.0 + torch.exp(-g_f32))).to(dtype)
        dfc = (dy_f32 * silu_g.float()).to(dtype)
        return (
            dg,
            dfc,
            dg.float().sum(dim=0).to(dtype),
            dfc.float().sum(dim=0).to(dtype),
        )

    expected = tuple(value.cpu() for value in eager_reference())
    summary = {
        "M": m,
        "N": n,
        "dtype": str(dtype),
        "rtol": 3e-2,
        "atol": 1e-1,
        "enable_graph_optimize": [False, True],
        "debug": True,
        "sanitize_overflow": True,
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
        "cann_version": torch_npu.npu.utils.get_cann_version(),
        "candidates": [],
    }

    def run_mode(enabled, block_m, block_n, attempt):
        fused_swiglu_bwd_b_graph_optimize_kernel.device_caches.clear()
        dg = torch.empty((m, n), dtype=dtype, device="npu")
        dfc = torch.empty((m, n), dtype=dtype, device="npu")
        db_g = torch.empty((n, ), dtype=dtype, device="npu")
        db_fc = torch.empty((n, ), dtype=dtype, device="npu")
        compiled = fused_swiglu_bwd_b_graph_optimize_kernel[(triton.cdiv(n, block_n), )](
            dy,
            g,
            fc,
            dg,
            dfc,
            db_g,
            db_fc,
            m,
            n,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            num_warps=4,
            num_stages=2,
            debug=True,
            sanitize_overflow=True,
            enable_graph_optimize=enabled,
        )
        torch.npu.synchronize()
        ttir = compiled.asm["ttir"]
        name = f"fused-swiglu-small-{int(enabled)}-{block_m}x{block_n}-{attempt}"
        assert_ttir_text_reparseable(ttir, tmp_path, name)
        (tmp_path / f"{name}.ttir").write_text(ttir)
        return tuple(value.cpu() for value in (dg, dfc, db_g, db_fc)), ttir

    selected = None
    outputs = {}
    ttirs = {}
    last_runtime_error = None

    def write_summary():
        (tmp_path / "fused-swiglu-small-tiling-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    def run_repeated_mode(enabled, block_m, block_n):
        repeated = []
        mode_ttir = []
        for attempt in range(2):
            result, ttir = run_mode(enabled, block_m, block_n, attempt)
            repeated.append(result)
            mode_ttir.append(ttir)
        for first, second in zip(repeated[0], repeated[1]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        return repeated[-1], mode_ttir[-1]

    for block_m, block_n in ((64, 8), (32, 8)):
        candidate = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}
        try:
            off_output, off_ttir = run_repeated_mode(False, block_m, block_n)
        except (RuntimeError, TritonError, subprocess.CalledProcessError) as error:
            # Only a configuration that cannot run even with graph-optimize
            # disabled is eligible for the smaller fallback tile.
            candidate["status"] = "off-runtime-error"
            candidate["error_type"] = type(error).__name__
            candidate["error"] = str(error)
            summary["candidates"].append(candidate)
            last_runtime_error = error
            continue

        try:
            on_output, on_ttir = run_repeated_mode(True, block_m, block_n)
        except (RuntimeError, TritonError, subprocess.CalledProcessError) as error:
            # An enabled-only failure is the regression under test, not a
            # resource reason to silently reduce the validation coverage.
            candidate["status"] = "on-runtime-error"
            candidate["error_type"] = type(error).__name__
            candidate["error"] = str(error)
            summary["candidates"].append(candidate)
            write_summary()
            raise AssertionError("graph-optimize failed for a tiling that succeeds with it disabled") from error

        selected = (block_m, block_n)
        outputs = {False: off_output, True: on_output}
        ttirs = {False: off_ttir, True: on_ttir}
        candidate["status"] = "selected"
        summary["candidates"].append(candidate)
        break

    if selected is None:
        write_summary()
        raise RuntimeError("all fused_swiglu validation tilings failed") from last_runtime_error

    for index, (off, on, reference) in enumerate(zip(outputs[False], outputs[True], expected)):
        torch.testing.assert_close(on, off, rtol=3e-2, atol=1e-1)
        torch.testing.assert_close(on, reference, rtol=3e-2, atol=1e-1)
        summary[f"output_{index}_max_abs_on_off"] = float((on.float() - off.float()).abs().max().item())
        summary[f"output_{index}_max_abs_eager"] = float((on.float() - reference.float()).abs().max().item())

    block_m, block_n = selected
    on_ttir = ttirs[True]
    off_ttir = ttirs[False]
    assert on_ttir.count("tt.assert") == off_ttir.count("tt.assert")
    assert on_ttir.count("tt.assert") > 0
    assert "overflow detected for operation" in on_ttir
    assert on_ttir.count("tt.auto_overflow_assert") == on_ttir.count("tt.assert")
    old_pointer_type = "tensor<{}x{}x!tt.ptr".format(block_n, block_m)
    new_pointer_type = "tensor<{}x{}x!tt.ptr".format(block_m, block_n)
    assert old_pointer_type in off_ttir
    assert old_pointer_type not in on_ttir
    assert new_pointer_type in on_ttir
    assert "tensor<1x{}xi64>".format(block_m) not in on_ttir
    assert "tensor<{}x{}xi64>".format(block_n, block_m) not in on_ttir
    assert "tensor<1x{}xi1>".format(block_m) not in on_ttir
    assert "tensor<{}x{}xi1>".format(block_n, block_m) not in on_ttir

    summary["selected_tiling"] = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "num_warps": 4,
        "num_stages": 2,
    }
    summary["graph_optimize_assert_count"] = on_ttir.count("tt.assert")
    summary["device"] = (torch.npu.get_device_name(torch.npu.current_device())
                         if hasattr(torch.npu, "get_device_name") else "npu")
    write_summary()


def make_diagonal_shift_ttir(kernel, options, block=16, rule_mask=None):
    source = ASTSource(
        kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32"},
        {"BLOCK": block},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, source, context, options, {}, {})
    module = make_ttir(module, {}, options)
    if rule_mask is not None:
        pm = ir.pass_manager(module.context)
        ascend.passes.ttir.add_graph_optimize(pm, rule_mask=rule_mask)
        pm.run(module, "")
    return str(module)


@pytest.mark.parametrize(
    "kernel",
    (diagonal_shift_forward_kernel, diagonal_shift_reverse_kernel),
    ids=("forward-left-shift", "reverse-right-shift"),
)
def test_diagonal_mask_removal_collapses_quadratic_shift(kernel):
    """The 16x16 intermediate and its reduction give way to one subtraction."""
    options = NPUOptions(arch="Ascend910B1", enable_graph_optimize=True)

    ttir = make_diagonal_shift_ttir(kernel, options)

    assert "arith.subf" in ttir
    assert "tt.reduce" not in ttir
    assert "tt.scan" in ttir
    # The quadratic intermediate is what the rule exists to remove.
    assert "tensor<16x16xf32>" not in ttir
    assert "tensor<16x16xi1>" not in ttir


@pytest.mark.parametrize(
    "kernel",
    (diagonal_shift_forward_kernel, diagonal_shift_reverse_kernel),
    ids=("forward-left-shift", "reverse-right-shift"),
)
def test_diagonal_mask_removal_is_gated_by_its_rule_bit(kernel):
    """Mask 127 is every other identity, so the pattern must survive intact."""
    options = NPUOptions(arch="Ascend910B1", enable_graph_optimize=False)

    ttir = make_diagonal_shift_ttir(kernel, options, rule_mask=127)

    assert "tt.reduce" in ttir
    assert "tensor<16x16xf32>" in ttir
    assert "arith.subf" not in ttir


def test_diagonal_mask_removal_applies_without_simt_route():
    """The rewrite is compute logic, so it must not depend on pure-SIMT mode.

    Its predecessor lived behind the 910_95 plus force-SIMT gate of
    processStridedLoadStoreRewriteOperations and therefore never ran on other
    targets.
    """
    for arch, compile_mode in (
        ("Ascend910B1", "simd"),
        ("Ascend910_9589", "simt_only"),
    ):
        options = NPUOptions(
            arch=arch,
            enable_graph_optimize=True,
            compile_mode=compile_mode,
        )

        ttir = make_diagonal_shift_ttir(diagonal_shift_forward_kernel, options)

        assert "arith.subf" in ttir
        assert "tt.reduce" not in ttir


def test_diagonal_mask_removal_numerical_equivalence(monkeypatch, tmp_path):
    """Compare the default graph path against the unoptimized kernel on hardware.

    The float rewrite is not bit-exact, since scan[i] - x[i] only recovers
    scan[i - 1] up to rounding, so this asserts closeness rather than equality.
    """
    torch = _require_npu()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    source = torch.rand(16, dtype=torch.float32, device="npu")
    expected = torch.cat([
        torch.zeros(1, dtype=torch.float32),
        torch.cumsum(source.cpu(), dim=0)[:-1],
    ])

    outputs = {}
    for enabled in (False, True):
        diagonal_shift_forward_kernel.device_caches.clear()
        output = torch.empty_like(source)
        compiled = diagonal_shift_forward_kernel[(1, )](
            source,
            output,
            BLOCK=16,
            enable_graph_optimize=enabled,
        )
        torch.npu.synchronize()
        assert_ttir_text_reparseable(
            compiled.asm["ttir"],
            tmp_path,
            f"diagonal-{int(enabled)}",
        )
        outputs[enabled] = output.cpu()

    assert "arith.subf" in compiled.asm["ttir"]
    torch.testing.assert_close(outputs[False], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(outputs[True], outputs[False], rtol=1e-5, atol=1e-6)


def make_wrapped_tile_ttir(options, block_k=8, block_n=16, bound=None, rule_mask=None):
    kernel = wrapped_tile_copy_kernel if bound is None else constexpr_wrapped_tile_copy_kernel
    signature = {"src_ptr": "*fp32", "dst_ptr": "*fp32"}
    constants = {"BLOCK_K": block_k, "BLOCK_N": block_n}
    if bound is None:
        signature["n"] = "i32"
    else:
        constants["N"] = bound

    source = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, source, context, options, {}, {})
    module = make_ttir(module, {}, options)
    if rule_mask is not None:
        pm = ir.pass_manager(module.context)
        ascend.passes.ttir.add_graph_optimize(pm, rule_mask=rule_mask)
        pm.run(module, "")
    return str(module)


def test_convert_modulo_to_mask_linearizes_wrapped_tile():
    """The wrap gives way to a linear offset plus a boundary mask on the load.

    The point of the rewrite is the address form: TritonToStructured can turn a
    linear masked access into one contiguous transfer, while it has no way to
    keep a wrapped address contiguous.
    """
    options = NPUOptions(arch="Ascend910B1", enable_graph_optimize=True)

    ttir = make_wrapped_tile_ttir(options)

    assert "arith.remsi" not in ttir
    assert "arith.cmpi slt" in ttir
    # The load carries a mask and a zero fill it did not have before.
    assert "tt.load" in ttir
    assert "arith.constant dense<0.000000e+00> : tensor<8x16xf32>" in ttir


def test_convert_modulo_to_mask_is_gated_by_its_rule_bit():
    """Mask 255 is every other identity, so the wrap must survive intact."""
    options = NPUOptions(arch="Ascend910B1", enable_graph_optimize=False)

    ttir = make_wrapped_tile_ttir(options, rule_mask=255)

    assert "arith.remsi" in ttir


def test_convert_modulo_to_mask_leaves_compile_time_bounds_alone():
    """A constant divisor belongs to TritonToStructured, which stays equivalent.

    visitOperandRem re-expresses such a wrap as a strided access instead of
    discarding it, so claiming this candidate would trade an exact rewrite for
    an approximate one.
    """
    options = NPUOptions(arch="Ascend910B1", enable_graph_optimize=True)

    ttir = make_wrapped_tile_ttir(options, bound=20)

    assert "arith.remsi" in ttir


def test_convert_modulo_to_mask_applies_without_simt_route():
    """The rewrite is address logic, so it must not depend on pure-SIMT mode."""
    for arch, compile_mode in (
        ("Ascend910B1", "simd"),
        ("Ascend910_9589", "simt_only"),
    ):
        options = NPUOptions(
            arch=arch,
            enable_graph_optimize=True,
            compile_mode=compile_mode,
        )

        ttir = make_wrapped_tile_ttir(options)

        assert "arith.remsi" not in ttir


def test_convert_modulo_to_mask_numerical_equivalence(monkeypatch, tmp_path):
    """The lanes the store keeps must read exactly what the wrap gave them.

    A bound that is not a multiple of the tile makes the last program a boundary
    tile, which is the only place the wrap ever mattered.
    """
    torch = _require_npu()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    block_k = 8
    block_n = 16
    bound = 20
    # The wrapped lanes now address past the last row instead of folding back
    # into the tensor, so the allocation carries one tile of slack.  Whether the
    # lowered transfer actually stops at the mask is a TritonToStructured
    # property, asserted structurally by the tests above.
    source = torch.rand(block_k * bound + block_n, dtype=torch.float32, device="npu")
    expected = source[:block_k * bound].cpu()

    outputs = {}
    for enabled in (False, True):
        wrapped_tile_copy_kernel.device_caches.clear()
        output = torch.zeros(block_k * bound, dtype=torch.float32, device="npu")
        grid = ((bound + block_n - 1) // block_n, )
        compiled = wrapped_tile_copy_kernel[grid](
            source,
            output,
            bound,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            enable_graph_optimize=enabled,
        )
        torch.npu.synchronize()
        assert_ttir_text_reparseable(
            compiled.asm["ttir"],
            tmp_path,
            f"wrapped-tile-{int(enabled)}",
        )
        outputs[enabled] = output.cpu()

    assert "arith.remsi" not in compiled.asm["ttir"]
    torch.testing.assert_close(outputs[False], expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[True], outputs[False], rtol=0, atol=0)


def test_graph_optimize_options_contribute_to_npu_hash():
    assert (NPUOptions(enable_graph_optimize=False).hash() != NPUOptions(enable_graph_optimize=True).hash())
