import re

import pytest
import torch
import triton
import triton.language as tl
import triton.language.core as tl_core
import triton.experimental.tle.language as tle

from triton._C import libtriton
from triton._C.libtriton import ir, passes
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, compile_to_ttir, mthreads_backend
from triton.experimental.tle.language.gpu.mthreads.layout import (
    MusaDotOperandEncoding,
    MusaSqmmaEncoding,
    MusaWmmaEncoding,
)


def _make_mthreads_builder():
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    return context, ir.builder(context)


def _make_mthreads_function_argument(shape=None):
    context, builder = _make_mthreads_builder()
    value_type = builder.get_float_ty()
    if shape is not None:
        value_type = builder.get_block_ty(value_type, shape)
    module = builder.create_module()
    function_type = builder.get_function_ty([value_type], [])
    function = builder.get_or_insert_function(module, "set_layout_binding_test", function_type, "public", False)
    entry = function.add_entry_block()
    builder.set_insertion_point_to_start(entry)
    return context, builder, module, function.args(0)


_CONVERSION_LAYOUTS = """
#layout_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#layout_b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice_a = #ttg.slice<{dim = 0, parent = #layout_a}>
#slice_b = #ttg.slice<{dim = 0, parent = #layout_b}>
#slice_dim1_a = #ttg.slice<{dim = 1, parent = #layout_a}>
#reshape_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#reshape_slice = #ttg.slice<{dim = 1, parent = #reshape_layout}>
#mma = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [64, 64, 32]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma}>
#wmma_explicit = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [2, 2], instrShape = [16, 8, 16]}>
#wmma_unsupported = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [8, 8, 8]}>
#lhs_unsupported = #ttg.dot_op<{opIdx = 0, parent = #wmma_unsupported}>
#rhs_unsupported = #ttg.dot_op<{opIdx = 1, parent = #wmma_unsupported}>
#wmma_bad_warps = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [2, 1], instrShape = [16, 16, 16]}>
#lhs_bad_warps = #ttg.dot_op<{opIdx = 0, parent = #wmma_bad_warps}>
#rhs_bad_warps = #ttg.dot_op<{opIdx = 1, parent = #wmma_bad_warps}>
#sqmma_explicit = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [32, 64, 16]}>
#sqmma_lhs_explicit = #ttg.dot_op<{opIdx = 0, parent = #sqmma_explicit}>
#sqmma_rhs_explicit = #ttg.dot_op<{opIdx = 1, parent = #sqmma_explicit}>
#sqmma_unsupported = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [8, 8, 8]}>
#sqmma_lhs_unsupported = #ttg.dot_op<{opIdx = 0, parent = #sqmma_unsupported}>
#sqmma_rhs_unsupported = #ttg.dot_op<{opIdx = 1, parent = #sqmma_unsupported}>
#layout_1d_a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout_1d_b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout_3d_a = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#layout_3d_b = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 1, 4], order = [2, 1, 0]}>
#shared_1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
"""


def _convert_set_layout_to_ttgir(tmp_path, body):
    fixture = tmp_path / "mthreads_tle_set_layout_conversion.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\nmodule {{\n{body}\n}}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    passes.ttir.add_convert_to_ttgpuir(pm, "musa:ph1", 4, 32, 1)
    pm.run(module, "mthreads_tle_set_layout_conversion")
    ttgir = module.str_nodebug()
    _assert_set_layout_lowered(ttgir)
    return ttgir


def _run_ttgir_coalesce(tmp_path, body):
    fixture = tmp_path / "mthreads_tle_set_layout_coalesce.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 4 : i32, "
                       "\"ttg.threads-per-warp\" = 32 : i32} {\n"
                       f"{body}\n"
                       "}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    passes.ttgpuir.add_coalesce(pm)
    pm.run(module, "mthreads_tle_set_layout_coalesce")
    return module.str_nodebug()


def _run_ttgir_coalesce_and_finalize_explicit_layouts(tmp_path, body):
    fixture = tmp_path / "mthreads_tle_set_layout_coalesce_finalize.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 4 : i32, "
                       "\"ttg.threads-per-warp\" = 32 : i32} {\n"
                       f"{body}\n"
                       "}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    passes.ttgpuir.add_coalesce(pm)
    pm.run(module, "mthreads_tle_set_layout_coalesce")
    after_coalesce = module.str_nodebug()

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_finalize_explicit_layouts(pm)
    pm.run(module, "mthreads_tle_finalize_explicit_layouts")
    return after_coalesce, module.str_nodebug()


def _run_ttgir_remove_layout_conversions(tmp_path, body, repeat=1, extra_module_attrs=""):
    fixture = tmp_path / "mthreads_tle_set_layout_remove_conversions.mlir"
    module_attrs = ('"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, '
                    '"ttg.threads-per-warp" = 32 : i32')
    if extra_module_attrs:
        module_attrs += f", {extra_module_attrs}"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\nmodule attributes {{{module_attrs}}} {{\n{body}\n}}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    for _ in range(repeat):
        passes.ttgpuir.add_remove_layout_conversions(pm)
    pm.run(module, "mthreads_tle_set_layout_remove_conversions")
    return module.str_nodebug()


def _run_ttgir_finalize_explicit_layouts(tmp_path, body, finalize_repeat=1, run_cse=False):
    fixture = tmp_path / "mthreads_tle_finalize_explicit_layouts.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 4 : i32, "
                       "\"ttg.threads-per-warp\" = 32 : i32} {\n"
                       f"{body}\n"
                       "}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    if run_cse:
        pm = ir.pass_manager(context)
        passes.common.add_cse(pm)
        pm.run(module, "mthreads_tle_explicit_layout_cse")
    before_finalize = module.str_nodebug()

    pm = ir.pass_manager(context)
    for _ in range(finalize_repeat):
        libtriton.mthreads.passes.ttgpuir.add_tle_finalize_explicit_layouts(pm)
    pm.run(module, "mthreads_tle_finalize_explicit_layouts")
    return before_finalize, module.str_nodebug()


def _run_ttgir_select_encodings(tmp_path, body):
    fixture = tmp_path / "mthreads_tle_set_layout_select_encodings.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 4 : i32, "
                       "\"ttg.threads-per-warp\" = 32 : i32} {\n"
                       f"{body}\n"
                       "}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_tle_select_encodings(pm)
    pm.run(module, "mthreads_tle_set_layout_select_encodings")
    return module.str_nodebug()


def _run_ttgir_optimize_thread_locality(tmp_path, body, num_warps=4):
    fixture = tmp_path / "mthreads_tle_set_layout_optimize_thread_locality.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       f"module attributes {{\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = {num_warps} : i32, "
                       "\"ttg.threads-per-warp\" = 32 : i32} {\n"
                       f"{body}\n"
                       "}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    pm.run(module, "mthreads_tle_set_layout_optimize_thread_locality")
    return module.str_nodebug()


def _run_ttgir_musa_pass(tmp_path, body, pass_name):
    fixture = tmp_path / f"mthreads_tle_set_layout_{pass_name}.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 4 : i32, "
                       "ttg.target = \"musa:ph1\", \"ttg.threads-per-warp\" = 32 : i32} {\n"
                       f"{body}\n"
                       "}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(fixture), context)

    pm = ir.pass_manager(context)
    musa_passes = libtriton.mthreads.passes.ttgpuir
    pass_module = musa_passes if hasattr(musa_passes, pass_name) else passes.ttgpuir
    getattr(pass_module, pass_name)(pm)
    pm.run(module, f"mthreads_tle_set_layout_{pass_name}")
    return module.str_nodebug()


def _assert_set_layout_lowered(ttgir):
    assert "musa_tle.set_layout" not in ttgir
    assert "tle.gpu.set_layout" not in ttgir


def _explicit_result_encoding(line):
    match = re.search(r"tle\.explicit_encoding\.0 = (#\w+)", line)
    assert match is not None, line
    return match.group(1)


def _type_encoding_aliases(line):
    return re.findall(r", (#\w+)>", line)


def _convert_layout_encodings(line):
    attr_prefix = "tle.explicit_encoding.0 = "
    assert attr_prefix in line, line
    explicit = line.split(attr_prefix, 1)[1].split("} : tensor", 1)[0]
    type_pair = line.split("} : ", 1)[1]
    source, result = type_pair.split(" -> ", 1)
    return explicit, source, result


def _assert_direct_native_dot_chain(ttgir, mnemonic):
    dot_lines = [line for line in ttgir.splitlines() if mnemonic in line]
    assert len(dot_lines) == 2, ttgir
    first_result = re.match(r"\s*(%\w+)\s*=", dot_lines[0])
    assert first_result is not None, dot_lines[0]
    assert re.search(rf"{re.escape(first_result.group(1))}(?=[,\s])", dot_lines[1]), dot_lines


def _musa_runtime_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


requires_musa_runtime = pytest.mark.skipif(
    not _musa_runtime_available(),
    reason="MUSA runtime is not available",
)


def _warmup_and_run(kernel, *args, grid=(1, ), **kwargs):
    compiled = kernel.warmup(*args, grid=grid, **kwargs)
    kernel[grid](*args, **kwargs)
    return compiled


def _assert_runtime_pipeline_ir(compiled, mnemonic=None, native_dot_count=1):
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "musa_tle.set_layout" in ttir
    assert "tle.gpu.set_layout" not in ttir
    _assert_set_layout_lowered(ttgir)
    assert "tle.explicit_encoding." not in ttgir
    assert "tle.explicit_memory_encoding" not in ttgir
    assert "tle.explicit_encoding." not in llir
    assert "tle.explicit_memory_encoding" not in llir
    assert "nvvm" not in llir.lower()
    if mnemonic:
        assert ttgir.count(mnemonic) == native_dot_count
        assert " tt.dot " not in ttgir


@tl_core.builtin
def _ensure_ttg_layout_attrs(num_warps: tl.constexpr, warp_size: tl.constexpr, num_ctas: tl.constexpr, _semantic=None):
    _semantic.builder.ensure_ttg_layout_attrs(
        int(tl_core._unwrap_if_constexpr(num_warps)),
        int(tl_core._unwrap_if_constexpr(warp_size)),
        int(tl_core._unwrap_if_constexpr(num_ctas)),
    )


@triton.jit
def _valid_layout_attrs_kernel():
    _ensure_ttg_layout_attrs(4, 32, 1)
    _ensure_ttg_layout_attrs(4, 32, 1)


@triton.jit
def _num_warps_conflict_kernel():
    _ensure_ttg_layout_attrs(4, 32, 1)
    _ensure_ttg_layout_attrs(8, 32, 1)


@triton.jit
def _num_ctas_conflict_kernel():
    _ensure_ttg_layout_attrs(4, 32, 1)
    _ensure_ttg_layout_attrs(4, 32, 2)


@triton.jit
def _invalid_layout_attrs_kernel(num_warps: tl.constexpr, warp_size: tl.constexpr, num_ctas: tl.constexpr):
    _ensure_ttg_layout_attrs(num_warps, warp_size, num_ctas)


@triton.jit
def _set_layout_block_numeric_kernel(src, out, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = rows * BLOCK + cols
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(pointers, values)


@triton.jit
def _set_layout_slice_numeric_kernel(src, out, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(pointers, values)


@triton.jit
def _set_layout_transpose_numeric_kernel(
    src,
    out,
    BLOCK: tl.constexpr,
    SRC_LAYOUT: tl.constexpr,
    DST_LAYOUT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = rows * BLOCK + cols
    src_offsets = tle.gpu.set_layout(offsets, SRC_LAYOUT)
    dst_offsets = tle.gpu.set_layout(offsets, DST_LAYOUT)
    values = tle.gpu.set_layout(tl.load(src + src_offsets), SRC_LAYOUT)
    values = tl.trans(values) * 2.0 + 1.0
    tl.store(out + dst_offsets, values)


@triton.jit
def _set_layout_dual_domain_numeric_kernel(
    out_a,
    out_b,
    BLOCK: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = rows * BLOCK + cols
    root = tl.full((BLOCK, BLOCK), 3.25, tl.float32)
    values_a = tle.gpu.set_layout(root, LAYOUT_A)
    values_b = tle.gpu.set_layout(root, LAYOUT_B)
    offsets_a = tle.gpu.set_layout(offsets, LAYOUT_A)
    offsets_b = tle.gpu.set_layout(offsets, LAYOUT_B)
    pointers_a = out_a + offsets_a
    pointers_b = out_b + offsets_b
    tl.store(pointers_a, values_a)
    tl.store(pointers_b, values_b)


@triton.jit
def _set_layout_shared_rank3_slice_numeric_kernel(
    out_rows,
    out_cols,
    BLOCK: tl.constexpr,
    ROW_LAYOUT: tl.constexpr,
    COL_LAYOUT: tl.constexpr,
):
    values = tl.arange(0, BLOCK)
    rows = tle.gpu.set_layout(values[None, :, None], ROW_LAYOUT)
    cols = tle.gpu.set_layout(values[None, None, :], COL_LAYOUT)
    tl.store(out_rows + rows, rows.to(tl.float32))
    tl.store(out_cols + cols, cols.to(tl.float32))


@triton.jit(noinline=True)
def _set_layout_noinline_return_helper(BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    values = tl.arange(0, BLOCK)
    return tle.gpu.set_layout(values, LAYOUT)


@triton.jit(noinline=True)
def _set_layout_noinline_bridge_helper(BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    return _set_layout_noinline_return_helper(BLOCK, LAYOUT)


@triton.jit
def _set_layout_noinline_numeric_kernel(out, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    values = _set_layout_noinline_bridge_helper(BLOCK, LAYOUT)
    tl.store(out + values, values.to(tl.float32))


@triton.jit
def _set_layout_sqmma_pipeline_kernel(out, LAYOUT: tl.constexpr):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a = tle.gpu.alloc((block_m, block_k), dtype=tl.float16, layout=None)
    b = tle.gpu.alloc((block_k, block_n), dtype=tl.float16, layout=None)
    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    acc = tle.gpu.set_layout(acc, LAYOUT)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    ptrs = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(ptrs, acc)


@triton.jit
def _ordinary_ttgir_pipeline_kernel(out):
    offsets = tl.arange(0, 128)
    tl.store(out + offsets, offsets.to(tl.float32))


@triton.jit
def _automatic_musa_mma_numeric_kernel(
    a_base,
    b_base,
    out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    reduction = tl.arange(0, BLOCK_K)
    lhs = tl.load(a_base + rows * BLOCK_K + reduction[None, :])
    rhs = tl.load(b_base + reduction[:, None] * BLOCK_N + cols)
    result = tl.dot(lhs, rhs, out_dtype=tl.float32)
    tl.store(out + rows * BLOCK_N + cols, result)


@triton.jit
def _explicit_musa_wmma_kernel(
    out,
    MMA_LAYOUT: tl.constexpr,
    LHS_LAYOUT: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
):
    size: tl.constexpr = 16
    lhs = tl.full((size, size), 1.0, tl.bfloat16)
    rhs = tl.full((size, size), 2.0, tl.bfloat16)
    lhs = tle.gpu.set_layout(lhs, LHS_LAYOUT)
    rhs = tle.gpu.set_layout(rhs, RHS_LAYOUT)
    acc = tle.gpu.set_layout(tl.zeros((size, size), tl.float32), MMA_LAYOUT)
    result = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    rows = tl.arange(0, size)[:, None]
    cols = tl.arange(0, size)[None, :]
    ptrs = tle.gpu.set_layout(out + rows * size + cols, MMA_LAYOUT)
    tl.store(ptrs, result)


@triton.jit
def _explicit_musa_sqmma_kernel(
    out,
    MMA_LAYOUT: tl.constexpr,
    LHS_LAYOUT: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 64
    block_k: tl.constexpr = 32
    lhs = tl.full((block_m, block_k), 1.0, tl.float16)
    rhs = tl.full((block_k, block_n), 2.0, tl.float16)
    lhs = tle.gpu.set_layout(lhs, LHS_LAYOUT)
    rhs = tle.gpu.set_layout(rhs, RHS_LAYOUT)
    acc = tle.gpu.set_layout(tl.zeros((block_m, block_n), tl.float32), MMA_LAYOUT)
    result = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    rows = tl.arange(0, block_m)[:, None]
    cols = tl.arange(0, block_n)[None, :]
    ptrs = tle.gpu.set_layout(out + rows * block_n + cols, MMA_LAYOUT)
    tl.store(ptrs, result)


@triton.jit
def _explicit_musa_wmma_chained_kernel(
    out,
    MMA_LAYOUT: tl.constexpr,
    LHS_LAYOUT: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
):
    size: tl.constexpr = 16
    lhs = tle.gpu.set_layout(tl.full((size, size), 1.0, tl.bfloat16), LHS_LAYOUT)
    rhs = tle.gpu.set_layout(tl.full((size, size), 2.0, tl.bfloat16), RHS_LAYOUT)
    acc = tle.gpu.set_layout(tl.zeros((size, size), tl.float32), MMA_LAYOUT)
    first = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    first = tle.gpu.set_layout(first, MMA_LAYOUT)
    result = tl.dot(lhs, rhs, acc=first, out_dtype=tl.float32)
    rows = tl.arange(0, size)[:, None]
    cols = tl.arange(0, size)[None, :]
    ptrs = tle.gpu.set_layout(out + rows * size + cols, MMA_LAYOUT)
    tl.store(ptrs, result)


@triton.jit
def _explicit_musa_sqmma_chained_kernel(
    out,
    MMA_LAYOUT: tl.constexpr,
    LHS_LAYOUT: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 64
    block_k: tl.constexpr = 32
    lhs = tle.gpu.set_layout(tl.full((block_m, block_k), 1.0, tl.float16), LHS_LAYOUT)
    rhs = tle.gpu.set_layout(tl.full((block_k, block_n), 2.0, tl.float16), RHS_LAYOUT)
    acc = tle.gpu.set_layout(tl.zeros((block_m, block_n), tl.float32), MMA_LAYOUT)
    first = tl.dot(lhs, rhs, acc=acc, out_dtype=tl.float32)
    first = tle.gpu.set_layout(first, MMA_LAYOUT)
    result = tl.dot(lhs, rhs, acc=first, out_dtype=tl.float32)
    rows = tl.arange(0, block_m)[:, None]
    cols = tl.arange(0, block_n)[None, :]
    ptrs = tle.gpu.set_layout(out + rows * block_n + cols, MMA_LAYOUT)
    tl.store(ptrs, result)


def test_mthreads_tle_layout_attr_builder_is_backend_local():
    _, builder = _make_mthreads_builder()

    assert hasattr(builder, "ensure_ttg_layout_attrs")
    assert hasattr(builder, "get_blocked_encoding")
    assert hasattr(builder, "get_sliced_encoding")
    assert hasattr(builder, "get_dot_operand_layout")
    assert hasattr(builder, "clone_tensor_type_with_encoding")
    with pytest.raises(
            ValueError,
            match="mthreads TLE cannot set ttg layout attributes without an insertion block",
    ):
        builder.ensure_ttg_layout_attrs(4, 32, 1)


def test_mthreads_tle_musa_mma_layout_builders_are_backend_local():
    _, builder = _make_mthreads_builder()

    assert hasattr(builder, "get_musa_wmma_layout")
    assert hasattr(builder, "get_musa_sqmma_layout")


def test_mthreads_tle_set_layout_builder_is_backend_local():
    _, builder = _make_mthreads_builder()

    assert hasattr(builder, "create_tle_gpu_set_layout")


@pytest.mark.parametrize(
    "layout,expected_type,expected_rank,parent_type",
    [
        (tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0]), tle.gpu.BlockEncoding, 2, None),
        (
            tle.gpu.SlicedEncoding(0, tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0])),
            tle.gpu.SlicedEncoding,
            1,
            tle.gpu.BlockEncoding,
        ),
        (MusaWmmaEncoding([3, 1], [4, 1], [16, 16, 16]), MusaWmmaEncoding, 2, None),
        (MusaSqmmaEncoding([3, 1], [4, 1], [64, 64, 32]), MusaSqmmaEncoding, 2, None),
        (
            MusaDotOperandEncoding(0, MusaWmmaEncoding([3, 1], [4, 1], [16, 16, 16])),
            MusaDotOperandEncoding,
            2,
            MusaWmmaEncoding,
        ),
        (
            MusaDotOperandEncoding(1, MusaSqmmaEncoding([3, 1], [4, 1], [64, 64, 32])),
            MusaDotOperandEncoding,
            2,
            MusaSqmmaEncoding,
        ),
    ],
)
def test_mthreads_tle_python_layout_type_matrix(layout, expected_type, expected_rank, parent_type):
    assert isinstance(layout, tle.gpu.distributed_encoding)
    assert type(layout) is expected_type
    assert layout.rank == expected_rank
    assert isinstance(hash(layout), int)
    assert expected_type.__name__ in repr(layout)
    if parent_type is not None:
        assert type(layout.parent) is parent_type


def test_mthreads_tle_distributed_layout_builders_create_native_attrs():
    _, builder = _make_mthreads_builder()
    blocked = builder.get_blocked_encoding([1, 1], [1, 32], [4, 1], [1, 0], [])
    sliced = builder.get_sliced_encoding(0, blocked)
    wmma = builder.get_musa_wmma_layout([3, 1], [4, 1], [], [16, 16, 16])
    sqmma = builder.get_musa_sqmma_layout([3, 1], [4, 1], [], [64, 64, 32])
    wmma_lhs = builder.get_dot_operand_layout(0, wmma, 0)
    wmma_rhs = builder.get_dot_operand_layout(1, wmma, 0)
    sqmma_lhs = builder.get_dot_operand_layout(0, sqmma, 0)
    sqmma_rhs = builder.get_dot_operand_layout(1, sqmma, 0)

    module = builder.create_module()
    module.set_attr("test.blocked", blocked)
    module.set_attr("test.sliced", sliced)
    module.set_attr("test.wmma_lhs", wmma_lhs)
    module.set_attr("test.wmma_rhs", wmma_rhs)
    module.set_attr("test.sqmma_lhs", sqmma_lhs)
    module.set_attr("test.sqmma_rhs", sqmma_rhs)
    printed = str(module)

    assert "#ttg.blocked" in printed
    assert "#ttg.slice" in printed
    wmma_alias = re.search(r"(#mma\d*) = #ttg\.musa_wmma", printed)
    sqmma_alias = re.search(r"(#mma\d*) = #ttg\.musa_sqmma", printed)
    assert wmma_alias is not None, printed
    assert sqmma_alias is not None, printed
    for operand_index in (0, 1):
        assert f"#ttg.dot_op<{{opIdx = {operand_index}, parent = {wmma_alias.group(1)}}}>" in printed
        assert f"#ttg.dot_op<{{opIdx = {operand_index}, parent = {sqmma_alias.group(1)}}}>" in printed
    assert "nvidia_mma" not in printed


def test_mthreads_tle_clone_tensor_type_with_distributed_encoding():
    _, builder = _make_mthreads_builder()
    blocked = builder.get_blocked_encoding([1, 1], [1, 32], [4, 1], [1, 0], [])
    element_type = builder.get_float_ty()
    tensor_type = builder.get_block_ty(element_type, [16, 16])

    cloned = builder.clone_tensor_type_with_encoding(tensor_type, blocked)

    assert str(cloned).startswith("tensor<16x16xf32, #ttg.blocked")


@pytest.mark.parametrize(
    "method,args,diagnostic",
    [
        (
            "get_blocked_encoding",
            ([], [], [], [], []),
            "mthreads TLE blocked encoding rank must be positive",
        ),
        (
            "get_blocked_encoding",
            ([1, 1], [1, 32], [4], [1, 0], []),
            "mthreads TLE blocked encoding fields must have the same rank",
        ),
        (
            "get_blocked_encoding",
            ([1, 1], [1, 32], [4, 1], [0, 0], []),
            "mthreads TLE blocked encoding order must be a permutation of 0..rank-1",
        ),
        (
            "get_blocked_encoding",
            ([3, 1], [1, 32], [4, 1], [1, 0], []),
            "mthreads TLE blocked encoding size_per_thread entries must be positive powers of two",
        ),
        (
            "get_blocked_encoding",
            ([1, 1], [2, 8], [4, 1], [1, 0], []),
            "mthreads TLE PH1 blocked encoding requires product(threads_per_warp) == 32",
        ),
        (
            "get_blocked_encoding",
            ([1, 1], [1, 32], [3, 1], [1, 0], []),
            "mthreads TLE blocked encoding warps_per_cta entries must be positive powers of two",
        ),
        (
            "get_blocked_encoding",
            ([1, 1], [1, 32], [4, 1], [1, 0], [[0]]),
            "mthreads TLE CGA layout basis 0 has rank 1, expected 2",
        ),
        (
            "get_blocked_encoding",
            ([1, 1], [1, 32], [4, 1], [1, 0], [[0, -1]]),
            "mthreads TLE CGA layout basis 0 contains a negative value",
        ),
    ],
)
def test_mthreads_tle_blocked_layout_builder_rejects_invalid_contracts(method, args, diagnostic):
    _, builder = _make_mthreads_builder()
    with pytest.raises(ValueError, match=re.escape(diagnostic)):
        getattr(builder, method)(*args)


def test_mthreads_tle_sliced_layout_builder_rejects_invalid_contracts():
    _, builder = _make_mthreads_builder()
    not_distributed = builder.get_string_attr("not-a-layout")
    rank_one = builder.get_blocked_encoding([1], [32], [4], [0], [])
    rank_two = builder.get_blocked_encoding([1, 1], [1, 32], [4, 1], [1, 0], [])

    with pytest.raises(ValueError, match="parent must be a distributed encoding"):
        builder.get_sliced_encoding(0, not_distributed)
    with pytest.raises(ValueError, match="parent rank must be at least 2"):
        builder.get_sliced_encoding(0, rank_one)
    with pytest.raises(ValueError, match="dim must be less than parent rank"):
        builder.get_sliced_encoding(2, rank_two)


def test_mthreads_tle_dot_operand_layout_builder_rejects_invalid_contracts():
    _, builder = _make_mthreads_builder()
    blocked = builder.get_blocked_encoding([1, 1], [1, 32], [4, 1], [1, 0], [])
    wmma = builder.get_musa_wmma_layout([3, 1], [4, 1], [], [16, 16, 16])

    with pytest.raises(ValueError, match="dot operand index must be 0 or 1"):
        builder.get_dot_operand_layout(2, wmma, 0)
    with pytest.raises(ValueError, match="parent must be a MUSA WMMA or SQMMA encoding"):
        builder.get_dot_operand_layout(0, blocked, 0)
    with pytest.raises(ValueError, match="MUSA dot operand requires k_width=0"):
        builder.get_dot_operand_layout(0, wmma, 1)


def test_mthreads_tle_clone_tensor_type_rejects_invalid_contracts():
    _, builder = _make_mthreads_builder()
    scalar_type = builder.get_float_ty()
    tensor_type = builder.get_block_ty(scalar_type, [16, 16])
    not_distributed = builder.get_string_attr("not-a-layout")
    rank_one = builder.get_blocked_encoding([1], [32], [4], [0], [])

    with pytest.raises(TypeError, match="only clone a ranked tensor type"):
        builder.clone_tensor_type_with_encoding(scalar_type, rank_one)
    with pytest.raises(TypeError, match="encoding must be a distributed encoding"):
        builder.clone_tensor_type_with_encoding(tensor_type, not_distributed)
    with pytest.raises(ValueError, match="tensor rank must match distributed encoding rank"):
        builder.clone_tensor_type_with_encoding(tensor_type, rank_one)


def test_mthreads_tle_set_layout_binding_rejects_invalid_contracts():
    _scalar_context, scalar_builder, _scalar_module, scalar = _make_mthreads_function_argument()
    rank_one = scalar_builder.get_blocked_encoding([1], [32], [4], [0], [])
    with pytest.raises(TypeError, match="set_layout source must be a ranked tensor"):
        scalar_builder.create_tle_gpu_set_layout(scalar, rank_one)

    _tensor_context, tensor_builder, _tensor_module, tensor = _make_mthreads_function_argument([16, 16])
    not_distributed = tensor_builder.get_string_attr("not-a-layout")
    with pytest.raises(TypeError, match="target_encoding must be a distributed encoding"):
        tensor_builder.create_tle_gpu_set_layout(tensor, not_distributed)

    shared = tensor_builder.make_swizzled_shared_encoding_attr(1, 1, 1, [1, 0], [1, 1], [1, 1], [1, 0])
    with pytest.raises(TypeError, match="target_encoding must be a distributed encoding"):
        tensor_builder.create_tle_gpu_set_layout(tensor, shared)

    rank_one = tensor_builder.get_blocked_encoding([1], [32], [4], [0], [])
    with pytest.raises(ValueError, match="target encoding rank 1 must match source tensor rank 2"):
        tensor_builder.create_tle_gpu_set_layout(tensor, rank_one)


@pytest.mark.parametrize(
    "method,args,expected_mnemonic",
    [
        ("get_musa_wmma_layout", ([3, 1], [4, 1], [], [16, 16, 16]), "#ttg.musa_wmma"),
        ("get_musa_sqmma_layout", ([3, 1], [4, 1], [], [64, 64, 32]), "#ttg.musa_sqmma"),
        ("get_musa_wmma_layout", ([3, 1], [2, 2, 1], [[0, 1, 0]], [16, 8, 8]), "#ttg.musa_wmma"),
        ("get_musa_sqmma_layout", ([3, 1], [4, 2, 1], [[0, 1, 0]], [32, 64, 16]), "#ttg.musa_sqmma"),
    ],
)
def test_mthreads_tle_musa_mma_layout_builders_create_native_attrs(method, args, expected_mnemonic):
    _, builder = _make_mthreads_builder()
    attr = getattr(builder, method)(*args)
    module = builder.create_module()
    module.set_attr("test.layout", attr)
    printed = str(module)

    assert expected_mnemonic in printed
    assert "versionMajor = 3" in printed
    assert "versionMinor = 1" in printed
    assert "instrShape" in printed
    assert "nvidia_mma" not in printed


@pytest.mark.parametrize(
    "method,args,diagnostic",
    [
        (
            "get_musa_wmma_layout",
            ([3], [4, 1], [], [16, 16, 16]),
            "mthreads TLE WMMA version must contain major and minor",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 2], [4, 1], [], [16, 16, 16]),
            "mthreads TLE WMMA currently supports only MUSA PH1 version [3, 1]",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 1], [4], [], [16, 16, 16]),
            "mthreads TLE WMMA warps_per_cta rank must be 2 or 3",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 1], [3, 1], [], [16, 16, 16]),
            "mthreads TLE WMMA warps_per_cta entries must be positive powers of two",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 1], [2, 2, 2], [], [16, 16, 16]),
            "mthreads TLE WMMA rank-3 warps_per_cta must end in 1",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 1], [4, 1], [], [16, 16]),
            "mthreads TLE WMMA instr_shape must contain logical (M, N, K)",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 1], [4, 1], [], [8, 8, 8]),
            "mthreads TLE WMMA instr_shape is not supported by any PH1 WMMA intrinsic",
        ),
        (
            "get_musa_sqmma_layout",
            ([3, 1], [2, 2], [], [64, 64, 32]),
            "mthreads TLE SQMMA warps_per_cta[0] must be a multiple of 4",
        ),
        (
            "get_musa_sqmma_layout",
            ([3, 1], [4, 1], [], [32, 128, 8]),
            "mthreads TLE SQMMA instr_shape is not supported by any PH1 SQMMA type contract",
        ),
        (
            "get_musa_wmma_layout",
            ([3, 1], [2, 2, 1], [[0, 1]], [16, 16, 16]),
            "mthreads TLE WMMA CGA layout basis 0 has rank 2, expected 3",
        ),
        (
            "get_musa_sqmma_layout",
            ([3, 1], [4, 1], [[0, -1]], [64, 64, 32]),
            "mthreads TLE SQMMA CGA layout basis 0 contains a negative value",
        ),
    ],
)
def test_mthreads_tle_musa_mma_layout_builders_reject_invalid_contracts(method, args, diagnostic):
    _, builder = _make_mthreads_builder()
    with pytest.raises(ValueError, match=re.escape(diagnostic)):
        getattr(builder, method)(*args)


def test_mthreads_tle_set_layout_dialect_round_trip(tmp_path):
    fixture = tmp_path / "mthreads_tle_set_layout.mlir"
    fixture.write_text("""
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
#wmma = #ttg.musa_wmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#sqmma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [64, 64, 32]}>
#wmma_lhs = #ttg.dot_op<{opIdx = 0, parent = #wmma}>
#sqmma_rhs = #ttg.dot_op<{opIdx = 1, parent = #sqmma}>
module {
  tt.func public @blocked(%arg0: tensor<16x16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #blocked} : tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
  tt.func public @slice(%arg0: tensor<16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #slice} : tensor<16xf32> -> tensor<16xf32>
    tt.return
  }
  tt.func public @wmma(%arg0: tensor<16x16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #wmma} : tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
  tt.func public @sqmma(%arg0: tensor<64x64xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #sqmma} : tensor<64x64xf32> -> tensor<64x64xf32>
    tt.return
  }
  tt.func public @wmma_lhs(%arg0: tensor<16x16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #wmma_lhs} : tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
  tt.func public @sqmma_rhs(%arg0: tensor<64x64xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #sqmma_rhs} : tensor<64x64xf32> -> tensor<64x64xf32>
    tt.return
  }
}
""")

    context, _ = _make_mthreads_builder()
    module = ir.parse_mlir_module(str(fixture), context)
    printed = str(module)

    assert printed.count("musa_tle.set_layout") == 6
    assert "tle.gpu.set_layout" not in printed
    assert "#ttg.blocked" in printed
    assert "#ttg.slice" in printed
    assert "#ttg.musa_wmma" in printed
    assert "#ttg.musa_sqmma" in printed
    assert "#ttg.dot_op" in printed


@pytest.mark.parametrize(
    "tensor_type,target_encoding,diagnostic",
    [
        ("tensor<16x16xf32>", "#shared_a", "target_encoding must be a distributed encoding"),
        ("tensor<16x16xf32>", '"not-a-layout"', "target_encoding must be a distributed encoding"),
        (
            "tensor<16x16xf32>",
            "#layout_1d_a",
            "target encoding rank 1 must match source tensor rank 2",
        ),
        ("tensor<16xf32>", "#layout_a", "target encoding rank 2 must match source tensor rank 1"),
    ],
)
def test_mthreads_tle_set_layout_verifier_rejects_invalid_target_encoding(tmp_path, capfd, tensor_type, target_encoding,
                                                                          diagnostic):
    fixture = tmp_path / "mthreads_tle_invalid_set_layout_target.mlir"
    fixture.write_text(f"""{_CONVERSION_LAYOUTS}
module {{
  tt.func public @invalid_target(%arg0: {tensor_type}) {{
    %0 = musa_tle.set_layout %arg0 {{target_encoding = {target_encoding}}} : {tensor_type} -> {tensor_type}
    tt.return
  }}
}}
""")

    context, _ = _make_mthreads_builder()
    with pytest.raises(RuntimeError):
        ir.parse_mlir_module(str(fixture), context)

    error = capfd.readouterr().err
    assert diagnostic in error
    assert "Assertion" not in error
    assert "signal caught" not in error
    assert "PassManager::run failed" not in error


@pytest.mark.parametrize(
    "tensor_type,target_layout,expected_layout",
    [
        ("tensor<16x16xf32>", "#layout_a", "#ttg.blocked"),
        ("tensor<16xf32>", "#slice_a", "#ttg.slice"),
    ],
)
def test_mthreads_tle_set_layout_lowers_identity_without_convert(tmp_path, tensor_type, target_layout, expected_layout):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        f"""
  tt.func public @identity(%arg0: {tensor_type}) {{
    %0 = musa_tle.set_layout %arg0 {{target_encoding = {target_layout}}} : {tensor_type} -> {tensor_type}
    tt.return
  }}
""",
    )

    function_line = next(line for line in ttgir.splitlines() if "@identity" in line)
    assert ", #" in function_line
    assert expected_layout in ttgir
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_updates_dense_constant_type(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @constant_source() {
    %value = arith.constant dense<0.0> : tensor<16x16xf32>
    %0 = musa_tle.set_layout %value {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
""",
    )

    constant_line = next(line for line in ttgir.splitlines() if "arith.constant" in line)
    target = _explicit_result_encoding(constant_line)
    assert _type_encoding_aliases(constant_line) == [target]
    assert re.search(
        rf"arith\.constant(?: \{{[^}}]*\}})? dense<0\.000000e\+00> : "
        rf"tensor<16x16xf32, {re.escape(target)}>",
        ttgir,
    ), ttgir
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_merges_shared_root_uses_in_one_domain(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @shared_root_one_domain() {
    %seed_value = arith.constant dense<0> : tensor<128xi32>
    %seed = musa_tle.set_layout %seed_value {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    %shared = arith.constant dense<1> : tensor<128xi32>
    %sum0 = arith.addi %seed, %shared : tensor<128xi32>
    %sum1 = arith.addi %seed, %shared : tensor<128xi32>
    tt.return
  }
""",
    )

    shared_lines = [line for line in ttgir.splitlines() if "arith.constant" in line and "dense<1>" in line]
    assert len(shared_lines) == 1
    shared_encoding = _explicit_result_encoding(shared_lines[0])
    add_lines = [line for line in ttgir.splitlines() if "arith.addi" in line]
    assert len(add_lines) == 2
    assert all(shared_encoding in line for line in add_lines)
    assert "ttg.convert_layout" not in ttgir


def test_mthreads_tle_set_layout_isolates_shared_rank3_slice_domains(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @shared_rank3_slice_domains() {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %shared = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %rows = tt.expand_dims %shared {axis = 2 : i32} : tensor<1x128xi32> -> tensor<1x128x1xi32>
    %cols = tt.expand_dims %shared {axis = 1 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
    %row_layout = musa_tle.set_layout %rows {target_encoding = #layout_3d_a} : tensor<1x128x1xi32> -> tensor<1x128x1xi32>
    %col_layout = musa_tle.set_layout %cols {target_encoding = #layout_3d_b} : tensor<1x1x128xi32> -> tensor<1x1x128xi32>
    tt.return
  }
""",
    )

    range_lines = [line for line in ttgir.splitlines() if "tt.make_range" in line]
    shared_expand_lines = [line for line in ttgir.splitlines() if "tt.expand_dims" in line and "axis = 0" in line]
    branch_expand_lines = [
        line for line in ttgir.splitlines() if "tt.expand_dims" in line and ("axis = 1" in line or "axis = 2" in line)
    ]
    assert len(range_lines) == 2, ttgir
    assert len(shared_expand_lines) == 2, ttgir
    assert len(branch_expand_lines) == 2, ttgir
    assert len({line.split(" -> ", 1)[1] for line in shared_expand_lines}) == 2
    assert len({_explicit_result_encoding(line) for line in branch_expand_lines}) == 2
    _assert_set_layout_lowered(ttgir)


@pytest.mark.parametrize("caller_first", [False, True])
def test_mthreads_tle_set_layout_synchronizes_noinline_function_abi(tmp_path, caller_first):
    callee = """
  tt.func private @layout_callee(%arg0: tensor<128xi32>) -> (tensor<128xi32>, i32) {
    %layout = musa_tle.set_layout %arg0 {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    %c7_i32 = arith.constant 7 : i32
    tt.return %layout, %c7_i32 : tensor<128xi32>, i32
  }
"""
    caller = """
  tt.func public @layout_caller() {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %first:2 = tt.call @layout_callee(%range) : (tensor<128xi32>) -> (tensor<128xi32>, i32)
    %second:2 = tt.call @layout_callee(%range) : (tensor<128xi32>) -> (tensor<128xi32>, i32)
    tt.return
  }
"""
    body = caller + callee if caller_first else callee + caller
    ttgir = _convert_set_layout_to_ttgir(tmp_path, body)

    callee_line = next(line for line in ttgir.splitlines() if "tt.func private @layout_callee" in line)
    call_lines = [line for line in ttgir.splitlines() if "tt.call @layout_callee" in line]
    return_line = next(line for line in ttgir.splitlines() if "tt.return" in line and "i32" in line)
    encoding = _type_encoding_aliases(callee_line)[0]
    assert len(call_lines) == 2, ttgir
    assert all(encoding in line for line in call_lines), ttgir
    assert encoding in return_line, ttgir
    assert all(line.count("i32") >= 1 for line in call_lines), ttgir


def test_mthreads_tle_set_layout_synchronizes_multilevel_call_results(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @layout_root() {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %result = tt.call @layout_bridge(%range) : (tensor<128xi32>) -> tensor<128xi32>
    tt.return
  }
  tt.func private @layout_bridge(%arg0: tensor<128xi32>) -> tensor<128xi32> {
    %result = tt.call @layout_leaf(%arg0) : (tensor<128xi32>) -> tensor<128xi32>
    tt.return %result : tensor<128xi32>
  }
  tt.func private @layout_leaf(%arg0: tensor<128xi32>) -> tensor<128xi32> {
    %layout = musa_tle.set_layout %arg0 {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    tt.return %layout : tensor<128xi32>
  }
""",
    )

    function_lines = [line for line in ttgir.splitlines() if "tt.func" in line and "@layout_" in line]
    call_lines = [line for line in ttgir.splitlines() if "tt.call @layout_" in line]
    assert len(function_lines) == 3, ttgir
    assert len(call_lines) == 2, ttgir
    encoding = _type_encoding_aliases(function_lines[1])[0]
    assert all(encoding in line for line in function_lines[1:]), ttgir
    assert all(encoding in line for line in call_lines), ttgir


def test_mthreads_tle_set_layout_propagates_call_result_contract_to_callee(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func private @layout_producer() -> tensor<128xi32> {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    tt.return %range : tensor<128xi32>
  }
  tt.func public @layout_consumer() {
    %result = tt.call @layout_producer() : () -> tensor<128xi32>
    %layout = musa_tle.set_layout %result {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    tt.return
  }
""",
    )

    producer_line = next(line for line in ttgir.splitlines() if "tt.func private @layout_producer" in line)
    call_line = next(line for line in ttgir.splitlines() if "tt.call @layout_producer" in line)
    producer_return = next(line for line in ttgir.splitlines() if "tt.return" in line and "tensor<128xi32" in line)
    encoding = _type_encoding_aliases(producer_line)[0]
    assert encoding in call_line, ttgir
    assert encoding in producer_return, ttgir


def test_mthreads_tle_set_layout_unifies_encoded_and_unencoded_returns(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func private @mixed_returns(%condition: i1) -> tensor<128xi32> {
    cf.cond_br %condition, ^bb1, ^bb2
  ^bb1:
    %range_a = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %layout_a = musa_tle.set_layout %range_a {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    tt.return %layout_a : tensor<128xi32>
  ^bb2:
    %range_b = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    tt.return %range_b : tensor<128xi32>
  }
""",
    )

    function_line = next(line for line in ttgir.splitlines() if "tt.func private @mixed_returns" in line)
    return_lines = [line for line in ttgir.splitlines() if "tt.return" in line]
    range_lines = [line for line in ttgir.splitlines() if "tt.make_range" in line]
    encoding = _type_encoding_aliases(function_line)[0]
    assert len(return_lines) == 2, ttgir
    assert len(range_lines) == 2, ttgir
    assert all(encoding in line for line in return_lines), ttgir
    assert all(encoding in line for line in range_lines), ttgir


def test_mthreads_tle_set_layout_rejects_conflicting_return_abi(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _convert_set_layout_to_ttgir(
            tmp_path,
            """
  tt.func private @conflicting_return() -> tensor<128xi32> {
    %condition = arith.constant true
    cf.cond_br %condition, ^bb1, ^bb2
  ^bb1:
    %range_a = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %layout_a = musa_tle.set_layout %range_a {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    tt.return %layout_a : tensor<128xi32>
  ^bb2:
    %range_b = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %layout_b = musa_tle.set_layout %range_b {target_encoding = #layout_1d_b} : tensor<128xi32> -> tensor<128xi32>
    tt.return %layout_b : tensor<128xi32>
  }
""",
        )

    error = capfd.readouterr().err
    assert "conflicting MUSA TLE ABI encodings for result #0" in error
    assert "sizePerThread = [1]" in error
    assert "sizePerThread = [2]" in error


def test_mthreads_tle_set_layout_rejects_conflicting_call_argument_abi(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _convert_set_layout_to_ttgir(
            tmp_path,
            """
  tt.func private @argument_callee(%arg0: tensor<128xi32>) {
    tt.return
  }
  tt.func public @argument_caller() {
    %range_a = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %range_b = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %layout_a = musa_tle.set_layout %range_a {target_encoding = #layout_1d_a} : tensor<128xi32> -> tensor<128xi32>
    %layout_b = musa_tle.set_layout %range_b {target_encoding = #layout_1d_b} : tensor<128xi32> -> tensor<128xi32>
    tt.call @argument_callee(%layout_a) : (tensor<128xi32>) -> ()
    tt.call @argument_callee(%layout_b) : (tensor<128xi32>) -> ()
    tt.return
  }
""",
        )

    error = capfd.readouterr().err
    assert "conflicting MUSA TLE ABI encodings for argument #0" in error
    assert "sizePerThread = [1]" in error
    assert "sizePerThread = [2]" in error


def test_mthreads_tle_set_layout_does_not_clone_memory_roots(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @shared_load(%ptrs: tensor<32x!tt.ptr<f32>>) {
    %loaded = tt.load %ptrs : tensor<32x!tt.ptr<f32>>
    %lhs = musa_tle.set_layout %loaded {target_encoding = #layout_1d_a} : tensor<32xf32> -> tensor<32xf32>
    %rhs = musa_tle.set_layout %loaded {target_encoding = #layout_1d_b} : tensor<32xf32> -> tensor<32xf32>
    tt.return
  }
""",
    )

    assert ttgir.count("tt.load") == 1
    assert ttgir.count("ttg.convert_layout") == 1
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_root_isolation_is_noop_without_set_layout(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @shared_root_without_set_layout() {
    %shared = arith.constant dense<1> : tensor<128xi32>
    %sum0 = arith.addi %shared, %shared : tensor<128xi32>
    %sum1 = arith.addi %shared, %shared : tensor<128xi32>
    tt.return
  }
""",
    )

    shared_lines = [line for line in ttgir.splitlines() if "arith.constant" in line and "dense<1>" in line]
    assert len(shared_lines) == 1
    assert "tle.explicit_encoding." not in ttgir


def test_mthreads_tle_set_layout_hard_hint_overrides_soft_hint(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @hard_over_soft(%arg0: tensor<16x16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = musa_tle.set_layout %0 {target_encoding = #layout_b} : tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
""",
    )

    convert_lines = [line for line in ttgir.splitlines() if "ttg.convert_layout" in line]
    assert len(convert_lines) == 1
    explicit, source, result = _convert_layout_encodings(convert_lines[0])
    assert source != result
    assert explicit in result
    assert explicit not in source
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_merges_matching_hard_hints(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @matching_hard_hints(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = musa_tle.set_layout %arg1 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %2 = arith.addf %0, %1 : tensor<16x16xf32>
    tt.return
  }
""",
    )

    add_line = next(line for line in ttgir.splitlines() if "arith.addf" in line)
    target = _explicit_result_encoding(add_line)
    assert _type_encoding_aliases(add_line) == [target]
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_rejects_conflicting_hard_hints(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _convert_set_layout_to_ttgir(
            tmp_path,
            """
  tt.func public @conflicting_hard_hints(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) {
    %0 = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %1 = musa_tle.set_layout %arg1 {target_encoding = #layout_b} : tensor<16x16xf32> -> tensor<16x16xf32>
    %2 = arith.addf %0, %1 : tensor<16x16xf32>
    tt.return
  }
""",
        )

    diagnostic = capfd.readouterr().err
    assert "found conflicting MUSA TLE encoding hints for value" in diagnostic
    assert "threadsPerWarp = [1, 32]" in diagnostic
    assert "threadsPerWarp = [32, 1]" in diagnostic


def test_mthreads_tle_hint_pass_is_noop_without_set_layout(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @without_set_layout(%arg0: f32, %arg1: f32) {
    %0 = arith.addf %arg0, %arg1 : f32
    tt.return
  }
""",
    )

    assert "musa_tle.set_layout" not in ttgir
    assert "arith.addf" in ttgir


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_mthreads_tle_set_layout_propagates_through_scf_for(tmp_path, direction):
    if direction == "forward":
        prelude = """
    %seed = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
"""
        init = "%seed"
        epilogue = """
    %use = arith.addf %loop, %loop : tensor<16x16xf32>
"""
    else:
        prelude = ""
        init = "%arg0"
        epilogue = """
    %encoded = musa_tle.set_layout %loop {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
"""

    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        f"""
  tt.func public @for_{direction}(%arg0: tensor<16x16xf32>) {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
{prelude}
    %loop = scf.for %iv = %c0 to %c2 step %c1 iter_args(%iter = {init}) -> (tensor<16x16xf32>) : i32 {{
      %next = arith.addf %iter, %iter : tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }}
{epilogue}
    tt.return
  }}
""",
    )

    add_line = next(line for line in ttgir.splitlines() if "arith.addf" in line)
    target = _explicit_result_encoding(add_line)
    for line in ttgir.splitlines():
        if "scf.for" in line or "scf.yield" in line or "arith.addf" in line:
            assert target in line, line
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_propagates_through_scf_while(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @while_forward(%arg0: tensor<16x16xf32>) {
    %true = arith.constant true
    %seed = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %loop = scf.while (%before = %seed) : (tensor<16x16xf32>) -> (tensor<16x16xf32>) {
      scf.condition(%true) %before : tensor<16x16xf32>
    } do {
    ^bb0(%after: tensor<16x16xf32>):
      %next = arith.addf %after, %after : tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    %use = arith.addf %loop, %loop : tensor<16x16xf32>
    tt.return
  }
""",
    )

    add_line = next(line for line in ttgir.splitlines() if "arith.addf" in line)
    target = _explicit_result_encoding(add_line)
    for line in ttgir.splitlines():
        if any(op in line for op in ("scf.while", "scf.condition", "scf.yield", "arith.addf")):
            assert target in line, line
    assert re.search(rf"\^bb0\([^\n]*{re.escape(target)}", ttgir), ttgir
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_propagates_from_while_condition_and_yield(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @while_condition(%arg0: tensor<16x16xf32>) {
    %true = arith.constant true
    %loop = scf.while (%before = %arg0) : (tensor<16x16xf32>) -> (tensor<16x16xf32>) {
      %encoded = musa_tle.set_layout %before {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
      scf.condition(%true) %encoded : tensor<16x16xf32>
    } do {
    ^bb0(%after: tensor<16x16xf32>):
      scf.yield %after : tensor<16x16xf32>
    }
    tt.return
  }
  tt.func public @while_yield(%arg0: tensor<16x16xf32>) {
    %true = arith.constant true
    %loop = scf.while (%before = %arg0) : (tensor<16x16xf32>) -> (tensor<16x16xf32>) {
      scf.condition(%true) %before : tensor<16x16xf32>
    } do {
    ^bb0(%after: tensor<16x16xf32>):
      %encoded = musa_tle.set_layout %after {target_encoding = #layout_b} : tensor<16x16xf32> -> tensor<16x16xf32>
      scf.yield %encoded : tensor<16x16xf32>
    }
    tt.return
  }
""",
    )

    function_lines = [line for line in ttgir.splitlines() if "tt.func public @while_" in line]
    assert len(function_lines) == 2
    condition_layout = _type_encoding_aliases(function_lines[0])[0]
    yield_layout = _type_encoding_aliases(function_lines[1])[0]
    assert condition_layout != yield_layout
    while_lines = [line for line in ttgir.splitlines() if "scf.while" in line]
    condition_lines = [line for line in ttgir.splitlines() if "scf.condition" in line]
    yield_lines = [line for line in ttgir.splitlines() if "scf.yield" in line]
    assert condition_layout in while_lines[0] and condition_layout in condition_lines[0]
    assert yield_layout in while_lines[1] and yield_layout in yield_lines[1]
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


def test_mthreads_tle_set_layout_propagates_through_scf_if(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @if_forward(%arg0: tensor<16x16xf32>) {
    %true = arith.constant true
    %seed = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %branch = scf.if %true -> (tensor<16x16xf32>) {
      scf.yield %seed : tensor<16x16xf32>
    } else {
      scf.yield %arg0 : tensor<16x16xf32>
    }
    %use = arith.addf %branch, %branch : tensor<16x16xf32>
    tt.return
  }
""",
    )

    add_line = next(line for line in ttgir.splitlines() if "arith.addf" in line)
    target = _explicit_result_encoding(add_line)
    for line in ttgir.splitlines():
        if "scf.if" in line or "scf.yield" in line or "arith.addf" in line:
            assert target in line, line
    assert "ttg.convert_layout" not in ttgir
    _assert_set_layout_lowered(ttgir)


@pytest.mark.parametrize("control_flow", ["for", "if"])
def test_mthreads_tle_set_layout_rejects_control_flow_hard_conflicts(tmp_path, capfd, control_flow):
    if control_flow == "for":
        body = """
  tt.func public @for_conflict(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %init = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %yielded = musa_tle.set_layout %arg1 {target_encoding = #layout_b} : tensor<16x16xf32> -> tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %c2 step %c1 iter_args(%iter = %init) -> (tensor<16x16xf32>) : i32 {
      scf.yield %yielded : tensor<16x16xf32>
    }
    tt.return
  }
"""
    else:
        body = """
  tt.func public @if_conflict(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) {
    %true = arith.constant true
    %then = musa_tle.set_layout %arg0 {target_encoding = #layout_a} : tensor<16x16xf32> -> tensor<16x16xf32>
    %else = musa_tle.set_layout %arg1 {target_encoding = #layout_b} : tensor<16x16xf32> -> tensor<16x16xf32>
    %branch = scf.if %true -> (tensor<16x16xf32>) {
      scf.yield %then : tensor<16x16xf32>
    } else {
      scf.yield %else : tensor<16x16xf32>
    }
    tt.return
  }
"""

    with pytest.raises(RuntimeError):
        _convert_set_layout_to_ttgir(tmp_path, body)

    diagnostic = capfd.readouterr().err
    assert "found conflicting MUSA TLE encoding hints for value" in diagnostic
    assert "threadsPerWarp = [1, 32]" in diagnostic
    assert "threadsPerWarp = [32, 1]" in diagnostic


def test_mthreads_tle_set_layout_propagates_dot_accumulator_only(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @chained_dot_accumulator() {
    %lhs_value = arith.constant dense<0.0> : tensor<16x16xbf16>
    %lhs_encoded = musa_tle.set_layout %lhs_value {target_encoding = #lhs} : tensor<16x16xbf16> -> tensor<16x16xbf16>
    %rhs_value = arith.constant dense<0.0> : tensor<16x16xbf16>
    %rhs_encoded = musa_tle.set_layout %rhs_value {target_encoding = #rhs} : tensor<16x16xbf16> -> tensor<16x16xbf16>
    %acc_value = arith.constant dense<0.0> : tensor<16x16xf32>
    %acc_encoded = musa_tle.set_layout %acc_value {target_encoding = #mma} : tensor<16x16xf32> -> tensor<16x16xf32>
    %first = tt.dot %lhs_encoded, %rhs_encoded, %acc_encoded : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
    %second = tt.dot %lhs_encoded, %rhs_encoded, %first : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
    tt.return
  }
""",
    )

    dot_lines = [line for line in ttgir.splitlines() if "tt.dot" in line]
    assert len(dot_lines) == 2
    for line in dot_lines:
        assert "#ttg.dot_op<{opIdx = 0, parent = #mma}>" in line
        assert "#ttg.dot_op<{opIdx = 1, parent = #mma}>" in line
        assert "#mma" in line
    assert "#ttg.musa_wmma" in ttgir
    assert "ttg.convert_layout" not in ttgir


def test_mthreads_tle_set_layout_propagates_into_local_pointers_and_load(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @local_pointer_forward(%idx: tensor<32xi32>) {
    %encoded_idx = musa_tle.set_layout %idx {target_encoding = #layout_1d_a} : tensor<32xi32> -> tensor<32xi32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %encoded_idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32>) -> tensor<32x!tt.ptr<f32, 3>>
    %loaded = tt.load %ptrs : tensor<32x!tt.ptr<f32, 3>>
    tt.return
  }
""",
    )

    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    load_line = next(line for line in ttgir.splitlines() if "tt.load" in line)
    local_encoding = re.search(r"tle\.explicit_encoding\.0 = (#\w+)", local_pointer_line).group(1)
    assert local_encoding in local_pointer_line
    assert f"tle.explicit_memory_encoding = {local_encoding}" in load_line
    assert local_encoding in load_line
    assert re.search(r"(?<!musa_)tle\.local_pointers", ttgir) is None


def test_mthreads_tle_set_layout_propagates_from_local_pointer_to_index(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @local_pointer_backward(%idx: tensor<32xi32>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32>) -> tensor<32x!tt.ptr<f32, 3>>
    %encoded_ptrs = musa_tle.set_layout %ptrs {target_encoding = #layout_1d_b} : tensor<32x!tt.ptr<f32, 3>> -> tensor<32x!tt.ptr<f32, 3>>
    tt.return
  }
""",
    )

    function_line = next(line for line in ttgir.splitlines() if "@local_pointer_backward" in line)
    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    local_encoding = re.search(r"tle\.explicit_encoding\.0 = (#\w+)", local_pointer_line).group(1)
    assert local_encoding in function_line
    assert local_encoding in local_pointer_line


def test_mthreads_tle_set_layout_marks_local_pointer_memory_ops(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @local_pointer_memory_ops(%idx: tensor<32xi32>) {
    %encoded_idx = musa_tle.set_layout %idx {target_encoding = #layout_1d_a} : tensor<32xi32> -> tensor<32xi32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %encoded_idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32>) -> tensor<32x!tt.ptr<f32, 3>>
    %value = arith.constant dense<1.0> : tensor<32xf32>
    %loaded = tt.load %ptrs : tensor<32x!tt.ptr<f32, 3>>
    tt.store %ptrs, %value : tensor<32x!tt.ptr<f32, 3>>
    %rmw = tt.atomic_rmw fadd, relaxed, cta, %ptrs, %value : (tensor<32x!tt.ptr<f32, 3>>, tensor<32xf32>) -> tensor<32xf32>
    %cas = tt.atomic_cas relaxed, cta, %ptrs, %value, %value : (tensor<32x!tt.ptr<f32, 3>>, tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    tt.return
  }
""",
    )

    memory_lines = [
        line for line in ttgir.splitlines()
        if any(op in line for op in ("tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas"))
    ]
    assert len(memory_lines) == 4
    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    local_encoding = re.search(r"tle\.explicit_encoding\.0 = (#\w+)", local_pointer_line).group(1)
    for line in memory_lines:
        assert f"tle.explicit_memory_encoding = {local_encoding}" in line, line
        assert local_encoding in line, line


def test_mthreads_tle_set_layout_does_not_mark_unconstrained_memory_ops(tmp_path):
    ttgir = _convert_set_layout_to_ttgir(
        tmp_path,
        """
  tt.func public @unconstrained_memory(%base: !tt.ptr<f32>) {
    %ptrs = tt.splat %base : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %loaded = tt.load %ptrs : tensor<32x!tt.ptr<f32>>
    tt.return
  }
""",
    )

    assert "tt.load" in ttgir
    assert "tle.explicit_memory_encoding" not in ttgir


def test_mthreads_tle_set_layout_rejects_conflicting_memory_encodings(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _convert_set_layout_to_ttgir(
            tmp_path,
            """
  tt.func public @conflicting_memory(%idx: tensor<32xi32>) {
    %encoded_idx = musa_tle.set_layout %idx {target_encoding = #layout_1d_a} : tensor<32xi32> -> tensor<32xi32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %encoded_idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32>) -> tensor<32x!tt.ptr<f32, 3>>
    %value = arith.constant dense<1.0> : tensor<32xf32>
    %encoded_value = musa_tle.set_layout %value {target_encoding = #layout_1d_b} : tensor<32xf32> -> tensor<32xf32>
    tt.store %ptrs, %encoded_value : tensor<32x!tt.ptr<f32, 3>>
    tt.return
  }
""",
        )

    diagnostic = capfd.readouterr().err
    assert "has conflicting explicit MUSA TLE memory encodings" in diagnostic
    assert "sizePerThread = [1]" in diagnostic
    assert "sizePerThread = [2]" in diagnostic


def test_mthreads_tle_coalesce_preserves_explicit_load_layout(tmp_path):
    ttgir = _run_ttgir_coalesce(
        tmp_path,
        """
  tt.func public @explicit_rebuild(%base: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout_1d_b>
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #layout_1d_b>
    %ptrs = tt.addptr %bases, %offsets : tensor<1024x!tt.ptr<f32>, #layout_1d_b>, tensor<1024xi32, #layout_1d_b>
    %loaded = tt.load %ptrs {tle.explicit_encoding.0 = #layout_1d_b, tle.explicit_memory_encoding = #layout_1d_b} : tensor<1024x!tt.ptr<f32>, #layout_1d_b>
    tt.return
  }
""",
    )

    load_line = next(line for line in ttgir.splitlines() if "tt.load" in line)
    memory_match = re.search(r"tle\.explicit_memory_encoding = (#\w+)", load_line)
    assert memory_match is not None, load_line
    memory_encoding = memory_match.group(1)
    result_match = re.search(r"tle\.explicit_encoding\.0 = (#\w+)", load_line)
    assert result_match is not None, load_line
    assert result_match.group(1) == memory_encoding
    assert memory_encoding in load_line
    assert "ttg.convert_layout" not in ttgir


def test_mthreads_tle_coalesce_preserves_explicit_store_layout(tmp_path):
    ttgir = _run_ttgir_coalesce(
        tmp_path,
        """
  tt.func public @explicit_store(%base: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout_1d_b>
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #layout_1d_b>
    %ptrs = tt.addptr %bases, %offsets : tensor<1024x!tt.ptr<f32>, #layout_1d_b>, tensor<1024xi32, #layout_1d_b>
    %value = arith.constant dense<1.0> : tensor<1024xf32, #layout_1d_b>
    tt.store %ptrs, %value {tle.explicit_memory_encoding = #layout_1d_b} : tensor<1024x!tt.ptr<f32>, #layout_1d_b>
    tt.return
  }
""",
    )

    store_line = next(line for line in ttgir.splitlines() if "tt.store" in line)
    memory_match = re.search(r"tle\.explicit_memory_encoding = (#\w+)", store_line)
    assert memory_match is not None, store_line
    assert memory_match.group(1) in store_line
    assert "ttg.convert_layout" not in ttgir


def test_mthreads_tle_coalesce_rejects_pointer_memory_layout_conflict(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _run_ttgir_coalesce(
            tmp_path,
            """
  tt.func public @conflicting_pointer_memory_layout(%base: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout_1d_a>
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #layout_1d_a>
    %ptrs = tt.addptr %bases, %offsets : tensor<1024x!tt.ptr<f32>, #layout_1d_a>, tensor<1024xi32, #layout_1d_a>
    %loaded = tt.load %ptrs {tle.explicit_memory_encoding = #layout_1d_b} : tensor<1024x!tt.ptr<f32>, #layout_1d_a>
    tt.return
  }
""",
        )

    diagnostic = capfd.readouterr().err
    assert "has explicit MUSA TLE memory encoding that does not match the pointer tensor encoding" in diagnostic
    assert "pointer:" in diagnostic
    assert "explicit memory:" in diagnostic
    assert "sizePerThread = [1]" in diagnostic
    assert "sizePerThread = [2]" in diagnostic


def test_mthreads_tle_coalesce_keeps_ordinary_memory_optimization(tmp_path):
    ttgir = _run_ttgir_coalesce(
        tmp_path,
        """
  tt.func public @ordinary_rebuild(%base: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout_1d_a>
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #layout_1d_a>
    %ptrs = tt.addptr %bases, %offsets : tensor<1024x!tt.ptr<f32>, #layout_1d_a>, tensor<1024xi32, #layout_1d_a>
    %loaded = tt.load %ptrs : tensor<1024x!tt.ptr<f32>, #layout_1d_a>
    %value = arith.constant dense<1.0> : tensor<1024xf32, #layout_1d_a>
    tt.store %ptrs, %value : tensor<1024x!tt.ptr<f32>, #layout_1d_a>
    tt.return
  }
""",
    )

    assert "tt.load" in ttgir
    assert "tt.store" in ttgir
    assert "ttg.convert_layout" in ttgir
    assert "tle.explicit_encoding." not in ttgir
    assert "tle.explicit_memory_encoding" not in ttgir


@pytest.mark.parametrize("hard", [False, True])
def test_mthreads_tle_coalesce_descriptor_transfers_explicit_result_layout(tmp_path, hard):
    attrs = " {tle.explicit_encoding.0 = #layout_a}" if hard else ""
    after_coalesce, finalized = _run_ttgir_coalesce_and_finalize_explicit_layouts(
        tmp_path,
        f"""
  tt.func public @descriptor_rebuild(%desc: !tt.tensordesc<tensor<16x16xf32, #shared_a>>) -> tensor<16x16xf32, #layout_a> {{
    %c0 = arith.constant 0 : i32
    %loaded = tt.descriptor_load %desc[%c0, %c0]{attrs} : !tt.tensordesc<tensor<16x16xf32, #shared_a>> -> tensor<16x16xf32, #layout_a>
    tt.return %loaded : tensor<16x16xf32, #layout_a>
  }}
""",
    )

    descriptor = next(line for line in after_coalesce.splitlines() if "tt.descriptor_load" in line)
    bridge = next(line for line in after_coalesce.splitlines() if "ttg.convert_layout" in line)
    optimized_encoding, hard_encoding = _type_encoding_aliases(bridge)
    assert optimized_encoding != hard_encoding
    assert _type_encoding_aliases(descriptor)[-1] == optimized_encoding
    assert "tle.explicit_encoding." not in descriptor
    assert "tle.explicit_memory_encoding" not in after_coalesce
    if hard:
        assert after_coalesce.count("tle.explicit_encoding.0") == 1
        assert _explicit_result_encoding(bridge) == hard_encoding
    else:
        assert "tle.explicit_encoding." not in after_coalesce

    assert "tle.explicit_encoding." not in finalized
    assert "tle.explicit_memory_encoding" not in finalized
    finalized_descriptor = next(line for line in finalized.splitlines() if "tt.descriptor_load" in line)
    finalized_bridge = next(line for line in finalized.splitlines() if "ttg.convert_layout" in line)
    finalized_source, finalized_target = _type_encoding_aliases(finalized_bridge)
    assert finalized_source != finalized_target
    assert finalized_source == optimized_encoding
    assert finalized_target == hard_encoding
    assert _type_encoding_aliases(finalized_descriptor)[-1] == finalized_source


def test_mthreads_tle_remove_layout_conversions_prefers_hard_encoding(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @hard_over_soft(%arg0: tensor<16x16xf32, #layout_a>, %arg1: tensor<16x16xf32, #layout_a>, %out: tensor<16x16x!tt.ptr<f32>, #layout_b>) {
    %soft = ttg.convert_layout %arg0 : tensor<16x16xf32, #layout_a> -> tensor<16x16xf32, #layout_b>
    %hard = ttg.convert_layout %arg1 {tle.explicit_encoding.0 = #layout_b} : tensor<16x16xf32, #layout_a> -> tensor<16x16xf32, #layout_b>
    %sum = arith.addf %soft, %hard : tensor<16x16xf32, #layout_b>
    tt.store %out, %sum : tensor<16x16x!tt.ptr<f32>, #layout_b>
    tt.return
  }
""",
    )

    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    hard_encoding, _, hard_result = _convert_layout_encodings(hard_line)
    assert hard_encoding in hard_result
    add_line = next(line for line in ttgir.splitlines() if "arith.addf" in line)
    assert hard_encoding in add_line


def test_mthreads_tle_remove_layout_conversions_preserves_rematerialization_anchor(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @hard_rematerialization_boundary(%lhs_scalar: f32, %rhs_scalar: f32, %out: tensor<16x16x!tt.ptr<f32>, #layout_b>) {
    %lhs = tt.splat %lhs_scalar : f32 -> tensor<16x16xf32, #layout_a>
    %rhs = tt.splat %rhs_scalar : f32 -> tensor<16x16xf32, #layout_a>
    %sum = arith.addf %lhs, %rhs : tensor<16x16xf32, #layout_a>
    %hard = ttg.convert_layout %sum {tle.explicit_encoding.0 = #layout_b} : tensor<16x16xf32, #layout_a> -> tensor<16x16xf32, #layout_b>
    %use = arith.mulf %hard, %hard : tensor<16x16xf32, #layout_b>
    tt.store %out, %use : tensor<16x16x!tt.ptr<f32>, #layout_b>
    tt.return
  }
""",
    )

    assert ttgir.count("tle.explicit_encoding.0") == 1
    assert ttgir.count("ttg.convert_layout") == 1
    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    explicit, _, result = _convert_layout_encodings(hard_line)
    assert explicit in result


def test_mthreads_tle_remove_layout_conversions_does_not_hoist_hard_convert(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @hard_hoist_boundary(%arg0: tensor<16x16xi8, #layout_a>, %out: tensor<16x16x!tt.ptr<i32>, #layout_b>) {
    %extended = arith.extui %arg0 : tensor<16x16xi8, #layout_a> to tensor<16x16xi32, #layout_a>
    %hard = ttg.convert_layout %extended {tle.explicit_encoding.0 = #layout_b} : tensor<16x16xi32, #layout_a> -> tensor<16x16xi32, #layout_b>
    %use = arith.addi %hard, %hard : tensor<16x16xi32, #layout_b>
    tt.store %out, %use : tensor<16x16x!tt.ptr<i32>, #layout_b>
    tt.return
  }
""",
    )

    lines = ttgir.splitlines()
    ext_index = next(i for i, line in enumerate(lines) if "arith.extui" in line)
    hard_index = next(i for i, line in enumerate(lines) if "tle.explicit_encoding.0" in line)
    ext_result = re.search(r"(%[\w]+) = arith\.extui", lines[ext_index]).group(1)
    assert ext_index < hard_index
    assert ext_result in lines[hard_index]
    assert ttgir.count("tle.explicit_encoding.0") == 1


def test_mthreads_tle_remove_layout_conversions_does_not_fold_hard_source(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @hard_local_store_boundary(%arg0: tensor<32xf32, #layout_1d_a>) {
    %hard = ttg.convert_layout %arg0 {tle.explicit_encoding.0 = #layout_1d_b} : tensor<32xf32, #layout_1d_a> -> tensor<32xf32, #layout_1d_b>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    ttg.local_store %hard, %buf : tensor<32xf32, #layout_1d_b> -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    tt.return
  }
""",
    )

    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    hard_result = re.search(r"(%[\w]+) = ttg\.convert_layout", hard_line).group(1)
    hard_encoding, _, _ = _convert_layout_encodings(hard_line)
    store_line = next(line for line in ttgir.splitlines() if "ttg.local_store" in line)
    assert hard_result in store_line
    assert hard_encoding in store_line


def test_mthreads_tle_remove_layout_conversions_preserves_hard_anchor_across_rounds(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @repeated_hard_anchor(%out: tensor<16x16x!tt.ptr<f32>, #layout_b>) {
    %value = arith.constant dense<1.0> : tensor<16x16xf32, #layout_a>
    %hard = ttg.convert_layout %value {tle.explicit_encoding.0 = #layout_b} : tensor<16x16xf32, #layout_a> -> tensor<16x16xf32, #layout_b>
    %use = arith.addf %hard, %hard : tensor<16x16xf32, #layout_b>
    tt.store %out, %use : tensor<16x16x!tt.ptr<f32>, #layout_b>
    tt.return
  }
""",
        repeat=2,
    )

    assert ttgir.count("ttg.convert_layout") == 1
    assert ttgir.count("tle.explicit_encoding.0") == 1
    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    explicit, _, result = _convert_layout_encodings(hard_line)
    assert explicit in result


def test_mthreads_tle_remove_layout_conversions_keeps_ordinary_cleanup(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @ordinary_redundant_convert(%arg0: tensor<16x16xf32, #layout_a>) {
    %redundant = ttg.convert_layout %arg0 : tensor<16x16xf32, #layout_a> -> tensor<16x16xf32, #layout_a>
    %use = arith.addf %redundant, %redundant : tensor<16x16xf32, #layout_a>
    tt.return
  }
""",
    )

    assert "ttg.convert_layout" not in ttgir
    assert "tle.explicit_encoding." not in ttgir


def test_mthreads_tle_remove_layout_conversions_does_not_retarget_hard_sqmma_store(tmp_path):
    ttgir = _run_ttgir_remove_layout_conversions(
        tmp_path,
        """
  tt.func public @hard_sqmma_store(%ptrs: tensor<64x64x!tt.ptr<f32>, #layout_a>) {
    %value = arith.constant dense<1.0> : tensor<64x64xf32, #sqmma>
    %hard = ttg.convert_layout %value {tle.explicit_encoding.0 = #layout_a} : tensor<64x64xf32, #sqmma> -> tensor<64x64xf32, #layout_a>
    tt.store %ptrs, %hard : tensor<64x64x!tt.ptr<f32>, #layout_a>
    tt.return
  }
""",
        extra_module_attrs='"tle.enable_encoding_rematerialization" = true',
    )

    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    hard_result = re.search(r"(%[\w]+) = ttg\.convert_layout", hard_line).group(1)
    hard_encoding, _, _ = _convert_layout_encodings(hard_line)
    store_line = next(line for line in ttgir.splitlines() if "tt.store" in line)
    assert hard_result in store_line
    assert hard_encoding in store_line


def test_mthreads_tle_select_encodings_prefers_hard_local_pointer_layout(tmp_path):
    ttgir = _run_ttgir_select_encodings(
        tmp_path,
        """
  tt.func public @hard_local_pointer(%idx: tensor<32xi32, #layout_1d_a>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %idx) {tle.explicit_encoding.0 = #layout_1d_a} : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32, #layout_1d_a>) -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_a>
    %adapted = ttg.convert_layout %ptrs : tensor<32x!tt.ptr<f32, 3>, #layout_1d_a> -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    %loaded = tt.load %adapted : tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    tt.return
  }
""",
    )

    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    load_line = next(line for line in ttgir.splitlines() if "tt.load" in line)
    hard_encoding = _explicit_result_encoding(local_pointer_line)
    assert hard_encoding in local_pointer_line.split(" -> ", 1)[1]
    assert hard_encoding in load_line


def test_mthreads_tle_select_encodings_uses_explicit_memory_layout(tmp_path):
    ttgir = _run_ttgir_select_encodings(
        tmp_path,
        """
  tt.func public @hard_memory_consumer(%idx: tensor<32xi32, #layout_1d_a>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32, #layout_1d_a>) -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_a>
    %adapted = ttg.convert_layout %ptrs : tensor<32x!tt.ptr<f32, 3>, #layout_1d_a> -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    %loaded = tt.load %adapted {tle.explicit_encoding.0 = #layout_1d_b, tle.explicit_memory_encoding = #layout_1d_b} : tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    tt.return
  }
""",
    )

    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    load_line = next(line for line in ttgir.splitlines() if "tt.load" in line)
    hard_encoding = _explicit_result_encoding(load_line)
    assert hard_encoding in local_pointer_line
    assert f"tle.explicit_memory_encoding = {hard_encoding}" in load_line
    assert not any("ttg.convert_layout" in line and "!tt.ptr" in line for line in ttgir.splitlines())


def test_mthreads_tle_select_encodings_rejects_conflicting_hard_layouts(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _run_ttgir_select_encodings(
            tmp_path,
            """
  tt.func public @conflicting_local_pointer(%idx: tensor<32xi32, #layout_1d_a>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %idx) {tle.explicit_encoding.0 = #layout_1d_a} : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32, #layout_1d_a>) -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_a>
    %loaded = tt.load %ptrs {tle.explicit_memory_encoding = #layout_1d_b} : tensor<32x!tt.ptr<f32, 3>, #layout_1d_a>
    tt.return
  }
""",
        )

    diagnostic = capfd.readouterr().err
    assert "has conflicting explicit MUSA TLE memory encodings" in diagnostic
    assert "sizePerThread = [1]" in diagnostic
    assert "sizePerThread = [2]" in diagnostic


def test_mthreads_tle_select_encodings_preserves_hard_pointer_boundary(tmp_path):
    ttgir = _run_ttgir_select_encodings(
        tmp_path,
        """
  tt.func public @hard_pointer_boundary(%idx: tensor<32xi32, #layout_1d_a>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32, #layout_1d_a>) -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_a>
    %hard = ttg.convert_layout %ptrs {tle.explicit_encoding.0 = #layout_1d_b} : tensor<32x!tt.ptr<f32, 3>, #layout_1d_a> -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    %loaded = tt.load %hard : tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    tt.return
  }
""",
    )

    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    assert "ttg.convert_layout" in hard_line
    explicit, source, result = _convert_layout_encodings(hard_line)
    assert explicit in result
    assert explicit not in source
    assert next(iter(_type_encoding_aliases(source))) in local_pointer_line


def test_mthreads_tle_select_encodings_keeps_ordinary_voting(tmp_path):
    ttgir = _run_ttgir_select_encodings(
        tmp_path,
        """
  tt.func public @ordinary_voting(%idx: tensor<32xi32, #layout_1d_a>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %ptrs = "musa_tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared_1d, #smem, mutable>, tensor<32xi32, #layout_1d_a>) -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_a>
    %adapted = ttg.convert_layout %ptrs : tensor<32x!tt.ptr<f32, 3>, #layout_1d_a> -> tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    %loaded0 = tt.load %adapted : tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    %loaded1 = tt.load %adapted : tensor<32x!tt.ptr<f32, 3>, #layout_1d_b>
    tt.return
  }
""",
    )

    local_pointer_line = next(line for line in ttgir.splitlines() if "musa_tle.local_pointers" in line)
    load_lines = [line for line in ttgir.splitlines() if "tt.load" in line]
    assert len(load_lines) == 2
    load_encoding = _type_encoding_aliases(load_lines[0])[0]
    assert load_encoding in local_pointer_line
    assert "tle.explicit_encoding." not in ttgir
    assert "tle.explicit_memory_encoding" not in ttgir


def test_mthreads_tle_optimize_thread_locality_preserves_hard_reshape(tmp_path):
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        """
  tt.func public @hard_reshape(%arg0: tensor<64x16xf32, #reshape_layout>) -> tensor<64xf32, #reshape_slice> {
    %reshaped = tt.reshape %arg0 allow_reorder {tle.explicit_encoding.0 = #reshape_layout} : tensor<64x16xf32, #reshape_layout> -> tensor<64x16xf32, #reshape_layout>
    %reduced = "tt.reduce"(%reshaped) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maximumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<64x16xf32, #reshape_layout>) -> tensor<64xf32, #reshape_slice>
    tt.return %reduced : tensor<64xf32, #reshape_slice>
  }
""",
        num_warps=2,
    )

    reshape_line = next(line for line in ttgir.splitlines() if "tt.reshape" in line)
    explicit = _explicit_result_encoding(reshape_line)
    assert explicit in reshape_line.split(" -> ", 1)[1]
    assert "efficient_layout" not in reshape_line
    assert "ttg.convert_layout" not in ttgir


def test_mthreads_tle_optimize_thread_locality_keeps_ordinary_reshape(tmp_path):
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        """
  tt.func public @ordinary_reshape(%arg0: tensor<64x16xf32, #reshape_layout>) -> tensor<64xf32, #reshape_slice> {
    %reshaped = tt.reshape %arg0 allow_reorder : tensor<64x16xf32, #reshape_layout> -> tensor<64x16xf32, #reshape_layout>
    %reduced = "tt.reduce"(%reshaped) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maximumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<64x16xf32, #reshape_layout>) -> tensor<64xf32, #reshape_slice>
    tt.return %reduced : tensor<64xf32, #reshape_slice>
  }
""",
        num_warps=2,
    )

    reshape_line = next(line for line in ttgir.splitlines() if "tt.reshape" in line)
    assert "efficient_layout" in reshape_line
    assert "ttg.convert_layout" in ttgir
    assert "tle.explicit_encoding." not in ttgir


def test_mthreads_tle_optimize_thread_locality_preserves_hard_gather(tmp_path):
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        """
  tt.func public @hard_gather(%src: tensor<64x64xf32, #layout_a>, %idx: tensor<64x64xi32, #layout_a>) -> tensor<64x64xf32, #layout_a> {
    %gathered = tt.gather %src[%idx] {axis = 0 : i32, tle.explicit_encoding.0 = #layout_a} : (tensor<64x64xf32, #layout_a>, tensor<64x64xi32, #layout_a>) -> tensor<64x64xf32, #layout_a>
    tt.return %gathered : tensor<64x64xf32, #layout_a>
  }
""",
    )

    gather_line = next(line for line in ttgir.splitlines() if "tt.gather" in line)
    explicit = _explicit_result_encoding(gather_line)
    assert explicit in gather_line.split(" -> ", 1)[1]
    assert "efficient_layout" not in gather_line
    assert "ttg.convert_layout" not in ttgir


def test_mthreads_tle_optimize_thread_locality_keeps_ordinary_gather(tmp_path):
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        """
  tt.func public @ordinary_gather(%src: tensor<64x64xf32, #layout_a>, %idx: tensor<64x64xi32, #layout_a>) -> tensor<64x64xf32, #layout_a> {
    %gathered = tt.gather %src[%idx] {axis = 0 : i32} : (tensor<64x64xf32, #layout_a>, tensor<64x64xi32, #layout_a>) -> tensor<64x64xf32, #layout_a>
    tt.return %gathered : tensor<64x64xf32, #layout_a>
  }
""",
    )

    gather_line = next(line for line in ttgir.splitlines() if "tt.gather" in line)
    assert "efficient_layout" in gather_line
    assert ttgir.count("ttg.convert_layout") == 3
    assert "tle.explicit_encoding." not in ttgir


def test_mthreads_tle_optimize_thread_locality_keeps_adjacent_hard_boundary(tmp_path):
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        """
  tt.func public @hard_gather_input(%src: tensor<64x64xf32, #layout_b>, %idx: tensor<64x64xi32, #layout_a>) -> tensor<64x64xf32, #layout_a> {
    %hard = ttg.convert_layout %src {tle.explicit_encoding.0 = #layout_a} : tensor<64x64xf32, #layout_b> -> tensor<64x64xf32, #layout_a>
    %gathered = tt.gather %hard[%idx] {axis = 0 : i32} : (tensor<64x64xf32, #layout_a>, tensor<64x64xi32, #layout_a>) -> tensor<64x64xf32, #layout_a>
    tt.return %gathered : tensor<64x64xf32, #layout_a>
  }
""",
    )

    hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
    explicit, source, result = _convert_layout_encodings(hard_line)
    assert explicit in result
    assert explicit not in source
    gather_line = next(line for line in ttgir.splitlines() if "tt.gather" in line)
    assert "efficient_layout" in gather_line
    assert ttgir.count("tle.explicit_encoding.0") == 1


@pytest.mark.parametrize("hard_anchor", ["reduce", "update"])
def test_mthreads_tle_optimize_thread_locality_preserves_hard_reduction(tmp_path, hard_anchor):
    reduce_attrs = " {tle.explicit_encoding.0 = #slice_dim1_a}" if hard_anchor == "reduce" else ""
    update_attrs = " {tle.explicit_encoding.0 = #slice_dim1_a}" if hard_anchor == "update" else ""
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        f"""
  tt.func public @hard_reduction(%ptrs: tensor<32x128x!tt.ptr<f32>, #layout_a>, %limit: i32) -> tensor<32xf32, #slice_dim1_a> {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.0> : tensor<32xf32, #slice_dim1_a>
    %loop = scf.for %iv = %c0 to %limit step %c1 iter_args(%acc = %init) -> (tensor<32xf32, #slice_dim1_a>) : i32 {{
      %loaded = tt.load %ptrs : tensor<32x128x!tt.ptr<f32>, #layout_a>
      %reduced = "tt.reduce"(%loaded) <{{axis = 1 : i32}}> ({{
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
      }}){reduce_attrs} : (tensor<32x128xf32, #layout_a>) -> tensor<32xf32, #slice_dim1_a>
      %updated = arith.addf %acc, %reduced{update_attrs} : tensor<32xf32, #slice_dim1_a>
      scf.yield %updated : tensor<32xf32, #slice_dim1_a>
    }}
    tt.return %loop : tensor<32xf32, #slice_dim1_a>
  }}
""",
    )

    assert ttgir.count('"tt.reduce"') == 1
    assert "tt.reshape" not in ttgir
    assert ttgir.count("tle.explicit_encoding.0") == 1


def test_mthreads_tle_optimize_thread_locality_keeps_ordinary_reduction(tmp_path):
    ttgir = _run_ttgir_optimize_thread_locality(
        tmp_path,
        """
  tt.func public @ordinary_reduction(%ptrs: tensor<32x128x!tt.ptr<f32>, #layout_a>, %limit: i32) -> tensor<32xf32, #slice_dim1_a> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.0> : tensor<32xf32, #slice_dim1_a>
    %loop = scf.for %iv = %c0 to %limit step %c1 iter_args(%acc = %init) -> (tensor<32xf32, #slice_dim1_a>) : i32 {
      %loaded = tt.load %ptrs : tensor<32x128x!tt.ptr<f32>, #layout_a>
      %reduced = "tt.reduce"(%loaded) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x128xf32, #layout_a>) -> tensor<32xf32, #slice_dim1_a>
      %updated = arith.addf %acc, %reduced : tensor<32xf32, #slice_dim1_a>
      scf.yield %updated : tensor<32xf32, #slice_dim1_a>
    }
    tt.return %loop : tensor<32xf32, #slice_dim1_a>
  }
""",
    )

    assert ttgir.count('"tt.reduce"') == 3
    assert "tt.reshape" in ttgir
    assert "efficient_layout" in ttgir
    assert "ttg.convert_layout" in ttgir
    assert "tle.explicit_encoding." not in ttgir


@pytest.mark.parametrize("hard", [True, False], ids=["hard", "ordinary"])
def test_mthreads_tle_optimize_sqmma_accumulator_layout_respects_hard_boundary(tmp_path, hard):
    attrs = " {tle.explicit_encoding.0 = #layout_a}" if hard else ""
    ttgir = _run_ttgir_musa_pass(
        tmp_path,
        f"""
  tt.func public @sqmma_accumulator(
      %a: !ttg.memdesc<64x32xf16, #shared_a, #smem, mutable>,
      %b: !ttg.memdesc<32x64xf16, #shared_b, #smem, mutable>,
      %mma_init: tensor<64x64xf32, #sqmma>,
      %blocked_init: tensor<64x64xf32, #layout_a>,
      %limit: index) -> tensor<64x64xf32, #layout_a> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %loop:2 = scf.for %iv = %c0 to %limit step %c1 iter_args(%mma = %mma_init, %blocked = %blocked_init) -> (tensor<64x64xf32, #sqmma>, tensor<64x64xf32, #layout_a>) {{
      %dot = ttmg.squad_dot %a, %b, %mma, %true {{eltTypeA = 4 : i32, eltTypeB = 4 : i32, eltTypeC = 7 : i32, isAsync = true, k = 32 : i32, layoutA = 0 : i32, layoutB = 0 : i32, m = 64 : i32, n = 64 : i32}} : !ttg.memdesc<64x32xf16, #shared_a, #smem, mutable> * !ttg.memdesc<32x64xf16, #shared_b, #smem, mutable> -> tensor<64x64xf32, #sqmma>
      %wait = ttmg.squad_dot_wait %dot : tensor<64x64xf32, #sqmma>
      %converted = ttg.convert_layout %wait{attrs} : tensor<64x64xf32, #sqmma> -> tensor<64x64xf32, #layout_a>
      scf.yield %wait, %converted : tensor<64x64xf32, #sqmma>, tensor<64x64xf32, #layout_a>
    }}
    tt.return %loop#1 : tensor<64x64xf32, #layout_a>
  }}
""",
        "add_optimize_sqmma_accumulator_layout",
    )

    if hard:
        assert "tt.return %0#1" in ttgir
        hard_line = next(line for line in ttgir.splitlines() if "tle.explicit_encoding.0" in line)
        assert "ttg.convert_layout" in hard_line
    else:
        assert "%blocked = %blocked_init" not in ttgir
        assert "tle.explicit_encoding." not in ttgir


@pytest.mark.parametrize("hard", [True, False], ids=["hard", "ordinary"])
def test_mthreads_tle_canonicalize_sqmma_result_conversion_respects_hard_boundary(tmp_path, hard):
    attrs = " {tle.explicit_encoding.0 = #layout_a}" if hard else ""
    ttgir = _run_ttgir_musa_pass(
        tmp_path,
        f"""
  tt.func public @sqmma_result(%arg0: tensor<64x64xf32, #sqmma>) -> tensor<64x64xf16, #layout_a> {{
    %converted = ttg.convert_layout %arg0{attrs} : tensor<64x64xf32, #sqmma> -> tensor<64x64xf32, #layout_a>
    %truncated = arith.truncf %converted : tensor<64x64xf32, #layout_a> to tensor<64x64xf16, #layout_a>
    tt.return %truncated : tensor<64x64xf16, #layout_a>
  }}
""",
        "add_canonicalize_sqmma_result_conversions",
    )

    lines = ttgir.splitlines()
    convert_idx = next(i for i, line in enumerate(lines) if "ttg.convert_layout" in line)
    trunc_idx = next(i for i, line in enumerate(lines) if "arith.truncf" in line)
    if hard:
        assert convert_idx < trunc_idx
        assert ttgir.count("tle.explicit_encoding.0") == 1
    else:
        assert trunc_idx < convert_idx
        assert "tle.explicit_encoding." not in ttgir


@pytest.mark.parametrize("hard", [True, False], ids=["hard", "ordinary"])
def test_mthreads_tle_coalesce_async_copy_respects_explicit_memory_layout(tmp_path, hard):
    attrs = " {tle.explicit_memory_encoding = #layout_1d_a}" if hard else ""
    ttgir = _run_ttgir_musa_pass(
        tmp_path,
        f"""
  tt.func public @async_copy(%ptrs: tensor<32x!tt.ptr<f32>, #layout_1d_a>) {{
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptrs, %buf{attrs} : tensor<32x!tt.ptr<f32>, #layout_1d_a> -> <32xf32, #shared_1d, #smem, mutable>
    tt.return
  }}
""",
        "add_coalesce_async_copy",
    )

    if hard:
        copy_line = next(line for line in ttgir.splitlines() if "ttg.async_copy_global_to_local" in line)
        assert "tle.explicit_memory_encoding" in copy_line
        explicit = re.search(r"tle\.explicit_memory_encoding = (#\w+)", copy_line).group(1)
        assert copy_line.count(explicit) >= 2
        assert "ttg.convert_layout" not in ttgir
    else:
        assert "tle.explicit_memory_encoding" not in ttgir


def test_mthreads_tle_coalesce_async_copy_rejects_conflicting_explicit_memory_layout(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            """
  tt.func public @async_copy_conflict(%ptrs: tensor<32x!tt.ptr<f32>, #layout_1d_a>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared_1d, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptrs, %buf {tle.explicit_memory_encoding = #layout_1d_b} : tensor<32x!tt.ptr<f32>, #layout_1d_a> -> <32xf32, #shared_1d, #smem, mutable>
    tt.return
  }
""",
            "add_coalesce_async_copy",
        )

    diagnostic = capfd.readouterr().err
    assert "conflicts with the async-copy source encoding" in diagnostic


def test_mthreads_tle_set_layout_survives_full_sqmma_ttgir_pipeline():
    layout = MusaSqmmaEncoding([3, 1], [4, 1], [128, 128, 64])
    compiled = compile_musa(
        _set_layout_sqmma_pipeline_kernel,
        {"out": "*fp32", "LAYOUT": "constexpr"},
        {"LAYOUT": layout},
    )
    ttgir = compiled.asm["ttgir"]

    assert "musa_tle.set_layout" not in ttgir
    assert "ttmg.squad_dot" in ttgir
    assert "#ttg.musa_sqmma" in ttgir
    assert "tle.explicit_encoding." not in ttgir
    assert "tle.explicit_memory_encoding" not in ttgir
    assert "tle.explicit_encoding." not in compiled.asm["llir"]
    assert "tle.explicit_memory_encoding" not in compiled.asm["llir"]


def test_mthreads_tle_finalize_explicit_layouts_keeps_ordinary_pipeline():
    compiled = compile_musa(_ordinary_ttgir_pipeline_kernel, {"out": "*fp32"})
    ttgir = compiled.asm["ttgir"]
    assert "tt.store" in ttgir
    assert "musa_tle.set_layout" not in ttgir
    assert "tle.explicit_encoding." not in ttgir
    assert "tle.explicit_memory_encoding" not in ttgir


def test_mthreads_tle_explicit_wmma_uses_fp32_carrier_for_fp16_result(tmp_path):
    ttgir = _run_ttgir_musa_pass(
        tmp_path,
        """
  tt.func public @explicit_wmma_fp16() -> tensor<16x16xf16, #mma> {
    %lhs = arith.constant dense<0.0> : tensor<16x16xf16, #lhs>
    %rhs = arith.constant dense<0.0> : tensor<16x16xf16, #rhs>
    %acc = arith.constant dense<0.0> : tensor<16x16xf16, #mma>
    %result = tt.dot %lhs, %rhs, %acc : tensor<16x16xf16, #lhs> * tensor<16x16xf16, #rhs> -> tensor<16x16xf16, #mma>
    tt.return %result : tensor<16x16xf16, #mma>
  }
""",
        "add_accelerate_matmul",
    )

    assert "ttmg.wmma_dot" in ttgir
    assert "-> tensor<16x16xf32, #mma>" in ttgir
    assert "arith.truncf" in ttgir
    assert "to tensor<16x16xf16, #mma>" in ttgir
    assert " tt.dot " not in ttgir


@pytest.mark.parametrize(
    "encoding,lhs_encoding,rhs_encoding,diagnostic",
    [
        (
            "#wmma_unsupported",
            "#lhs_unsupported",
            "#rhs_unsupported",
            "instruction shape and element types are unsupported",
        ),
    ],
)
def test_mthreads_tle_explicit_wmma_rejects_unsupported_contracts(tmp_path, capfd, encoding, lhs_encoding, rhs_encoding,
                                                                  diagnostic):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            f"""
  tt.func public @invalid_explicit_wmma() {{
    %lhs = arith.constant dense<0.0> : tensor<16x16xbf16, {lhs_encoding}>
    %rhs = arith.constant dense<0.0> : tensor<16x16xbf16, {rhs_encoding}>
    %acc = arith.constant dense<0.0> : tensor<16x16xf32, {encoding}>
    %result = tt.dot %lhs, %rhs, %acc : tensor<16x16xbf16, {lhs_encoding}> * tensor<16x16xbf16, {rhs_encoding}> -> tensor<16x16xf32, {encoding}>
    tt.return
  }}
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "cannot lower explicit MUSA WMMA dot" in error
    assert diagnostic in error


def test_mthreads_tle_explicit_wmma_rejects_module_warp_mismatch(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            """
  tt.func public @invalid_explicit_wmma_warps() {
    %lhs = arith.constant dense<0.0> : tensor<16x16xbf16, #lhs_bad_warps>
    %rhs = arith.constant dense<0.0> : tensor<16x16xbf16, #rhs_bad_warps>
    %acc = arith.constant dense<0.0> : tensor<16x16xf32, #wmma_bad_warps>
    %result = tt.dot %lhs, %rhs, %acc : tensor<16x16xbf16, #lhs_bad_warps> * tensor<16x16xbf16, #rhs_bad_warps> -> tensor<16x16xf32, #wmma_bad_warps>
    tt.return
  }
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "Layout has 2 warps per CTA" in error
    assert "the context requires 4 warps per CTA" in error


def test_mthreads_tle_explicit_wmma_rejects_disable_wmma(monkeypatch, tmp_path, capfd):
    monkeypatch.setenv("DISABLE_WMMA", "1")
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            """
  tt.func public @disabled_explicit_wmma() {
    %lhs = arith.constant dense<0.0> : tensor<16x16xbf16, #lhs>
    %rhs = arith.constant dense<0.0> : tensor<16x16xbf16, #rhs>
    %acc = arith.constant dense<0.0> : tensor<16x16xf32, #mma>
    %result = tt.dot %lhs, %rhs, %acc : tensor<16x16xbf16, #lhs> * tensor<16x16xbf16, #rhs> -> tensor<16x16xf32, #mma>
    tt.return
  }
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "cannot lower explicit MUSA WMMA dot" in error
    assert "DISABLE_WMMA conflicts with an explicit MUSA WMMA layout" in error


@pytest.mark.parametrize("kind", ["block", "slice", "transpose", "dual-domain"])
@requires_musa_runtime
def test_mthreads_tle_layout_runtime(kind):
    block = 16
    layout_a = tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0])
    layout_b = tle.gpu.BlockEncoding([1, 1], [32, 1], [1, 4], [0, 1])
    if kind == "block":
        kernel = _set_layout_block_numeric_kernel
        shape = (block, block)
        reference = torch.arange(block * block, dtype=torch.float32).reshape(shape) * 1.25 - 7.0
        source = reference.to("musa")
        actual = torch.empty_like(source)
        compiled = _warmup_and_run(kernel, source, actual, block, layout_a, num_warps=4)
        expected = reference
    elif kind == "slice":
        kernel = _set_layout_slice_numeric_kernel
        layout = tle.gpu.SlicedEncoding(0, layout_a)
        shape = (block, )
        reference = torch.arange(block, dtype=torch.float32) * 1.25 - 7.0
        source = reference.to("musa")
        actual = torch.empty_like(source)
        compiled = _warmup_and_run(kernel, source, actual, block, layout, num_warps=4)
        expected = reference
    elif kind == "transpose":
        shape = (block, block)
        reference = torch.arange(block * block, dtype=torch.float32).reshape(shape) * 0.5 - 3.0
        source = reference.to("musa")
        actual = torch.empty_like(source)
        compiled = _warmup_and_run(
            _set_layout_transpose_numeric_kernel,
            source,
            actual,
            block,
            layout_a,
            layout_b,
            num_warps=4,
        )
        trans_lines = [line for line in compiled.asm["ttgir"].splitlines() if "tt.trans" in line]
        assert len(trans_lines) == 1
        assert trans_lines[0].count("#") >= 2
        expected = reference.T * 2.0 + 1.0
    else:
        shape = (block, block)
        reference = torch.full(shape, 3.25, dtype=torch.float32)
        actual = torch.empty(shape, device="musa", dtype=torch.float32)
        alternate = torch.empty_like(actual)
        compiled = _warmup_and_run(
            _set_layout_dual_domain_numeric_kernel,
            actual,
            alternate,
            block,
            layout_a,
            layout_b,
            num_warps=4,
        )
        torch.testing.assert_close(alternate.cpu(), reference, rtol=0, atol=0)
        expected = reference

    _assert_runtime_pipeline_ir(compiled)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@requires_musa_runtime
def test_mthreads_tle_shared_rank3_slice_runtime():
    block = 128
    row_layout = tle.gpu.BlockEncoding([1, 1, 1], [1, 32, 1], [1, 4, 1], [2, 1, 0])
    col_layout = tle.gpu.BlockEncoding([1, 1, 1], [1, 1, 32], [1, 1, 4], [2, 1, 0])
    actual_rows = torch.empty(block, device="musa", dtype=torch.float32)
    actual_cols = torch.empty_like(actual_rows)

    compiled = _warmup_and_run(
        _set_layout_shared_rank3_slice_numeric_kernel,
        actual_rows,
        actual_cols,
        block,
        row_layout,
        col_layout,
        num_warps=4,
    )

    _assert_runtime_pipeline_ir(compiled)
    expected = torch.arange(block, dtype=torch.float32)
    torch.testing.assert_close(actual_rows.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_cols.cpu(), expected, rtol=0, atol=0)


@requires_musa_runtime
def test_mthreads_tle_noinline_function_abi_runtime():
    block = 128
    parent = tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0])
    layout = tle.gpu.SlicedEncoding(0, parent)
    actual = torch.empty(block, device="musa", dtype=torch.float32)

    compiled = _warmup_and_run(
        _set_layout_noinline_numeric_kernel,
        actual,
        block,
        layout,
        num_warps=4,
    )

    _assert_runtime_pipeline_ir(compiled)
    expected = torch.arange(block, dtype=torch.float32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["wmma", "sqmma"])
@requires_musa_runtime
def test_mthreads_tle_explicit_and_automatic_mma_runtime(monkeypatch, kind):
    monkeypatch.delenv("DISABLE_WMMA", raising=False)
    monkeypatch.delenv("DISABLE_SQMMA", raising=False)
    if kind == "wmma":
        kernel = _explicit_musa_wmma_kernel
        mma = MusaWmmaEncoding([3, 1], [2, 2], [16, 8, 16])
        block_m, block_n, block_k = 16, 16, 16
        dtype = torch.bfloat16
        mnemonic = "ttmg.wmma_dot"
    else:
        kernel = _explicit_musa_sqmma_kernel
        mma = MusaSqmmaEncoding([3, 1], [4, 1], [32, 64, 16])
        block_m, block_n, block_k = 64, 64, 32
        dtype = torch.float16
        mnemonic = "ttmg.squad_dot"
    lhs_layout = MusaDotOperandEncoding(0, mma)
    rhs_layout = MusaDotOperandEncoding(1, mma)
    lhs = torch.full((block_m, block_k), 1.0, device="musa", dtype=dtype)
    rhs = torch.full((block_k, block_n), 2.0, device="musa", dtype=dtype)
    actual = torch.empty((block_m, block_n), device="musa", dtype=torch.float32)
    automatic = torch.empty_like(actual)

    explicit_compiled = _warmup_and_run(
        kernel,
        actual,
        mma,
        lhs_layout,
        rhs_layout,
        num_warps=4,
    )
    automatic_compiled = _warmup_and_run(
        _automatic_musa_mma_numeric_kernel,
        lhs,
        rhs,
        automatic,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    _assert_runtime_pipeline_ir(explicit_compiled, mnemonic)
    automatic_ttgir = automatic_compiled.asm["ttgir"]
    assert automatic_ttgir.count(mnemonic) == 1
    assert " tt.dot " not in automatic_ttgir
    assert "nvvm" not in automatic_compiled.asm["llir"].lower()
    reference = lhs.cpu().float() @ rhs.cpu().float()
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(automatic.cpu(), reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual, automatic, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("kind", ["layout", "rank3-shared-slice", "noinline-abi", "wmma", "sqmma"])
@pytest.mark.skipif(_musa_runtime_available(), reason="covered by MUSA runtime tests")
def test_mthreads_tle_runtime_compile_fallback(kind):
    if kind == "layout":
        layout_a = tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0])
        layout_b = tle.gpu.BlockEncoding([1, 1], [32, 1], [1, 4], [0, 1])
        compiled = compile_musa(
            _set_layout_dual_domain_numeric_kernel,
            {
                "out_a": "*fp32",
                "out_b": "*fp32",
                "BLOCK": "constexpr",
                "LAYOUT_A": "constexpr",
                "LAYOUT_B": "constexpr",
            },
            {"BLOCK": 16, "LAYOUT_A": layout_a, "LAYOUT_B": layout_b},
        )
        _assert_runtime_pipeline_ir(compiled)
        return

    if kind == "rank3-shared-slice":
        row_layout = tle.gpu.BlockEncoding([1, 1, 1], [1, 32, 1], [1, 4, 1], [2, 1, 0])
        col_layout = tle.gpu.BlockEncoding([1, 1, 1], [1, 1, 32], [1, 1, 4], [2, 1, 0])
        compiled = compile_musa(
            _set_layout_shared_rank3_slice_numeric_kernel,
            {
                "out_rows": "*fp32",
                "out_cols": "*fp32",
                "BLOCK": "constexpr",
                "ROW_LAYOUT": "constexpr",
                "COL_LAYOUT": "constexpr",
            },
            {"BLOCK": 128, "ROW_LAYOUT": row_layout, "COL_LAYOUT": col_layout},
        )
        _assert_runtime_pipeline_ir(compiled)
        return

    if kind == "noinline-abi":
        parent = tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0])
        layout = tle.gpu.SlicedEncoding(0, parent)
        compiled = compile_musa(
            _set_layout_noinline_numeric_kernel,
            {
                "out": "*fp32",
                "BLOCK": "constexpr",
                "LAYOUT": "constexpr",
            },
            {"BLOCK": 128, "LAYOUT": layout},
        )
        _assert_runtime_pipeline_ir(compiled)
        return

    if kind == "wmma":
        kernel = _explicit_musa_wmma_kernel
        mma = MusaWmmaEncoding([3, 1], [2, 2], [16, 8, 16])
        mnemonic = "ttmg.wmma_dot"
    else:
        kernel = _explicit_musa_sqmma_kernel
        mma = MusaSqmmaEncoding([3, 1], [4, 1], [32, 64, 16])
        mnemonic = "ttmg.squad_dot"
    lhs_layout = MusaDotOperandEncoding(0, mma)
    rhs_layout = MusaDotOperandEncoding(1, mma)
    compiled = compile_musa(
        kernel,
        {
            "out": "*fp32",
            "MMA_LAYOUT": "constexpr",
            "LHS_LAYOUT": "constexpr",
            "RHS_LAYOUT": "constexpr",
        },
        {"MMA_LAYOUT": mma, "LHS_LAYOUT": lhs_layout, "RHS_LAYOUT": rhs_layout},
    )
    _assert_runtime_pipeline_ir(compiled, mnemonic)


def test_mthreads_tle_explicit_sqmma_uses_fp32_carrier_for_fp16_result(tmp_path):
    ttgir = _run_ttgir_musa_pass(
        tmp_path,
        """
  tt.func public @explicit_sqmma_fp16() -> tensor<64x64xf16, #sqmma_explicit> {
    %lhs = arith.constant dense<0.0> : tensor<64x32xf16, #sqmma_lhs_explicit>
    %rhs = arith.constant dense<0.0> : tensor<32x64xf16, #sqmma_rhs_explicit>
    %acc = arith.constant dense<0.0> : tensor<64x64xf16, #sqmma_explicit>
    %result = tt.dot %lhs, %rhs, %acc : tensor<64x32xf16, #sqmma_lhs_explicit> * tensor<32x64xf16, #sqmma_rhs_explicit> -> tensor<64x64xf16, #sqmma_explicit>
    tt.return %result : tensor<64x64xf16, #sqmma_explicit>
  }
""",
        "add_accelerate_matmul",
    )

    assert "ttmg.squad_dot" in ttgir
    assert "-> tensor<64x64xf32, #mma>" in ttgir
    assert "arith.truncf" in ttgir
    assert "to tensor<64x64xf16, #mma>" in ttgir
    assert " tt.dot " not in ttgir


@pytest.mark.parametrize(
    "encoding,lhs_encoding,rhs_encoding,diagnostic",
    [
        (
            "#sqmma_unsupported",
            "#sqmma_lhs_unsupported",
            "#sqmma_rhs_unsupported",
            "instruction shape and element types are unsupported",
        ),
    ],
)
def test_mthreads_tle_explicit_sqmma_rejects_unsupported_contracts(
    tmp_path,
    capfd,
    encoding,
    lhs_encoding,
    rhs_encoding,
    diagnostic,
):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            f"""
  tt.func public @invalid_explicit_sqmma() {{
    %lhs = arith.constant dense<0.0> : tensor<64x32xbf16, {lhs_encoding}>
    %rhs = arith.constant dense<0.0> : tensor<32x64xbf16, {rhs_encoding}>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32, {encoding}>
    %result = tt.dot %lhs, %rhs, %acc : tensor<64x32xbf16, {lhs_encoding}> * tensor<32x64xbf16, {rhs_encoding}> -> tensor<64x64xf32, {encoding}>
    tt.return
  }}
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "cannot lower explicit MUSA SQMMA dot" in error
    assert diagnostic in error


def test_mthreads_tle_explicit_sqmma_rejects_unmaterializable_shared_operand(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            """
  tt.func public @invalid_explicit_sqmma_shared(%raw_lhs: tensor<64x32xbf16>) {
    %lhs = ttg.convert_layout %raw_lhs : tensor<64x32xbf16> -> tensor<64x32xbf16, #sqmma_lhs_explicit>
    %rhs = arith.constant dense<0.0> : tensor<32x64xbf16, #sqmma_rhs_explicit>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32, #sqmma_explicit>
    %result = tt.dot %lhs, %rhs, %acc : tensor<64x32xbf16, #sqmma_lhs_explicit> * tensor<32x64xbf16, #sqmma_rhs_explicit> -> tensor<64x64xf32, #sqmma_explicit>
    tt.return
  }
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "cannot lower explicit MUSA SQMMA dot" in error
    assert "operand cannot be materialized in shared memory" in error


def test_mthreads_tle_explicit_sqmma_rejects_invalid_squad_layout(tmp_path, capfd):
    fixture = tmp_path / "mthreads_tle_invalid_explicit_sqmma_squad.mlir"
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       "#sqmma_bad_squad = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, "
                       "warpsPerCTA = [2, 2], instrShape = [32, 64, 16]}>\n"
                       "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 4 : i32, "
                       "ttg.target = \"musa:ph1\", \"ttg.threads-per-warp\" = 32 : i32} {}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    with pytest.raises(RuntimeError):
        ir.parse_mlir_module(str(fixture), context)
    assert "SQMMA expects warpsPerCTA[0] to be a multiple of 4" in capfd.readouterr().err


def test_mthreads_tle_explicit_sqmma_rejects_mismatched_parent(tmp_path, capfd):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            """
  tt.func public @invalid_explicit_sqmma_parent() {
    %lhs = arith.constant dense<0.0> : tensor<64x32xbf16, #lhs>
    %rhs = arith.constant dense<0.0> : tensor<32x64xbf16, #sqmma_rhs_explicit>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32, #sqmma_explicit>
    %result = tt.dot %lhs, %rhs, %acc : tensor<64x32xbf16, #lhs> * tensor<32x64xbf16, #sqmma_rhs_explicit> -> tensor<64x64xf32, #sqmma_explicit>
    tt.return
  }
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "Incompatible parent encoding" in error


def test_mthreads_tle_explicit_sqmma_rejects_disable_sqmma(monkeypatch, tmp_path, capfd):
    monkeypatch.setenv("DISABLE_SQMMA", "1")
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            """
  tt.func public @disabled_explicit_sqmma() {
    %lhs = arith.constant dense<0.0> : tensor<64x32xbf16, #sqmma_lhs_explicit>
    %rhs = arith.constant dense<0.0> : tensor<32x64xbf16, #sqmma_rhs_explicit>
    %acc = arith.constant dense<0.0> : tensor<64x64xf32, #sqmma_explicit>
    %result = tt.dot %lhs, %rhs, %acc : tensor<64x32xbf16, #sqmma_lhs_explicit> * tensor<32x64xbf16, #sqmma_rhs_explicit> -> tensor<64x64xf32, #sqmma_explicit>
    tt.return
  }
""",
            "add_accelerate_matmul",
        )
    error = capfd.readouterr().err
    assert "cannot lower explicit MUSA SQMMA dot" in error
    assert "DISABLE_SQMMA conflicts with an explicit MUSA SQMMA layout" in error


@pytest.mark.parametrize(
    "lhs_type,rhs_type,result_type,diagnostic",
    [
        (
            "tensor<16x16xbf16, #rhs>",
            "tensor<16x16xbf16, #rhs>",
            "tensor<16x16xf32, #mma>",
            "Wrong opIdx",
        ),
        (
            "tensor<16x16xbf16, #lhs>",
            "tensor<16x16xbf16, #lhs>",
            "tensor<16x16xf32, #mma>",
            "Wrong opIdx",
        ),
        (
            "tensor<16x16xbf16, #sqmma_lhs_explicit>",
            "tensor<16x16xbf16, #rhs>",
            "tensor<16x16xf32, #mma>",
            "Incompatible parent encoding",
        ),
        (
            "tensor<16x16xbf16, #lhs>",
            "tensor<16x16xbf16, #sqmma_rhs_explicit>",
            "tensor<16x16xf32, #mma>",
            "Incompatible parent encoding",
        ),
        (
            "tensor<64x32xbf16, #sqmma_rhs_explicit>",
            "tensor<32x64xbf16, #sqmma_rhs_explicit>",
            "tensor<64x64xf32, #sqmma_explicit>",
            "Wrong opIdx",
        ),
        (
            "tensor<64x32xbf16, #sqmma_lhs_explicit>",
            "tensor<32x64xbf16, #sqmma_lhs_explicit>",
            "tensor<64x64xf32, #sqmma_explicit>",
            "Wrong opIdx",
        ),
        (
            "tensor<64x32xbf16, #lhs>",
            "tensor<32x64xbf16, #sqmma_rhs_explicit>",
            "tensor<64x64xf32, #sqmma_explicit>",
            "Incompatible parent encoding",
        ),
        (
            "tensor<64x32xbf16, #sqmma_lhs_explicit>",
            "tensor<32x64xbf16, #rhs>",
            "tensor<64x64xf32, #sqmma_explicit>",
            "Incompatible parent encoding",
        ),
    ],
)
def test_mthreads_tle_explicit_mma_rejects_operand_index_and_parent_matrix(
    tmp_path,
    capfd,
    lhs_type,
    rhs_type,
    result_type,
    diagnostic,
):
    with pytest.raises(RuntimeError):
        _run_ttgir_musa_pass(
            tmp_path,
            f"""
  tt.func public @invalid_explicit_musa_mma_contract() {{
    %lhs = arith.constant dense<0.0> : {lhs_type}
    %rhs = arith.constant dense<0.0> : {rhs_type}
    %acc = arith.constant dense<0.0> : {result_type}
    %result = tt.dot %lhs, %rhs, %acc : {lhs_type} * {rhs_type} -> {result_type}
    tt.return
  }}
""",
            "add_accelerate_matmul",
        )
    assert diagnostic in capfd.readouterr().err


@pytest.mark.parametrize(
    "parent,operand_index,kind",
    [
        ("#mma", 0, "WMMA"),
        ("#mma", 1, "WMMA"),
        ("#sqmma_explicit", 0, "SQMMA"),
        ("#sqmma_explicit", 1, "SQMMA"),
    ],
)
def test_mthreads_tle_explicit_mma_rejects_nonzero_k_width(
    tmp_path,
    capfd,
    parent,
    operand_index,
    kind,
):
    fixture = (tmp_path / f"mthreads_tle_invalid_{kind.lower()}_k_width_{operand_index}.mlir")
    fixture.write_text(f"{_CONVERSION_LAYOUTS}\n"
                       f"#invalid_k_width = #ttg.dot_op<{{opIdx = {operand_index}, parent = {parent}, kWidth = 1}}>\n"
                       "module {}\n")

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    with pytest.raises(RuntimeError):
        ir.parse_mlir_module(str(fixture), context)
    error = capfd.readouterr().err
    assert "ttg.dot_op kWidth parameter is not supported" in error
    assert f"for MUSA {kind} parent" in error


def test_mthreads_tle_set_layout_rejects_chained_dot_accumulator_conflict(
    tmp_path,
    capfd,
):
    with pytest.raises(RuntimeError):
        _convert_set_layout_to_ttgir(
            tmp_path,
            """
  tt.func public @conflicting_chained_dot_accumulator() {
    %lhs_value = arith.constant dense<1.0> : tensor<16x16xbf16>
    %lhs = musa_tle.set_layout %lhs_value {target_encoding = #lhs} : tensor<16x16xbf16> -> tensor<16x16xbf16>
    %rhs_value = arith.constant dense<2.0> : tensor<16x16xbf16>
    %rhs = musa_tle.set_layout %rhs_value {target_encoding = #rhs} : tensor<16x16xbf16> -> tensor<16x16xbf16>
    %acc_value = arith.constant dense<0.0> : tensor<16x16xf32>
    %acc = musa_tle.set_layout %acc_value {target_encoding = #mma} : tensor<16x16xf32> -> tensor<16x16xf32>
    %first = tt.dot %lhs, %rhs, %acc : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
    %conflict = musa_tle.set_layout %first {target_encoding = #wmma_explicit} : tensor<16x16xf32> -> tensor<16x16xf32>
    %second = tt.dot %lhs, %rhs, %conflict : tensor<16x16xbf16> * tensor<16x16xbf16> -> tensor<16x16xf32>
    tt.return
  }
""",
        )
    error = capfd.readouterr().err
    assert "Incompatible parent encoding" in error
    assert "failed to infer returned types" in error
    assert "#ttg.musa_wmma" in error


@pytest.mark.parametrize("kind", ["wmma", "sqmma"])
@requires_musa_runtime
def test_mthreads_tle_explicit_mma_chained_runtime(kind):
    if kind == "wmma":
        kernel = _explicit_musa_wmma_chained_kernel
        mma = MusaWmmaEncoding([3, 1], [4, 1], [16, 16, 16])
        block_m, block_n, block_k = 16, 16, 16
        mnemonic = "ttmg.wmma_dot"
    else:
        kernel = _explicit_musa_sqmma_chained_kernel
        mma = MusaSqmmaEncoding([3, 1], [4, 1], [32, 64, 16])
        block_m, block_n, block_k = 64, 64, 32
        mnemonic = "ttmg.squad_dot"
    lhs_layout = MusaDotOperandEncoding(0, mma)
    rhs_layout = MusaDotOperandEncoding(1, mma)
    actual = torch.empty((block_m, block_n), device="musa", dtype=torch.float32)

    compiled = _warmup_and_run(
        kernel,
        actual,
        mma,
        lhs_layout,
        rhs_layout,
        num_warps=4,
    )

    _assert_runtime_pipeline_ir(compiled, mnemonic, native_dot_count=2)
    _assert_direct_native_dot_chain(compiled.asm["ttgir"], mnemonic)
    reference = torch.full((block_m, block_n), 4.0 * block_k, dtype=torch.float32)
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-3, atol=1e-3)


def test_mthreads_tle_finalize_explicit_layouts_preserves_final_encodings(tmp_path):
    before, finalized = _run_ttgir_finalize_explicit_layouts(
        tmp_path,
        """
  tt.func public @finalize_layouts(%arg0: tensor<16x16xf32, #layout_a>, %out: tensor<16x16x!tt.ptr<f32>, #layout_b>) {
    %hard = ttg.convert_layout %arg0 {tle.explicit_encoding.0 = #layout_b} : tensor<16x16xf32, #layout_a> -> tensor<16x16xf32, #layout_b>
    %identity = ttg.convert_layout %hard {tle.explicit_encoding.0 = #layout_b} : tensor<16x16xf32, #layout_b> -> tensor<16x16xf32, #layout_b>
    tt.store %out, %identity {tle.explicit_memory_encoding = #layout_b} : tensor<16x16x!tt.ptr<f32>, #layout_b>
    tt.return
  }
""",
    )

    assert before.count("tle.explicit_encoding.0") == 2
    assert "tle.explicit_memory_encoding" in before
    assert "tle.explicit_encoding." not in finalized
    assert "tle.explicit_memory_encoding" not in finalized
    assert finalized.count("ttg.convert_layout") == 1
    conversion = next(line for line in finalized.splitlines() if "ttg.convert_layout" in line)
    source_encoding, target_encoding = _type_encoding_aliases(conversion)
    assert source_encoding != target_encoding
    store = next(line for line in finalized.splitlines() if "tt.store" in line)
    assert target_encoding in store


def test_mthreads_tle_finalize_explicit_layouts_is_idempotent_after_cse(tmp_path):
    before, finalized = _run_ttgir_finalize_explicit_layouts(
        tmp_path,
        """
  tt.func public @finalize_after_cse(%out: tensor<16x16x!tt.ptr<f32>, #layout_a>) {
    %lhs = arith.constant {tle.explicit_encoding.0 = #layout_a} dense<1.0> : tensor<16x16xf32, #layout_a>
    %rhs = arith.constant {tle.explicit_encoding.0 = #layout_a} dense<1.0> : tensor<16x16xf32, #layout_a>
    %sum = arith.addf %lhs, %rhs {tle.explicit_encoding.0 = #layout_a} : tensor<16x16xf32, #layout_a>
    tt.store %out, %sum {tle.explicit_memory_encoding = #layout_a} : tensor<16x16x!tt.ptr<f32>, #layout_a>
    tt.return
  }
""",
        finalize_repeat=2,
        run_cse=True,
    )

    assert before.count("arith.constant") == 1
    assert "tle.explicit_encoding.0" in before
    assert "tle.explicit_memory_encoding" in before
    assert "tle.explicit_encoding." not in finalized
    assert "tle.explicit_memory_encoding" not in finalized
    assert finalized.count("arith.constant") == 1
    assert "tensor<16x16xf32" in finalized


def test_mthreads_tle_finalize_explicit_layouts_keeps_ordinary_ir(tmp_path):
    before, finalized = _run_ttgir_finalize_explicit_layouts(
        tmp_path,
        """
  tt.func public @ordinary_finalize(%arg0: tensor<16x16xf32, #layout_a>, %out: tensor<16x16x!tt.ptr<f32>, #layout_a>) {
    %value = arith.addf %arg0, %arg0 : tensor<16x16xf32, #layout_a>
    tt.store %out, %value : tensor<16x16x!tt.ptr<f32>, #layout_a>
    tt.return
  }
""",
    )
    assert finalized == before


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (
            """
  tt.func public @malformed_result_attr() {
    %value = arith.constant {tle.explicit_encoding.bad = #layout_a} dense<1.0> : tensor<16x16xf32, #layout_a>
    tt.return
  }
""",
            "has malformed explicit MUSA TLE result encoding",
        ),
        (
            """
  tt.func public @missing_result_attr() {
    %value = arith.constant {tle.explicit_encoding.1 = #layout_a} dense<1.0> : tensor<16x16xf32, #layout_a>
    tt.return
  }
""",
            "has explicit MUSA TLE encoding for missing result 1",
        ),
        (
            """
  tt.func public @mismatched_result_attr() {
    %value = arith.constant {tle.explicit_encoding.0 = #layout_b} dense<1.0> : tensor<16x16xf32, #layout_a>
    tt.return
  }
""",
            "does not match the final tensor type encoding",
        ),
        (
            """
  tt.func public @memory_attr_on_non_memory_op() {
    %value = arith.constant {tle.explicit_memory_encoding = #layout_a} dense<1.0> : tensor<16x16xf32, #layout_a>
    tt.return
  }
""",
            "explicit MUSA TLE memory encoding on a non-memory operation",
        ),
        (
            """
  tt.func public @mismatched_memory_attr(%out: tensor<16x16x!tt.ptr<f32>, #layout_a>) {
    %value = arith.constant dense<1.0> : tensor<16x16xf32, #layout_a>
    tt.store %out, %value {tle.explicit_memory_encoding = #layout_b} : tensor<16x16x!tt.ptr<f32>, #layout_a>
    tt.return
  }
""",
            "does not match the final pointer encoding",
        ),
    ],
)
def test_mthreads_tle_finalize_explicit_layouts_rejects_invalid_contracts(tmp_path, capfd, body, diagnostic):
    with pytest.raises(RuntimeError):
        _run_ttgir_finalize_explicit_layouts(tmp_path, body)
    assert diagnostic in capfd.readouterr().err


def test_mthreads_tle_make_ttgir_records_explicit_layout_lifecycle(monkeypatch, tmp_path):
    repro_prefix = tmp_path / "mthreads_tle_layout_lifecycle"
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_prefix))

    layout = MusaSqmmaEncoding([3, 1], [4, 1], [128, 128, 64])
    compile_musa(
        _set_layout_sqmma_pipeline_kernel,
        {"out": "*fp32", "LAYOUT": "constexpr"},
        {"LAYOUT": layout},
    )

    reproducer = (tmp_path / "mthreads_tle_layout_lifecycle.make_ttgir.repro.mlir").read_text()
    pipeline_match = re.search(r'pipeline: "([^"]+)"', reproducer)
    assert pipeline_match is not None, reproducer
    pipeline = pipeline_match.group(1)
    cleanup = "tritongpu-remove-layout-conversions"
    finalize = "tritonmusa-tle-finalize-explicit-layouts"
    assert pipeline.count(cleanup) == 3
    assert pipeline.count(finalize) == 1

    ordered_passes = [
        "convert-triton-to-tritongpu",
        "tritongpu-coalesce",
        cleanup,
        "tritongpu-optimize-thread-locality",
        "tritonmusa-tle-select-encodings",
        "tritonmusa-tle-lower-sqmma",
        "tritonmusa-accelerate-matmul",
        cleanup,
        "tritonmusa-optimize-dot-operands",
        "tritonmusa-pipeline",
        "tritongpu-prefetch",
        "tritongpu-coalesce-async-copy",
        "tritonmusa-tme-lowering",
        "tritonmusa-canonicalize-sqmma-result-conversions",
        cleanup,
        "tritonmusa-convert-sqmma-to-mtgpu",
        "tritonmusa-finalize-barriers",
        "tritonmusa-tle-prepare-warp-specialize",
        finalize,
    ]
    cursor = 0
    for pass_name in ordered_passes:
        position = pipeline.find(pass_name, cursor)
        assert position >= cursor, f"missing or misordered pass {pass_name}: {pipeline}"
        cursor = position + len(pass_name)


def test_mthreads_tle_layout_attrs_are_written_and_idempotent():
    ttir = compile_to_ttir(_valid_layout_attrs_kernel, {})
    assert '"ttg.num-warps" = 4 : i32' in ttir
    assert '"ttg.threads-per-warp" = 32 : i32' in ttir
    assert '"ttg.num-ctas" = 1 : i32' in ttir


@pytest.mark.parametrize(
    "kernel,diagnostic",
    [
        (
            _num_warps_conflict_kernel,
            r"mthreads TLE layout attribute 'ttg\.num-warps' mismatch: module has 4 : i32, requested 8 : i32",
        ),
        (
            _num_ctas_conflict_kernel,
            r"mthreads TLE layout attribute 'ttg\.num-ctas' mismatch: module has 1 : i32, requested 2 : i32",
        ),
    ],
)
def test_mthreads_tle_layout_attrs_reject_conflicts(kernel, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        compile_to_ttir(kernel, {})


@pytest.mark.parametrize(
    "constexprs,diagnostic",
    [
        (
            {"num_warps": 0, "warp_size": 32, "num_ctas": 1},
            "mthreads TLE num_warps must be a positive power of two",
        ),
        (
            {"num_warps": 3, "warp_size": 32, "num_ctas": 1},
            "mthreads TLE num_warps must be a positive power of two",
        ),
        (
            {"num_warps": 4, "warp_size": 64, "num_ctas": 1},
            r"mthreads TLE PH1 requires warp_size \(threads per warp\) to be 32",
        ),
        (
            {"num_warps": 4, "warp_size": 32, "num_ctas": 0},
            "mthreads TLE num_ctas must be positive",
        ),
    ],
)
def test_mthreads_tle_layout_attrs_reject_invalid_parameters(constexprs, diagnostic):
    signature = {
        "num_warps": "constexpr",
        "warp_size": "constexpr",
        "num_ctas": "constexpr",
    }
    with pytest.raises(CompilationError, match=re.escape(diagnostic) if "\\" not in diagnostic else diagnostic):
        compile_to_ttir(_invalid_layout_attrs_kernel, signature, constexprs)
