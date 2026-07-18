import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _extract_static_scalar_kernel(x_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tile = tle.extract_tile(src, index=0, tile_shape=(16, 16))
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tl.store(out_ptr + tr * 16 + tc, tile)


@triton.jit
def _extract_static_multidim_kernel(x_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tile = tle.extract_tile(src, index=[1, 1], tile_shape=(16, 16))
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tl.store(out_ptr + tr * 16 + tc, tile)


@triton.jit
def _extract_dynamic_scalar_kernel(x_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    index = tl.program_id(0)
    tile = tle.extract_tile(src, index=index, tile_shape=(16, 16))
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tl.store(out_ptr + tr * 16 + tc, tile)


@triton.jit
def _extract_dynamic_multidim_kernel(x_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    idx_m = tl.program_id(0)
    idx_n = tl.program_id(1)
    tile = tle.extract_tile(src, index=[idx_m, idx_n], tile_shape=(16, 16))
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tl.store(out_ptr + tr * 16 + tc, tile)


@triton.jit
def _insert_static_scalar_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tile = tl.load(tile_ptr + tr * 16 + tc)
    result = tle.insert_tile(src, tile, index=0)
    tl.store(out_ptr + rows * 32 + cols, result)


@triton.jit
def _insert_static_multidim_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tile = tl.load(tile_ptr + tr * 16 + tc)
    result = tle.insert_tile(src, tile, index=[1, 1])
    tl.store(out_ptr + rows * 32 + cols, result)


@triton.jit
def _insert_static_2x32_same_encoding_runtime_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 2)[:, None]
    tc = tl.arange(0, 32)[None, :]
    tile = tl.load(tile_ptr + tr * 32 + tc)
    result = tle.insert_tile(src, tile, index=0)
    tl.store(out_ptr + rows * 32 + cols, result)


@triton.jit
def _insert_dynamic_scalar_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tile = tl.load(tile_ptr + tr * 16 + tc)
    index = tl.program_id(0)
    result = tle.insert_tile(src, tile, index=index)
    tl.store(out_ptr + rows * 32 + cols, result)


@triton.jit
def _insert_dynamic_multidim_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tile = tl.load(tile_ptr + tr * 16 + tc)
    idx_m = tl.program_id(0)
    idx_n = tl.program_id(1)
    result = tle.insert_tile(src, tile, index=[idx_m, idx_n])
    tl.store(out_ptr + rows * 32 + cols, result)


@triton.jit
def _extract_runtime_kernel(x_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr, INDEX: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tile = tle.extract_tile(src, index=INDEX, tile_shape=(16, 16))
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tl.store(out_ptr + tr * 16 + tc, tile)


@triton.jit
def _insert_runtime_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr, INDEX: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tile = tl.load(tile_ptr + tr * 16 + tc)
    result = tle.insert_tile(src, tile, index=INDEX)
    tl.store(out_ptr + rows * 32 + cols, result)


@triton.jit
def _extract_dynamic_runtime_kernel(x_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    index = tl.program_id(0)
    tile = tle.extract_tile(src, index=index, tile_shape=(16, 16))
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    tl.store(out_ptr + index * 16 * 16 + tr * 16 + tc, tile)


@triton.jit
def _insert_dynamic_runtime_kernel(x_ptr, tile_ptr, out_ptr, sxm: tl.constexpr, sxn: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    src = tl.load(x_ptr + rows * sxm + cols * sxn)
    tr = tl.arange(0, 16)[:, None]
    tc = tl.arange(0, 16)[None, :]
    index = tl.program_id(0)
    tile = tl.load(tile_ptr + index * 16 * 16 + tr * 16 + tc)
    result = tle.insert_tile(src, tile, index=index)
    tl.store(out_ptr + index * 32 * 32 + rows * 32 + cols, result)


def _compile(fn):
    return compile_musa(
        fn,
        signature={
            "x_ptr": "*fp32",
            "tile_ptr": "*fp32",
            "out_ptr": "*fp32",
            "sxm": "constexpr",
            "sxn": "constexpr",
        },
        constexprs={"sxm": 32, "sxn": 1},
    )


def _compile_extract(fn):
    return compile_musa(
        fn,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "sxm": "constexpr", "sxn": "constexpr"},
        constexprs={"sxm": 32, "sxn": 1},
    )


def _assert_static_non_cta_default_blocked_tile(ttgir, *, tile_shape=(16, 16), same_tile_encoding=True):
    tile_m, tile_n = tile_shape
    assert f"tile_shape = array<i64: {tile_m}, {tile_n}>" in ttgir, ttgir
    assert "tensor<32x32xf32, #blocked>" in ttgir, ttgir
    if same_tile_encoding:
        assert f"tensor<{tile_m}x{tile_n}xf32, #blocked>" in ttgir, ttgir
    assert "sizePerThread = [1, 1]" in ttgir, ttgir
    assert "threadsPerWarp = [1, 32]" in ttgir, ttgir
    assert "warpsPerCTA = [4, 1]" in ttgir, ttgir


_SAME_THREAD_EXTRACT_TTGIR = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @same_thread_extract(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %cst32 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %cst32_2 = arith.constant dense<32> : tensor<2x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %rows32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows32_1 = tt.expand_dims %rows32 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %cols32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols32_1 = tt.expand_dims %cols32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %src_row_off = arith.muli %rows32_1, %cst32 : tensor<32x1xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %x_row_ptr = tt.addptr %x_splat, %src_row_off : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %x_row_ptr_b = tt.broadcast %x_row_ptr : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %cols32_b = tt.broadcast %cols32_1 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %x_ptrs = tt.addptr %x_row_ptr_b, %cols32_b : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %src = tt.load %x_ptrs : tensor<32x32x!tt.ptr<f32>, #blocked>
    %tile = musa_tle.extract_tile %src[%c0_i32] {tile_shape = array<i64: 2, 32>} : tensor<32x32xf32, #blocked>, i32 -> tensor<2x32xf32, #blocked>
    %rows2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2_1 = tt.expand_dims %rows2 {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<2x1xi32, #blocked>
    %out_row_off = arith.muli %rows2_1, %cst32_2 : tensor<2x1xi32, #blocked>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>, #blocked>
    %out_row_ptr = tt.addptr %out_splat, %out_row_off : tensor<2x1x!tt.ptr<f32>, #blocked>, tensor<2x1xi32, #blocked>
    %out_row_ptr_b = tt.broadcast %out_row_ptr : tensor<2x1x!tt.ptr<f32>, #blocked> -> tensor<2x32x!tt.ptr<f32>, #blocked>
    %cols32_out = tt.broadcast %cols32_1 : tensor<1x32xi32, #blocked> -> tensor<2x32xi32, #blocked>
    %out_ptrs = tt.addptr %out_row_ptr_b, %cols32_out : tensor<2x32x!tt.ptr<f32>, #blocked>, tensor<2x32xi32, #blocked>
    tt.store %out_ptrs, %tile : tensor<2x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
"""

_SAME_THREAD_INSERT_TTGIR = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @same_thread_insert(%x_ptr: !tt.ptr<f32>, %tile_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %cst32 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %cst32_2 = arith.constant dense<32> : tensor<2x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %rows32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows32_1 = tt.expand_dims %rows32 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %cols32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols32_1 = tt.expand_dims %cols32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %src_row_off = arith.muli %rows32_1, %cst32 : tensor<32x1xi32, #blocked>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %x_row_ptr = tt.addptr %x_splat, %src_row_off : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %x_row_ptr_b = tt.broadcast %x_row_ptr : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %cols32_b = tt.broadcast %cols32_1 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %x_ptrs = tt.addptr %x_row_ptr_b, %cols32_b : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %src = tt.load %x_ptrs : tensor<32x32x!tt.ptr<f32>, #blocked>
    %rows2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2_1 = tt.expand_dims %rows2 {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<2x1xi32, #blocked>
    %tile_row_off = arith.muli %rows2_1, %cst32_2 : tensor<2x1xi32, #blocked>
    %tile_splat = tt.splat %tile_ptr : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>, #blocked>
    %tile_row_ptr = tt.addptr %tile_splat, %tile_row_off : tensor<2x1x!tt.ptr<f32>, #blocked>, tensor<2x1xi32, #blocked>
    %tile_row_ptr_b = tt.broadcast %tile_row_ptr : tensor<2x1x!tt.ptr<f32>, #blocked> -> tensor<2x32x!tt.ptr<f32>, #blocked>
    %cols32_tile = tt.broadcast %cols32_1 : tensor<1x32xi32, #blocked> -> tensor<2x32xi32, #blocked>
    %tile_ptrs = tt.addptr %tile_row_ptr_b, %cols32_tile : tensor<2x32x!tt.ptr<f32>, #blocked>, tensor<2x32xi32, #blocked>
    %tile = tt.load %tile_ptrs : tensor<2x32x!tt.ptr<f32>, #blocked>
    %result = musa_tle.insert_tile %src[%c0_i32] = %tile {tile_shape = array<i64: 2, 32>} : tensor<32x32xf32, #blocked>, i32, tensor<2x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %out_row_ptr = tt.addptr %out_splat, %src_row_off : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %out_row_ptr_b = tt.broadcast %out_row_ptr : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.addptr %out_row_ptr_b, %cols32_b : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    tt.store %out_ptrs, %result : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
"""


def _lower_ttgir_fixture_to_llvm(tmp_path, ttgir):
    from triton._C import libtriton
    from triton._C.libtriton import ir, passes

    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    fixture_path = tmp_path / "tile_fixture.ttgir"
    fixture_path.write_text(ttgir)
    module = ir.parse_mlir_module(str(fixture_path), context)

    pm = ir.pass_manager(context)
    passes.convert.add_scf_to_cf(pm)
    passes.convert.add_index_to_llvmir(pm)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(module, "lower_tile_fixture")
    return module.str()


@pytest.mark.parametrize(
    "kernel",
    [
        _extract_static_scalar_kernel,
        _extract_static_multidim_kernel,
        _extract_dynamic_scalar_kernel,
        _extract_dynamic_multidim_kernel,
    ],
)
def test_extract_tile_compiles_through_mthreads_tle(kernel):
    compiled = _compile_extract(kernel)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "musa_tle.extract_tile" in ttgir, ttgir
    assert "musa_tle.extract_tile" not in llir, llir
    assert "nvvm" not in llir.lower(), llir


@pytest.mark.parametrize(
    "kernel",
    [
        _insert_static_scalar_kernel,
        _insert_static_multidim_kernel,
        _insert_dynamic_scalar_kernel,
        _insert_dynamic_multidim_kernel,
    ],
)
def test_insert_tile_compiles_through_mthreads_tle(kernel):
    compiled = _compile(kernel)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]

    assert "musa_tle.insert_tile" in ttgir, ttgir
    assert "musa_tle.insert_tile" not in llir, llir
    assert "nvvm" not in llir.lower(), llir


def test_dynamic_tile_ops_lower_with_musa_local_barriers():
    extract = _compile_extract(_extract_dynamic_scalar_kernel)
    insert = _compile(_insert_dynamic_scalar_kernel)

    assert "llvm.musa.syncthreads.lm" in extract.asm["llir"], extract.asm["llir"]
    assert "llvm.musa.syncthreads.lm" in insert.asm["llir"], insert.asm["llir"]


def test_static_non_cta_extract_lowers_with_smem_barrier(tmp_path):
    llir = _lower_ttgir_fixture_to_llvm(tmp_path, _SAME_THREAD_EXTRACT_TTGIR)

    assert "musa_tle.extract_tile" in _SAME_THREAD_EXTRACT_TTGIR
    _assert_static_non_cta_default_blocked_tile(_SAME_THREAD_EXTRACT_TTGIR, tile_shape=(2, 32))
    assert "musa_tle.extract_tile" not in llir, llir
    assert "llvm.musa.syncthreads.lm" in llir, llir


def test_static_non_cta_insert_lowers_with_smem_barrier(tmp_path):
    llir = _lower_ttgir_fixture_to_llvm(tmp_path, _SAME_THREAD_INSERT_TTGIR)

    assert "musa_tle.insert_tile" in _SAME_THREAD_INSERT_TTGIR
    _assert_static_non_cta_default_blocked_tile(_SAME_THREAD_INSERT_TTGIR, tile_shape=(2, 32))
    assert "musa_tle.insert_tile" not in llir, llir
    assert "llvm.musa.syncthreads.lm" in llir, llir


def test_static_non_cta_insert_encoding_mismatch_uses_smem_barrier():
    compiled = _compile(_insert_static_scalar_kernel)

    _assert_static_non_cta_default_blocked_tile(compiled.asm["ttgir"], same_tile_encoding=False)
    assert "tensor<16x16xf32, #blocked1>" in compiled.asm["ttgir"], compiled.asm["ttgir"]
    assert "llvm.musa.syncthreads.lm" in compiled.asm["llir"], compiled.asm["llir"]


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("index, row, col", [(0, 0, 0), (3, 16, 16)])
def test_extract_tile_runtime_matches_source_tile(index, row, col):
    x = torch.arange(32 * 32, device="musa", dtype=torch.float32).reshape(32, 32)
    out = torch.empty((16, 16), device="musa", dtype=torch.float32)

    _extract_runtime_kernel[(1, )](x, out, x.stride(0), x.stride(1), INDEX=index, num_warps=4)

    torch.testing.assert_close(out.cpu(), x[row:row + 16, col:col + 16].cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_insert_tile_runtime_updates_selected_region_only():
    x = torch.arange(32 * 32, device="musa", dtype=torch.float32).reshape(32, 32)
    tile = torch.full((16, 16), 777.0, device="musa", dtype=torch.float32)
    out = torch.empty_like(x)

    _insert_runtime_kernel[(1, )](x, tile, out, x.stride(0), x.stride(1), INDEX=3, num_warps=4)

    expected = x.clone()
    expected[16:32, 16:32] = tile
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_insert_tile_static_2x32_same_encoding_runtime_updates_selected_region_only():
    x = torch.arange(32 * 32, device="musa", dtype=torch.float32).reshape(32, 32)
    tile = torch.full((2, 32), 777.0, device="musa", dtype=torch.float32)
    out = torch.empty_like(x)

    _insert_static_2x32_same_encoding_runtime_kernel[(1, )](x, tile, out, x.stride(0), x.stride(1), num_warps=2)

    expected = x.clone()
    expected[0:2, 0:32] = tile
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_extract_tile_dynamic_runtime_matches_all_source_tiles():
    x = torch.arange(32 * 32, device="musa", dtype=torch.float32).reshape(32, 32)
    out = torch.empty((4, 16, 16), device="musa", dtype=torch.float32)

    _extract_dynamic_runtime_kernel[(4, )](x, out, x.stride(0), x.stride(1), num_warps=4)

    expected = torch.stack([
        x[0:16, 0:16],
        x[0:16, 16:32],
        x[16:32, 0:16],
        x[16:32, 16:32],
    ])
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_insert_tile_dynamic_runtime_updates_each_selected_region():
    x = torch.arange(32 * 32, device="musa", dtype=torch.float32).reshape(32, 32)
    tile = torch.arange(4 * 16 * 16, device="musa", dtype=torch.float32).reshape(4, 16, 16)
    tile = tile + 10000
    out = torch.empty((4, 32, 32), device="musa", dtype=torch.float32)

    _insert_dynamic_runtime_kernel[(4, )](x, tile, out, x.stride(0), x.stride(1), num_warps=4)

    expected = x.expand(4, 32, 32).clone()
    expected[0, 0:16, 0:16] = tile[0]
    expected[1, 0:16, 16:32] = tile[1]
    expected[2, 16:32, 0:16] = tile[2]
    expected[3, 16:32, 16:32] = tile[3]
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)
