"""No-op layout selection must retain the proved gfx936 BUFFER width."""
import pytest
from triton._C.libtriton import ir
try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires HCU backend", allow_module_level=True)

BASE = r"""#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @outer_factor_inner_unknown_stride(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i32 {tt.divisibility = 8 : i32}, %stride: i32 {tt.divisibility = 16 : i32}) {
    %m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #sM>
    %k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<128xi32, #sM> -> tensor<128x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<32xi32, #sK> -> tensor<1x32xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<128x32xi32, #blocked>
    %sixteen = arith.constant dense<32> : tensor<128x32xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<128x32xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<128x32xi32, #blocked>
    %stridef = tt.splat %stride : i32 -> tensor<128x32xi32, #blocked>
    %row = arith.muli %mf, %stridef : tensor<128x32xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<128x32xi32, #blocked>
    %us = tt.splat %U : i32 -> tensor<128x32xi32, #blocked>
    %sum = arith.addi %us, %v0 : tensor<128x32xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<128x32xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<128x32xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<128x32xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<128x32xbf16, #blocked>) -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%sum] into %dst  : <bf16>[tensor<128x32xi32, #blocked>] -> <128x32xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> tensor<128x32xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<128x32xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<128x32xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<128x32x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<128x32x!tt.ptr<bf16>, #blocked>, tensor<128x32xi32, #blocked>
    tt.store %optrs, %got : tensor<128x32x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}
"""


def fixture(kind, dtype):
    text = BASE.replace('bf16', dtype)
    if kind in ['width8', 'width16']:
        width = 8 if kind == 'width8' else 16
        text = text.replace('into %dst  :', 'into %dst {contiguity = ' + str(width) + ' : i32} :')
    elif kind in ['mask_coarse', 'mask_fine']:
        text = text.replace('%A[%sum] into', '%A[%sum] mask = %mask into')
        if kind == 'mask_fine':
            text = text.replace('dense<8>', 'dense<1>')
    elif kind == 'uniform_mask':
        text = text.replace('%stride: i32 {tt.divisibility = 16 : i32})',
                            '%stride: i32 {tt.divisibility = 16 : i32}, %take: i1)')
        text = text.replace('    %t = hcug.buffer_load_to_local',
                            '    %uniform_mask = tt.splat %take : i1 -> tensor<128x32xi1, #blocked>\n'
                            '    %t = hcug.buffer_load_to_local')
        text = text.replace('%A[%sum] into', '%A[%sum] mask = %uniform_mask into')
    elif kind == 'unknown_offset':
        text = text.replace('%U: i32 {tt.divisibility = 8 : i32}', '%U: i32')
    elif kind == 'unknown_stride':
        text = text.replace('%stride: i32 {tt.divisibility = 16 : i32}', '%stride: i32')
    return text


def run_pass(tmp_path, text):
    path = tmp_path / 'coalesce-width.mlir'
    path.write_text(text)
    ctx = ir.context()
    ir.load_dialects(ctx)
    hcu.load_dialects(ctx)
    module = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    hcu.passes.ttgpuir.add_coalesce_async_copy(pm, 'gfx936')
    pm.run(module, 'coalesce_width')
    first = str(module)
    # Each run follows the real pipeline's fresh manager/context lifecycle.
    second_path = tmp_path / 'coalesce-width-second.mlir'
    second_path.write_text(first)
    second_ctx = ir.context()
    ir.load_dialects(second_ctx)
    hcu.load_dialects(second_ctx)
    second = ir.parse_mlir_module(str(second_path), second_ctx)
    second_pm = ir.pass_manager(second_ctx)
    hcu.passes.ttgpuir.add_coalesce_async_copy(second_pm, 'gfx936')
    second_pm.run(second, 'coalesce_width_idempotence')
    assert str(second) == first
    return first


@pytest.mark.parametrize('dtype', ['bf16', 'f16'])
@pytest.mark.parametrize('kind', ['default', 'width8', 'width16', 'uniform_mask', 'mask_coarse',
                                 'mask_fine', 'unknown_offset', 'unknown_stride'])
def test_existing_layout_width_and_legality(tmp_path, kind, dtype):
    text = fixture(kind, dtype)
    out = run_pass(tmp_path, text)
    # AxisInfo currently does not prove the bitwise grouped mask's alignment.
    # Preserve that conservative fallback rather than bypassing mask analysis.
    fallback = kind in ['mask_coarse', 'mask_fine', 'unknown_offset', 'unknown_stride']
    assert out.count('hcug.buffer_load_to_local') == int(not fallback)
    assert out.count('hcug.buffer_load ') == int(fallback)
    if not fallback:
        assert 'contiguity = 8 : i32' in out
    assert out.count('ttg.local_barrier') == text.count('ttg.local_barrier')
    assert out.count('ttg.async_wait') == text.count('ttg.async_wait')
