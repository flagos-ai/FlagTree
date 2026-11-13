"""Packed legacy MMAC must preserve the requested accumulator transpose."""
import pytest
from triton._C.libtriton import ir

try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip('Requires HCU backend', allow_module_level=True)


def run(tmp_path, dtype, k, pack, transpose_requested):
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#dotA = #ttg.dot_op<{{opIdx = 0, parent = #blocked, kWidth = 0}}>
#dotB = #ttg.dot_op<{{opIdx = 1, parent = #blocked, kWidth = 0}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32, "ttg.hcu.mmac_transposed" = {transpose_requested} : i32}} {{
  tt.func @fixture(%a: tensor<64x{k}x{dtype}, #dotA>, %b: tensor<{k}x128x{dtype}, #dotB>, %c: tensor<64x128xf32, #blocked>) -> tensor<64x128xf32, #blocked> {{
    %out = tt.dot %a, %b, %c : tensor<64x{k}x{dtype}, #dotA> * tensor<{k}x128x{dtype}, #dotB> -> tensor<64x128xf32, #blocked>
    tt.return %out : tensor<64x128xf32, #blocked>
  }}
}}
'''
    f = tmp_path / 'input.mlir'
    f.write_text(text)
    ctx = ir.context()
    ir.load_dialects(ctx)
    hcu.load_dialects(ctx)
    module = ir.parse_mlir_module(str(f), ctx)
    pm = ir.pass_manager(ctx)
    hcu.passes.ttgpuir.add_accelerate_matmul(pm, 'gfx936', 16, pack, 1)
    pm.run(module, 'transpose_pack2_domain')
    return str(module)


@pytest.mark.parametrize('dtype', ['bf16', 'f16'])
@pytest.mark.parametrize(('k', 'pack'), [(16, 1), (32, 1), (32, 2), (64, 1), (64, 2)])
@pytest.mark.parametrize('transpose_requested', [0, 1])
def test_requested_transpose_with_packed_k_groups(tmp_path, dtype, k, pack, transpose_requested):
    out = run(tmp_path, dtype, k, pack, transpose_requested)
    assert '#ttg.amd_mfma' in out
    assert f'kWidth = {4 * pack}' in out
    selected = bool(transpose_requested)
    assert ('isTransposed = true' in out) == selected
