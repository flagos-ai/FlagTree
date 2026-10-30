"""A staged KV allocation can serve both K-transpose and V consumers."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_warp_specialize import _compile_ws_ir


@triton.jit
def shared_kv_kernel(QK, PV, DTYPE: tl.constexpr, STAGES: tl.constexpr):
    q = tle.gpu.alloc((32, 256), DTYPE)
    kv = tle.gpu.alloc((STAGES, 64, 256), DTYPE)
    p = tle.gpu.alloc((32, 64), DTYPE)
    tile = kv.slot(STAGES - 1)
    scores = tle.gpu.wgmma(q, tile, trans_b=True)
    scores = tle.gpu.wgmma_wait(0, scores)
    output = tle.gpu.wgmma(p, tile)
    output = tle.gpu.wgmma_wait(0, output)
    rows = tl.arange(0, 32)[:, None]
    tl.store(QK + rows * 64 + tl.arange(0, 64)[None, :], scores)
    tl.store(PV + rows * 256 + tl.arange(0, 256)[None, :], output)


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_transpose_and_value(stages):
    _, ttgir = _compile_ws_ir(
        shared_kv_kernel,
        {"QK": "*fp32", "PV": "*fp32", "DTYPE": "constexpr", "STAGES": "constexpr"},
        {"DTYPE": tl.float8e4nv, "STAGES": stages},
        consumer_warps=4,
    )
    # Both consumers must survive without allocating a second KV tile.
    assert ttgir.count(" = ttmg.squad_dot ") == 2
    assert ttgir.count(" = ttg.local_alloc ") == 3


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_rejects_incompatible_physical_layouts(stages, capfd):
    # The BF16 version needs different physical swizzles for these two views.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ws_ir(
            shared_kv_kernel,
            {"QK": "*fp32", "PV": "*fp32", "DTYPE": "constexpr", "STAGES": "constexpr"},
            {"DTYPE": tl.bfloat16, "STAGES": stages},
            consumer_warps=4,
        )
    assert "SQMMA consumers require incompatible physical shared layouts" in capfd.readouterr().err
