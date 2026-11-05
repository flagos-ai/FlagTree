"""A staged KV allocation can serve both K-transpose and V consumers."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_warp_specialize import _compile_ws_ir


@triton.jit
def shared_kv_kernel(Q, KV, P, QK, PV, DTYPE: tl.constexpr, STAGES: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    query = tl.load(Q + rows * 256 + tl.arange(0, 256)[None, :])
    key_value = tl.load(KV + tl.arange(0, STAGES)[:, None, None] * (64 * 256) + tl.arange(0, 64)[None, :, None] * 256 +
                        tl.arange(0, 256)[None, None, :])
    probability = tl.load(P + rows * 64 + tl.arange(0, 64)[None, :])
    q = tle.gpu.alloc((32, 256), DTYPE, init_value=query)
    kv = tle.gpu.alloc((STAGES, 64, 256), DTYPE, init_value=key_value)
    p = tle.gpu.alloc((32, 64), DTYPE, init_value=probability)
    tl.debug_barrier()
    tile = kv.slot(STAGES - 1)
    scores = tle.gpu.wgmma(q, tile, trans_b=True)
    scores = tle.gpu.wgmma_wait(0, scores)
    output = tle.gpu.wgmma(p, tile)
    output = tle.gpu.wgmma_wait(0, output)
    tl.store(QK + rows * 64 + tl.arange(0, 64)[None, :], scores)
    tl.store(PV + rows * 256 + tl.arange(0, 256)[None, :], output)


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_transpose_and_value(stages):
    _, ttgir, lowered = _compile_ws_ir(
        shared_kv_kernel,
        {
            "Q": "*fp8e4nv",
            "KV": "*fp8e4nv",
            "P": "*fp8e4nv",
            "QK": "*fp32",
            "PV": "*fp32",
            "DTYPE": "constexpr",
            "STAGES": "constexpr",
        },
        {"DTYPE": tl.float8e4nv, "STAGES": stages},
        consumer_warps=4,
        include_late=True,
    )
    # Both consumers must survive without allocating a second KV tile.
    assert ttgir.count(" = ttmg.squad_dot ") == 2
    assert ttgir.count(" = ttg.local_alloc ") == 3
    assert "llvm.musa.sqmma.e4m3" in lowered


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_rejects_incompatible_physical_layouts(stages, capfd):
    # The BF16 version needs different physical swizzles for these two views.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ws_ir(
            shared_kv_kernel,
            {
                "Q": "*bf16",
                "KV": "*bf16",
                "P": "*bf16",
                "QK": "*fp32",
                "PV": "*fp32",
                "DTYPE": "constexpr",
                "STAGES": "constexpr",
            },
            {"DTYPE": tl.bfloat16, "STAGES": stages},
            consumer_warps=4,
        )
    assert "SQMMA consumers require incompatible physical shared layouts" in capfd.readouterr().err


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_fp8_runtime_precision(stages):
    import torch_musa

    if not torch_musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(1450)
    torch_dtype = torch.float8_e4m3fn
    query = (torch.randn((32, 256)) * 0.25).to(torch_dtype)
    key_value = (torch.randn((stages, 64, 256)) * 0.25).to(torch_dtype)
    probability = (torch.randn((32, 64)) * 0.25).to(torch_dtype)
    tile = key_value[-1].double()
    expected_qk = query.double() @ tile.T
    expected_pv = probability.double() @ tile

    qk = torch.empty((32, 64), dtype=torch.float32, device="musa")
    pv = torch.empty((32, 256), dtype=torch.float32, device="musa")
    kernel = shared_kv_kernel[(1, )](
        query.to("musa"),
        key_value.to("musa"),
        probability.to("musa"),
        qk,
        pv,
        tl.float8e4nv,
        stages,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(qk.cpu(), expected_qk.float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(pv.cpu(), expected_pv.float(), rtol=1e-3, atol=2e-3)
    assert "llvm.musa.sqmma.e4m3" in kernel.asm["llir"]
