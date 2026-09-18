"""Masked loads/stores and tail tiles under RLC off vs on."""

import pytest

from rlc_gate_common import launch_masked_gemm, rlc_compile_env
from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()

_RLC_CASES = (
    pytest.param(False, 15, id="rlc-off"),
    pytest.param(True, 15, id="rlc-on"),
)
_SHAPE_CASES = (
    pytest.param(128, 128, 64, 128, 128, 64, 4, "sqmma-full", 5e-2, 5e-2, id="sqmma-full"),
    pytest.param(100, 90, 80, 128, 128, 64, 4, "sqmma-tail", 5e-2, 5e-2, id="sqmma-tail"),
    pytest.param(100, 128, 64, 128, 128, 64, 4, "sqmma-mask-m", 5e-2, 5e-2, id="sqmma-mask-m"),
    pytest.param(20, 20, 24, 16, 16, 16, 4, "wmma-tail", 8e-2, 8e-2, id="wmma-tail"),
    pytest.param(16, 16, 16, 16, 16, 16, 4, "wmma-full", 8e-2, 8e-2, id="wmma-full"),
)


@pytest.mark.parametrize("enhance,phase_mask", _RLC_CASES)
@pytest.mark.parametrize("m,n,k,block_m,block_n,block_k,num_warps,case_id,atol,rtol", _SHAPE_CASES)
def test_masked_gemm_tails_runtime(enhance, phase_mask, m, n, k, block_m, block_n, block_k,
                                   num_warps, case_id, atol, rtol, tmp_path):
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    a_cpu = torch.randn((m, k), dtype=torch.float16)
    b_cpu = torch.randn((k, n), dtype=torch.float16)
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"{case_id}-{int(enhance)}")):
        a = a_cpu.to("musa")
        b = b_cpu.to("musa")
        out = torch.empty((m, n), device="musa", dtype=torch.float32)
        kernel = launch_masked_gemm(a, b, out, block_m, block_n, block_k, num_warps=num_warps)
        torch.musa.synchronize()
        expected = torch.matmul(a_cpu.float(), b_cpu.float())
        torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
        ttgir = kernel.asm["ttgir"]
        if case_id.startswith("wmma"):
            assert "#ttg.musa_wmma" in ttgir
        assert "tt.store" in ttgir
        assert "tt.load" in ttgir
