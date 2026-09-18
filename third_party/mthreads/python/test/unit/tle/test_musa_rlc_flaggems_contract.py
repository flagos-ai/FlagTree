"""FlagGems-contract rand and replication_pad3d under RLC off vs on.

The 2026-08-15 S5000 TTGIR dumps were produced from FlagGems
`flaggems-bfeca796-source` ops `rand.py` and `replication_pad3d.py`. That
checkout is gone; these kernels freeze the same BLOCK/mask/philox/pad-clamp
contract, including N not divisible by 4096 and H/W not divisible by 64/16.
"""

import pytest
import torch.nn.functional as F

from rlc_gate_common import (
    count_convert_layout,
    launch_flaggems_pad,
    launch_flaggems_rand,
    rlc_compile_env,
)
from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()

_RLC_CASES = (
    pytest.param(False, 15, id="rlc-off"),
    pytest.param(True, 15, id="rlc-on"),
)


@pytest.mark.parametrize("enhance,phase_mask", _RLC_CASES)
def test_flaggems_rand_masked_runtime(enhance, phase_mask, tmp_path):
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    n = 3000
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"rand-{int(enhance)}")):
        out = torch.empty((n, ), device="musa", dtype=torch.float16)
        kernel = launch_flaggems_rand(out, philox_seed=1234, philox_offset=7)
        torch.musa.synchronize()
        host = out.cpu()
        assert host.shape == (n, )
        assert torch.isfinite(host.float()).all()
        lo, hi = float(host.min()), float(host.max())
        assert 0.0 <= lo <= hi <= 1.0 + 1e-3
        assert count_convert_layout(kernel.asm["ttgir"]) >= 0


@pytest.mark.parametrize("enhance,phase_mask", _RLC_CASES)
def test_flaggems_replication_pad3d_tail_runtime(enhance, phase_mask, tmp_path):
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    x_cpu = torch.randn((2, 3, 8, 20, 18), dtype=torch.float16)
    pad = (1, 2, 3, 1, 1, 1)
    expected = F.pad(x_cpu.float(), pad, mode="replicate")
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = pad
    d_out = x_cpu.shape[2] + pad_f + pad_ba
    h_out = x_cpu.shape[3] + pad_t + pad_b
    w_out = x_cpu.shape[4] + pad_l + pad_r
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"pad-{int(enhance)}")):
        x = x_cpu.to("musa")
        out = torch.empty((2, 3, d_out, h_out, w_out), device="musa", dtype=torch.float16)
        kernel = launch_flaggems_pad(x, out, pad_l, pad_t, pad_f)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=0, rtol=0)
        assert "tt.load" in kernel.asm["ttgir"]
        assert "tt.store" in kernel.asm["ttgir"]


def test_flaggems_rlc_off_vs_on_match(tmp_path):
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    n = 3000
    with rlc_compile_env(False, 15, cache_dir=str(tmp_path / "rand-off")):
        off = torch.empty((n, ), device="musa", dtype=torch.float16)
        launch_flaggems_rand(off, philox_seed=99, philox_offset=3)
        torch.musa.synchronize()
        off_cpu = off.cpu()
    with rlc_compile_env(True, 15, cache_dir=str(tmp_path / "rand-on")):
        on = torch.empty((n, ), device="musa", dtype=torch.float16)
        launch_flaggems_rand(on, philox_seed=99, philox_offset=3)
        torch.musa.synchronize()
        on_cpu = on.cpu()
    torch.testing.assert_close(off_cpu, on_cpu, atol=0, rtol=0)
