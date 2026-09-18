"""Broader FlagGems-category screen: convert counts and off-vs-on correctness.

These are self-contained kernels matching FlagGems softmax / rms_norm /
dropout / gelu_and_mul / addmm contracts. They exist because rand+pad did
not change convert counts, so they cannot decide whether RLC is profitable.
"""

import math

import pytest
import torch

from rlc_gate_common import (
    count_convert_layout,
    launch_flaggems_addmm,
    launch_flaggems_dropout,
    launch_flaggems_gelu_and_mul,
    launch_flaggems_rms_norm,
    launch_flaggems_softmax,
    rlc_compile_env,
)
from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()

_RLC = (
    pytest.param(False, 15, id="rlc-off"),
    pytest.param(True, 15, id="rlc-on"),
)


def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


@pytest.mark.parametrize("enhance,phase_mask", _RLC)
def test_flaggems_softmax_runtime(enhance, phase_mask, tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(0)
    x_cpu = torch.randn((32, 100), dtype=torch.float16)
    expected = torch.softmax(x_cpu.float(), dim=-1)
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"softmax-{int(enhance)}")):
        x = x_cpu.to("musa")
        out = torch.empty_like(x)
        kernel = launch_flaggems_softmax(x, out, block_n=128)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=2e-3, rtol=2e-3)
        assert count_convert_layout(kernel.asm["ttgir"]) >= 0


@pytest.mark.parametrize("enhance,phase_mask", _RLC)
def test_flaggems_rms_norm_runtime(enhance, phase_mask, tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(1)
    x_cpu = torch.randn((16, 96), dtype=torch.float16)
    w_cpu = torch.randn((96, ), dtype=torch.float16)
    xf = x_cpu.float()
    rrms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = (xf * rrms).to(torch.float16).float() * w_cpu.float()
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"rms-{int(enhance)}")):
        x = x_cpu.to("musa")
        w = w_cpu.to("musa")
        out = torch.empty_like(x)
        kernel = launch_flaggems_rms_norm(x, w, out)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=3e-3, rtol=3e-3)
        assert count_convert_layout(kernel.asm["ttgir"]) >= 0


@pytest.mark.parametrize("enhance,phase_mask", _RLC)
def test_flaggems_gelu_and_mul_runtime(enhance, phase_mask, tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(2)
    x_cpu = torch.randn((3000, ), dtype=torch.float16)
    y_cpu = torch.randn((3000, ), dtype=torch.float16)
    expected = _gelu(x_cpu.float()) * y_cpu.float()
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"gelu-{int(enhance)}")):
        x = x_cpu.to("musa")
        y = y_cpu.to("musa")
        out = torch.empty_like(x)
        kernel = launch_flaggems_gelu_and_mul(x, y, out)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=3e-3, rtol=3e-3)
        assert count_convert_layout(kernel.asm["ttgir"]) >= 0


@pytest.mark.parametrize("enhance,phase_mask", _RLC)
def test_flaggems_addmm_runtime(enhance, phase_mask, tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(3)
    a_cpu = torch.randn((100, 80), dtype=torch.float16)
    b_cpu = torch.randn((80, 90), dtype=torch.float16)
    c_cpu = torch.randn((100, 90), dtype=torch.float16)
    expected = torch.addmm(c_cpu.float(), a_cpu.float(), b_cpu.float())
    with rlc_compile_env(enhance, phase_mask, cache_dir=str(tmp_path / f"addmm-{int(enhance)}")):
        a = a_cpu.to("musa")
        b = b_cpu.to("musa")
        c = c_cpu.to("musa")
        out = torch.empty_like(c)
        kernel = launch_flaggems_addmm(a, b, c, out)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=8e-2, rtol=8e-2)
        assert "tt.dot" in kernel.asm["ttgir"] or "musa_sqmma" in kernel.asm["ttgir"] or "musa_wmma" in kernel.asm["ttgir"]


def test_flaggems_dropout_off_vs_on(tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(4)
    x_cpu = torch.randn((3000, ), dtype=torch.float16)
    with rlc_compile_env(False, 15, cache_dir=str(tmp_path / "drop-off")):
        x = x_cpu.to("musa")
        off = torch.empty_like(x)
        k_off = launch_flaggems_dropout(x, off, p=0.1, philox_seed=11, philox_offset=3)
        torch.musa.synchronize()
        off_cpu = off.cpu()
        off_cvt = count_convert_layout(k_off.asm["ttgir"])
    with rlc_compile_env(True, 15, cache_dir=str(tmp_path / "drop-on")):
        x = x_cpu.to("musa")
        on = torch.empty_like(x)
        k_on = launch_flaggems_dropout(x, on, p=0.1, philox_seed=11, philox_offset=3)
        torch.musa.synchronize()
        on_cpu = on.cpu()
        on_cvt = count_convert_layout(k_on.asm["ttgir"])
    torch.testing.assert_close(off_cpu, on_cpu, atol=0, rtol=0)
    assert off_cvt >= 0 and on_cvt >= 0
