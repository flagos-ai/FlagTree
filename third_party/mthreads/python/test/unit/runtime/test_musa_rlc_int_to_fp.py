"""Exercise the RLC vector-width lowering with exact CPU rounding references."""

import pytest
import triton
from triton._C import libtriton

pytestmark = pytest.mark.skipif(
    not hasattr(libtriton, "mthreads"), reason="MThreads backend is required"
)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("tail", [0, 3])
def test_preserved_int_to_fp_rounding(signed, width, tail, tmp_path):
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    block = 128 * width
    n = block - tail
    values = [0, 1, (1 << 24) - 1, (1 << 24) + 1, (1 << 24) + 3,
              (1 << 25) + 2, (1 << 25) + 6, (1 << 31) - 1]
    values += [-1, -(1 << 24) - 1, -(1 << 24) - 3, -(1 << 31)] if signed else [
        1 << 31, (1 << 31) + 129, (1 << 32) - 129, (1 << 32) - 1
    ]
    values = (values * ((n + len(values) - 1) // len(values)))[:n]
    bits = [v if v < (1 << 31) else v - (1 << 32) for v in values]
    src = torch.tensor(bits, dtype=torch.int32, device="musa")
    dst = torch.full((block,), float("nan"), dtype=torch.float32, device="musa")
    op = "sitofp" if signed else "uitofp"
    ir = f"""
#b = #ttg.blocked<{{sizePerThread = [{width}], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "musa:31"}} {{
  tt.func public @convert(%input: !tt.ptr<i32>, %output: !tt.ptr<f32>, %n: i32) {{
    %i = tt.make_range {{start = 0 : i32, end = {block} : i32}} : tensor<{block}xi32, #b>
    %ns = tt.splat %n : i32 -> tensor<{block}xi32, #b>
    %mask = arith.cmpi slt, %i, %ns : tensor<{block}xi32, #b>
    %ip = tt.splat %input : !tt.ptr<i32> -> tensor<{block}x!tt.ptr<i32>, #b>
    %ips = tt.addptr %ip, %i : tensor<{block}x!tt.ptr<i32>, #b>, tensor<{block}xi32, #b>
    %x = tt.load %ips, %mask : tensor<{block}x!tt.ptr<i32>, #b>
    %y = arith.{op} %x {{"ttg.rlc-preserve-int-to-fp-vector-width" = {width} : i32}} : tensor<{block}xi32, #b> to tensor<{block}xf32, #b>
    %out = tt.splat %output : !tt.ptr<f32> -> tensor<{block}x!tt.ptr<f32>, #b>
    %outs = tt.addptr %out, %i : tensor<{block}x!tt.ptr<f32>, #b>, tensor<{block}xi32, #b>
    tt.store %outs, %y, %mask : tensor<{block}x!tt.ptr<f32>, #b>
    tt.return
  }}
}}
"""
    path = tmp_path / "convert.ttgir"
    path.write_text(ir)
    kernel = triton.compile(str(path))
    kernel[(1, 1, 1)](src, dst, n)
    actual = dst.cpu()
    # Binary64 represents all i32/u32 inputs exactly; CPU conversion is the oracle.
    expected = torch.tensor(values, dtype=torch.float64).float()
    torch.testing.assert_close(actual[:n], expected, atol=0, rtol=0)
    assert torch.isnan(actual[n:]).all()
