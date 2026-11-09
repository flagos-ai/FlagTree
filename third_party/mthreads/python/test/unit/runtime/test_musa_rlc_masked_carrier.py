"""Device regression for grouped BF16 masked reads and their fallback path."""
from pathlib import Path
import re

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("holes", [False, True])
@pytest.mark.parametrize("other", [False, True])
@pytest.mark.parametrize("n", [0, 144, 1024])
def test_bf16_masked_group_values(paired, holes, other, n, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_musa")
    if not hasattr(libtriton, "mthreads") or not torch.musa.is_available():
        pytest.skip("MThreads backend and MUSA device are required")

    mask = """
    %mask = arith.cmpi slt, %r, %bound : tensor<1024xi32, #b>
"""
    if holes:
        mask = """
    %upper = arith.cmpi slt, %r, %bound : tensor<1024xi32, #b>
    %bit = arith.constant dense<128> : tensor<1024xi32, #b>
    %zero = arith.constant dense<0> : tensor<1024xi32, #b>
    %bits = arith.andi %r, %bit : tensor<1024xi32, #b>
    %hole = arith.cmpi ne, %bits, %zero : tensor<1024xi32, #b>
    %mask = arith.andi %upper, %hole : tensor<1024xi1, #b>
"""
    a_other = ", %other_a" if other else ""
    b_other = ", %other_b" if other else ""
    peer = f"""
    %base_b = tt.splat %input_b : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #b>
    %p_b = tt.addptr %base_b, %offset : tensor<1024x!tt.ptr<bf16>, #b>, tensor<1024xi32, #b>
    %v_b = tt.load %p_b, %mask{b_other} : tensor<1024x!tt.ptr<bf16>, #b>
""" if paired else ""
    store_peer = """
    %out_b = tt.splat %output_b : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #b>
    %q_b = tt.addptr %out_b, %r : tensor<1024x!tt.ptr<bf16>, #b>, tensor<1024xi32, #b>
    tt.store %q_b, %v_b : tensor<1024x!tt.ptr<bf16>, #b>
""" if paired else ""
    source = f"""
#b = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"}} {{
  tt.func public @masked_values(%input: !tt.ptr<bf16>, %input_b: !tt.ptr<bf16>, %output: !tt.ptr<bf16>, %output_b: !tt.ptr<bf16>, %n: i32 {{tt.divisibility = 16 : i32}}) {{
    %r = tt.make_range {{start = 0 : i32, end = 1024 : i32}} : tensor<1024xi32, #b>
    %two = arith.constant dense<2> : tensor<1024xi32, #b>
    %offset = arith.muli %r, %two : tensor<1024xi32, #b>
    %base = tt.splat %input : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #b>
    %p = tt.addptr %base, %offset : tensor<1024x!tt.ptr<bf16>, #b>, tensor<1024xi32, #b>
    %bound = tt.splat %n : i32 -> tensor<1024xi32, #b>
    {mask}
    %other_a = arith.constant dense<7.0> : tensor<1024xbf16, #b>
    %other_b = arith.constant dense<-3.0> : tensor<1024xbf16, #b>
    %v = tt.load %p, %mask{a_other} : tensor<1024x!tt.ptr<bf16>, #b>
    {peer}
    %out = tt.splat %output : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #b>
    %q = tt.addptr %out, %r : tensor<1024x!tt.ptr<bf16>, #b>, tensor<1024xi32, #b>
    tt.store %q, %v : tensor<1024x!tt.ptr<bf16>, #b>
    {store_peer}
    tt.return
  }}
}}
"""
    path = tmp_path / "masked_values.ttgir"
    path.write_text(source)
    kernel = triton.compile(str(path), target=GPUTarget("musa", 31, 32))
    if not holes:
        assert len(re.findall(r"\bbr i1\b", kernel.asm["llir"])) <= 2
    cpu_a = torch.arange(2048, dtype=torch.float32).to(torch.bfloat16)
    cpu_a[0], cpu_a[2], cpu_a[4], cpu_a[6] = float("nan"), float("inf"), -float("inf"), -0.0
    cpu_b = cpu_a.roll(8)
    x = cpu_a.to("musa") if n else torch.empty(0, dtype=torch.bfloat16, device="musa")
    y = cpu_b.to("musa") if n else torch.empty(0, dtype=torch.bfloat16, device="musa")
    out_a = torch.empty(1024, dtype=torch.bfloat16, device="musa")
    out_b = torch.empty_like(out_a)
    kernel[(1, 1, 1)](x, y, out_a, out_b, n)
    torch.musa.synchronize()
    valid = torch.arange(1024) < n
    if holes:
        valid &= (torch.arange(1024) & 128) != 0
    for actual, source_cpu, fill in [(out_a, cpu_a, 7.0)] + ([(out_b, cpu_b, -3.0)] if paired else []):
        expected = source_cpu[::2].clone()
        actual = actual.cpu()
        if other:
            expected[~valid] = fill
        else:
            actual, expected = actual[valid], expected[valid]
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
