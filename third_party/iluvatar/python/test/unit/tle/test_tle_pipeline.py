import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from utils import compile_iluvatar


@triton.jit
def _pipeline_sum_kernel(x_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tle.gpu.pipeline(0, n_blocks, 1, num_stages=NUM_STAGES):
        acc += tl.load(x_ptr + i * BLOCK + offs)
    tl.store(out_ptr + offs, acc)


@triton.jit
def _range_sum_kernel(x_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.range(0, n_blocks, 1):
        acc += tl.load(x_ptr + i * BLOCK + offs)
    tl.store(out_ptr + offs, acc)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_tle_pipeline_num_stages_codegen(num_stages):
    """tle.gpu.pipeline must be recognized as a loop iterator and propagate its
    num_stages hint onto the generated scf.for (tt.num_stages)."""
    compiled = compile_iluvatar(
        _pipeline_sum_kernel,
        signature={
            "x_ptr": "*fp32", "out_ptr": "*fp32", "n_blocks": "i32", "BLOCK": "constexpr", "NUM_STAGES": "constexpr"
        },
        constexprs={"BLOCK": 64, "NUM_STAGES": num_stages},
    )
    ttir = compiled.asm["ttir"]
    if num_stages > 1:
        # The pipeline hint must reach the loop; num_stages == 1 is a no-op hint.
        assert "tt.num_stages" in ttir, ttir


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_tle_pipeline_matches_range(device, num_stages):
    """tle.gpu.pipeline must be numerically equivalent to a plain tl.range loop."""
    block = 64
    n_blocks = 17
    torch.manual_seed(0)
    x = torch.randn(n_blocks * block, device=device, dtype=torch.float32)
    out_pipeline = torch.zeros(block, device=device, dtype=torch.float32)
    out_range = torch.zeros(block, device=device, dtype=torch.float32)

    grid = (1, )
    _pipeline_sum_kernel[grid](x, out_pipeline, n_blocks, BLOCK=block, NUM_STAGES=num_stages)
    _range_sum_kernel[grid](x, out_range, n_blocks, BLOCK=block)

    expected = x.reshape(n_blocks, block).sum(dim=0)
    torch.testing.assert_close(out_pipeline, out_range, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_pipeline, expected, atol=1e-4, rtol=1e-4)
