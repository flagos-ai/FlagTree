# flagtree tle
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _rematerialize_index_kernel(output_i32, output_i64, BLOCK: tl.constexpr):
    program_id = tle.gpu.rematerialize_index(tl.program_id(0).to(tl.int64))
    source_i32 = tl.arange(0, BLOCK)
    source_i64 = source_i32.to(tl.int64)
    first_i32 = tle.gpu.rematerialize_index(source_i32)
    second_i32 = tle.gpu.rematerialize_index(source_i32)
    first_i64 = tle.gpu.rematerialize_index(source_i64)
    second_i64 = tle.gpu.rematerialize_index(source_i64)
    tl.store(output_i32 + source_i32, first_i32 + second_i32)
    tl.store(output_i64 + source_i32, first_i64 + second_i64 + program_id)


def _require_cuda():
    try:
        torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


def test_rematerialize_index_preserves_value_and_program_points():
    _require_cuda()
    output_i32 = torch.empty(64, device="cuda", dtype=torch.int32)
    output_i64 = torch.empty(64, device="cuda", dtype=torch.int64)
    compiled = _rematerialize_index_kernel.warmup(
        output_i32,
        output_i64,
        BLOCK=64,
        grid=(1,),
        num_warps=4,
    )

    assert compiled.asm["ttir"].count("tt.elementwise_inline_asm") == 5
    assert compiled.asm["ttir"].count("tle.rematerialize_index") == 5
    _rematerialize_index_kernel[(1,)](
        output_i32,
        output_i64,
        BLOCK=64,
        num_warps=4,
    )
    expected = 2 * torch.arange(64, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(output_i32, expected, atol=0, rtol=0)
    torch.testing.assert_close(output_i64, expected.to(torch.int64), atol=0, rtol=0)
