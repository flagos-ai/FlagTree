"""Regression tests for the Iluvatar 4-row BF16 SME async-copy path."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.compiler import ASTSource, GPUTarget


ROWS = 4
COLS = 512


def _compile_4row_bf16(num_warps=8):
    return triton.compile(
        ASTSource(
            _copy_4row_bf16,
            signature={"src": "*bf16", "dst": "*bf16"},
            constexprs={},
        ),
        target=GPUTarget("corex", 71, 64),
        options={"num_warps": num_warps},
    )


@triton.jit
def _copy_4row_bf16(src, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    src_ptrs = src + rows * 512 + cols
    src_ptrs = tle.gpu.set_layout(
        src_ptrs,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512],
            tl.bfloat16,
            [1, 4],
            [2, 32],
            [2, 4],
            [1, 0],
        ),
    )

    storage = tle.gpu.alloc(
        [4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [4, 512], tl.bfloat16, [1, 0], [1, 1], [1, 1], [1, 0]),
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(src_ptrs, storage, [4, 512], is_async=True, input_stride=512)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)

    values = tl.load(tle.gpu.local_ptr(storage))
    tl.store(dst + rows * 512 + cols, values)


def test_tle_sme_4row_bf16_codegen_without_gpu():
    compiled = _compile_4row_bf16()
    assert "isSme = true" in compiled.asm["ttgir"]
    assert "inputStride" in compiled.asm["ttgir"]
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" in compiled.asm["llir"]


@pytest.mark.parametrize("num_warps", [8])
def test_tle_sme_4row_bf16_async_copy(num_warps):
    if not torch.cuda.is_available():
        pytest.skip("requires an Iluvatar GPU")

    torch.manual_seed(17)
    source = torch.randn((ROWS, COLS), device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(source)
    compiled = _copy_4row_bf16[(1,)](source, result, num_warps=num_warps)

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "isSme = true" in ttgir
    assert "inputStride" in ttgir
    assert "llvm.bi.sme.load.4x1b64.rowxfb16" in llir

    for _ in range(5):
        source.normal_()
        _copy_4row_bf16[(1,)](source, result, num_warps=num_warps)
        torch.testing.assert_close(result, source, rtol=0, atol=0)


@triton.jit
def _copy_4row_bf16_invalid_stride(src, dst):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 512)[None, :]
    src_ptrs = tle.gpu.set_layout(
        src + rows * 500 + cols,
        tle.gpu.IluvatarSmeBlockEncoding(
            [4, 512], tl.bfloat16, [1, 4], [2, 32], [2, 4], [1, 0]
        ),
    )
    storage = tle.gpu.alloc(
        [4, 512],
        dtype=tl.bfloat16,
        layout=tle.gpu.iluvatar_sme_shared_layout(
            [4, 512], tl.bfloat16, [1, 0], [1, 1], [1, 1], [1, 0]
        ),
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(src_ptrs, storage, [4, 512], is_async=True, input_stride=500)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)


def test_tle_sme_rejects_invalid_row_stride():
    with pytest.raises(Exception, match="64-byte.*stride|SME.*stride"):
        triton.compile(
            ASTSource(
                _copy_4row_bf16_invalid_stride,
                signature={"src": "*bf16", "dst": "*bf16"},
                constexprs={},
            ),
            target=GPUTarget("corex", 71, 64),
            options={"num_warps": 8},
        )
