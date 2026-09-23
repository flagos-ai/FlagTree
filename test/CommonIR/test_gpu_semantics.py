"""Full-buffer GPU copy contract and frontend diagnostics for CommonIR."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

pytestmark = pytest.mark.skipif(not ENABLED, reason="requires FLAGTREE_COMMON_IR build")


@triton.jit
def _roundtrip(src, dst, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([BLOCK], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tle.gpu.copy(src + idx, buf, [BLOCK])
    value = buf.load(writable=False)
    buf.store(value + 1)
    tle.gpu.copy(buf, dst + idx, [BLOCK])


@pytest.mark.parametrize("block", [16, 128, 1024])
def test_copy_roundtrip(block):
    src = torch.arange(block, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    kernel = _roundtrip[(1, )](src, dst, block)
    torch.testing.assert_close(dst, src + 1, rtol=0, atol=0)
    assert "tile." not in kernel.asm["ttgir"]
    assert "unrealized_conversion_cast" not in kernel.asm["ttgir"]


@triton.jit
def _invalid(src, CASE: tl.constexpr):
    idx = tl.arange(0, 16)
    buf = tle.gpu.alloc([16], tl.float32, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    if CASE == 0:
        tle.gpu.copy(src + idx, buf, [8])
    elif CASE == 1:
        tle.gpu.copy(src + tl.arange(0, 8), buf, [16])
    elif CASE == 2:
        tle.gpu.copy(src + idx, buf, [16], offsets=[1])
    elif CASE == 3:
        buf.load(target_shape=[8])
    elif CASE == 4:
        buf.store(tl.full([8], 0, tl.float32))
    elif CASE == 5:
        buf.store(tl.full([16], 0, tl.int32))
    elif CASE == 6:
        tle.gpu.copy((src + idx).to(tl.pointer_type(tl.int32)), buf, [16])
    elif CASE == 7:
        tle.gpu.alloc([32, 32], tl.float32, scope=tle.gpu.tmem)
    else:
        barrier = tle.gpu.alloc_barrier(expect_bytes=64)
        tle.gpu.copy(src + idx, buf, [16], barrier=barrier)


@pytest.mark.parametrize("case, message", [
    (0, "shape and pointer tensor shape"),
    (1, "shape and pointer tensor shape"),
    (2, "normal copy does not support offsets"),
    (3, "buffered_tensor.load requires the full buffer shape"),
    (4, "buffered_tensor.store requires the full buffer shape and element type"),
    (5, "buffered_tensor.store requires the full buffer shape and element type"),
    (6, "copy requires pointers with the buffer element type"),
    (7, "currently supports only shared-memory buffers"),
    (8, "barrier is only supported for global-to-shared TMA copy"),
])
def test_invalid_buffer_semantics(case, message):
    src = torch.empty(16, dtype=torch.float32, device="cuda")
    with pytest.raises(triton.CompilationError, match=message):
        _invalid[(1, )](src, case)
