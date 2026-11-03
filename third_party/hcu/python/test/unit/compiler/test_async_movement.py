"""Runtime coverage for HCU shared-memory movement and buffer views."""
import pytest
import torch

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _move(A, C, MODE: tl.constexpr, VIEW: tl.constexpr, TAIL: tl.constexpr):
    r = tl.broadcast_to(tl.arange(0, 16)[:, None], (16, 64))
    c = tl.broadcast_to(tl.arange(0, 64)[None, :], (16, 64))
    if VIEW == 1:
        storage = tle.gpu.alloc((2, 16, 64), dtype=A.dtype.element_ty,
                                nv_mma_shared_layout=False)
    elif VIEW == 2:
        storage = tle.gpu.alloc((16, 128), dtype=A.dtype.element_ty,
                                nv_mma_shared_layout=False)
    else:
        storage = tle.gpu.alloc((16, 64), dtype=A.dtype.element_ty,
                                nv_mma_shared_layout=False)
    for i in range(3):
        if VIEW == 1:
            buf = storage.slot(i % 2)
        elif VIEW == 2:
            buf = storage.subslice(64, 64, dim=1)
        else:
            buf = storage
        if MODE < 2:
            value = tle.load(A + r * 64 + c, c < TAIL, 0, is_async=MODE == 1)
            buf.store(value)
        else:
            tle.gpu.copy(A + r * 64 + c, buf, (16, 64), mask=c < TAIL)
        if MODE == 3:
            tle.gpu.copy(buf, C + r * 64 + c, (16, 64), mask=c < TAIL)
        else:
            tl.store(C + r * 64 + c, buf.load() + i, c < TAIL)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize("view", [0, 1, 2])
@pytest.mark.parametrize("tail", [63, 64])
def test_shared_movement(dtype, mode, view, tail):
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("gfx936 direct-to-LDS regression")
    a = torch.randn((16, 64), dtype=dtype, device="cuda")
    c = torch.full_like(a, -99)
    _move[(1,)](a, c, mode, view, tail, num_warps=4, num_stages=1)
    expected = torch.full_like(a, -99)
    expected[:, :tail] = a[:, :tail] + (0 if mode == 3 else 2)
    torch.testing.assert_close(c, expected)
