import sys

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton._C.libtriton import llvm


# TODO: temporary workaround, fix later.
# noinline=True emits a real device-side function call. On the CoreX 4.4.0
# runtime that call sequence faults with an illegal instruction (XID 22),
# which wedges the GPU and hangs any later launch, so keep this inlineable for now.
# @triton.jit(noinline=True)
@triton.jit
def add_one(x_ptr, SQRT: tl.constexpr) -> None:
    x = tl.load(x_ptr)
    if SQRT:
        x = libdevice.sqrt(x)
    tl.store(x_ptr, x + 1.0)


@triton.jit
def add_one_indirect(x_ptr, SQRT: tl.constexpr) -> None:
    add_one(x_ptr, SQRT)


@pytest.mark.parametrize("use_libdevice", (False, True))
@pytest.mark.parametrize("kernel", (add_one, add_one_indirect))
def test_link_extern_libs(use_libdevice, kernel):
    link_called: bool = False

    def callback(frame, event, arg):
        nonlocal link_called
        if event == "c_call" and arg is llvm.link_extern_libs:
            link_called = True

    x = torch.full((1, ), 4.0, device="cuda")
    prior_callback = sys.getprofile()
    try:
        sys.setprofile(callback)
        with (compilation := triton.knobs.compilation).scope():
            compilation.always_compile = True
            kernel[(1, )](x, SQRT=use_libdevice)
    finally:
        sys.setprofile(prior_callback)

    assert (link_called == use_libdevice)

    # sqrt(4) + 1 vs 4 + 1, so the value also tells whether libdevice was taken.
    # Reading it back synchronizes, turning a device-side fault into a failure
    # instead of a test that passes while the kernel never produced a result.
    expected = 3.0 if use_libdevice else 5.0
    torch.testing.assert_close(x, torch.full_like(x, expected))
