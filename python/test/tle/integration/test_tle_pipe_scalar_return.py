"""Noinline pipe helpers preserve ordinary scalar return values."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _returning_sender(endpoint, value):
    endpoint.commit(0)
    return value + 1, value + 2


@triton.jit(noinline=True)
def _returning_receiver(endpoint, output):
    endpoint.wait(0)
    tl.store(output + 1, 7)


@triton.jit
def _scalar_return_kernel(output):
    edge = tle.pipe(capacity=1, one_shot=True)
    first, second = _returning_sender(edge.writer(), tl.program_id(0) + 5)
    tl.store(output, first)
    tl.store(output + 2, second)
    _returning_receiver(edge.reader(), output)


@pytest.mark.skip_if_tle_unsupported("pipe")
def test_pipe_helper_preserves_scalar_return():
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer CUDA GPU")
    output = torch.zeros(3, dtype=torch.int32, device="cuda")
    kernel = _scalar_return_kernel[(1, )](output)
    torch.testing.assert_close(output.cpu(), torch.tensor([6, 7, 7], dtype=torch.int32), rtol=0, atol=0)
    assert "_returning_sender" in kernel.asm["llir"]


@triton.jit(noinline=True)
def _nested_sender(endpoint, value):
    first, second = _returning_sender(endpoint, value)
    return first + second


@triton.jit
def _nested_return_kernel(output):
    first = tle.pipe(capacity=1, one_shot=True)
    second = tle.pipe(capacity=1, one_shot=True)
    result0 = _nested_sender(first.writer(), tl.program_id(0) + 3)
    _returning_receiver(first.reader(), output)
    result1 = _nested_sender(second.writer(), tl.program_id(0) + 5)
    _returning_receiver(second.reader(), output)
    tl.store(output, result0)
    tl.store(output + 2, result1)


@pytest.mark.skip_if_tle_unsupported("pipe")
def test_nested_pipe_helpers_share_multiple_result_abi():
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer CUDA GPU")
    output = torch.zeros(3, dtype=torch.int32, device="cuda")
    kernel = _nested_return_kernel[(1, )](output)
    torch.testing.assert_close(output.cpu(), torch.tensor([9, 7, 13], dtype=torch.int32), rtol=0, atol=0)
    assert "_nested_sender" in kernel.asm["llir"]
