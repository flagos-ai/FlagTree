import pytest
import torch

from triton.runtime import driver
from triton.tools.tensor_descriptor import TensorDescriptor


class _FakeDriver:

    def __init__(self, stride_alignment):
        self.stride_alignment = stride_alignment

    def get_tensor_descriptor_stride_alignment(self):
        return self.stride_alignment


def _set_stride_alignment(monkeypatch, stride_alignment):
    monkeypatch.setattr(driver, "_active", _FakeDriver(stride_alignment))


def test_tensor_descriptor_default_backend_requires_sixteen_byte_strides(monkeypatch):
    _set_stride_alignment(monkeypatch, 16)
    tensor = torch.empty((2, 538), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="strides must be 16-byte aligned"):
        TensorDescriptor.from_tensor(tensor, [1, 2])


def test_tensor_descriptor_uses_backend_stride_alignment(monkeypatch):
    _set_stride_alignment(monkeypatch, 4)
    tensor = torch.empty((2, 538), dtype=torch.bfloat16)

    descriptor = TensorDescriptor.from_tensor(tensor, [1, 2])

    assert tuple(descriptor.strides) == tensor.stride()


def test_tensor_descriptor_rejects_stride_below_backend_alignment(monkeypatch):
    _set_stride_alignment(monkeypatch, 4)
    tensor = torch.empty((2, 537), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="strides must be 4-byte aligned"):
        TensorDescriptor.from_tensor(tensor, [1, 2])


def test_tensor_descriptor_stride_alignment_is_not_caller_configurable(monkeypatch):
    _set_stride_alignment(monkeypatch, 16)
    tensor = torch.empty((2, 16), dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="stride_alignment"):
        TensorDescriptor.from_tensor(tensor, [1, 2], stride_alignment=4)


def test_interpreter_uses_backend_stride_alignment(monkeypatch):
    from triton.runtime.interpreter import GridExecutor

    _set_stride_alignment(monkeypatch, 4)
    tensor = torch.empty((2, 538), dtype=torch.bfloat16)
    descriptor = TensorDescriptor.from_tensor(tensor, [1, 2])
    executor = GridExecutor(lambda: None, [], (1, ))

    (host_descriptor, ), _ = executor._init_args_hst((descriptor, ), {})

    assert tuple(host_descriptor.strides) == tensor.stride()
