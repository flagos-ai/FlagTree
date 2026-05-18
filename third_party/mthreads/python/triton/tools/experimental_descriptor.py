import torch

from triton.tools.tensor_descriptor import TensorDescriptor


class _RawPointerTensor:

    def __init__(self, ptr: int, dtype: torch.dtype, device: str = "musa"):
        self._ptr = int(ptr)
        self.dtype = dtype
        self.device = torch.device(device)

    def data_ptr(self) -> int:
        return self._ptr

    def element_size(self) -> int:
        return int(self.dtype.itemsize)


def _dtype_from_element_size(element_size: int) -> torch.dtype:
    if element_size == 1:
        return torch.uint8
    if element_size == 2:
        return torch.int16
    if element_size == 4:
        return torch.int32
    raise ValueError(f"unsupported descriptor element_size={element_size}")


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _create_descriptor(ptr, shape, block_shape, element_size):
    dtype = _dtype_from_element_size(int(element_size))
    base = _RawPointerTensor(ptr, dtype=dtype, device="musa")
    return TensorDescriptor(base, list(shape), _contiguous_strides(shape), list(block_shape))


def create_1d_tma_descriptor(ptr, dim, block_dim, element_size):
    return _create_descriptor(ptr, (dim, ), (block_dim, ), element_size)


def create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    return _create_descriptor(ptr, (dim1, dim0), (block_dim1, block_dim0), element_size)


def create_3d_tma_descriptor(ptr, dim2, dim1, dim0, block_dim2, block_dim1, block_dim0, element_size):
    return _create_descriptor(ptr, (dim2, dim1, dim0), (block_dim2, block_dim1, block_dim0), element_size)
