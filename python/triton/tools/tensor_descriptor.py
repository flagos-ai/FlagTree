# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape


# FlagTree-only additive extension (no matching upstream Triton PR and no
# original block replaced): query the active backend for its TensorDescriptor
# stride-alignment contract, falling back to Triton's 16-byte rule when no
# runtime driver is active.
def _get_stride_alignment():
    # Import lazily to avoid importing runtime drivers while triton.tools is
    # being initialized.  A tools-only process may have no active driver; in
    # that case retain the existing conservative contract.
    from triton.runtime import driver

    try:
        alignment = driver.active.get_tensor_descriptor_stride_alignment()
    except RuntimeError as error:
        if not str(error).startswith("0 active drivers"):
            raise
        alignment = 16

    assert isinstance(alignment, int) and not isinstance(alignment, bool), \
        "tensor descriptor stride alignment must be an integer"
    assert alignment > 0 and (alignment & (alignment - 1)) == 0, \
        "tensor descriptor stride alignment must be a positive power of two"
    return alignment


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    padding: str = "zero"

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        ty = type(self.base)
        if ty.__name__ not in ("FakeTensor", "FunctionalTensor"):
            assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)

        # Original Triton validation (kept for review context):
        #
        # elem_bytes = self.base.dtype.itemsize
        # for stride in self.strides[:-1]:
        #     assert (stride * elem_bytes) % 16 == 0, \
        #         "strides must be 16-byte aligned"
        #
        # FlagTree/MThreads extension (no matching upstream Triton PR): replace
        # the fixed 16-byte check with the backend contract. Other backends keep
        # the 16-byte default; MThreads uses its hardware-supported 4-byte rule.
        stride_alignment = _get_stride_alignment()
        elem_bytes = self.base.dtype.itemsize
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % stride_alignment == 0, \
                f"strides must be {stride_alignment}-byte aligned"
        for shape_dim in self.shape:
            assert shape_dim > 0, "shape must be positive"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], padding="zero"):
        return TensorDescriptor(tensor, tensor.shape, tensor.stride(), block_shape, padding)
