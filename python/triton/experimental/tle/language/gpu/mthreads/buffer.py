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

from math import prod

from triton._utils import TRITON_MAX_TENSOR_NUMEL, is_power_of_two

from .. import types as gpu_types


def is_backend_builder(builder) -> bool:
    return hasattr(builder, "mark_musa_tle_auto_shared_layout")


def needs_non_power_of_two_leading_dim(builder, shape) -> bool:
    if not is_backend_builder(builder) or not shape:
        return False
    return isinstance(shape[0], int) and shape[0] > 0 and not is_power_of_two(shape[0])


def validate_shape(shape) -> tuple[int, ...]:
    shape = tuple(shape)
    if len(shape) < 2:
        raise ValueError("mthreads TLE non-power-of-two buffered tensor requires rank >= 2")
    for index, dim in enumerate(shape):
        if not isinstance(dim, int):
            raise TypeError(f"Shape element {index} must have type `constexpr[int]`, got `constexpr[{type(dim)}]")
        if dim <= 0:
            raise ValueError(f"Shape element {index} must be positive")
        if index > 0 and not is_power_of_two(dim):
            raise ValueError(f"Shape element {index} must be a power of 2")
    numel = prod(shape)
    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return shape


class buffered_tensor_type(gpu_types.buffered_tensor_type):

    def __init__(self, element_ty, shape, storage, layout=None, semantic=None, alloc_shape=None):
        shape = validate_shape(shape)
        self.element_ty = element_ty
        self.shape = shape
        self.numel = prod(shape)
        self.name = f"<{self.shape}, {self.element_ty}>"
        self.storage = storage
        self.layout = layout
        self.alloc_shape = list(shape if alloc_shape is None else alloc_shape)
        assert semantic, "buffered_tensor array must be created with a builder"
        self.semantic = semantic

    def _unflatten_ir(self, handles, cursor):
        value = buffered_tensor(
            handles[cursor],
            self.scalar,
            self.shape,
            self.storage,
            self.layout,
            self.semantic,
            alloc_shape=self.alloc_shape,
        )
        if hasattr(self, "_tle_remote_shard_id"):
            shard_id = getattr(self, "_tle_remote_shard_id")
            scope = getattr(self, "_tle_remote_scope", None)
            setattr(value, "_tle_remote_shard_id", shard_id)
            setattr(value, "_tle_remote_scope", scope)
            setattr(value.type, "_tle_remote_shard_id", shard_id)
            setattr(value.type, "_tle_remote_scope", scope)
        return value, cursor + 1

    def with_element_ty(self, scalar_ty):
        return buffered_tensor_type(
            scalar_ty,
            self.shape,
            self.storage,
            self.layout,
            self.semantic,
            alloc_shape=self.alloc_shape,
        )


class buffered_tensor(gpu_types.buffered_tensor):

    def __init__(self, handle, element_ty, shape, storage, layout=None, semantic=None, alloc_shape=None):
        self.handle = handle
        self.shape = list(shape)
        self.type = buffered_tensor_type(
            element_ty,
            shape,
            storage,
            layout,
            semantic,
            alloc_shape=alloc_shape,
        )
        self.dtype = element_ty


def create_buffered_tensor(handle, element_ty, shape, storage, layout, semantic, alloc_shape=None):
    return buffered_tensor(
        handle,
        element_ty,
        shape,
        storage,
        layout,
        semantic,
        alloc_shape=alloc_shape,
    )
