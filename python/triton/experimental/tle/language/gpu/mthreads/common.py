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

import os

import triton.language.core as tl

try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ModuleNotFoundError:
    FLAGTREE_BACKEND = os.environ.get("FLAGTREE_BACKEND", "")


def _has_mthreads_libtriton() -> bool:
    try:
        from triton._C import libtriton
    except ImportError:
        return False
    return hasattr(libtriton, "mthreads")


def enabled() -> bool:
    return FLAGTREE_BACKEND == "mthreads" or _has_mthreads_libtriton()


@tl.builtin
def local_barrier(_semantic=None) -> None:
    """Synchronize a CTA and make shared-memory accesses visible."""
    builder = _semantic.builder
    if not hasattr(builder, "create_mthreads_local_barrier"):
        raise RuntimeError("MThreads TLE local barrier requires a newer native libtriton")
    builder.create_mthreads_local_barrier()


@tl.builtin
def device_tensor_descriptor(address, shape, strides, block_shape, dtype, _semantic=None):
    """Import a host-encoded MUSA TME descriptor from device memory.

    ``address`` is a scalar global byte pointer to a 64-byte-aligned, immutable
    64-byte ``MUtensorDescriptor``. Its encoded dtype/shape/strides must match
    the supplied logical metadata. Keep both descriptor and payload alive
    until every consuming launch completes. Publish descriptor updates before
    launch with normal stream ordering; this operation supplies no memory fence.

    Unlike descriptor arguments passed by value, the launch ABI uses an
    ordinary device pointer. The caller owns descriptor allocation and reuse.
    """
    address = _semantic.to_tensor(address)
    if (
        not isinstance(address.type, tl.pointer_type)
        or address.type.address_space != 1
        or address.type.element_ty not in (tl.int8, tl.uint8)
    ):
        raise ValueError("device tensor descriptor requires a scalar global byte pointer")
    shape, strides = list(shape), list(strides)
    block_shape = tl._unwrap_shape(block_shape)
    dtype = tl._unwrap_if_constexpr(dtype)
    if not 1 <= len(shape) <= 5 or len(shape) != len(strides) or len(shape) != len(block_shape):
        raise ValueError("device tensor descriptor shape/strides/block rank must agree in 1..5")
    if any(dim <= 0 for dim in block_shape):
        raise ValueError("device tensor descriptor block dimensions must be positive")
    if not isinstance(dtype, tl.dtype):
        raise ValueError("device tensor descriptor dtype must be a Triton dtype")
    builder = _semantic.builder
    if not hasattr(builder, "create_mthreads_device_tensor_descriptor"):
        raise RuntimeError("device tensor descriptors require a newer native FlagTree binding")
    block_type = tl.block_type(dtype, block_shape)
    handle = builder.create_mthreads_device_tensor_descriptor(
        address.handle, block_type.to_ir(builder), dtype.is_int_signed()
    )
    shape = [_semantic.make_scalar(x, tl.int32) for x in shape]
    strides = [_semantic.make_scalar(tl._unwrap_if_constexpr(x), tl.int64) for x in strides]
    return tl.tensor_descriptor(handle, shape, strides, block_type)
