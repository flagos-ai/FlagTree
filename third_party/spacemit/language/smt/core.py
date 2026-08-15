# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT

from triton.language.core import (
    _unwrap_if_constexpr,
    builtin,
    constexpr,
    range,
)
from . import semantic as smt_semantic
from triton.language import core as tl


def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


@builtin
def compile_hint(ptr, hint_name, hint_val=None, _semantic=None):
    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    smt_semantic.compile_hint(ptr, hint_name, hint_val, _semantic)


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple tensor cores participate in the loop.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 bind_sub_block: bool = True):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block


@builtin
def descriptor_load(base, offsets, destination=None, _semantic=None):
    """Descriptor-based block load operation with two forms:

    Form 1: descriptor_load(base, offsets)
    Form 2: descriptor_load(base, offsets, destination)

    :param base: the base tensor pointer to load from
    :param offsets: offset values (e.g., [s * SUB_BLK_M, 0])
    :param destination: destination

    Examples
    *******
    .. code-block:: python

        # Form 1
        a = smt.descriptor_load(a_block_ptr, [s * SUB_BLK_M, 0])
        # Form 2
        a = smt.descriptor_load(a_block_ptr, [s * SUB_BLK_M, 0], destination)
    """

    if destination is None:
        return smt_semantic.descriptor_load(base, offsets, _semantic)

    return smt_semantic.descriptor_load_to_destination(base, offsets, destination, _semantic)


@builtin
def view(base, offsets, shape, packed_size=None, destination=None, _semantic=None):
    """Create a local view of a tensor with new shape and packed tile size.

    :param base: the base tensor to create view from
    :param offsets: offset values (e.g., [s * SUB_BLK_M, 0])
    :param shape: new shape for the view (e.g., [SUB_BLK_M, BLOCK_SIZE_K])
    :param packed_size: packed tile size for tensor cores (e.g., [16, 8])
    :param destination: optional destination tensor for in-place write (DPS)

    Example
    *******
    .. code-block:: python

        accumulator = tl.view(accumulator, [s * SUB_BLK_M, 0], [SUB_BLK_M, BLOCK_SIZE_K], [16, 8])
    """
    rank = len(shape)
    if packed_size is None:
        packed_size = [0] * rank
    return smt_semantic.view(base, offsets, shape, packed_size, destination, _semantic)


@builtin
def alloc(shape, type=tl.float32, storage="l2", _semantic=None):
    """Allocate a tensor in shared memory with specified shape and packed tile size.

    :param shape: shape of the tensor to allocate (e.g., [BLOCK_SIZE_N, BLOCK_SIZE_K])
    :param type: data type of the tensor elements (default: float32)

    Example
    *******
    .. code-block:: python

        b_packed_shared = tl.alloc([BLOCK_SIZE_N, BLOCK_SIZE_K])
    """

    return smt_semantic.alloc(shape, type, storage, _semantic)


@builtin
def alloc_copies(shape, dtype=tl.float32, storage="l2", copies=1, _semantic=None):

    return smt_semantic.alloc_copies(shape, dtype, copies, storage, _semantic)


@builtin
def dot(a_packed, b_packed, out_unpacked=None, _semantic=None):
    """
    Args:
        a_packed: packed A matrix, shape [MB, KB, mb, kb] (from MxK)
        b_packed: packed B matrix, shape [NB, KB, kb, nb] (from NxK)
    Returns:
        Matrix multiplication result, shape [MB, NB, mb, nb]

    """

    return smt_semantic.mmt4d(a_packed, b_packed, out_unpacked, _semantic)


@builtin
def mbarrier(flag=tl.constexpr(0), arrive_count=tl.constexpr(0), transaction_count=tl.constexpr(0),
             expect_count=tl.constexpr(1), _semantic=None):
    """Initialize a memory barrier for thread synchronization.

    :param flag: Barrier mode flag (0=normal, 1=async, 2=with fence)
    :param arrive_count: Initial arrival thread count
    :param transaction_count: Total threads to wait for
    :param expect_count: Expected version value after release

    Example:
        bar = smt.mbarrier(flag=0, arrive_count=0, transaction_count=0, expect_count=1)
    """
    return smt_semantic.mbarrier(flag, arrive_count, transaction_count, expect_count, _semantic)


@builtin
def barrier_arrive(bar, _semantic=None):
    """Signal thread arrival at barrier."""
    return smt_semantic.barrier_arrive(bar, _semantic)


@builtin
def barrier_wait(bar, flag=tl.constexpr(0), arrive_count=tl.constexpr(0), _semantic=None):
    """Wait for barrier to reach expected version."""
    return smt_semantic.barrier_wait(bar, flag, arrive_count, _semantic)


@builtin
def get_num_of_thread(_semantic=None):
    return smt_semantic.get_num_of_thread(_semantic)


@builtin
def global_mbarrier(id, _semantic=None):
    return smt_semantic.global_mbarrier(id, _semantic)


@builtin
def barrier_set_expect(bar, expect_count=tl.constexpr(1), _semantic=None):
    return smt_semantic.barrier_set_expect(bar, expect_count, _semantic)


@builtin
def mbarrier_copies(
        flag=tl.constexpr(0),
        arrive_count=tl.constexpr(0),
        transaction_count=tl.constexpr(0),
        expect_count=tl.constexpr(1),
        copies=1,
        _semantic=None,
):

    return smt_semantic.mbarrier_copies(flag, arrive_count, transaction_count, expect_count, copies, _semantic)
