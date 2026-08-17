# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT

from triton.language import core as tl
from triton.language import semantic as tl_semantic
from . import types as smt

from typing import TypeVar

T = TypeVar('T')
TensorTy = TypeVar('TensorTy')


def _ceil_div(lhs, rhs):
    return (lhs + rhs - 1) // rhs


def compile_hint(ptr: tl.tensor, hint_name: str, hint_val, _semantic=None):
    if not hint_val:
        hint_val = _semantic.builder.get_unit_attr()
    elif isinstance(hint_val, bool):
        hint_val = _semantic.builder.get_bool_attr(hint_val)
    elif isinstance(hint_val, int):
        hint_val = _semantic.builder.get_int32_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    _semantic.builder.create_annotation(ptr.handle, hint_name, hint_val)


def descriptor_load(base: tl.tensor, offsets, _semantic=None) -> tl.tensor:

    semantic_instance = tl_semantic.TritonSemantic(_semantic.builder)
    offsets = semantic_instance._convert_to_ir_values(offsets, require_i64=False)
    handle = _semantic.builder.create_descriptor_load(base.handle, offsets)
    dst_ty = base.type.element_ty
    return tl.tensor(handle, dst_ty)


def descriptor_load_to_destination(base: tl.tensor, offsets, destination, _semantic=None) -> tl.tensor:
    semantic_instance = tl_semantic.TritonSemantic(_semantic.builder)
    offsets = semantic_instance._convert_to_ir_values(offsets, require_i64=False)

    _semantic.builder.create_descriptor_load_to_destination(base.handle, offsets, destination.handle)


def view(base: tl.tensor, offsets, shape, packed_size, destination=None, _semantic=None) -> tl.tensor:
    semantic_instance = tl_semantic.TritonSemantic(_semantic.builder)
    dest_handle = destination.handle if destination is not None else None
    # Save raw offsets before IR conversion for no-op detection
    raw_offsets = [o.value if isinstance(o, tl.constexpr) else o for o in offsets]
    offsets = semantic_instance._convert_to_ir_values(offsets, require_i64=False)

    shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in shape]
    packed_size = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in packed_size]

    assert len(shape) == len(packed_size), \
        "Shape and packed_size must have the same length"

    pointee_type = base.type.element_ty

    is_block_ptr = False
    if hasattr(pointee_type, 'is_block') and pointee_type.is_block():
        is_block_ptr = True

    if is_block_ptr:
        element_ty = pointee_type.element_ty
        handle = _semantic.builder.create_viewptr(base.handle, offsets, shape, packed_size)

    else:
        element_ty = pointee_type
        # Dispatch to pack, unpack, or repack
        if all(s == 1 for s in packed_size):
            # packed_size=[1,1] → unpack (4D→2D)
            handle = _semantic.builder.create_unpack(base.handle, offsets, shape, destination=dest_handle)
        else:
            base_shape = base.type.shape if hasattr(base.type, 'shape') else []
            base_is_4d = len(base_shape) == 4
            if base_is_4d and not all(s == 0 for s in packed_size):
                base_packed = [base_shape[2], base_shape[3]]
                if list(packed_size) != base_packed:
                    # 4D input with different packed → repack
                    handle = _semantic.builder.create_repack(base.handle, offsets, shape, packed_size,
                                                             destination=dest_handle)
                else:
                    base_logical = [base_shape[0] * base_packed[0], base_shape[1] * base_packed[1]]
                    is_noop = (list(shape) == base_logical and all(isinstance(o, int) and o == 0 for o in raw_offsets))
                    if is_noop:
                        handle = base.handle
                    else:
                        handle = _semantic.builder.create_repack(base.handle, offsets, shape, packed_size,
                                                                 destination=dest_handle)
            else:
                handle = _semantic.builder.create_pack(base.handle, offsets, shape, packed_size,
                                                       destination=dest_handle)

    # Compute result type (same logic as before, unchanged)
    if all(s == 0 for s in packed_size):
        if is_block_ptr:
            base_tensor = tl.tensor(handle, base.type.element_ty)
        else:
            base_tensor = tl.tensor(handle, base.type)

        actualPackedSize = [base_tensor.shape[2], base_tensor.shape[3]]
        result_shape = [
            _ceil_div(shape[0], actualPackedSize[0]),
            _ceil_div(shape[1], actualPackedSize[1]), actualPackedSize[0], actualPackedSize[1]
        ]
        if is_block_ptr:
            result_tensor = tl.tensor(handle, tl.pointer_type(tl.block_type(element_ty, result_shape)))
        else:
            result_tensor = tl.tensor(handle, tl.block_type(element_ty, result_shape))
    elif all(s == 1 for s in packed_size):
        if is_block_ptr:
            result_tensor = tl.tensor(handle, tl.pointer_type(tl.block_type(element_ty, shape)))
        else:
            result_tensor = tl.tensor(handle, tl.block_type(element_ty, shape))
    else:
        result_shape = [
            _ceil_div(shape[0], packed_size[0]),
            _ceil_div(shape[1], packed_size[1]), packed_size[0], packed_size[1]
        ]
        if is_block_ptr:
            result_tensor = tl.tensor(handle, tl.pointer_type(tl.block_type(element_ty, result_shape)))
        else:
            result_tensor = tl.tensor(handle, tl.block_type(element_ty, result_shape))
    return result_tensor


def alloc(shape, dtype, storage: str, _semantic=None):

    shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in shape]

    ptr_type = tl.pointer_type(tl.block_type(dtype, shape))
    dtype_ir = ptr_type.to_ir(_semantic.builder)
    storage = storage.value if hasattr(storage, 'value') else storage

    handle = _semantic.builder.create_alloc(shape, dtype_ir, storage)

    return tl.tensor(handle, ptr_type)


def alloc_copies(shape, dtype, copies, storage: str, _semantic=None):
    unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
    unwrapped_num = tl._unwrap_if_constexpr(copies)
    storage = storage.value if hasattr(storage, 'value') else storage

    num_val = unwrapped_num.value if isinstance(unwrapped_num, tl.constexpr) else unwrapped_num
    base_shape = [d.value if isinstance(d, tl.constexpr) else d for d in unwrapped_shape]

    full_shape = [num_val] + base_shape

    element_ty_ir = dtype.to_ir(_semantic.builder)

    handle = _semantic.builder.create_alloc_copies(full_shape, element_ty_ir, storage)
    return smt.buffered_tensor(handle, dtype, full_shape, num_val, storage, _semantic)


def mmt4d(a_packed: tl.tensor, b_packed: tl.tensor, out_unpacked: tl.tensor, _semantic=None):
    assert len(a_packed.shape) == 4, f"A must be 4D packed, got {a_packed.shape}D"
    assert len(b_packed.shape) == 4, f"B must be 4D packed, got {b_packed.shape}D"

    if a_packed.shape[1] == b_packed.shape[0] and a_packed.shape[3] == b_packed.shape[2]:
        mb = a_packed.shape[0]
        nb = b_packed.shape[1]
        mb_packed_sizes = a_packed.shape[2]
        nb_packed_sizes = b_packed.shape[3]
    elif a_packed.shape[1] == b_packed.shape[1] and a_packed.shape[3] == b_packed.shape[3]:
        mb = a_packed.shape[0]
        nb = b_packed.shape[0]
        mb_packed_sizes = a_packed.shape[2]
        nb_packed_sizes = b_packed.shape[2]
    else:
        raise ValueError(f"Unsupported packing shapes A{a_packed.shape} B{b_packed.shape}")
    output_shape = [mb, nb, mb_packed_sizes, nb_packed_sizes]
    ret_type = tl.block_type(a_packed.type.scalar, output_shape)
    if out_unpacked is None:
        out = _semantic.builder.create_mmt4d(a_packed.handle, b_packed.handle, None)
    else:
        out = _semantic.builder.create_mmt4d(a_packed.handle, b_packed.handle, out_unpacked.handle)
    return tl.tensor(out, ret_type)


def mbarrier(flag, arrive_count, transaction_count, expect_count, _semantic=None) -> tl.tensor:
    from triton.language.core import _unwrap_if_constexpr

    flag = _unwrap_if_constexpr(flag)
    arrive_count = _unwrap_if_constexpr(arrive_count)
    transaction_count = _unwrap_if_constexpr(transaction_count)
    expect_count = _unwrap_if_constexpr(expect_count)

    semantic_inst = tl_semantic.TritonSemantic(_semantic.builder)
    flag_val = semantic_inst.to_tensor(flag)
    arrive_count_val = semantic_inst.to_tensor(arrive_count)
    transaction_count_val = semantic_inst.to_tensor(transaction_count)
    exp_val = semantic_inst.to_tensor(expect_count)

    flag_val = semantic_inst.cast(flag_val, tl.int16)
    arrive_count_val = semantic_inst.cast(arrive_count_val, tl.int16)
    transaction_count_val = semantic_inst.cast(transaction_count_val, tl.int16)
    exp_val = semantic_inst.cast(exp_val, tl.int16)

    bar_handle = _semantic.builder.create_mbarrier(flag_val.handle, arrive_count_val.handle,
                                                   transaction_count_val.handle, exp_val.handle)
    return tl.tensor(bar_handle, tl.int64)


def mbarrier_copies(
        flag=tl.constexpr(0),
        arrive_count=tl.constexpr(0),
        transaction_count=tl.constexpr(0),
        expect_count=tl.constexpr(1),
        copies=1,
        _semantic=None,
):
    """
    Allocate multiple mbarrier copies.

    Args:
        flag: barrier flag (0 or 1)
        arrive_count: initial arrive count
        transaction_count: transaction byte count for async operations
        expect_count: expected arrive count before barrier can proceed
        copies: number of barrier copies
        _semantic: triton semantic context

    Returns:
        mbarrier: handle to the allocated barriers

    Example:
        bar = smt.mbarrier_copies(copies=2, transaction_count=1024)
        bar0 = bar[0]  # I64 handle
        bar1 = bar[1]  # I64 handle
    """

    def unwrap(val):
        v = tl._unwrap_if_constexpr(val)
        return v.value if isinstance(v, tl.constexpr) else v

    flag_val = unwrap(flag)
    arrive_val = unwrap(arrive_count)
    txn_val = unwrap(transaction_count)
    expect_val = unwrap(expect_count)
    num_copies = unwrap(copies)

    handle = _semantic.builder.create_mbarrier_copies(num_copies, flag_val, arrive_val, txn_val, expect_val)

    return smt.mbarrier(handle, num_copies, flag_val, arrive_val, txn_val, expect_val, _semantic)


def barrier_arrive(bar: tl.tensor, _semantic=None):
    semantic_inst = tl_semantic.TritonSemantic(_semantic.builder)
    if not isinstance(bar, tl.tensor):
        bar = semantic_inst.to_tensor(bar)
    bar = semantic_inst.cast(bar, tl.int64)
    _semantic.builder.create_barrier_arrive(bar.handle)


def barrier_wait(bar: tl.tensor, flag, arrive_count, _semantic=None):
    from triton.language.core import _unwrap_if_constexpr

    semantic_inst = tl_semantic.TritonSemantic(_semantic.builder)
    if not isinstance(bar, tl.tensor):
        bar = semantic_inst.to_tensor(bar)
    bar = semantic_inst.cast(bar, tl.int64)

    flag = _unwrap_if_constexpr(flag)
    arrive_count = _unwrap_if_constexpr(arrive_count)

    flag_tensor = semantic_inst.to_tensor(flag)
    arr_tensor = semantic_inst.to_tensor(arrive_count)

    flag_tensor = semantic_inst.cast(flag_tensor, tl.int16)
    arr_tensor = semantic_inst.cast(arr_tensor, tl.int16)

    _semantic.builder.create_barrier_wait(bar.handle, flag_tensor.handle, arr_tensor.handle)


def get_num_of_thread(_semantic=None):
    _semantic.builder.create_get_num_of_thread()


def global_mbarrier(id, _semantic=None) -> tl.tensor:
    raise NotImplementedError("global_mbarrier Not supported: The syntax is not currently implemented in the lowering.")
    from triton.language.core import _unwrap_if_constexpr

    id = _unwrap_if_constexpr(id)
    semantic_inst = tl_semantic.TritonSemantic(_semantic.builder)
    id_val = semantic_inst.to_tensor(id)
    id_val = semantic_inst.cast(id_val, tl.int16)

    bar_handle = _semantic.builder.create_global_mbarrier(id_val.handle)
    return tl.tensor(bar_handle, tl.int64)


def barrier_set_expect(bar: tl.tensor, expect_count, _semantic=None):
    from triton.language.core import _unwrap_if_constexpr
    expect_count = _unwrap_if_constexpr(expect_count)

    semantic_inst = tl_semantic.TritonSemantic(_semantic.builder)
    exp_tensor = semantic_inst.to_tensor(expect_count)
    exp_tensor = semantic_inst.cast(exp_tensor, tl.int16)

    _semantic.builder.create_barrier_set_expect(bar.handle, exp_tensor.handle)
