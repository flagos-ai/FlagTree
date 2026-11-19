import triton.language.core as tl
from triton._C.libtriton import ir


def custom_op(builder: ir.builder, op_name: str, **kwargs):
    if op_name == "sync_block_all":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["mode"], kwargs["event_id"])

    elif op_name == "sync_block_set":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])

    elif op_name == "sync_block_wait":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])

    raise ValueError(f"Unsupported custom op: {op_name}")


def _is_int_like_elem(x) -> bool:
    """Accept int / tl.constexpr(int) / tl.tensor(int*)."""
    if isinstance(x, int):
        return True
    if isinstance(x, tl.constexpr):
        # constexpr value should be python int
        return isinstance(x.value, int)
    if isinstance(x, tl.tensor):
        # Offsets/strides must be integer typed (i32/i64 etc.)
        return x.dtype.is_int()
    return False


def _assert_int_like_tuple(name: str, xs):
    assert isinstance(xs, (list, tuple, tl.tuple)), f"{name} should be a tuple/list, but got {type(xs)}"
    assert all(_is_int_like_elem(x) for x in xs), f"{name} should be integer"


def _convert_elem_to_ir_value(builder, elem, require_i64):
    if isinstance(elem, int):
        elem = tl.constexpr(elem)
    if isinstance(elem, tl.constexpr):
        if require_i64:
            assert -2**63 <= elem.value < 2**63, f"Block pointers only support 64 bit `shape/strides`, " \
                f"got a value {elem.value} which is out of the range"
            return builder.get_int64(elem.value)
        else:
            assert -2**31 <= elem.value < 2**31, f"Block pointers only support 32 bit `offsets/block_shape`, " \
                f"got a value {elem.value} which is out of the range"
            return builder.get_int32(elem.value)
    elif isinstance(elem, tl.tensor):
        if require_i64:
            return builder.create_int_cast(elem.handle, builder.get_int64_ty(), elem.dtype.is_int_signed())
        else:
            return builder.create_int_cast(elem.handle, builder.get_int32_ty(), elem.dtype.is_int_signed())
    else:
        assert False, f"Unsupported element type in shape/strides/offsets: {type(elem)}"
