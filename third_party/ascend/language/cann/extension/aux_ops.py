import triton.language as tl
from triton.language import core
from triton.language.core import (_unwrap_if_constexpr, _tensor_member_fn, _unwrap_iterable, builtin, constexpr, dtype,
                                  tensor, check_bit_width, _unwrap_if_constexpr, range)

from typing import Optional, Tuple, List, overload, Union
from triton._C.libtriton import ir
from ._utils import custom_op


@_tensor_member_fn
@builtin
def sync_block_all(mode, event_id, _semantic=None):
    import warnings

    warnings.warn(
        ("This method would be deprecated. Use al.sync_block_all instead."),
        DeprecationWarning,
        stacklevel=1,
    )
    mode = _unwrap_if_constexpr(mode)
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode == "all_cube" or mode == "all_vector" or mode == "all", f"ERROR: mode = {mode}, only supports all_cube/all_vector/all"
    custom_op(_semantic.builder, "sync_block_all", mode=mode, event_id=event_id)


@_tensor_member_fn
@builtin
def sync_block_set(sender, receiver, event_id, _semantic=None):
    import warnings

    warnings.warn(
        ("This method would be deprecated. Use al.sync_block_set instead."),
        DeprecationWarning,
        stacklevel=1,
    )
    sender = _unwrap_if_constexpr(sender)
    receiver = _unwrap_if_constexpr(receiver)
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(sender, str) and (sender == "cube"
                                        or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver
                                          == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    custom_op(_semantic.builder, "sync_block_set", sender=sender, event_id=event_id)


@_tensor_member_fn
@builtin
def sync_block_wait(sender, receiver, event_id, _semantic=None):
    import warnings

    warnings.warn(
        ("This method would be deprecated. Use al.sync_block_wait instead."),
        DeprecationWarning,
        stacklevel=1,
    )
    sender = _unwrap_if_constexpr(sender)
    receiver = _unwrap_if_constexpr(receiver)
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(sender, str) and (sender == "cube"
                                        or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver
                                          == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    custom_op(_semantic.builder, "sync_block_wait", sender=sender, event_id=event_id)


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 bind_sub_block: bool = False):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block


def compile_hint_impl(ptr: tensor, hint_name: str, hint_val, builder: ir.builder):
    # simt mode does not support hint annotations
    # FIXME: is_simt_mode
    # if builder.is_simt_mode():
    #     return
    # Check isinstance(hint_val, bool) first to handle False explicitly
    if isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, core.constexpr):
        hint_val = builder.get_string_attr(hint_val.value)
    elif isinstance(hint_val, (list, core.tuple)):
        # only support i64 array attr for now
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation_mark(ptr.handle, hint_name, hint_val)


@builtin
def compile_hint(ptr, hint_name, hint_val=None, _semantic=None):
    # simt mode does not support hint annotations
    if _semantic.builder.is_simt_mode():
        return

    def _unwrap(val):
        return _unwrap_if_constexpr(val) if val else val

    hint_name = _unwrap_if_constexpr(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    if isinstance(hint_val, (list, core.tuple)):
        hint_val = [_unwrap(val) for val in hint_val]
    else:
        hint_val = _unwrap(hint_val)
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    compile_hint_impl(ptr, hint_name, hint_val, _semantic.builder)


@builtin
def multibuffer(src: tensor, size, _semantic=None):
    """
    Set multi_buffer for an existing tensor
    :src: tensor set to bufferize multiple time
    :size: number of copies
    """
    buffer_size = _unwrap_if_constexpr(size)
    assert isinstance(buffer_size, int) and buffer_size == 2, f"only support bufferize equals 2"
    compile_hint_impl(src, "hivm.multi_buffer", buffer_size, _semantic.builder)
