import triton.language as tl
from triton.language import core
from triton.language.core import (_unwrap_if_constexpr, _unwrap_iterable, builtin, constexpr, dtype, tensor,
                                  check_bit_width, _unwrap_if_constexpr, range)

from typing import Optional, Tuple, List, overload, Union
from triton._C.libtriton import ir


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's ``range`` in the context of
    ``triton.jit`` functions. In addition, it allows the user to pass extra attributes to the compiler.

    :param arg1: If ``arg2`` is not given, the end of the range; otherwise the start of the range.
    :type arg1: int
    :param arg2: The end of the range, if given.
    :type arg2: int
    :param step: The step size of the iteration. Defaults to 1.
    :type step: int
    :param num_stages: Number of pipeline stages for the loop.
    :type num_stages: int
    :param loop_unroll_factor: The unroll factor applied to the loop body.
    :type loop_unroll_factor: int
    :param bind_sub_block: Tells the compiler whether multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the
        number of iterations in this loop. Currently on 910B, at most 2 vector cores can be used.
    :type bind_sub_block: bool
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
    """Attaches a compiler hint annotation to a tensor, guiding Ascend code generation.

    Hints influence how the compiler handles memory layout, buffering, and other
    optimization decisions. The hint is attached to ``ptr`` and serialized as an
    MLIR annotation in the generated code.

    :param ptr: The tensor to annotate with the compiler hint.
    :type ptr: tensor
    :param hint_name: Hint identifier string (e.g., ``"hivm.multi_buffer"``).
    :type hint_name: str
    :param hint_val: Hint value. Can be a bool, int, constexpr string, or list of
        integers (serialized as an i64 array attribute).
    :type hint_val: bool | int | constexpr | list[int] | None
    """
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
    Set multi_buffer for an existing tensor.

    :param src: The tensor set to bufferize multiple time.
    :type src: tensor
    :param size: The number of copies.
    :type size: int or constexpr
    """
    buffer_size = _unwrap_if_constexpr(size)
    assert isinstance(buffer_size, int) and buffer_size == 2, f"only support bufferize equals 2"
    compile_hint_impl(src, "hivm.multi_buffer", buffer_size, _semantic.builder)
