# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__all__ = ["custom", "custom_semantic", "register_custom_op", "SyncEventSlot"]

import inspect
from warnings import warn
import types
import typing
import itertools
import triton.language.core as tl
from . import core

# Registry for custom op, mapping name to its configuration.
_custom_op_registry = {}


class SyncEventSlot:
    """One sync_event_slots entry for custom macro ops."""

    def __init__(self, set_pipe=None, wait_pipe=None, sync=None, event=None):
        self.set_pipe = set_pipe
        self.wait_pipe = wait_pipe
        self.sync = sync if sync is not None else core.SYNC_HINT.WAIT
        self.event = event


def _validate_pipe(pipe):
    if isinstance(pipe, core.PIPE):
        return
    if isinstance(pipe, (list, tuple)) and len(pipe) == 2:
        assert all(isinstance(p, core.PIPE) for p in pipe), \
            "macro custom op pipe sequence must contain PIPE values"
        return
    assert False, ("Invalid 'pipe' field: single PIPE lowers to hivm.hir.custom; "
                   "a sequence of two PIPE values lowers to hivm.hir.custom_macro")


def _is_macro_pipe(pipe):
    return isinstance(pipe, (list, tuple)) and len(pipe) == 2


def _sync_hint_value(sync):
    if isinstance(sync, core.SYNC_HINT):
        return sync.value
    if isinstance(sync, str):
        return getattr(core.SYNC_HINT, sync.upper()).value
    return sync


def _event_id_value(event):
    if isinstance(event, core.EVENT_ID):
        return event.value
    if isinstance(event, str):
        return getattr(core.EVENT_ID, event).value
    return event


def _build_sync_event_slot_attr(slot, builder):
    if isinstance(slot, SyncEventSlot):
        set_pipe = slot.set_pipe.value if slot.set_pipe is not None else None
        wait_pipe = slot.wait_pipe.value if slot.wait_pipe is not None else None
        macro_sync = _sync_hint_value(slot.sync)
        event = _event_id_value(slot.event) if slot.event is not None else None
    elif isinstance(slot, (list, tuple)):
        if len(slot) == 1 and slot[0] == 'internal':
            set_pipe = wait_pipe = event = None
            macro_sync = core.SYNC_HINT.INTERNAL.value
        elif len(slot) >= 2:
            set_pipe = slot[0].value
            wait_pipe = slot[1].value
            macro_sync = _sync_hint_value(slot[2]) if len(slot) > 2 else core.SYNC_HINT.WAIT.value
            event = _event_id_value(slot[3]) if len(slot) > 3 else None
        else:
            assert False, "invalid sync_event_slots entry"
    else:
        assert False, "sync_event_slots entries must be SyncEventSlot or tuple"

    return builder.get_sync_event_slot_attr(
        set_pipe,
        wait_pipe,
        macro_sync,
        event,
    )


def _get_op_class(name):
    # Try to get op class in _custom_op_registry.
    op_class = _custom_op_registry.get(name)
    if op_class is None:
        # Allow bulitin custom ops used without registry.
        assert name.startswith('__builtin_'), f"Custom Op '{name}' not registered."
        # Return a dummy op class for builtin custom op.
        op_class = type(
            "_builtin_custom_op", (object, ), {
                "name": name,
                "core": core.CORE.VECTOR,
                "pipe": core.PIPE.PIPE_V,
                "mode": core.MODE.SIMT,
                "signature": inspect.signature(object),
            })
    return op_class


def _unwrap_constexpr(arg):
    if isinstance(arg, tl.constexpr):
        return arg.value
    if isinstance(arg, (tuple, tl.tuple)):
        return tuple(_unwrap_constexpr(x) for x in arg)
    if isinstance(arg, list):
        return [_unwrap_constexpr(x) for x in arg]
    if isinstance(arg, dict):
        return {k: _unwrap_constexpr(v) for k, v in arg.items()}
    return arg


def _to_value(value, _semantic=None, ty=None):
    # Try to use 'type' attribute if ty not set.
    ty = getattr(value, 'type', ty) if ty is None else ty
    if isinstance(value, tl.tensor):
        if not value.type.is_block() and isinstance(ty, tl.dtype) and value.type != ty:
            # For a scalar variable, if its type is not the expected one
            # that specified by type hint 'ty', insert a cast for it.
            return _semantic.cast(value, ty).handle
        return value.handle
    if isinstance(value, bool):
        return _semantic.builder.get_int1(value)
    if isinstance(value, int):
        if isinstance(ty, tl.dtype):
            if ty.is_int64():
                return _semantic.builder.get_int64(value)
            if ty.is_uint64():
                return _semantic.builder.get_uint64(value)
            if ty.is_int32():
                return _semantic.builder.get_int32(value)
            if ty.is_uint32():
                return _semantic.builder.get_uint32(value)
            if ty.is_int16():
                return _semantic.builder.get_int16(value)
            if ty.is_uint16():
                return _semantic.builder.get_uint16(value)
            if ty.is_int8():
                return _semantic.builder.get_int8(value)
            if ty.is_uint8():
                return _semantic.builder.get_uint8(value)
        # default int32
        return _semantic.builder.get_int32(value)
    if isinstance(value, float):
        if isinstance(ty, tl.dtype):
            if ty.is_fp64():
                return _semantic.builder.get_fp64(value)
            if ty.is_fp32():
                return _semantic.builder.get_fp32(value)
            if ty.is_fp16():
                return _semantic.builder.get_fp16(value)
            if ty.is_bf16():
                return _semantic.builder.get_bf16(value)
        # default float32
        return _semantic.builder.get_fp32(value)
    if isinstance(value, tl.constexpr):
        return _to_value(value.value, _semantic)
    raise TypeError(f"Unsupported argument type {value} : {type(value)}")


def _to_operands(args, _semantic=None):
    operands = []
    for value in args:
        if value is None:
            continue
        if isinstance(value, (list, tuple, tl.tuple)):
            for item in value:
                operands.append(_to_value(item, _semantic))
        else:
            operands.append(_to_value(value, _semantic))
    return operands


def _get_element_type(ty):
    if isinstance(ty, types.GenericAlias):
        return typing.get_args(ty)[0]
    return ty


def _args_to_operands(op, _semantic, args, kwargs):
    if not op.signature.parameters:
        # Without parameters in signature, use the actual parameter order.
        return _to_operands(itertools.chain(args, kwargs.values()), _semantic)

    # Convert arguments to operands according the signature.
    operands = []
    bind = op.signature.bind(*args, **kwargs)
    for param in op.signature.parameters.values():
        value = bind.arguments.get(param.name)
        if value is None:
            continue
        ty = op.arg_type.get(param.name, param.annotation)
        if isinstance(value, (list, tuple, tl.tuple)):
            ty = _get_element_type(ty)
            for item in value:
                operands.append(_to_value(item, _semantic, ty))
        else:
            operands.append(_to_value(value, _semantic, ty))
    return operands


def _bind_op_arguments(op, args, kwargs):
    if not op.signature.parameters:
        return None
    return op.signature.bind(*args, **kwargs)


def _make_align_dim_attrs(op, builder, arg_attrs):
    # Find op argument by name using op.align_dim's key
    # We want to return a dict mapping for each align_dim key -> int attribute for the actual bound argument value.
    name = 'align_dim'
    if not hasattr(op, name):
        return

    # To find argument indices matching each align_dim key, check the op.signature parameters
    # and map align_dim key (argument name) to its index position.
    align_arg_indices = {}
    if hasattr(op, "signature"):
        param_names = list(op.signature.parameters.keys())
        for arg_name in op.align_dim.keys():
            if arg_name in param_names:
                align_arg_indices[arg_name] = param_names.index(arg_name)

    for arg, align_val in op.align_dim.items():
        if isinstance(arg, str) and arg in align_arg_indices:
            arg_attrs[align_arg_indices[arg]] = {name: builder.get_int_attr(align_val)}
            print(arg_attrs[align_arg_indices[arg]])
        elif isinstance(arg, int):
            arg_attrs[arg] = {name: builder.get_int_attr(align_val)}
            print(arg_attrs[arg])
        else:
            assert False, f"{name}'s keys should be string or int"


def _make_arg_attrs(op, builder):
    num_args = len(op.signature.parameters) if hasattr(op, "signature") else 0
    arg_attrs = [{} for _ in range(num_args)]

    _make_align_dim_attrs(op, builder, arg_attrs)
    return arg_attrs


def _add_optional_attr(op, name, builder, attrs):
    if hasattr(op, name):
        attrs[name] = builder.get_string_attr(getattr(op, name))


def _add_bitcode_attr(op, builder, attrs):
    name = 'bitcode'
    if not hasattr(op, name):
        return

    from pathlib import Path
    bitcode = Path(getattr(op, name))
    assert bitcode.exists(), f"Provided bitcode ({name}) not exist"
    attrs[name] = builder.get_string_attr(str(bitcode.absolute()))


def _add_optional_extra_buffer_attr(op, builder, attrs):
    name = 'extra_buffers'
    if not hasattr(op, name):
        return

    extra_buffers = getattr(op, name)
    if isinstance(extra_buffers, tuple):
        extra_buffers = [extra_buffers]

    extra_buffer_types, extra_buffer_sizes = zip(*extra_buffers)
    attrs[name + "_types"] = builder.get_type_array_attr([ty.to_ir(builder) for ty in extra_buffer_types])
    attrs[name + "_sizes"] = builder.get_i64_array_attr(list(extra_buffer_sizes))


def _add_optional_indexing_map_attr(op, builder, attrs):
    # Optional indexing map attribute:
    # `indexing_map` should be an iterable of al.affine_map (MLIR AffineMap) objects.
    name = 'indexing_map'
    if not hasattr(op, name):
        return

    indexing_map = getattr(op, name)
    attrs[name] = builder.get_affine_map_array_attr(indexing_map)


def _add_optional_iterator_types_attr(op, builder, attrs):
    name = 'iterator_types'
    if not hasattr(op, name):
        return

    attrs[name] = builder.get_iterator_types_attr([iterator_type.value for iterator_type in getattr(op, name)])


def _add_sync_event_slots_attr(op, builder, attrs):
    if not hasattr(op, 'sync_event_slots'):
        return
    slots = getattr(op, 'sync_event_slots')
    if isinstance(slots, tuple):
        slots = list(slots)
    attrs['sync_event_slots'] = builder.get_array_attr([_build_sync_event_slot_attr(slot, builder) for slot in slots])


def _make_attrs(op, builder, is_macro):
    attrs = {
        'hivm.tcore_type': builder.get_core_type_attr(op.core.value),
    }
    # VF mode is only required for non-CUBE custom ops (see HIVM requiresVFMode()).
    if op.core != core.CORE.CUBE:
        attrs['hivm.vf_mode'] = builder.get_vf_mode_attr(op.mode.value)
    if is_macro:
        pipe_in, pipe_out = op.pipe
        attrs['hivm.pipe_in'] = builder.get_pipe_attr(pipe_in.value)
        attrs['hivm.pipe_out'] = builder.get_pipe_attr(pipe_out.value)
        _add_sync_event_slots_attr(op, builder, attrs)
    else:
        attrs['hivm.pipe'] = builder.get_pipe_attr(op.pipe.value)

    if not op.name.startswith('__builtin_'):
        assert hasattr(op, 'symbol'), f"Non builtin custom op, symbol is required."
        assert hasattr(op, 'bitcode'), f"Non builtin custom op, bitcode path is required."

    # Add bit code path attribute, formalize to abosulte path.
    _add_bitcode_attr(op, builder, attrs)

    _add_optional_indexing_map_attr(op, builder, attrs)
    _add_optional_iterator_types_attr(op, builder, attrs)

    _add_optional_extra_buffer_attr(op, builder, attrs)

    _add_optional_attr(op, 'symbol', builder, attrs)
    _add_optional_attr(op, 'source', builder, attrs)
    _add_optional_attr(op, 'compile', builder, attrs)
    # Extra attributes can be added here, such as op.extra_attr="attr_a=xx"
    _add_optional_attr(op, 'extra_attr', builder, attrs)

    return attrs


def _to_result(res, res_types):
    assert (len(res) == len(res_types))
    n_res = len(res)
    if n_res == 0:
        return None
    if n_res == 1:
        return tl.tensor(res[0], res_types[0])
    return tl.tuple(tl.tensor(res[i], res_types[i]) for i in range(n_res))


def _init_op(op_class, *args, **kwargs):
    op = op_class.__new__(op_class)
    # Add arg_type dict to support dynamic argument type specifying.
    setattr(op, 'arg_type', {})
    if op_class.signature.parameters:
        # Init with arguments validate.
        op_class.__init__(op, *args, **kwargs)
    return op


def custom_semantic(name: str, *args, _semantic=None, **kwargs):
    """Internal semantic handler for custom operations (used by code generator).

    User code should use :func:`custom` instead.
    """
    name = _unwrap_constexpr(name)
    # Get op class according the name.
    op_class = _get_op_class(name)
    deprecated_message = getattr(op_class, "_deprecated_message", None)
    if deprecated_message:
        warn(deprecated_message, FutureWarning, stacklevel=3)
    # Convert constexpr to value in arguments.
    args = _unwrap_constexpr(args)
    kwargs = _unwrap_constexpr(kwargs)
    kwargs.pop('sync_related_args', None)
    # Create op instance from op class with the arguments.
    op = _init_op(op_class, *args, **kwargs)
    is_macro = _is_macro_pipe(op.pipe)
    if hasattr(op, 'sync_event_slots'):
        assert is_macro, "sync_event_slots is only supported for macro custom ops"
    # Prepare inputs and outputs operands.
    out = kwargs.pop('out', [])
    outs = out if isinstance(out, (list, tuple, tl.tuple)) else [out]
    outputs = _to_operands(outs, _semantic)
    inputs = _args_to_operands(op, _semantic, args, kwargs)
    main_builder = _semantic.builder
    builder = getattr(main_builder, '_ascend_builder')
    # Setup attributes. These only read context-level state (types, attrs), so
    # the delegate builder is fine here.
    attrs = _make_attrs(op, builder, is_macro)
    arg_attrs = _make_arg_attrs(op, builder)
    # Emit through the unified-builder wrappers so the delegate uses the main
    # builder's current insertion point and location, not stale delegate state.
    if is_macro:
        res = main_builder.create_custom_macro_op(name, attrs, inputs, outputs, arg_attrs)
    else:
        res = main_builder.create_custom_op(name, attrs, inputs, outputs, arg_attrs)
    # Results with same types as outputs.
    res_types = [out.type for out in outs]
    return _to_result(res, res_types)


@core.builtin
def custom(name: str, *args, _semantic=None, **kwargs):
    """Invoke a custom operation with the given name and arguments."""
    return custom_semantic(name, *args, _semantic=_semantic, **kwargs)


def register_custom_op(op):
    """Register a custom operation so that it can be invoked via :func:`custom`.

    Used as a class decorator. The decorated class must define a ``core`` field
    (a :class:`CORE` value), a ``pipe`` field (a :class:`PIPE` value), and for
    non-CUBE ops a ``mode`` field (a :class:`MODE` value).

    :param op: the custom op class to register.
    """
    assert inspect.isclass(op), "@register_custom_op should decorate on a class."
    # Use class name if name not set.
    if not hasattr(op, 'name'):
        setattr(op, 'name', op.__name__)
    # The op name should not be used.
    assert op.name not in _custom_op_registry, f"Custom op name '{op.name}' already used."

    # Check required core, pipe fields; mode is required for non-CUBE ops only.
    assert hasattr(op, 'core'), "'core' field is required."
    assert hasattr(op, 'pipe'), "'pipe' field is required."
    assert isinstance(op.core, core.CORE), "Invalid 'core' field, CORE type is required."
    _validate_pipe(op.pipe)
    if op.core != core.CORE.CUBE:
        assert hasattr(op, 'mode'), "'mode' field is required for non-CUBE custom ops."
        assert isinstance(op.mode, core.MODE), "Invalid 'mode' field, MODE type is required."
    if hasattr(op, 'sync_event_slots'):
        assert _is_macro_pipe(op.pipe), "sync_event_slots requires a two-pipe macro custom op"
    # Retrieve arguments signature from __init__ method and save it.
    signature = inspect.signature(op)
    setattr(op, 'signature', signature)
    # Register the custom op configuration.
    _custom_op_registry[op.name] = op
    return op


_dtype_cname_dict = {
    'int1': 'bool',
    'int8': 'int8_t',
    'int16': 'int16_t',
    'int32': 'int32_t',
    'int64': 'int64_t',
    'uint8': 'uint8_t',
    'uint16': 'uint16_t',
    'uint32': 'uint32_t',
    'uint64': 'uint64_t',
    'fp16': 'half',
    'bf16': 'bfloat16_t',
    'fp32': 'float',
    'fp64': 'double',
    'fp8e5': 'float8_e5m2_t',
    'fp8e4nv': 'float8_e4m3_t',
    # other float8 types are not supported yet,
    # such as 'fp8e4b8', 'fp8e4b15', 'fp8e5b16'.
}


def _cname(self):
    """Return the corresponding C name of the given tl.dtype"""
    return _dtype_cname_dict.get(self.name, self.name)


# Add 'cname' property to tl.dtype class.
tl.dtype.cname = property(_cname, None)
