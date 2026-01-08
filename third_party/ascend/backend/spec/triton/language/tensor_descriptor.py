# TODO: When upgrading to Triton 3.4.0, remove this file,
#       use the upstream Triton functions, and update core.py and semantic.py accordingly.
from __future__ import annotations

import builtins
from typing import List, Tuple, Sequence, TypeVar
from enum import Enum

from triton._C.libtriton import ir
from triton.language.core import (
    builtin,
    constexpr,
    tensor,
    _value,
    void as real_void,
)

from triton.language.semantic import (
    _convert_to_ir_values,
    _str_to_load_cache_modifier,
    _str_to_eviction_policy,
)

from triton.language._utils import validate_block_shape


def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return builtins.tuple(_unwrap_if_constexpr(x) for x in o)
    if isinstance(o, tuple):
        return tuple(_unwrap_if_constexpr(x) for x in o)
    return o.value if isinstance(o, constexpr) else o


def _unwrap_shape(shape):
    shape = _unwrap_if_constexpr(shape)
    return [_unwrap_if_constexpr(s) for s in shape]


def _normalize_tuple(t):
    normalized_tuple = _unwrap_if_constexpr(t)
    if isinstance(normalized_tuple, (list, builtins.tuple)):
        normalized_tuple = tuple(normalized_tuple)
    return normalized_tuple


def descriptor_load(desc: tensor_descriptor_base, offsets, cache_modifier: str,
                    eviction_policy: str, builder: ir.builder) -> tensor:
    assert isinstance(desc, tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)
    x = builder.create_descriptor_load(desc.handle, offsets, _str_to_load_cache_modifier(cache_modifier),
                                            _str_to_eviction_policy(eviction_policy))
    return tensor(x, desc.block_type)


def validate_store_like(desc: tensor_descriptor_base, value: tensor, offsets) -> None:
    assert isinstance(desc, tensor_descriptor_base)
    ndim = len(desc.block_shape)
    assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
    assert value.shape == desc.block_shape


def descriptor_store(desc: tensor_descriptor_base, value: tensor, offsets, builder: ir.builder) -> tensor:
    validate_store_like(desc, value, offsets)
    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)
    return tensor(builder.create_descriptor_store(desc.handle, value.handle, offsets), real_void)



class base_value(_value):
    """Base class of values that exist in the triton IR (i.e. not constexprs).
    """
    type: base_type

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        """Flatten frontend value into a sequence of mlir handles, which are appended
        to the output list
        """
        raise NotImplementedError


class base_type:

    def __eq__(self, other):
        raise NotImplementedError("Types must implement __eq__")

    def __ne__(self, other):
        return not (self == other)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError

    def mangle(self) -> str:
        raise NotImplementedError(f"NYI: Type mangling for type {self.__class__}")

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        raise NotImplementedError


class tuple(base_value):

    def __init__(self, args: Sequence, type: tuple_type = None):
        self.values = [i for i in args]

        def get_type(x):
            if isinstance(x, dtype):
                return dtype
            if isinstance(x, (int, float)):
                return constexpr
            return x.type

        self.type = type or tuple_type([get_type(x) for x in self.values])

    def __getitem__(self, idx: constexpr):
        if isinstance(idx, int):
            idx = constexpr(idx)
        if isinstance(idx, constexpr):
            return self.values[idx]
        else:
            assert isinstance(idx, (slice, builtins.slice))
            return tuple(self.values[idx.start:idx.stop:idx.step])

    def __getattr__(self, name):
        return self.values[self.type.fields.index(name)]

    def __setitem__(self, idx: constexpr, value):
        if isinstance(idx, int):
            idx = constexpr(idx)
        assert isinstance(idx, constexpr)
        self.values[idx] = value

    def __add__(self, other):
        other = _normalize_tuple(other)
        return tuple(self.values + other.values)

    def __mul__(self, other):
        assert isinstance(other, constexpr)
        return tuple(self.values * other.value)

    def __eq__(self, other):
        other = _normalize_tuple(other)
        return constexpr(self.values == other.values)

    def __hash__(self):
        return hash(builtins.tuple(self.values))

    def __str__(self):
        return str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def _flatten_ir(self, handles: List[ir.value]):
        for v in self.values:
            print("[debug]tuple _flatten_ir: value:", v)
            v._flatten_ir(handles)
            print("[debug]tuple _flatten_ir: handles:", handles)

    def __repr__(self):
        return f"({' ,'.join(repr(x) for x in self.values)})"


class tuple_type(base_type):

    def __init__(self, types, fields=None):
        self.types = types
        self.fields = fields or [''] * len(types)
        self.name = '[' + ','.join([f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + ']'

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]):
        for ty in self.types:
            if not isinstance(ty, constexpr):
                ty._flatten_ir_types(builder, out)

    def __getitem__(self, index: int) -> dtype:
        return self.types[index]

    def __eq__(self, other):
        return type(self) is type(other) and self.types == other.types and self.fields == other.fields

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tuple, int]:
        values = []
        for ty in self.types:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values.append(value)
        return tuple(values, self), cursor

    def mangle(self):
        return 'T' + '_'.join(ty.mangle for ty in self.types) + 'T'


class dtype(base_type):
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    class KIND(Enum):
        BOOLEAN = 0
        INTEGRAL = 1
        FLOATING = 2

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        # flagtree backend specialization
        from ._utils import spec_get_primitive_bitwidth
        get_primitive_bitwidth = spec_get_primitive_bitwidth
        self.primitive_bitwidth = get_primitive_bitwidth(name)
        self.itemsize = self.primitive_bitwidth // 8
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.exponent_bias = 16
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 52
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def kind(self):
        # Return int value following the type ordering bool < integer < fp
        if self.is_bool():
            return dtype.KIND.BOOLEAN
        elif self.is_int():
            return dtype.KIND.INTEGRAL
        else:
            assert self.is_floating()
            return dtype.KIND.FLOATING

    def get_int_max_value(self):
        if self.is_int_signed():
            return 2**(self.int_bitwidth - 1) - 1
        if self.is_int_unsigned():
            return 2**self.int_bitwidth - 1
        assert False

    def get_int_min_value(self):
        if self.is_int_signed():
            return -2**(self.int_bitwidth - 1)
        if self.is_int_unsigned():
            return 0
        assert False

    @staticmethod
    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, ))

    @property
    def scalar(self):
        return self

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name.startswith("fp8"):
            if self.name not in builder.options.supported_fp8_dtypes:
                raise ValueError(f'type {self} not supported in this architecture. '
                                 f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')

        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_ty()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_ty()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_ty()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):
        return self.name

    def codegen_name(self):
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        if self.is_int():
            SIGNED = dtype.SIGNEDNESS.SIGNED
            prefix = 'i' if self.int_signedness == SIGNED else 'u'
            return prefix + str(self.int_bitwidth)
        if self.is_floating():
            return str(self)
        if self.is_void():
            return 'V'
        return super().mangle()

    def with_element_ty(self, element_ty: dtype):
        assert not self.is_block()
        return element_ty


class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty

        # Note that block_type's shape is a list of int
        # while tensor's shape is a list of constexpr.
        assert (isinstance(shape, (list, tuple)))

        # shape can be empty ([]) when an input is a 0D tensor.
        self.shape = tuple(_unwrap_shape(shape))
        if not self.shape:
            raise TypeError('0d block_type is forbidden')

        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> Tuple[int]:
        return self.shape

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return block_type(scalar_ty, self.shape)

    def __eq__(self, other) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    @property
    def scalar(self):
        return self.element_ty

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        return f'{elt}S{shape}S'


class tuple_type(base_type):

    def __init__(self, types, fields=None):
        self.types = types
        self.fields = fields or [''] * len(types)
        self.name = '[' + ','.join([f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + ']'

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]):
        for ty in self.types:
            if not isinstance(ty, constexpr):
                ty._flatten_ir_types(builder, out)

    def __getitem__(self, index: int) -> dtype:
        return self.types[index]

    def __eq__(self, other):
        return type(self) is type(other) and self.types == other.types and self.fields == other.fields

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tuple, int]:
        values = []
        for ty in self.types:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values.append(value)
        return tuple(values, self), cursor

    def mangle(self):
        return 'T' + '_'.join(ty.mangle for ty in self.types) + 'T'


class tensor_descriptor_base_type(base_type):

    def __init__(self, block_type: block_type):
        self.block_type = block_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        value = tensor_descriptor_base(handles[cursor], self.block_type)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        is_signed = self.block_type.element_ty.is_int_signed()
        out.append(builder.create_tensor_descriptor_type(self.block_type.to_ir(builder), is_signed))

    def __str__(self) -> str:
        # ex. "tensor_descriptor<float32[16, 32]>"
        return f"tensor_descriptor<{self.block_type}>"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False
        return self.block_type == other.block_type

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}"


class tensor_descriptor_base(base_value):
    """"
    A tensor descriptor with unknown shape and strides
    """

    def __init__(self, handle, block_type: block_type):
        """Not called by user code."""
        super().__init__(handle)

        self.handle = handle  # IR handle
        self.type = tensor_descriptor_base_type(block_type)  # Tensor type (block_type)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, offsets: Sequence[constexpr | tensor], _builder=None) -> tensor:
        """Load a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be filled with zeros.

        :note: Offset must be a multiple of 16-bytes
        """
        return descriptor_load(self, offsets, "", "", _builder)

    @builtin
    def store(self, offsets: Sequence[constexpr | tensor], value: tensor, _builder=None) -> tensor:
        """Store a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be ignored.

        :note: Offset must be a multiple of 16-bytes
        """
        return descriptor_store(self, value, offsets, _builder)


class tensor_descriptor_type(tensor_descriptor_base_type):

    def __init__(self, block_type: block_type, shape_type: tuple_type, strides_type: tuple_type):
        self.block_type = block_type
        self.shape_type = shape_type
        self.strides_type = strides_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        shape = shape.values
        strides = strides.values
        value = tensor_descriptor(handle, shape, strides, self.block_type)
        return value, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        super()._flatten_ir_types(builder, out)
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return super().__eq__(other) and (self.shape_type == other.shape_type) and (self.strides_type
                                                                                    == other.strides_type)


class tensor_descriptor(tensor_descriptor_base):
    """A descriptor representing a tensor in global memory.
    """

    def __init__(self, handle, shape: List[tensor], strides: List[tensor], block_type: block_type):
        """Not called by user code."""
        # IR handle
        super().__init__(handle, block_type)
        # Global shape
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.type = tensor_descriptor_type(
            block_type,
            shape_type=self.shape.type,
            strides_type=self.strides.type,
        )

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)
