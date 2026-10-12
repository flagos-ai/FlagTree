"""Shared support for C-interoperable custom types (structs and arrays).

This module implements the frontend half of the "custom types" feature
described in ``docs/custom-types/RFC.md``.  It is imported by both the main
frontend (``triton/language/core.py``) and the Ascend frontend copy
(``triton/spec/ascend/language/core.py``) so that the two cannot drift.

The design is **reference-only**: a struct or array is always accessed through
a pointer and is never a Triton SSA value.  See the RFC for the full rationale.

This module deliberately avoids importing the sibling ``core`` module by name
at import time (to keep the circular import with ``core.py`` safe); it holds a
module reference and resolves attributes lazily.
"""

from __future__ import annotations

import ctypes
from typing import Any, List, Optional, Sequence, Tuple

# Module reference, not a name import: ``core`` imports this module near the
# end of its own body, so ``core.base_type`` exists by the time it is used.
from . import core as _core

# The Triton interpreter executes the kernel's Python directly and therefore
# cannot inject ``_semantic`` into attribute access.  It installs its semantic
# here so struct/array value objects can resolve field/element access.
_default_semantic = None


def set_default_semantic(semantic) -> None:
    global _default_semantic
    _default_semantic = semantic


# ---------------------------------------------------------------------------
# ctypes -> Triton mapping
# ---------------------------------------------------------------------------

# Scalar ctypes types that map to a Triton scalar dtype.  Only fixed-width,
# platform-independent types are listed (see RFC 5.2).
#
# Caveat: ctypes aliases some names to the same type object.  On LP64 hosts
# ``c_long``/``c_ssize_t`` *are* ``c_int64`` and are therefore accepted as
# ``int64`` (they cannot be rejected independently of ``c_int64``), while the
# distinct objects ``c_ulong``/``c_size_t`` are rejected.  Since the layout is
# always taken from the host, this is consistent with the design's
# host-is-authoritative stance.
_CTYPES_SCALAR_MAP = {
    ctypes.c_bool: "int1",
    ctypes.c_byte: "int8",
    ctypes.c_int8: "int8",
    ctypes.c_ubyte: "uint8",
    ctypes.c_uint8: "uint8",
    ctypes.c_short: "int16",
    ctypes.c_int16: "int16",
    ctypes.c_ushort: "uint16",
    ctypes.c_uint16: "uint16",
    ctypes.c_int: "int32",
    ctypes.c_int32: "int32",
    ctypes.c_uint: "uint32",
    ctypes.c_uint32: "uint32",
    ctypes.c_longlong: "int64",
    ctypes.c_int64: "int64",
    ctypes.c_ulonglong: "uint64",
    ctypes.c_uint64: "uint64",
    ctypes.c_float: "float32",
    ctypes.c_double: "float64",
}

# Triton dtype name -> the token used in signature strings.  These tokens are
# parsed back by :func:`parse_signature`.
_DTYPE_TOKEN = {
    "int1": "i1",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "fp16": "fp16",
    "bf16": "bf16",
    "fp32": "fp32",
    "fp64": "fp64",
}
_TOKEN_TO_CORE_ATTR = {
    "i1": "int1",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
}


def _is_structure(ct: Any) -> bool:
    return isinstance(ct, type) and issubclass(ct, ctypes.Structure)


def _classify_ctype(ct: Any, path: str) -> "_core.base_type":
    """Map a ctypes type to a Triton type, or raise a clear error."""
    if ct in _CTYPES_SCALAR_MAP:
        return getattr(_core, _CTYPES_SCALAR_MAP[ct])

    if ct is ctypes.c_void_p:
        return _core.pointer_type(_core.int8)

    # Typed pointer: ``ctypes.POINTER(T)`` / ``ctypes.pointer`` results.
    pointer_base = getattr(ctypes, "_Pointer", None)
    if pointer_base is not None and isinstance(ct, type) and issubclass(ct, pointer_base):
        return _core.pointer_type(_classify_ctype(ct._type_, path))

    if isinstance(ct, type) and issubclass(ct, ctypes.Array):
        element_ty = _classify_ctype(ct._type_, path)
        return _array_type(element_ty, ct._length_)

    if _is_structure(ct):
        st = getattr(ct, "_triton_struct_type", None)
        if st is None:
            fields, size, align = _struct_layout(ct, path)
            st = _struct_type(fields, size, align, name=getattr(ct, "__name__", None))
        return st

    raise TypeError(
        f"field {path}: unsupported ctypes type {getattr(ct, '__name__', ct)!r}. "
        f"Only fixed-width C scalars, pointers, fixed-size arrays, and nested "
        f"structures are supported.")


def _struct_layout(cls: Any, path: str):
    """Compute ``(fields, size, align)`` for a ctypes.Structure subclass."""
    fields: List[Tuple[str, Any, int]] = []
    for entry in cls._fields_:
        if len(entry) != 2:
            raise TypeError(f"field {path}: bitfields are not supported")
        name, ct = entry
        ty = _classify_ctype(ct, f"{path}.{name}")
        offset = getattr(cls, name).offset
        fields.append((name, ty, int(offset)))
    return fields, int(ctypes.sizeof(cls)), int(ctypes.alignment(cls))


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class _array_type(_core.base_type):
    """A fixed-size C array ``element_ty[count]``.

    Appears only as a pointer pointee or as a struct field, never as a value.
    """

    def __init__(self, element_ty: "_core.base_type", count: int):
        if int(count) < 1:
            raise ValueError("array element count must be >= 1")
        self.element_ty = element_ty
        self.count = int(count)

    # base_type interface -------------------------------------------------
    def to_ir(self, builder) -> Any:
        return builder.get_array_ty(self.element_ty.to_ir(builder), self.count)

    def _flatten_ir_types(self, builder, out: List[Any]) -> None:
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        raise NotImplementedError("array types have reference semantics only")

    def __eq__(self, other) -> bool:
        return isinstance(other, _array_type) and self.element_ty == other.element_ty and self.count == other.count

    def __hash__(self) -> int:
        return hash(("array", self.element_ty, self.count))

    def __str__(self) -> str:
        return f"array<{self.element_ty},{self.count}>"

    def mangle(self) -> str:
        return f"AR{self.count}_{self.element_ty.mangle()}"


class _struct_type(_core.base_type):
    """A C struct with an explicit byte layout.

    ``fields`` is an ordered list of ``(name, triton_type, byte_offset)``.
    ``size``/``align`` are the ctypes-provided values.  Like arrays, a struct
    appears only as a pointer pointee or as a nested struct field.
    """

    def __init__(self, fields: Sequence[Tuple[str, Any, int]], size: int, align: int, name: Optional[str] = None):
        self.fields = tuple((str(n), t, int(o)) for (n, t, o) in fields)
        self.size = int(size)
        self.align = int(align)
        self.name = name

    # -- helpers ----------------------------------------------------------
    def field(self, name: str) -> Tuple[str, Any, int]:
        for f in self.fields:
            if f[0] == name:
                return f
        available = ", ".join(f[0] for f in self.fields)
        raise AttributeError(f"{self.name or 'struct'} has no field '{name}' (fields: {available})")

    # base_type interface -------------------------------------------------
    def to_ir(self, builder) -> Any:
        names = [f[0] for f in self.fields]
        types = [f[1].to_ir(builder) for f in self.fields]
        offsets = [f[2] for f in self.fields]
        return builder.get_struct_ty(names, types, offsets, self.size, self.align)

    def _flatten_ir_types(self, builder, out: List[Any]) -> None:
        out.append(self.to_ir(builder))

    def _unflatten_ir(self, handles, cursor):
        raise NotImplementedError("struct types have reference semantics only")

    def __eq__(self, other) -> bool:
        return (isinstance(other, _struct_type) and self.fields == other.fields and self.size == other.size
                and self.align == other.align)

    def __hash__(self) -> int:
        return hash(("struct", self.fields, self.size, self.align))

    def __str__(self) -> str:
        return self.signature()

    def mangle(self) -> str:
        body = "_".join(f"{n}_{t.mangle()}_{o}" for (n, t, o) in self.fields)
        return f"ST{len(self.fields)}_{body}_s{self.size}a{self.align}"

    def signature(self) -> str:
        """Self-contained string reconstructible by :func:`parse_signature`.

        Must not contain commas: the Ascend launcher joins/splits the argument
        signature on commas (to flatten tuples), so a comma here would corrupt
        the kernel argument list.  Fields are separated by `;` and an array is
        written `array<elem[N]>`.
        """
        label = f"<{self.name}>" if self.name else ""
        body = ";".join(f"{n}:{_type_sig(t)}@{o}" for (n, t, o) in self.fields)
        return f"struct{label}{{{body}}};size={self.size};align={self.align}"


def type_sizeof(ty) -> int:
    """Size in bytes of a struct/array field type (host layout)."""
    if isinstance(ty, _array_type):
        return type_sizeof(ty.element_ty) * ty.count
    if isinstance(ty, _struct_type):
        return ty.size
    if isinstance(ty, _core.pointer_type):
        return 8
    bitwidth = getattr(ty, "primitive_bitwidth", None)
    if bitwidth is not None:
        return max(1, bitwidth // 8)
    raise TypeError(f"cannot compute the size of {ty!r}")


# ---------------------------------------------------------------------------
# Signature strings
# ---------------------------------------------------------------------------


def _type_sig(ty: Any) -> str:
    if isinstance(ty, _array_type):
        return f"array<{_type_sig(ty.element_ty)}[{ty.count}]>"
    if isinstance(ty, _struct_type):
        return ty.signature()
    if isinstance(ty, _core.pointer_type):
        return "*" + _type_sig(ty.element_ty)
    token = _DTYPE_TOKEN.get(getattr(ty, "name", None))
    if token is None:
        raise TypeError(f"cannot serialize field type {ty!r}")
    return token


class _SigParser:

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _expect(self, ch: str) -> None:
        if self._peek() != ch:
            raise ValueError(f"expected {ch!r} at position {self.pos} in {self.text!r}")
        self.pos += 1

    def _ident(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] in "_."):
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"expected identifier at position {self.pos} in {self.text!r}")
        return self.text[start:self.pos]

    def _int(self) -> int:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if start == self.pos:
            raise ValueError(f"expected integer at position {self.pos} in {self.text!r}")
        return int(self.text[start:self.pos])

    def parse_type(self) -> Any:
        if self._peek() == "*":
            self.pos += 1
            return _core.pointer_type(self.parse_type())
        name = self._ident()
        if self._peek() == "<":
            self._expect("<")
            if name == "array":
                element_ty = self.parse_type()
                self._expect("[")
                count = self._int()
                self._expect("]")
                self._expect(">")
                return _array_type(element_ty, count)
            if name == "struct":
                label = self._ident()
                self._expect(">")
                return self._parse_struct_body(label)
            raise ValueError(f"unknown parameterized type {name!r}")
        token = _TOKEN_TO_CORE_ATTR.get(name)
        if token is None:
            raise ValueError(f"unknown type token {name!r} in {self.text!r}")
        return getattr(_core, token)

    def _parse_struct_body(self, label: str) -> _struct_type:
        self._expect("{")
        fields: List[Tuple[str, Any, int]] = []
        if self._peek() != "}":
            while True:
                field_name = self._ident()
                self._expect(":")
                field_ty = self.parse_type()
                self._expect("@")
                offset = self._int()
                fields.append((field_name, field_ty, offset))
                if self._peek() == ";":
                    self.pos += 1
                    continue
                break
        self._expect("}")
        size = align = None
        if self.text.startswith(";size=", self.pos):
            self.pos += len(";size=")
            size = self._int()
        if self.text.startswith(";align=", self.pos):
            self.pos += len(";align=")
            align = self._int()
        if size is None:
            raise ValueError(f"struct signature missing ;size= in {self.text!r}")
        if align is None:
            align = max((t.align for (_, t, _) in fields if hasattr(t, "align")), default=1)
        return _struct_type(fields, size, align, name=label)


def parse_signature(text: str) -> "_core.base_type":
    """Parse a struct/array signature produced by :meth:`_struct_type.signature`."""
    parser = _SigParser(text)
    ty = parser.parse_type()
    if parser.pos != len(text):
        raise ValueError(f"trailing characters in signature {text!r}")
    return ty


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def struct_type(cls):
    """Decorator turning a ``ctypes.Structure`` subclass into a Triton struct type.

    Usage::

        @tl.struct_type
        class TensorView(ctypes.Structure):
            _fields_ = [...]
    """
    if not _is_structure(cls):
        raise TypeError("@struct_type can only be applied to a ctypes.Structure subclass")
    fields, size, align = _struct_layout(cls, cls.__name__)
    cls._triton_struct_type = _struct_type(fields, size, align, name=cls.__name__)
    return cls


# ---------------------------------------------------------------------------
# Frontend values (reference semantics)
# ---------------------------------------------------------------------------


class struct_pointer(_core.base_value):
    """A pointer to a struct, with field access in kernels."""

    def __init__(self, handle, type):
        self.handle = handle
        self.type = type

    def _flatten_ir(self, handles: List[Any]) -> None:
        handles.append(self.handle)

    # The interpreter executes Python directly, so field access arrives as a
    # plain attribute access.  Under the JIT code generator these are handled
    # by CodeGenerator.visit_Attribute instead and never reach here.
    def __getattr__(self, name):
        if name.startswith("_") or name in ("handle", "type"):
            raise AttributeError(name)
        if _default_semantic is None:
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
        return _default_semantic.struct_field(self, name)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in ("handle", "type"):
            object.__setattr__(self, name, value)
            return
        if _default_semantic is None:
            object.__setattr__(self, name, value)
            return
        _default_semantic.struct_field_store(self, name, value)


class array_pointer(_core.base_value):
    """A pointer to an array, with element access in kernels."""

    def __init__(self, handle, type):
        self.handle = handle
        self.type = type

    def _flatten_ir(self, handles: List[Any]) -> None:
        handles.append(self.handle)

    def __getitem__(self, index, _semantic=None):
        semantic = _semantic if _semantic is not None else _default_semantic
        if semantic is None:
            raise ValueError("no semantic available for array element access")
        return semantic.array_element(self, index)

    def __setitem__(self, index, value):
        if _default_semantic is None:
            raise ValueError("no semantic available for array element store")
        _default_semantic.array_element_store(self, index, value)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


class StructView:
    """Pairs a struct type with a device buffer holding its bytes."""

    def __init__(self, struct_type: _struct_type, buffer):
        self.struct_type = struct_type
        self.buffer = buffer

    @property
    def nbytes(self) -> int:
        return self.struct_type.size

    def data_ptr(self) -> int:
        return self.buffer.data_ptr()


def _require_struct_type(instance) -> _struct_type:
    st = getattr(type(instance), "_triton_struct_type", None)
    if st is None:
        raise TypeError("expected an instance of a class decorated with @tl.struct_type")
    return st


def to_device(instance, buffer=None, device=None) -> StructView:
    """Upload a host ctypes struct instance to a device buffer.

    If ``buffer`` is omitted, a ``torch`` uint8 tensor is allocated on
    ``device`` (default: the current device).  ``buffer`` may instead be any
    writable byte buffer (e.g. a ``bytearray``) or a device tensor exposing
    ``copy_``/``device``.
    """
    st = _require_struct_type(instance)
    raw = ctypes.string_at(ctypes.addressof(instance), ctypes.sizeof(type(instance)))
    if buffer is None:
        import torch
        buffer = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device=device)
    elif hasattr(buffer, "copy_"):
        import torch
        buffer.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(buffer.device))
    else:
        buffer[:] = raw
    return StructView(st, buffer)
