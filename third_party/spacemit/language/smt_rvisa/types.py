# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Type metadata for @smt_rvisa kernel parameters.

Parameter annotations are structured objects, not MLIR text: ``sr.mem(f16)`` /
``sr.mem(f32, out=True)`` / ``sr.index`` build ``Ty`` instances directly, and
codegen reads shape/element/layout/space as attributes. Types cross into C++
structurally too: ``Ty.build(builder)`` calls the official MLIR type
constructors exposed on the builder (``get_memref_type`` / ``get_vector_type``
/ ``get_float_type`` / ...). No type text is assembled in Python anywhere.

The only type *text* that enters the system is what kernel authors write as
literals on the LLVM channel (e.g. ``call_intrinsic(..., result_t="!llvm.ptr")``);
that text is fed to the official MLIR parser (``builder.parse_type``) and
re-derived structurally by ``classify_type`` — no Python-side parser.

Usage:
    def my_fn(A: sr.mem(f16), n: sr.index, C: sr.mem(f32, out=True)):
        ...
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Structured MLIR type metadata (Ty hierarchy).
#
# codegen tracks the type of every SSA value as a Ty object instead of an MLIR
# text string: shape/element/layout queries are attribute reads (no regex, no
# substring sniffing), and MLIR types are materialised only at the C++ builder
# boundary via ``Ty.build(builder)`` (structured constructor bindings on
# ir.builder, triton_shared.cc init_triton_smt_rvisa_ir). Nothing round-trips
# structured -> text -> structured.
# ---------------------------------------------------------------------------


def _is_scalar_name(tok: str) -> bool:
    """index / iN / fN / bfN — the builtin scalar type names the DSL accepts."""
    if tok == "index":
        return True
    for prefix in ("bf", "f", "i"):
        if tok.startswith(prefix) and tok[len(prefix):].isdigit():
            return True
    return False


class Ty:
    """Base of the structured type hierarchy.

    ``build(b)`` materialises the MLIR type through the builder's structured
    type constructors (b = ir.builder / TritonOpBuilder pybind wrapper).
    """

    __slots__ = ()

    def build(self, b):
        raise NotImplementedError


class ScalarTy(Ty):
    """A builtin scalar type: ``index``, ``i1``..``i64``, ``f16``/``bf16``/``f32``/``f64``.

    ``kind``/``bits``/``is_bf`` are resolved once at construction; ``name`` is
    kept because a few consumers key on it (e.g. mma_cube dtype dispatch).
    """

    __slots__ = ("name", "kind", "bits", "is_bf")

    def __init__(self, name: str):
        if not isinstance(name, str) or not _is_scalar_name(name):
            raise ValueError(f"ScalarTy: expected index/iN/fN/bfN, got {name!r}")
        self.name = name
        if name == "index":
            self.kind, self.bits, self.is_bf = "index", 0, False
        elif name.startswith("bf"):
            self.kind, self.bits, self.is_bf = "float", int(name[2:]), True
        elif name.startswith("f"):
            self.kind, self.bits, self.is_bf = "float", int(name[1:]), False
        else:  # iN
            self.kind, self.bits, self.is_bf = "int", int(name[1:]), False

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    def build(self, b):
        if self.kind == "index":
            return b.get_index_type()
        if self.kind == "int":
            return b.get_integer_type(self.bits)
        return b.get_float_type(self.bits, self.is_bf)

    def __repr__(self) -> str:
        return f"ScalarTy({self.name!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, ScalarTy) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("ScalarTy", self.name))


class ScalableDim:
    """A scalable vector dimension ``[N]`` (vscale * N lanes).

    Only ever produced by ``classify_type`` from types parsed by the official
    MLIR parser (LLVM-channel user literals); codegen constructs static dims.
    """

    __slots__ = ("n", )

    def __init__(self, n: int):
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"ScalableDim: expected a positive int, got {n!r}")
        self.n = n

    def __repr__(self) -> str:
        return f"ScalableDim({self.n})"

    def __eq__(self, other) -> bool:
        return isinstance(other, ScalableDim) and other.n == self.n

    def __hash__(self) -> int:
        return hash(("ScalableDim", self.n))


class VecTy(Ty):
    """``vector<D0xD1x...xT>``.

    ``dims`` entries are ints (static) or ``ScalableDim`` (``[N]``, parse-only:
    MLIR vectors have no dynamic dims). ``elem`` is the element Ty (a ScalarTy
    for everything the DSL constructs).
    """

    __slots__ = ("dims", "elem")

    def __init__(self, dims, elem: Ty):
        self.dims = tuple(dims)
        self.elem = elem

    def build(self, b):
        dims = []
        scalable = []
        for d in self.dims:
            if isinstance(d, ScalableDim):
                dims.append(d.n)
                scalable.append(True)
            elif isinstance(d, int) and not isinstance(d, bool):
                dims.append(d)
                scalable.append(False)
            else:
                raise TypeError(f"VecTy.build: bad dimension {d!r}")
        return b.get_vector_type(dims, scalable, self.elem.build(b))

    @property
    def first_dim(self) -> int:
        d = self.dims[0]
        if not isinstance(d, int):
            raise ValueError(f"Cannot extract a static size from {self!r}")
        return d

    def __repr__(self) -> str:
        return f"VecTy({self.dims!r}, {self.elem!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, VecTy) and other.dims == self.dims and other.elem == self.elem

    def __hash__(self) -> int:
        return hash(("VecTy", self.dims, self.elem))


class TensorTy(Ty):
    """``tensor<D0xD1x...xT>``; dims entries are ints (static) or None (``?``)."""

    __slots__ = ("dims", "elem")

    def __init__(self, dims, elem: Ty):
        self.dims = tuple(dims)
        self.elem = elem

    def build(self, b):
        return b.get_tensor_type(list(self.dims), self.elem.build(b))

    def __repr__(self) -> str:
        return f"TensorTy({self.dims!r}, {self.elem!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, TensorTy) and other.dims == self.dims and other.elem == self.elem

    def __hash__(self) -> int:
        return hash(("TensorTy", self.dims, self.elem))


class StridedLayout:
    """``strided<[S0, S1, ...]>`` layout metadata.

    ``strides`` entries are ints (static) or None (``?``). ``has_offset`` is
    True for a dynamic offset (the only form codegen constructs; MLIR treats
    "no offset clause" identically) or an int for a static offset.
    """

    __slots__ = ("strides", "has_offset")

    def __init__(self, strides, has_offset=False):
        self.strides = tuple(strides)
        self.has_offset = has_offset

    def __eq__(self, other) -> bool:
        return (isinstance(other, StridedLayout) and other.strides == self.strides
                and other.has_offset == self.has_offset)

    def __hash__(self) -> int:
        return hash(("StridedLayout", self.strides, self.has_offset))

    def __repr__(self) -> str:
        return f"StridedLayout({self.strides!r}, has_offset={self.has_offset!r})"


class _BridgeSpace:
    """Sentinel: resolve the tt.ptr<->memref bridge space at build time.

    ``MemTy.space`` carries this sentinel instead of any attr text; build()
    swaps it for the Attribute returned by the C++ binding
    ``builder.get_bridge_memory_space()`` (MemorySpaceUtils.h
    getDefaultBridgeMemorySpace), keeping the C++ header the single source of
    truth for the bridge space. A stale libtriton without the binding fails
    loudly at build time — never a silent hardcoded fallback, which would mask
    exactly the version skew that breaks DSLRegionOpPattern's operand trace
    (illegal cross-space memref.cast, the MR20 regression class).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<bridge memory space>"


BRIDGE = _BridgeSpace()


class MemTy(Ty):
    """``memref`` type.

    ``dims`` is None for the unranked form (``memref<*xT[, space]>``, the raw
    pointer-param annotation) or a tuple of ints/None (``?``) for ranked forms.
    ``layout`` is a StridedLayout or None; ``space`` is the ``BRIDGE``
    sentinel, an MLIR Attribute, or None.
    """

    __slots__ = ("dims", "elem", "layout", "space")

    def __init__(self, dims, elem: Ty, layout: StridedLayout | None = None, space=None):
        self.dims = None if dims is None else tuple(dims)
        self.elem = elem
        self.layout = layout
        self.space = space

    @property
    def unranked(self) -> bool:
        return self.dims is None

    def _build_space(self, b):
        if self.space is None:
            return None
        if self.space is BRIDGE:
            return b.get_bridge_memory_space()
        return self.space  # already an Attribute object

    def build(self, b):
        elem = self.elem.build(b)
        space = self._build_space(b)
        if self.dims is None:
            return b.get_unranked_memref_type(elem, space)
        dims = [None if d is None else int(d) for d in self.dims]
        strides = None
        offset = None
        if self.layout is not None:
            strides = [None if s is None else int(s) for s in self.layout.strides]
            ho = self.layout.has_offset
            if isinstance(ho, int) and not isinstance(ho, bool):
                offset = ho  # static offset
            elif ho is True:
                offset = None  # C++ contract: None -> ShapedType::kDynamic ("offset: ?")
            else:
                offset = 0  # has_offset False == no offset clause == offset 0
        return b.get_memref_type(dims, elem, strides, offset, space)

    def as_dynamic_ranked(self) -> "MemTy":
        """The _ranked_cast transform: memref<*xT, ...> -> memref<?xT, ...>."""
        if not self.unranked:
            raise ValueError(f"as_dynamic_ranked on a ranked type {self!r}")
        return MemTy((None, ), self.elem, self.layout, self.space)

    def __repr__(self) -> str:
        return (f"MemTy({self.dims!r}, {self.elem!r}, "
                f"layout={self.layout!r}, space={self.space!r})")

    def __eq__(self, other) -> bool:
        return (isinstance(other, MemTy) and other.dims == self.dims and other.elem == self.elem
                and other.layout == self.layout and other.space is self.space)

    def __hash__(self) -> int:
        return hash(("MemTy", self.dims, self.elem, self.layout, id(self.space)))


class OpaqueTy(Ty):
    """User-literal MLIR type text carried through uninterpreted.

    Only for the LLVM-direct channel, where kernel authors pass type strings
    verbatim ("!llvm.ptr", "!llvm.struct<...>", "()" for void). build() hands
    the text to the official MLIR parser (``builder.parse_type``) — Python
    never parses it. Never constructed by codegen.
    """

    __slots__ = ("text", )

    def __init__(self, text: str):
        self.text = text

    def build(self, b):
        t = b.parse_type(self.text)
        if t is None:
            raise ValueError(f"OpaqueTy.build: MLIR parser rejected type text {self.text!r}")
        return t

    def __repr__(self) -> str:
        return f"OpaqueTy({self.text!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, OpaqueTy) and other.text == self.text

    def __hash__(self) -> int:
        return hash(("OpaqueTy", self.text))


# Frequently used singletons.
INDEX = ScalarTy("index")
I1 = ScalarTy("i1")
I8 = ScalarTy("i8")
I16 = ScalarTy("i16")
I32 = ScalarTy("i32")
I64 = ScalarTy("i64")
F16 = ScalarTy("f16")
BF16 = ScalarTy("bf16")
F32 = ScalarTy("f32")
F64 = ScalarTy("f64")
LLVM_PTR = OpaqueTy("!llvm.ptr")
LLVM_DESC = OpaqueTy("!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>")
VOID = OpaqueTy("()")  # compared/returned only; never built (no MLIR parse for "()")

# ---------------------------------------------------------------------------
# classify_type — re-derive the Ty hierarchy from an MLIR Type that the
# official C++ parser produced (builder.parse_type of an LLVM-channel user
# literal). Uses the structured introspection bindings; the OpaqueTy fallback
# keeps the official MLIR printer text for passthrough types ("!llvm.*").
# ---------------------------------------------------------------------------


def classify_type(b, T) -> Ty:
    """Structured Ty for a parsed MLIR Type (b = builder, T = pybind Type)."""
    if b.type_is_vector(T):
        dims = tuple(
            ScalableDim(d) if sc else d for d, sc in zip(b.type_vector_dims(T), b.type_vector_scalable_dims(T)))
        return VecTy(dims, classify_type(b, b.type_vector_elem(T)))
    if b.type_is_index(T):
        return INDEX
    w = b.type_int_width(T)
    if w:
        return ScalarTy(f"i{w}")
    fi = b.type_float_info(T)
    if fi is not None:
        width, is_bf = fi
        return ScalarTy(("bf" if is_bf else "f") + str(width))
    return OpaqueTy(str(T))


# ---------------------------------------------------------------------------
# Parameter annotations (document-facing sugar, feishu 3.3):
# sr.mem(f16) / sr.mem(f32, out=True) / sr.index.
# ---------------------------------------------------------------------------


class _TypedAnnotation:
    """Parameter annotation carrying a structured Ty.

    ``kind`` is fixed once, at construction: "mem" for MemTy pointer params,
    "scalar" for ScalarTy. Consumers must branch on ``kind`` instead of
    sniffing types; ``ty.build(builder)`` materialises the MLIR type.
    """

    def __init__(self, ty: Ty, writable: bool):
        if isinstance(ty, MemTy):
            kind = "mem"
        elif isinstance(ty, ScalarTy):
            kind = "scalar"
        else:
            raise TypeError(f"annotation type must be MemTy or ScalarTy, got {ty!r}")
        self.ty = ty
        self.writable = writable
        self.kind = kind

    def __repr__(self) -> str:
        return f"<{'rw' if self.writable else 'ro'} {self.ty!r}>"


def mem(dtype: str, out: bool = False) -> _TypedAnnotation:
    """Pointer-parameter annotation for a raw kernel.

    sr.mem(f16)           -> read-only  ``memref<*xf16, <bridge space>>``
    sr.mem(f32, out=True) -> writable   ``memref<*xf32, <bridge space>>``

    The memory space is the ``BRIDGE`` sentinel, resolved at build time by
    ``builder.get_bridge_memory_space()`` (C++ getDefaultBridgeMemorySpace) so
    it always matches the tt.ptr <-> memref bridge space of the
    spine-triton-opt pipeline; a mismatch would make DSLRegionOpPattern's
    operand trace fail to match the raw body's block-arg types and
    InlineSmtRvisaRegion emit an illegal cross-space memref.cast.
    """
    if not isinstance(dtype, str) or dtype == "index" or not _is_scalar_name(dtype):
        raise ValueError(f"mem: dtype must be a scalar element type (iN/fN/bfN), got {dtype!r}")
    ty = MemTy(None, ScalarTy(dtype), None, BRIDGE)
    return _TypedAnnotation(ty, writable=out)


# Scalar index parameter annotation, e.g.  K: sr.index
index = _TypedAnnotation(INDEX, writable=False)
