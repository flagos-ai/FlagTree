# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Translate the renderer's index/predicate expression dialect into C++.

The TIR call renderers in ``codegen/triton`` compute every address, extent and
predicate as a string of integer algebra over dense local coordinates. That
algebra is buffer-plan logic, not Triton logic: the same offsets address the
same bytes whichever language loads them. This module reuses it rather than
re-deriving a parallel C++ copy that could drift.

The strings are valid Python expressions, so the translation is an AST walk,
not a regex rewrite. Two operators differ from C++ and are handled explicitly:

* ``//`` is Python floor division. Every operand the renderers produce is a
  non-negative index, where floor and truncating division agree, so it maps to
  ``/``. :func:`translate` rejects an expression whose operands it cannot
  prove non-negative rather than emitting a silently different value.
* ``&`` / ``|`` are Triton's elementwise boolean operators. On the scalar
  values a C++ kernel evaluates they are ``&&`` / ``||``.

Anything outside the grammar below raises, so a new renderer construct shows
up as a codegen error instead of wrong arithmetic.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping

from triton.flagmega.errors import CodegenError


# Scalar dtype spellings. Only the types the generated kernels actually load
# are listed; an unlisted dtype must raise rather than guess a C++ name.
_CPP_SCALAR_TYPES: Mapping[str, str] = {
    "bool": "bool",
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long long",
    "uint64": "unsigned long long",
    "float16": "_Float16",
    "bfloat16": "ftt::bfloat16_t",
    "float32": "float",
    "float64": "double",
}

# Triton's spelling of the same types, as they appear inside the pointer casts
# the renderers emit (``value.to(tl.pointer_type(tl.bfloat16))``).
_TRITON_TO_CPP_SCALAR = {
    "tl.int1": "bool",
    "tl.int8": "signed char",
    "tl.uint8": "unsigned char",
    "tl.int16": "short",
    "tl.uint16": "unsigned short",
    "tl.int32": "int",
    "tl.uint32": "unsigned int",
    "tl.int64": "long long",
    "tl.uint64": "unsigned long long",
    "tl.float16": "_Float16",
    "tl.bfloat16": "ftt::bfloat16_t",
    "tl.float32": "float",
    "tl.float64": "double",
}

# Triton scalar intrinsics that have a direct FTT or libm equivalent. The
# values are C++ callables taking the same argument count.
_CALL_TRANSLATIONS: Mapping[str, str] = {
    "tl.where": "",  # handled structurally as a conditional operator
    "tl.maximum": "ftt::max",
    "tl.minimum": "ftt::min",
    "tl.abs": "ftt::abs",
    "tl.exp": "ftt::exp",
    "tl.log": "ftt::log",
    "tl.sqrt": "ftt::sqrt",
    "tl.rsqrt": "ftt::rsqrt",
    "tl.sigmoid": "ftt::sigmoid",
    "tl.cdiv": "ftt::cdiv",
    "libdevice.llrint": "ftt::llrint",
}

_BINARY_OPERATORS: Mapping[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Mod: "%",
    ast.BitAnd: "&&",
    ast.BitOr: "||",
    ast.BitXor: "!=",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_COMPARISONS: Mapping[type, str] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def cpp_scalar_type(dtype: str) -> str:
    """Return the C++ spelling of one serialized FlagMega scalar dtype."""

    try:
        return _CPP_SCALAR_TYPES[str(dtype)]
    except KeyError as error:
        raise CodegenError(
            f"FTT has no C++ scalar spelling for dtype {dtype!r}."
        ) from error


def translate(expression: object) -> str:
    """Translate one renderer expression string into a C++ expression."""

    # The renderers emit multi-line expressions and indented argument lists, so
    # strip surrounding whitespace before parsing; leading indentation is an
    # `IndentationError` in `eval` mode.
    text = str(expression).strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise CodegenError(
            f"Cannot parse renderer expression for C++ codegen: {text!r} ({error})."
        ) from error
    return _Translator(text).visit(tree.body)


class _Translator:
    """Recursive-descent emitter over the renderer's expression grammar."""

    def __init__(self, source: str) -> None:
        self.source = source

    def reject(self, node: ast.AST, reason: str):
        raise CodegenError(
            f"C++ codegen cannot translate {reason} in renderer expression "
            f"{self.source!r}."
        )

    def visit(self, node: ast.AST) -> str:
        handler = getattr(self, f"_{type(node).__name__}", None)
        if handler is None:
            self.reject(node, f"a {type(node).__name__} node")
        return handler(node)

    def _dotted_name(self, node: ast.AST) -> str:
        """Spell a dotted reference (``tl.where``) back as written."""

        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            self.reject(node, "a computed attribute reference")
        parts.append(current.id)
        return ".".join(reversed(parts))

    def _Constant(self, node: ast.Constant) -> str:
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return f"{value}LL"
        if isinstance(value, float):
            # Renderer float literals are mask fill values and scale factors;
            # emit them as float to match the kernels' accumulator type.
            return f"{value!r}f"
        if value is None:
            self.reject(node, "a None literal")
        self.reject(node, f"a {type(value).__name__} literal")

    def _Name(self, node: ast.Name) -> str:
        return node.id

    def _Attribute(self, node: ast.Attribute) -> str:
        # Dotted names reaching here are type spellings (``tl.bfloat16``);
        # pointer casts consume them in _Call before this point.
        dotted = self._dotted_name(node)
        if dotted in _TRITON_TO_CPP_SCALAR:
            return _TRITON_TO_CPP_SCALAR[dotted]
        self.reject(node, f"attribute reference {dotted!r}")

    def _UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Invert):
            return f"(~{operand})"
        if isinstance(node.op, ast.Not):
            return f"(!{operand})"
        self.reject(node, f"unary {type(node.op).__name__}")

    def _BoolOp(self, node: ast.BoolOp) -> str:
        operator = "&&" if isinstance(node.op, ast.And) else "||"
        joined = f" {operator} ".join(self.visit(value) for value in node.values)
        return f"({joined})"

    def _BinOp(self, node: ast.BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.FloorDiv):
            # Floor and truncating division agree only for non-negative
            # operands. The renderers derive every divisor from a positive
            # extent and every dividend from a dense coordinate, so this
            # holds -- but prove it per expression instead of assuming it.
            self._require_nonnegative(node.left)
            self._require_nonnegative(node.right)
            return f"({left} / {right})"
        if isinstance(node.op, ast.Mod):
            self._require_nonnegative(node.left)
            self._require_nonnegative(node.right)
            return f"({left} % {right})"
        if isinstance(node.op, ast.Div):
            return f"((float)({left}) / (float)({right}))"
        try:
            operator = _BINARY_OPERATORS[type(node.op)]
        except KeyError:
            self.reject(node, f"binary {type(node.op).__name__}")
        return f"({left} {operator} {right})"

    def _Compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            self.reject(node, "a chained comparison")
        try:
            operator = _COMPARISONS[type(node.ops[0])]
        except KeyError:
            self.reject(node, f"comparison {type(node.ops[0]).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        return f"({left} {operator} {right})"

    def _IfExp(self, node: ast.IfExp) -> str:
        return (
            f"({self.visit(node.test)} ? {self.visit(node.body)}"
            f" : {self.visit(node.orelse)})"
        )

    def _Subscript(self, node: ast.Subscript) -> str:
        """Drop a Triton broadcast subscript.

        The renderers write 2-D tile addresses as ``row[:, None]`` /
        ``col[None, :]`` so Triton broadcasts two 1-D index vectors into a
        matrix. A C++ kernel evaluates one scalar element at a time, where both
        operands are already scalars and the broadcast is the identity.

        Only a pure broadcast is dropped. A subscript that actually selects an
        element (``table[i]``) is a different operation and must not be
        silently erased, so anything other than slices and ``None`` raises.
        """

        index = node.slice
        items = index.elts if isinstance(index, ast.Tuple) else [index]
        for item in items:
            if isinstance(item, ast.Slice):
                if item.lower is not None or item.upper is not None or item.step is not None:
                    self.reject(node, "a bounded slice")
                continue
            if isinstance(item, ast.Constant) and item.value is None:
                continue
            self.reject(node, "a subscript that is not a broadcast")
        return self.visit(node.value)

    def _Call(self, node: ast.Call) -> str:
        if node.keywords:
            self.reject(node, "a call with keyword arguments")
        function = node.func
        # ``pointer.to(tl.pointer_type(tl.bfloat16))`` retypes a byte pool.
        if isinstance(function, ast.Attribute) and function.attr == "to":
            return self._pointer_cast(node)
        dotted = self._dotted_name(function)
        if dotted == "tl.where":
            if len(node.args) != 3:
                self.reject(node, "tl.where with other than three arguments")
            condition, true_value, false_value = (
                self.visit(value) for value in node.args
            )
            return f"({condition} ? {true_value} : {false_value})"
        if dotted == "tl.full":
            # `tl.full((), value, dtype)` materializes a scalar as a rank-zero
            # tensor so a literal crossing a Triton function boundary stays
            # dynamic instead of specializing the callee. A C++ parameter is
            # already dynamic, so this is just the value.
            if len(node.args) != 3:
                self.reject(node, "tl.full with other than three arguments")
            return self.visit(node.args[1])
        if dotted == "_flagmega_block_scope_base":
            # Select this CTA's slice of a replicated byte pool. The Triton
            # module template defines the same helper; see `block_scope_base`
            # in the C++ entry template.
            if len(node.args) != 2:
                self.reject(node, "a block-scope base with other than two arguments")
            pool, scope_nbytes = (self.visit(value) for value in node.args)
            return f"ftt::runtime::block_scope_base({pool}, {scope_nbytes})"
        if dotted in _CALL_TRANSLATIONS:
            arguments = ", ".join(self.visit(value) for value in node.args)
            return f"{_CALL_TRANSLATIONS[dotted]}({arguments})"
        self.reject(node, f"call to {dotted!r}")

    def _pointer_cast(self, node: ast.Call) -> str:
        """Translate ``base.to(tl.pointer_type(T))`` into a C++ cast."""

        if len(node.args) != 1:
            self.reject(node, "a .to() cast with other than one argument")
        argument = node.args[0]
        if (
            not isinstance(argument, ast.Call)
            or self._dotted_name(argument.func) != "tl.pointer_type"
            or len(argument.args) != 1
        ):
            self.reject(node, "a .to() cast that is not a pointer retype")
        element = self._dotted_name(argument.args[0])
        try:
            spelling = _TRITON_TO_CPP_SCALAR[element]
        except KeyError:
            self.reject(node, f"pointer element type {element!r}")
        base = self.visit(node.func.value)
        return f"reinterpret_cast<{spelling} *>({base})"

    def _require_nonnegative(self, node: ast.AST) -> None:
        """Reject an operand of ``//``/``%`` that may be negative.

        Dense coordinates, extents and their sums/products are non-negative by
        construction. A subtraction or unary minus is not, and would make C++
        truncating division disagree with the Python floor division the
        renderer expression means.
        """

        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and node.value >= 0:
                return
            self.reject(node, "a negative or non-integer division operand")
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Sub,)):
                self.reject(node, "a subtraction under floor division")
            self._require_nonnegative(node.left)
            self._require_nonnegative(node.right)
            return
        if isinstance(node, ast.Call):
            # ``tl.where`` over dense coordinates stays non-negative; a
            # clamped index is exactly this shape.
            return
        if isinstance(node, ast.Subscript):
            # A broadcast subscript is transparent; recurse into its operand.
            self._require_nonnegative(node.value)
            return
        if isinstance(node, ast.UnaryOp):
            self.reject(node, "a unary-signed division operand")
        # Comparisons and boolean operators yield 0/1.
        if isinstance(node, (ast.Compare, ast.BoolOp, ast.IfExp)):
            return
        self.reject(node, "an unprovable division operand")


__all__ = ["cpp_scalar_type", "translate"]
