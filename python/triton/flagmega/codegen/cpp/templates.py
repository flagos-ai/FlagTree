# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic Jinja template lookup for C++ kernel implementations.

The resolution rules match ``codegen/triton/templates.py`` exactly -- same
``kernel/variant`` identity, same platform/architecture specialization
fallback -- so a kernel implemented for both backends is named the same way in
both trees. Only the file extension and template root differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from triton.flagmega.codegen.cpp.expressions import cpp_scalar_type, translate
from triton.flagmega.errors import CodegenError


_SNAKE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class CppKernelTemplateSpec:
    """Template identity derived from a selected TIR candidate."""

    kernel: str
    variant: str
    platform: str | None = None
    architecture: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("kernel", "variant", "platform", "architecture"):
            value = getattr(self, field_name)
            if value is not None and not _SNAKE_NAME.fullmatch(value):
                raise CodegenError(
                    f"C++ template {field_name} must be snake_case, got {value!r}."
                )
        if self.architecture is not None and self.platform is None:
            raise CodegenError("A C++ architecture specialization requires a platform.")


class CppTemplateRegistry:
    """Resolve per-kernel C++ implementations with the Triton fallback rules."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = (
            Path(root) if root is not None else Path(__file__).parent / "templates"
        )
        self.environment = Environment(
            loader=FileSystemLoader(str(self.root)),
            undefined=StrictUndefined,
            extensions=("jinja2.ext.do",),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )
        # `cppexpr` is how templates consume the renderer's index/predicate
        # algebra: the strings are Python expressions, and the filter is the
        # only sanctioned path that turns them into C++. A template that
        # interpolates one directly would emit Python `//` into C++.
        self.environment.filters["cppexpr"] = translate
        self.environment.filters["cpptype"] = cpp_scalar_type
        self.environment.filters["cppargs"] = _argument_list
        self.environment.filters["cppfloat"] = _float_literal

    def candidates(self, spec: CppKernelTemplateSpec) -> tuple[str, ...]:
        spec = CppKernelTemplateSpec(
            spec.kernel,
            spec.variant,
            spec.platform,
            spec.architecture,
        )
        base = f"kernels/{spec.kernel}"
        values: list[str] = []
        if spec.platform is not None and spec.architecture is not None:
            values.append(
                f"{base}/platforms/{spec.platform}/{spec.architecture}/{spec.variant}.cpp.jinja"
            )
        if spec.platform is not None:
            values.append(f"{base}/platforms/{spec.platform}/{spec.variant}.cpp.jinja")
        values.append(f"{base}/{spec.variant}.cpp.jinja")
        return tuple(values)

    def available(self, spec: CppKernelTemplateSpec) -> bool:
        return any(
            (self.root / relative).is_file() for relative in self.candidates(spec)
        )

    def resolve(self, spec: CppKernelTemplateSpec) -> str:
        candidates = self.candidates(spec)
        for relative in candidates:
            if (self.root / relative).is_file():
                return relative
        raise CodegenError(
            f"No C++ template implements {spec.kernel}/{spec.variant}; "
            f"searched {list(candidates)}. Either add the template under "
            f"{self.root}, or compile this model with the Triton backend."
        )

    def render(self, template_name: str, context: Mapping[str, object]) -> str:
        try:
            return self.environment.get_template(template_name).render(**context)
        except TemplateError as error:
            raise CodegenError(
                f"Failed to render C++ template {template_name}: {error}"
            ) from error


def _argument_list(value: object) -> str:
    """Translate a comma-separated renderer argument list into C++.

    A call event's ``arguments`` is one string holding several expressions, so
    translating it as a single expression would fail on the commas. Split on
    top-level commas only: an argument can itself contain commas inside
    parentheses or brackets.
    """

    text = str(value)
    arguments: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    arguments.append(text[start:])
    return ", ".join(
        translate(argument) for argument in arguments if argument.strip()
    )


def _float_literal(value: object) -> str:
    """Spell a renderer constant as a C++ float literal.

    Pad values and reduction neutrals arrive as Python numbers (``0``,
    ``-inf``), and ``0`` would otherwise emit as an int and change the type of
    the conditional expression it sits in.
    """

    number = float(value)
    if number != number:
        return "__builtin_nanf(\"\")"
    if number == float("inf"):
        return "__builtin_inff()"
    if number == float("-inf"):
        return "(-__builtin_inff())"
    return f"{number!r}f"


__all__ = ["CppKernelTemplateSpec", "CppTemplateRegistry"]
