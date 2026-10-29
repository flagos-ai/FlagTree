# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""gfx1100: pin device-function pointer parameters to SGPRs.

gfx1100 megakernels keep ConvertToBufferOps on (the gfx1201 ones turn it off;
see ``runtime.nvidia_launch.compilation_knobs``). A noinline device function
receives its pointer parameters in VGPRs, so every buffer op whose base is such
a parameter is wrapped in a readfirstlane waterfall loop. Triton scalars are
uniform per program, so a ``v_readfirstlane`` of each pointer at function entry
is exact and lets the buffer ops use the SGPR base directly.
"""

from __future__ import annotations

import ast

HELPER = "_flagmega_amd_uniform_parameter"
# The RDNA GEMV templates compute every address in 64 bits, so no buffer op
# takes a parameter base there; pinning would only add SGPR pressure across
# their hot loops (measured slower on W7900).
SKIPPED_FAMILIES = frozenset({"block_fp8", "matmul_glu"})


def amd_uniform_parameters(platform: str | None, architecture: str | None) -> bool:
    return platform == "amd" and architecture == "gfx1100"


def _is_noinline_jit(decorator: ast.expr) -> bool:
    return (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "jit" and any(
                keyword.arg == "noinline" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in decorator.keywords))


def skipped_symbols(render_calls) -> frozenset[str]:
    return frozenset(str(call["symbol"]) for call in render_calls if call.get("family") in SKIPPED_FAMILIES)


def add_uniform_parameter_prologue(source: str, skip: frozenset[str] = frozenset()) -> str:
    """Prefix every ``@triton.jit(noinline=True)`` function body with one
    ``name = _flagmega_amd_uniform_parameter(name)`` per runtime parameter,
    except the device functions of the calls in ``skip``."""
    lines = source.split("\n")
    inserts = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or not any(map(_is_noinline_jit, node.decorator_list)):
            continue
        if any(node.name == symbol or node.name.startswith(symbol + "__") for symbol in skip):
            continue
        arguments = node.args
        defaulted = len(arguments.defaults)
        positional = arguments.posonlyargs + arguments.args
        names = [
            argument.arg for index, argument in enumerate(positional)
            if argument.annotation is None and index < len(positional) - defaulted
        ]
        if not names:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str) and len(node.body) > 1):
            first = node.body[1]
        indent = " " * first.col_offset
        inserts.append((first.lineno - 1, [f"{indent}{name} = {HELPER}({name})" for name in names]))
    for line, statements in sorted(inserts, reverse=True):
        lines[line:line] = statements
    return "\n".join(lines)
