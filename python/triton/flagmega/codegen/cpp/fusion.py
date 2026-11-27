# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validated C++ register PreOps/PostOps and ordered numeric round trips."""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from triton.flagmega.codegen.cpp.expressions import register_cast, translate_register
from triton.flagmega.errors import CodegenError


_TARGET = re.compile(r'^(?:pre_[a-z][a-z0-9_]*_[0-9]+|base_result|post_result_[0-9]+)$')


@dataclass(frozen=True)
class RegisterProgram:
    lines: tuple[str, ...]
    value: str


def elementwise_program(call: Mapping[str, object]) -> RegisterProgram:
    """Translate a closed descriptor program; never interpolate Python lines."""
    if call.get('family') != 'elementwise' or call.get('variant') not in {
        'add', 'mul', 'div', 'sigmoid', 'silu', 'cast'
    }:
        raise CodegenError(f'C++ register fusion has no owner for {call.get("family")}/{call.get("variant")}.')
    lines = call.get('fusion_lines', ())
    if not isinstance(lines, (tuple, list)) or not lines:
        raise CodegenError('C++ register fusion requires ordered assignment lines.')
    variables = {'lhs_value'}
    if call.get('variant') in {'add', 'mul', 'div'}:
        variables.add('rhs_value')
    declared = set()
    emitted = []
    for line in lines:
        if not isinstance(line, str):
            raise CodegenError('C++ register fusion requires assignment strings.')
        try:
            tree = ast.parse(line.strip(), mode='exec')
        except SyntaxError as error:
            raise CodegenError(f'Invalid C++ register assignment {line!r}.') from error
        if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
                or len(tree.body[0].targets) != 1 or not isinstance(tree.body[0].targets[0], ast.Name)):
            raise CodegenError('C++ register fusion permits one simple assignment per line.')
        assignment = tree.body[0]
        target = assignment.targets[0].id
        if not _TARGET.fullmatch(target) or '__' in target or target in variables:
            raise CodegenError(f'Invalid or repeated C++ fusion register {target!r}.')
        expression = ast.get_source_segment(line.strip(), assignment.value)
        if expression is None:
            raise CodegenError('C++ fusion assignment has no expression.')
        emitted.append(f'const float {target} = {translate_register(expression, variables)};')
        variables.add(target)
        declared.add(target)
    value = call.get('fusion_value')
    if not isinstance(value, str) or value not in declared or 'base_result' not in declared:
        raise CodegenError('C++ fusion output must name a declared result register.')
    return RegisterProgram(tuple(emitted), value)


def ordered_casts(register: str, dtypes: Sequence[str]) -> str:
    """Preserve every BF16/FP32 addend conversion in its specified order."""
    if not isinstance(dtypes, (tuple, list)):
        raise CodegenError('C++ ordered casts require a dtype sequence.')
    if not isinstance(register, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', register) or '__' in register:
        raise CodegenError('C++ ordered casts require a safe register identifier.')
    expression = translate_register(register, {register})
    for dtype in dtypes:
        expression = register_cast(expression, dtype)
    return expression


def validate_call(call: Mapping[str, object]) -> None:
    """Check feature ownership before the renderer emits an implementation."""
    if call.get('fusion_lines'):
        elementwise_program(call)
    elif call.get('fusion_value'):
        raise CodegenError('C++ fusion output has no register program.')
    for key, value in call.items():
        if key.startswith('fusion_') and key not in {'fusion_lines', 'fusion_value'} and value:
            raise CodegenError(f'C++ does not implement {key} for {call.get("family")}.')
    if call.get('addend_cast_types'):
        if (call.get('family'), call.get('variant')) != ('dense_matmul', 'packed_k_major_gemv_norm_stats'):
            raise CodegenError('C++ ordered addend casts have no implementing kernel owner.')
        ordered_casts('dense_residual_loaded', call['addend_cast_types'])
    if (call.get('family'), call.get('variant')) == ('dense_matmul', 'packed_k_major_gemv_norm_stats'):
        ordered_casts('dense_accumulator', (call.get('projection_type', 'tl.bfloat16'),))
