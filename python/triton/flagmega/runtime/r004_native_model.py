# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Verify model native bindings before normal prepare loads any CUfunction."""
from triton.flagmega.errors import ArtifactError
from triton.flagmega.codegen.triton.package_plan import plain_package_value


def verified_model_programs(codegen, kernels, module):
    from triton.flagmega.codegen.triton.r004_native_model import PROGRAM_SCHEMA, PROGRAM_SCHEMA_V6, PROGRAM_SCHEMA_R031
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    from triton.flagmega.runtime.kernel_programs import _verify_source_program_literals
    expected = describe_tir_package(module)
    for key in ('kernel_program_schema', 'native_model_binding', 'program_execution', 'kernels'):
        if plain_package_value(codegen.get(key)) != plain_package_value(expected.get(key)):
            raise ArtifactError('Native model bindings differ from the typed caller/arena contract.')
    raw = codegen.get('kernels')
    if codegen.get('kernel_program_schema') not in (PROGRAM_SCHEMA, PROGRAM_SCHEMA_V6, PROGRAM_SCHEMA_R031) or len(raw) != len(kernels):
        raise ArtifactError('Native model entry multiplicity differs from its contract.')
    _verify_source_program_literals(codegen, raw, kernels)
    entries = {}
    for spec, kernel in zip(raw, kernels, strict=True):
        function = getattr(kernel, 'fn', kernel)
        symbol = spec['entry_symbol']
        if (getattr(function, '__name__', None) != symbol
                or entries.setdefault(symbol, kernel) is not kernel
                or plain_package_value(function.__globals__.get('FLAGMEGA_PROGRAM_EXECUTION'))
                != plain_package_value(codegen['program_execution'])):
            raise ArtifactError('Native model does not resolve the declared shared function and schedule.')
    return tuple({**spec, 'grid':tuple(spec['grid']),
                  'dynamic_argument_indices':tuple(spec['dynamic_argument_indices'])} for spec in raw)
