# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from .helpers import scheduled_nested_module


def test_kernel_ops_are_not_executable_function_boundaries():
    module = scheduled_nested_module()
    assert not module.prim_functions
    assert module.kernel_definitions
    main = module.execution_function_map["main"]
    prepared, nested, output = main.body.fields
    assert isinstance(prepared, fm.KernelInvoke)
    assert isinstance(output, fm.KernelInvoke)
    assert isinstance(nested, fm.PrimFunctionCall)
    assert not isinstance(prepared, fm.PrimFunctionCall)
    assert nested.callee == "worker"
    assert all(isinstance(value.dispatch, fm.KernelDispatch) for value in module.kernel_definitions)
    assert all("body" not in value.to_data() for value in module.kernel_definitions)


def test_function_and_kernel_symbol_domains_cannot_be_interchanged():
    module = scheduled_nested_module()
    main = module.execution_function_map["main"]
    first, *rest = main.body.fields
    for invalid, message in (
        (replace(first, kernel="worker"), "missing kernel definition"),
        (fm.T.prim_function_call(first.call_id, first.kernel, arguments=first.arguments,
                                 results=first.results, reads=first.reads, writes=first.writes), "non-function kernel"),
    ):
        edited = replace(module, execution_functions=tuple(
            replace(function, body=fm.T.sequential((invalid, *rest))) if function.name == "main" else function
            for function in module.execution_functions
        ))
        with pytest.raises(IRVerificationError, match=message):
            fm.verify_module(edited)
