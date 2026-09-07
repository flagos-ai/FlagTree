# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import ast

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package


@pytest.mark.parametrize("auxiliary", [False, True])
def test_entry_role_wrappers_are_inline_but_reusable_functions_are_not(compile_pipeline_module, auxiliary):
    implementation = "tir.dense_matmul.tensor_descriptor_smem_pipeline_" + (
        "aux_gemv" if auxiliary else "gemv")
    package = describe_tir_package(compile_pipeline_module(reusable=True, implementation=implementation))
    tree = ast.parse(render_tir_package(package, "unit"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def noinline(name):
        return any(
            isinstance(decorator, ast.Call)
            and any(keyword.arg == "noinline" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in decorator.keywords)
            for decorator in functions[name].decorator_list
        )

    roles = ["consumer", "producer", *(["auxiliary_consumer"] if auxiliary else [])]
    for role in roles:
        assert not noinline(f"flagmega_main__{role}"), "entry role is a scheduling wrapper, not a reusable function"
        assert noinline(f"_flagmega_function_worker__{role}")
