# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner sums preserve one result rounding and all observable graph edges."""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.promote_partial_matmul import promote_partial_matmul
from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_partial_accumulator_precision import matmul_module


def test_promotion_preserves_output_packets_and_closed_function_abi():
    module = matmul_module(False)
    before = module.node_map["dot"].type
    result = promote_partial_matmul(module)
    after = result.node_map["dot"].type
    assert after.partial == before.partial
    assert after.axis_policies == before.axis_policies
    assert after.tensor.dtype.elem_type == fm.DType.FLOAT32
    assert result.node_map["reduce"].type == module.node_map["reduce"].type
    assert result.functions == module.functions
    assert not any(node.op == "tensors.pack" for node in result.nodes)
    assert promote_partial_matmul(result) is result


def test_each_shared_materialization_rounds_its_own_result():
    module = matmul_module(False)
    original = module.node_map["reduce"]
    second = replace(original, id="second_materialization")
    entry = module.function_map[module.entry]
    module = fm.verify_module(replace(module, nodes=(*module.nodes, second), functions=(
        replace(entry, outputs=(*entry.outputs, second.id)),
    )))
    result = promote_partial_matmul(module)
    assert result.node_map[original.id].op == "ntt.vectorized_cast"
    assert result.node_map[second.id].op == "ntt.vectorized_cast"
    assert result.node_map[original.id].inputs != result.node_map[second.id].inputs
    assert result.functions == module.functions


def test_partial_function_output_is_not_retyped_behind_callers():
    module = matmul_module(False)
    entry = module.function_map[module.entry]
    module = fm.verify_module(replace(module, functions=(replace(entry, outputs=(*entry.outputs, "dot")),)))
    assert promote_partial_matmul(module) is module


def test_already_wide_partial_remains_unchanged():
    module = matmul_module(True)
    assert promote_partial_matmul(module) is module
