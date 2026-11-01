# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _add_module(shape):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", shape)
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_exact_binary_materializes_specialized_vector_op_and_lowers_one_kernel():
    module = _add_module((1, 16))
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module
    assert [node.op for node in vectorized.nodes[-2:]] == ["math.vectorized_binary", "tensors.unpack"]
    assert not any(node.op == 'tensors.pack' for node in vectorized.nodes)
    assert vectorized.node_map['output.vectorized.compute'].attrs['lhs_pack']['lanes'] == (8,)
    assert vectorized.node_map["output.vectorized.compute"].type.dtype == fm.vector_type("bfloat16", (8,))

    lhs = torch.randn(1, 16, dtype=torch.bfloat16)
    rhs = torch.randn(1, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(vectorized, {"lhs": lhs, "rhs": rhs})[0],
        evaluator.run(module, {"lhs": lhs, "rhs": rhs})[0],
    )

    lowered = Compiler().compile(module).module
    kernels = [
        fm.kernel_dispatch_for_call(lowered, node)
        for node in lowered.nodes
        if fm.kernel_dispatch_for_call(lowered, node) is not None
    ]
    # Complete canonical inputs are read through proven chip-local views.
    # The vectorized arithmetic owns lane addressing without staging copies.
    compute = [
        kernel
        for kernel in kernels
        if kernel.semantic_op == "math.vectorized_binary"
    ]
    assert len(compute) == 1
    assert sum(kernel.semantic_op == "distributed.boxing" for kernel in kernels) == 0
    assert compute[0].parameters["vector_schedule"]["contract"]["lanes"] == (8,)
    assert compute[0].parameters["vector_schedule"]["physical"] == {
        "elements_per_program": 256,
    }


def test_non_divisible_binary_owns_its_exact_load_and_store_tail():
    module = _add_module((3, 10))
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module
    ops = tuple(node.op for node in vectorized.nodes)
    assert ops.count("tensors.pad") == 0
    assert ops.count("tensors.pack") == 0
    assert ops.count('tensors.slice_to_shape') == 0
    output = vectorized.node_map['output']
    assert output.op == 'math.vectorized_binary'
    assert output.attrs['lhs_pack']['padding']['pad_end'] == (0, 6)
    assert output.attrs['result_unpack'] == {'axes': (1,), 'lanes': (8,), 'shape': (3, 10)}
    assert output.type == module.node_map['output'].type

    lhs = torch.randn(3, 10, dtype=torch.bfloat16)
    rhs = torch.randn(3, 10, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(vectorized, {"lhs": lhs, "rhs": rhs})[0],
        lhs + rhs,
    )
