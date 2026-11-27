# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def _binary_module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="imported", entry="main"
            )

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (2, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            output = fm.F.math.add(lhs, rhs, name="output")
            self.function("main", (lhs, rhs), (output,))

    return Graph().build()


def test_auto_packing_does_not_erase_an_unrelated_typed_vector_expression():
    module = Compiler().compile(
        _binary_module(), stop_after="apply-packing"
    ).module

    compute = module.node_map["output.vectorized.compute"]
    assert compute.op == "math.vectorized_binary"
    assert isinstance(compute.type, fm.TensorType)
    assert compute.type.dtype == fm.vector_type("bfloat16", (8,))


def test_auto_distribution_preserves_vector_type_until_explicit_lowering_pass():
    distributed = Compiler().compile(
        _binary_module(), stop_after="auto-distributed"
    ).module

    compute = distributed.node_map["output.vectorized.compute"]
    assert compute.op == "math.vectorized_binary"
    assert isinstance(compute.type, fm.DistributedType)
    assert compute.type.tensor.dtype == fm.vector_type("bfloat16", (8,))
    assert isinstance(distributed.node_map["output"].type, fm.DistributedType)

    lowered = Compiler().compile(
        _binary_module(), stop_after="lower-vectorization-contracts"
    ).module
    output = lowered.node_map["output"]
    # nncase's post-boundary UnpackToBitcast canonicalization keeps the
    # physical vector producer and exposes its scalar result as a storage
    # view; explicit contract lowering must not scalarize that producer.
    assert output.op == "tensors.bitcast"
    assert isinstance(output.type, fm.DistributedType)
    assert output.type.tensor.dtype == fm.DType.BFLOAT16
    compute = lowered.node_map[output.inputs[0]]
    assert compute.op == "math.vectorized_binary"
    assert compute.type.tensor.dtype == fm.vector_type("bfloat16", (8,))
    assert compute.metadata["selected_vector_lanes"] == (8,)
