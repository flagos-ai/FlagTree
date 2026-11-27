# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target


class _TupleProjection(fm.Module):
    def __init__(self, stage):
        super().__init__(dialect="ntt", stage=stage, entry="main")

    def forward(self):
        first = self.input("first", fm.tensor_type("float32", (2,)), id="first")
        second = self.input("second", fm.tensor_type("float32", (3,)), id="second")
        pair = fm.F.builtin.tuple(first, second, name="pair")
        result = fm.F.tensors.get_item(pair, 1, name="result")
        self.function("main", (first, second), (result,))


def test_auto_packing_dataflow_registers_neutral_tuple_fold():
    module = _TupleProjection("vectorized").build()

    packed = Compiler().compile(module, stop_after="apply-packing").module

    assert packed.function_map["main"].outputs == ("second",)
    assert not any(node.op in {"builtin.tuple", "builtin.get_item"} for node in packed.nodes)


def test_post_boundary_egraph_registers_neutral_tuple_fold():
    module = _TupleProjection("boundary_layout_propagated").build()

    rewritten = post_function_boundary_pack_propagation(
        module, NvidiaSm90Target()
    )

    assert rewritten.function_map["main"].outputs == ("second",)
    assert not any(node.op in {"builtin.tuple", "builtin.get_item"} for node in rewritten.nodes)
