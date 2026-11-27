# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    propagate_function_boundary_layouts,
)


class _PackedIdentity(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        logical_type = fm.tensor_type("float32", (2, 16))
        parameter = self.input("parameter", logical_type, id="parameter")
        packed = fm.F.tensors.pack(
            parameter, (8,), axes=(1,), name="packed"
        )
        restored = fm.F.tensors.unpack(
            packed, axes=(1,), name="restored"
        )
        value = self.input("value", logical_type, id="value")
        call = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="identity",
            name="call",
        )
        self.function("main", (value,), (call,))
        self.function("identity", (parameter,), (restored,))


def test_pack_unpack_identity_becomes_a_typed_vector_function_boundary():
    original = _PackedIdentity().build()
    rewritten = propagate_function_boundary_layouts(original)

    identity = rewritten.function_map["identity"]
    assert identity.outputs == identity.parameters
    assert isinstance(
        rewritten.node_map[identity.parameters[0]].type.dtype,
        fm.VectorType,
    )
    assert sum(node.op == "tensors.pack" for node in rewritten.nodes) == 1
    assert sum(node.op == "tensors.unpack" for node in rewritten.nodes) == 1

    value = torch.randn(2, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
