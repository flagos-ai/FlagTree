# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""KDA output layout propagation preserves the FP32 state and effect order."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.ir.ops.nn.test_kda_gated_recurrent_core import _inputs, _operand_types


class KdaConsumers(fm.Module):
    def __init__(self, lanes=(2, 2), *, scalar_users=True, output_lanes=(), gated=True):
        super().__init__(dialect="high_level", stage="packed", entry="main")
        self.lanes, self.scalar_users, self.output_lanes = lanes, scalar_users, output_lanes
        self.gated = gated

    def forward(self):
        types = _operand_types()
        names = ("q", "k", "v", "g", "decay_rate", "gate_bias", "beta", "state")
        if not self.gated:
            names = ("q", "k", "v", "decay", "beta", "state")
            types = (*types[:3], types[4], fm.tensor_type("float32", (1, 2)), types[-1])
        operands = tuple(self.input(name, kind, id=name) for name, kind in zip(names, types))
        operation = fm.F.nn.kda_gated_recurrent_core if self.gated else fm.F.nn.kda_recurrent_core
        attrs = {"lower_bound": -5.} if self.gated else {}
        if self.output_lanes:
            attrs["output_lanes"] = self.output_lanes
        pair = operation(*operands, **attrs, name="rec")
        output, state = fm.F.tensors.get_items(pair, 0, 1)
        packed = fm.F.tensors.pack(output, self.lanes, axes=(-1,) * len(self.lanes))
        stats = fm.F.nn.norm_stats(packed, axis=-1, use_mean=False)
        outputs = [packed, stats, state]
        if self.scalar_users:
            outputs.extend((output, pair))
        # A second step must read the first step's state exactly once.
        next_pair = operation(*operands[:-1], state, name="next_rec", **({"lower_bound": -5.} if self.gated else {}))
        outputs.append(next_pair)
        self.function("main", operands, tuple(outputs))


@pytest.mark.parametrize("lanes", [(2,), (2, 2)])
@pytest.mark.parametrize("gated", [True, False])
def test_kda_packed_output_keeps_state_users_and_recurrence_effect_order(lanes, gated):
    original = KdaConsumers(lanes, gated=gated).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    recurrences = [node for node in rewritten.nodes if node.op in {"nn.kda_gated_recurrent_core", "nn.kda_recurrent_core"}]
    assert len(recurrences) == 2
    first, second = recurrences
    assert first.attrs.get("output_lanes") == lanes
    assert first.type.fields[1] == original.node_map["rec"].type.fields[1]
    assert not second.attrs.get("output_lanes")
    assert [(node.op, node.effect) for node in rewritten.nodes if not node.effect.is_pure] == [
        (node.op, node.effect) for node in original.nodes if not node.effect.is_pure]
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    values = _inputs()
    if not gated:
        values = {**{name: values[name] for name in ("q", "k", "v", "state")},
                  "decay": -5 * torch.sigmoid(values["decay_rate"] * (values["g"].float() + values["gate_bias"])),
                  "beta": torch.sigmoid(values["beta"].float())}
    torch.testing.assert_close(evaluator.run(rewritten, values), evaluator.run(original, values), rtol=0, atol=0)
    assert post_function_boundary_pack_propagation(rewritten, NvidiaSm90Target()).semantic_hash == rewritten.semantic_hash


def test_kda_output_lanes_compose_with_existing_activation_lanes():
    original = KdaConsumers(lanes=(2,), output_lanes=(2,)).build()
    rewritten = post_function_boundary_pack_propagation(original, NvidiaSm90Target())
    first, second = [node for node in rewritten.nodes if node.op == "nn.kda_gated_recurrent_core"]
    assert first.attrs["output_lanes"] == (2, 2)
    assert first.type.fields[1] == second.type.fields[1]
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, _inputs()), evaluator.run(original, _inputs()), rtol=0, atol=0)
