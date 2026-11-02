# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A residual add whose operand pack only rebuilds an already packed value.

Folding operand Packs into VectorizedBinary left such adds reading a function
parameter's boundary restore (Unpack / lane Bitcast) and an unpacked
projection. Their operand types then no longer matched the add, so
FormAddNormStats stopped fusing the residual add with the next norm's
statistics on every target.
"""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.binary import binary_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.producer_demand import final_axis_pack_demand

PACK = {"lanes": (8, ), "axes": (1, )}
VECTOR = fm.tensor_type(fm.vector_type("bfloat16", (8, )), (1, 4))


def _module(view, packing=PACK):

    class Graph(fm.Module):

        def forward(self):
            hidden = self.input("hidden", VECTOR, id="hidden")
            update = self.input("update", fm.tensor_type("bfloat16", (1, 32)), id="update")
            if view == "unpack":
                restored = fm.F.tensors.unpack(hidden, axes=(1, ), name="restore")
            else:
                restored = fm.F.tensors.bitcast(hidden, "bfloat16", name="restore")
            output = fm.F.math.vectorized_binary(restored, update, binary_op="add", lhs_pack=packing,
                                                 rhs_pack=packing, name="residual")
            self.function("main", (hidden, update), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _run(module, hidden, update):
    return TorchEvaluator(DictWeightResolver({})).run(module, {"hidden": hidden, "update": update})[0]


@pytest.mark.parametrize("view", ("unpack", "bitcast"))
def test_add_reads_the_packed_value_instead_of_repacking_its_restore(view):
    module = _module(view)
    rewritten = DataflowPass("restore", binary_propagation_rules()).run(module)
    residual = rewritten.node_map["residual"]
    assert residual.inputs[0] == "hidden" and "lhs_pack" not in residual.attrs
    assert residual.attrs["rhs_pack"] == PACK and "restore" not in rewritten.node_map
    hidden = (torch.arange(32).float().reshape(1, 4, 8) / 16).bfloat16()
    update = (torch.arange(32).float().reshape(1, 32) % 5 / 8).bfloat16()
    torch.testing.assert_close(_run(rewritten, hidden, update), _run(module, hidden, update), rtol=0, atol=0)


def test_a_pack_that_does_not_rebuild_the_value_is_kept():
    module = _module("unpack", {"lanes": (2, 4), "axes": (1, 1)})
    rewritten = DataflowPass("restore", binary_propagation_rules()).run(module)
    assert rewritten.node_map["residual"].inputs[0] == "restore"


def test_an_operand_pack_is_a_producer_lane_demand():
    module = _module("unpack")
    assert final_axis_pack_demand(module, {"update"}) == (8, )
    assert final_axis_pack_demand(module, {"hidden"}) is None


@pytest.mark.parametrize("target", ("amd-gfx1100", "amd-gfx1201", "nvidia-sm90"))
def test_qwen3_residual_adds_form_norm_stats_combines(target):
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.importer import VLLM_INDUCTOR_LEVEL3, apply_numerical_profile, import_qwen3_model
    from triton.flagmega.options import CompileOptions
    from python.test.flagmega.qwen3.helpers import full_checkpoint

    module = apply_numerical_profile(import_qwen3_model(full_checkpoint(num_hidden_layers=2)), VLLM_INDUCTOR_LEVEL3)
    packed = Compiler(CompileOptions(target=target)).compile(module, stop_after="AutoPackingPass").module
    # Both residual adds of the reusable decode layer combine with the next norm's statistics.
    assert sum(node.op == "ntt.add_norm_stats" for node in packed.nodes) == 2
    assert not any(node.op == "nn.norm_stats" and packed.node_map[node.inputs[0]].op == "math.vectorized_binary"
                   for node in packed.nodes)
