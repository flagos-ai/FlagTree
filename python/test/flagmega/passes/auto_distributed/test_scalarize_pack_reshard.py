# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Removing schedule Pack must also translate its real reshard contract."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


@pytest.mark.parametrize("adapter", ["distributed.sharded_view", "distributed.boxing"])
@pytest.mark.parametrize("axis", [0, 1])
def test_pack_reshard_keeps_communication_and_restores_scalar_coordinates(adapter, axis):
    mesh = fm.Placement((2, 2), "yx", "bb")
    rhs_tensor = fm.tensor_type("bfloat16", (32, 32))
    policies = [fm.SBP.broadcast()] * 2
    policies[axis] = fm.SBP.split_contiguous((0, 1), 8)
    rhs_type = fm.DistributedType(rhs_tensor, tuple(policies), mesh)
    lhs_type = fm.DistributedType(fm.tensor_type("bfloat16", (1, 32)), (fm.SBP.broadcast(),) * 2, mesh)
    b = fm.IRBuilder(dialect="ntt", stage="norm_bindings_finalized")
    lhs = b.var("lhs", lhs_type, id="lhs")
    rhs = b.var("rhs", rhs_type, id="rhs")
    metadata = {"vectorization_root": "result", "vectorization_internal": True, "vectorization_role": "pack"}
    pack_type = fm.get_definition("tensors.pack").infer_type((rhs,), {"lanes": (4,), "axes": (0,)})
    pack = b.call("tensors.pack", (rhs,), pack_type, id="pack", attrs={"lanes": (4,), "axes": (0,)}, metadata=metadata)
    broadcast = fm.DistributedType(pack_type.tensor, (fm.SBP.broadcast(),) * 2, mesh)
    bridge = b.call(adapter, (pack,), broadcast, id="bridge", attrs={"new_type": broadcast})
    semantic = {"vectorized_from": "math.matmul", "vectorization_attrs": {"transpose_a": False, "transpose_b": True},
                "vectorization_candidate": "vectorization.matmul.n", "vector_axes": (1,), "vector_lanes": (4,)}
    compute_type = fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", 4), (1, 8)), (fm.SBP.broadcast(),) * 2, mesh)
    compute = b.call("math.vectorized_matmul", (lhs, bridge), compute_type, id="compute", attrs={
        "lhs_axes": (), "rhs_axes": (0,), "output_axes": (1,), "output_lanes": (4,),
        "transpose_a": False, "transpose_b": True}, metadata={**metadata, **semantic, "vectorization_role": "compute"})
    result = b.call("tensors.unpack", (compute,), lhs_type, id="result", attrs={"axes": (1,)}, metadata=semantic)
    b.function("main", (lhs, rhs), (result,))
    original = fm.verify_module(b.build(entry="main"))
    lowered = fm.verify_module(lower_vectorization_contracts(original))
    assert "pack" not in lowered.node_map
    lowered_bridge = lowered.node_map["bridge"]
    assert lowered_bridge.op == adapter and lowered_bridge.inputs == ("rhs",)
    assert lowered_bridge.type.tensor == rhs_tensor
    assert lowered.node_map["result"].op == "math.matmul"
    assert lowered.node_map["result"].inputs == ("lhs", "bridge")
    torch.manual_seed(13)
    inputs = {"lhs": torch.randn(1, 32).bfloat16(), "rhs": torch.randn(32, 32).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(lowered, inputs), evaluator.run(original, inputs), rtol=0, atol=0)
