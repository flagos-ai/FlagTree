# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A reshard that consumes a scalarized propagated reshape must scalarize too."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts


@pytest.mark.parametrize("adapter", ["distributed.sharded_view", "distributed.boxing"])
def test_reshard_after_scalarized_reshape_follows_its_scalar_source(adapter):
    placement = fm.Placement((2, 2), "yx", "bb")
    lanes = (8,)
    native_vector = fm.tensor_type(fm.vector_type("bfloat16", lanes), (1, 16))
    native_type = fm.DistributedType(
        native_vector, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 8)), placement
    )
    # A propagated reshape carries the packed physical shape; its logical
    # (scalar) shape lives in ``vectorization_attrs``.
    reshaped_vector = fm.tensor_type(fm.vector_type("bfloat16", lanes), (1, 8, 2))
    reshaped_type = fm.DistributedType(
        reshaped_vector,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 4), fm.SBP.broadcast()),
        placement,
    )
    view_type = fm.DistributedType(reshaped_vector, (fm.SBP.broadcast(),) * 3, placement)

    b = fm.IRBuilder(dialect="ntt", stage="add_norm_stats_lowered")
    source = b.var("source", native_type, id="source")
    native = b.call(
        "math.vectorized_unary", (source,), native_type, id="heads.vectorized.compute",
        attrs={"unary_op": "silu"},
        metadata={
            "vectorization_root": "heads.vectorized", "vectorization_role": "compute",
            "vectorization_internal": True, "vectorization_attrs": {},
            "vectorization_candidate": "vectorization.unary",
            "vectorized_from": "math.silu", "vector_axes": (1,), "vector_lanes": lanes,
        },
    )
    b.call(
        "tensors.reshape", (native,), reshaped_type, id="heads.propagated.reshape",
        attrs={"shape": (1, 8, 2)},
        metadata={
            "vectorization_root": "heads.vectorized",
            "vectorization_semantic_id": "heads",
            "vectorization_role": "propagated-reshape", "vectorization_internal": True,
            "vectorization_attrs": {"shape": (1, 8, 16)},
            "vectorization_candidate": "vectorization.unary",
            "vectorized_from": "tensors.reshape",
            "vector_axes": (2,), "vector_lanes": lanes,
        },
    )
    bridge = b.call(
        adapter, ("heads.propagated.reshape",), view_type, id="bridge",
        attrs={"new_type": view_type},
    )
    b.function("main", (source,), (bridge,))
    original = fm.verify_module(b.build(entry="main"))

    lowered = fm.verify_module(lower_vectorization_contracts(original))
    result = lowered.node_map["heads"]
    assert result.op == "tensors.reshape"
    assert result.type.tensor.dtype == fm.DType.BFLOAT16
    lowered_bridge = lowered.node_map["bridge"]
    assert lowered_bridge.op == adapter
    # The reshard must relate the same logical tensor type its source now has.
    assert lowered_bridge.type.tensor == lowered.node_map[lowered_bridge.inputs[0]].type.tensor

    inputs = {"source": torch.linspace(-3, 3, 128).reshape(1, 16, 8).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(lowered, inputs), evaluator.run(original, inputs), rtol=0, atol=0
    )
