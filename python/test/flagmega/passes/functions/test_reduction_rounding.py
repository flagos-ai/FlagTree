# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A rounding boundary may move into its owner only with a complete proof."""

from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions.reduction_rounding import fuse_materialized_reduction_rounding


def graph(
    *,
    lanes=(8,),
    mode="partial",
    parts=4,
    chips=4,
    observer=None,
    hidden=128,
    block_shape=(2, 2),
    hidden_owners=False,
    producer=False,
):
    mesh = fm.Placement((chips, *block_shape), "cyx", "cbb")
    tokens, hc = 1, 4
    b = fm.SBP.broadcast()
    policies = (
        fm.SBP.split_contiguous((0,)) if mode == "partial" else b,
        b,
        fm.SBP.split_contiguous((0, 1, 2), 1)
        if mode == "gather"
        else fm.SBP.split_contiguous((1, 2), 4)
        if hidden_owners
        else b,
    )

    def dist(kind, policies=None):
        return fm.DistributedType(kind, policies or (b,) * kind.rank, mesh)

    class Graph(fm.Module):
        def forward(self):
            kind = fm.tensor_type(
                fm.vector_type("bfloat16", lanes) if lanes else "bfloat16", (parts, tokens, hidden // prod(lanes))
            )
            value = self.input("value", dist(kind, policies), id="value")
            residual = self.input("residual", dist(fm.tensor_type("float32", (tokens, hc, hidden))), id="residual")
            post = self.input("post", dist(fm.tensor_type("float32", (tokens, hc))), id="post")
            comb = self.input("comb", dist(fm.tensor_type("float32", (tokens, hc, hc))), id="comb")
            parameter = value
            if producer:
                value = (
                    fm.F.math.vectorized_binary(value, value, binary_op="add", name="added")
                    if lanes
                    else fm.F.math.add(value, value, name="added")
                )
            if lanes:
                wide = fm.F.ntt.vectorized_cast(
                    value, new_type=fm.vector_type("float32", (2, 4)), vectorize_axes=(-1,), name="wide"
                )
                scalar = fm.F.tensors.bitcast(wide, "float32", name="scalar")
            else:
                wide = scalar = fm.F.tensors.cast(value, "float32", name="wide")
            reduced = fm.F.math.reduce_sum(scalar, axes=(0,), keep_dims=False, name="sum")
            if mode != "full":
                reduced = fm.F.distributed.force_boxing(reduced, dist(reduced.type.tensor), name="materialized")
            narrow = fm.F.tensors.cast(reduced, "bfloat16", name="narrow")
            output = fm.F.nn.mhc_post(
                narrow,
                residual,
                post,
                comb,
                accumulation_mode="value_fma_first",
                residual_input_dtype="bfloat16",
                output_dtype="float32",
                name="output",
            )
            outputs = [output]
            if observer == "wide":
                outputs.append(wide)
            elif observer == "narrow":
                outputs.append(narrow)
            elif observer == "scalar_user":
                outputs.append(fm.F.math.add(narrow, narrow, name="scalar_user"))
            self.function("main", (parameter, residual, post, comb), tuple(outputs))

    return Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()


def values(module):
    generator = torch.Generator().manual_seed(20261018)
    result = {}
    for identity in module.function_map[module.entry].parameters:
        kind = fm.logical_type(module.node_map[identity].type)
        shape = tuple(d.fixed_value for d in kind.shape) + tuple(getattr(kind.dtype, "lanes", ()))
        dtype = getattr(kind.dtype, "elem_type", kind.dtype).value
        result[identity] = torch.randn(shape, generator=generator).to(getattr(torch, dtype))
    return result


@pytest.mark.parametrize("lanes,mode", [((), "full"), ((8,), "full"), ((2, 4), "partial"), ((8,), "gather")])
@pytest.mark.parametrize("producer", [False, True])
def test_rounding_fuses_into_reduce_and_materialization_without_compute_adapters(tmp_path, lanes, mode, producer):
    original = graph(lanes=lanes, mode=mode, producer=producer)
    assert sum(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in original.nodes) == 2
    result = fuse_materialized_reduction_rounding(original)
    assert not any(
        node.op in {"tensors.cast", "ntt.vectorized_cast", "tensors.pack", "tensors.unpack", "tensors.bitcast"}
        for node in result.nodes
    )
    assert result.function_map["main"].outputs == original.function_map["main"].outputs
    assert result.node_map["output"].type == original.node_map["output"].type
    assert len([node for node in result.nodes if node.op == "math.reduce_sum"]) == 1
    assert len(result.metadata["materialized_reduction_rounding"]) == 1
    assert fm.load_module(fm.emit_module(result, tmp_path / "result.py")).semantic_hash == result.semantic_hash
    assert fuse_materialized_reduction_rounding(result).semantic_hash == result.semantic_hash
    evaluator = TorchEvaluator(DictWeightResolver({}))
    inputs = values(original)
    for mutation in range(3):
        (a,) = evaluator.run(original, inputs)
        (b,) = evaluator.run(result, inputs)
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        inputs["value"].add_(1 / 128)


@pytest.mark.parametrize(
    "changes",
    [
        {"parts": 8},
        {"observer": "wide"},
        {"observer": "narrow"},
        {"observer": "scalar_user"},
    ],
)
def test_partial_fusion_rejects_early_rounding_or_an_unconverted_observer(changes):
    original = graph(**changes)
    assert fuse_materialized_reduction_rounding(original).semantic_hash == original.semantic_hash


def test_materialized_scalar_result_keeps_its_public_abi():
    original = graph(lanes=(), mode="full", observer="narrow")
    result = fuse_materialized_reduction_rounding(original)
    assert result.node_map["narrow"].type == original.node_map["narrow"].type
    assert result.function_map["main"].outputs == original.function_map["main"].outputs
    assert not any(node.op == "tensors.cast" for node in result.nodes)


def test_two_owner_local_values_are_a_real_rounding_counterexample():
    inputs = torch.tensor([[1.0, 1 / 256], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16)
    correct = inputs.float().sum(1).sum().bfloat16()
    early = inputs.float().sum(1).bfloat16().float().sum().bfloat16()
    assert correct.item() == 1 / 256 and early.item() == 0.0


def test_vector_contract_lowering_retains_the_new_physical_reduction_owner():
    from dataclasses import replace
    from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts

    original = graph(lanes=(8,), mode="partial")
    reduction = original.node_map["sum"]
    metadata = {
        **reduction.metadata,
        "vectorization_internal": True,
        "vectorization_role": "compute",
        "vectorization_semantic_id": "sum.semantic",
        "vectorization_root": "sum.semantic",
        "vectorization_candidate": "vectorization.sum.components",
        "vectorized_from": "math.reduce_sum",
        "vector_axes": (1,),
        "vector_lanes": (8,),
        "vectorization_attrs": {"axes": (0,), "keep_dims": False},
        "vectorization_inputs": ("scalar",),
    }
    original = fm.verify_module(
        replace(
            original,
            nodes=tuple(replace(node, metadata=metadata) if node.id == "sum" else node for node in original.nodes),
        )
    )
    fused = fuse_materialized_reduction_rounding(original)
    lowered = fm.verify_module(lower_vectorization_contracts(fused))
    physical = fused.node_map["narrow.rounded_sum"]
    assert lowered.node_map[physical.id].inputs == physical.inputs
    assert lowered.node_map[physical.id].type == physical.type
    assert not any(
        node.op in {"tensors.cast", "ntt.vectorized_cast", "tensors.pack", "tensors.unpack"} for node in lowered.nodes
    )
