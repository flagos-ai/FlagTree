# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.targets import NvidiaSm90Target


def _graph(*, key_dim=128, dtype_input=None, tokens=1, replicated=False):
    config = GatedDeltaNetStateConfig(1, 2, 4, key_dim, 16, 4, 32)
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = broadcast if replicated else fm.SBP.split_contiguous((0, 1))
    specs = {
        "qkv": (tokens, config.conv_dim), "z": (tokens, 64),
        "projection_input": (tokens, 32), "b_weight": (4, 32), "a_weight": (4, 32),
        "a_log": (4,), "dt_bias": (4,), "norm_weight": (16,),
    }
    builder = fm.IRBuilder(dialect="distributed", stage="frozen_constants")
    state = builder.var("state", config.ref_type, id="state")
    inputs = [state]
    for name, shape in specs.items():
        dtype = "float32" if name in (dtype_input, "a_log") else "bfloat16"
        policies = (broadcast, split) if name == "z" else (broadcast,) * len(shape)
        inputs.append(builder.var(name, fm.DistributedType(fm.tensor_type(dtype, shape), policies, placement), id=name))
    output = fm.TupleType((inputs[2].type, state.type))
    result = builder.call("nn.gdn_recurrent_core", tuple(inputs), output,
                          attrs={"key_head_dim": key_dim, "value_head_dim": 16,
                                 "num_key_heads": 2, "num_value_heads": 4, "epsilon": 1e-6}, id="recurrent")
    builder.function("main", tuple(inputs), (result,))
    return builder.build(entry="main")


def _propose(graph):
    target = NvidiaSm90Target()
    context = TritonCandidateContext(graph, target, {}, {}, {}, frozenset(), True,
                                    target.triton_implementation_model)
    before = graph.semantic_hash
    proposal = GdnCandidateProvider().propose(graph.node_map["recurrent"], context)
    assert graph.semantic_hash == before
    return proposal


@pytest.mark.parametrize("name", ["qkv", "z", "projection_input", "b_weight", "a_weight"])
def test_state_pipeline_rejects_unsupported_dtype_before_selection(name):
    proposal = _propose(_graph(dtype_input=name))
    assert all("state_smem_pipeline" not in candidate.id for candidate in proposal.candidates)


@pytest.mark.parametrize("options", [{"tokens": 2}, {"replicated": True}])
def test_state_pipeline_rejects_non_decode_or_duplicate_owners(options):
    proposal = _propose(_graph(**options))
    assert all("state_smem_pipeline" not in candidate.id for candidate in proposal.candidates)


def test_state_pipeline_rejects_unaligned_state_row_before_selection():
    graph = _graph(key_dim=12)
    nodes = tuple(replace(node, attrs={**node.attrs, "key_head_dim": 13})
                  if node.id == "recurrent" else node for node in graph.nodes)
    proposal = _propose(replace(graph, nodes=nodes))
    assert all("state_smem_pipeline" not in candidate.id for candidate in proposal.candidates)


@pytest.mark.parametrize("dimension", [12, 128, 256, 512])
def test_every_recurrent_candidate_covers_whole_key_dimension(dimension):
    proposal = _propose(_graph(key_dim=dimension))
    assert proposal is not None
    assert all(candidate.parameters["tile_state"][0] >= dimension for candidate in proposal.candidates)
    assert any("state_smem_pipeline" in candidate.id for candidate in proposal.candidates)
    assert ".persistent" in proposal.default_candidate
