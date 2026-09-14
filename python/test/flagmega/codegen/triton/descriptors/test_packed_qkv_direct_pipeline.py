# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Weight-only staging does not constrain the owner's input K capacity."""

from dataclasses import replace
from functools import lru_cache

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels.packed_qkv import _applicable
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.targets import NvidiaSm90Target

from .conftest import _packed_qkv_mma_pipeline_module
from .test_packed_qkv_mma_candidate import _context
from .test_packed_qkv_n_tiles import check_exact_coordinates_and_partials, implementation_id, tiled_module


def direct_id(kind, descriptor):
    return implementation_id(kind, descriptor).removesuffix("_pipeline") + "_direct_pipeline"


@lru_cache(maxsize=None)
def direct_module(kind, descriptor, input_extent):
    target = NvidiaSm90Target()
    model = target.triton_implementation_model
    identity = direct_id(kind, descriptor)
    original = model.implementation(identity)
    assert original is not None
    # K=128 exercises a partial final transfer for local K=192, and transfers
    # larger than the input capacity for ragged/empty K owners.
    weight_stage = original.shared_workspaces[0]
    shape = ((1,) if descriptor == "single" else ()) + (8, 8, 2, 64)
    candidate = replace(original, parameters={**original.parameters, "block_k": 128},
                        shared_workspaces=(replace(weight_stage, type=fm.tensor_type("bfloat16", (2, *shape))),))
    target.triton_implementation_model = replace(model, implementations=tuple(
        candidate if value.id == identity else value for value in model.implementations))
    return _packed_qkv_mma_pipeline_module(identity, target=target, input_extent=input_extent,
                                          projection_widths=(2048, 384, 384), num_kv_heads=3)


def test_partial_axis_proof_is_independent_of_full_tile_proof():
    module = tiled_module("mma", "table", (4096, 512, 512), 2)
    context = _context(module)
    contract = dict(context.implementation_model.implementation(implementation_id("mma", "table")).contract)
    del contract["requires_uniform_full_input_reduction_tiles"]
    assert _applicable(context, contract)
    output = context.dispatch.outputs[0]
    wrong = replace(context.function.parameter_map[output].type, partial=fm.SBP.partial((1,), fm.ReduceOp.SUM))
    assert not _applicable(_context(module, parameter_types={output: wrong}), contract)


@pytest.mark.parametrize("kind", ("gemv", "mma"))
@pytest.mark.parametrize("descriptor", ("single", "table"))
@pytest.mark.parametrize("input_extent", (384, 1536, 4096))
def test_direct_pipeline_describes_k_tail_without_padding_storage(kind, descriptor, input_extent):
    module = direct_module(kind, descriptor, input_extent)
    call, = (value for value in describe_tir_package(module)["render_calls"]
             if value["implementation"] == direct_id(kind, descriptor))
    assert call["direct_lhs"] is True
    assert call["local_k_capacity"] == {384: 64, 1536: 192, 4096: 512}[input_extent]
    assert call["num_k_tiles"] == (call["local_k_capacity"] + 127) // 128
    assert len(call["shared_workspaces"]) == 1
    assert not call["pipeline_consumer_workspaces"]
    request, = call["host_tensor_descriptor_requests"]
    shape = request["shape"] if descriptor == "single" else request["entries"][0]["shape"]
    assert shape[2 if descriptor == "single" else 1] * 16 == call["local_k_capacity"]
    model = NvidiaSm90Target().triton_implementation_model
    contract = model.implementation(direct_id(kind, descriptor)).contract
    assert "required_local_reduction_extent" not in contract
    assert "requires_uniform_full_input_reduction_tiles" not in contract
    assert direct_id(kind, descriptor) not in model.preferences["qkv_parallel_linear"]


@pytest.mark.parametrize("kind", ("gemv", "mma"))
@pytest.mark.parametrize("descriptor", ("single", "table"))
@pytest.mark.parametrize("input_extent", (384, 1536, 4096))
def test_direct_pipeline_exact_k_tiles_and_empty_owners(kind, descriptor, input_extent, tmp_path):
    module = direct_module(kind, descriptor, input_extent)
    check_exact_coordinates_and_partials(module, direct_id(kind, descriptor), (2048, 384, 384), input_extent, tmp_path)


@pytest.mark.parametrize("violation", ("rows", "reduction", "output_dtype", "partial_axes", "missing_partial",
                                      "weight_k", "weight_n"))
def test_direct_candidate_rejects_incompatible_physical_abi(violation):
    from .test_packed_qkv_mma_candidate import _candidate_ids

    module = direct_module("mma", "table", 1536)
    context = _context(module)
    if violation in {"rows", "reduction"}:
        name = context.dispatch.arguments[0]
    elif violation.startswith("weight_"):
        name = context.dispatch.arguments[1]
    else:
        name = context.dispatch.outputs[1]
    original = context.function.parameter_map[name].type
    if violation in {"rows", "reduction"}:
        changed = replace(original, tensor=fm.tensor_type("bfloat16", (2, 1536) if violation == "rows" else (1, 2048)))
    elif violation.startswith("weight_"):
        shape = list(original.shape)
        shape[1 if violation == "weight_k" else 2] += 1
        changed = replace(original, shape=tuple(shape))
    elif violation == "output_dtype":
        changed = replace(original, tensor=replace(original.tensor, dtype=fm.VectorType(fm.DType.FLOAT32, (8,))))
    else:
        changed = replace(original, partial=None if violation == "missing_partial" else fm.SBP.partial((1,), fm.ReduceOp.SUM))
    ids = _candidate_ids(_context(module, parameter_types={name: changed}))
    assert not {direct_id(kind, descriptor) for kind in ("gemv", "mma")
                for descriptor in ("single", "table")}.intersection(ids)
