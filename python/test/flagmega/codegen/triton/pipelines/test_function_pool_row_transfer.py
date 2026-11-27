# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A formal row can inherit only a complete closed-call owned-pool proof."""
from copy import deepcopy
import pytest
from triton.flagmega.codegen.triton.function_pool_abi import function_pool_origins
from triton.flagmega.codegen.triton.function_pool_abi import plan_function_pool_abis
from triton.flagmega.codegen.triton.row_transfer import validate_async_row_copy
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.triton.pipelines.test_function_pool_alignment import bindings


def row_abi():
    return dict(local_capacity_shape=(1, 128), scalar_lane_count=1, scalar_itemsize=2, scalar_storage_strides=(128, 1),
                pooled=False, alignment_bytes=1, pool_byte_offset=0, coordinate_space="canonical_global",
                distributed_type=None)


def test_closed_pool_proof_allows_formal_without_relabeling_storage():
    values = bindings()
    before = deepcopy(values)
    origins = function_pool_origins(values, entry="main")
    abi = row_abi()
    with pytest.raises(CodegenError, match="owned pool"):
        validate_async_row_copy(abi, 128)
    validate_async_row_copy(abi, 128, owned_pool_origin=origins["worker"]["weight"])
    assert values == before and abi == row_abi()
    assert origins["worker"]["weight"]["calls"] == (("main", "call0"), ("main", "call1"))


def test_one_external_actual_prevents_owned_pool_claim():
    values = bindings()
    edge = values["main"]["call_abi"]["events"][1]["arguments"][0]
    edge["actual_abi"]["pooled"] = False
    origins = function_pool_origins(values, entry="main")
    assert "weight" not in origins.get("worker", {})


def test_owned_origin_does_not_depend_on_argument_count_savings():
    values = bindings()
    values["worker"]["pools"] = []
    assert "worker" not in plan_function_pool_abis(values, entry="main")[0]
    proof = function_pool_origins(values, entry="main")["worker"]["weight"]
    validate_async_row_copy(row_abi(), 128, owned_pool_origin=proof)


@pytest.mark.parametrize("change", [dict(alignment_bytes=2), dict(calls=()), dict(schema="unproved")])
def test_incomplete_or_weak_pool_proof_does_not_relax_copy_guard(change):
    proof = function_pool_origins(bindings(), entry="main")["worker"]["weight"]
    with pytest.raises(CodegenError):
        validate_async_row_copy(row_abi(), 128, owned_pool_origin={**proof, **change})


def test_shifted_formal_and_unknown_dynamic_view_remain_guarded():
    proof = function_pool_origins(bindings(), entry="main")["worker"]["weight"]
    for update in (dict(pool_byte_offset=2), dict(view_byte_offset={"unknown": True})):
        with pytest.raises(CodegenError):
            validate_async_row_copy({**row_abi(), **update}, 128, owned_pool_origin=proof)
