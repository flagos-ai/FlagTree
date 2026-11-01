# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Remote reductions retain their sum tree and require separate physical proofs."""

from copy import deepcopy

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError

from . import test_remote_work_partition as remote_contract
from .test_local_narrowing import local_abi
from .test_partial_reduce_local_abi import _raw


@pytest.fixture(scope="module")
def partition():
    return remote_contract.partition.__wrapped__()


def reduction_fixture():
    call, leaf, source, result, context = remote_contract.fixture()
    partial = {"axes": [0], "reduce_op": "sum"}
    source["abi"]["distributed_type"]["partial"] = partial
    leaf.update(reduction=True, partial_owner_count=1, partial_owner_tile=1,
                partial_owner="boxing_partial_member", source_offset="boxing_offsets // 128",
                accumulator_type="tl.float32", remote_peers=4)
    leaf.pop("remote_owner")
    leaf.pop("remote_source_offset")
    return call, leaf, source, result, context


def test_reduction_assigns_each_output_tile_once_without_changing_chip_writers(partition):
    reduce_partition = partition.__globals__["reduce_partition"]
    writer = reduce_partition(*reduction_fixture())
    assert writer is not None
    for chip in range(4):
        tiles = [tile for cta in range(128) for tile in range(cta * 1024, 8192, 128 * 1024)
                 if eval(writer, {"__builtins__": {}}, {"shard_coord0": chip})]
        assert tiles == (list(range(0, 8192, 1024)) if chip == 1 else [])


def test_chip_compact_nonoverlap_uses_physical_components_not_peer_instances(partition):
    call, leaf, source, result, context = reduction_fixture()
    source["abi"].update(storage_kind="compact_per_chip", owner_count=128)
    result["abi"]["pool_byte_offset"] = 128 * 16 * 4
    assert partition.__globals__["byte_span"](source["abi"]) == (0, 128 * 16 * 4)
    assert partition.__globals__["reduce_partition"](call, leaf, source, result, context) is not None


@pytest.mark.parametrize("physical_scope", ("chip", "block"))
def test_cooperative_reduction_derives_component_from_destination_not_executing_cta(physical_scope):
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (1, 4096))
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, mesh)
    partial = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 2), 32)),
        mesh, partial=fm.SBP.partial((0,)))
    source = local_abi(partial)
    source.update(storage="workspace", memory_sharing_scope="chip", storage_kind="compact_per_chip",
                  component_stride_scalar_elements=32, owner_index_expression="block_index")
    result = local_abi(broad)
    result.update(storage="workspace", memory_sharing_scope=physical_scope, coordinate_space="canonical_global",
                  storage_kind="canonical_global", pool_byte_offset=4096 * 4)
    raw = _raw((source,), (result,), ("gather_reduce_scatter",))
    raw["facts"] = {"require_cooperative_remote_reduce": True}
    if physical_scope == "block":
        with pytest.raises(CodegenError, match="chip-shared canonical result"):
            prepare_kernel_calls((raw,), function_name="main")
        return
    leaf = prepare_kernel_calls((raw,), function_name="main")[0]["leaves"][0]
    assert leaf["capacity"] == 4096
    assert leaf["partial_owner_count"] == 1
    assert leaf["remote_peers"] == 4
    for offset in range(4096):
        assert eval(leaf["source_offset"], {"__builtins__": {}}, {"boxing_offsets": offset}) == offset


@pytest.mark.parametrize("cause", ("block_partial", "max", "private_source", "private_result", "alias",
    "cta_offset", "cta_owner", "wrong_peer_count", "local_owners", "fp16_accumulator", "peer_selection",
    "partial_backing", "result_partial", "truncated_grid", "participation"))
def test_reduction_rejects_unproven_ownership_or_sum_tree(partition, cause):
    call, leaf, source, result, context = reduction_fixture()
    if cause == "block_partial":
        source["abi"]["distributed_type"]["partial"]["axes"] = [0, 1]
    elif cause == "max":
        source["abi"]["distributed_type"]["partial"]["reduce_op"] = "max"
    elif cause == "private_source": source["abi"]["memory_sharing_scope"] = "block"
    elif cause == "private_result": result["abi"]["memory_sharing_scope"] = "block"
    elif cause == "alias": result["abi"]["pool_byte_offset"] = 0
    elif cause == "cta_offset": leaf["source_offset"] += " + shard_coord1 * 32"
    elif cause == "cta_owner": leaf["partial_owner"] = "block_index"
    elif cause == "wrong_peer_count": leaf["remote_peers"] = 2
    elif cause == "local_owners": leaf["partial_owner_count"] = 8
    elif cause == "fp16_accumulator": leaf["accumulator_type"] = "tl.float16"
    elif cause == "peer_selection": leaf["remote_owner"] = "boxing_offsets // 2048"
    elif cause == "partial_backing": source["abi"]["distributed_backing_type"] = deepcopy(source["abi"]["distributed_type"])
    elif cause == "result_partial": result["abi"]["distributed_type"]["partial"] = {"axes": [0]}
    elif cause == "truncated_grid": context["grid"] = [64, 1, 1]
    elif cause == "participation": call["participation_active"] = "shard_coord2 == 0"
    assert partition.__globals__["reduce_partition"](call, leaf, source, result, context) is None
