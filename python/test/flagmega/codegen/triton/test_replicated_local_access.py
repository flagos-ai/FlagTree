# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import _access_in_result_domain
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.triton.kernels.distributed_boxing.test_partial_reduce_local_abi import _abi


def _operand(*, split=False):
    tensor = fm.tensor_type("bfloat16", (1, 16))
    placement = fm.Placement((2, 4), "yx", "bb")
    value_type = fm.DistributedType(tensor, (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((0, ), 8) if split else fm.SBP.broadcast(),
    ), placement)
    result = _abi((1, 16), local_shape=(1, 8 if split else 16), storage_kind="compact_local", coordinate_space="local")
    result["distributed_type"] = value_type.to_data()
    return result


def test_private_broadcast_operand_uses_logical_coordinates_in_a_narrower_cyclic_domain():
    source = _operand()
    target = {
        **source, "local_capacity_shape": (1, 8), "logical_coordinate_expressions":
        ("local_coord_0", "local_coord_1 * 2 + shard_coord_0")
    }
    domain = {"local_coordinates": ("0", "index"), "logical_coordinates": ("0", "index * 2 + shard_y")}
    offset = _access_in_result_domain(source, target, domain)
    for owner in range(2):
        for index in range(8):
            assert eval(offset, {"__builtins__": {}}, {"index": index, "shard_y": owner}) == index * 2 + owner


def test_private_split_operand_is_not_treated_as_a_full_replica():
    source = _operand(split=True)
    target = _operand()
    domain = {"local_coordinates": ("0", "index"), "logical_coordinates": ("0", "index")}
    with pytest.raises(CodegenError, match="different owner mappings"):
        _access_in_result_domain(source, target, domain)
