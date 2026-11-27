# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.targets import NvidiaSm90Target


class _ViewLoweringTarget(NvidiaSm90Target):
    def plan_launch(self, module, kernel_nodes):
        del module, kernel_nodes
        return {"num_warps": 1}

    def plan_codegen_package(self, module, kernel_nodes):
        del module, kernel_nodes
        return {"kind": "view-test"}


def _unpack_module(*, axis):
    builder = fm.IRBuilder(dialect="high_level", stage="selected_tir_variants")
    packed_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 4))
    value = builder.var("value", packed_type, id="value")
    unpacked_type = fm.tensor_type("bfloat16", (2, 32)) if axis == 1 else (
        fm.tensor_type("bfloat16", (16, 4))
    )
    unpacked = builder.call(
        "tensors.unpack",
        (value,),
        unpacked_type,
        id="unpacked",
        attrs={"axes": (axis,)},
    )
    builder.function("main", (value,), (unpacked,))
    return builder.build(entry="main")


def _bitcast_module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level",
                stage="selected_tir_variants",
                entry="main",
            )

        def forward(self):
            packed_type = fm.tensor_type(
                fm.vector_type("bfloat16", (8,)), (2, 4)
            )
            value = self.input("value", packed_type, id="value")
            result = fm.F.tensors.bitcast(
                value, fm.DType.BFLOAT16, name="result"
            )
            self.function("main", (value,), (result,))

    return Graph().build()


def test_last_axis_unpack_lowers_to_explicit_zero_copy_tir_view():
    lowered = _ViewLoweringTarget().lower_to_tir(_unpack_module(axis=1))

    view = lowered.node_map["unpacked"]
    assert view.op == "tir.buffer_view"
    assert view.attrs == {"alias_kind": "vector_reinterpret"}
    assert view.metadata["lowered_from"] == "tensors.unpack"


def test_unpack_requiring_lane_permutation_is_not_silently_made_a_view():
    with pytest.raises(
        IRVerificationError,
        match="No reviewed Triton TIR candidate.*tensors.unpack",
    ):
        _ViewLoweringTarget().lower_to_tir(_unpack_module(axis=0))


def test_storage_bitcast_lowers_to_typed_alias_without_a_kernel_candidate():
    lowered = _ViewLoweringTarget().lower_to_tir(_bitcast_module())

    view = lowered.node_map["result"]
    assert view.op == "tir.buffer_view"
    assert view.attrs == {"alias_kind": "vector_reinterpret"}
    assert view.metadata["lowered_from"] == "tensors.bitcast"
