# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


class _ViewLoweringTarget(NvidiaSm90Target):
    def plan_launch(self, module, kernel_nodes):
        del module, kernel_nodes
        return {"num_warps": 1}

    def plan_codegen_package(self, module, kernel_nodes):
        del module, kernel_nodes
        return {"kind": "view-test"}


def test_dense_static_reshape_lowers_to_explicit_zero_copy_tir_view():
    builder = fm.IRBuilder(dialect="high_level", stage="selected_tir_variants")
    source_type = fm.tensor_type("bfloat16", (1, 16))
    result_type = fm.tensor_type("bfloat16", (1, 2, 8))
    source = builder.var("source", source_type, id="source")
    result = builder.call(
        "tensors.reshape",
        (source,),
        result_type,
        id="result",
        attrs={"shape": (1, 2, 8)},
    )
    builder.function("main", (source,), (result,))

    lowered = _ViewLoweringTarget().lower_to_tir(builder.build(entry="main"))

    view = lowered.node_map["result"]
    assert view.op == "tir.buffer_view"
    assert view.attrs == {"alias_kind": "reshape"}
    assert view.metadata["lowered_from"] == "tensors.reshape"
