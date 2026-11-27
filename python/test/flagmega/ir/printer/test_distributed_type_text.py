# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.printer import il_source


def _sharded_view_module(tensor_type: fm.TensorType, distributed_type: fm.DistributedType):
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    value = builder.var("value", tensor_type, id="value")
    shard = builder.call(
        "distributed.sharded_view",
        [value],
        distributed_type,
        id="shard",
        attrs={"new_type": distributed_type},
    )
    builder.function("main", [value], [shard])
    return builder.build(entry="main")


def test_distributed_attribute_and_result_match_nncase_il_conventions():
    tensor_type = fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), [10240, 160])
    placement = fm.Placement((8, 16), "yx", "bb")
    distributed_type = fm.DistributedType(
        tensor_type,
        (
            fm.SBP.split_block_cyclic((1,), block_size=64),
            fm.SBP.split_contiguous((0,), granularity=20),
        ),
        placement,
    )

    source = il_source(_sharded_view_module(tensor_type, distributed_type))

    assert (
        "NewType: F8_E4M3<2,16>[10240,160], "
        "(S(BC(64)@[1]),S(C(20)@[0])), [y:8,x:16], Partial: "
    ) in source
    assert (
        "// {F8_E4M3<2,16>[10240,160], "
        "(S(BC(64)@[1]),S(C(20)@[0])), [640@x,20@y], }"
    ) in source
    for leaked_repr in ("DistributedType(", "TensorType(", "DType.", "DimConst(", "Placement("):
        assert leaked_repr not in source


def test_distributed_result_shows_dynamic_local_shape_and_partial_state():
    tensor_type = fm.tensor_type("bfloat16", [fm.dim("tokens", minimum=1, maximum=4096), 5120])
    placement = fm.Placement((8,), "b", "b")
    distributed_type = fm.DistributedType(
        tensor_type,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0,), fm.ReduceOp.SUM),
    )

    source = il_source(_sharded_view_module(tensor_type, distributed_type))

    assert "NewType: bf16[tokens,5120], (S(C@[0]),B), [b:8], Partial: P([0], Sum)" in source
    assert "// {bf16[tokens,5120], (S(C@[0]),B), [ceil_div(tokens,8)@b,5120], P([0], Sum)}" in source
