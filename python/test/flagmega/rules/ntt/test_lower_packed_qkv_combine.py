# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt import lower_packed_qkv_parallel_linear_combine_rule


def _partial_module():
    placement = fm.Placement((8, 16), "yx", "bb")
    tensors = tuple(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, extent))
        for extent in (256, 128, 128)
    )
    output_type = fm.TupleType(tuple(
        fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for tensor in tensors
    ))
    partial_type = fm.TupleType(tuple(
        fm.DistributedType(
            tensor,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
            partial=fm.SBP.partial((0,)),
        )
        for tensor in tensors
    ))
    builder = fm.IRBuilder(dialect="ntt", stage="qkv_combine_folded")
    value = builder.var("qkv", partial_type, id="qkv")
    combine = builder.call(
        "ntt.packed_qkv_parallel_linear_combine",
        (value,),
        output_type,
        id="combine",
        attrs={"output_type": output_type},
    )
    builder.function("main", (value,), (combine,))
    return builder.build(entry="main")


def test_lower_partial_combine_to_generic_tuple_boxing():
    source = _partial_module()
    module = DataflowPass(
        "LowerPackedQKVParallelLinearCombine",
        (lower_packed_qkv_parallel_linear_combine_rule(),),
    ).run(source)

    lowered = module.node_map["combine"]
    assert lowered.op == "distributed.boxing"
    assert lowered.inputs == ("qkv",)
    assert lowered.type == source.node_map["combine"].type
    assert lowered.attrs == {"new_type": lowered.type}
    assert lowered.metadata["lowered_by"] == (
        "LowerPackedQKVParallelLinearCombine"
    )
