# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Equal capacity/strides do not erase a parent's shard-coordinate contract."""

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import BufferizationOptions, NttBufferizationPolicy


def test_binding_specializes_distinct_parent_backings_before_formal_buffers():
    mesh = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (1, 16))
    parent = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), mesh)
    own = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 4)), mesh)
    dispatch = fm.T.kernel_dispatch(semantic_op="math.silu", semantic_candidate="tir.unary.silu",
        arguments=("input",), outputs=("output",), reads=("input",), writes=("output",),
        inplace_alias_candidates=())
    kernel = fm.T.prim_function("shared", "triton", (
        fm.T.prim_parameter("input", own, fm.T.PrimParameterRole.INPUT),
        fm.T.prim_parameter("output", own, fm.T.PrimParameterRole.OUTPUT)),
        fm.T.sequential((dispatch,)), fm.T.return_((fm.T.return_binding(fm.T.value_ref("output", own), "output"),)))
    b = fm.IRBuilder(dialect="semantic_tir", stage="selected_microkernels")
    b.prim_function(kernel)
    x = b.var("x", own, id="x")
    y = b.var("y", parent, id="y")
    plain = b.call("math.silu", (x,), own, id="plain")
    source = b.call("math.silu", (y,), parent, id="source")
    view = b.call("distributed.sharded_view", (source,), own, id="view", attrs={"new_type": own})
    first = b.call("tir.call", (plain,), own, id="first", attrs={"callee": "shared"})
    second = b.call("tir.call", (view,), own, id="second", attrs={"callee": "shared"})
    b.function("main", (x, y), (first, second))
    result = NttBufferizationPolicy(BufferizationOptions.generic()).bufferize(b.build(entry="main"))
    assert result.node_map["first"].attrs["callee"] != result.node_map["second"].attrs["callee"]
    assert len(result.prim_functions) == 2
    buffers = [f.runtime_parameters[0].buffers[0] for f in result.prim_functions]
    assert buffers[0].strides == buffers[1].strides
    assert {v.distributed_backing_type for v in buffers} == {None, parent}
