# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from triton.flagmega import ir as fm


def module(*, export_source=False, dynamic=False, distributed=False, nested=False):
    narrow = fm.tensor_type("bfloat16", (8, ))
    wide = fm.tensor_type("float32", (8, ))
    if distributed:
        placement = fm.Placement((2, 2), "yx", "bb")
        narrow = fm.DistributedType(narrow, (fm.SBP.split_contiguous((0, 1)), ), placement)
        wide = fm.DistributedType(wide, (fm.SBP.split_contiguous((0, 1)), ), placement)
    builder = fm.IRBuilder(dialect="nn", stage="fused_norm")
    weights = [
        builder.weight(f"weight{i}", narrow, source="memory", key=f"weight{i}", id=f"weight{i}",
                       metadata={"rdata_group": {"name": "unit.weights", "index": i, "count": 2}}) for i in range(2)
    ]
    runtime = builder.var("runtime", wide, id="runtime")
    dynamic_weight = builder.var("dynamic_weight", narrow, id="dynamic_weight") if dynamic else None
    p = builder.var("p", narrow, id="p")
    x = builder.var("x", wide, id="x")
    cast = builder.call("tensors.cast", (p, ), wide, attrs={"dtype": "float32"}, id="cast")
    transformed = builder.call("math.add", (cast, cast), wide, id="transformed")
    output = builder.call("math.add", (x, transformed), wide, id="output")
    outputs = (output, p) if export_source else (output, )
    result_type = fm.TupleType((wide, narrow)) if export_source else wide
    builder.function("worker", (p, x), outputs, attrs={"reusable": True, "calling_convention": "device"})
    callee = "worker"
    if nested:
        outer_p = builder.var("outer_p", narrow, id="outer_p")
        outer_x = builder.var("outer_x", wide, id="outer_x")
        inner = builder.call("builtin.call", (outer_p, outer_x), result_type, attrs={"callee": "worker"}, id="inner")
        builder.function("outer", (outer_p, outer_x), (inner, ), attrs={"reusable": True})
        callee = "outer"
    calls = [
        builder.call("builtin.call", (dynamic_weight if dynamic and i == 1 else weights[i], runtime), result_type,
                     attrs={"callee": callee}, id=f"call{i}") for i in range(2)
    ]
    builder.function("main", (runtime, dynamic_weight) if dynamic else (runtime, ), calls)
    result = builder.build(entry="main")
    if distributed:
        result = replace(result, metadata={"auto_distribution": {"placement": placement.to_data()}})
    return fm.verify_module(result)
