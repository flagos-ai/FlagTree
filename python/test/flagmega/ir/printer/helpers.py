# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def arithmetic_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("x", tensor, id="x")
    weight = builder.weight("weight", tensor, source="/missing/checkpoint", key="tensor.key", id="long_weight_id")
    constant = builder.node(op="builtin.splat_const", type=tensor, attrs={"value": 2.0}, id="long_constant_id")
    first = builder.call("math.add", (x, weight), tensor, id="very_long_original_output_name")
    last = builder.call("math.mul", (first, constant), tensor, id="out")
    builder.function("main", (x, ), (last, ))
    return fm.verify_module(builder.build(entry="main"))
