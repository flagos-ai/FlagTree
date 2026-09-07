# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def make_add_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported", metadata={"model": "artifact-unit"})
    value_type = fm.tensor_type("float32", [17])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")
