# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def _module():
    batch = fm.dim("batch", minimum=1, maximum=8)
    value_type = fm.tensor_type("float32", [batch, 4])
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    add = builder.call("math.add", (lhs, rhs), value_type, id="add")
    output = builder.call("math.silu", (add,), value_type, id="output")
    dead = builder.call("math.silu", (rhs,), value_type, id="dead")
    del dead
    builder.function("main", (lhs, rhs), (add, output))
    scalar_type = fm.tensor_type("int32", [])
    prim = fm.T.prim_function(
        "constant_one",
        "triton",
        (),
        fm.T.sequential((fm.T.evaluate(fm.T.immediate(1, scalar_type)),)),
        fm.T.return_(),
    )
    builder.prim_function(prim)
    return fm.verify_module(builder.build(entry="main"))


class RecordingVisitor(fm.IRVisitor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ops = []
        self.types = []
        self.dimensions = []
        self.tir = []

    def visit_node(self, node):
        self.ops.append(node.op)

    def visit_math_add(self, node):
        self.ops.append(f"special:{node.op}")

    def visit_type(self, value):
        self.types.append(type(value).__name__)

    def visit_dimension(self, value):
        self.dimensions.append(str(value))

    def visit_tir_node(self, node):
        self.tir.append(type(node).__name__)


def test_whole_ir_visitor_is_bottom_up_memoized_and_op_dispatched():
    visitor = RecordingVisitor()
    visitor.run(_module())

    assert visitor.ops == ["builtin.var", "builtin.var", "special:math.add", "math.silu"]
    assert "dead" not in visitor.ops
    assert visitor.ops.count("special:math.add") == 1  # shared by two function outputs
    assert visitor.types.count("TensorType") == 2  # shared graph type + scalar TIR type
    assert visitor.dimensions.count("batch") == 1
    assert visitor.tir.count("Immediate") == 1
    assert visitor.tir[-1] == "PrimFunction"


def test_whole_ir_visitor_can_explicitly_include_dead_nodes():
    visitor = RecordingVisitor(visit_dead_nodes=True, visit_prim_functions=False)
    visitor.run(_module())

    assert visitor.ops.count("math.silu") == 2
