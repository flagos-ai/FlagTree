# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SSA recurrence is structural and never represented by cyclic node edges."""

from dataclasses import replace

import pytest
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import F, IRBuilder, TensorType, verify_module
from triton.flagmega.ir.ops.core import construction_scope


def loop_module(stop=3, step=1):
    builder = IRBuilder(dialect="high_level", stage="imported")
    tensor = TensorType("float32", (4,))
    index_type = TensorType("int64", ())
    iv = builder.var("iv", index_type, id="iv")
    carried = builder.var("carried", tensor, id="carried")
    capture = builder.var("capture", tensor, id="capture")
    with construction_scope(builder):
        next_value = F.math.add(carried, capture, name="body_add")
    region = builder.region("body", (iv, carried, capture), (next_value,), iter_count=1)
    value = builder.var("value", tensor, id="value")
    increment = builder.var("increment", tensor, id="increment")
    with construction_scope(builder):
        lower = F.builtin.scalar_const(index_type, 0, name="lower")
        upper = F.builtin.scalar_const(index_type, stop, name="upper")
        stride = F.builtin.scalar_const(index_type, step, name="stride")
        result = F.control.for_loop(lower, upper, stride, value, body=region,
                                    captures=(increment,), name="loop")
        output = F.tensors.get_item(result, 0, name="output")
    builder.function("main", (value, increment), (output,))
    return builder.build(entry="main")


@pytest.mark.parametrize("stop,step,trips", [(0, 1, 0), (-1, 1, 0), (1, 1, 1), (5, 2, 3)])
def test_for_evaluates_recurrence(stop, step, trips):
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import TorchEvaluator

    module = verify_module(loop_module(stop, step))
    value = torch.arange(4, dtype=torch.float32)
    increment = torch.tensor([1., 2., 3., 4.])
    result, = TorchEvaluator({}).run(module, {"value": value, "increment": increment})
    torch.testing.assert_close(result, value + trips * increment, rtol=0, atol=0)


def test_for_rejects_implicit_region_capture():
    module = loop_module()
    nodes = tuple(replace(node, inputs=("carried", "increment")) if node.id == "body_add" else node
                  for node in module.nodes)
    # Put the deliberately illegal outer operand before the region for topology.
    outer = next(node for node in nodes if node.id == "increment")
    nodes = (outer, *(node for node in nodes if node is not outer))
    with pytest.raises(IRVerificationError, match="(capture|closed|scope)"):
        verify_module(replace(module, nodes=nodes))


def test_for_rejects_nonpositive_step():
    with pytest.raises(IRVerificationError, match="step"):
        verify_module(loop_module(step=0))


def test_for_compiler_preserves_loop_to_execution_ir(tmp_path):
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.ir.tir.visitor import TIRVisitor

    result = Compiler().compile(loop_module()).module
    loops = []

    class Visitor(TIRVisitor):
        def visit_for(self, node):
            loops.append(node)
            return self.generic_visit(node)

    for function in result.execution_functions:
        Visitor().visit(function)
    assert len(loops) == 1
    assert loops[0].iter_args
    from triton.flagmega.ir import emit_module, load_module
    restored = load_module(emit_module(result, tmp_path / "for.py"))
    assert restored.semantic_hash == result.semantic_hash
