# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator import DictWeightResolver, EvaluationContext
from triton.flagmega.ir.ops.math.add import Add


def _add_module():
    class AddGraph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value_type = fm.tensor_type("float32", [2])
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            result = fm.F.math.add(lhs, rhs, name="result")
            self.function("main", [lhs, rhs], [result])

    return AddGraph().build()


def test_context_reads_named_parameter_and_restores_parent_frame():
    module = _add_module()
    lhs = torch.tensor([1.0, 2.0])
    rhs = torch.tensor([3.0, 4.0])
    context = EvaluationContext(
        module,
        torch=torch,
        inputs={"lhs": lhs, "rhs": rhs},
        weights=DictWeightResolver({}),
        constant_assets={},
    )
    result = module.node_map["result"]
    values = {"lhs": lhs, "rhs": rhs}

    with context.call_scope(result, (lhs, rhs), values.__getitem__):
        assert context.current_node is result
        assert context.return_type == result.type
        assert context.get_argument_value(Add, Add.lhs) is lhs
        assert context.get_argument_value(Add, Add.rhs) is rhs
        assert context.evaluate(module.node_map["lhs"]) is lhs

        with context.call_scope(result, (rhs, lhs), values.__getitem__):
            assert context.get_argument_value(Add, Add.lhs) is rhs

        assert context.get_argument_value(Add, Add.lhs) is lhs

    with pytest.raises(EvaluationError, match="Current evaluator call is not set"):
        _ = context.current_node


def test_context_rejects_parameter_from_another_definition():
    module = _add_module()
    lhs = torch.tensor([1.0, 2.0])
    rhs = torch.tensor([3.0, 4.0])
    context = EvaluationContext(
        module,
        torch=torch,
        inputs={"lhs": lhs, "rhs": rhs},
        weights=DictWeightResolver({}),
        constant_assets={},
    )
    result = module.node_map["result"]
    with context.call_scope(result, (lhs, rhs), lambda node_id: None):
        with pytest.raises(EvaluationError, match="does not belong"):
            context.get_argument_value(Add, fm.get_definition("math.mul").lhs)
