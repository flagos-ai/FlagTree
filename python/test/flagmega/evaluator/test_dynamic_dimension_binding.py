# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _dynamic_add_module():
    class DynamicAdd(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            batch = fm.dim("batch", minimum=1, maximum=8)
            value_type = fm.tensor_type("float32", [batch, 4])
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            output = fm.F.math.add(lhs, rhs, name="output")
            self.function("main", [lhs, rhs], [output])

    return DynamicAdd().build()


def test_repeated_dynamic_dimension_is_bound_once_for_the_run():
    module = _dynamic_add_module()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    result = evaluator.run_result(
        module,
        {"lhs": torch.ones(2, 4), "rhs": torch.ones(2, 4)},
    )
    assert tuple(result.outputs[0].shape) == (2, 4)
    with pytest.raises(TypeError):
        result.trace["output"] = torch.zeros(2, 4)


def test_repeated_dynamic_dimension_rejects_inconsistent_inputs():
    module = _dynamic_add_module()
    with pytest.raises(EvaluationError, match="expects 2, got 3"):
        TorchEvaluator(DictWeightResolver({})).run(
            module,
            {"lhs": torch.ones(2, 4), "rhs": torch.ones(3, 4)},
        )


def test_dynamic_dimension_enforces_declared_bounds():
    module = _dynamic_add_module()
    with pytest.raises(EvaluationError, match="above 8"):
        TorchEvaluator(DictWeightResolver({})).run(
            module,
            {"lhs": torch.ones(9, 4), "rhs": torch.ones(9, 4)},
        )
