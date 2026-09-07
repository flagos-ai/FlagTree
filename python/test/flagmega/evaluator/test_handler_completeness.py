# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator import (
    DictWeightResolver,
    TorchEvaluator,
    inspect_evaluation_support,
)


def _call_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", [2])
    callee_arg = builder.var("callee_arg", value_type, id="callee_arg")
    callee_result = builder.call("math.silu", [callee_arg], value_type, id="callee_result")
    builder.function("layer", [callee_arg], [callee_result])
    main_arg = builder.var("main_arg", value_type, id="main_arg")
    main_result = builder.call(
        "builtin.call",
        [main_arg],
        value_type,
        id="main_result",
        attrs={"callee": "layer"},
    )
    builder.function("main", [main_arg], [main_result])
    return builder.build(entry="main")


def _unsupported_tir_module():
    builder = fm.IRBuilder(dialect="tir", stage="tir-lowered")
    value_type = fm.tensor_type("float32", [2])
    source = builder.var("source", value_type, id="source")
    output = builder.call(
        "tir.kernel",
        [source],
        value_type,
        id="kernel",
        attrs={
            "semantic_op": "math.silu",
            "candidate": "unit_test",
            "parameters": {},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", [source], [output])
    return builder.build(entry="main")


def test_completeness_distinguishes_op_handler_from_visitor_call_dispatch():
    module = _call_module()
    support = inspect_evaluation_support(module)
    assert support.is_complete
    assert support.visitor_handled_ops == ("builtin.call",)
    assert {"builtin.var", "math.silu"}.issubset(support.supported_ops)

    value = torch.tensor([1.0, -1.0])
    output = TorchEvaluator(DictWeightResolver({})).run(module, {"main_arg": value})[0]
    torch.testing.assert_close(output, torch.nn.functional.silu(value))


def test_missing_live_handler_is_reported_before_execution():
    module = _unsupported_tir_module()
    support = inspect_evaluation_support(module)
    assert not support.is_complete
    assert support.gaps[0].node_id == "kernel"
    assert support.gaps[0].op == "tir.kernel"

    with pytest.raises(EvaluationError, match="kernel:tir.kernel"):
        TorchEvaluator(DictWeightResolver({})).run(
            module,
            {"source": torch.ones(2)},
        )
