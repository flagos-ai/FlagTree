# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Returning a reference result must not retain a recursive visitor's tensors."""
import gc
import weakref

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _module(nested):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (8,))
    value = builder.var("value", value_type, id="value")
    hidden = builder.call("math.mul", (value, value), value_type, id="hidden")
    output = builder.call("math.add", (hidden, hidden), value_type, id="output")
    if nested:
        builder.function("worker", (value,), (output,))
        argument = builder.var("argument", value_type, id="argument")
        result = builder.call("builtin.call", (argument,), value_type, id="result", attrs={"callee": "worker"})
        builder.function("main", (argument,), (result,))
    else:
        builder.function("main", (value,), (output,))
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("nested", [False, True])
def test_dropping_a_trace_releases_intermediates_without_cyclic_gc(nested):
    module = _module(nested)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(4):
            result = evaluator.run_result(module, {"argument" if nested else "value": torch.ones(8)})
            hidden = weakref.ref(result.trace['hidden'])
            output, = result.outputs
            del result
            assert hidden() is None, 'a recursive visitor retained the completed run'
            torch.testing.assert_close(output, torch.full((8,), 2.))
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_live_trace_retains_intermediates_and_remains_immutable():
    result = TorchEvaluator(DictWeightResolver({})).run_result(_module(False), {'value': torch.ones(8)})
    hidden = weakref.ref(result.trace['hidden'])
    assert hidden() is result.trace['hidden']
    with pytest.raises(TypeError):
        result.trace['hidden'] = None
