# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
import pytest
from python.test.flagmega.passes.tir.bufferize.test_ref_slice import state_slice_graph


@pytest.mark.parametrize("reusable", [False, True])
def test_write_to_reference_view_is_a_write_to_the_function_parameter(reusable):
    module = Compiler().compile(state_slice_graph(reusable=reusable)).module
    plan = fm.verify_buffer_plan(module)
    function = module.execution_function_map["main"]
    state_id = "entry_state" if reusable else "state"
    state = set(dict(plan.function_map["main"].parameters)[state_id])
    # The convolution dispatch declares the aggregate state read_write; the
    # alias closure must reach both parent fields, not just the view's SSA ids.
    assert state <= set(function.attrs["written_parameters"])
    if reusable:
        for call in fm.execution_calls_of(function):
            assert state <= set(call.writes)
