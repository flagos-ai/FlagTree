# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def _definition():
    tensor = fm.tensor_type("float32", (16,))
    return fm.T.kernel_definition("silu", "triton", (
        fm.T.prim_parameter("source", tensor),
        fm.T.prim_parameter("result", tensor, fm.PrimParameterRole.OUTPUT),
    ), fm.T.kernel_dispatch("math.silu", arguments=("source",), outputs=("result",),
                            reads=("source",), writes=("result",)),
        fm.T.return_((fm.T.return_binding(fm.T.value_ref("result", tensor), "result"),)))


def test_kernel_contract_is_not_a_function_or_region():
    definition = _definition()
    assert not isinstance(definition, fm.PrimFunction)
    assert "body" not in definition.to_data()
    assert fm.tir_from_data(definition.to_data()) == definition
    body = fm.T.sequential((fm.T.pipeline_stage("stage", definition.dispatch),))
    region = fm.T.producer_consumer_region(body, body)
    with pytest.raises(IRSchemaError, match="exactly one KernelDispatch"):
        replace(definition, dispatch=region)


@pytest.mark.parametrize("key", ["calling_convention", "noinline"])
def test_kernel_contract_cannot_own_a_function_calling_convention(key):
    with pytest.raises(IRSchemaError, match="calling convention"):
        replace(_definition(), attrs={key: True})


def test_kernel_signature_preserves_parameter_order_and_return_storage_validation():
    definition = _definition()
    with pytest.raises(IRSchemaError, match="ordered inputs"):
        replace(definition, parameters=tuple(reversed(definition.parameters)))
    with pytest.raises(IRSchemaError, match="not an input/output"):
        replace(definition, results=fm.T.return_((fm.T.return_binding(
            fm.T.value_ref("result", definition.parameters[1].type), "missing",
        ),)))


def test_kernel_invoke_has_actual_buffer_effects_but_no_memory_pool_frame():
    invoke = fm.T.kernel_invoke("op", "silu", arguments=(fm.T.prim_call_binding("source", "x"),),
                                results=(fm.T.prim_call_binding("result", "y"),), reads=("x",), writes=("y",))
    assert not isinstance(invoke, fm.PrimFunctionCall)
    assert fm.tir_from_data(invoke.to_data()) == invoke
    assert "callee" not in invoke.to_data()
    assert "memory_pools" not in invoke.to_data()
    with pytest.raises(TypeError, match="memory_pools"):
        replace(invoke, memory_pools=(fm.T.memory_pool_frame("workspace", None, 0, 32),))
    with pytest.raises(IRSchemaError, match="effects must reference"):
        replace(invoke, writes=("unbound",))
