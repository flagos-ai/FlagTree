# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A non-owning state view must stay addressable across outlined stages."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.passes.functions import FunctionStage, outline_function_stages
from triton.flagmega.passes.functions import function_nodes


def reference_view_graph(*, single_effect=False):
    config = GatedDeltaNetStateConfig(3, 1, 2, 4, 4, 4, 16)

    class Graph(fm.Module):
        def forward(self):
            qkv = self.input("qkv", fm.tensor_type("bfloat16", (1, 16)), id="qkv")
            state = self.input("state", config.ref_type, id="state")
            weight = self.input("weight", fm.tensor_type("bfloat16", (16, 1, 4)), id="weight")
            layer = self.input("layer", fm.tensor_type("int32", ()), id="layer")
            view = fm.F.nn.gated_delta_net_state_slice(state, layer, name="view")
            first = fm.F.nn.gated_delta_net_convolution(qkv, view, weight, conv_kernel_size=4, name="first_conv")
            value = fm.F.tensors.get_item(first, 0, name="first_value")
            updated = fm.F.tensors.get_item(first, 1, name="first_state")
            if single_effect:
                output = value
            else:
                second = fm.F.nn.gated_delta_net_convolution(value, updated, weight, conv_kernel_size=4, name="second_conv")
                output = fm.F.tensors.get_item(second, 0, name="second_value")
            parameters = (qkv, state, weight, layer)
            self.function("worker", parameters, (output, state), attrs={"noinline": True, "reusable": True})
            actual = tuple(self.input("entry_" + n.id, n.type, id="entry_" + n.id) for n in parameters)
            call = fm.F.builtin.call(*actual, result_type=fm.TupleType((output.type, state.type)),
                                     callee="worker", effect=fm.effect("read_write", "state"), name="call")
            self.function("main", actual, (fm.F.tensors.get_item(call, 0), fm.F.tensors.get_item(call, 1)))

    return Graph(dialect="high_level", stage="imported", entry="main",
                 metadata={"auto_distribution": {"placement": fm.Placement((2, 2), "yx", "bb").to_data()}}).build()


def outlined_reference_view_module(*, single_effect=False):
    source = reference_view_graph(single_effect=single_effect)
    source = Compiler().compile(source, stop_after="auto-distributed").module
    function = source.function_map["worker"]
    foreign = {n.id for f in source.functions if f.name != "worker" for n in function_nodes(source, f)}
    owned = [n.id for n in function_nodes(source, function) if n.id not in set(function.parameters) | foreign]
    cut = owned.index("first_value" if single_effect else "second_conv")
    stages = (FunctionStage("first", tuple(owned[:cut])), FunctionStage("second", tuple(owned[cut:])))
    outlined = outline_function_stages(source, "worker", stages,
                                      **({"effect_stage": 0} if single_effect else {"effect_stages": (0, 1)}))
    return Compiler().compile(outlined).module


def test_sequential_state_view_stages_keep_an_identity_result_abi():
    compiled = outlined_reference_view_module()
    plan = fm.verify_buffer_plan(compiled)
    assert any(n.op == "tir.ref_slice" for n in compiled.nodes)
    first = plan.function_map["worker.first"]
    assert first.result_aliases
    for result, parameter in first.result_aliases:
        assert plan.buffer_map[result].mem_span.must_alias(plan.buffer_map[parameter].mem_span)


def test_one_effect_stage_returning_a_tuple_keeps_reference_view_identity():
    compiled = outlined_reference_view_module(single_effect=True)
    plan = fm.verify_buffer_plan(compiled)
    first = plan.function_map["worker.first"]
    assert first.result_aliases
    for result, parameter in first.result_aliases:
        assert plan.buffer_map[result].mem_span.must_alias(plan.buffer_map[parameter].mem_span)


@pytest.mark.parametrize("single_effect", [False, True])
def test_device_state_views_preserve_ordered_updates(tmp_path, single_effect):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
    module = outlined_reference_view_module(single_effect=single_effect)
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    try:
        state = create_gdn_state(GatedDeltaNetStateConfig(3, 1, 2, 4, 4, 4, 16), device="cuda:0")
        state.convolution.copy_(torch.arange(state.convolution.numel(), device="cuda").reshape_as(state.convolution))
        state.recurrent.fill_(17)
        expected = state.convolution.clone()
        qkv = torch.arange(16, dtype=torch.bfloat16, device="cuda").reshape(1, 16)
        weight = torch.zeros((16, 1, 4), dtype=torch.bfloat16, device="cuda")
        weight[:, :, -1] = 1
        output = torch.empty_like(qkv)
        parameters = dict(runtime.buffer_plan.function_map["main"].parameters)
        fields = dict(zip(parameters["entry_state"], (state.convolution, state.recurrent), strict=True))
        tensors = {"entry_qkv": qkv, "entry_weight": weight}
        def arguments(index):
            values = {**tensors, "entry_layer": index}
            return tuple(output if a["role"] == "result" else
                         fields[a["buffer"]] if a["value"] == "entry_state" else values[a["value"]]
                         for a in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments(0))
        for index in (0, 2, 1, 2):
            qkv.add_(1)
            first = torch.nn.functional.silu(qkv)
            for value in (qkv,) if single_effect else (qkv, first):
                expected[index, :, :-1, :] = expected[index, :, 1:, :].clone()
                expected[index, :, -1, :] = value.reshape(2, 8)
            output.fill_(float("nan"))
            GeneratedTirCallGraphModule.run_into(runtime, *arguments(index))
            torch.cuda.synchronize()
            torch.testing.assert_close(output, first if single_effect else torch.nn.functional.silu(first), rtol=0, atol=0)
            torch.testing.assert_close(state.convolution, expected, rtol=0, atol=0)
            assert bool((state.recurrent == 17).all())
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("invalid", ["data_access", "duplicate", "foreign"])
def test_reference_view_contract_cannot_hide_data_access_or_foreign_operands(invalid):
    from triton.flagmega.ir.ops.core import OpDefinition, input_parameter, op_definition
    from triton.flagmega.ir.ops.nn.gdn_state_slice import GatedDeltaNetStateSlice
    from triton.flagmega.ir.type_pattern import is_ref

    class Bad(OpDefinition):
        source = input_parameter(is_ref(), memory_effect="read" if invalid == "data_access" else "none")
    Bad.reference_view_input_parameters = (
        (Bad.source, Bad.source) if invalid == "duplicate" else
        (GatedDeltaNetStateSlice.state,) if invalid == "foreign" else (Bad.source,))
    with pytest.raises(TypeError, match="reference_view_input_parameters"):
        op_definition("test.reference_view_" + invalid)(Bad)
