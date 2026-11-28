# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Early numeric boundaries retain BF16 rounding and shape-only references."""
import pytest
import torch
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.passes.functions.post_boundary_pack import post_function_boundary_pack_propagation
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.memory_effect import MemoryEffect


def residual_graph(axes=(1, )):

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", fm.tensor_type(fm.vector_type("float32", (2, 2)), (1, 4)), id="x")
            y = self.input("y", fm.tensor_type(fm.vector_type("float32", (4, )), (1, 4)), id="y")
            narrow = fm.F.ntt.vectorized_cast(x, fm.vector_type("bfloat16", (4, )), axes, rounding_contract="preserve",
                                              name="rounded")
            wide = fm.F.ntt.vectorized_cast(narrow, fm.vector_type("float32", (4, )), axes,
                                            rounding_contract="preserve", name="widened")
            result = fm.F.math.vectorized_binary(wide, y, binary_op="add", name="sum")
            self.function("main", (x, y), (result, x))

    return Graph(dialect="high_level", stage="boundary_layout_propagated", entry="main").build()


def direct_embedding_graph(shared=None):
    state_config = PagedAttentionStateConfig(1, 2, 16)

    class Graph(fm.Module):

        def forward(self):
            ids = self.input("ids", fm.tensor_type("int64", (2, )))
            weight = self.input("weight", fm.tensor_type(fm.vector_type("bfloat16", (4, )), (7, 4)))
            state = self.input("state", state_config.ref_type)
            embedded = fm.F.nn.embedding(ids, weight, padding_idx=0, name="embedded")
            positions = fm.F.nn.rotary_embedding(embedded, state, head_dim=16, theta=10000., name="positions")
            cosine = fm.F.tensors.get_item(positions, index=0)
            wide = fm.F.ntt.vectorized_cast(embedded, fm.vector_type("float32", (4, )), (1, ), name="wide")
            outputs = [wide, cosine]
            if shared == "public":
                outputs.append(embedded)
            elif shared == "data":
                outputs.append(fm.F.math.add(embedded, embedded))
            self.function("main", (ids, weight, state), tuple(outputs))

    return Graph(dialect="high_level", stage="boundary_layout_propagated", entry="main").build(), state_config


def test_residual_rounding_absorbed_at_original_packing_fixed_point(tmp_path):
    source = residual_graph()
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    assert not any(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in result.nodes)
    fused = next(n for n in result.nodes if n.id == "sum")
    casts = [n for n in fused.attrs["pre_ops"]["lhs"].nodes if n.op == "ntt.vectorized_cast"]
    assert [n.attrs["new_type"]["elem_type"] for n in casts] == ["bfloat16", "float32"]
    assert all(n.attrs["rounding_contract"] == "preserve" for n in casts)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    values = {
        "x": torch.linspace(.991, 1.009, 16).reshape(1, 4, 2, 2), "y":
        torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    }
    old, new = evaluator.run(source, values), evaluator.run(result, values)
    assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(old, new))
    assert fm.load_module(fm.emit_module(result, tmp_path / "rounding.py")).semantic_hash == result.semantic_hash
    assert post_function_boundary_pack_propagation(result, NvidiaSm90Target()).semantic_hash == result.semantic_hash


def test_direct_embedding_shape_reference_absorbs_only_boundary(tmp_path):
    source, config = direct_embedding_graph()
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    assert not any(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in result.nodes)
    assert len([n for n in result.nodes if n.op == "nn.embedding"]) == 1
    old_positions = source.node_map["positions"]
    new_positions = result.node_map["positions"]
    reference = get_definition("nn.rotary_embedding").reference
    assert reference.memory_effect == MemoryEffect.NONE
    assert (new_positions.type, new_positions.effect) == (old_positions.type, old_positions.effect)
    assert [n.id for n in result.nodes if not n.effect.is_pure] == [n.id for n in source.nodes if not n.effect.is_pure]
    alias = result.node_map["wide"]
    assert alias.op == "tensors.bitcast" and result.node_map[alias.inputs[0]].type == alias.type
    evaluator = TorchEvaluator(DictWeightResolver({}))
    values = {
        "ids": torch.tensor([0, 6]), "weight": torch.arange(112).reshape(7, 4, 4).bfloat16() / 32, "state":
        create_paged_attention_state(config, device="cpu")
    }
    torch.testing.assert_close(evaluator.run(result, values), evaluator.run(source, values), rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(result, tmp_path / "embedding.py")).semantic_hash == result.semantic_hash


@pytest.mark.parametrize("shared", ("public", "data"))
def test_direct_producer_shared_data_and_public_value_rejected(shared):
    source, _ = direct_embedding_graph(shared)
    result = post_function_boundary_pack_propagation(source, NvidiaSm90Target())
    assert any(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in result.nodes)
    assert result.node_map["embedded"].type == source.node_map["embedded"].type


def test_large_model_path_uses_the_same_supported_absorption():
    source = residual_graph()
    padding = tuple(
        fm.Node(f"unused{i}", "builtin.scalar_const", (), fm.tensor_type("int64", ()), attrs={"value": 0})
        for i in range(20_001))
    large = replace(source, nodes=(*source.nodes, *padding))
    result = post_function_boundary_pack_propagation(large, NvidiaSm90Target())
    assert not any(n.op in {"tensors.cast", "ntt.vectorized_cast"} for n in result.nodes)
    body = result.node_map["sum"].attrs["pre_ops"]["lhs"]
    assert [n.attrs["new_type"]["elem_type"] for n in body.nodes[1:]] == ["bfloat16", "float32"]


def test_early_fused_casts_keep_normal_distribution_relations():
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    fused = post_function_boundary_pack_propagation(residual_graph(), NvidiaSm90Target())
    source = replace(fused, stage="unused_functions_removed")
    compiler = Compiler(CompileOptions(codegen_backend="triton"))
    proposed = compiler.compile(source, stop_after="propose-distribution").module
    assert proposed.metadata["auto_distribution_proposal"]["status"] == "OPTIMAL"
    # Single-candidate buckets intentionally have no editable SelectionPoint.
    # Normal application proves the selected storage relation is materializable.
    materialized = compiler.run_stage(proposed, "auto-distributed").module
    assert isinstance(materialized.node_map["sum"].type, fm.DistributedType)


@pytest.mark.parametrize("axes", ((1, ), (-1, ), (1, 1), (-1, -1)))
def test_fusion_forward_and_inverse_keep_storage_choices_and_split_units(axes):
    source = post_function_boundary_pack_propagation(residual_graph(axes), NvidiaSm90Target())
    node = source.node_map["sum"]
    definition = get_definition(node.op)
    mesh = fm.Placement((2, 2), "yx", "bb")
    broad, split = fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ), 1)
    choices = tuple(
        tuple(fm.DistributedType(source.node_map[name].type, (broad, policy), mesh)
              for policy in (broad, split))
        for name in node.inputs)
    inputs = tuple(definition.distributed_input_type_tuples(choices, node.attrs))
    assert inputs and all(all(value in offered for value, offered in zip(values, choices)) for values in inputs)
    outputs = definition.distributed_output_type_candidates(choices, node.type, node.attrs)
    assert outputs
    logical = tuple(source.node_map[name].type for name in node.inputs)
    for output in outputs:
        recovered = definition.infer_distributed_input_types(output, logical, node.attrs)
        assert recovered
        for storage in recovered:
            operands = tuple(replace(source.node_map[name], type=kind) for name, kind in zip(node.inputs, storage))
            assert definition.infer_call_type(operands, node.attrs) == output
    partial = fm.DistributedType(node.type, (broad, broad), mesh, partial=fm.SBP.partial((0, ), "sum"))
    assert definition.infer_distributed_input_types(partial, logical, node.attrs) == ()


def test_no_fusion_relations_delegate_to_original_protocols():
    definition = get_definition("math.add")
    mesh = fm.Placement((2, 2), "yx", "bb")
    kind = fm.DistributedType(fm.tensor_type("float32", (4, )), (fm.SBP.broadcast(), ), mesh)
    choices, attrs = ((kind, ), (kind, )), {}
    for name, args in (("distributed_input_type_tuples", (choices, attrs)), ("distributed_output_type_candidates",
                                                                             (choices, kind, attrs)),
                       ("infer_distributed_input_types", (kind, (kind.tensor, kind.tensor), attrs))):
        method = getattr(definition, name)
        old = method.__func__.__wrapped__(definition, *args)
        new = method(*args)
        assert tuple(new) == tuple(old)


def test_postops_output_and_inverse_use_preconversion_result_type():
    from triton.flagmega.ir.op_fusion import _inverse_post_types
    raw = fm.tensor_type("bfloat16", (4, ))

    @fm.fusion(raw)
    def widen(x):
        return fm.F.tensors.cast(x, "float32", rounding_contract="preserve")

    mesh = fm.Placement((2, 2), "yx", "bb")
    value = fm.DistributedType(raw, (fm.SBP.split_contiguous((0, ), 1), ), mesh)
    attrs = {"post_ops": (widen, )}
    definition = get_definition("math.add")
    output, = definition.distributed_output_type_candidates(((value, ), (value, )), widen.output_type, attrs)
    assert definition.infer_distributed_input_types(output, (raw, raw), attrs) == ((value, value), )
    tuple_result = fm.TupleType((output, value))
    assert _inverse_post_types(tuple_result, {"post_ops": (widen, None)}) == (fm.TupleType((value, value)), )


def test_unknown_fusion_inverse_stays_undeclared():
    raw = fm.tensor_type("float32", (4, ))

    @fm.fusion(raw)
    def shared_expression(x):
        return fm.F.math.mul(x, x)

    mesh = fm.Placement((2, 2), "yx", "bb")
    value = fm.DistributedType(raw, (fm.SBP.broadcast(), ), mesh)
    definition = get_definition("math.add")
    assert definition.infer_distributed_input_types(value, (raw, raw), {"pre_ops": {"lhs": shared_expression}}) is None
