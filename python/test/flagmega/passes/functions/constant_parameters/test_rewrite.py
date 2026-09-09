# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator, materialize_constant_assets
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.functions import function_nodes, lift_constant_parameter_expressions as lift
from .helpers import module


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("export_source", [False, True])
@pytest.mark.parametrize("distributed", [False, True])
def test_all_calls_keep_exact_values_reuse_and_exported_parameters(tmp_path, nested, export_source, distributed):
    original = module(nested=nested, export_source=export_source, distributed=distributed)
    before = original.semantic_hash
    result = lift(original)
    assert original.semantic_hash == before
    assert lift(result) == result
    worker = result.function_map["worker"]
    assert ("p" in worker.parameters) == export_source
    assert worker.attrs == original.function_map["worker"].attrs
    assert not any(n.op == "tensors.cast" for n in function_nodes(result, worker))
    weights = {f"weight{i}": (torch.arange(8, dtype=torch.float32) * .017 + i).bfloat16() for i in range(2)}
    resolver = DictWeightResolver(weights)
    evaluator = TorchEvaluator(resolver)
    inputs = {"runtime": torch.linspace(-2, 2, 8)}
    torch.testing.assert_close(evaluator.run(result, inputs), evaluator.run(original, inputs), rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(result, tmp_path / "module.py")) == result
    frozen = freeze_constant_islands(result)
    assets = materialize_constant_assets(frozen, resolver)
    for i in range(2):
        call = frozen.node_map[f"call{i}"]
        value = frozen.node_map[call.inputs[-1]]
        assert value.op == "builtin.const_asset"
        torch.testing.assert_close(assets[value.id], weights[f"weight{i}"].float() * 2, rtol=0, atol=0)
        assert value.metadata["rdata_group"] == {"name": "unit.weights", "index": i, "count": 2}


def test_shared_cast_boundary_becomes_one_extra_parameter_not_a_lost_user():
    original = module()
    output = original.node_map["output"]
    extra = replace(output, id="extra", inputs=("x", "cast"))
    updated = replace(
        original, nodes=(*original.nodes, extra), functions=tuple(
            replace(f, outputs=("output", "extra")) if f.name == "worker" else f for f in original.functions))
    updated = replace(
        updated, nodes=tuple(
            replace(n, type=fm.TupleType((output.type, output.type))) if n.op == "builtin.call" else n
            for n in updated.nodes))
    result = lift(fm.verify_module(updated))
    assert len(result.function_map["worker"].parameters) == 3
    assert result.node_map["extra"].inputs[-1] != result.node_map["output"].inputs[-1]
    assert result.node_map["extra"].inputs[-1] in result.function_map["worker"].parameters


def test_moved_selection_owners_are_removed_without_losing_realized_node_contracts():
    original = module()
    points = tuple(
        fm.SelectionPoint(f"choice.{name}", "distribution", (fm.Candidate("only"), ), "only", owner=name)
        for name in ("cast", "output"))
    records = tuple(fm.SelectionRecord(p.id, "only", "test", "test") for p in points)
    original = replace(original, selection_points=points, selections=records)
    result = lift(original)
    assert [p.owner for p in result.selection_points] == ["output"]
    assert [s.point_id for s in result.selections] == ["choice.output"]
