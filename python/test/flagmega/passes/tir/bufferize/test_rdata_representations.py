# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.evaluator import DictWeightResolver, materialize_constant_assets
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def _sliced_weights(*, indices=(0, 1), prefix="", starts=(0, 2)):
    builder = fm.IRBuilder(dialect="high_level", stage="selected_tir")
    outputs = []
    for index in indices:
        weight = builder.weight(
            f"{prefix}layers.{index}.weight",
            fm.tensor_type("float32", (4, 4)),
            source="memory",
            key=f"{prefix}layers.{index}.weight",
            id=f"{prefix}w{index}",
            metadata={"rdata_group": {"name": "layers.weight", "index": index, "count": 2}},
        )
        for branch, start in enumerate(starts):
            outputs.append(
                builder.call(
                    "tensors.slice",
                    (weight, ),
                    fm.tensor_type("float32", (2, 4)),
                    attrs={"starts": (start, ), "ends": (start + 2, ), "axes": (0, ), "steps": (1, )},
                    id=f"{prefix}slice_{index}_{branch}",
                ))
    builder.function("main", (), outputs)
    return freeze_constant_islands(builder.build(entry="main"))


def _groups(plan):
    groups = {}
    for buffer in plan.buffers:
        if buffer.rdata_group:
            groups.setdefault(buffer.rdata_group, []).append(buffer)
    return groups


def test_slice_outputs_of_one_weight_group_have_independent_contiguous_storage():
    module = _sliced_weights(indices=(1, 0))
    plan = fm.make_buffer_plan(module)
    groups = _groups(plan)

    assert len(groups) == 2
    for name, members in groups.items():
        assert name.startswith("layers.weight.repr_")
        assert [member.group_index for member in members] == [0, 1]
        assert all(member.group_count == 2 for member in members)
        assert members[1].offset == members[0].offset + 256
        assert members[0].mem_span.buffer != members[1].mem_span.buffer
    spans = {(buffer.offset, buffer.nbytes) for members in groups.values() for buffer in members}
    assert len(spans) == 4  # Layout equivalence must never deduplicate different layer values.

    values = materialize_constant_assets(
        module,
        DictWeightResolver({
            f"layers.{index}.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4) + index * 32
            for index in range(2)
        }))
    for index in range(2):
        for branch in range(2):
            expected = torch.arange(branch * 8, branch * 8 + 8, dtype=torch.float32).reshape(2, 4) + index * 32
            torch.testing.assert_close(values[f"slice_{index}_{branch}"], expected, rtol=0, atol=0)


def test_representation_identity_ignores_layer_keys_node_names_and_recipe_output_order():
    original = _sliced_weights()
    renamed = _sliced_weights(indices=(1, 0), prefix="renamed_", starts=(2, 0))
    assert set(_groups(fm.make_buffer_plan(original))) == set(_groups(fm.make_buffer_plan(renamed)))
    assert original.constant_recipes[0].fingerprint != renamed.constant_recipes[0].fingerprint


def test_representation_identity_is_preserved_by_python_resume_and_bufferization(tmp_path):
    frozen = _sliced_weights()
    restored = fm.load_module(fm.emit_module(frozen, tmp_path / "frozen.py"))
    names = set(_groups(fm.make_buffer_plan(frozen)))
    assert set(_groups(fm.make_buffer_plan(restored))) == names
    buffered = NttBufferizationPolicy(BufferizationOptions.generic()).bufferize(restored)
    assert set(_groups(fm.make_buffer_plan(buffered))) == names


def test_edited_recipe_recomputes_physical_group_identity():
    frozen = _sliced_weights(starts=(0, ))
    edited = replace(
        frozen, constant_recipes=tuple(
            replace(
                recipe, nodes=tuple(
                    replace(node, attrs={**dict(node.attrs), "starts": (2, ), "ends": (
                                             4, )}) if node.op == "tensors.slice" else node
                    for node in recipe.nodes))
            for recipe in frozen.constant_recipes))
    assert set(_groups(fm.make_buffer_plan(frozen))).isdisjoint(_groups(fm.make_buffer_plan(edited)))


@pytest.mark.parametrize("kwargs", [{"indices": (0, )}, {"starts": (0, 0)}])
def test_each_representation_still_requires_complete_nonduplicate_indices(kwargs):
    with pytest.raises(IRVerificationError, match="requires indices 0..1 with no duplicates"):
        fm.make_buffer_plan(_sliced_weights(**kwargs))
