# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import TorchEvaluator, DictWeightResolver
from triton.flagmega.ir.ops.nn.mla_projection import MlaProjection
from python.test.flagmega.ir.ops.nn._mla_head_fixtures import graph, policies


@pytest.mark.parametrize("direction", ["absorb_query", "expand_value"])
@pytest.mark.parametrize("fp8,scale_sharded", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("cyclic,staged,levels", [(False, False, "bb"), (True, False, "bc"), (False, True, "bb")])
def test_complete_head_ownership_roundtrip_and_logical_values(
    tmp_path, direction, fp8, scale_sharded, cyclic, staged, levels
):
    mesh = fm.Placement((2, 2), "yx", levels)
    heads = 7 if cyclic else 8
    hp, rp = policies(mesh, heads=heads, cyclic=cyclic, staged=staged)
    m = graph(
        direction=direction, heads=heads, mesh=mesh, head_policy=hp, row_policy=rp, fp8=fp8, scale_sharded=scale_sharded
    )
    n = next(n for n in m.nodes if n.op == "nn.mla_projection")
    assert n.type.axis_policies == (fm.SBP.broadcast(), hp, fm.SBP.broadcast())
    restored = fm.load_module(fm.emit_module(m, tmp_path / "module.py"))
    assert restored.semantic_hash == m.semantic_hash
    width = 16 if direction == "absorb_query" else 32
    x = torch.arange(heads * width).float().remainder(17).sub_(8).mul_(0.125).bfloat16().reshape(1, heads, width)
    w = torch.arange(heads * 32 * 32).float().remainder(19).sub_(9).mul_(0.125).bfloat16().reshape(heads * 32, 32)
    values = {"x": x.reshape(1, heads, width // 8, 2, 4), "w": w}
    if fp8:
        values.update(
            w=w.to(torch.float8_e4m3fn),
            scale=torch.arange(heads * 4 * 4).float().remainder(4).add_(1).mul_(0.25).reshape(heads * 4, 4),
        )
        w = (values["w"].float() * values["scale"].repeat_interleave(8, 0).repeat_interleave(8, 1)).bfloat16()
    selected = w.reshape(heads, 32, 32)
    selected = selected[:, :16] if direction == "absorb_query" else selected[:, 16:].transpose(1, 2)
    expected = torch.bmm(x.transpose(0, 1).float(), selected.float()).bfloat16().transpose(0, 1)
    (actual,) = TorchEvaluator(DictWeightResolver({})).run(restored, values)
    assert torch.equal(actual.reshape(expected.shape), expected)


def test_correlated_candidates_preserve_split_ownership():
    mesh = fm.Placement((2, 2), "yx", "bb")
    hp, rp = policies(mesh)
    # Build both choices with legal inputs independently; infer_type must
    # reject cross-pairs, not discard the legal split candidate altogether.
    m = graph(mesh=mesh, head_policy=hp, row_policy=rp)
    n = next(n for n in m.nodes if n.op == "nn.mla_projection")
    split = tuple(m.node_map[i].type for i in n.inputs)
    broadcast = tuple(
        fm.DistributedType(t.tensor, (fm.SBP.broadcast(),) * t.tensor.rank, mesh)
        if isinstance(t, fm.DistributedType)
        else t
        for t in split
    )
    choices = tuple(tuple(dict.fromkeys((a, b))) for a, b in zip(split, broadcast))
    actual = tuple(MlaProjection.distributed_input_type_tuples(choices, n.attrs))
    assert split in actual and broadcast in actual
    assert len(actual) == 2


@pytest.mark.parametrize("which", ["feature", "weight_rows", "weight_columns", "scale_rows"])
def test_incomplete_or_mismatched_head_ownership_rejected(which):
    mesh = fm.Placement((2, 2), "yx", "bb")
    hp, rp = policies(mesh)
    changes = {
        "feature": {"source_changes": {1: fm.SBP.broadcast(), 2: hp}},
        "weight_rows": {"row_policy": fm.SBP.split_contiguous((0,), 128)},
        "weight_columns": {"weight_changes": {0: fm.SBP.broadcast(), 1: hp}},
        "scale_rows": {"scale_changes": {0: fm.SBP.split_contiguous((0,), 14)}},
    }[which]
    with pytest.raises(IRSchemaError, match="MLA projection"):
        graph(
            mesh=mesh,
            head_policy=hp,
            row_policy=rp if which != "weight_rows" else changes.pop("row_policy"),
            fp8=True,
            **changes,
        )
