# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import pytest
from pathlib import Path
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.dsa_paged_attention import DsaPagedAttention
from python.test.flagmega.ir.ops.nn._mla_head_fixtures import policies
from python.test.flagmega.ir.ops.nn._dsa_head_fixtures import graph, compile_attention


@pytest.mark.parametrize("cyclic,levels", [(False, "bb"), (True, "bc")])
def test_dsa_enumerates_complete_head_shards(tmp_path, cyclic, levels):
    mesh = fm.Placement((2, 2), "yx", levels)
    heads = 7 if cyclic else 8
    hp, _ = policies(mesh, heads=heads, cyclic=cyclic)
    m = graph(mesh=mesh, head_policy=hp, heads=heads)
    n = next(n for n in m.nodes if n.op == "nn.dsa_paged_attention")
    assert n.type.axis_policies[1] == hp
    restored = fm.load_module(fm.emit_module(m, tmp_path / "model.py"))
    assert restored.semantic_hash == m.semantic_hash
    split = tuple(m.node_map[i].type for i in n.inputs)
    assert split in tuple(DsaPagedAttention.distributed_input_type_tuples(tuple((t,) for t in split), n.attrs))


@pytest.mark.parametrize("which", ["feature", "indices"])
def test_dsa_rejects_incomplete_reduction_or_indices(which):
    mesh = fm.Placement((2, 2), "yx", "bb")
    hp, _ = policies(mesh)
    change = (
        {"query_changes": {1: fm.SBP.broadcast(), 2: fm.SBP.split_contiguous((1,), 2)}}
        if which == "feature"
        else {"topk_policy": fm.SBP.split_contiguous((0,), 65)}
    )
    with pytest.raises(IRSchemaError, match="DSA"):
        graph(mesh=mesh, head_policy=hp, **change)


@pytest.mark.parametrize("schedule,traversal", [("bf16_mma", "compact"), ("bf16_mma", "slots"), ("simt", "slots")])
def test_dsa_head_shards_lower_through_normal_pipeline(tmp_path, schedule, traversal):
    mesh = fm.Placement((2, 2), "yx", "bb")
    hp, _ = policies(mesh, heads=7, cyclic=True)
    _, p = compile_attention(tmp_path, mesh=mesh, head_policy=hp, heads=7, schedule=schedule, traversal=traversal)
    s = (p / "generated_kernels.py").read_text()
    assert "flagmega-kernel: pack/" not in s and "flagmega-kernel: unpack/" not in s
