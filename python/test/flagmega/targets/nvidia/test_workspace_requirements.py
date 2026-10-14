# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
import pytest
from triton.flagmega.targets.nvidia import attach_workspace_requirements


def _candidate(family: str, variant: str) -> fm.Candidate:
    return fm.Candidate(
        f"tir.{family}.{variant}",
        {"family": family, "variant": variant},
        {},
    )


def _attach(node, candidate, inputs=()):
    module = fm.IRModule("semantic_tir", "target_proposed", (*inputs, node), (), "main")
    return attach_workspace_requirements(
        node,
        (candidate,),
        module,
        mesh_hierarchy=(8, 16),
    )[0].parameters.get("workspaces", ())


def test_local_rms_norm_has_no_private_partial_stats():
    node = fm.Node(
        "norm",
        "nn.rms_norm",
        (),
        fm.tensor_type("bfloat16", (1, 2048)),
    )

    workspaces = _attach(node, _candidate("rms_norm", "local"))

    assert workspaces == ()

@pytest.mark.parametrize("batch,split,local_batch", [(1, False, 1), (3, False, 3), (3, True, 2)])
def test_distributed_argmax_declares_value_and_index_reductions(batch, split, local_batch):
    logits_type = fm.tensor_type("float32", (batch, 37))
    if split:
        logits_type = fm.DistributedType(logits_type,
            (fm.SBP.split_block_cyclic((0,), 2), fm.SBP.broadcast()),
            fm.Placement((8, 16), "yx", "bb"))
    logits = fm.Node("logits", "builtin.var", (), logits_type, attrs={"name": "logits"})
    node = fm.Node(
        "sample",
        "nn.greedy_sample",
        (logits.id,),
        fm.tensor_type("int32", (batch,)),
    )

    workspaces = _attach(node, _candidate("greedy_sample", "distributed_argmax"), (logits,))

    assert tuple(value["name"] for value in workspaces) == (
        "partial_max",
        "partial_index",
    )
    assert workspaces[0]["type"] == fm.tensor_type("float32", (local_batch, 128))
    assert workspaces[1]["type"] == fm.tensor_type("int32", (local_batch, 128))


def test_gather_reduce_norm_stats_partials_follow_the_partial_input_row_count():
    # The partials kernel writes `shard_index * local_rows + row`, where local_rows
    # is the PARTIAL input's local row count. Here the partial input is token-
    # replicated with 4 local rows while the statistics output is token-sharded to
    # a single row -- sizing the workspace by the output would undersize it to
    # (mesh_size,); it must follow the input rows: (mesh_size, 4).
    mesh = fm.Placement((8, 16), "yx", "bb")
    partial_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (4, 2048)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        mesh,
    )
    partial = fm.Node("partial", "builtin.var", (), partial_type, attrs={"name": "partial"})
    addend = fm.Node("addend", "builtin.var", (), partial_type, attrs={"name": "addend"})
    node = fm.Node(
        "stats",
        "nn.add_norm_stats",
        (partial.id, addend.id),
        fm.TupleType((
            fm.tensor_type("bfloat16", (1, 2048)),
            fm.tensor_type("float32", (1, 1, 1)),
        )),
    )

    workspaces = _attach(
        node,
        _candidate("gather_reduce_add_norm_stats", "sum_rms"),
        (partial, addend),
    )

    partials = {value["name"]: value["type"] for value in workspaces}
    assert "norm_stats_partials" in partials
    assert partials["norm_stats_partials"] == fm.tensor_type("float32", (128, 4))


def test_gdn_recurrent_declares_cross_owner_core_scratch():
    node = fm.Node(
        "recurrent",
        "nn.gdn_recurrent_core",
        (),
        fm.tensor_type("bfloat16", (1, 6144)),
        attrs={"num_value_heads": 48, "value_head_dim": 128},
    )

    workspaces = _attach(node, _candidate("gdn_recurrent", "persistent"))

    assert tuple(value["name"] for value in workspaces) == ("core_scratch",)
    assert workspaces[0]["type"] == fm.tensor_type("float32", (48, 128))
    assert workspaces[0]["alignment"] == 128
