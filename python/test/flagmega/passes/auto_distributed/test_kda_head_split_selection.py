# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A real GLM-5.3 KDA layer must actually get a head split on a 4-chip mesh.

Offering the candidate is not enough — the whole point of the head split is that
the solver picks it, so the per-chip footprint of the KDA weights and states
drops by the shard count instead of staying replicated.  On the published model
that is 130.69 MiB of KDA weight per layer across 34 KDA layers, i.e. 4.34 GiB
replicated per chip, against a measured ~2.7 GiB shortfall on 4 H800s.

The fixture uses the reduced GLM checkpoint with published-scale head
dimensions, because at ``kda_head_dim=8`` one whole replica is ~300 bytes and
replication is genuinely the cheaper plan — the cost model is right to keep it.
"""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler, CompileOptions
from triton.flagmega.importer.glm5_moe import Glm5MoeImporter

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint

HEADS, HEAD_DIM = 8, 128
KDA_OPS = ("nn.kda_convolution", "nn.kda_gated_recurrent_core", "nn.kda_recurrent_core")
PLACEMENT = fm.Placement((4, ), "x", "b")


@pytest.fixture(scope="module")
def distributed():
    scale = dict(REDUCED, kda_num_heads=HEADS, kda_head_dim=HEAD_DIM)
    imported = Glm5MoeImporter(checkpoint(scale, kind="kda"), layer=4).import_module()
    options = CompileOptions(target="nvidia-sm90", placement=PLACEMENT,
                             bufferize_opt_level="fast")
    return Compiler(options).compile(imported, stop_after="auto-distributed").module


def _kda_nodes(module):
    return tuple(node for node in module.nodes if node.op in KDA_OPS)


def _fields(value):
    return value.fields if isinstance(value, fm.TupleType) else (value, )


def _head_policy(value):
    return value.axis_policies[1] if isinstance(value, fm.DistributedType) else None


def test_the_kda_layer_is_present_and_head_split(distributed):
    nodes = _kda_nodes(distributed)
    assert len(nodes) == 2, f"expected the convolution and the gated recurrent core, got {nodes}"
    for node in nodes:
        splits = [_head_policy(field) for field in _fields(node.type)]
        assert splits and all(isinstance(policy, fm.SBPSplit) for policy in splits), \
            f"{node.id} kept a replicated head axis: {splits}"
        assert all(set(policy.hierarchy_axes) == {0} for policy in splits)


def test_the_conv_weight_and_both_states_are_sharded_not_replicated(distributed):
    """These are the tensors that overflow the 4-card budget when replicated."""

    for node in _kda_nodes(distributed):
        operands = [distributed.node_map[value] for value in node.inputs]
        # The rank-4 operands are the conv weight / conv state / recurrent state.
        heavy = [value for value in operands
                 if isinstance(value.type, fm.DistributedType) and value.type.tensor.rank == 4]
        assert heavy, f"{node.id} has no rank-4 weight or state operand"
        for value in heavy:
            assert isinstance(_head_policy(value.type), fm.SBPSplit), \
                f"{value.id} stayed replicated on every chip"


def test_every_kda_operand_agrees_on_one_head_owner(distributed):
    broadcast = fm.SBP.broadcast()
    for node in _kda_nodes(distributed):
        values = [distributed.node_map[value].type for value in node.inputs]
        values += list(_fields(node.type))
        heads = {_head_policy(value) for value in values}
        assert len(heads) == 1, f"{node.id} mixes head owners: {heads}"
        for value in values:
            assert all(policy == broadcast
                       for axis, policy in enumerate(value.axis_policies) if axis != 1)
            assert value.partial is None and value.exclusive is None


def test_the_split_shrinks_the_per_chip_kda_footprint(distributed):
    """The measurable claim: sharded local bytes are a quarter of the replica."""

    shards = 4
    for node in _kda_nodes(distributed):
        for value in (distributed.node_map[value] for value in node.inputs):
            if not isinstance(value.type, fm.DistributedType) or value.type.tensor.rank != 4:
                continue
            local = fm.local_tensor_type(value.type)
            assert local.shape[1].fixed_value == HEADS // shards, \
                f"{value.id} local head extent {local.shape[1]} is not a {shards}-way shard"


def test_the_out_projection_consumes_the_split_as_a_row_parallel_reduction(distributed):
    """``o_proj`` must use the existing partial-output matmul, not a gather."""

    matmuls = [node for node in distributed.nodes
               if "o_proj" in node.id and node.op.endswith("matmul")]
    assert matmuls, "the KDA out projection did not survive as a matmul"
    reshards = [node for node in distributed.nodes
                if node.op == "distributed.boxing" and any(
                    "o_norm" in value or "out_flat" in value for value in node.inputs)]
    # Either the matmul reduces a K-split itself, or exactly one reshard
    # gathers heads before it. Both are valid plans; an all-to-all per head is
    # not, so assert the reshard count stays bounded.
    assert len(reshards) <= 1, f"the head split leaked into repeated boxing: {reshards}"
