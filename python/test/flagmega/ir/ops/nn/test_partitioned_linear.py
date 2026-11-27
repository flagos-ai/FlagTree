# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Partition rounding, packed ABI, and only numerically legal K ownership."""

from types import SimpleNamespace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.partitioned_linear import PartitionedLinear


def graph(*, k=2048, n=80, parts=4, lanes=(), input_lanes=(), placement=None,
          k_policy=None, n_policy=None, materialize=False, **changes):
    class Graph(fm.Module):
        def forward(self):
            dtype = fm.vector_type("bfloat16", input_lanes) if input_lanes else fm.DType.BFLOAT16
            x = self.input("x", changes.get("x", fm.tensor_type(dtype, (1, k // getattr(dtype, "lane_count", 1)))), id="x")
            w = self.input("w", changes.get("w", fm.tensor_type("bfloat16", (n, k))), id="w")
            value, weight = x, w
            if placement:
                kp, np = k_policy or fm.SBP.broadcast(), n_policy or fm.SBP.broadcast()
                value = fm.F.distributed.sharded_view(x, fm.DistributedType(x.type, (fm.SBP.broadcast(), kp), placement))
                weight = fm.F.distributed.sharded_view(w, fm.DistributedType(w.type, (np, kp), placement))
            result = fm.F.nn.partitioned_linear(value, weight, partitions=parts, output_lanes=lanes, name="projection")
            if materialize:
                result = fm.F.distributed.force_boxing(result, fm.logical_type(result.type), name="result")
            self.function("main", (x, w), (result,))

    return Graph(dialect="distributed" if placement else "high_level",
                 stage="frozen_constants" if placement else "imported", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}} if placement else {}).build()


@pytest.mark.parametrize("lanes,input_lanes", [((), ()), ((2, 4), (2, 4))])
def test_partition_rounding_evaluation_and_resume(tmp_path, lanes, input_lanes):
    module = graph(k=128, lanes=lanes, input_lanes=input_lanes)
    restored = fm.load_module(fm.emit_module(module, tmp_path / "projection.py"))
    assert restored.semantic_hash == module.semantic_hash
    generator = torch.Generator().manual_seed(873)
    values = {node.id: torch.randint(-7, 8, tuple(d.fixed_value for d in node.type.shape)
                                    + tuple(getattr(node.type.dtype, "lanes", ())), generator=generator).float().div_(128).bfloat16()
              for node in module.nodes if node.op == "builtin.var"}
    actual, = TorchEvaluator(DictWeightResolver({})).run(restored, values)
    x, w = values["x"].reshape(1, 128).float(), values["w"].float()
    expected = sum((x[:, i:i + 32] @ w[:, i:i + 32].T).bfloat16().float() for i in range(0, 128, 32)).bfloat16()
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)
    node = module.node_map["projection"]
    assert node.effect == fm.effect("pure")
    assert PartitionedLinear.cost(node).bytes_written == 160


def test_rounding_each_partition_is_not_one_complete_matmul():
    module = graph(k=4, n=1, parts=2)
    x = torch.ones((1, 4), dtype=torch.bfloat16)
    w = torch.tensor([[1., 1 / 256, 1 / 256, 0.]], dtype=torch.bfloat16)
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, {"x": x, "w": w})
    assert actual.item() == 1.
    assert (x.float() @ w.float().T).bfloat16().item() == 1.0078125


@pytest.mark.parametrize("change", [
    {"parts": 0}, {"parts": True}, {"parts": 3}, {"lanes": (3,)}, {"lanes": (True,)},
    {"x": fm.tensor_type("float32", (1, 2048))}, {"x": fm.tensor_type("bfloat16", (2, 2048))},
    {"w": fm.tensor_type("bfloat16", (80, 1024))},
])
def test_invalid_shape_dtype_and_partition_contracts(change):
    with pytest.raises(IRSchemaError, match="PartitionedLinear"):
        graph(**change)


@pytest.mark.parametrize("axis", ["n", "k"])
def test_distribution_and_inverse_are_in_scalar_units(axis):
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    split = fm.SBP.split_contiguous((0,))
    module = graph(n=128, lanes=(2, 4), input_lanes=(2, 4), placement=placement,
                   **{f"{axis}_policy": split})
    node = module.node_map["projection"]
    assert node.type.partial == (fm.SBPPartial((0,)) if axis == "k" else None)
    assert node.type.axis_policies[1] == (split if axis == "n" else fm.SBP.broadcast())
    types = tuple(module.node_map[name].type for name in node.inputs)
    assert PartitionedLinear.infer_distributed_input_types(node.type, tuple(t.tensor for t in types), node.attrs) == (types,)
    assert tuple(PartitionedLinear.distributed_input_type_tuples(tuple((t,) for t in types), node.attrs)) == (types,)
    factors = PartitionedLinear.cost_factors(tuple(module.node_map[name] for name in node.inputs), node.attrs, node.type)
    assert factors.chip_aggregate_memory_load_bytes >= 128 * 2048 * 2 // 4


@pytest.mark.parametrize("chips,lanes,split", [
    (2, (), fm.SBP.split_contiguous((0,))), (8, (), fm.SBP.split_contiguous((0,))),
    (4, (), fm.SBP.split_contiguous((1,))), (4, (), fm.SBP.split_block_cyclic((0,), 32)),
])
def test_k_sharding_must_match_semantic_partition_boundaries(chips, lanes, split):
    with pytest.raises(IRSchemaError, match="PartitionedLinear"):
        graph(placement=fm.Placement((chips, 128), "cy", "cb"), k_policy=split, input_lanes=lanes)


def test_reject_partial_inputs_instead_of_merging_rounded_partitions():
    module = graph(placement=fm.Placement((4, 128), "cy", "cb"))
    x, w = (module.node_map[name] for name in module.node_map["projection"].inputs)
    bad = SimpleNamespace(type=fm.DistributedType(x.type.tensor, x.type.axis_policies, x.type.placement,
                                               partial=fm.SBPPartial((0,))))
    with pytest.raises(IRSchemaError, match="no input partials"):
        PartitionedLinear.infer_type((bad, w), module.node_map["projection"].attrs)


def test_lane_group_must_not_cross_semantic_partition():
    # A distributed type already rejects this split into more owners than
    # input groups; keep that rejection as part of the complete contract.
    with pytest.raises(IRSchemaError, match="policies are not distributable"):
        graph(placement=fm.Placement((4, 128), "cy", "cb"),
              k_policy=fm.SBP.split_contiguous((0,)), input_lanes=(1024,))
