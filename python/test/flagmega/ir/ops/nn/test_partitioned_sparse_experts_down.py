# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep projection partitions material until their declared rounding is done."""

import subprocess
import sys
from dataclasses import replace
from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


class PartitionedProjection(fm.Module):
    def __init__(self, *, packed=False, lanes=(), rounded=False, mode="dynamic_fp8_ue8m0",
                 partitions=4, tokens=1, routes=2, hidden=16, reduction=128, block=128, combine=False, experts=3,
                 consumer_lanes=(), scalar_consumer=False, large=False, constants=False):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self.packed, self.lanes, self.rounded, self.mode = packed, lanes, rounded, mode
        self.partitions, self.tokens, self.routes = partitions, tokens, routes
        self.hidden, self.reduction, self.block = hidden, reduction, block
        self.combine, self.experts = combine, experts
        self.consumer_lanes, self.scalar_consumer, self.large = consumer_lanes, scalar_consumer, large
        self.constants = constants

    def forward(self):
        p, t, r, n, k = self.partitions, self.tokens, self.routes, self.hidden, self.reduction
        activation = (fm.tensor_type(fm.vector_type("bfloat16", (2, 2)), (t, r, p, k // 4)) if self.packed
                      else fm.tensor_type("bfloat16", (t, r, p, k)))
        types = {"activations": activation, "router_expert_ids": fm.tensor_type("int32", (t, r)),
                 "down_input_scale": fm.tensor_type("float32", (self.experts, 1)),
                 "down_weight": fm.tensor_type("float8_e4m3fn", (self.experts, p, n, k)),
                 "down_proj_scale": fm.tensor_type("float32", (self.experts, p, -(-n // self.block), -(-k // self.block)))}
        inputs, parameters = {}, []
        for name, kind in types.items():
            if self.constants and name in {"down_weight", "down_proj_scale", "down_input_scale"}:
                inputs[name] = self.weight(name, kind, source="memory", key=name, id=name)
            else:
                inputs[name] = self.input(name, kind, id=name)
                parameters.append(inputs[name])
        result = fm.F.nn.partitioned_sparse_experts_down(**inputs, weight_block_n=self.block,
            weight_block_k=self.block, activation_quantization=self.mode,
            round_projection=self.rounded, output_lanes=self.lanes, name="projection")
        outputs = (result,)
        if self.consumer_lanes:
            outputs = tuple(fm.F.tensors.pack(result, self.consumer_lanes, axes=(-1,) * len(self.consumer_lanes))
                            for _ in range(2))
            if self.scalar_consumer:
                outputs += (result,)
        if self.combine:
            coefficients = self.input("router_expert_weights", fm.tensor_type("float32", (t, r)), id="router_expert_weights")
            inputs["router_expert_weights"] = coefficients
            parameters.append(coefficients)
            coefficients = fm.F.tensors.broadcast_to(coefficients, shape=(p, t, r))
            coefficients = fm.F.tensors.reshape(coefficients, (p * t, r))
            scalar_hidden = n // prod(self.lanes)
            projected = fm.F.tensors.reshape(result, (p * t, r, scalar_hidden))
            bf16 = fm.vector_type("bfloat16", self.lanes) if self.lanes else fm.DType.BFLOAT16
            fp32 = fm.vector_type("float32", self.lanes) if self.lanes else fm.DType.FLOAT32
            local = fm.F.nn.sparse_experts_combine(projected, coefficients, output_dtype=bf16,
                                                  round_weighted_output=True, name="local_combine")
            local = fm.F.tensors.reshape(local, (p, t, scalar_hidden))
            wide = fm.F.tensors.cast(local, fp32)
            if self.lanes:
                wide = fm.F.tensors.unpack(wide, axes=(2,) * len(self.lanes))
            reduced = fm.F.math.reduce_sum(wide, axes=(0,), keep_dims=False, output_lanes=self.lanes)
            reduced = fm.F.tensors.cast(reduced, bf16)
            outputs = (local, reduced)
        if self.large:
            padding = inputs["activations"]
            for _ in range(20_001):
                padding = fm.F.math.add(padding, inputs["activations"])
            outputs += (padding,)
        self.function("main", parameters, outputs)


def operands(graph):
    p, t, r, n, k = graph.partitions, graph.tokens, graph.routes, graph.hidden, graph.reduction
    value = torch.zeros((t, r, p, k), dtype=torch.bfloat16)
    value[..., 0] = 129.
    if graph.packed:
        value = value.reshape(t, r, p, k // 4, 2, 2)
    weight = torch.zeros((graph.experts, p, n, k), dtype=torch.float8_e4m3fn)
    weight[..., 0] = 1.
    scale = torch.full((graph.experts, p, -(-n // graph.block), -(-k // graph.block)), 1.00390625)
    # Distinct partition scales expose any accidental P/route/N reordering.
    scale *= torch.arange(1, p + 1).float()[None, :, None, None]
    values = {"activations": value, "router_expert_ids": torch.zeros((t, r), dtype=torch.int32),
              "down_input_scale": torch.ones((graph.experts, 1)), "down_weight": weight, "down_proj_scale": scale}
    if graph.combine:
        values["router_expert_weights"] = torch.tensor((.3, .7)).expand(t, r).contiguous()
    return values


@pytest.mark.parametrize("packed,lanes", ((False, ()), (True, (2, 2))))
@pytest.mark.parametrize("rounded", (False, True))
def test_partitions_keep_wide_projections_and_declared_rounding(tmp_path, packed, lanes, rounded):
    graph = PartitionedProjection(packed=packed, lanes=lanes, rounded=rounded)
    module = graph.build()
    result, = TorchEvaluator(DictWeightResolver({})).run(module, operands(graph))
    expected = torch.arange(1, 5).float() * 128.5
    if rounded:
        expected = expected.bfloat16().float()
    expected = expected[:, None, None, None].expand(4, 1, 2, 16)
    torch.testing.assert_close(result.reshape(4, 1, 2, 16), expected, atol=0, rtol=0)
    loaded = fm.load_module(fm.emit_module(module, tmp_path / "module.py"))
    assert loaded.semantic_hash == module.semantic_hash


@pytest.mark.parametrize("lanes", ((), (2, 2)))
def test_route_rounding_precedes_partition_reduction(lanes):
    graph = PartitionedProjection(lanes=lanes, combine=True)
    local, actual = TorchEvaluator(DictWeightResolver({})).run(graph.build(), operands(graph))
    projections = (torch.arange(1, 5).float() * 128.5)[:, None].expand(4, 2)
    weights = torch.tensor((.3, .7))
    expected_local = (projections * weights).bfloat16().float().sum(1).bfloat16()
    expected = expected_local.float().sum().bfloat16()
    late = (projections.sum(0) * weights).bfloat16().float().sum().bfloat16()
    assert expected != late, "the fixture must distinguish local and global rounding"
    torch.testing.assert_close(local.reshape(4, 1, 16), expected_local[:, None, None].expand(4, 1, 16), atol=0, rtol=0)
    torch.testing.assert_close(actual.reshape(1, 16), torch.full((1, 16), expected.item(), dtype=torch.bfloat16), atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ("weight_dtype", "activation_dtype", "partition_shape", "scale_shape",
                                     "input_scale_shape", "rounding", "block", "lanes", "mode"))
def test_schema_rejects_unsupported_partition_contracts(invalid):
    module = PartitionedProjection().build()
    definition = fm.get_definition("nn.partitioned_sparse_experts_down")
    node = module.node_map["projection"]
    inputs = [module.node_map[name] for name in node.inputs]
    attrs = dict(node.attrs)
    replacements = {
        "weight_dtype": (3, fm.tensor_type("bfloat16", (3, 4, 16, 128))),
        "activation_dtype": (0, fm.tensor_type("float32", (1, 2, 4, 128))),
        "partition_shape": (0, fm.tensor_type("bfloat16", (1, 2, 3, 128))),
        "scale_shape": (4, fm.tensor_type("float32", (3, 3, 1, 1))),
        "input_scale_shape": (2, fm.tensor_type("float32", (3, 4, 1))),
    }
    if invalid in replacements:
        index, kind = replacements[invalid]
        inputs[index] = replace(inputs[index], type=kind)
    else:
        name, value = {"rounding": ("round_projection", 1), "block": ("weight_block_k", 0),
                       "lanes": ("output_lanes", (3,)), "mode": ("activation_quantization", "unknown")}[invalid]
        attrs[name] = value
    with pytest.raises(IRSchemaError):
        definition.prepare(inputs, attrs)


@pytest.mark.parametrize("invalid", ("split_k", "partition_mismatch", "overlapping_roles"))
def test_distribution_cannot_hide_a_reduction_or_change_partition_owners(invalid):
    module = PartitionedProjection(lanes=(2, 2)).build()
    node = module.node_map["projection"]
    d = fm.get_definition(node.op)
    mesh = fm.Placement((2, 2), "cy", "cb")
    b, p = fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1)
    n = fm.SBP.split_block_cyclic((1,), 8)
    k = fm.SBP.split_contiguous((1,), 64) if invalid == "split_k" else b
    weight_p = fm.SBP.split_contiguous((1,), 1) if invalid == "partition_mismatch" else p
    if invalid in {"split_k", "partition_mismatch"}:
        n = b
    if invalid == "overlapping_roles":
        n = fm.SBP.split_contiguous((0,), 8)
        weight_p = b
    policies = ((b, b, p, k), (b, b), (b, b), (b, weight_p, n, k), (b, b, b, b))
    inputs = tuple(replace(module.node_map[name], type=fm.DistributedType(module.node_map[name].type, policy, mesh))
                   for name, policy in zip(node.inputs, policies, strict=True))
    with pytest.raises(IRSchemaError):
        d.prepare(inputs, node.attrs)


@pytest.mark.parametrize("large", (False, True))
@pytest.mark.parametrize("scalar_consumer", (False, True))
def test_both_layout_paths_rebuild_one_shared_packed_producer(large, scalar_consumer):
    from triton.flagmega.passes.functions import post_function_boundary_pack_propagation
    from triton.flagmega.targets import NvidiaSm90Target

    graph = PartitionedProjection(consumer_lanes=(2, 2), scalar_consumer=scalar_consumer, large=large)
    original = graph.build()
    target = NvidiaSm90Target()
    rewritten = post_function_boundary_pack_propagation(original, target)
    producers = [node for node in rewritten.nodes if node.op == "nn.partitioned_sparse_experts_down"]
    assert len(producers) == 1 and producers[0].attrs["output_lanes"] == (2, 2)
    assert not any(node.op == "tensors.pack" for node in rewritten.nodes)
    assert post_function_boundary_pack_propagation(rewritten, target).semantic_hash == rewritten.semantic_hash
    if large:
        original = replace(original, functions=(replace(original.functions[0], outputs=original.functions[0].outputs[:-1]),))
        rewritten = replace(rewritten, functions=(replace(rewritten.functions[0], outputs=rewritten.functions[0].outputs[:-1]),))
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for got, expected in zip(evaluator.run(rewritten, operands(graph)), evaluator.run(original, operands(graph)), strict=True):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_scalar_and_packed_constructors_resume_in_a_fresh_process(tmp_path):
    arguments = []
    for lanes in ((), (2, 2)):
        module = PartitionedProjection(lanes=lanes).build()
        path = fm.emit_module(module, tmp_path / f"projection-{len(lanes)}.py")
        arguments.extend((str(path), module.semantic_hash))
    subprocess.run([sys.executable, "-c", '''
import sys
from triton.flagmega import ir as fm
for path, expected in zip(sys.argv[1::2], sys.argv[2::2]):
    module = fm.load_module(path)
    assert module.semantic_hash == expected
''', *arguments], check=True)
