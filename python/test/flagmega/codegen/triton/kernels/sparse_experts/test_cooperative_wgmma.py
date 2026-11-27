# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Chip-cooperative FP8 blocks retain the ordered WGMMA scale merge."""

from copy import deepcopy
from dataclasses import replace
from math import prod
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.sparse_experts import sparse_experts_cooperative_implementations

from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_wgmma_runtime import expert_module

IDENTITY = "tir.sparse_experts_gate_up.cooperative_wgmma"


def cooperative_target(warps=8):
    target = NvidiaSm90Target()
    base = target.triton_implementation_model
    implementations = tuple(
        replace(value, parameters={**value.parameters, "compute_num_warps": warps})
        for value in sparse_experts_cooperative_implementations()
    )
    target.triton_implementation_model = replace(
        base,
        implementations=(*base.implementations, *implementations),
        preferences={**base.preferences, "sparse_experts_gate_up": (IDENTITY,)},
    )
    return target


def compile_cooperative(graph, warps=8):
    compiler = Compiler()
    compiler.target = cooperative_target(warps)
    return compiler.compile(graph).module


def test_cooperative_workspace_and_effect_contract(tmp_path):
    compiled = compile_cooperative(expert_module("sparse_experts_gate_up"))
    (dispatch,) = (
        value.dispatch
        for value in compiled.kernel_definitions
        if value.dispatch.semantic_op == "nn.sparse_experts_gate_up"
    )
    assert dispatch.microkernel.implementation == IDENTITY
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 1
    assert dispatch.microkernel.facts["source_memory_scope"] == "chip"
    assert {"mma_v3", "fp8", "grid_sync", "cooperative_grid"} <= set(dispatch.microkernel.requires)
    partials, scales = dispatch.workspaces
    assert partials.type == fm.tensor_type("float32", (1, 2, 2, 2, 2, 64))
    assert scales.type == fm.tensor_type("float32", (1, 2, 2))
    assert all(
        value.memory_space == "workspace" and value.lifetime.value == "invocation" for value in dispatch.workspaces
    )
    assert all(effect.scope.value == "chip" for name, effect in dispatch.memory_effects if name in dispatch.arguments)
    resumed = fm.load_module(fm.emit_module(compiled, tmp_path / "selected.py"))
    assert resumed.semantic_hash == compiled.semantic_hash


def test_cooperative_codegen_roundtrip(tmp_path):
    compiled = compile_cooperative(expert_module("sparse_experts_gate_up"))
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: sparse_experts_gate_up/cooperative_wgmma" in source
    assert "__partials(" in source and "__merge(" in source


def stage_graph(
    *, tokens=1, routes=8, rows=512, reduction=4096, experts=11, packed=False, placement=None, readonly=False,
    private_result=False, publish_result=False,
):
    lanes = (8,) if packed else ()
    activation_dtype = fm.vector_type("bfloat16", lanes) if packed else "bfloat16"
    lane_count = prod(lanes)
    types = {
        "dispatched": fm.tensor_type(activation_dtype, (tokens, routes, reduction // lane_count)),
        "router_expert_ids": fm.tensor_type("int32", (tokens, routes)),
    }
    broadcast = fm.SBP.broadcast()
    block_axes = tuple(i for i, level in enumerate(placement.hierarchy_levels) if level == "b") if placement else ()
    row_axes = (
        tuple(i for i, level in enumerate(placement.hierarchy_levels) if level != "b") + block_axes[-1:]
        if placement
        else ()
    )
    route_axes = block_axes[:-1]
    route = (
        fm.SBP.split_contiguous(route_axes, -(-routes // prod(placement.hierarchy[i] for i in route_axes)))
        if placement
        else broadcast
    )
    row_capacity = (
        -(-rows // (prod(placement.hierarchy[i] for i in row_axes) * lane_count)) * lane_count if placement else rows
    )
    row = fm.SBP.split_contiguous(row_axes, row_capacity) if placement else broadcast
    for field in ("gate", "up"):
        types[field + "_weight"] = fm.tensor_type("float8_e4m3fn", (experts, rows, reduction))
        types[field + "_input_scale"] = fm.tensor_type("float32", (experts, 1))
        types[field + "_proj_scale"] = fm.tensor_type("float32", (experts, -(-rows // 128), reduction // 128))
    if placement:
        policies = {
            "dispatched": (broadcast, route, broadcast),
            "router_expert_ids": (broadcast, route),
            "gate_weight": (broadcast, row, broadcast),
            "up_weight": (broadcast, row, broadcast),
        }
        types = {
            name: fm.DistributedType(t, policies.get(name, (broadcast,) * t.rank), placement)
            for name, t in types.items()
        }

    class Graph(fm.Module):
        def forward(self):
            parameters, operands = [], []
            for parameter in SparseExpertsGateUp.input_parameters:
                name = parameter.name
                if readonly and name.endswith("_weight"):
                    value = self.weight(name, types[name], source="memory", key=name, id=name)
                else:
                    value = self.input(name, types[name], id=name)
                    parameters.append(value)
                operands.append(value)
            value = fm.F.nn.sparse_experts_gate_up(
                *operands,
                output_dtype=activation_dtype,
                activation_quantization="dynamic_fp8_ue8m0",
                weight_block_n=128,
                weight_block_k=128,
                round_projections=True,
                round_activation=True,
                gate_clamp_max=10.0,
                up_clamp=10.0,
                name="projection",
            )
            if private_result:
                dtype = fm.vector_type("float32", lanes) if packed else fm.DType.FLOAT32
                value = fm.F.tensors.cast(value, dtype, name="widened")
            if publish_result:
                value = fm.F.distributed.force_boxing(value, fm.logical_type(value.type), name="published_gate_up")
            sentinel = self.input("sentinel", fm.tensor_type("float32", (128,)), id="sentinel")
            copied = fm.F.math.add(sentinel, sentinel)
            if placement:
                sentinel_granularity = -(-128 // prod(placement.hierarchy[i] for i in block_axes))
                split = fm.DistributedType(sentinel.type, (fm.SBP.split_contiguous(block_axes, sentinel_granularity),), placement)
                local = fm.F.distributed.force_boxing(sentinel, split)
                copied = fm.F.distributed.force_boxing(fm.F.math.add(local, local), sentinel.type)
            self.function("main", (*parameters, sentinel), (value, copied))

    return Graph(
        dialect="distributed" if placement else "nn",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}} if placement else {},
    ).build()


def compile_stage(graph, placement, cooperative, warps=8):
    compiler = Compiler(CompileOptions(placement=placement))
    if cooperative:
        target = cooperative_target(warps)
    else:
        target = NvidiaSm90Target()
        base = target.triton_implementation_model
        target.triton_implementation_model = replace(
            base,
            implementations=tuple(
                replace(value, parameters={**value.parameters, "compute_num_warps": warps})
                if value.id == "tir.sparse_experts_gate_up.wgmma"
                else value
                for value in base.implementations
            ),
            preferences={**base.preferences, "sparse_experts_gate_up": ("tir.sparse_experts_gate_up.wgmma",)},
        )
    compiler.target = target
    return compiler.compile(graph).module


class TensorCheckpoint:
    def __init__(self, values):
        self.values = values

    def load_tensor(self, key, *, device="cpu"):
        return self.values[key].to(device)


def stage_values(torch, graph):
    rng = torch.Generator().manual_seed(1019)
    values = {}
    for node in graph.nodes:
        if node.op not in {"builtin.var", "builtin.weight"}:
            continue
        name = node.id
        kind = fm.logical_type(node.type)
        shape = tuple(dim.fixed_value for dim in kind.shape)
        if name == "router_expert_ids":
            experts = fm.logical_type(graph.node_map["gate_weight"].type).shape[0].fixed_value
            values[name] = (torch.arange(prod(shape)).reshape(shape) * (experts - 1)).remainder(experts).int()
        elif name == "sentinel":
            values[name] = torch.arange(128, dtype=torch.float32)
        elif name.endswith("_scale"):
            values[name] = (torch.rand(shape, generator=rng) + 0.25) * (
                2.0**-12 if name.endswith("proj_scale") else 1.0
            )
        else:
            lanes = kind.dtype.lanes if isinstance(kind.dtype, fm.VectorType) else ()
            dtype = kind.dtype.elem_type if lanes else kind.dtype
            values[name] = (torch.randint(-16, 17, (*shape, *lanes), generator=rng).float() / 16).to(
                getattr(torch, dtype.value)
            )
    return values


@pytest.mark.parametrize(
    "tokens,routes,rows,reduction,packed,warps,distributed",
    [
        (1, 2, 72, 256, False, 8, False),
        (1, 8, 512, 4096, False, 8, True),
        (2, 16, 640, 1536, True, 16, True),
    ],
)
def test_cooperative_complete_stage_bitwise(tmp_path, tokens, routes, rows, reduction, packed, warps, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    placement = fm.Placement((8, 16), "yx", "bb") if distributed else None
    graph = stage_graph(
        tokens=tokens,
        routes=routes,
        rows=rows,
        reduction=reduction,
        packed=packed,
        placement=placement,
        readonly=True,
        experts=max(11, routes + 1),
    )
    values = stage_values(torch, graph)
    checkpoint = TensorCheckpoint(values)
    runtimes, outputs, guards = [], [], []
    for cooperative in (False, True):
        compiled = compile_stage(graph, placement, cooperative, warps)
        artifact = write_artifact(
            compiled, tmp_path / str(cooperative), target="nvidia-sm90", emit_executable=True, checkpoint=checkpoint
        )
        runtime = load(artifact, device="cuda:0")
        runtimes.append(runtime)
        current = {}
        for spec in runtime.external_arguments:
            if spec["role"] != "result":
                continue
            name = spec["value"]
            shape = (tokens, routes, rows // 8, 8) if packed else (tokens, routes, rows)
            dtype = torch.bfloat16
            if name != compiled.function_map[compiled.entry].outputs[0]:
                shape, dtype = (128,), torch.float32
            storage = torch.full((prod(shape) + 64,), 123.0, dtype=dtype, device="cuda")
            current[name] = storage[: prod(shape)].reshape(shape)
            guards.append(storage[prod(shape) :])
        outputs.append(current)
    try:
        device_values = {name: value.cuda() for name, value in values.items() if not name.endswith("_weight")}
        for mutation in range(3):
            if mutation:
                device_values["dispatched"].add_(1 / 128)
                device_values["router_expert_ids"].add_(1).remainder_(values["gate_weight"].shape[0])
                device_values["gate_proj_scale"].mul_(1.03125)
            for runtime, output in zip(runtimes, outputs, strict=True):
                arguments = tuple(({**device_values, **output})[spec["value"]] for spec in runtime.external_arguments)
                GeneratedTirCallGraphModule.prepare(runtime, *arguments)
                GeneratedTirCallGraphModule.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                assert runtime.resource_report["spill_bytes"] == 0
                assert runtime.resource_report["grid"] == ([128, 1, 1] if distributed else [1, 1, 1])
                torch.testing.assert_close(tuple(output.values())[1], device_values["sentinel"] * 2, rtol=0, atol=0)
            for before, after in zip(outputs[0].values(), outputs[1].values(), strict=True):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
            assert all(torch.all(guard == 123.0) for guard in guards)
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.fixture(scope="module")
def cooperative_raw():
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis

    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    graph = stage_graph(placement=placement, rows=2048, readonly=True)
    compiled = compile_stage(graph, placement, True)
    raw = next(x for x in describe_kernel_call_abis(compiled) if x["family"] == "sparse_experts_gate_up")
    for group in ("inputs", "outputs", "workspaces"):
        for arg in raw[group]:
            for binding in arg["buffers"]:
                binding.setdefault("runtime_argument", arg["formal"])
    return raw


@pytest.mark.parametrize("chip", [0, 3])
def test_chip_peer_addresses_match_compact_component_abi(cooperative_raw, chip):
    import torch
    from triton.flagmega.codegen.triton.sparse_experts.gate_up import sparse_experts_gate_up_call

    raw = deepcopy(cooperative_raw)
    call = sparse_experts_gate_up_call(raw)
    weight = next(arg["buffers"][0] for arg in raw["inputs"] if arg["formal"] == "gate_weight")["abi"]
    assert weight["storage_kind"] == "compact_per_chip"
    local_rows, k = weight["local_capacity_shape"][-2:]

    class Pointer(int):
        def to(self, dtype):
            return self

    tl = SimpleNamespace(pointer_type=lambda dtype: None, float8e4nv=None, bfloat16=None, float32=None, int32=None)
    for tile in (0, 7):
        n = tile * 64 + torch.arange(64, dtype=torch.int64)
        ks = 31 * 128 + torch.arange(128, dtype=torch.int64)
        env = {
            "tl": tl,
            "shard_coord0": chip,
            "shard_coord1": 7,
            "shard_coord2": 9,
            "_fm_token": 0,
            "_fm_local_route": 0,
            "_fm_route_owner": 6,
            "_fm_expert": 10,
            "_fm_row_owner": n // local_rows,
            "_fm_local_n": n % local_rows,
            "_fm_k": ks,
            "_fm_k_start": 31 * 128,
            "_fm_k_block": 31,
        }
        for group in ("inputs", "outputs", "workspaces"):
            for arg in raw[group]:
                for binding in arg["buffers"]:
                    env[binding["runtime_argument"]] = Pointer(0)
        actual = eval(call["gate_peer_pointer"], {}, env) + eval(call["gate_peer_offset"], {}, env)
        expected = (
            weight["pool_byte_offset"]
            + n[:, None] // local_rows * weight["component_stride_scalar_elements"]
            + 10 * local_rows * k
            + (n[:, None] % local_rows) * k
            + ks[None, :]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        scale = eval(call["gate_peer_scale_offset"], {}, env)
        torch.testing.assert_close(scale, 10 * 16 * 32 + (chip * 512 + n) // 128 * 32 + 31, rtol=0, atol=0)
        source = eval(call["peer_input_offset"], {}, env)
        torch.testing.assert_close(source, 6 * 4096 + ks, rtol=0, atol=0)


@pytest.mark.parametrize(
    "group,formal,changes",
    [
        ("inputs", "dispatched", {"memory_sharing_scope": "block"}),
        ("inputs", "gate_weight", {"memory_sharing_scope": "block"}),
        ("inputs", "gate_weight", {"distributed_backing_type": {"partial": "sum"}}),
        ("inputs", "gate_weight", {"view_byte_offset": 0, "view_byte_offset_expression": "0"}),
        ("workspaces", "projection_partials", {"scalar_dtype": "bfloat16"}),
        ("workspaces", "activation_scales", {"memory_sharing_scope": "block"}),
    ],
)
def test_cooperative_rejects_unproved_physical_access(cooperative_raw, group, formal, changes):
    from triton.flagmega.codegen.triton.sparse_experts.gate_up import sparse_experts_gate_up_call
    from triton.flagmega.errors import CodegenError

    raw = deepcopy(cooperative_raw)
    arg = next(value for value in raw[group] if value["formal"] == formal)
    arg["buffers"][0]["abi"].update(changes)
    with pytest.raises(CodegenError, match="Cooperative FP8 experts"):
        sparse_experts_gate_up_call(raw)


def test_optional_implementation_does_not_change_generic_target():
    assert NvidiaSm90Target().triton_implementation_model.implementation(IDENTITY) is None
    (implementation,) = sparse_experts_cooperative_implementations()
    from triton.flagmega.targets.nvidia_sm120 import NvidiaSm120Target

    assert "mma_v3" in NvidiaSm120Target().capability.missing(implementation.requires)


@pytest.mark.parametrize(
    "mode,block,reduction,supported",
    [
        ("dynamic_fp8_ue8m0", 128, 256, True),
        ("dynamic_fp8_ue8m0", 128, 1536, True),
        ("dynamic_fp8_ue8m0", 64, 256, False),
        ("dynamic_fp8_ue8m0", 128, 192, False),
        ("dynamic_fp8", 128, 256, False),
    ],
)
def test_cooperative_candidate_keeps_the_block_quantization_contract(mode, block, reduction, supported):
    graph = expert_module("sparse_experts_gate_up", quantization=mode, block=block, reduction=reduction)
    proposed = cooperative_target().propose_tir(graph)
    point = next(value for value in proposed.selection_points if value.owner == "projection")
    assert (IDENTITY in {value.id for value in point.candidates}) is supported
