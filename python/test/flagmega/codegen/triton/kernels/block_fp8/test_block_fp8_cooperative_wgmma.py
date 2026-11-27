# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered FP8 projections cooperate over raw blocks without changing FMA order."""

from copy import deepcopy
from dataclasses import replace
from math import ceil, prod
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.block_fp8 import block_fp8_cooperative_implementations

IDENTITY = "tir.block_fp8.cooperative_wgmma"


def projection_graph(
    *,
    n=72,
    k=256,
    lanes=(),
    placement=None,
    readonly=False,
    block_n=128,
    block_k=128,
    quantization="dynamic_fp8_sm90",
    dequant=None,
    private_result=False,
    private_input=False,
):
    types = {
        "value": fm.tensor_type("float32" if private_input else "bfloat16", (1, k)),
        "weight": fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (n, k // 32)),
        "weight_scale": fm.tensor_type("float32", (ceil(n / block_n), ceil(k / block_k))),
    }
    axes = tuple(i for i, level in enumerate(placement.hierarchy_levels) if level == "b") if placement else ()
    if placement:
        owners = prod(placement.hierarchy)
        row = fm.SBP.split_contiguous(tuple(range(placement.rank)), n // owners)
        broad = fm.SBP.broadcast()
        types = {
            name: fm.DistributedType(t, (row, broad) if name == "weight" else (broad, broad), placement)
            for name, t in types.items()
        }

    class Graph(fm.Module):
        def forward(self):
            operands, parameters = {}, []
            for name, kind in types.items():
                if name == "weight" and readonly:
                    operands[name] = self.weight(name, kind, source="memory", key=name, id=name)
                else:
                    operands[name] = self.input(name, kind, id=name)
                    parameters.append(operands[name])
            if private_input:
                operands["value"] = fm.F.tensors.cast(operands["value"], "bfloat16", name="private_value")
            output = fm.F.math.packed_block_scaled_matmul(
                operands["value"],
                operands["weight"],
                operands["weight_scale"],
                k_pack=2,
                k_vector=16,
                weight_block_n=block_n,
                weight_block_k=block_k,
                output_lanes=lanes,
                activation_quantization=quantization,
                weight_dequant_dtype=dequant,
                name="projection",
            )
            if private_result:
                dtype = fm.vector_type("float32", lanes) if lanes else fm.DType.FLOAT32
                output = fm.F.tensors.cast(output, dtype, name="widened")
            sentinel = self.input("sentinel", fm.tensor_type("float32", (128,)), id="sentinel")
            copied = fm.F.math.add(sentinel, sentinel)
            if placement:
                kind = fm.DistributedType(sentinel.type, (fm.SBP.split_contiguous(axes, 1),), placement)
                local = fm.F.distributed.force_boxing(sentinel, kind)
                copied = fm.F.distributed.force_boxing(fm.F.math.add(local, local), sentinel.type)
            self.function("main", (*parameters, sentinel), (output, copied))

    return Graph(
        dialect="distributed" if placement else "nn",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": placement.to_data()}} if placement else {},
    ).build()


def projection_target(cooperative=True, warps=8):
    target = NvidiaSm90Target()
    base = target.triton_implementation_model
    implementations = tuple(
        replace(value, parameters={**value.parameters, "compute_num_warps": warps, "scaled_accumulation": "fma"})
        if value.id == "tir.block_fp8.mma"
        else value
        for value in base.implementations
    )
    if cooperative:
        implementations += tuple(
            replace(value, parameters={**value.parameters, "compute_num_warps": warps})
            for value in block_fp8_cooperative_implementations()
        )
    target.triton_implementation_model = replace(
        base,
        implementations=implementations,
        preferences={
            **base.preferences,
            "block_fp8": ((IDENTITY,) if cooperative else ()) + ("tir.block_fp8.mma", "tir.block_fp8.simt"),
        },
    )
    return target


def compile_projection(graph, placement=None, cooperative=True, warps=8):
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = projection_target(cooperative, warps)
    module = compiler.compile(graph).module
    if cooperative:
        dispatches = [d.dispatch for d in module.kernel_definitions if d.dispatch.microkernel.family == "block_fp8"]
        assert dispatches and all(d.microkernel.implementation == IDENTITY for d in dispatches)
    return module


def test_projection_workspace_and_effect_contract(tmp_path):
    module = compile_projection(projection_graph())
    (dispatch,) = (d.dispatch for d in module.kernel_definitions if d.dispatch.microkernel.family == "block_fp8")
    assert dispatch.microkernel.implementation == IDENTITY
    assert dispatch.microkernel.facts["source_memory_scope"] == "chip"
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 1
    assert [w.type for w in dispatch.workspaces] == [
        fm.tensor_type("float32", (2, 72)),
        fm.tensor_type("float32", (2,)),
    ]
    assert all(w.lifetime.value == "invocation" and w.memory_space == "workspace" for w in dispatch.workspaces)
    assert all(effect.scope.value == "chip" for name, effect in dispatch.memory_effects if name in dispatch.arguments)
    assert fm.load_module(fm.emit_module(module, tmp_path / "selected.py")).semantic_hash == module.semantic_hash
    assert NvidiaSm90Target().triton_implementation_model.implementation(IDENTITY) is None


def test_projection_ordinary_codegen(tmp_path):
    module = compile_projection(projection_graph())
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (path / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: block_fp8/cooperative_wgmma" in source
    assert "__partials(" in source and "__merge(" in source


class TensorCheckpoint:
    def __init__(self, values):
        self.values = values

    def load_tensor(self, key, *, device="cpu"):
        return self.values[key].to(device)


def projection_values(torch, graph):
    rng = torch.Generator().manual_seed(1023)
    values = {}
    for node in graph.nodes:
        if node.op not in {"builtin.var", "builtin.weight"}:
            continue
        kind = fm.logical_type(node.type)
        shape = tuple(dim.fixed_value for dim in kind.shape)
        lanes = kind.dtype.lanes if isinstance(kind.dtype, fm.VectorType) else ()
        dtype = kind.dtype.elem_type if lanes else kind.dtype
        if node.id == "weight_scale":
            values[node.id] = (torch.rand(shape, generator=rng) + 0.125) * 2.0**-12
        else:
            values[node.id] = (torch.randint(-16, 17, (*shape, *lanes), generator=rng).float() / 16).to(
                getattr(torch, dtype.value)
            )
    return values


@pytest.mark.parametrize(
    "n,k,lanes,warps,distributed",
    [
        (72, 256, (), 8, False),
        (1024, 4096, (8,), 8, True),
        (5120, 1536, (2, 4), 16, True),
    ],
)
def test_projection_complete_stage_bitwise(tmp_path, n, k, lanes, warps, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    placement = fm.Placement((8, 16), "yx", "bb") if distributed else None
    graph = projection_graph(n=n, k=k, lanes=lanes, placement=placement, readonly=True)
    values = projection_values(torch, graph)
    runtimes, results, guards, arguments = [], [], [], []
    for cooperative in (False, True):
        module = compile_projection(graph, placement, cooperative, warps)
        path = write_artifact(
            module,
            tmp_path / str(cooperative),
            target="nvidia-sm90",
            emit_executable=True,
            checkpoint=TensorCheckpoint(values),
        )
        runtime = load(path, device="cuda:0")
        current = {}
        for spec in runtime.external_arguments:
            if spec["role"] == "result":
                is_projection = spec["value"] == module.function_map[module.entry].outputs[0]
                shape, dtype = (
                    ((1, n // prod(lanes), *lanes), torch.bfloat16) if is_projection else ((128,), torch.float32)
                )
                storage = torch.full((prod(shape) + 64,), 123.0, dtype=dtype, device="cuda")
                current[spec["value"]] = storage[: prod(shape)].reshape(shape)
                guards.append(storage[prod(shape) :])
        runtimes.append(runtime)
        results.append(current)
    device_values = {name: value.cuda() for name, value in values.items() if name != "weight"}
    for runtime, current in zip(runtimes, results, strict=True):
        arguments.append(tuple(({**device_values, **current})[spec["value"]] for spec in runtime.external_arguments))
    for mutation in range(4):
        if mutation == 1:
            device_values["value"].zero_()
        elif mutation:
            device_values["value"].fill_(0.125 * mutation)
            device_values["weight_scale"].mul_(1.03125)
        for runtime, operands in zip(runtimes, arguments, strict=True):
            GeneratedTirCallGraphModule.prepare(runtime, *operands)
            GeneratedTirCallGraphModule.run_into(runtime, *operands)
            torch.cuda.synchronize()
            assert runtime.resource_report["spill_store_bytes"] == runtime.resource_report["spill_load_bytes"] == 0
        for before, after in zip(results[0].values(), results[1].values(), strict=True):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        assert all(torch.all(guard == 123.0) for guard in guards)
        torch.testing.assert_close(tuple(results[1].values())[1], device_values["sentinel"] * 2, rtol=0, atol=0)
    for runtime in runtimes:
        runtime.close()


@pytest.mark.parametrize(
    "quantization,block_k,k,dequant,supported",
    [
        ("dynamic_fp8_sm90", 128, 256, None, True),
        ("dynamic_fp8_sm90", 128, 1536, None, True),
        ("dynamic_fp8_sm90", 128, 160, None, False),
        ("dynamic_fp8_sm90", 64, 256, None, False),
        ("dynamic_fp8", 128, 256, None, False),
        ("none", 128, 256, "bfloat16", False),
    ],
)
def test_projection_candidate_keeps_its_numerical_contract(quantization, block_k, k, dequant, supported):
    graph = projection_graph(k=k, quantization=quantization, block_k=block_k, dequant=dequant)
    proposed = projection_target().propose_tir(graph)
    point = next(p for p in proposed.selection_points if p.owner == "projection")
    assert (IDENTITY in {candidate.id for candidate in point.candidates}) is supported


@pytest.fixture(scope="module", params=("cbb", "bcb"))
def projection_raw(request):
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis

    placement = (
        fm.Placement((4, 8, 16), "cyx", "cbb") if request.param == "cbb" else fm.Placement((8, 4, 16), "ycx", "bcb")
    )
    graph = projection_graph(n=4096, k=4096, lanes=(8,), placement=placement, readonly=True)
    module = compile_projection(graph, placement)
    raw = next(call for call in describe_kernel_call_abis(module) if call["family"] == "block_fp8")
    for group in ("inputs", "outputs", "workspaces"):
        for argument in raw[group]:
            for binding in argument["buffers"]:
                binding.setdefault("runtime_argument", argument["formal"])
    return raw, request.param


@pytest.mark.parametrize("chip", [0, 3])
def test_projection_chip_addresses_follow_the_actual_compact_abi(projection_raw, chip):
    import torch
    from triton.flagmega.codegen.triton.kernel_call_renderers import _block_fp8_call

    raw, levels = projection_raw
    call = _block_fp8_call(deepcopy(raw))
    weight = next(a["buffers"][0]["abi"] for a in raw["inputs"] if a["formal"] == "weight")
    assert weight["storage_kind"] == "compact_per_chip"
    local_n, packed_k = weight["local_capacity_shape"]
    k = packed_k * weight["scalar_lane_count"]
    assert local_n == 8 and k == 4096

    class Pointer(int):
        def to(self, dtype):
            return self

    tl = SimpleNamespace(
        pointer_type=lambda dtype: None, float8e4nv=None, bfloat16=None, float32=None, int64=None, uint8=None
    )
    for tile in (0, 15):
        rows = tile * 64 + torch.arange(64)
        ks = 31 * 128 + torch.arange(128)
        env = {
            "tl": tl,
            "shard_coord0": chip if levels == "cbb" else 7,
            "shard_coord1": 7 if levels == "cbb" else chip,
            "shard_coord2": 9,
            "_fm_owner": rows // local_n,
            "_fm_local_n": rows % local_n,
            "_fm_k": ks,
        }
        for group in ("inputs", "outputs", "workspaces"):
            for argument in raw[group]:
                for binding in argument["buffers"]:
                    env[binding["runtime_argument"]] = Pointer(0)
        actual = eval(call["weight_peer_pointer"], {}, env) + eval(call["weight_peer_offset"], {}, env)
        expected = (
            weight["pool_byte_offset"]
            + (rows[:, None] // local_n) * weight["component_stride_scalar_elements"]
            + (rows[:, None] % local_n) * k
            + ks[None, :]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        env["local_n_offsets"] = torch.arange(16)
        env["_fm_global_n"] = eval(call["global_n"], {}, env)
        expected_n = (
            (chip * 128 + 7 * 16 + 9) * local_n + torch.arange(16)
            if levels == "cbb"
            else ((7 * 4 + chip) * 16 + 9) * local_n + torch.arange(16)
        )
        torch.testing.assert_close(env["_fm_global_n"], expected_n, rtol=0, atol=0)
        env["_fm_merge_k"] = torch.arange(32)
        actual_scale = eval(call["merge_scale_offset"], {}, env)
        torch.testing.assert_close(
            actual_scale, expected_n[None, :] // 128 * 32 + torch.arange(32)[:, None], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "group,formal,changes",
    [
        ("inputs", "value", {"memory_sharing_scope": "block"}),
        ("inputs", "weight", {"memory_sharing_scope": "block"}),
        ("inputs", "weight", {"distributed_backing_type": {"partial": "sum"}}),
        ("inputs", "weight", {"view_byte_offset": 0, "view_byte_offset_expression": "0"}),
        ("workspaces", "projection_partials", {"scalar_dtype": "bfloat16"}),
        ("workspaces", "activation_scales", {"scalar_storage_strides": [2]}),
    ],
)
def test_projection_rejects_unproved_physical_access(projection_raw, group, formal, changes):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _block_fp8_call
    from triton.flagmega.errors import CodegenError

    raw = deepcopy(projection_raw[0])
    argument = next(value for value in raw[group] if value["formal"] == formal)
    argument["buffers"][0]["abi"].update(changes)
    with pytest.raises(CodegenError, match="Cooperative FP8 projections"):
        _block_fp8_call(raw)


def test_projection_keeps_combined_scale_fma_cancellation(tmp_path):
    import numpy as np

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from python.test.flagmega.codegen.triton.kernels.block_fp8.test_dequantized_runtime import _run

    class Cancellation(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 256)), id="x")
            w = self.input("w", fm.tensor_type("float8_e4m3fn", (16, 256)), id="w")
            s = self.input("s", fm.tensor_type("float32", (1, 2)), id="s")
            output = fm.F.math.block_scaled_matmul(
                x, w, s, weight_block_n=16, weight_block_k=128, activation_quantization="dynamic_fp8_sm90"
            )
            self.function("main", (x, w, s), (output,))

    module = compile_projection(Cancellation(dialect="nn", stage="frozen_constants", entry="main").build())
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    weights = torch.ones((16, 256))
    weights[:, 128:] = -1
    scale = torch.tensor([[1.0, np.nextafter(np.float32(1.0), np.float32(2.0))]])
    values = {"x": torch.ones((1, 256), dtype=torch.bfloat16), "w": weights.to(torch.float8_e4m3fn), "s": scale}
    raw = np.float32(128 * 448)
    a_scale = np.float32(1.0 / 448.0)
    first = np.float32(raw * a_scale)
    factor = np.float32(a_scale * scale[0, 1].item())
    # These FP32 operands' exact product/cancellation fit in FP64. Only the
    # final cast rounds, independently witnessing the original ordered FMA.
    expected = np.float32(float(first) - float(raw) * float(factor))
    actual = _run(path, values, (1, 16), "bfloat16")
    torch.testing.assert_close(actual, torch.full((1, 16), float(expected), dtype=torch.bfloat16), rtol=0, atol=0)
