# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete ordered GLU stages share raw work while retaining BF16 boundaries."""

from copy import deepcopy
from dataclasses import replace
from math import ceil, prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.matmul_glu import matmul_glu_cooperative_implementations

IDENTITY = "tir.matmul_glu.cooperative_wgmma"


def glu_graph(
    *, n=72, k=256, placement=None, readonly=False, limits=(10.0, 10.0), block_k=128,
    quantization="dynamic_fp8_sm90", private_result=False, private_input=False,
):
    types = {
        "value": fm.tensor_type("float32" if private_input else "bfloat16", (1, k)),
        "gate_weight": fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (n, k // 32)),
        "up_weight": fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (n, k // 32)),
        "gate_scale": fm.tensor_type("float32", (ceil(n / 128), ceil(k / block_k))),
        "up_scale": fm.tensor_type("float32", (ceil(n / 128), ceil(k / block_k))),
    }
    axes = tuple(i for i, level in enumerate(placement.hierarchy_levels) if level == "b") if placement else ()
    if placement:
        row = fm.SBP.split_contiguous(tuple(range(placement.rank)), n // prod(placement.hierarchy))
        broad = fm.SBP.broadcast()
        types = {
            name: fm.DistributedType(t, (row, broad) if name.endswith("_weight") else (broad, broad), placement)
            for name, t in types.items()
        }

    class Graph(fm.Module):
        def forward(self):
            values, parameters = {}, []
            for name, kind in types.items():
                if readonly and name.endswith("_weight"):
                    values[name] = self.weight(name, kind, source="memory", key=name, id=name)
                else:
                    values[name] = self.input(name, kind, id=name)
                    parameters.append(values[name])
            if private_input:
                values["value"] = fm.F.tensors.cast(values["value"], "bfloat16", name="private_value")
            output = fm.F.nn.packed_matmul_glu(
                *(values[name] for name in types),
                activation="silu",
                weight_block_n=128,
                weight_block_k=block_k,
                k_pack=2,
                k_vector=16,
                activation_quantization=quantization,
                gate_clamp_max=limits[0],
                up_clamp=limits[1],
                name="glu",
            )
            if private_result:
                output = fm.F.tensors.cast(output, "float32", name="widened")
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


def glu_target(cooperative=True, warps=8):
    target = NvidiaSm90Target()
    base = target.triton_implementation_model
    implementations = tuple(
        replace(v, parameters={**v.parameters, "compute_num_warps": warps, "scaled_accumulation": "fma"})
        if v.id == "tir.matmul_glu.mma"
        else v
        for v in base.implementations
    )
    if cooperative:
        implementations += tuple(
            replace(v, parameters={**v.parameters, "compute_num_warps": warps})
            for v in matmul_glu_cooperative_implementations()
        )
    target.triton_implementation_model = replace(
        base,
        implementations=implementations,
        preferences={
            **base.preferences,
            "matmul_glu": ((IDENTITY,) if cooperative else ()) + ("tir.matmul_glu.mma", "tir.matmul_glu.simt"),
        },
    )
    return target


def compile_glu(graph, placement=None, cooperative=True, warps=8):
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = glu_target(cooperative, warps)
    module = compiler.compile(graph).module
    assert any(
        d.dispatch.microkernel.implementation == (IDENTITY if cooperative else "tir.matmul_glu.mma")
        for d in module.kernel_definitions
    )
    return module


class GluCheckpoint:
    def __init__(self, values):
        self.values = values

    def load_tensor(self, key, *, device="cpu"):
        return self.values[key].to(device)


def glu_values(torch, graph):
    rng = torch.Generator().manual_seed(1031)
    values = {}
    for node in graph.nodes:
        if node.op not in {"builtin.var", "builtin.weight"}:
            continue
        kind = fm.logical_type(node.type)
        shape = tuple(d.fixed_value for d in kind.shape)
        lanes = kind.dtype.lanes if isinstance(kind.dtype, fm.VectorType) else ()
        dtype = kind.dtype.elem_type if lanes else kind.dtype
        if node.id.endswith("_scale"):
            values[node.id] = (torch.rand(shape, generator=rng) + 0.125) / 16
        else:
            values[node.id] = (torch.randint(-16, 17, (*shape, *lanes), generator=rng).float() / 4).to(
                getattr(torch, dtype.value)
            )
    return values


def test_glu_workspace_effects_and_optional_factory(tmp_path):
    module = compile_glu(glu_graph())
    dispatch = next(d.dispatch for d in module.kernel_definitions if d.dispatch.microkernel.family == "matmul_glu")
    assert dispatch.microkernel.facts["source_memory_scope"] == "chip"
    assert dispatch.microkernel.facts["internal_grid_barriers"] == 1
    assert [w.type for w in dispatch.workspaces] == [fm.tensor_type("float32", (2, 72))] * 2 + [
        fm.tensor_type("float32", (2,))
    ]
    assert all(w.lifetime.value == "invocation" and w.memory_space == "workspace" for w in dispatch.workspaces)
    assert all(effect.scope.value == "chip" for name, effect in dispatch.memory_effects if name in dispatch.arguments)
    assert fm.load_module(fm.emit_module(module, tmp_path / "selected.py")).semantic_hash == module.semantic_hash
    assert NvidiaSm90Target().triton_implementation_model.implementation(IDENTITY) is None


def test_glu_ordinary_codegen(tmp_path):
    module = compile_glu(glu_graph())
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (path / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: matmul_glu/cooperative_wgmma" in source
    assert "__partials(" in source and "__merge(" in source


@pytest.mark.parametrize(
    "k,block_k,quantization,supported",
    [
        (256, 128, "dynamic_fp8_sm90", True),
        (4096, 128, "dynamic_fp8_sm90", True),
        (160, 128, "dynamic_fp8_sm90", False),
        (256, 64, "dynamic_fp8_sm90", False),
        (256, 128, "dynamic_fp8", False),
    ],
)
def test_glu_candidate_preserves_quantization_domain(k, block_k, quantization, supported):
    proposed = glu_target().propose_tir(glu_graph(k=k, block_k=block_k, quantization=quantization))
    point = next(p for p in proposed.selection_points if p.owner == "glu")
    assert (IDENTITY in {c.id for c in point.candidates}) is supported


@pytest.mark.parametrize(
    "n,k,warps,distributed,limits",
    [(72, 256, 8, False, (None, None)), (512, 4096, 8, True, (10.0, 10.0)),
        (3072, 4096, 8, True, (10.0, 10.0)), (768, 1536, 16, True, (0.75, 0.5))],
)
def test_glu_complete_stage_bitwise(tmp_path, n, k, warps, distributed, limits):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    placement = fm.Placement((8, 16), "yx", "bb") if distributed else None
    graph = glu_graph(n=n, k=k, placement=placement, readonly=True, limits=limits)
    values = glu_values(torch, graph)
    runtimes = []
    results = []
    guards = []
    for cooperative in (False, True):
        module = compile_glu(graph, placement, cooperative, warps)
        path = write_artifact(
            module,
            tmp_path / str(cooperative),
            target="nvidia-sm90",
            emit_executable=True,
            checkpoint=GluCheckpoint(values),
        )
        runtime = load(path, device="cuda:0")
        runtimes.append(runtime)
        current = {}
        for spec in runtime.external_arguments:
            if spec["role"] == "result":
                is_glu = spec["value"] == module.function_map[module.entry].outputs[0]
                shape, dtype = ((1, n), torch.bfloat16) if is_glu else ((128,), torch.float32)
                storage = torch.full((prod(shape) + 64,), 123.0, dtype=dtype, device="cuda")
                current[spec["value"]] = storage[: prod(shape)].reshape(shape)
                guards.append(storage[prod(shape) :])
        results.append(current)
    device_values = {name: value.cuda() for name, value in values.items() if not name.endswith("_weight")}
    arguments = [
        tuple(({**device_values, **current})[spec["value"]] for spec in runtime.external_arguments)
        for runtime, current in zip(runtimes, results, strict=True)
    ]
    for mutation in range(4):
        if mutation == 1:
            device_values["value"].zero_()
        elif mutation:
            device_values["value"].fill_(0.125 * mutation)
            device_values["gate_scale"].mul_(1.03125)
            device_values["up_scale"].mul_(0.96875)
        for runtime, args in zip(runtimes, arguments, strict=True):
            GeneratedTirCallGraphModule.prepare(runtime, *args)
            GeneratedTirCallGraphModule.run_into(runtime, *args)
            torch.cuda.synchronize()
            assert runtime.resource_report["spill_store_bytes"] == runtime.resource_report["spill_load_bytes"] == 0
        for before, after in zip(results[0].values(), results[1].values(), strict=True):
            torch.testing.assert_close(after.view(torch.uint8), before.view(torch.uint8), rtol=0, atol=0)
        assert all(torch.all(guard == 123.0) for guard in guards)
        torch.testing.assert_close(tuple(results[1].values())[1], device_values["sentinel"] * 2, rtol=0, atol=0)
    for runtime in runtimes:
        runtime.close()


@pytest.fixture(scope="module", params=(("cbb", True), ("bcb", True), ("cbb", False)))
def glu_raw(request):
    from triton.flagmega.codegen.triton.call_abi import describe_kernel_call_abis

    levels, readonly = request.param
    placement = fm.Placement((4, 8, 16), "cyx", "cbb") if levels == "cbb" else fm.Placement((8, 4, 16), "ycx", "bcb")
    module = compile_glu(glu_graph(n=2048, k=4096, placement=placement, readonly=readonly), placement)
    raw = next(c for c in describe_kernel_call_abis(module) if c["family"] == "matmul_glu")
    for group in ("inputs", "outputs", "workspaces"):
        for argument in raw[group]:
            for binding in argument["buffers"]:
                binding.setdefault("runtime_argument", argument["formal"])
    return raw, levels


@pytest.mark.parametrize("chip", [0, 3])
def test_glu_actual_gate_up_peer_addresses_and_scale_rows(glu_raw, chip):
    from types import SimpleNamespace

    import torch
    from triton.flagmega.codegen.triton.kernel_call_renderers import _matmul_glu_call

    raw, levels = glu_raw
    call = _matmul_glu_call(deepcopy(raw))

    class Pointer(int):
        def to(self, dtype):
            return self

    tl = SimpleNamespace(
        pointer_type=lambda dtype: None, float8e4nv=None, bfloat16=None, float32=None, int64=None, uint8=None
    )
    for tile in (0, 7):
        rows = tile * 64 + torch.arange(64)
        ks = 31 * 128 + torch.arange(128)
        env = {
            "tl": tl,
            "shard_coord0": chip if levels == "cbb" else 7,
            "shard_coord1": 7 if levels == "cbb" else chip,
            "shard_coord2": 9,
            "_fm_owner": rows // 4,
            "_fm_local_n": rows % 4,
            "_fm_k": ks,
        }
        for group in ("inputs", "outputs", "workspaces"):
            for argument in raw[group]:
                for binding in argument["buffers"]:
                    env[binding["runtime_argument"]] = Pointer(0)
        for name, pointer, offset in (
            ("gate_weight", "weight_peer_pointer", "weight_peer_offset"),
            ("up_weight", "up_weight_peer_pointer", "up_weight_peer_offset"),
        ):
            abi = next(a["buffers"][0]["abi"] for a in raw["inputs"] if a["formal"] == name)
            actual = eval(call[pointer], {}, env) + eval(call[offset], {}, env)
            if abi["storage_kind"] in {"compact_per_chip", "compact_per_owner"}:
                expected = (
                    abi["pool_byte_offset"]
                    + (rows[:, None] // 4) * abi["component_stride_scalar_elements"]
                    + (rows[:, None] % 4) * 4096
                    + ks[None, :]
                )
            else:
                owner = rows // 4
                global_row = (
                    chip * 512 + rows
                    if levels == "cbb"
                    else ((owner // 16 * 4 + chip) * 16 + owner % 16) * 4 + rows % 4
                )
                expected = abi["pool_byte_offset"] + global_row[:, None] * 4096 + ks[None, :]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        env["local_n_offsets"] = torch.arange(16)
        env["_fm_global_n"] = eval(call["global_n"], {}, env)
        expected_n = (
            (chip * 128 + 7 * 16 + 9) * 4 + torch.arange(16)
            if levels == "cbb"
            else ((7 * 4 + chip) * 16 + 9) * 4 + torch.arange(16)
        )
        torch.testing.assert_close(env["_fm_global_n"], expected_n, rtol=0, atol=0)
        env["_fm_merge_k"] = torch.arange(32)
        for offset in ("merge_scale_offset", "up_merge_scale_offset"):
            torch.testing.assert_close(
                eval(call[offset], {}, env), expected_n[None, :] // 128 * 32 + torch.arange(32)[:, None], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "group,formal,changes",
    [
        ("inputs", "value", {"memory_sharing_scope": "block"}),
        ("inputs", "gate_weight", {"memory_sharing_scope": "block"}),
        ("inputs", "up_weight", {"memory_sharing_scope": "block"}),
        ("inputs", "up_weight", {"distributed_backing_type": {"partial": "sum"}}),
        ("inputs", "up_scale", {"view_byte_offset": 0, "view_byte_offset_expression": "0"}),
        ("outputs", "result", {"memory_sharing_scope": "block"}),
        ("workspaces", "gate_partials", {"scalar_dtype": "bfloat16"}),
        ("workspaces", "up_partials", {"scalar_storage_strides": [0, 1]}),
        ("workspaces", "activation_scales", {"scalar_storage_strides": [2]}),
    ],
)
def test_glu_rejects_unproved_physical_access(glu_raw, group, formal, changes):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _matmul_glu_call
    from triton.flagmega.errors import CodegenError

    raw = deepcopy(glu_raw[0])
    argument = next(a for a in raw[group] if a["formal"] == formal)
    argument["buffers"][0]["abi"].update(changes)
    with pytest.raises(CodegenError, match="Cooperative FP8 GLU"):
        _matmul_glu_call(raw)


def test_glu_combined_scale_cancellation_and_bf16_postops(tmp_path):
    import numpy as np

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from python.test.flagmega.codegen.triton.kernels.block_fp8.test_dequantized_runtime import _run

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 256)), id="x")
            gate = self.input("gate", fm.tensor_type("float8_e4m3fn", (16, 256)), id="gate")
            up = self.input("up", gate.type, id="up")
            gs = self.input("gs", fm.tensor_type("float32", (1, 2)), id="gs")
            us = self.input("us", gs.type, id="us")
            result = fm.F.nn.matmul_glu(
                x,
                gate,
                up,
                gs,
                us,
                activation="silu",
                weight_block_n=128,
                weight_block_k=128,
                activation_quantization="dynamic_fp8_sm90",
                name="glu",
            )
            self.function("main", (x, gate, up, gs, us), (result,))

    module = compile_glu(Graph(dialect="nn", stage="frozen_constants", entry="main").build())
    path = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    weights = torch.ones((16, 256))
    weights[:, 128:] = -1
    scales = torch.tensor([[1.0, np.nextafter(np.float32(1.0), np.float32(2.0))]])
    values = {
        "x": torch.ones((1, 256), dtype=torch.bfloat16),
        "gate": weights.to(torch.float8_e4m3fn),
        "up": torch.ones((16, 256), dtype=torch.float8_e4m3fn),
        "gs": scales,
        "us": torch.ones_like(scales),
    }
    raw = np.float32(128 * 448)
    a_scale = np.float32(1 / 448)
    first = np.float32(raw * a_scale)
    factor = np.float32(a_scale * scales[0, 1].item())
    gate = np.float32(float(first) - float(raw) * float(factor))
    assert gate != 0
    gate_bf16 = torch.tensor(float(gate)).bfloat16().float()
    expected = (torch.nn.functional.silu(gate_bf16).bfloat16().float() * 256).bfloat16()
    actual = _run(path, values, (1, 16), "bfloat16")
    torch.testing.assert_close(actual, torch.full((1, 16), expected.item(), dtype=torch.bfloat16), rtol=0, atol=0)
