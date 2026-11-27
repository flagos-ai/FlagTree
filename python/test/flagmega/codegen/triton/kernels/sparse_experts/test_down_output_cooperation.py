# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Wide resident expert projections preserve FP8 scales, owners and rounding."""

import ast
from dataclasses import replace
from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target

IDENTITY = "tir.partitioned_sparse_experts_down.wgmma"


def graph(*, n=4096, k=512, parts=1, experts=16, rounded=False, lanes=(2, 4), shard_n=True):
    mesh = fm.Placement((8, 16), "yx", "bb")
    b = fm.SBP.broadcast()
    route = fm.SBP.split_contiguous((0,), 1)
    feature = fm.SBP.split_contiguous((1,)) if shard_n else b
    types = {
        "activations": fm.DistributedType(
            fm.tensor_type(fm.vector_type("bfloat16", (2, 2)), (1, 8, parts, k // 4)), (b, route, b, b), mesh
        ),
        "router_expert_ids": fm.DistributedType(fm.tensor_type("int32", (1, 8)), (b, route), mesh),
        "down_input_scale": fm.DistributedType(fm.tensor_type("float32", (experts, 1)), (b, b), mesh),
        "down_weight": fm.DistributedType(
            fm.tensor_type("float8_e4m3fn", (experts, parts, n, k)), (b, b, feature, b), mesh
        ),
        "down_proj_scale": fm.DistributedType(
            fm.tensor_type("float32", (experts, parts, (n + 127) // 128, k // 128)), (b, b, b, b), mesh
        ),
    }

    class Graph(fm.Module):
        def forward(self):
            inputs = {name: self.input(name, t, id=name) for name, t in types.items()}
            out = fm.F.nn.partitioned_sparse_experts_down(
                **inputs,
                weight_block_n=128,
                weight_block_k=128,
                activation_quantization="dynamic_fp8_ue8m0",
                round_projection=rounded,
                output_lanes=lanes,
                name="projection",
            )
            self.function("main", tuple(inputs.values()), (out,))

    return Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()


def compile_graph(module, tile=None, *, identity=IDENTITY):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            p = super().propose_tir(module)
            return apply_plan(
                p,
                override_plan(
                    p,
                    [("tir.projection", identity)],
                    policy="test.resident_down",
                    rationale="Exercise the real FP8 UE8M0 down stage at eight warps.",
                ),
            )

    mesh = fm.Placement((8, 16), "yx", "bb")
    base = NvidiaSm90Target().triton_implementation_model

    def configure(v):
        if v.id != identity:
            return v
        return replace(
            v,
            parameters={
                **v.parameters,
                "compute_num_warps": 8,
                **({"output_tile_n": tile} if tile is not None else {}),
            },
        )

    target = Target(
        triton_implementation_model=replace(base, implementations=tuple(configure(v) for v in base.implementations))
    )
    c = Compiler(CompileOptions(placement=mesh))
    c.target = target
    result = c.compile(module).module
    assert len(result.kernel_definitions) == 1
    assert result.kernel_definitions[0].dispatch.microkernel.implementation == identity
    return result


@pytest.mark.parametrize("tile", [64, 128, 256])
def test_requested_output_cooperation_reaches_the_generated_kernel(tmp_path, tile):
    m = compile_graph(graph(), tile)
    p = write_artifact(m, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    t = ast.parse((p / "generated_kernels.py").read_text())
    (loop,) = (
        n
        for n in ast.walk(t)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "_fm_n_start"
    )
    assert ast.literal_eval(loop.iter.args[2]) == tile


@pytest.mark.parametrize("tile", [0, -1, 32, 192, True, 1.5])
def test_invalid_output_cooperation_is_rejected_before_device_codegen(tmp_path, tile):
    m = compile_graph(graph(), tile)
    with pytest.raises(CodegenError, match="output_tile_n"):
        write_artifact(m, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_wide_tail_scale_addresses_stay_inside_the_scale_buffer(tmp_path):
    import numpy as np

    n, k, parts, experts = 65, 384, 2, 16
    m = compile_graph(graph(n=n, k=k, parts=parts, lanes=(), shard_n=False), 256)
    p = write_artifact(m, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    tree = ast.parse((p / "generated_kernels.py").read_text())
    (assignment,) = (
        a
        for a in ast.walk(tree)
        if isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_fm_scale" for t in a.targets)
    )
    (call,) = (
        a
        for a in ast.walk(assignment.value)
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "load"
    )
    predicate = next(k.value for k in call.keywords if k.arg == "mask")

    def mask(node):
        if isinstance(node, ast.Name) and node.id == "_fm_n_mask":
            return np.arange(256) < n
        if isinstance(node, ast.Constant):
            return np.full(256, bool(node.value))
        raise AssertionError(f"Unproved scale mask: {ast.dump(node)}")

    # The final expert/partition/K group places an unsafe padded row directly
    # beyond the logical buffer. Evaluate the emitted predicate, not a presumed
    # source-code guard, against independently defined scalar storage bounds.
    addresses = ((experts * parts - 1) * ((n + 127) // 128) + np.arange(256) // 128) * (k // 128) + (k // 128 - 1)
    capacity = experts * parts * ((n + 127) // 128) * (k // 128)
    assert np.all((addresses[mask(predicate)] >= 0) & (addresses[mask(predicate)] < capacity))
    assert np.any(addresses >= capacity), "Fixture must expose unmasked padded-row reads"


@pytest.mark.parametrize(
    "n,k,parts,rounded,lanes,shard_n",
    [
        (4096, 512, 1, False, (2, 4), True),
        (65, 384, 2, True, (), False),
        (384, 256, 1, False, (2, 4), False),
    ],
)
def test_resident_down_cooperation_and_scale_block_tails(tmp_path, n, k, parts, rounded, lanes, shard_n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    config = {"n": n, "k": k, "parts": parts, "rounded": rounded, "lanes": lanes, "shard_n": shard_n}
    source = graph(**config)
    experts = 16
    runtimes = {}
    outputs = {}
    shared = {}
    storages = []
    arguments = {}

    def allocate(kind):
        t = fm.logical_type(kind)
        shape = tuple(d.fixed_value for d in t.shape) + tuple(getattr(t.dtype, "lanes", ()))
        dtype = getattr(torch, getattr(t.dtype, "elem_type", t.dtype).value)
        storage = torch.full((prod(shape) + 32,), 128, dtype=dtype, device="cuda")
        storages.append(storage)
        return storage[16:-16].reshape(shape)

    try:
        for tile in (64, 128, 256):
            m = compile_graph(source, tile)
            r = load(
                write_artifact(m, tmp_path / str(tile), target="nvidia-sm90", emit_executable=True), device="cuda:0"
            )
            runtimes[tile] = r
            assert not r.codegen.get("host_tensor_descriptor_specs")
            values = {}
            for spec in r.external_arguments:
                if spec["role"] == "result":
                    outputs[tile] = allocate(m.node_map[spec["value"]].type)
                    values[spec["value"]] = outputs[tile]
                else:
                    if spec["value"] not in shared:
                        shared[spec["value"]] = allocate(m.node_map[spec["value"]].type)
                    values[spec["value"]] = shared[spec["value"]]
            arguments[tile] = tuple(values[s["value"]] for s in r.external_arguments)
            Run.prepare(r, *arguments[tile])
            assert r.resource_report["spill_bytes"] == 0
        gen = torch.Generator().manual_seed(71993)
        for mutation in range(6):
            x = (torch.randint(-16, 17, (1, 8, parts, k), generator=gen).float() / 16).bfloat16()
            w = (torch.randint(-16, 17, (experts, parts, n, k), generator=gen).float() / 16).to(torch.float8_e4m3fn)
            scales = 2.0 ** torch.randint(-3, 3, (experts, parts, (n + 127) // 128, k // 128), generator=gen).float()
            input_scale = (2.0 ** (torch.arange(experts).remainder(3) - 2).float()).reshape(experts, 1)
            ids = torch.randperm(experts, generator=gen)[:8].reshape(1, 8).int()
            if mutation in (1, 2, 3, 4):
                x = (torch.randn((1, 8, parts, k), generator=gen) * (0.5 + mutation)).bfloat16()
                w = (torch.randn((experts, parts, n, k), generator=gen) * 0.125).to(torch.float8_e4m3fn)
                scales = torch.rand(scales.shape, generator=gen) + 0.125
                input_scale = torch.rand((experts, 1), generator=gen) + 0.125
                if mutation == 4:
                    ids.fill_(7)
            if mutation == 5:
                x.zero_()
                x[..., 0] = 129
                w.zero_()
                w[..., 0] = 1
                scales = torch.rand(scales.shape, generator=gen) + 0.125
            cpu_values = {
                "activations": x.reshape(shared["activations"].shape),
                "router_expert_ids": ids,
                "down_input_scale": input_scale,
                "down_weight": w,
                "down_proj_scale": scales,
            }
            for name, value in cpu_values.items():
                shared[name].copy_(value.to("cuda"))
            for tile, r in runtimes.items():
                outputs[tile].fill_(float("nan"))
                Run.run_into(r, *arguments[tile])
            torch.cuda.synchronize()
            for tile in (128, 256):
                assert torch.equal(outputs[64].view(torch.uint8), outputs[tile].view(torch.uint8)), (mutation, tile)
            if mutation == 0:
                (expected,) = TorchEvaluator(DictWeightResolver({})).run(source, cpu_values)
                torch.testing.assert_close(outputs[256].cpu(), expected, rtol=0, atol=0)
            if mutation == 5:
                expected = torch.stack(
                    [
                        scales[ids[0, r], p, torch.arange(n) // 128, 0] * 128 * input_scale[ids[0, r], 0]
                        for p in range(parts)
                        for r in range(8)
                    ]
                ).reshape(parts, 1, 8, n)
                if rounded:
                    expected = expected.bfloat16().float()
                torch.testing.assert_close(outputs[256].cpu().reshape(parts, 1, 8, n), expected, rtol=0, atol=0)
            assert all(bool((s[:16].float() == 128).all()) and bool((s[-16:].float() == 128).all()) for s in storages)
    finally:
        for r in runtimes.values():
            r.close()


def test_wide_output_tile_rejects_a_non_wgmma_implementation(tmp_path):
    module = compile_graph(graph(), 128, identity="tir.partitioned_sparse_experts_down.simt")
    with pytest.raises(CodegenError, match="output_tile_n requires"):
        write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


def test_default_output_tile_preserves_the_legacy_source(tmp_path):
    sources = []
    for index, tile in enumerate((None, 64)):
        module = compile_graph(graph(), tile)
        path = write_artifact(module, tmp_path / str(index), target="nvidia-sm90", emit_executable=True)
        sources.append((path / "generated_kernels.py").read_bytes())
    assert sources[0] == sources[1]
