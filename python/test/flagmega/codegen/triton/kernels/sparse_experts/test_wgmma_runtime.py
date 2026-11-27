# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected expert WGMMA agrees with an independent CUDA scaled-GEMM oracle."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.math._ue8m0_quantization import ue8m0_quantize
from triton.flagmega.ir.ops.nn.partitioned_sparse_experts_down import PartitionedSparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target


def expert_module(family, packed=False, *, quantization="dynamic_fp8_ue8m0", block=128,
                  reduction=256, rounded=False):
    definition = SparseExpertsGateUp if family == "sparse_experts_gate_up" else PartitionedSparseExpertsDown
    lanes = (2, 2) if packed else ()
    dtype = fm.vector_type("bfloat16", lanes) if lanes else "bfloat16"
    capacity = reduction // 4 if packed else reduction
    common = {"router_expert_ids": fm.tensor_type("int32", (1, 2))}
    attrs = {"activation_quantization": quantization, "weight_block_n": 128, "weight_block_k": block}
    if family == "sparse_experts_gate_up":
        types = {**common, "dispatched": fm.tensor_type(dtype, (1, 2, capacity))}
        for field in ("gate", "up"):
            types[field + "_weight"] = fm.tensor_type("float8_e4m3fn", (3, 72, reduction))
            types[field + "_input_scale"] = fm.tensor_type("float32", (3, 1))
            types[field + "_proj_scale"] = fm.tensor_type("float32", (3, 1, -(-reduction // block)))
        attrs.update(output_dtype=dtype, round_projections=True, round_activation=True,
                     gate_clamp_max=10., up_clamp=10.)
    else:
        types = {**common, "activations": fm.tensor_type(dtype, (1, 2, 2, capacity)),
                 "down_weight": fm.tensor_type("float8_e4m3fn", (3, 2, 72, reduction)),
                 "down_input_scale": fm.tensor_type("float32", (3, 1)),
                 "down_proj_scale": fm.tensor_type("float32", (3, 2, 1, -(-reduction // block)))}
        attrs.update(output_lanes=lanes, round_projection=rounded)

    class Graph(fm.Module):
        def forward(self):
            values = tuple(self.input(p.name, types[p.name], id=p.name) for p in definition.input_parameters)
            result = definition.construct(*values, **attrs, name="projection")
            self.function("main", values, (result,))

    return Graph(dialect="nn", stage="frozen_constants", entry="main").build()


@pytest.mark.parametrize("family", ["sparse_experts_gate_up", "partitioned_sparse_experts_down"])
@pytest.mark.parametrize("mode,block,k,supported", [
    ("dynamic_fp8_ue8m0", 128, 256, True), ("dynamic_fp8_ue8m0", 64, 256, False),
    ("dynamic_fp8_ue8m0", 128, 192, False), ("dynamic_fp8", 128, 256, False),
])
def test_wgmma_candidate_requires_its_block_quantization_contract(family, mode, block, k, supported):
    module = expert_module(family, quantization=mode, block=block, reduction=k)
    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(p for p in proposed.selection_points if p.owner == "projection")
    assert (f"tir.{family}.wgmma" in {c.id for c in point.candidates}) is supported


def test_expert_wgmma_requires_hopper_mma_capability():
    from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target
    from triton.flagmega.targets.nvidia_sm120 import NvidiaSm120Target

    for family in ("sparse_experts_gate_up", "partitioned_sparse_experts_down"):
        identity = f"tir.{family}.wgmma"
        implementation = NvidiaSm90Target().triton_implementation_model.implementation(identity)
        assert identity not in {i.id for i in IluvatarBiV150Target().triton_implementation_model.implementations}
        assert NvidiaSm120Target().capability.missing(implementation.requires) == ("mma_v3",)


def cuda_projection(value, weight, scale):
    """cuBLAS FP8 fast accumulation, with explicit block-scale rounding."""
    n, k = weight.shape
    rows = -(-n // 128) * 128
    padded = torch.zeros((rows, k), dtype=weight.dtype, device=weight.device)
    padded[:n].copy_(weight)
    q, a_scale = ue8m0_quantize(value.reshape(-1, 128))
    q = q.reshape(-1)
    one = torch.ones(1, device=value.device)
    acc = torch.zeros((rows, 16), device=value.device)
    for block in range(k // 128):
        rhs = q[block * 128:(block + 1) * 128].expand(16, 128).contiguous().T
        part = torch._scaled_mm(padded[:, block * 128:(block + 1) * 128], rhs,
                                scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=True)
        b_scale = scale[:, block].repeat_interleave(128)[:, None]
        acc = torch.addcmul(acc, part * b_scale, a_scale[block, 0])
    return acc[:n, 0]


@pytest.mark.parametrize("family,rounded", [("sparse_experts_gate_up", False),
                                         ("partitioned_sparse_experts_down", False),
                                         ("partitioned_sparse_experts_down", True)])
@pytest.mark.parametrize("packed", [False, True])
def test_wgmma_expert_projection_matches_cuda(tmp_path, family, rounded, packed):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = expert_module(family, packed, rounded=rounded)
    base = NvidiaSm90Target().triton_implementation_model
    identity = f"tir.{family}.wgmma"
    target = NvidiaSm90Target(triton_implementation_model=replace(base, preferences={
        **base.preferences, family: (identity, *base.preferences.get(family, ())),
    }))
    compiler = Compiler()
    compiler.target = target
    compiled = compiler.compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert f"# flagmega-kernel: {family}/wgmma" in source
    assert "# flagmega-kernel: pack/" not in source
    generator = torch.Generator().manual_seed(1729)
    values = {}
    for name in module.function_map[module.entry].parameters:
        tensor = module.node_map[name].type
        shape = tuple(d.fixed_value for d in tensor.shape)
        if name == "router_expert_ids":
            value = torch.tensor([[2, 0]], dtype=torch.int32)
        elif name.endswith("_scale"):
            value = torch.linspace(.75, 1.25, steps=tensor.shape[0].fixed_value).reshape(3, *([1] * (tensor.rank - 1)))
            value = value.expand(shape).contiguous()
        else:
            lanes = tensor.dtype.lanes if isinstance(tensor.dtype, fm.VectorType) else ()
            dtype = tensor.dtype.elem_type if lanes else tensor.dtype
            value = torch.randn((*shape, *lanes), generator=generator).to(getattr(torch, dtype.value))
        values[name] = value.cuda()
    if family == "sparse_experts_gate_up":
        expected = []
        for route, expert in enumerate((2, 0)):
            x = values["dispatched"].reshape(1, 2, 256)[0, route]
            gate, up = (cuda_projection(x, values[field + "_weight"][expert],
                                       values[field + "_proj_scale"][expert])
                        * values[field + "_input_scale"][expert] for field in ("gate", "up"))
            gate, up = gate.bfloat16().clamp(max=10), up.bfloat16().clamp(-10, 10)
            expected.append(torch.nn.functional.silu(gate) * up)
        expected = torch.stack(expected).reshape(1, 2, 72)
    else:
        expected = torch.empty((2, 1, 2, 72), device="cuda")
        activation = values["activations"].reshape(1, 2, 2, 256)
        for partition in range(2):
            for route, expert in enumerate((2, 0)):
                projected = cuda_projection(activation[0, route, partition],
                    values["down_weight"][expert, partition], values["down_proj_scale"][expert, partition])
                projected = projected * values["down_input_scale"][expert]
                expected[partition, 0, route] = projected.bfloat16().float() if rounded else projected
    if packed:
        expected = expected.reshape(*expected.shape[:-1], 18, 2, 2)
    runtime = load(artifact, device="cuda:0")
    try:
        arguments = tuple(values[n] for n in module.function_map[module.entry].parameters)
        runtime.prepare(*arguments)
        actual = runtime.run(*arguments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
