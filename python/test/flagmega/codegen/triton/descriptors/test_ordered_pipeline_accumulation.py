# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TMA slot turnover preserves a complete ordered MMA accumulator."""

from dataclasses import replace

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega import ir as fm
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from .conftest import _packed_descriptor_pipeline_module


@pytest.mark.parametrize("identity,block_k", [
    ("tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv", 1024),
    ("tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv", 512),
    ("tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512", 256),
])
def test_ordered_pipeline_preserves_mma_bits_across_slots(tmp_path, identity, block_k):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    generator = torch.Generator().manual_seed(467)
    k = n = 2048
    weight = (torch.randn((n, k), generator=generator) * .1).bfloat16()
    checkpoint = MemoryCheckpoint({}, {"weight": TensorInfo("weight", DType.BFLOAT16, (n, k), "memory")},
                                  {"weight": weight})
    base = NvidiaSm90Target().triton_implementation_model
    entries = []
    for implementation in base.implementations:
        if implementation.id == identity:
            indices = {i for c in implementation.transfer_pipeline.channels for i in c.shared_workspace_indices}
            workspaces = tuple(replace(w, type=fm.tensor_type(
                w.type.dtype, (w.type.shape[0], block_k // 16, *w.type.shape[2:])))
                if i in indices else w for i, w in enumerate(implementation.shared_workspaces))
            parameters = {**implementation.parameters, "block_k": block_k,
                          "accumulation_mode": "ordered_mma", "mma_block_k": 128}
            if "reduction_unroll" in parameters:
                parameters["reduction_unroll"] = min(parameters["reduction_unroll"], block_k // parameters["reduction_group"])
            implementation = replace(implementation, parameters=parameters, shared_workspaces=workspaces,
                                     facts={**implementation.facts, "ordered_mma_accumulation": True})
        entries.append(implementation)
    model = replace(base, implementations=tuple(entries))
    runtimes = []
    try:
        for index, implementation in enumerate(("tir.dense_matmul.packed_k_major_mma", identity)):
            compiler = Compiler()
            compiler.target = NvidiaSm90Target(triton_implementation_model=model)
            module = _packed_descriptor_pipeline_module(implementation, compiler=compiler, k=k, n=n,
                                                       reduction_order="mma_k16",
                                                       output_policy=fm.SBP.split_contiguous((1,), 16)
                                                       if "table" in identity else None)
            path = write_artifact(module, tmp_path / str(index), target="nvidia-sm90", checkpoint=checkpoint,
                                  emit_executable=True)
            runtimes.append(load(path, device="cuda:0"))
        value = torch.empty((1, k), device="cuda", dtype=torch.bfloat16)
        outputs = [r.create_outputs() for r in runtimes]
        for r, output in zip(runtimes, outputs, strict=True):
            r.prepare(value, output=output)
        for _ in range(3):
            value.copy_(torch.randn((1, k), generator=generator))
            for r, output in zip(runtimes, outputs, strict=True):
                output.fill_(float("nan"))
                r.run_into(output, value)
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        assert runtimes[1].resource_report["spill_bytes"] == 0
    finally:
        for runtime in runtimes:
            runtime.close()
