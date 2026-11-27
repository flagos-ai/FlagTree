# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact QKV basis vectors test packing, MMA coordinates and projection slices.

Small random values plus an absolute tolerance can conceal permutations or
wrong slice boundaries. An integer-valued RHS and one-hot LHS require exact
answers even with BF16 split-K partials.
"""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


@pytest.mark.parametrize("module_fixture", (
    "packed_qkv_mma_pipeline_module",
    "packed_qkv_mma_descriptor_table_pipeline_module",
    "packed_qkv_mma_aligned_pipeline_module",
))
def test_packed_qkv_exact_basis_coordinates(tmp_path, request, module_fixture):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module = request.getfixturevalue(module_fixture)
    names = ("q_weight", "k_weight", "v_weight")
    weights = {
        name: ((torch.arange(2048)[:, None] // 16 * 3
                + torch.arange(2048)[:, None] % 16
                + torch.arange(n)[None, :] * 5 + index * 7) % 61 - 30).to(torch.bfloat16)
        for index, (name, n) in enumerate(zip(names, (2048, 1024, 1024)))
    }
    checkpoint = MemoryCheckpoint({}, {
        name: TensorInfo(name, DType.BFLOAT16, tuple(value.shape), "memory")
        for name, value in weights.items()
    }, weights)
    artifact = write_artifact(module, tmp_path / "qkv", target="nvidia-sm90",
                              checkpoint=checkpoint, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
    outputs = tuple(torch.empty((1, weights[name].shape[1]), device="cuda", dtype=torch.bfloat16)
                    for name in names)
    runtime.prepare(value, *outputs)
    # Packed atom, MMA K tile, split-K owner and final logical reduction edges.
    for k in (0, 15, 16, 63, 64, 255, 256, 2047):
        value.zero_()
        value[0, k] = 1
        for output in outputs:
            output.fill_(float("nan"))
        runtime.run_into(value, *outputs)
        torch.cuda.synchronize()
        for name, output in zip(names, outputs):
            torch.testing.assert_close(output.cpu(), weights[name][k:k + 1], rtol=0, atol=0,
                                       msg=lambda message: f"{name}, K={k}: {message}")

    # This fixture selects K64 block-cyclic partitioning on eight K owners.
    # Dyadic operands make all FP32 dot/add operations exact. Only the typed
    # BF16 partial write and the final BF16 boxing write may round; a full-K
    # torch.matmul oracle would incorrectly erase the first boundary.
    k = torch.arange(2048)
    dyadic = (((k * 7) % 17 - 8).double() / 128).reshape(1, -1)
    value.copy_(dyadic.to(torch.bfloat16))
    runtime.run_into(value, *outputs)
    torch.cuda.synchronize()
    distinguishes_full_k = False
    for name, output in zip(names, outputs):
        partials = []
        for owner in range(8):
            indices = (k // 64) % 8 == owner
            partial = dyadic[:, indices] @ weights[name][indices].double()
            partials.append(partial.to(torch.bfloat16).double())
        expected = torch.stack(partials).sum(0).to(torch.bfloat16)
        full_k = (dyadic @ weights[name].double()).to(torch.bfloat16)
        distinguishes_full_k |= not torch.equal(expected, full_k)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert distinguishes_full_k, "The oracle must exercise BF16 partial rounding"
    assert runtime.resource_report["spill_bytes"] == 0
