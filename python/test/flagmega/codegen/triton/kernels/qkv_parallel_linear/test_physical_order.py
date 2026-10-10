# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed QKV physical-order traversal, including both active-axis tails."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


@pytest.mark.parametrize("k,active_k,capacities,tile_n,block_k", [
    (32, 32, (16, 8, 8), 16, 32),
    (48, 43, (24, 8, 16), 16, 32),
    (16, 13, (8, 8, 8), 32, 64),
])
def test_packed_qkv_physical_order_and_active_tails(tmp_path, k, active_k, capacities, tile_n, block_k):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    torch.manual_seed(79)
    total_n = sum(capacities)
    outputs = []
    start = 0
    for i, capacity in enumerate(capacities):
        outputs.append({"result": f"Y{i}", "local_n_capacity": capacity,
                        "active": f"qkv_local_n_offsets < {capacity - i}",
                        "result_offset": "qkv_local_n_offsets", "weight_group_start": start})
        start += capacity // 8
    call = {"family": "qkv_parallel_linear", "variant": "packed_fused_gemv",
            "symbol": "run", "signature": "X, W, Y0, Y1, Y2",
            "source": "X", "weight": "W", "source_offset": "qkv_local_k_offsets",
            "source_active": f"qkv_local_k_offsets < {active_k}", "local_k_capacity": k,
            "tile_n": tile_n, "block_k": block_k, "k_lane": 16, "n_lane": 8,
            "weight_owner_stride": k * total_n, "weight_k_group_stride": total_n * 16,
            "weight_n_group_stride": 128, "outputs": outputs, "output_type": "tl.bfloat16"}
    body = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("qkv_parallel_linear", "packed_fused_gemv"),
        {"render_calls": [call], "distributed_entry": False, "mesh_hierarchy": (1, 1)},
    ).source
    path = tmp_path / "packed_qkv_physical.py"
    path.write_text("import triton\nimport triton.language as tl\n" + body)
    spec = importlib.util.spec_from_file_location("packed_qkv_physical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.randint(-3, 4, (k,), device="cuda").bfloat16()
    dense = torch.randint(-3, 4, (total_n, k), device="cuda").bfloat16()
    packed = dense.reshape(total_n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous()
    results = [torch.full((capacity,), 777., device="cuda", dtype=torch.bfloat16) for capacity in capacities]
    module.run[(1,)](x, packed, *results)
    expected = (dense[:, :active_k].float() * x[:active_k].float()).sum(-1).bfloat16()
    start = 0
    for i, (result, capacity) in enumerate(zip(results, capacities)):
        active = capacity - i
        torch.testing.assert_close(result[:active], expected[start:start + active], rtol=0, atol=0)
        torch.testing.assert_close(result[active:], torch.full_like(result[active:], 777.), rtol=0, atol=0)
        start += capacity
