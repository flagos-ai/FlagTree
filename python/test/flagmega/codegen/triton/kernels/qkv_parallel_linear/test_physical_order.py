# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed QKV physical-order traversal, including both active-axis tails."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


_GENERIC = KernelTemplateSpec("qkv_parallel_linear", "packed_fused_gemv")
_AMD = KernelTemplateSpec("qkv_parallel_linear", "packed_fused_gemv", "amd", "gfx1100")
_GFX1201 = KernelTemplateSpec("qkv_parallel_linear", "packed_fused_gemv", "amd", "gfx1201")


@pytest.mark.parametrize("shard_index", [0, 3])
@pytest.mark.parametrize("template_spec", [_GENERIC, _AMD, _GFX1201], ids=["generic", "amd", "gfx1201"])
@pytest.mark.parametrize("k,active_k,capacities,tile_n,block_k", [
    (32, 32, (16, 8, 8), 16, 32),
    (48, 43, (24, 8, 16), 16, 32),
    (16, 13, (8, 8, 8), 32, 64),
    # Qwen3-1.7B decode, one of 128 owners: the AMD form caps tile_n at 64.
    (256, 256, (128, 64, 64), 128, 64),
])
def test_packed_qkv_physical_order_and_active_tails(tmp_path, template_spec, k, active_k, capacities, tile_n, block_k,
                                                   shard_index):
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
            "source": "X", "weight": "W", "local_m": 1, "source_offset": "qkv_local_k_offsets",
            "source_active": f"qkv_local_k_offsets < {active_k}", "local_k_capacity": k,
            "tile_n": tile_n, "block_k": block_k, "k_lane": 16, "n_lane": 8,
            "weight_owner_stride": k * total_n, "weight_k_group_stride": total_n * 16,
            "weight_n_group_stride": 128, "outputs": outputs, "output_type": "tl.bfloat16"}
    body = TritonTemplateRegistry().render_kernel(
        template_spec,
        {"render_calls": [call], "distributed_entry": False, "mesh_hierarchy": (1, 1)},
    ).source
    assert f"platform={template_spec.platform or 'generic'}" in body.splitlines()[0]
    # gfx1201 starts each owner on a different N tile; run a non-zero owner too.
    body = body.replace("    shard_index = 0\n", f"    shard_index = tl.full((), {shard_index}, tl.int64)\n")
    path = tmp_path / "packed_qkv_physical.py"
    path.write_text("import triton\nimport triton.language as tl\n" + body)
    spec = importlib.util.spec_from_file_location("packed_qkv_physical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.randint(-3, 4, (k,), device="cuda").bfloat16()
    dense = torch.randint(-3, 4, (total_n, k), device="cuda").bfloat16()
    packed = dense.reshape(total_n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous()
    # The owner reads its own slab; earlier slabs hold NaN so a wrong base shows.
    slabs = torch.full(((shard_index + 1) * packed.numel(),), float("nan"), device="cuda", dtype=torch.bfloat16)
    slabs[shard_index * packed.numel():] = packed.reshape(-1)
    results = [torch.full((capacity,), 777., device="cuda", dtype=torch.bfloat16) for capacity in capacities]
    module.run[(1,)](x, slabs, *results)
    expected = (dense[:, :active_k].float() * x[:active_k].float()).sum(-1).bfloat16()
    start = 0
    for i, (result, capacity) in enumerate(zip(results, capacities)):
        active = capacity - i
        torch.testing.assert_close(result[:active], expected[start:start + active], rtol=0, atol=0)
        torch.testing.assert_close(result[active:], torch.full_like(result[active:], 777.), rtol=0, atol=0)
        start += capacity


@pytest.mark.parametrize("platform,architecture,expected", [
    ("amd", "gfx1100", "kernels/qkv_parallel_linear/platforms/amd/packed_fused_gemv.py.jinja"),
    ("amd", "gfx1201", "kernels/qkv_parallel_linear/platforms/amd/gfx1201/packed_fused_gemv.py.jinja"),
    ("nvidia", "sm90", "kernels/qkv_parallel_linear/packed_fused_gemv.py.jinja"),
    ("corex", "bi_v150", "kernels/qkv_parallel_linear/packed_fused_gemv.py.jinja"),
    (None, None, "kernels/qkv_parallel_linear/packed_fused_gemv.py.jinja"),
])
def test_packed_qkv_rdna_template_is_amd_only(platform, architecture, expected):
    spec = KernelTemplateSpec("qkv_parallel_linear", "packed_fused_gemv", platform, architecture)
    assert TritonTemplateRegistry().resolve(spec) == expected


def test_gfx1201_qkv_rotates_only_multi_tile_outputs():
    outputs = [{"result": f"Y{i}", "local_n_capacity": capacity, "active": "True",
                "result_offset": "qkv_local_n_offsets", "weight_group_start": 0}
               for i, capacity in enumerate((128, 64, 64))]
    call = {"family": "qkv_parallel_linear", "variant": "packed_fused_gemv",
            "symbol": "run", "signature": "X, W, Y0, Y1, Y2",
            "source": "X", "weight": "W", "local_m": 1, "source_offset": "qkv_local_k_offsets",
            "source_active": "True", "local_k_capacity": 256,
            "tile_n": 128, "block_k": 64, "k_lane": 16, "n_lane": 8,
            "weight_owner_stride": 65536, "weight_k_group_stride": 4096,
            "weight_n_group_stride": 128, "outputs": outputs, "output_type": "tl.bfloat16"}
    context = {"render_calls": [call], "distributed_entry": False, "mesh_hierarchy": (1, 1)}
    rotated = TritonTemplateRegistry().render_kernel(_GFX1201, context).source
    assert rotated.count("tl.cast(shard_index % 2, tl.int32) * 64") == 1
    assert rotated.count("qkv_local_n_offsets = qkv_local_n_start + tl.arange(0, 64)") == 2
    for spec in (_AMD, _GENERIC):
        assert "tl.cast(shard_index" not in TritonTemplateRegistry().render_kernel(spec, context).source
    # gfx1201 differs from the gfx1100 form only by the rotation.
    plain = TritonTemplateRegistry().render_kernel(_AMD, context).source
    assert rotated.count("\n") == plain.count("\n") + 3
