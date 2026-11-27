# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed QKV SME transports exactly the consumer warp's four fragments."""
from copy import deepcopy
import importlib.util

import pytest

from triton.flagmega.codegen.triton.corex_sme_abi import attach_flat_sme_calls
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _call(k=192, owners=4, n_lane=4, active_k=None):
    strides = (k * 256, 4096, n_lane * 16)
    weight_abi = dict(scalar_dtype="bfloat16", local_capacity_shape=[owners, k // 16, 256 // n_lane],
                      logical_shape=[owners, k // 16, 256 // n_lane],
                      active_shape_expressions=list(map(str, (owners, k // 16, 256 // n_lane))),
                      scalar_lane_shape=[n_lane, 2, 8], scalar_storage_strides=list(strides),
                      distributed_type=None, coordinate_space="canonical_global",
                      storage_kind="compact_local", alignment_bytes=256, pool_byte_offset=0)
    source_abi = dict(scalar_dtype="bfloat16", scalar_lane_count=1, local_capacity_shape=[1, k],
                      scalar_storage_strides=[k, 1])
    outputs = [dict(result=f"Y{i}", local_n_capacity=n, active=f"qkv_local_n_offsets < {n - i}",
                    result_offset=f"shard_index*{n}+qkv_local_n_offsets", weight_group_start=start // n_lane)
               for i, (n, start) in enumerate(((128, 0), (64, 128), (64, 192)))]
    return dict(family="qkv_parallel_linear", variant="packed_fused_gemv", symbol="run",
                signature="X, W, Y0, Y1, Y2", source="X", weight="W",
                source_offset=f"shard_index*{k}+qkv_local_k_offsets",
                source_active=f"qkv_local_k_offsets < {k if active_k is None else active_k}",
                local_k_capacity=k, tile_n=128, block_k=64, n_lane=n_lane, k_lane=16,
                weight_owner_stride=strides[0], weight_k_group_stride=strides[1],
                weight_n_group_stride=strides[2], outputs=outputs, output_type="tl.float32",
                parameters=dict(distribution=dict(placement=dict(hierarchy=[owners, 1]))),
                inputs=[dict(buffers=[dict(formal="input", runtime_argument="X", abi=source_abi)]),
                        dict(buffers=[dict(formal="q_weight_qkv_fused", runtime_argument="W", abi=weight_abi)])])


@pytest.mark.parametrize("n_lane,k", [(4, 128), (8, 192), (4, 2048)])
def test_complete_packed_qkv_selects_owned_sme(n_lane, k):
    call = _call(k=k, n_lane=n_lane)
    attach_flat_sme_calls({"qkv": [call]}, None)
    assert call["corex_flat_sme_qkv"]


@pytest.mark.parametrize("change", ["alignment", "group_stride", "n_stride", "owner_stride",
                                    "k_tail", "active_weight", "view", "owner_count", "dtype"])
def test_qkv_sme_requires_complete_aligned_weight_rows(change):
    call = _call()
    w = call["inputs"][1]["buffers"][0]["abi"]
    if change == "alignment":
        w["alignment_bytes"] = 16
    elif change == "group_stride":
        call["weight_k_group_stride"] = 8192
    elif change == "n_stride":
        call["weight_n_group_stride"] = 128
    elif change == "owner_stride":
        call["weight_owner_stride"] -= 16
    elif change == "k_tail":
        call["local_k_capacity"] -= 16
    elif change == "active_weight":
        w["active_shape_expressions"][1] = "11"
    elif change == "view":
        w["view_byte_offset"] = 0
    elif change == "owner_count":
        call["parameters"]["distribution"]["placement"]["hierarchy"] = [2, 1]
    elif change == "dtype":
        w["scalar_dtype"] = "float32"
    attach_flat_sme_calls({"qkv": [call]}, None)
    assert "corex_flat_sme_qkv" not in call


def test_qkv_formal_uses_closed_call_alignment_proof():
    call = _call()
    call["inputs"][1]["buffers"][0]["abi"]["alignment_bytes"] = 16
    plan = dict(device_functions=[dict(function="qkv", function_pool_abi=dict(bindings=[dict(formal="W", alignment_bytes=256)]))])
    attach_flat_sme_calls({"qkv": [call]}, plan)
    assert call["corex_flat_sme_qkv"]


@pytest.mark.parametrize("k,n_lane,active_k,cancel", [(128, 4, 128, False), (192, 8, 173, False),
                                                      (2048, 4, 2048, False), (192, 4, 177, True), (192, 8, 0, False)])
def test_qkv_sme_matches_original_folds_and_owner_outputs(tmp_path, k, n_lane, active_k, cancel):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    call = _call(k=k, n_lane=n_lane, active_k=active_k)
    attach_flat_sme_calls({"qkv": [call]}, None)
    assert call["corex_flat_sme_qkv"]
    torch.manual_seed(873)
    x = torch.randn((4, k), device="cuda", dtype=torch.bfloat16)
    dense = (torch.randn((4, 256, k), device="cuda") / 32).bfloat16()
    if cancel:
        x.fill_(1)
        dense.zero_()
        dense[:, :, 0:4] = torch.tensor([33554432., 1., -33554432., 1.], device="cuda").bfloat16()
    if active_k == 0:
        dense.fill_(float("inf"))
    weight = dense.reshape(4, 256 // n_lane, n_lane, k // 16, 16).permute(0, 3, 1, 2, 4).contiguous()
    results = []
    for sme in (False, True):
        test_call = deepcopy(call)
        if not sme:
            test_call.pop("corex_flat_sme_qkv")
        rendered = TritonTemplateRegistry().render_kernel(
            KernelTemplateSpec("qkv_parallel_linear", "packed_fused_gemv", "corex", "bi_v150"),
            dict(render_calls=[test_call], num_warps=16, distributed_entry=True, mesh_hierarchy=(4, 1), mesh_axis_names=("block_y", "block_x")),
        ).source
        path = tmp_path / f"qkv_{sme}.py"
        path.write_text("import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n"
                        "FLAGMEGA_GRID_MESH = tl.constexpr(tle.device_mesh({'block':[('block_y',4),('block_x',1)]}))\n" + rendered)
        spec = importlib.util.spec_from_file_location(f"qkv_{sme}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        y = [torch.full((4 * n + 17,), -7., device="cuda", dtype=torch.float32) for n in (128, 64, 64)]
        fn = module.run
        compiled = fn[(4,)](x, weight, *y, num_warps=16, enable_fp_fusion=False)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            fn[(4,)](x, weight, *y, num_warps=16, enable_fp_fusion=False)
        for _ in range(12):
            graph.replay()
        torch.cuda.synchronize()
        results.append(y)
        if sme:
            # Shared loads retain the [4,4] consumer layout, without CTA sync.
            assert "warpsPerCTA = [4, 4]" in compiled.asm["ttgir"]
        for i, (output, n) in enumerate(zip(y, (128, 64, 64))):
            assert torch.equal(output[-17:], torch.full_like(output[-17:], -7.))
            assert torch.equal(output[:4 * n].reshape(4, n)[:, n - i:], torch.full_like(output[:4 * n].reshape(4, n)[:, n - i:], -7.))
    assert all(torch.equal(a, b) for a, b in zip(*results))
