# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A 1024-thread launch preserves the previous packed GEMV reduction trees."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _packed_offset(n, k):
    return f"(({k}[None, :] // 16) * 2048 + ({n}[:, None] // 8) * 128 + ({n}[:, None] % 8) * 16 + {k}[None, :] % 16)"


def _call(family, variant):
    if family == "qkv_parallel_linear":
        outputs = []
        start = 0
        for index, capacity in enumerate((128, 64, 64)):
            outputs.append(dict(result=f"Y{index}", local_n_capacity=capacity,
                                active="True", result_offset="qkv_local_n_offsets",
                                weight_group_start=start))
            start += capacity // 8
        return dict(family=family, variant=variant, symbol="run", signature="X, W, Y0, Y1, Y2",
                    source="X", weight="W", source_offset="qkv_local_k_offsets",
                    source_active="True", local_m=1, local_k_capacity=512, tile_n=512, block_k=64,
                    k_lane=16, n_lane=8, weight_owner_stride=512 * 256,
                    weight_k_group_stride=256 * 16, weight_n_group_stride=128,
                    outputs=outputs, output_type="tl.float32")
    call = dict(family=family, variant=variant, symbol="run", local_m=1, local_m_capacity=1,
                local_n_capacity=128, local_k_capacity=512, source="X", weight="W",
                source_active="True", result_active="True", result="Y",
                global_n="dense_local_n_offsets", global_k="dense_local_k_offsets",
                source_offset="dense_local_k_offsets", result_offset="dense_local_n_offsets",
                weight_offset=_packed_offset("dense_global_n", "dense_global_k"),
                output_type="tl.float32", tile_n=128, block_k=64,
                signature="X, W, Y")
    if family == "dense_matmul_glu":
        call.update(signature="X, W, U, Y", block_k=32,
                    global_n="dense_glu_local_n_offsets", global_k="dense_glu_local_k_offsets",
                    source_offset="dense_glu_local_k_offsets", result_offset="dense_glu_local_n_offsets",
                    gate_weight="W", up_weight="U",
                    gate_weight_offset=_packed_offset("dense_glu_global_n", "dense_glu_global_k"),
                    up_weight_offset=_packed_offset("dense_glu_global_n", "dense_glu_global_k"),
                    gate_clamp_max=None, up_clamp=None, output_type="tl.bfloat16")
    elif variant == "packed_k_major_gemv_norm_stats":
        call.update(signature="X, W, R, Y, S", tile_n=64, block_k=128,
                    residual="R", residual_offset="dense_local_n_offsets", addend_cast_types=[],
                    projection_type="tl.float32", result_writer_active="True",
                    stats="S", stats_offset="0", stats_writer_active="True")
    return call


def _render(call, corex, warps):
    return TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec(call["family"], call["variant"],
                           "corex" if corex else None, "bi_v150" if corex else None),
        dict(render_calls=[call], num_warps=warps, distributed_entry=False,
             mesh_hierarchy=(1, 1)),
    ).source


@pytest.mark.parametrize("family,variant", [
    ("qkv_parallel_linear", "packed_fused_gemv"),
    ("dense_matmul", "packed_k_major_gemv"),
    ("dense_matmul", "split_k_n_packed_k_major_gemv"),
    ("dense_matmul", "packed_k_major_gemv_norm_stats"),
    ("dense_matmul_glu", "packed_k_major_gemv"),
])
@pytest.mark.parametrize("seed,masked_tail", [(7, False), (32, False), (41, False), (71, True)])
def test_aligned_packed_gemv_preserves_all_results(tmp_path, family, variant, seed, masked_tail, qkv_n_lane=8, source_active=True, output_width=None):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    call = _call(family, variant)
    if family == "qkv_parallel_linear":
        call["n_lane"] = qkv_n_lane
        call["weight_n_group_stride"] = qkv_n_lane * 16
        for index, start in enumerate((0, 128, 192)):
            call["outputs"][index]["weight_group_start"] = start // qkv_n_lane
    if family == "qkv_parallel_linear" and not source_active:
        call["source_active"] = "False"
    if output_width is not None:
        call["local_n_capacity"] = output_width
        call["weight_offset"] = call["weight_offset"].replace("* 2048", f"* {output_width * 16}")
    reduction = 516 if masked_tail else 512
    width = 256 if family == "qkv_parallel_linear" else (output_width or 128)
    capacity = (width - 3 if masked_tail else width) if family != "qkv_parallel_linear" else 128
    call["local_k_capacity"] = reduction
    if family != "qkv_parallel_linear":
        call["local_n_capacity"] = capacity
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(reduction, generator=generator).bfloat16().cuda()

    def weight():
        groups = (reduction + 15) // 16
        dense = (torch.randn((width, groups * 16), generator=generator) / 32).bfloat16()
        return dense.reshape(width // 8, 8, groups, 16).permute(2, 0, 1, 3).contiguous().cuda()

    w = weight()
    if not source_active:
        w.fill_(float("inf"))
    u = weight() if family == "dense_matmul_glu" else None
    residual = torch.randn(capacity, generator=generator).cuda()
    results = []
    for corex, warps in ((False, 32), (True, 16)):
        path = tmp_path / f"aligned_{warps}.py"
        path.write_text("import triton\nimport triton.language as tl\n"
                        "import triton.experimental.tle.language as tle\n" + _render(call, corex, warps))
        spec = importlib.util.spec_from_file_location(f"aligned_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if family == "qkv_parallel_linear":
            output = [torch.empty(item["local_n_capacity"], device="cuda") for item in call["outputs"]]
            arguments = (x, w, *output)
        elif family == "dense_matmul_glu":
            output = [torch.empty(capacity, device="cuda", dtype=torch.bfloat16)]
            arguments = (x, w, u, *output)
        elif variant == "packed_k_major_gemv_norm_stats":
            output = [torch.empty(capacity, device="cuda"), torch.empty(1, device="cuda")]
            arguments = (x, w, residual, *output)
        else:
            output = [torch.empty(capacity, device="cuda")]
            arguments = (x, w, *output)
        module.run[(1,)](*arguments, num_warps=warps, enable_fp_fusion=False)
        torch.cuda.synchronize()
        results.append(output)
    for before, aligned in zip(*results):
        assert torch.equal(before, aligned)


@pytest.mark.parametrize("seed,masked_tail", [(7, False), (32, False), (41, False), (71, True)])
def test_batched_qkv_n4_groups_preserve_original_results(tmp_path, seed, masked_tail):
    test_aligned_packed_gemv_preserves_all_results(
        tmp_path, "qkv_parallel_linear", "packed_fused_gemv", seed, masked_tail, qkv_n_lane=4,
    )


@pytest.mark.parametrize("qkv_n_lane", [4, 8])
def test_batched_qkv_inactive_source_masks_nonfinite_weights(tmp_path, qkv_n_lane):
    test_aligned_packed_gemv_preserves_all_results(
        tmp_path, "qkv_parallel_linear", "packed_fused_gemv", 7, False,
        qkv_n_lane=qkv_n_lane, source_active=False,
    )


@pytest.mark.parametrize("variant", ("portable", "warps32", "block128", "width256"))
def test_batched_qkv_selection_preserves_other_template_contracts(variant):
    call = _call("qkv_parallel_linear", "packed_fused_gemv")
    warps = 32 if variant == "warps32" else 16
    if variant == "block128":
        call["block_k"] = 128
    if variant == "width256":
        call["outputs"][0]["local_n_capacity"] = 256
    source = _render(call, variant != "portable", warps)
    compile(source, variant, "exec")
    assert "batched packed outputs" not in source


@pytest.mark.parametrize("seed,masked_tail,width", [(7, False, 256), (32, True, 280), (41, False, 512)])
def test_projection_n256_rows_preserve_original_k64_tree(tmp_path, seed, masked_tail, width):
    test_aligned_packed_gemv_preserves_all_results(
        tmp_path, "dense_matmul", "split_k_n_packed_k_major_gemv", seed, masked_tail,
        output_width=width,
    )
