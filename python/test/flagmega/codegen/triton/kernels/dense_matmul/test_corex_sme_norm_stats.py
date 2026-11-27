# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Warp-owned packed down projection retains publication and RMS folds."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


@pytest.mark.parametrize("n,k,projection,output", (
    (32, 256, "tl.float32", "tl.float32"),
    (64, 384, "tl.bfloat16", "tl.float32"),
    (96, 512, "tl.bfloat16", "tl.bfloat16"),
    (128, 512, "tl.float32", "tl.float32"),
    (128, 6144, "tl.float32", "tl.float32"),
    (128, 6144, "tl.float32", "tl.bfloat16"),
    (128, 512, "tl.float32", "cancellation"),
))
def test_sme_down_matches_original_results_and_statistics(tmp_path, n, k, projection, output):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    call = dict(family="dense_matmul", variant="packed_k_major_gemv_norm_stats",
                symbol="run", signature="X, W, R, Y, S", local_m_capacity=1,
                local_n_capacity=n, local_k_capacity=k, tile_n=64, block_k=128,
                source="X", weight="W", result="Y", stats="S", residual="R",
                source_offset="dense_local_k_offsets", result_offset="dense_local_n_offsets",
                residual_offset="dense_local_n_offsets", stats_offset="0",
                source_active="True", result_active="True",
                result_writer_active="True", stats_writer_active="True",
                global_n="dense_local_n_offsets", global_k="dense_local_k_offsets",
                projection_type=projection, output_type=output, addend_cast_types=["tl.bfloat16"],
                weight_offset=f"(dense_global_k[None, :] // 16) * {n * 16} + "
                              "(dense_global_n[:, None] // 8) * 128 + "
                              "(dense_global_n[:, None] % 8) * 16 + dense_global_k[None, :] % 16")
    generator = torch.Generator().manual_seed(n + k)
    x = torch.randn(k, generator=generator).bfloat16().cuda()
    dense = (torch.randn((n, k), generator=generator) / 32).bfloat16()
    weight = dense.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().cuda()
    if output == "cancellation":
        # A balanced four-value fold returns zero; the original sequential
        # register fold returns one before the subsequent XOR reductions.
        x.fill_(1)
        dense.zero_()
        dense[:, :4] = torch.tensor([33554432., 1., -33554432., 1.]).bfloat16()
        weight = dense.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().cuda()
        call["output_type"] = "tl.float32"
    dtype = torch.bfloat16 if output == "tl.bfloat16" else torch.float32
    residual = torch.randn(n, generator=generator).to(dtype).cuda()
    if output == "cancellation":
        residual.zero_()
    results = []
    for corex, warps in ((False, 32), (True, 16)):
        if corex:
            call["corex_flat_sme_down"] = dict(source="X", weight="W", weight_k_group_stride=n * 16)
        source = TritonTemplateRegistry().render_kernel(
            KernelTemplateSpec("dense_matmul", "packed_k_major_gemv_norm_stats",
                               "corex" if corex else None, "bi_v150" if corex else None),
            dict(render_calls=[call], num_warps=warps,
                 distributed_entry=False, mesh_hierarchy=(1, 1)),
        ).source
        path = tmp_path / f"down_{warps}.py"
        path.write_text("import triton\nimport triton.language as tl\n"
                        "import triton.experimental.tle.language as tle\n" + source)
        spec = importlib.util.spec_from_file_location(f"down_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        y = torch.empty(n, device="cuda", dtype=dtype)
        stats = torch.empty(1, device="cuda")
        module.run[(1,)](x, weight, residual, y, stats, num_warps=warps, enable_fp_fusion=False)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            module.run[(1,)](x, weight, residual, y, stats, num_warps=warps, enable_fp_fusion=False)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        results.append((y, stats))
    for before, after in zip(*results):
        assert torch.equal(before, after)


@pytest.mark.parametrize("n,k", ((32, 256), (128, 6144)))
def test_sme_down_all_ctas_preserve_outputs_statistics_and_guards(tmp_path, n, k):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    owners, guard = 16, 17
    owner = "tl.program_id(0)"
    call = dict(
        family="dense_matmul", variant="packed_k_major_gemv_norm_stats", symbol="run",
        signature="X, W, R, Y, S", local_m_capacity=1, local_n_capacity=n,
        local_k_capacity=k, tile_n=64, block_k=128, source="X", weight="W",
        result="Y", stats="S", residual="R", source_offset="dense_local_k_offsets",
        result_offset=f"{guard}+{owner}*{n}+dense_local_n_offsets",
        residual_offset=f"{owner}*{n}+dense_local_n_offsets", stats_offset=f"{guard}+{owner}",
        source_active="True", result_active="True", result_writer_active="True",
        stats_writer_active="True", global_n="dense_local_n_offsets",
        global_k="dense_local_k_offsets", projection_type="tl.float32",
        output_type="tl.float32", addend_cast_types=[],
        weight_offset=f"{owner}*{n*k}+(dense_global_k[None,:]//16)*{n*16}"
                      "+dense_global_n[:,None]*16+dense_global_k[None,:]%16",
    )
    generator = torch.Generator().manual_seed(owners + n + k)
    x = torch.randn(k, generator=generator).bfloat16().cuda()
    dense = (torch.randn(owners, n, k, generator=generator) / 32).bfloat16()
    weight = dense.reshape(owners, n // 8, 8, k // 16, 16).permute(0, 3, 1, 2, 4).contiguous().cuda()
    residual = torch.randn(owners * n, generator=generator).cuda()
    results = []
    for corex, warps in ((False, 32), (True, 16)):
        if corex:
            call["corex_flat_sme_down"] = dict(
                source="X", weight=f"(W+{owner}*{n*k})", weight_k_group_stride=n * 16)
        source = TritonTemplateRegistry().render_kernel(
            KernelTemplateSpec("dense_matmul", "packed_k_major_gemv_norm_stats",
                               "corex" if corex else None, "bi_v150" if corex else None),
            dict(render_calls=[call], num_warps=warps,
                 distributed_entry=False, mesh_hierarchy=(1, 1)),
        ).source
        path = tmp_path / f"down_grid_{warps}.py"
        path.write_text("import triton\nimport triton.language as tl\n"
                        "import triton.experimental.tle.language as tle\n" + source)
        spec = importlib.util.spec_from_file_location(f"down_grid_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        y = torch.full((guard + owners * n + guard,), -7., device="cuda")
        stats = torch.full((guard + owners + guard,), -7., device="cuda")
        module.run[(owners,)](x, weight, residual, y, stats,
                             num_warps=warps, enable_fp_fusion=False)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            module.run[(owners,)](x, weight, residual, y, stats,
                                 num_warps=warps, enable_fp_fusion=False)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        for value in (y, stats):
            assert bool(torch.isfinite(value).all())
            assert torch.equal(value[:guard], torch.full_like(value[:guard], -7.))
            assert torch.equal(value[-guard:], torch.full_like(value[-guard:], -7.))
        results.append((y, stats))
    for before, after in zip(*results):
        assert torch.equal(before, after)
