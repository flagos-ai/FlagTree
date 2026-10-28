# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Batched gather keeps call offsets, aliasing and the original RMS tree."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _call(partial_base=128, stats_base=0, norm_base=16640, active="True"):
    offset = "shard_y*512+gather_local_offsets"
    return dict(
        family="gather_reduce_add_norm_apply", variant="sum", symbol="run",
        signature="P, R, W, Y, N, S", tile=16, local_capacity=512,
        work_partition_count=4, work_partition_index="shard_x",
        partial_owner_count=4, partial_reduction_width=4,
        stats_owner_count=16, reduction_width=16, normalization_size=2048,
        partial=f"P+{partial_base}", owner_stride=512,
        partial_owner="shard_y*4+gather_partial_lane",
        partial_offset="gather_local_offsets", addend="R", addend_offset=offset,
        value_result="Y", value_offset=offset, norm_result=f"N+{norm_base}",
        norm_offset=offset, scale="W", scale_offset=offset,
        bias="W", bias_offset=offset, stats_partials=f"S+{stats_base}",
        stats_square_base=0, stats_writer_index="shard_y*4+shard_x",
        participant_active="True", active=active, epsilon="1e-6",
        use_mean=False, has_bias=False, round_before_scale=False,
        output_type="tl.float32", norm_output_type="tl.bfloat16",
    )


def _render(call, corex=True, warps=16):
    return TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gather_reduce_add_norm_apply", "sum",
                           "corex" if corex else None, "bi_v150" if corex else None),
        dict(render_calls=[call], num_warps=warps, distributed_entry=True,
             mesh_hierarchy=(4, 4), mesh_axis_names=("block_y", "block_x")),
    ).source


@pytest.mark.parametrize("neighbor", ["warps32", "mean", "bias", "round", "tile", "capacity"])
def test_other_collective_contracts_keep_their_template(neighbor):
    call = _call()
    warps = 32 if neighbor == "warps32" else 16
    if neighbor in ("mean", "bias", "round"):
        call[{"mean": "use_mean", "bias": "has_bias", "round": "round_before_scale"}[neighbor]] = True
    elif neighbor == "tile":
        call["tile"] = 8
    elif neighbor == "capacity":
        call["local_capacity"] = 511
    source = _render(call, warps=warps)
    assert "gather_start in tl.range" in source
    compile(source, neighbor, "exec")


@pytest.mark.parametrize("partial_base,stats_base,norm_base,alias,cancellation,masked", [
    (128, 0, 16640, True, False, False),
    (1024, 9216, 0, True, False, False),
    (37, 19, 23, False, False, True),
    (128, 0, 16640, True, True, False),
])
def test_batched_gather_matches_original_values_statistics_and_normalization(
    tmp_path, partial_base, stats_base, norm_base, alias, cancellation, masked,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    call = _call(partial_base, stats_base, norm_base,
                 "gather_local_offsets%11 != 0" if masked else "True")
    torch.manual_seed(41)
    partial = torch.randn(partial_base + 16*512, device="cuda") / 32
    residual = torch.randn(2048, device="cuda")
    scale = torch.randn(2048, device="cuda").bfloat16()
    if cancellation:
        # Each lane sums a large square followed by seven ones. A balanced
        # eight-value fold retains small terms which the original loop loses.
        partial.zero_()
        residual.fill_(1)
        for y in range(4):
            residual[y*512:y*512+64] = 4096
    results = []
    for corex, warps in ((False, 32), (True, 16)):
        path = tmp_path / f"gather_{warps}.py"
        source = ("import triton\nimport triton.language as tl\n"
                  "import triton.experimental.tle.language as tle\n"
                  "FLAGMEGA_GRID_MESH = tl.constexpr(tle.device_mesh({'block': [('block_y',4),('block_x',4)]}))\n")
        path.write_text(source + _render(call, corex, warps))
        spec = importlib.util.spec_from_file_location(f"gather_{warps}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        addend = residual.clone()
        value = addend if alias else torch.full_like(addend, -7)
        norm = torch.full((norm_base+2048+17,), -7, device="cuda", dtype=torch.bfloat16)
        stats = torch.full((stats_base+16+17,), -7, device="cuda", dtype=torch.float32)
        arguments = (partial, addend, scale, value, norm, stats)
        module.run__partials[(16,)](*arguments, num_warps=warps, enable_fp_fusion=False)
        module.run__finalize[(16,)](*arguments, num_warps=warps, enable_fp_fusion=False)
        torch.cuda.synchronize()
        results.append((value.clone(), norm.clone(), stats.clone()))
    for before, after in zip(*results):
        assert torch.equal(before, after)
    if cancellation:
        assert torch.equal(results[0][2][stats_base:stats_base+16],
                           torch.full((16,), 16*4096**2, device="cuda", dtype=torch.float32))
