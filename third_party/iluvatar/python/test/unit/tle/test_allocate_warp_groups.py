"""Padding partitions must exactly fill the reserved physical warp groups."""

import re

import pytest
from triton._C.libtriton import iluvatar, ir, passes


@pytest.mark.parametrize("worker_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("max_worker_warps", [4, 8, 16])
def test_warp_group_padding_fills_without_overshooting(tmp_path, worker_warps, max_worker_warps):
    regions = []
    for warps in (worker_warps, max_worker_warps):
        regions.append(f"""
    ttg.warp_specialize()
    default {{
      ttg.warp_yield
    }}
    partition0() num_warps({warps}) {{
      ttg.warp_return
    }} : () -> ()
""")
    source = tmp_path / "warp_padding.ttgir"
    source.write_text("""
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:71", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @kernel() {
""" + "".join(regions) + "    tt.return\n  }\n}\n")
    context = ir.context()
    ir.load_dialects(context)
    iluvatar.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    module.context = context
    pm = ir.pass_manager(context)
    passes.ttgpuir.add_allocate_warp_groups(pm)
    pm.run(module, "warp_group_padding")

    extra_warps = max(worker_warps, max_worker_warps)
    assert module.get_int_attr("ttg.total-num-warps") == 4 + extra_warps
    allocated_regions = module.str_nodebug().split("ttg.warp_specialize")[1:]
    assert len(allocated_regions) == 2
    for original, region in zip((worker_warps, max_worker_warps), allocated_regions):
        sizes = [int(size) for size in re.findall(r"num_warps\((\d+)\)", region)]
        assert sum(sizes) == extra_warps, region
        expected_sizes = [original]
        remaining = extra_warps - original
        while remaining:
            size = 1 << (remaining.bit_length() - 1)
            expected_sizes.append(size)
            remaining -= size
        assert sizes == expected_sizes, region
        starts = re.search(r"warpGroupStartIds = array<i32: ([\d, ]+)>", region)
        assert starts, region
        start_ids = [int(start) for start in starts.group(1).split(",")]
        assert len(start_ids) == len(sizes)
        next_start = 4
        for start, size in sorted(zip(start_ids, sizes)):
            assert start == next_start, region
            next_start += size
        assert next_start == 4 + extra_warps, region
