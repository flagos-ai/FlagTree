# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Uniform-row specialization preserves masking, tails and projection rounding."""

import pytest

from python.test.flagmega.codegen.triton.kernels.gdn_recurrent.projection_helpers import build_projection_kernel


@pytest.mark.parametrize("value_tile,k,tile_k", [(4, 32, 128), (8, 2056, 1024), (16, 2048, 512)])
@pytest.mark.parametrize("case", ["uniform", "inactive_first", "empty", "mixed"])
def test_dense_projection_has_exact_dyadic_results(tmp_path, value_tile, k, tile_k, case):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    kernel, _ = build_projection_kernel(tmp_path, value_tile)
    generator = torch.Generator().manual_seed(2217)
    source = (torch.randint(-8, 9, (k,), generator=generator) / 16).bfloat16().cuda()
    weight = (torch.randint(-8, 9, (4, k), generator=generator) / 64).bfloat16().cuda()
    rows = torch.full((value_tile,), 2, dtype=torch.int64, device="cuda")
    active = torch.ones(value_tile, dtype=torch.bool, device="cuda")
    if case == "inactive_first":
        rows[0] = 0x7000000000000000  # Never dereference an inactive row.
        active[0] = False
    elif case == "empty":
        rows.fill_(-1)
        active.fill_(False)
    elif case == "mixed":
        rows.copy_(torch.arange(value_tile, device="cuda") % 4)
    output = torch.empty(value_tile, dtype=torch.float32, device="cuda")
    projections = (weight.float() @ source.float()).bfloat16().float()
    expected = torch.where(active, projections[rows.clamp(0, 3)], 0.0)
    for uniform in (False, True) if case != "mixed" else (False,):
        output.fill_(float("nan"))
        kernel[(1,)](source, weight, rows, active, output, k, tile_k, value_tile, uniform, num_warps=8)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
