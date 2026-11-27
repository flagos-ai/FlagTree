# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Weight reuse must preserve sequential arithmetic and every output span."""

import pytest
import torch

from triton.flagmega.codegen.triton.iq4_q8_kernels import dp4a_m1, q8_prepare
from triton.flagmega.codegen.triton.iq4_q8_multi_column import dp4a_columns


@pytest.mark.parametrize("rows", [2, 5])
@pytest.mark.parametrize("mode", [0, 2])
def test_projection_matches_sequential_rows(rows, mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Accepted r003 numerical subject requires SM120")
    torch.manual_seed(35)
    n, k = (5120, 17408) if mode == 0 else (17408, 5120) if mode == 1 else (12288, 5120)
    guard = 64
    width = n // 2 if mode == 2 else n
    dtype = torch.float32 if mode == 0 else torch.bfloat16
    x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda")
    weights = torch.randint(0, 256, (n, k // 256 * 136), dtype=torch.uint8, device="cuda")
    # Keep random IQ4 codes/scales while giving each K256 block a finite d.
    weights.view(torch.float16).reshape(n, -1)[:, ::68] = 1 / 128
    up_weights = torch.randint(0, 256, weights.shape, dtype=torch.uint8, device="cuda")
    up_weights.view(torch.float16).reshape(n, -1)[:, ::68] = 1 / 256
    q = torch.empty((rows, k // 32, 36), dtype=torch.uint8, device="cuda")
    q8_prepare[(k // 128,)](x, q, K=k, ROWS=rows, num_warps=4)

    def project(record, count):
        fields = [
            torch.full((count * width + 2 * guard,), -7, dtype=dtype, device="cuda")
            for _ in range(2 if mode == 2 else 1)
        ]
        outputs = [field[guard:-guard] for field in fields]
        for output in outputs:
            output.fill_(float("nan"))
        y, z = outputs[0], outputs[-1]
        budget = (
            (80 if mode == 1 else 80 if k == 17408 else 56)
            if count > 1
            else (104 if mode == 1 else 96 if mode == 2 else 80)
        )
        fn = dp4a_m1 if count == 1 else dp4a_columns
        extra = {} if count == 1 else {"COLS": 4}
        budget = budget if count == 1 else 80
        kernel = fn[(n if count == 1 else (n + 3) // 4,)](
            record,
            weights,
            up_weights,
            y,
            z,
            N=n,
            K=k,
            MODE=mode,
            ROWS=count,
            num_warps=4,
            num_stages=3,
            maxnreg=budget,
            enable_static_shared=True,
            enable_fp_fusion=False,
            ptx_options="--minnctapersm=1",
            **extra,
        )
        torch.cuda.synchronize()
        assert kernel.metadata.ptxas_spill_store_bytes == 0
        assert kernel.metadata.ptxas_spill_load_bytes == 0
        for field in fields:
            assert bool((field[:guard] == -7).all())
            assert bool((field[-guard:] == -7).all())
        return [output.reshape(count, width) for output in outputs]

    actual = project(q, rows)
    expected = [project(q[row], 1) for row in range(rows)]
    for field in range(len(actual)):
        assert bool(torch.isfinite(actual[field]).all())
        reference = torch.cat([value[field] for value in expected])
        torch.testing.assert_close(actual[field], reference, rtol=0, atol=0)
        # Check the published bits too, including the sign of zero for F32.
        bits = torch.int32 if dtype == torch.float32 else torch.int16
        assert bool((actual[field].view(bits) == reference.view(bits)).all())
