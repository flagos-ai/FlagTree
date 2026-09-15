# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compile and execute Cube boundaries around a TLE dot product."""
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def boundary_dot(A, B, Out):
    with tle.scope(core_mode="cube"):
        pid = tl.program_id(0)
        tle.dsa.ascend.raw("cube_begin", pid)
        lane = tl.arange(0, 16)
        a = tl.load(A + lane[:, None] * 16 + lane[None, :])
        b = tl.load(B + lane[:, None] * 16 + lane[None, :])
        result = tl.dot(a, b)
        tl.store(Out + lane[:, None] * 16 + lane[None, :], result)
        tle.dsa.ascend.raw("cube_end", pid)


def test_cube_boundaries():
    a = torch.eye(16, dtype=torch.float16, device="npu")
    b = torch.arange(256, dtype=torch.float16, device="npu").reshape(16, 16)
    out = torch.empty_like(b)
    boundary_dot[(1, )](a, b, out)
    torch.testing.assert_close(out, b, atol=0, rtol=0)
