# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: MIT

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_all_blocks(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + index, index < N, other=0.0)
    tl.store(dst + index, value, index < N)


@pytest.mark.parametrize("environment_enabled", [False, True])
@pytest.mark.parametrize("kernel_enabled", [None, False, True])
def test_a2_a3_auto_blockify_covers_full_grid(monkeypatch, tmp_path, environment_enabled, kernel_enabled):
    from triton.backends.ascend.driver import NPUUtils

    arch = NPUUtils().get_arch()
    if not arch.startswith(("Ascend910B", "Ascend910_93")):
        pytest.skip("requires an A2/A3 compilation target")
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1" if environment_enabled else "0")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    _copy_all_blocks.device_caches.clear()
    # Several times the physical core count, including a partial final tile.
    count = (NPUUtils().get_aivector_core_num() * 4 + 3) * 256 - 17
    source = torch.arange(count, dtype=torch.float32, device="npu")
    output = torch.full_like(source, -1)
    _copy_all_blocks[(triton.cdiv(count, 256), )](
        source,
        output,
        count,
        256,
        enable_auto_blockify=kernel_enabled,
    )
    torch.testing.assert_close(output.cpu(), source.cpu(), rtol=0, atol=0)
