"""Run FlagTune's baseline tuning path on an installed Ascend backend."""

import pytest

import triton
import triton.language as tl
from triton.flagtune import flagtune
from triton.flagtune.runtime.device import probe_flagtune_device

# Keep collection possible on machines without the optional Ascend runtime.
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ascend_event_autotuning(dtype, monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if triton.runtime.driver.active.get_current_target().backend != "npu":
        pytest.skip("requires Triton's Ascend driver")
    monkeypatch.setenv("FLAGTUNE_BENCHMARK_MODE", "event")

    @flagtune(
        configs=[triton.Config({"BLOCK_SIZE": block}) for block in (256, 512)],
        key=["N"],
        warmup=5,
        rep=20,
    )
    @triton.jit
    def add(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + offsets, offsets < N, other=0)
        y = tl.load(Y + offsets, offsets < N, other=0)
        tl.store(Z + offsets, x + y, offsets < N)

    descriptor = probe_flagtune_device()
    assert descriptor.vendor == "huawei"
    assert descriptor.torch_device_type == "npu"
    x = torch.randn(4097, dtype=dtype, device="npu")
    y = torch.randn_like(x)
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]), )
    # First call tunes; the second exercises the selected configuration cache.
    for _ in range(2):
        add[grid](x, y, z, x.numel())
        torch.npu.synchronize()
        torch.testing.assert_close(z, x + y)
    assert add.benchmark_protocol.resolved_mode.value == "event"
    assert add.cache
