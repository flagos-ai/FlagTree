import re
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


def _has_nvidia_hopper_gpu() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(
    not _has_nvidia_hopper_gpu(),
    reason="warp-specialize helper barriers require NVIDIA Hopper (sm90+)",
)


def _assert_barrier(ptx: str, kind: str, barrier_id: int, num_threads: int) -> None:
    pattern = rf"\bbar\.{kind}(?:\.[a-z]+)*\s+{barrier_id},\s*{num_threads}\s*;"
    assert re.search(pattern, ptx), f"missing bar.{kind} {barrier_id}, {num_threads} in PTX:\n{ptx}"


@triton.jit(noinline=True)
def _debug_barrier_fill(out):
    tl.debug_barrier()
    offsets = tl.arange(0, 128)
    tl.store(out + offsets, 1)


@triton.jit
def _debug_barrier_default(out):
    _debug_barrier_fill(out)


@triton.jit
def _debug_barrier_worker(out):
    _debug_barrier_fill(out + 128)
    _debug_barrier_fill(out + 128)


@triton.jit
def _debug_barrier_helper_kernel(out):
    tle.gpu.warp_specialize(
        [
            (_debug_barrier_default, (out, )),
            (_debug_barrier_worker, (out, )),
        ],
        [2],
        [168],
    )


@pytest.mark.require_tle("gpu.warp_specialize")
def test_debug_barrier_in_outlined_default_and_worker_helpers():
    out = torch.zeros(256, device="cuda", dtype=torch.int32)

    compiled = _debug_barrier_helper_kernel[(1, )](out, num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, torch.ones_like(out))
    ptx = compiled.asm["ptx"]
    _assert_barrier(ptx, "sync", 0, 128)
    _assert_barrier(ptx, "sync", 2, 64)


@triton.jit(noinline=True)
def _nested_barrier_helper_b(out):
    tl.debug_barrier()
    offsets = tl.arange(0, 64)
    tl.store(out + offsets, 7)


@triton.jit(noinline=True)
def _nested_barrier_helper_a(out):
    _nested_barrier_helper_b(out)


@triton.jit
def _nested_barrier_default(out, n):
    tl.store(out + 64, n + 7)


@triton.jit
def _nested_barrier_worker(out, n):
    for i in range(n):
        if i == n - 1:
            _nested_barrier_helper_a(out)


@triton.jit
def _nested_barrier_helper_kernel(out, n):
    tle.gpu.warp_specialize(
        [
            (_nested_barrier_default, (out, n)),
            (_nested_barrier_worker, (out, n)),
        ],
        [2],
        [168],
    )


@pytest.mark.require_tle("gpu.warp_specialize")
def test_debug_barrier_propagates_through_nested_helpers_and_control_flow():
    out = torch.zeros(65, device="cuda", dtype=torch.int32)

    compiled = _nested_barrier_helper_kernel[(1, )](out, 2, num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(out[:64], torch.full_like(out[:64], 7))
    assert out[64].item() == 9
    _assert_barrier(compiled.asm["ptx"], "sync", 2, 64)


@dialect(
    name="cuda",
    file=Path(__file__).with_name("tle_warp_specialize_raw_helper.cu"),
    extern_func_name="tle_ws_raw_increment",
)
def _raw_increment(*args, **kwargs):
    ...


@triton.jit(noinline=True)
def _raw_debug_barrier_helper(counter):
    tl.debug_barrier()
    tle_raw.call(_raw_increment, [counter])


@triton.jit
def _raw_debug_barrier_default(counters):
    _raw_debug_barrier_helper(counters)


@triton.jit
def _raw_debug_barrier_worker(counters):
    _raw_debug_barrier_helper(counters + 1)
    _raw_debug_barrier_helper(counters + 1)


@triton.jit
def _raw_debug_barrier_helper_kernel(counters):
    tle.gpu.warp_specialize(
        [
            (_raw_debug_barrier_default, (counters, )),
            (_raw_debug_barrier_worker, (counters, )),
        ],
        [2],
        [168],
    )


@pytest.mark.require_tle("gpu.warp_specialize", "raw.call")
def test_raw_call_and_debug_barrier_in_outlined_helpers():
    counters = torch.zeros(2, device="cuda", dtype=torch.int32)

    compiled = _raw_debug_barrier_helper_kernel[(1, )](counters, num_warps=4)
    torch.cuda.synchronize()

    torch.testing.assert_close(counters.cpu(), torch.tensor([4, 4], dtype=torch.int32))
    ptx = compiled.asm["ptx"]
    _assert_barrier(ptx, "sync", 0, 128)
    _assert_barrier(ptx, "sync", 2, 64)

    llir = compiled.asm["llir"]
    raw_definition = re.search(
        r"^define [^{\n]*@tle_ws_raw_increment\((?P<args>.*)\) [^{]*\{$",
        llir,
        re.MULTILINE,
    )
    assert raw_definition
    assert "ptr addrspace(1)" in raw_definition.group("args")
    assert "," not in raw_definition.group("args")
    assert "@tle_ws_raw_increment_ws" not in llir
