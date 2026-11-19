# flagtree tle
"""Representative-workload benchmarks for the tsingmicro TLE DSA backend.

Workloads (tsingmicro TX8100, float32 unless noted):
  - tle.cumsum (tle-lite)      vs torch.cumsum
  - tle.dsa.copy  GM->SPM->GM  vs torch clone (same memory traffic)
  - tle.dsa.add   SPM elementwise vs torch add
  - tle.dsa.tsingmicro.randn hardware TRNG  vs torch.randn

Usage: python test/tle/benchmark/bench_tle_dsa.py
"""

import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.testing import do_bench

DEVICE = "txda"


def _is_txda():
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        return False
    return triton.runtime.driver.active.get_current_target().backend == "txda"


# ----------------------------- cumsum ---------------------------------------


@triton.jit
def _cumsum_kernel(x_ptr, ex_ptr, t_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    exclusive, total = tle.cumsum(x, axis=0)
    tl.store(ex_ptr + offs, exclusive, mask=mask)
    tl.store(t_ptr, total)


def bench_cumsum(sizes):
    for n in sizes:
        block = triton.next_power_of_2(n)
        x = torch.randn(n, device=DEVICE, dtype=torch.float32)
        ex = torch.empty(n, device=DEVICE, dtype=torch.float32)
        t = torch.empty(1, device=DEVICE, dtype=torch.float32)
        ms = do_bench(lambda: _cumsum_kernel[(1, )](x, ex, t, n, BLOCK=block), warmup=25, rep=100)
        ref = torch.cumsum(x, dim=0)
        err = (ex[:n] - (ref - x)).abs().max().item()
        torch_ms = do_bench(lambda: torch.cumsum(x, dim=0), warmup=25, rep=100)
        print(f"cumsum  n={n:>7}  tle.cumsum={ms:8.4f} ms  torch.cumsum={torch_ms:8.4f} ms  max_err={err:.2e}")


# ----------------------------- copy -----------------------------------------


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, shape: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    buf = tle.dsa.alloc([BLOCK], dtype=tl.float32, scope=tle.dsa.tsingmicro.SPM)
    tle.dsa.copy(src_ptr + offs, buf, [BLOCK])
    tle.dsa.copy(buf, dst_ptr + offs, [BLOCK])


def bench_copy(sizes):
    for n in sizes:
        block = triton.next_power_of_2(n)
        x = torch.randn(n, device=DEVICE, dtype=torch.float32)
        y = torch.empty_like(x)
        ms = do_bench(lambda: _copy_kernel[(1, )](x, y, n, BLOCK=block), warmup=25, rep=100)
        torch_ms = do_bench(lambda: y.clone(), warmup=25, rep=100)
        ok = torch.equal(x, y)
        print(f"copy    n={n:>7}  dsa.copy={ms:8.4f} ms  torch.clone={torch_ms:8.4f} ms  ok={ok}")


# ----------------------------- add ------------------------------------------


@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, shape: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tle.dsa.alloc([BLOCK], dtype=tl.float32, scope=tle.dsa.tsingmicro.SPM)
    b = tle.dsa.alloc([BLOCK], dtype=tl.float32, scope=tle.dsa.tsingmicro.SPM)
    c = tle.dsa.alloc([BLOCK], dtype=tl.float32, scope=tle.dsa.tsingmicro.SPM)
    tle.dsa.copy(a_ptr + offs, a, [BLOCK])
    tle.dsa.copy(b_ptr + offs, b, [BLOCK])
    tle.dsa.add(a, b, c)
    t = tle.dsa.to_tensor(c)
    tl.store(c_ptr + offs, t)


def bench_add(sizes):
    for n in sizes:
        block = triton.next_power_of_2(n)
        a = torch.randn(n, device=DEVICE, dtype=torch.float32)
        b = torch.randn(n, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)
        ms = do_bench(lambda: _add_kernel[(1, )](a, b, c, n, BLOCK=block), warmup=25, rep=100)
        torch_ms = do_bench(lambda: a + b, warmup=25, rep=100)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)
        print(f"add     n={n:>7}  dsa.add={ms:8.4f} ms  torch.add={torch_ms:8.4f} ms")


# ----------------------------- randn ----------------------------------------


@triton.jit
def _randn_kernel(out_ptr, seed, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed + 1
    val, s0, s1 = tle.dsa.tsingmicro.randn(s0, s1, BLOCK)
    tl.store(out_ptr + offs, val, mask=mask)


def bench_randn(sizes):
    for n in sizes:
        block = triton.next_power_of_2(n)
        out = torch.empty(n, device=DEVICE, dtype=torch.float32)
        ms = do_bench(lambda: _randn_kernel[(1, )](out, 42, n, BLOCK=block), warmup=25, rep=100)
        torch_ms = do_bench(lambda: torch.randn(n, device=DEVICE, dtype=torch.float32), warmup=25, rep=100)
        m, s = out.float().mean().item(), out.float().std().item()
        print(
            f"randn   n={n:>7}  dsa.tsingmicro.randn={ms:8.4f} ms  torch.randn={torch_ms:8.4f} ms  mean={m:+.3f} std={s:.3f}"
        )


if __name__ == "__main__":
    if not _is_txda():
        print("This benchmark requires the TsingMicro (txda) backend")
        sys.exit(1)
    print(f"device: TX8100 (txda), torch {torch.__version__}, triton {triton.__version__}")
    bench_cumsum([1024, 4096, 16384])
    bench_copy([1024, 4096, 16384, 32768, 65536])
    bench_add([1024, 4096, 16384, 32768])
    bench_randn([1024, 4096, 16384])
