"""Correctness sweep + three-way timing for a data-movement operator on KL3.

Copy this next to the operator you are working on and edit `CASES` / `entry`. Run it
on an idle card with a warm cache:

    CUDA_VISIBLE_DEVICES=1 FLAGGEMS_CACHE_DIR=/tmp/gems_cache python sweep_copy.py

What it checks, and why each case is here:

  * windows on either side       -- gappy rows, the common non-contiguous layout
  * tails in both tile dims      -- `sizes` handling, where a mask would be wrong
  * broadcast rows / broadcast cols -- stride 0 outside the run is fine, inside is not
  * transpose                    -- the on-chip path, and its int64 refusal
  * rank 3-5                     -- outer dimensions riding the grid
  * every dtype, both directions of conversion, bool
  * `dst[::2]`                   -- must fall back, silently wrong otherwise

`taken=False` is a pass, not a skip: it means the entry point refused and the caller's
fallback runs. What must never happen is `taken=True` with wrong values.
"""

import importlib
import sys
import time

import torch

# --- edit these two ---------------------------------------------------------------
FLAGGEMS_SRC = "/home/users/jinchengxiong/baidu/public/FlagGems/src"
ENTRY = ("flag_gems.runtime.backend._kunlunxin.utils.tle_copy", "tle_copy")
# ----------------------------------------------------------------------------------

sys.path.insert(0, FLAGGEMS_SRC)
DEV = "cuda:0"
entry = getattr(importlib.import_module(ENTRY[0]), ENTRY[1])
FAILED = []


def rand(shape, dtype):
    x = torch.randn(shape, dtype=torch.float32) * 4
    return x.to(dtype).to(DEV)


def iota(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return torch.arange(n, dtype=dtype, device=DEV).reshape(shape)


def check(name, src, dst):
    """Run the entry point and compare against a host-side copy_."""
    ref = dst.clone().cpu()
    ref.copy_(src.cpu())
    taken = entry(src, dst)
    torch.cuda.synchronize()
    if not taken:
        print(f"SKIP {name}: fell back", flush=True)
        return
    bad = (dst.cpu() != ref).sum().item()
    if bad:
        FAILED.append(name)
        print(f"FAIL {name}: {bad}/{ref.numel()} mismatched", flush=True)
        print("      got", dst.cpu().reshape(-1)[:6].float().tolist())
        print("      ref", ref.reshape(-1)[:6].float().tolist())
    else:
        print(f"PASS {name}", flush=True)


def med(fn, iters=11):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return sorted(ts)[len(ts) // 2]


def timing(name, src, dst, native):
    """Your path vs the fallback vs native, in one process."""
    taken = entry(src, dst)
    torch.cuda.synchronize()
    if not taken:
        print(f"{name:<34} fell back", flush=True)
        return
    mine = med(lambda: entry(src, dst))
    fallback = med(lambda: dst.copy_(src))  # replace with the real fallback
    nat = med(native)
    print(
        f"{name:<34} mine={mine:9.1f}us  fallback={fallback:9.1f}us  native={nat:9.1f}us",
        flush=True,
    )


def correctness():
    f32, f16, bf16 = torch.float32, torch.float16, torch.bfloat16
    i8, i16, i32, i64 = torch.int8, torch.int16, torch.int32, torch.int64

    check("contig f32", rand((512, 256), f32), torch.zeros(512, 256, dtype=f32, device=DEV))
    check("dst window", rand((64, 100), f32), torch.zeros(64, 128, dtype=f32, device=DEV)[:, :100])
    check("src window", rand((64, 128), f32)[:, :100], torch.zeros(64, 100, dtype=f32, device=DEV))
    check(
        "both gappy tails",
        rand((37, 300), f32)[:, :251],
        torch.zeros(37, 400, dtype=f32, device=DEV)[:, :251],
    )
    check(
        "broadcast rows",
        rand((1, 251), f32).expand(37, 251),
        torch.zeros(37, 251, dtype=f32, device=DEV),
    )
    check(
        "broadcast cols (must fall back)",
        rand((37, 1), f32).expand(37, 251),
        torch.zeros(37, 251, dtype=f32, device=DEV),
    )
    check("transpose f32", rand((128, 64), f32).t(), torch.zeros(64, 128, dtype=f32, device=DEV))
    check(
        "dst step2 (must fall back)",
        rand((64, 100), f32),
        torch.zeros(64, 200, dtype=f32, device=DEV)[:, ::2],
    )
    check("rank3", rand((5, 17, 300), f32)[:, :, :251], torch.zeros(5, 17, 251, dtype=f32, device=DEV))
    check(
        "rank4",
        rand((3, 5, 17, 300), f32)[:, :, :, :251],
        torch.zeros(3, 5, 17, 251, dtype=f32, device=DEV),
    )
    for dt, tag in ((f16, "f16"), (bf16, "bf16"), (i8, "i8"), (i16, "i16")):
        check(f"both gappy {tag}", rand((37, 300), dt)[:, :251], torch.zeros(37, 251, dtype=dt, device=DEV))
    for dt, tag in ((i32, "i32"), (i64, "i64")):
        check(f"both gappy {tag}", iota((37, 300), dt)[:, :251], torch.zeros(37, 251, dtype=dt, device=DEV))
    check("transpose i64", iota((128, 64), i64).t(), torch.zeros(64, 128, dtype=i64, device=DEV))
    check("convert f32->f16", rand((37, 251), f32), torch.zeros(37, 251, dtype=f16, device=DEV))
    check("convert i8->f32 (SDNN cast is zeros)", rand((37, 251), i8), torch.zeros(37, 251, dtype=f32, device=DEV))
    b = rand((37, 251), f32) > 0
    check("bool->f32", b, torch.zeros(37, 251, dtype=f32, device=DEV))
    check("f32->bool", rand((37, 251), f32), torch.zeros(37, 251, dtype=torch.bool, device=DEV))
    check("single element", rand((1, ), f32), torch.zeros(1, dtype=f32, device=DEV))


def perf():
    f16 = torch.float16
    for n in (256, 1024, 2048, 4096):
        src = rand((n, n), f16)
        dst = torch.zeros(n, n, dtype=f16, device=DEV)
        timing(f"transpose f16 {n}x{n}", src.t(), dst, lambda s=src: s.t().contiguous())
    for rows, cols in ((256, 1024), (2048, 2048), (4096, 4096)):
        src = rand((rows, cols + 64), f16)[:, :cols]
        dst = torch.zeros(rows, cols + 32, dtype=f16, device=DEV)[:, :cols]
        timing(f"window f16 {rows}x{cols}", src, dst, lambda d=dst, s=src: d.copy_(s))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "correctness"):
        correctness()
        print("\nFAILED:", FAILED if FAILED else "none")
    if what in ("all", "perf"):
        print()
        perf()
