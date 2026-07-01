"""
Mixed Kernel Routing
====================

This tutorial demonstrates per-kernel routing in one Python process.  With
``FLAGTREE_USE_TILEIR=1`` enabled, a plain Triton kernel routes to TileIR, while
a kernel using non-TileIR TLE shared-memory APIs falls back to the native NVIDIA
path.
"""

import os
import pathlib

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.experimental.tle.language as tle  # noqa: E402


@triton.jit
def _plain_add_kernel(x, y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(y + offsets, values + 1.0, mask=mask)


@triton.jit
def _tle_shared_memory_kernel(x, y, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    smem = tle.gpu.alloc([BLOCK_SIZE], dtype=tl.float32, scope=tle.gpu.smem)
    smem_ptrs = tle.gpu.local_ptr(smem, (offsets, ))
    values = tl.load(x + offsets)
    tl.store(smem_ptrs, values)
    copied = tl.load(smem_ptrs)
    tl.store(y + offsets, copied + 2.0)


def _cache_files(cache_dir: pathlib.Path):
    return sorted(path.name for path in cache_dir.glob("**/*") if path.is_file())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("this tutorial requires a CUDA GPU")

    cache_dir = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", "/tmp/flagtree_tileir_tutorial_cache/02"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ["FLAGTREE_USE_TILEIR"] = "1"

    n_elements = 1024
    block = 256
    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y_tileir = torch.empty_like(x)
    y_native = torch.empty(block, device="cuda", dtype=torch.float32)

    plain_grid = (triton.cdiv(n_elements, block), )
    _plain_add_kernel[plain_grid](x, y_tileir, n_elements, BLOCK_SIZE=block, num_warps=4)
    _tle_shared_memory_kernel[(1, )](x, y_native, BLOCK_SIZE=block, num_warps=4)

    torch.testing.assert_close(y_tileir, x + 1)
    torch.testing.assert_close(y_native, x[:block] + 2)

    files = _cache_files(cache_dir)
    plain_used_tileir = any(name == "_plain_add_kernel.tileir" for name in files)
    tle_used_native = any(name == "_tle_shared_memory_kernel.ptx" for name in files)
    tle_used_tileir = any(name == "_tle_shared_memory_kernel.tileir" for name in files)
    if not plain_used_tileir:
        raise AssertionError("plain Triton kernel did not produce TileIR cache output")
    if not tle_used_native or tle_used_tileir:
        raise AssertionError("TLE shared-memory kernel did not fall back to native NVIDIA")

    print("[OK] FLAGTREE_USE_TILEIR=1: plain Triton kernel routed to TileIR")
    print("[OK] FLAGTREE_USE_TILEIR=1: non-TileIR TLE kernel routed to native NVIDIA in the same process")
    print(f"[OK] TRITON_CACHE_DIR={cache_dir}")


if __name__ == "__main__":
    main()
