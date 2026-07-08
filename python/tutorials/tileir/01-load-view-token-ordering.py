"""
Load View Token Ordering
========================

This tutorial runs a minimal TileIR-only TLE view-token kernel.  The kernel
creates tensor views, chains memory tokens through ``load_view_tko`` and
``store_view_tko``, and checks the result against PyTorch.
"""

import os
import subprocess
import sys

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/flagtree_tileir_tutorial_cache/01")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.experimental.tle.language as tle  # noqa: E402
from triton.compiler.errors import CompilationError  # noqa: E402


@triton.jit
def _copy_with_token_kernel(x, y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_view = tle.make_view(
        base=x,
        shapes=[n_elements],
        strides=[1],
        tile_shape=[BLOCK_SIZE],
        tile_dim_map=[0],
    )
    y_view = tle.make_view(
        base=y,
        shapes=[n_elements],
        strides=[1],
        tile_shape=[BLOCK_SIZE],
        tile_dim_map=[0],
    )

    start_token = tle.create_mem_token()
    values, load_token = tle.load_view_tko(
        x_view,
        [pid],
        memToken=start_token,
        semantics="acquire",
        scope="device",
        has_result_token=True,
    )
    joined = tle.join_mem_tokens(start_token, load_token)
    store_token = tle.store_view_tko(
        y_view,
        values + 1.0,
        [pid],
        memToken=joined,
        semantics="release",
        scope="device",
        has_result_token=True,
    )
    # The second store writes the same tile after the first store token to
    # demonstrate token-ordered memory side effects.
    tle.store_view_tko(
        y_view,
        values + 2.0,
        [pid],
        memToken=store_token,
        semantics="weak",
        has_result_token=False,
    )


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("this tutorial requires a CUDA GPU")


def run_positive():
    os.environ["FLAGTREE_USE_TILEIR"] = "1"
    _require_cuda()
    n_elements = 1024
    block = 256
    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, block), )
    _copy_with_token_kernel[grid](x, y, n_elements, BLOCK_SIZE=block, num_warps=4)
    torch.testing.assert_close(y, x + 2)
    print("[OK] FLAGTREE_USE_TILEIR=1: TileIR TKO view-token kernel")


def run_expected_native_failure():
    os.environ.pop("FLAGTREE_USE_TILEIR", None)
    _require_cuda()
    n_elements = 1024
    block = 256
    x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    try:
        grid = (triton.cdiv(n_elements, block), )
        _copy_with_token_kernel[grid](x, y, n_elements, BLOCK_SIZE=block, num_warps=4)
    except CompilationError as exc:
        text = str(exc)
        expected_message = "TileIR path" in text or "FLAGTREE_USE_TILEIR" in text
        if not expected_message:
            raise
        print("[OK] FLAGTREE_USE_TILEIR=0: native NVIDIA path reports the expected TileIR-only API error")
        return
    raise AssertionError("expected native NVIDIA path to reject TKO view-token APIs")


def run_negative_subprocess():
    env = os.environ.copy()
    env.pop("FLAGTREE_USE_TILEIR", None)
    env["FLAGTREE_TILEIR_TUTORIAL_EXPECT_NATIVE_FAILURE"] = "1"
    subprocess.run([sys.executable, __file__], env=env, check=True)


def main():
    if os.environ.get("FLAGTREE_TILEIR_TUTORIAL_EXPECT_NATIVE_FAILURE") == "1":
        run_expected_native_failure()
        return

    run_positive()
    run_negative_subprocess()
    print(f"[OK] TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")


if __name__ == "__main__":
    main()
