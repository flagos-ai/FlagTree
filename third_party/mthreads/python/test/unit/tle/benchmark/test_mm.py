"""Benchmark native Triton MM, optimized TLE-WS MM, and Torch on MUSA.

Pytest:
    python -m pytest -q -s third_party/mthreads/python/test/unit/tle/benchmark/test_mm.py

Command line:
    python third_party/mthreads/python/test/unit/tle/benchmark/test_mm.py \
        --shape 1024 1024 1024 --stages 1 2 3
"""

import argparse
import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("MUSA_LAUNCH_BLOCKING", "0")

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
PANEL_WIDTH = 4
NUM_WARPS = 16
RTOL = 1.25e-1
ATOL = 1.25e-1

DEFAULT_SHAPES = (
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (8192, 7168, 16384),
)
PYTEST_SHAPES = ((1024, 1024, 1024), )
PYTEST_STAGES = (1, 2, 3)
PYTEST_WARMUP = 5
PYTEST_REP = 20
BENCH_COLUMNS = (
    "M",
    "N",
    "K",
    "Stages",
    "Torch (ms)",
    "Triton (ms)",
    "TLE-WS (ms)",
    "Triton vs Torch",
    "TLE-WS vs Torch",
    "TLE-WS vs Triton",
)
_PYTEST_ROWS = []


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


if not _musa_available() and __name__ != "__main__":
    pytest.skip("MUSA device is not available", allow_module_level=True)


@triton.jit
def _native_rasterization_2d_column(
    block_idx,
    grid_x,
    grid_y,
    panel_width: tl.constexpr,
):
    panel_size = panel_width * grid_y
    residual_panel_width = grid_x % panel_width
    full_panels_size = grid_x // panel_width * panel_width * grid_y
    panel_idx = block_idx // panel_size
    panel_offset = block_idx % panel_size

    width = tl.where(
        block_idx >= full_panels_size,
        residual_panel_width,
        panel_width,
    )
    row_idx = panel_offset // width
    mini_x = panel_offset % width
    mini_x = tl.where(row_idx % 2 == 1, width - 1 - mini_x, mini_x)
    row_idx = tl.where(panel_idx % 2 == 1, grid_y - 1 - row_idx, row_idx)
    col_idx = panel_idx * panel_width + mini_x
    return col_idx, row_idx


@triton.jit
def native_triton_mm_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
    grid_y = tl.cdiv(M, block_m)
    grid_x = tl.cdiv(N, block_n)
    block_idx = raw_bx + raw_by * grid_x
    pid_n, pid_m = _native_rasterization_2d_column(
        block_idx,
        grid_x,
        grid_y,
        panel_width,
    )

    offset_a_m = pid_m * block_m
    offset_b_n = pid_n * block_n
    offset_m = offset_a_m + tl.arange(0, block_m)
    offset_n = offset_b_n + tl.arange(0, block_n)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in tl.range(0, K, block_k, num_stages=pipeline_stages):
        a = tl.load_tensor_descriptor(a_desc, [offset_a_m, k])
        b = tl.load_tensor_descriptor(b_desc, [k, offset_b_n])
        acc = tl.dot(a, b, acc=acc)

    c_block_ptr = c_ptr + N * offset_m[:, None] + offset_n[None, :]
    tl.store(c_block_ptr, acc.to(tl.float16), mask=mask)


@triton.jit
def _tle_ws_rasterization_2d_column(
    block_idx,
    grid_x,
    grid_y,
    panel_width: tl.constexpr,
    full_panel: tl.constexpr,
):
    panel_size = panel_width * grid_y
    panel_idx = block_idx // panel_size
    panel_offset = block_idx % panel_size

    if full_panel:
        width = panel_width
    else:
        residual_panel_width = grid_x % panel_width
        full_panels_size = grid_x // panel_width * panel_width * grid_y
        width = tl.where(
            block_idx >= full_panels_size,
            residual_panel_width,
            panel_width,
        )
    row_idx = panel_offset // width
    mini_x = panel_offset % width
    mini_x = tl.where(row_idx % 2 == 1, width - 1 - mini_x, mini_x)
    row_idx = tl.where(panel_idx % 2 == 1, grid_y - 1 - row_idx, row_idx)
    col_idx = panel_idx * panel_width + mini_x
    return col_idx, row_idx


@triton.jit
def _tle_ws_mm_consumer(
    a_reader,
    b_reader,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offset_m = pid_m * block_m + tl.arange(0, block_m)
    offset_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in tl.range(
            0,
            k_tiles,
            num_stages=1,
            loop_unroll_factor=k_tiles,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    c_block_ptr = c_ptr + N * offset_m[:, None] + offset_n[None, :]
    tl.store(c_block_ptr, acc.to(tl.float16), mask=mask)


@triton.jit
def _tle_ws_mm_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    K: tl.constexpr,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offset_a_m = pid_m * block_m
    offset_b_n = pid_n * block_n
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    for k_iter in tl.range(
            0,
            k_tiles,
            num_stages=1,
            loop_unroll_factor=k_tiles,
    ):
        offset_k = k_iter * block_k
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            (block_m, block_k),
            (offset_a_m, offset_k),
        )
        tle.gpu.copy(
            b_desc,
            b_slot.b,
            (block_k, block_n),
            (offset_k, offset_b_n),
        )
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def tle_ws_mm_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
    full_panel: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
    grid_x = tl.num_programs(axis=0)
    grid_y = tl.num_programs(axis=1)
    block_idx = raw_bx + raw_by * grid_x
    pid_n, pid_m = _tle_ws_rasterization_2d_column(
        block_idx,
        grid_x,
        grid_y,
        panel_width,
        full_panel,
    )

    a_smem = tle.gpu.alloc(
        (pipeline_stages, block_m, block_k),
        dtype=tl.float16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (pipeline_stages, block_k, block_n),
        dtype=tl.float16,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="mm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="mm_b", b=b_smem)

    tle.gpu.warp_specialize(
        [
            (
                _tle_ws_mm_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    M,
                    N,
                    K,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
            (
                _tle_ws_mm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    K,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def _validate_case(m, n, k, stages, block_m, block_n, block_k, panel_width):
    if min(m, n, k) <= 0:
        raise ValueError("M, N, and K must be positive")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("M/N/K must be divisible by BLOCK_M/BLOCK_N/BLOCK_K")
    if stages not in (1, 2, 3):
        raise ValueError("the optimized mthreads TLE-WS MM supports stages 1, 2, or 3")
    if panel_width <= 0:
        raise ValueError("panel_width must be positive")


def _bench(fn, warmup, rep):
    return float(triton.testing.do_bench(
        fn,
        device_type="musa",
        warmup=warmup,
        rep=rep,
        return_mode="median",
    ))


def _speedup(reference_ms, candidate_ms):
    return reference_ms / candidate_ms if candidate_ms else float("inf")


def _run_case(
    m,
    n,
    k,
    stages,
    warmup,
    rep,
    block_m=BLOCK_M,
    block_n=BLOCK_N,
    block_k=BLOCK_K,
    panel_width=PANEL_WIDTH,
    num_warps=NUM_WARPS,
    seed=42,
):
    _validate_case(m, n, k, stages, block_m, block_n, block_k, panel_width)
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float16, device="musa")
    b = torch.randn((k, n), dtype=torch.float16, device="musa")
    desc_a = TensorDescriptor.from_tensor(a, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, block_n])
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m), 1)
    full_panel = grid[0] % panel_width == 0

    def launch_torch():
        return torch.mm(a, b)

    def launch_triton():
        output = torch.empty((m, n), dtype=torch.float16, device="musa")
        native_triton_mm_kernel[grid](
            desc_a,
            desc_b,
            output,
            m,
            n,
            k,
            block_m,
            block_n,
            block_k,
            panel_width,
            stages,
            num_warps=num_warps,
            num_stages=stages,
        )
        return output

    def launch_tle_ws():
        output = torch.empty((m, n), dtype=torch.float16, device="musa")
        tle_ws_mm_kernel[grid](
            desc_a,
            desc_b,
            output,
            m,
            n,
            k,
            block_m,
            block_n,
            block_k,
            panel_width,
            stages,
            full_panel,
            num_warps=num_warps,
            num_stages=stages,
        )
        return output

    torch_output = launch_torch()
    triton_output = launch_triton()
    tle_ws_output = launch_tle_ws()
    torch.musa.synchronize()

    reference = torch.mm(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(torch_output.float(), reference, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(triton_output.float(), reference, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(tle_ws_output.float(), reference, rtol=RTOL, atol=ATOL)
    torch.musa.synchronize()

    torch_ms = _bench(launch_torch, warmup, rep)
    triton_ms = _bench(launch_triton, warmup, rep)
    tle_ws_ms = _bench(launch_tle_ws, warmup, rep)
    return {
        "M": m,
        "N": n,
        "K": k,
        "Stages": stages,
        "Torch (ms)": torch_ms,
        "Triton (ms)": triton_ms,
        "TLE-WS (ms)": tle_ws_ms,
        "Triton vs Torch": _speedup(torch_ms, triton_ms),
        "TLE-WS vs Torch": _speedup(torch_ms, tle_ws_ms),
        "TLE-WS vs Triton": _speedup(triton_ms, tle_ws_ms),
    }


def _format_value(column, value):
    if column.endswith("(ms)"):
        return f"{value:.6f}"
    if " vs " in column:
        return f"{value:.4f}x"
    return str(value)


def _print_table(title, rows):
    string_rows = [[_format_value(column, row[column]) for column in BENCH_COLUMNS] for row in rows]
    widths = [max(len(column), *(len(row[index]) for row in string_rows)) for index, column in enumerate(BENCH_COLUMNS)]
    print(title)
    print("  ".join(f"{column:>{widths[index]}}" for index, column in enumerate(BENCH_COLUMNS)))
    for row in string_rows:
        print("  ".join(f"{value:>{widths[index]}}" for index, value in enumerate(row)))


@pytest.fixture(scope="session", autouse=True)
def _print_pytest_results_once():
    _PYTEST_ROWS.clear()
    yield
    if _PYTEST_ROWS:
        _print_table("mthreads MM benchmark (correctness passed)", _PYTEST_ROWS)


@pytest.mark.parametrize("shape", PYTEST_SHAPES)
@pytest.mark.parametrize("stages", PYTEST_STAGES, ids=lambda value: f"stage{value}")
def test_mm_correctness_and_benchmark(shape, stages):
    _PYTEST_ROWS.append(_run_case(
        *shape,
        stages=stages,
        warmup=PYTEST_WARMUP,
        rep=PYTEST_REP,
    ))


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark native Triton MM, optimized TLE-WS MM, and Torch on MUSA.")
    parser.add_argument(
        "--shape",
        dest="shapes",
        type=int,
        nargs=3,
        action="append",
        metavar=("M", "N", "K"),
        help="matrix shape; repeat to benchmark multiple shapes",
    )
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--warmup", type=int, default=25, help="do_bench warmup in milliseconds")
    parser.add_argument("--rep", type=int, default=100, help="do_bench measurement time in milliseconds")
    parser.add_argument("--block-m", type=int, default=BLOCK_M)
    parser.add_argument("--block-n", type=int, default=BLOCK_N)
    parser.add_argument("--block-k", type=int, default=BLOCK_K)
    parser.add_argument("--panel-width", type=int, default=PANEL_WIDTH)
    parser.add_argument("--num-warps", type=int, default=NUM_WARPS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("warmup and rep must be positive")
    if not _musa_available():
        raise RuntimeError("MUSA device is not available")
    shapes = [tuple(shape) for shape in args.shapes] if args.shapes else DEFAULT_SHAPES
    rows = [
        _run_case(
            *shape,
            stages=stages,
            warmup=args.warmup,
            rep=args.rep,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            panel_width=args.panel_width,
            num_warps=args.num_warps,
            seed=args.seed,
        ) for shape in shapes for stages in args.stages
    ]
    _print_table(
        "mthreads MM benchmark (correctness passed; "
        f"llc_opt={os.environ.get('TRITON_MUSA_ENABLE_LLC_OPT', '0')})",
        rows,
    )


if __name__ == "__main__":
    main()
