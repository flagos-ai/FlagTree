import argparse

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

DEFAULT_SHAPES = [
    (64, 4096),
    (128, 4096),
    (256, 4096),
    (1024, 4096),
    (1024, 8192),
    (1024, 11008),
]

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

FP8_MAX = 448.0
MIN_AMAX = 1.0e-4
QUANT_STORAGE_DTYPE = torch.float16
QUANT_STORAGE_NAME = "fp16"
BENCH_COLUMNS = [
    "M",
    "N",
    "Group Size",
    "Groups/Program",
    "TLE-TileOps (ms)",
    "Triton-Quant (ms)",
    "Torch-Quant (ms)",
    "TLE vs Triton",
    "TLE vs Torch",
]
PYTEST_BENCH_COLUMNS = ["DType", *BENCH_COLUMNS]
_PYTEST_BENCH_ROWS = []


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


def _environment_skip_reason():
    if not _musa_available():
        return "MUSA device is not available"
    return None


_SKIP_REASON = _environment_skip_reason()
if _SKIP_REASON is not None and __name__ != "__main__":
    pytest.skip(_SKIP_REASON, allow_module_level=True)


# reference implementation from flaggems
@triton.jit
def _native_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    MIN_AMAX_VALUE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, GROUP_SIZE)
    cols = group * GROUP_SIZE + offsets

    x = tl.load(x_ptr + row * N + cols).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, MIN_AMAX_VALUE)
    scale = amax / FP8_MAX_VALUE
    q = tl.clamp(x / scale, -FP8_MAX_VALUE, FP8_MAX_VALUE)

    tl.store(q_ptr + row * N + cols, q)
    tl.store(scale_ptr + row * NUM_GROUPS + group, scale)


@triton.jit
def _tle_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    MIN_AMAX_VALUE: tl.constexpr,
):
    row = tl.program_id(0)
    group_block = tl.program_id(1)
    first_group = group_block * GROUPS_PER_PROGRAM

    offsets = tl.arange(0, BLOCK_N)
    cols = first_group * GROUP_SIZE + offsets
    mask = cols < N

    row_values = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    q_values = tl.full((BLOCK_N, ), 0.0, tl.float32)

    for local_group in tl.static_range(0, GROUPS_PER_PROGRAM):
        group_id = first_group + local_group
        x_tile = tle.extract_tile(row_values, index=local_group, tile_shape=(GROUP_SIZE, ))
        amax = tl.max(tl.abs(x_tile), axis=0)
        amax = tl.maximum(amax, MIN_AMAX_VALUE)
        scale = amax / FP8_MAX_VALUE
        q_tile = tl.clamp(x_tile / scale, -FP8_MAX_VALUE, FP8_MAX_VALUE)
        q_values = tle.insert_tile(q_values, q_tile, index=local_group)
        tl.store(
            scale_ptr + row * NUM_GROUPS + group_id,
            scale,
            mask=group_id < NUM_GROUPS,
        )

    tl.store(q_ptr + row * N + cols, q_values, mask=mask)


def _torch_quant(x, group_size):
    m, n = x.shape
    num_groups = n // group_size
    x_grouped = x.reshape(m, num_groups, group_size).to(torch.float32)
    amax = torch.amax(torch.abs(x_grouped), dim=-1)
    min_amax = torch.full_like(amax, MIN_AMAX)
    scale = torch.maximum(amax, min_amax) / FP8_MAX
    q = torch.clamp(x_grouped / scale[..., None], -FP8_MAX, FP8_MAX)
    q = q.to(QUANT_STORAGE_DTYPE).reshape(m, n)
    return q, scale


def _launch_native(x, q, scale, group_size):
    m, n = x.shape
    num_groups = n // group_size
    _native_quant_kernel[(m, num_groups)](
        x,
        q,
        scale,
        N=n,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        FP8_MAX_VALUE=FP8_MAX,
        MIN_AMAX_VALUE=MIN_AMAX,
        num_warps=4,
    )
    return q, scale


def _launch_tle(x, q, scale, group_size, groups_per_program):
    m, n = x.shape
    num_groups = n // group_size
    _tle_quant_kernel[(m, triton.cdiv(num_groups, groups_per_program))](
        x,
        q,
        scale,
        N=n,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        GROUPS_PER_PROGRAM=groups_per_program,
        BLOCK_N=group_size * groups_per_program,
        FP8_MAX_VALUE=FP8_MAX,
        MIN_AMAX_VALUE=MIN_AMAX,
        num_warps=4,
    )
    return q, scale


def _native_quant(x, group_size):
    m, n = x.shape
    num_groups = n // group_size
    q = torch.empty((m, n), device=x.device, dtype=QUANT_STORAGE_DTYPE)
    scale = torch.empty((m, num_groups), device=x.device, dtype=torch.float32)
    return _launch_native(x, q, scale, group_size)


def _tle_quant(x, group_size, groups_per_program):
    m, n = x.shape
    num_groups = n // group_size
    q = torch.empty((m, n), device=x.device, dtype=QUANT_STORAGE_DTYPE)
    scale = torch.empty((m, num_groups), device=x.device, dtype=torch.float32)
    return _launch_tle(x, q, scale, group_size, groups_per_program)


def _parse_shape(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be M,N")
    try:
        m, n = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers") from exc
    if m <= 0 or n <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return m, n


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark per-token group FP8 quant on MUSA.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-program", type=int, default=4, choices=(2, 4))
    return parser.parse_args()


def _require_musa():
    if not _musa_available():
        raise RuntimeError("MUSA device is not available")
    return torch.device("musa")


def _tolerances(dtype_name):
    if dtype_name == "fp32":
        return 1.0e-5, 1.0e-6
    return 1.0e-2, 1.0e-2


def _bench(fn, warmup, rep):
    return float(triton.testing.do_bench(
        fn,
        device_type="musa",
        warmup=warmup,
        rep=rep,
        return_mode="median",
    ))


def _format_table_value(column, value):
    if column.endswith("(ms)"):
        return f"{value:.5f}"
    if column.startswith("TLE vs "):
        return f"{value:.2f}"
    if isinstance(value, (float, int)):
        return f"{float(value):.1f}"
    return str(value)


def _speedup(reference_ms, tle_ms):
    if tle_ms == 0.0:
        return float("inf")
    return reference_ms / tle_ms


def _print_table(title, rows, columns):
    string_rows = [[_format_table_value(column, row[column]) for column in columns] for row in rows]
    widths = [max(len(column), *(len(row[i]) for row in string_rows)) for i, column in enumerate(columns)]
    index_width = max(1, len(str(len(rows) - 1)))

    print(f"{title}:")
    print(" " * (index_width + 2) + "  ".join(f"{column:>{widths[i]}}" for i, column in enumerate(columns)))
    for index, row in enumerate(string_rows):
        print(f"{index:<{index_width}}  " + "  ".join(f"{value:>{widths[i]}}" for i, value in enumerate(row)))


def _validate_shape(m, n, group_size):
    if n % group_size != 0:
        raise ValueError(f"shape ({m}, {n}) requires N divisible by group_size={group_size}")


def _assert_quant_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0)


def _run_shape(m, n, dtype_name, warmup, rep, group_size, groups_per_program, device):
    _validate_shape(m, n, group_size)
    dtype = DTYPES[dtype_name]
    rtol, atol = _tolerances(dtype_name)
    num_groups = n // group_size
    torch.manual_seed(42)

    x = (torch.randn((m, n), device=device, dtype=dtype) * 0.5).contiguous()
    expected_q, expected_scale = _torch_quant(x, group_size)
    native_q, native_scale = _native_quant(x, group_size)
    tle_q, tle_scale = _tle_quant(x, group_size, groups_per_program)
    torch.musa.synchronize()

    _assert_quant_close(native_q, expected_q)
    _assert_quant_close(tle_q, expected_q)
    torch.testing.assert_close(native_scale, expected_scale, rtol=rtol, atol=atol)
    torch.testing.assert_close(tle_scale, expected_scale, rtol=rtol, atol=atol)

    native_q_out = torch.empty((m, n), device=device, dtype=QUANT_STORAGE_DTYPE)
    native_scale_out = torch.empty((m, num_groups), device=device, dtype=torch.float32)
    tle_q_out = torch.empty((m, n), device=device, dtype=QUANT_STORAGE_DTYPE)
    tle_scale_out = torch.empty((m, num_groups), device=device, dtype=torch.float32)
    _launch_native(x, native_q_out, native_scale_out, group_size)
    _launch_tle(x, tle_q_out, tle_scale_out, group_size, groups_per_program)
    torch.musa.synchronize()

    torch_ms = _bench(lambda: _torch_quant(x, group_size), warmup, rep)
    native_ms = _bench(
        lambda: _launch_native(x, native_q_out, native_scale_out, group_size),
        warmup,
        rep,
    )
    tle_ms = _bench(
        lambda: _launch_tle(x, tle_q_out, tle_scale_out, group_size, groups_per_program),
        warmup,
        rep,
    )

    return {
        "DType": dtype_name,
        "M": m,
        "N": n,
        "Group Size": group_size,
        "Groups/Program": groups_per_program,
        "TLE-TileOps (ms)": tle_ms,
        "Triton-Quant (ms)": native_ms,
        "Torch-Quant (ms)": torch_ms,
        "TLE vs Triton": _speedup(native_ms, tle_ms),
        "TLE vs Torch": _speedup(torch_ms, tle_ms),
    }


@pytest.fixture(scope="session", autouse=True)
def _print_pytest_bench_table_once():
    _PYTEST_BENCH_ROWS.clear()
    yield
    if not _PYTEST_BENCH_ROWS:
        return
    _print_table(
        f"tle-per-token-group-quant-fp8-vs-triton-vs-torch quant_storage={QUANT_STORAGE_NAME}",
        _PYTEST_BENCH_ROWS,
        PYTEST_BENCH_COLUMNS,
    )


@pytest.mark.parametrize("shape", DEFAULT_SHAPES)
@pytest.mark.parametrize("dtype_name", tuple(DTYPES))
@pytest.mark.parametrize("groups_per_program", (2, 4))
def test_per_token_group_quant_fp8_correctness_and_bench(shape, dtype_name, groups_per_program):
    row = _run_shape(
        shape[0],
        shape[1],
        dtype_name,
        warmup=1,
        rep=1,
        group_size=128,
        groups_per_program=groups_per_program,
        device=torch.device("musa"),
    )
    _PYTEST_BENCH_ROWS.append(row)


def main():
    args = _parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    device = _require_musa()
    shapes = args.shape if args.shape else DEFAULT_SHAPES
    rows = [
        _run_shape(
            m,
            n,
            args.dtype,
            args.warmup,
            args.rep,
            args.group_size,
            args.groups_per_program,
            device,
        ) for m, n in shapes
    ]
    _print_table(
        ("tle-per-token-group-quant-fp8-vs-triton-vs-torch "
         f"dtype={args.dtype} quant_storage={QUANT_STORAGE_NAME}"),
        rows,
        BENCH_COLUMNS,
    )


if __name__ == "__main__":
    main()
