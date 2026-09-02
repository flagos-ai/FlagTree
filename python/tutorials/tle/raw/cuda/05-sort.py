import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle_gpu
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

DEVICE = triton.runtime.driver.active.get_active_torch_device()
_TLE_RAW_SOURCE = Path(__file__).parent / "05-sort.cu"


def is_cuda():
    return (torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda")


def _configure_tle_raw_cccl_include():
    if not is_cuda():
        return
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    cccl_include = cuda_home / "include" / "cccl"
    if not (cccl_include / "cub").is_dir():
        return
    try:
        from triton import knobs
    except (AttributeError, ImportError):
        return
    try:
        flags = knobs.nvidia.tle_raw_clang_flags or ""
    except AttributeError:
        return

    required_flags = [f"-I{cccl_include}"]
    try:
        cuda_version = json.loads((cuda_home / "version.json").read_text())
        cuda_major = int(cuda_version["cuda"]["version"].split(".", 1)[0])
    except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
    ):
        cuda_major = 0
    if cuda_major >= 13:
        # Clang's CUDA wrapper suppresses the header defining this CUDA 13
        # macro but still includes math_functions.hpp.
        required_flags.append("-D_NV_RSQRT_SPECIFIER=")

    existing_flags = flags.split()
    missing_flags = [flag for flag in required_flags if flag not in existing_flags]
    if missing_flags:
        knobs.nvidia.tle_raw_clang_flags = " ".join((flags, *missing_flags)).strip()


_configure_tle_raw_cccl_include()


@dialect(
    name="cuda",
    file=_TLE_RAW_SOURCE,
    extern_func_name="RadixHistogramDigits8x2048",
    deferred=True,
)
def _radix_histogram_digits_8x2048_raw(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=_TLE_RAW_SOURCE,
    extern_func_name="RadixRank8x2048Precomputed",
    deferred=True,
)
def _radix_rank_8x2048_precomputed_raw(*args, **kwargs):
    ...


@triton.jit
def _float16_to_ordered_uint(value, descending: tl.constexpr = False):
    bits = value.to(tl.uint16, bitcast=True)
    signed_bits = value.to(tl.int16, bitcast=True)

    sign_mask = tl.full((), 0x8000, tl.uint16)
    shift = tl.full((), 15, tl.int16)
    mask = sign_mask | (signed_bits >> shift).to(tl.uint16, bitcast=True)

    if descending:
        return bits ^ (~mask)
    return bits ^ mask


@triton.jit
def _radix_tile_histogram_kernel_raw(
    input_ptr,
    counts_ptr,
    rows,
    n,
    descending: tl.constexpr,
    BIT_OFFSET: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    tl.static_assert(TILE_N == 2048)
    program = tl.program_id(0)
    tiles = tl.cdiv(n, TILE_N)
    row = program // tiles
    tile = program - row * tiles
    columns = tile * TILE_N + tl.arange(0, TILE_N)
    mask = columns < n
    digit_mask: tl.constexpr = NUM_BINS - 1
    values = tl.load(input_ptr + row * n + columns, mask=mask)
    keys = _float16_to_ordered_uint(values, descending)
    digits = ((keys >> BIT_OFFSET) & digit_mask).to(tl.uint16)

    digits_smem = tle_gpu.alloc(
        shape=[TILE_N],
        dtype=tl.uint16,
        layout=None,
        scope=tle_gpu.smem,
        nv_mma_shared_layout=False,
    )
    counts_smem = tle_gpu.alloc(
        shape=[NUM_BINS],
        dtype=tl.int32,
        layout=None,
        scope=tle_gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle_gpu.local_ptr(digits_smem, (columns, )), digits)
    valid_count = tl.minimum(TILE_N, n - tile * TILE_N)
    counts_smem = tle_raw.call_smem(
        _radix_histogram_digits_8x2048_raw,
        [digits_smem, counts_smem, valid_count],
        output_indices=[1],
    )

    bins = tl.arange(0, NUM_BINS)
    counts = tl.load(tle_gpu.local_ptr(counts_smem, (bins, )))
    tl.store(counts_ptr + (row * tiles + tile) * NUM_BINS + bins, counts)


@triton.jit
def _radix_tile_offsets_kernel(
    counts_ptr,
    offsets_ptr,
    tiles,
    NUM_BINS: tl.constexpr,
):
    row = tl.program_id(0)
    bins = tl.arange(0, NUM_BINS)
    row_base = row * tiles * NUM_BINS

    bin_totals = tl.zeros((NUM_BINS, ), dtype=tl.int32)
    for tile in range(0, tiles):
        counts = tl.load(counts_ptr + row_base + tile * NUM_BINS + bins)
        bin_totals += counts

    bin_bases = tl.cumsum(bin_totals, axis=0) - bin_totals
    running = bin_bases
    for tile in range(0, tiles):
        offset = row_base + tile * NUM_BINS + bins
        counts = tl.load(counts_ptr + offset)
        tl.store(offsets_ptr + offset, running)
        running += counts


@triton.jit
def _sweep_cub_local_rank_precomputed(
    arr_ptr,
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr32,
    associate_out_ptr64,
    tile_offsets_ptr,
    bit_offset,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    NUM_BINS: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
    final_pass,
):
    tl.static_assert(TILE_N == 2048)
    tl.static_assert(NUM_BINS == (1 << k_bits))
    tl.static_assert(k_bits == 8)

    pid = tl.program_id(0)
    pid_m = pid // OUT_N
    pid_n = pid - pid_m * OUT_N
    cols = tl.arange(0, TILE_N)
    n_offsets = pid_n * TILE_N + cols
    mask = n_offsets < N
    digit_mask: tl.constexpr = NUM_BINS - 1
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = _float16_to_ordered_uint(arr, descending)
    digits = ((arr_u >> bit_offset) & digit_mask).to(tl.uint16)
    digits = tl.where(mask, digits, digit_mask).to(tl.uint16)

    digits_smem = tle_gpu.alloc(
        shape=[TILE_N],
        dtype=tl.uint16,
        layout=None,
        scope=tle_gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle_gpu.local_ptr(digits_smem, (cols, )), digits)
    valid_count = tl.minimum(TILE_N, N - pid_n * TILE_N)
    tle_raw.call_smem(
        _radix_rank_8x2048_precomputed_raw,
        [
            digits_smem,
            arr_ptr,
            associate_arr_ptr,
            out_ptr,
            associate_out_ptr32,
            associate_out_ptr64,
            tile_offsets_ptr,
            pid_m,
            pid_n,
            N,
            OUT_N,
            valid_count,
            final_pass,
        ],
        output_indices=[],
    )


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"radix_sort only supports float16/bfloat16, got {dtype}")
    num_bits = 16
    num_warps = 8

    rows = arr.numel() // n
    tile_size = 2048
    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    tiles = triton.cdiv(n, tile_size)
    tile_grid = (rows * tiles, )
    sweep_grid = (rows * tiles, )

    with torch.cuda.device(arr.device):
        arr_in = torch.clone(arr)
        arr_out = torch.empty_like(arr)
        temporary_indices = torch.empty_like(arr, dtype=torch.int32)
        final_indices = torch.empty_like(arr, dtype=torch.int64)
        tile_counts = torch.empty((rows, tiles, num_bins), device=arr.device, dtype=torch.int32)
        tile_offsets = torch.empty_like(tile_counts)

        for pass_id in range(n_passes):
            bit_offset = pass_id * k_bits
            _radix_tile_histogram_kernel_raw[tile_grid](
                arr_in,
                tile_counts,
                rows,
                n,
                descending=descending,
                BIT_OFFSET=bit_offset,
                TILE_N=tile_size,
                NUM_BINS=num_bins,
                num_warps=num_warps,
            )
            _radix_tile_offsets_kernel[(rows, )](
                tile_counts,
                tile_offsets,
                tiles,
                NUM_BINS=num_bins,
                num_warps=num_warps,
            )
            _sweep_cub_local_rank_precomputed[sweep_grid](
                arr_in,
                temporary_indices,
                arr_out,
                temporary_indices,
                final_indices,
                tile_offsets,
                bit_offset,
                n,
                tiles,
                tile_size,
                num_bins,
                k_bits,
                int(descending),
                int(pass_id == n_passes - 1),
                num_warps=num_warps,
            )
            arr_in, arr_out = arr_out, arr_in

    return arr_in, final_indices


def sort(inp, dim=-1, descending=False):
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    _ = stable
    if inp.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"sort currently only supports float16/bfloat16, got {inp.dtype}")

    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0:
        return torch.empty_like(inp), torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim += inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    out, out_index = radix_sort(inp, descending=descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index


if __name__ == "__main__":
    torch.manual_seed(0)

    # test
    for dtype in (torch.float16, torch.bfloat16):
        for descending in (False, True):
            x = torch.randn((1024, 65536), device=DEVICE, dtype=dtype)

            ref_values, ref_indices = torch.sort(
                x,
                dim=-1,
                descending=descending,
                stable=True,
            )
            values, indices = sort(
                x,
                dim=-1,
                descending=descending,
            )

            torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
            torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)

    # perf
    x = torch.randn((1024, 65536), device=DEVICE, dtype=torch.float16)
    torch_ms = triton.testing.do_bench(lambda: torch.sort(x, dim=-1, descending=False, stable=True))
    triton_ms = triton.testing.do_bench(lambda: sort(x, dim=-1, descending=False))
    print(f"Torch: {torch_ms} \nTriton: {triton_ms} \nSpeedup: {torch_ms/triton_ms}x")
