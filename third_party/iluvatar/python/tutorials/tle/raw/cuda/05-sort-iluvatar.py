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
_TLE_RAW_SOURCE = Path(__file__).parent / "05-sort-iluvatar.cu"
# ====================== [OPT] 0 ======================
BLOCK_THREADS = 256
NUM_WARPS = BLOCK_THREADS // triton.runtime.driver.active.get_current_target().warp_size


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
    name="corex",
    file=_TLE_RAW_SOURCE,
    extern_func_name="RadixHistogramDigits8x2048",
    deferred=True,
)
def _radix_histogram_digits_8x2048_raw(*args, **kwargs):
    ...


@dialect(
    name="corex",
    file=_TLE_RAW_SOURCE,
    extern_func_name="RadixRank8x2048Precomputed",
    deferred=True,
)
def _radix_rank_8x2048_precomputed_raw(*args, **kwargs):
    ...


# ====================== [OPT] 2 ======================
# Single-tile entry: one CTA sorts a whole row with a full 16-bit key.
# items_per_thread (2 or 8) selects the CUB instantiation inside the .cu.
@dialect(
    name="corex",
    file=_TLE_RAW_SOURCE,
    extern_func_name="RadixSortFullTile",
    deferred=True,
)
def _radix_sort_full_tile_raw(*args, **kwargs):
    ...


# ====================== [OPT] 3 ======================
# Several short rows packed into one 512-wide tile.
@dialect(
    name="corex",
    file=_TLE_RAW_SOURCE,
    extern_func_name="RadixSortRowsPerTile512",
    deferred=True,
)
def _radix_sort_rows_per_tile_512_raw(*args, **kwargs):
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
    valid_n,
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
    local_cols = tl.arange(0, TILE_N)
    columns = tile * TILE_N + local_cols
    # ====================== [OPT] 1 ======================
    # Padding lanes are excluded here, so they never reach the histogram.
    # Otherwise every padded element carries the same digit and all of them
    # atomicAdd the same bin, which serializes the whole tile.
    mask = columns < valid_n
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
    # ====================== [OPT] 1 ======================
    # digits_smem is TILE_N wide, so it is indexed by the in-tile column.
    # `columns` is the global column and runs past the buffer once tile > 0.
    tl.store(tle_gpu.local_ptr(digits_smem, (local_cols, )), digits)
    valid_count = tl.maximum(tl.minimum(TILE_N, valid_n - tile * TILE_N), 0)
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
    VALID_N,
    OUT_N,
    TILE_N: tl.constexpr,
    NUM_BINS: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
    first_pass,
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
    # ====================== [OPT] 1 ======================
    # Padding lanes get digit 255 below and sort to the tail of the tile, so
    # `local < valid_count` in the .cu selects exactly the real elements.
    mask = n_offsets < VALID_N
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
    # ====================== [FIX] 1 ======================
    tl.debug_barrier()
    # ====================== [OPT] 1 ======================
    valid_count = tl.maximum(tl.minimum(TILE_N, VALID_N - pid_n * TILE_N), 0)
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
            first_pass,
            final_pass,
        ],
        output_indices=[],
    )


# ====================== [OPT] 2 ======================
# One CTA per row, used when the whole row fits in a single tile. The rank
# inside the tile is already the final rank, so there is no tile histogram,
# no global bin offsets and no second pass over global memory.
@triton.jit
def _sort_single_tile_kernel_raw(
    arr_ptr,
    out_ptr,
    out_indices_ptr,
    N,
    TILE_N: tl.constexpr,
    descending: tl.constexpr,
):
    tl.static_assert((TILE_N == 512) or (TILE_N == 2048))
    row = tl.program_id(0)
    cols = tl.arange(0, TILE_N)
    mask = cols < N
    arr = tl.load(arr_ptr + row * N + cols, mask=mask)
    # Padding lanes take the largest key so they sort to the tail. A real
    # element with the same key keeps a lower column, and the sort breaks
    # ties by column, so the real one still lands before the padding.
    keys = _float16_to_ordered_uint(arr, descending)
    keys = tl.where(mask, keys, 0xFFFF).to(tl.uint16)

    keys_smem = tle_gpu.alloc(
        shape=[TILE_N],
        dtype=tl.uint16,
        layout=None,
        scope=tle_gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle_gpu.local_ptr(keys_smem, (cols, )), keys)
    # ====================== [FIX] 1 ======================
    # Required: the .cu reads the tile in CUB blocked order, which is not the
    # order Triton stored it in. Even at TILE_N 512, where sizePerThread=2
    # lines up with items_per_thread=2, the layout drops to one element per
    # thread when N is not a multiple of 2, and the pairs stop matching.
    tl.debug_barrier()
    # ====================== [OPT] 2 ======================
    # 512-wide tile: 256 threads * 2 items. 2048-wide tile: 256 * 8.
    items_per_thread: tl.constexpr = 2 if TILE_N == 512 else 8
    tle_raw.call_smem(
        _radix_sort_full_tile_raw,
        [keys_smem, arr_ptr, out_ptr, out_indices_ptr, row, N, N, items_per_thread],
        output_indices=[],
    )


# ====================== [OPT] 3 ======================
# One CTA sorts TILE_N // ROW_LEN whole rows at once. Only valid when ROW_LEN
# divides TILE_N, i.e. a power of two, so the rows are contiguous in the tile
# and the row index is a plain shift of the in-tile column.
@triton.jit
def _sort_rows_per_tile_kernel_raw(
    arr_ptr,
    out_ptr,
    out_indices_ptr,
    rows,
    ROW_LEN: tl.constexpr,
    ROW_BITS: tl.constexpr,
    TILE_N: tl.constexpr,
    descending: tl.constexpr,
):
    tl.static_assert(TILE_N == 512)
    tl.static_assert(ROW_LEN == (1 << ROW_BITS))
    ROWS_PER_CTA: tl.constexpr = TILE_N // ROW_LEN
    tl.static_assert(ROWS_PER_CTA > 1)
    # Bits the row index occupies above the 16-bit key. Sorting fewer than
    # these would let rows mix; sorting more only costs extra CUB passes.
    ROW_INDEX_BITS: tl.constexpr = 9 - ROW_BITS

    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_CTA
    cols = tl.arange(0, TILE_N)
    valid_count = tl.maximum(tl.minimum(TILE_N, (rows - base_row) * ROW_LEN), 0)
    mask = cols < valid_count
    arr = tl.load(arr_ptr + base_row * ROW_LEN + cols, mask=mask)

    # Row index in the high bits keeps rows from mixing: the sort orders by
    # row first, then by value inside the row.
    keys = _float16_to_ordered_uint(arr, descending).to(tl.uint32)
    keys = ((cols >> ROW_BITS).to(tl.uint32) << 16) | keys
    keys = tl.where(mask, keys, 0xFFFFFFFF).to(tl.uint32)

    keys_smem = tle_gpu.alloc(
        shape=[TILE_N],
        dtype=tl.uint32,
        layout=None,
        scope=tle_gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle_gpu.local_ptr(keys_smem, (cols, )), keys)
    # ====================== [FIX] 1 ======================
    tl.debug_barrier()
    tle_raw.call_smem(
        _radix_sort_rows_per_tile_512_raw,
        [
            keys_smem,
            arr_ptr,
            out_ptr,
            out_indices_ptr,
            base_row,
            ROW_LEN,
            valid_count,
            16 + ROW_INDEX_BITS,
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
    num_warps = NUM_WARPS

    rows = arr.numel() // n
    # ====================== [OPT] 3 ======================
    # Short power-of-two rows: pack several rows per CTA. Below 512 the tile
    # is mostly padding otherwise, and the per-CTA cost does not shrink with
    # the element count.
    if n <= 256 and (n & (n - 1)) == 0:
        tile_n = 512
        rows_per_cta = tile_n // n
        with torch.cuda.device(arr.device):
            arr_2d = arr.reshape(rows, n)
            out = torch.empty_like(arr_2d)
            out_indices = torch.empty((rows, n), device=arr.device, dtype=torch.int64)
            _sort_rows_per_tile_kernel_raw[(triton.cdiv(rows, rows_per_cta), )](
                arr_2d,
                out,
                out_indices,
                rows,
                ROW_LEN=n,
                ROW_BITS=n.bit_length() - 1,
                TILE_N=tile_n,
                descending=descending,
                num_warps=num_warps,
            )
        return out.reshape(arr.shape), out_indices.reshape(arr.shape)

    # ====================== [OPT] 2 ======================
    # A row that fits in one tile is sorted by a single kernel. No padded
    # buffer, no clone, no intermediate int32 indices.
    if n <= 2048:
        single_tile_n = 512 if n <= 512 else 2048
        with torch.cuda.device(arr.device):
            arr_2d = arr.reshape(rows, n)
            out = torch.empty_like(arr_2d)
            out_indices = torch.empty((rows, n), device=arr.device, dtype=torch.int64)
            _sort_single_tile_kernel_raw[(rows, )](
                arr_2d,
                out,
                out_indices,
                n,
                TILE_N=single_tile_n,
                descending=descending,
                num_warps=num_warps,
            )
        return out.reshape(arr.shape), out_indices.reshape(arr.shape)

    tile_size = 2048

    # n % 2048 != 0, need padding
    padding = n % tile_size != 0
    # ====================== [OPT] 1 ======================
    # Row length the kernels treat as real. Padded columns are masked out
    # instead of being sorted along with the data.
    valid_n = n
    if padding:
        original_shape = arr.shape
        original_n = n
        n = triton.cdiv(n, tile_size) * tile_size

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    tiles = triton.cdiv(n, tile_size)
    tile_grid = (rows * tiles, )
    sweep_grid = (rows * tiles, )

    with torch.cuda.device(arr.device):
        if not padding:
            arr_in = torch.clone(arr)
        else:
            arr_in = torch.empty((rows, n), device=arr.device, dtype=arr.dtype)
            arr_in[:, :original_n] = arr.reshape(rows, original_n)
            # ====================== [OPT] 1 ======================
            # The tail used to be filled with +-inf so it would sort last.
            # valid_n now masks it out, so the fill is dead work.

        arr_out = torch.empty_like(arr_in)
        temporary_indices_a = torch.empty_like(arr_in, dtype=torch.int32)
        temporary_indices_b = (torch.empty_like(arr_in, dtype=torch.int32) if n_passes > 2 else temporary_indices_a)
        final_indices = torch.empty_like(arr_in, dtype=torch.int64)
        tile_counts = torch.empty((rows, tiles, num_bins), device=arr.device, dtype=torch.int32)
        tile_offsets = torch.empty_like(tile_counts)

        for pass_id in range(n_passes):
            bit_offset = pass_id * k_bits
            if pass_id == 0:
                compact_indices_in = temporary_indices_a
                compact_indices_out = temporary_indices_a
            elif pass_id < n_passes - 1:
                compact_indices_out = (temporary_indices_b
                                       if compact_indices_in is temporary_indices_a else temporary_indices_a)

            _radix_tile_histogram_kernel_raw[tile_grid](
                arr_in,
                tile_counts,
                rows,
                n,
                valid_n,
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
                compact_indices_in,
                arr_out,
                compact_indices_out,
                final_indices,
                tile_offsets,
                bit_offset,
                n,
                valid_n,
                tiles,
                tile_size,
                num_bins,
                k_bits,
                int(descending),
                int(pass_id == 0),
                int(pass_id == n_passes - 1),
                num_warps=num_warps,
            )

            compact_indices_in = compact_indices_out
            arr_in, arr_out = arr_out, arr_in

    if not padding:
        return arr_in, final_indices
    else:
        values = arr_in[:, :original_n].reshape(original_shape)
        indices = final_indices[:, :original_n].reshape(original_shape)
        return values, indices


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

    out, out_index = radix_sort(inp, k_bits=8, descending=descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index


if __name__ == "__main__":
    torch.manual_seed(0)

    SHAPES = [
        (1024, 1),
        (1024, 2),
        (1024, 64),
        (1024, 4095),
        (1024, 4096),
        (1024, 4097),
        (1024, 65536),
        (16, 131072),
        (8, 262144),
    ]

    # test
    for shape in SHAPES:
        for dtype in (torch.float16, torch.bfloat16):
            for descending in (False, True):
                x = torch.randn(shape, device=DEVICE, dtype=dtype)

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
    print("[√] accurency passed!")

    # perf
    # x = torch.randn((1024, 65536), device=DEVICE, dtype=torch.float16)
    # x = torch.randn((1024, 64), device=DEVICE, dtype=torch.float16)
    for shape in SHAPES:
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(shape, device=DEVICE, dtype=dtype)
            torch_ms = triton.testing.do_bench(lambda: torch.sort(x, dim=-1, descending=False, stable=True))
            triton_ms = triton.testing.do_bench(lambda: sort(x, dim=-1, descending=False))
            print(
                f"shape: {shape}, dtype: {dtype}, Torch: {torch_ms} Triton: {triton_ms} Speedup: {torch_ms/triton_ms}x")
            print()
