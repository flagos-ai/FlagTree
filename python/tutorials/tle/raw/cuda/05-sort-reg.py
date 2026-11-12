from pathlib import Path

import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle_gpu
import triton.experimental.tle.language.raw as tle_raw


_TLE_RAW_SOURCE_ = Path(__file__).parent / "05-sort.cu"

# Launch with num_warps=8 and num_ctas=1: 256 threads own 8 elements each.
_HISTOGRAM_LAYOUT = tl.constexpr(tle_gpu.BlockEncoding(
    size_per_thread=[8],
    threads_per_warp=[32],
    warps_per_cta=[8],
    order=[0],
))


@tle_raw.external(language='cuda', source=_TLE_RAW_SOURCE_)
def RadixHistogramDigits8x2048(*args, **kwargs):
    ...


@triton.jit
def _radix_tile_histogram_kernel_raw(
    input_ptr,
    counts_ptr,
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

    # Match fragment coordinates and cover each of the 2048 elements once.
    digits = tle_gpu.set_layout(digits, _HISTOGRAM_LAYOUT)
    mask = tle_gpu.set_layout(mask, _HISTOGRAM_LAYOUT)

    counts = tle_raw.call(
        RadixHistogramDigits8x2048,(
            tle_raw.fragment(digits),
            tle_raw.fragment(mask),
            tle_raw.output_buffer(
                init_value = tl.zeros([NUM_BINS], dtype=tl.uint32),
                shape=[NUM_BINS],
                dtype=tl.uint32,
            )
        )
    )

    bins = tl.arange(0, NUM_BINS)
    tl.store(counts_ptr + (row * tiles + tile) * NUM_BINS + bins, counts)


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
