#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace {

constexpr int kBlockThreads = 256;

template <int items_per_thread>
__device__ void radix_rank_precomputed(
    __attribute__((address_space(3))) uint16_t *digits, int64_t digits_offset,
    int64_t digits_stride,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) const int32_t *input_indices,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int32_t *output_indices32,
    __attribute__((address_space(1))) int64_t *output_indices64,
    __attribute__((address_space(1))) const int32_t *tile_offsets, int row,
    int tile, int n, int tiles, int valid_count, int first_pass,
    int final_pass) {
  constexpr int block_threads = kBlockThreads;
  constexpr int radix_bits = 8;
  constexpr int radix_bins = 1 << radix_bits;
  using BlockSort =
      cub::BlockRadixSort<uint16_t, block_threads, items_per_thread, int>;

  __shared__ typename BlockSort::TempStorage temp_storage;
  __shared__ uint32_t bin_starts[radix_bins];
  __shared__ uint32_t global_bin_starts[radix_bins];
  __shared__ uint16_t thread_last_digit[block_threads];
  uint16_t thread_digits[items_per_thread];
  int source_cols[items_per_thread];

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    thread_digits[item] = digits[digits_offset + local * digits_stride];
    source_cols[item] = local;
  }

  BlockSort(temp_storage).Sort(thread_digits, source_cols, 0, radix_bits);
  __syncthreads();

  bin_starts[threadIdx.x] = static_cast<uint32_t>(valid_count);
  thread_last_digit[threadIdx.x] = thread_digits[items_per_thread - 1];
  const int64_t offset =
      (static_cast<int64_t>(row) * tiles + tile) * radix_bins +
      static_cast<int>(threadIdx.x);
  global_bin_starts[threadIdx.x] = static_cast<uint32_t>(tile_offsets[offset]);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const uint16_t digit = thread_digits[item];
      bool starts_bin = local == 0;
      if (!starts_bin) {
        const uint16_t previous_digit =
            item == 0 ? thread_last_digit[static_cast<int>(threadIdx.x) - 1]
                      : thread_digits[item - 1];
        starts_bin = digit != previous_digit;
      }
      if (starts_bin) {
        bin_starts[digit] = static_cast<uint32_t>(local);
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const int digit = static_cast<int>(thread_digits[item]);
      const uint32_t local_rank =
          static_cast<uint32_t>(local) - bin_starts[digit];
      const int output_col =
          static_cast<int>(global_bin_starts[digit] + local_rank);
      const int input_col =
          tile * block_threads * items_per_thread + source_cols[item];
      const int64_t input_offset = static_cast<int64_t>(row) * n + input_col;
      const int64_t output_offset = static_cast<int64_t>(row) * n + output_col;
      output[output_offset] = input[input_offset];
      const int original_col =
          first_pass != 0 ? input_col : input_indices[input_offset];
      if (final_pass != 0) {
        output_indices64[output_offset] = static_cast<int64_t>(original_col);
      } else {
        output_indices32[output_offset] = original_col;
      }
    }
  }
  __syncthreads();
}

template <int items_per_thread>
__device__ void
radix_histogram_digits(__attribute__((address_space(3))) const uint16_t *digits,
                       int64_t digits_offset, int64_t digits_stride,
                       __attribute__((address_space(3))) int32_t *counts,
                       int64_t counts_offset, int64_t counts_stride,
                       int valid_count) {
  constexpr int radix_bins = 256;

  __shared__ uint32_t histogram[radix_bins];
  histogram[threadIdx.x] = 0;
  __syncthreads();

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const uint16_t digit = digits[digits_offset + local * digits_stride];
      atomicAdd(histogram + static_cast<int>(digit), 1u);
    }
  }
  __syncthreads();

  counts[counts_offset + static_cast<int64_t>(threadIdx.x) * counts_stride] =
      static_cast<int32_t>(histogram[threadIdx.x]);
  __syncthreads();
}

// ====================== [OPT] 2 ======================
// Whole row fits in one tile, so the rank inside the tile already is the
// final rank: no tile histogram, no global bin offsets, and both byte passes
// collapse into a single 16-bit CUB sort with no global round trip.
template <int items_per_thread>
__device__ void
radix_sort_full_tile(__attribute__((address_space(3))) const uint16_t *keys,
                     int64_t keys_offset, int64_t keys_stride,
                     __attribute__((address_space(1))) const uint16_t *input,
                     __attribute__((address_space(1))) uint16_t *output,
                     __attribute__((address_space(1))) int64_t *output_indices,
                     int row, int n, int valid_count) {
  constexpr int block_threads = kBlockThreads;
  using BlockSort =
      cub::BlockRadixSort<uint16_t, block_threads, items_per_thread, int>;

  __shared__ typename BlockSort::TempStorage temp_storage;
  uint16_t thread_keys[items_per_thread];
  int source_cols[items_per_thread];

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    thread_keys[item] = keys[keys_offset + local * keys_stride];
    source_cols[item] = local;
  }

  // Full 16-bit key, so one Sort is the complete ordering.
  BlockSort(temp_storage).Sort(thread_keys, source_cols, 0, 16);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const int input_col = source_cols[item];
      const int64_t row_base = static_cast<int64_t>(row) * n;
      output[row_base + local] = input[row_base + input_col];
      output_indices[row_base + local] = static_cast<int64_t>(input_col);
    }
  }
  __syncthreads();
}

// ====================== [OPT] 3 ======================
// Several whole rows packed into one tile. The row index rides in the high
// bits of the key, so after one sort the elements of row r occupy exactly
// [r * row_len, (r + 1) * row_len) and the output address is the sorted
// position itself. Per-CTA cost is dominated by the 256-bin block scan, so
// folding 8 rows into one CTA cuts that cost 8x for short rows.
template <int items_per_thread>
__device__ void radix_sort_rows_per_tile(
    __attribute__((address_space(3))) const uint32_t *keys, int64_t keys_offset,
    int64_t keys_stride,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int64_t *output_indices, int base_row,
    int row_len, int valid_count, int end_bit) {
  constexpr int block_threads = kBlockThreads;
  using BlockSort =
      cub::BlockRadixSort<uint32_t, block_threads, items_per_thread, int>;

  __shared__ typename BlockSort::TempStorage temp_storage;
  uint32_t thread_keys[items_per_thread];
  int source_cols[items_per_thread];

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    thread_keys[item] = keys[keys_offset + local * keys_stride];
    source_cols[item] = local;
  }

  BlockSort(temp_storage).Sort(thread_keys, source_cols, 0, end_bit);
  __syncthreads();

  const int64_t row_base = static_cast<int64_t>(base_row) * row_len;
  const int col_mask = row_len - 1;
#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const int source_col = source_cols[item];
      output[row_base + local] = input[row_base + source_col];
      // Sorted position and source sit in the same row, so masking off the
      // row field gives the column inside that row.
      output_indices[row_base + local] =
          static_cast<int64_t>(source_col & col_mask);
    }
  }
  __syncthreads();
}

} // namespace

// Fine-grained replacement for Triton's tl.histogram. Loading, ordered-key
// conversion, digit extraction, and the final global store stay in Triton.
__device__ void RadixHistogramDigits8x2048(
    __attribute__((address_space(3))) const uint16_t *digits_allocated,
    __attribute__((address_space(3))) const uint16_t *digits_aligned,
    int64_t digits_offset, int64_t digits_size, int64_t digits_stride,
    __attribute__((address_space(3))) int32_t *counts_allocated,
    __attribute__((address_space(3))) int32_t *counts_aligned,
    int64_t counts_offset, int64_t counts_size, int64_t counts_stride,
    int valid_count) {
  (void)digits_allocated;
  (void)digits_size;
  (void)counts_allocated;
  (void)counts_size;
  radix_histogram_digits<8>(digits_aligned, digits_offset, digits_stride,
                            counts_aligned, counts_offset, counts_stride,
                            valid_count);
}

__device__ void RadixRank8x2048Precomputed(
    __attribute__((address_space(3))) uint16_t *digits_allocated,
    __attribute__((address_space(3))) uint16_t *digits_aligned,
    int64_t digits_offset, int64_t digits_size, int64_t digits_stride,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) const int32_t *input_indices,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int32_t *output_indices32,
    __attribute__((address_space(1))) int64_t *output_indices64,
    __attribute__((address_space(1))) const int32_t *tile_offsets, int row,
    int tile, int n, int tiles, int valid_count, int first_pass,
    int final_pass) {
  (void)digits_allocated;
  (void)digits_size;
  radix_rank_precomputed<8>(digits_aligned, digits_offset, digits_stride, input,
                            input_indices, output, output_indices32,
                            output_indices64, tile_offsets, row, tile, n, tiles,
                            valid_count, first_pass, final_pass);
}

// ====================== [OPT] 2 ======================
// One exported symbol for both tile widths. items_per_thread is 2 for a
// 512-wide tile and 8 for 2048. The two CUB instantiations stay separate;
// this branch only picks which one runs.
__device__ void RadixSortFullTile(
    __attribute__((address_space(3))) const uint16_t *keys_allocated,
    __attribute__((address_space(3))) const uint16_t *keys_aligned,
    int64_t keys_offset, int64_t keys_size, int64_t keys_stride,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int64_t *output_indices, int row, int n,
    int valid_count, int items_per_thread) {
  (void)keys_allocated;
  (void)keys_size;
  if (items_per_thread == 2) {
    radix_sort_full_tile<2>(keys_aligned, keys_offset, keys_stride, input,
                            output, output_indices, row, n, valid_count);
  } else {
    radix_sort_full_tile<8>(keys_aligned, keys_offset, keys_stride, input,
                            output, output_indices, row, n, valid_count);
  }
}

// ====================== [OPT] 3 ======================
__device__ void RadixSortRowsPerTile512(
    __attribute__((address_space(3))) const uint32_t *keys_allocated,
    __attribute__((address_space(3))) const uint32_t *keys_aligned,
    int64_t keys_offset, int64_t keys_size, int64_t keys_stride,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int64_t *output_indices, int base_row,
    int row_len, int valid_count, int end_bit) {
  (void)keys_allocated;
  (void)keys_size;
  radix_sort_rows_per_tile<2>(keys_aligned, keys_offset, keys_stride, input,
                              output, output_indices, base_row, row_len,
                              valid_count, end_bit);
}
