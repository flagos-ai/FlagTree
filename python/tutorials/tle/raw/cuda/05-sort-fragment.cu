#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace {

__attribute__((device, always_inline)) void radix_rank_8x2048_precomputed(
    uint16_t (&digits_fragment)[8],

    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) const int32_t *input_indices,

    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int32_t *output_indices32,
    __attribute__((address_space(1))) int64_t *output_indices64,

    __attribute__((address_space(1))) const int32_t *tile_offsets, int row,
    int tile, int n, int tiles, int valid_count, int first_pass,
    int final_pass) {
  constexpr int block_threads = 256;
  constexpr int items_per_thread = 8;
  constexpr int radix_bits = 8;
  constexpr int radix_bins = 1 << radix_bits;
  using BlockSort =
      cub::BlockRadixSort<uint16_t, block_threads, items_per_thread, int>;

  __shared__ typename BlockSort::TempStorage temp_storage;
  __shared__ uint32_t bin_starts[radix_bins];
  __shared__ uint32_t global_bin_starts[radix_bins];
  __shared__ uint16_t thread_last_digit[block_threads];
  int source_cols[items_per_thread];

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    source_cols[item] = local;
  }

  BlockSort(temp_storage).Sort(digits_fragment, source_cols, 0, radix_bits);
  __syncthreads();

  bin_starts[threadIdx.x] = static_cast<uint32_t>(valid_count);
  thread_last_digit[threadIdx.x] = digits_fragment[items_per_thread - 1];
  const int64_t offset =
      (static_cast<int64_t>(row) * tiles + tile) * radix_bins +
      static_cast<int>(threadIdx.x);
  global_bin_starts[threadIdx.x] = static_cast<uint32_t>(tile_offsets[offset]);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const uint16_t digit = digits_fragment[item];
      bool starts_bin = local == 0;
      if (!starts_bin) {
        const uint16_t previous_digit =
            item == 0 ? thread_last_digit[static_cast<int>(threadIdx.x) - 1]
                      : digits_fragment[item - 1];
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
      const int digit = static_cast<int>(digits_fragment[item]);
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

__attribute__((device, always_inline)) void radix_histogram_digits_8x2048(
    uint16_t *digits_fragment, bool *mask_fragment, int64_t fragment_size,
    __attribute__((address_space(3))) uint32_t *counts, int64_t counts_stride) {
  counts[static_cast<int64_t>(threadIdx.x) * counts_stride] = 0;
  __syncthreads();
  for (unsigned i = 0; i < fragment_size; i++) {
    if (mask_fragment[i]) {
      atomicAdd((uint32_t *)(counts + counts_stride * digits_fragment[i]), 1u);
    }
  }
  __syncthreads();
}

} // namespace

__attribute__((device, always_inline)) void RadixHistogramDigits8x2048(
    uint16_t *digits_fragment, int64_t digits_fragment_size,
    bool *mask_fragment, int64_t mask_fragment_size,
    __attribute__((address_space(3))) uint32_t *counts_allocated,
    __attribute__((address_space(3))) uint32_t *counts_aligned,
    int64_t counts_offset, int64_t counts_size, int64_t counts_stride) {
  (void)counts_allocated;
  (void)counts_size;
  (void)digits_fragment_size;
  radix_histogram_digits_8x2048(digits_fragment, mask_fragment,
                                mask_fragment_size,
                                counts_aligned + counts_offset, counts_stride);
}

__attribute__((device, always_inline)) void RadixRank8x2048Precomputed(
    uint16_t *digits_fragment, int64_t digits_fragment_size,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) const int32_t *input_indices,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int32_t *output_indices32,
    __attribute__((address_space(1))) int64_t *output_indices64,
    __attribute__((address_space(1))) const int32_t *tile_offsets, int row,
    int tile, int n, int tiles, int valid_count, int first_pass,
    int final_pass) {
  // The 2048-element blocked layout gives each of the 256 threads eight items.
  (void)digits_fragment_size;
  radix_rank_8x2048_precomputed(
      *reinterpret_cast<uint16_t(*)[8]>(digits_fragment), input, input_indices,
      output, output_indices32, output_indices64, tile_offsets, row, tile, n,
      tiles, valid_count, first_pass, final_pass);
}
