#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>

namespace {

__device__ void radix_rank_8x2048_precomputed(
    __attribute__((address_space(3))) uint16_t *digits, int64_t digits_offset,
    int64_t digits_stride,
    __attribute__((address_space(1))) const uint16_t *input,
    __attribute__((address_space(1))) const int32_t *input_indices,
    __attribute__((address_space(1))) uint16_t *output,
    __attribute__((address_space(1))) int32_t *output_indices32,
    __attribute__((address_space(1))) int64_t *output_indices64,
    __attribute__((address_space(1))) const int32_t *tile_offsets, int row,
    int tile, int n, int tiles, int valid_count, int final_pass) {
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
          final_pass != 0 ? input_indices[input_offset] : input_col;
      if (final_pass != 0) {
        output_indices64[output_offset] = static_cast<int64_t>(original_col);
      } else {
        output_indices32[output_offset] = original_col;
      }
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
  constexpr int items_per_thread = 8;
  constexpr int radix_bins = 256;

  __shared__ uint32_t histogram[radix_bins];
  histogram[threadIdx.x] = 0;
  __syncthreads();

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int local = static_cast<int>(threadIdx.x) * items_per_thread + item;
    if (local < valid_count) {
      const uint16_t digit =
          digits_aligned[digits_offset + local * digits_stride];
      atomicAdd(histogram + static_cast<int>(digit), 1u);
    }
  }
  __syncthreads();

  counts_aligned[counts_offset +
                 static_cast<int64_t>(threadIdx.x) * counts_stride] =
      static_cast<int32_t>(histogram[threadIdx.x]);
  __syncthreads();
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
    int tile, int n, int tiles, int valid_count, int final_pass) {
  (void)digits_allocated;
  (void)digits_size;
  radix_rank_8x2048_precomputed(digits_aligned, digits_offset, digits_stride,
                                input, input_indices, output, output_indices32,
                                output_indices64, tile_offsets, row, tile, n,
                                tiles, valid_count, final_pass);
}
