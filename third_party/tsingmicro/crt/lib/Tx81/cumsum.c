//===---------------------- cumsum.c -------------------------------------===//
//
// Runtime API for exclusive cumsum on Tx81.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"
#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

void __Memcpy(uint64_t *src, uint64_t *dst, uint32_t elem_count, uint16_t fmt);
void __Memset(char *dst, int value, int *dst_shape, int *dst_stride, int rank,
              uint16_t fmt);
void __AddVV(uint64_t *src0, uint64_t *src1, uint64_t *dst, uint32_t elem_count,
             RND_MODE round, uint16_t fmt);

void __GatherScatter(uint64_t *src, uint64_t *dst, uint32_t bytes,
                     uint32_t src_strideN, uint32_t src_strideH,
                     uint32_t src_strideW, uint32_t src_iterN,
                     uint32_t src_iterH, uint32_t src_iterW,
                     uint32_t dst_strideN, uint32_t dst_strideH,
                     uint32_t dst_strideW, uint32_t dst_iterN,
                     uint32_t dst_iterH, uint32_t dst_iterW);

#define CUMSUM_BATCH_MIN_ROWS 4
#define CUMSUM_BATCH_MAX_ELEMS (1u << 20)

static void copy_elements(char *dst, char *src, uint32_t elem_count,
                          uint16_t fmt) {
  if (elem_count == 0)
    return;
  __Memcpy((uint64_t *)src, (uint64_t *)dst, elem_count, fmt);
}

static void zero_elements(char *dst, uint32_t elem_count, uint16_t fmt) {
  if (elem_count == 0)
    return;
  assert(elem_count <= INT_MAX);
  int shape[1] = {(int)elem_count};
  int stride[1] = {1};
  __Memset(dst, 0, shape, stride, 1, fmt);
}

static void add_vectors(char *dst, char *lhs, char *rhs, uint32_t elem_count,
                        uint16_t fmt) {
  if (elem_count == 0)
    return;
  __AddVV((uint64_t *)lhs, (uint64_t *)rhs, (uint64_t *)dst, elem_count,
          RND_NEAREST_EVEN, fmt);
}

static void copy_strided_rows(char *dst, char *src, uint32_t rows,
                              uint32_t elem_count, uint32_t dst_row_elems,
                              uint32_t src_row_elems, uint32_t bytes) {
  if (rows == 0 || elem_count == 0)
    return;
  uint64_t copy_bytes = (uint64_t)elem_count * bytes;
  uint64_t src_stride = (uint64_t)src_row_elems * bytes;
  uint64_t dst_stride = (uint64_t)dst_row_elems * bytes;
  assert(copy_bytes <= UINT32_MAX);
  assert(src_stride <= UINT32_MAX);
  assert(dst_stride <= UINT32_MAX);
  __GatherScatter((uint64_t *)src, (uint64_t *)dst, (uint32_t)copy_bytes, 1, 1,
                  (uint32_t)src_stride, 1, 1, rows, 1, 1, (uint32_t)dst_stride,
                  1, 1, rows);
}

static int64_t min_i64(int64_t lhs, int64_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

static void cumsum_row(char *src_row, char *exclusive_row, char *total,
                       char *scratch_row, uint32_t n, uint32_t bytes,
                       uint32_t pad, uint16_t fmt) {
  char *buf0 = scratch_row;
  char *buf1 = scratch_row + (uint64_t)pad * bytes;
  char *cur = src_row;
  char *next = buf0;

  if (n == 1) {
    zero_elements(exclusive_row, 1, fmt);
    copy_elements(total, src_row, 1, fmt);
    return;
  }

  for (uint32_t step = 1;; step <<= 1) {
    int final_step = (step << 1) >= n;

    if (final_step) {
      uint32_t tail_count = n - step - 1;
      zero_elements(exclusive_row, 1, fmt);
      copy_elements(exclusive_row + bytes, cur, step, fmt);
      add_vectors(exclusive_row + ((uint64_t)step + 1) * bytes,
                  cur + (uint64_t)step * bytes, cur, tail_count, fmt);
      add_vectors(total, cur + ((uint64_t)n - 1) * bytes,
                  cur + ((uint64_t)n - 1 - step) * bytes, 1, fmt);
      break;
    }

    copy_elements(next, cur, step, fmt);
    add_vectors(next + (uint64_t)step * bytes, cur + (uint64_t)step * bytes,
                cur, n - step, fmt);
    if (cur == src_row) {
      cur = next;
      next = buf1;
    } else {
      char *old_cur = cur;
      cur = next;
      next = old_cur;
    }
  }
}

static void build_shifted_tile(char *shifted, char *cur, uint32_t rows,
                               uint32_t n, uint32_t step, uint32_t prev_step,
                               uint32_t bytes, uint16_t fmt) {
  uint32_t zero_count = step - prev_step;
  for (uint32_t row = 0; row < rows; ++row) {
    uint64_t row_base = (uint64_t)row * n;
    zero_elements(shifted + (row_base + prev_step) * bytes, zero_count, fmt);
  }
  copy_strided_rows(shifted + (uint64_t)step * bytes, cur, rows, n - step, n, n,
                    bytes);
}

static void cumsum_rows_batched(char *src, char *exclusive, char *total,
                                char *scratch, uint32_t rows, uint32_t n,
                                uint32_t bytes, uint16_t fmt) {
  uint64_t tile_elems = (uint64_t)rows * n;
  char *buf0 = scratch;
  char *buf1 = scratch + tile_elems * bytes;
  char *shifted = scratch + 2 * tile_elems * bytes;
  char *cur = src;
  char *next = buf0;
  uint32_t prev_step = 0;

  for (uint32_t step = 1; step < n; step <<= 1) {
    build_shifted_tile(shifted, cur, rows, n, step, prev_step, bytes, fmt);
    add_vectors(next, cur, shifted, (uint32_t)tile_elems, fmt);

    cur = next;
    next = next == buf0 ? buf1 : buf0;
    prev_step = step;
  }

  for (uint32_t row = 0; row < rows; ++row) {
    char *exclusive_row = exclusive + (uint64_t)row * n * bytes;
    zero_elements(exclusive_row, 1, fmt);
  }
  copy_strided_rows(exclusive + bytes, cur, rows, n - 1, n, n, bytes);
  copy_strided_rows(total, cur + ((uint64_t)n - 1) * bytes, rows, 1, 1, n,
                    bytes);
}

void __CumsumPad(void *src, void *exclusive, void *total, void *scratch,
                 int rank, int axis, int *shape, int pad, uint16_t fmt) {
  assert(rank > 0);
  assert(axis == rank - 1);

  Data_Format dtype = (Data_Format)fmt;
  uint32_t bytes = get_dtype_size_new(dtype);
  int64_t last_dim = shape[rank - 1];
  assert(last_dim > 0);
  assert(pad >= last_dim);
  assert(last_dim <= UINT32_MAX);
  assert(pad <= UINT32_MAX);

  int64_t outer = 1;
  for (int i = 0; i < rank - 1; ++i)
    outer *= shape[i];

  char *src_bytes = (char *)src;
  char *exclusive_bytes = (char *)exclusive;
  char *total_bytes = (char *)total;
  char *scratch_bytes = (char *)scratch;
  int64_t scratch_row_elems = (int64_t)pad + last_dim;
  uint32_t n = (uint32_t)last_dim;
  uint32_t pad_count = (uint32_t)pad;

  if (n == 1) {
    assert(outer <= UINT32_MAX);
    zero_elements(exclusive_bytes, (uint32_t)outer, fmt);
    copy_elements(total_bytes, src_bytes, (uint32_t)outer, fmt);
    return;
  }

  int64_t scratch_elems = outer * scratch_row_elems;
  int64_t max_tile_by_scratch = scratch_elems / (3 * (int64_t)n);
  int64_t max_tile_by_add =
      CUMSUM_BATCH_MAX_ELEMS / n > 0 ? CUMSUM_BATCH_MAX_ELEMS / n : 1;
  int64_t tile_rows =
      min_i64(outer, min_i64(max_tile_by_scratch, max_tile_by_add));

  int64_t row = 0;
  if (tile_rows >= CUMSUM_BATCH_MIN_ROWS) {
    for (; row < outer;) {
      int64_t rows = min_i64(tile_rows, outer - row);
      if (rows < CUMSUM_BATCH_MIN_ROWS)
        break;
      cumsum_rows_batched(src_bytes + row * (int64_t)n * bytes,
                          exclusive_bytes + row * (int64_t)n * bytes,
                          total_bytes + row * bytes, scratch_bytes,
                          (uint32_t)rows, n, bytes, fmt);
      row += rows;
    }
  }

  for (; row < outer; ++row) {
    cumsum_row(src_bytes + row * (int64_t)n * bytes,
               exclusive_bytes + row * (int64_t)n * bytes,
               total_bytes + row * bytes,
               scratch_bytes + row * scratch_row_elems * bytes, n, bytes,
               pad_count, fmt);
  }
}
