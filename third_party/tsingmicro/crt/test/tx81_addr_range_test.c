#include "tx81_run.h"

#include <assert.h>
#include <stdio.h>

#define CHECK_RANGE(name, got, base, exp_min, exp_max)                         \
  do {                                                                         \
    uintptr_t exp_min_addr = (uintptr_t)((int64_t)(base) + (exp_min));         \
    uintptr_t exp_max_addr = (uintptr_t)((int64_t)(base) + (exp_max));         \
    if ((got).min_addr != exp_min_addr || (got).max_addr != exp_max_addr) {      \
      fprintf(stderr,                                                          \
              "FAIL %s: min=%#lx max=%#lx expected [%#lx, %#lx)\n", (name),  \
              (unsigned long)(got).min_addr, (unsigned long)(got).max_addr,  \
              (unsigned long)exp_min_addr, (unsigned long)exp_max_addr);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_empty_shape(void) {
  char buf[64];
  int shape[] = {0, 8};
  int src_stride[] = {16, 1};
  int dst_shape[] = {0, 8};
  int dst_stride[] = {16, 1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, dst_shape, dst_stride, 2, 4, Fmt_FP32);
  CHECK_RANGE("empty_shape", r, buf, 0, 0);
  return 0;
}

static int test_1d_contiguous(void) {
  char buf[4096];
  int shape[] = {100};
  int src_stride[] = {1};
  int dst_stride[] = {1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, shape, dst_stride, 1, 4, Fmt_FP32);
  CHECK_RANGE("1d_contiguous", r, buf, 0, 400);
  return 0;
}

static int test_2d_inner_line_with_padding(void) {
  char buf[4096];
  int shape[] = {4, 8};
  int src_stride[] = {16, 1};
  int dst_stride[] = {8, 1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, shape, dst_stride, 2, 4, Fmt_FP32);
  // max outer offset (3 * 16 * 4) + line (8 * 4) = 192 + 32 = 224
  CHECK_RANGE("2d_padding", r, buf, 0, 224);
  return 0;
}

static int test_inner_stride_nonunit(void) {
  char buf[4096];
  int shape[] = {4, 8};
  int src_stride[] = {16, 2};
  int dst_stride[] = {8, 1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, shape, dst_stride, 2, 4, Fmt_FP32);
  // max index offset: 3*64 + 7*8 = 192 + 56 = 248, + elem 4 -> 252
  CHECK_RANGE("inner_stride2", r, buf, 0, 252);
  return 0;
}

static int test_4d_fast_path(void) {
  char buf[8192];
  int shape[] = {2, 3, 4, 8};
  int src_stride[] = {96, 32, 8, 1};
  int dst_shape[] = {2, 3, 4, 8};
  int dst_stride[] = {96, 32, 8, 1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, dst_shape, dst_stride, 4, 4, Fmt_FP32);
  // outer max: 384 + 256 + 96 = 736, inner line 32 -> 768
  CHECK_RANGE("4d_contiguous", r, buf, 0, 768);
  return 0;
}

static int test_dst_noncontiguous_uses_inner_line(void) {
  char buf[4096];
  int shape[] = {2, 8};
  int src_stride[] = {8, 1};
  int dst_shape[] = {2, 8};
  int dst_stride[] = {16, 1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, dst_shape, dst_stride, 2, 4, Fmt_FP32);
  CHECK_RANGE("dst_noncontiguous", r, buf, 0, 64);
  return 0;
}

static int test_rank_too_large(void) {
  char buf[64];
  int shape[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  int stride[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, stride, shape, stride, 9, 4, Fmt_FP32);
  CHECK_RANGE("rank_too_large", r, buf, 0, 0);
  return 0;
}

static int test_negative_stride_axis(void) {
  char buf[4096];
  int shape[] = {4};
  int src_stride[] = {-1};
  int dst_stride[] = {1};
  Tx81SrcAddrRange r = compute_rdma_src_addr_range(
      buf, shape, src_stride, shape, dst_stride, 1, 4, Fmt_FP32);
  // offsets: 0, -4, -8, -12; envelope [base-12, base+4)
  CHECK_RANGE("rdma_negative_stride", r, buf, -12, 4);
  return 0;
}

static int test_wdma_empty_shape(void) {
  char buf[64];
  int shape[] = {0, 8};
  int src_stride[] = {16, 1};
  int dst_stride[] = {16, 1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 2, 4, Fmt_FP32);
  CHECK_RANGE("wdma_empty_shape", r, buf, 0, 0);
  return 0;
}

static int test_wdma_1d_contiguous(void) {
  char buf[4096];
  int shape[] = {100};
  int src_stride[] = {1};
  int dst_stride[] = {1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 1, 4, Fmt_FP32);
  CHECK_RANGE("wdma_1d_contiguous", r, buf, 0, 400);
  return 0;
}

static int test_wdma_2d_inner_line_with_dst_padding(void) {
  char buf[4096];
  int shape[] = {4, 8};
  int src_stride[] = {8, 1};
  int dst_stride[] = {16, 1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 2, 4, Fmt_FP32);
  CHECK_RANGE("wdma_2d_dst_padding", r, buf, 0, 224);
  return 0;
}

static int test_wdma_inner_stride_nonunit(void) {
  char buf[4096];
  int shape[] = {4, 8};
  int src_stride[] = {8, 1};
  int dst_stride[] = {16, 2};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 2, 4, Fmt_FP32);
  CHECK_RANGE("wdma_inner_stride2", r, buf, 0, 252);
  return 0;
}

static int test_wdma_4d_fast_path(void) {
  char buf[8192];
  int shape[] = {2, 3, 4, 8};
  int src_stride[] = {96, 32, 8, 1};
  int dst_stride[] = {96, 32, 8, 1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 4, 4, Fmt_FP32);
  CHECK_RANGE("wdma_4d_contiguous", r, buf, 0, 768);
  return 0;
}

static int test_wdma_src_noncontiguous_uses_inner_line(void) {
  char buf[4096];
  int shape[] = {2, 8};
  int src_stride[] = {16, 1};
  int dst_stride[] = {8, 1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 2, 4, Fmt_FP32);
  CHECK_RANGE("wdma_src_noncontiguous", r, buf, 0, 64);
  return 0;
}

static int test_wdma_rank_too_large(void) {
  char buf[64];
  int shape[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  int stride[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, stride, shape, stride, 9, 4, Fmt_FP32);
  CHECK_RANGE("wdma_rank_too_large", r, buf, 0, 0);
  return 0;
}

static int test_wdma_negative_stride_axis(void) {
  char buf[4096];
  int shape[] = {4};
  int src_stride[] = {1};
  int dst_stride[] = {-1};
  Tx81DstAddrRange r = compute_wdma_dst_addr_range(
      buf, shape, src_stride, shape, dst_stride, 1, 4, Fmt_FP32);
  CHECK_RANGE("wdma_negative_stride", r, buf, -12, 4);
  return 0;
}

<<<<<<< HEAD
int main(void) {
  int failed = 0;
=======
static int test_counter_initial_state(void) {
  if (get_dma_oob_count() != 0) {
    fprintf(stderr, "FAIL: oob counter not zero initially: %u\n", get_dma_oob_count());
    return 1;
  }
  if (get_dma_bad_magic_count() != 0) {
    fprintf(stderr, "FAIL: bad_magic counter not zero initially: %u\n", get_dma_bad_magic_count());
    return 1;
  }
  return 0;
}

static int test_abort_default_off(void) {
  if (get_dma_check_abort()) {
    fprintf(stderr, "FAIL: abort should be off by default\n");
    return 1;
  }
  return 0;
}

int main(void) {
  int failed = 0;
  failed |= test_counter_initial_state();
  failed |= test_abort_default_off();
>>>>>>> 295c0ad4 ([Feature] DMA OOB test framework)
  failed |= test_empty_shape();
  failed |= test_1d_contiguous();
  failed |= test_2d_inner_line_with_padding();
  failed |= test_inner_stride_nonunit();
  failed |= test_4d_fast_path();
  failed |= test_dst_noncontiguous_uses_inner_line();
  failed |= test_rank_too_large();
  failed |= test_negative_stride_axis();
  failed |= test_wdma_empty_shape();
  failed |= test_wdma_1d_contiguous();
  failed |= test_wdma_2d_inner_line_with_dst_padding();
  failed |= test_wdma_inner_stride_nonunit();
  failed |= test_wdma_4d_fast_path();
  failed |= test_wdma_src_noncontiguous_uses_inner_line();
  failed |= test_wdma_rank_too_large();
  failed |= test_wdma_negative_stride_axis();

  if (failed) {
    fprintf(stderr, "tx81_addr_range_test: %d case(s) failed\n", failed);
    return 1;
  }
  printf("tx81_addr_range_test: all tests passed\n");
  return 0;
}
