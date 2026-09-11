#include "tx81_run.h"

#include <assert.h>
#include <stdio.h>

#define CHECK_RANGE(name, got, base, exp_min, exp_max)                         \
  do {                                                                         \
    uintptr_t exp_min_addr = (uintptr_t)((int64_t)(base) + (exp_min));         \
    uintptr_t exp_max_addr = (uintptr_t)((int64_t)(base) + (exp_max));         \
    if ((got).min_addr != exp_min_addr || (got).max_addr != exp_max_addr) {    \
      fprintf(stderr, "FAIL %s: min=%#lx max=%#lx expected [%#lx, %#lx)\n",    \
              (name), (unsigned long)(got).min_addr,                           \
              (unsigned long)(got).max_addr, (unsigned long)exp_min_addr,      \
              (unsigned long)exp_max_addr);                                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
//  src address range tests
// ---------------------------------------------------------------------------

static int test_src_empty_zero_iterW(void) {
  char buf[64];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/1,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/0);
  CHECK_RANGE("src_empty_zero_iterW", r, buf, 0, 0);
  return 0;
}

static int test_src_empty_zero_iterH(void) {
  char buf[64];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/1,
      /*iterN=*/1, /*iterH=*/0, /*iterW=*/8);
  CHECK_RANGE("src_empty_zero_iterH", r, buf, 0, 0);
  return 0;
}

static int test_src_empty_zero_iterN(void) {
  char buf[64];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/1,
      /*iterN=*/0, /*iterH=*/4, /*iterW=*/8);
  CHECK_RANGE("src_empty_zero_iterN", r, buf, 0, 0);
  return 0;
}

// Simple 1D contiguous block copy: 10 iterations, each advancing 4 bytes,
// with bytes=4.  The pointer pattern is offsets 0,4,8,...,36, each
// touching [offset, offset+4).  max_addr = 36 + 4 = 40.
static int test_src_1d_contiguous(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/4,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/10);
  CHECK_RANGE("src_1d_contiguous", r, buf, 0, 40);
  return 0;
}

// Memcpy-like degenerate case: single copy of 1024 bytes.
static int test_src_memcpy_like(void) {
  char buf[2048];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/1024,
      /*strideN=*/1, /*strideH=*/1, /*strideW=*/1,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/1);
  CHECK_RANGE("src_memcpy_like", r, buf, 0, 1024);
  return 0;
}

// 2D with non-zero strideH after each inner (W) loop.
// W: 8 iterations, strideW=2, H: 4 iterations, strideH=16.
// coeff_W=2, coeff_H = 8*2 + 16 = 32.
// max_off = (4-1)*32 + (8-1)*2 = 96 + 14 = 110.
// max_addr = 110 + 4 = 114.
static int test_src_2d_with_strideH(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/16, /*strideW=*/2,
      /*iterN=*/1, /*iterH=*/4, /*iterW=*/8);
  CHECK_RANGE("src_2d_strideH", r, buf, 0, 114);
  return 0;
}

// 2D gather pattern: strideW=16, strideH=0.  After W loop the pointer
// naturally lands at the start of the next row (iterW * strideW bytes
// ahead), so strideH=0.
// coeff_W=16, coeff_H = 8*16 + 0 = 128.
// max_off = (4-1)*128 + (8-1)*16 = 384 + 112 = 496.
// max_addr = 496 + 4 = 500.
static int test_src_2d_gather(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/16,
      /*iterN=*/1, /*iterH=*/4, /*iterW=*/8);
  CHECK_RANGE("src_2d_gather", r, buf, 0, 500);
  return 0;
}

// 3D full: N=2, H=3, W=4 with all non-trivial strides.
// coeff_W=16, coeff_H = 4*16 + 8 = 72, coeff_N = 3*72 + 32 = 248.
// max_off = (2-1)*248 + (3-1)*72 + (4-1)*16 = 248 + 144 + 48 = 440.
// max_addr = 440 + 8 = 448.
static int test_src_3d_full(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/8,
      /*strideN=*/32, /*strideH=*/8, /*strideW=*/16,
      /*iterN=*/2, /*iterH=*/3, /*iterW=*/4);
  CHECK_RANGE("src_3d_full", r, buf, 0, 448);
  return 0;
}

// Only W dimension active (H and N each iterate once with zero stride).
// W=50, strideW=8 → max_off = 49*8 = 392, max_addr = 392 + 1 = 393.
static int test_src_1d_byte_stride(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_src_addr_range(
      buf, /*bytes=*/1,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/8,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/50);
  CHECK_RANGE("src_1d_byte_stride", r, buf, 0, 393);
  return 0;
}

// ---------------------------------------------------------------------------
//  dst address range tests — identical formula, different base / strides
// ---------------------------------------------------------------------------

static int test_dst_empty_zero_iterW(void) {
  char buf[64];
  Tx81MemAddrRange r = compute_gatherscatter_dst_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/1,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/0);
  CHECK_RANGE("dst_empty_zero_iterW", r, buf, 0, 0);
  return 0;
}

static int test_dst_1d_contiguous(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_dst_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/4,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/10);
  CHECK_RANGE("dst_1d_contiguous", r, buf, 0, 40);
  return 0;
}

// Scatter pattern on dst side: each inner copy advances by 16 bytes.
// coeff_W=16, coeff_H = 8*16 + 0 = 128.
// max_off = (4-1)*128 + (8-1)*16 = 496, max_addr = 496 + 4 = 500.
static int test_dst_2d_scatter(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_dst_addr_range(
      buf, /*bytes=*/4,
      /*strideN=*/0, /*strideH=*/0, /*strideW=*/16,
      /*iterN=*/1, /*iterH=*/4, /*iterW=*/8);
  CHECK_RANGE("dst_2d_scatter", r, buf, 0, 500);
  return 0;
}

static int test_dst_3d_full(void) {
  char buf[4096];
  Tx81MemAddrRange r = compute_gatherscatter_dst_addr_range(
      buf, /*bytes=*/8,
      /*strideN=*/32, /*strideH=*/8, /*strideW=*/16,
      /*iterN=*/2, /*iterH=*/3, /*iterW=*/4);
  CHECK_RANGE("dst_3d_full", r, buf, 0, 448);
  return 0;
}

// dst memcpy-like: single copy of 1024 bytes.
static int test_dst_memcpy_like(void) {
  char buf[2048];
  Tx81MemAddrRange r = compute_gatherscatter_dst_addr_range(
      buf, /*bytes=*/1024,
      /*strideN=*/1, /*strideH=*/1, /*strideW=*/1,
      /*iterN=*/1, /*iterH=*/1, /*iterW=*/1);
  CHECK_RANGE("dst_memcpy_like", r, buf, 0, 1024);
  return 0;
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------

int main(void) {
  int failed = 0;
  failed |= test_src_empty_zero_iterW();
  failed |= test_src_empty_zero_iterH();
  failed |= test_src_empty_zero_iterN();
  failed |= test_src_1d_contiguous();
  failed |= test_src_memcpy_like();
  failed |= test_src_2d_with_strideH();
  failed |= test_src_2d_gather();
  failed |= test_src_3d_full();
  failed |= test_src_1d_byte_stride();

  failed |= test_dst_empty_zero_iterW();
  failed |= test_dst_1d_contiguous();
  failed |= test_dst_2d_scatter();
  failed |= test_dst_3d_full();
  failed |= test_dst_memcpy_like();

  if (failed) {
    fprintf(stderr, "tx81_gatherscatter_addr_range_test: %d case(s) failed\n",
            failed);
    return 1;
  }
  printf("tx81_gatherscatter_addr_range_test: all tests passed\n");
  return 0;
}
