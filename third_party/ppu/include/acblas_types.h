/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_ACBLAS_TYPES_H
#define TRITON_ACBLAS_TYPES_H

#include <cstddef>
#include <cstdint>

// Forward declarations of acBLAS types and functions.

/* ACBLAS status type returns */
typedef enum {
  ACBLAS_STATUS_SUCCESS = 0,
  ACBLAS_STATUS_NOT_INITIALIZED = 1,
  ACBLAS_STATUS_ALLOC_FAILED = 3,
  ACBLAS_STATUS_INVALID_VALUE = 7,
  ACBLAS_STATUS_ARCH_MISMATCH = 8,
  ACBLAS_STATUS_MAPPING_ERROR = 11,
  ACBLAS_STATUS_EXECUTION_FAILED = 13,
  ACBLAS_STATUS_INTERNAL_ERROR = 14,
  ACBLAS_STATUS_NOT_SUPPORTED = 15,
  ACBLAS_STATUS_LICENSE_ERROR = 16
} acblasStatus_t;

typedef enum {
  ACBLAS_COMPUTE_16F = 64,          /* half - default */
  ACBLAS_COMPUTE_16F_PEDANTIC = 65, /* half - pedantic */
  ACBLAS_COMPUTE_32F = 68,          /* float - default */
  ACBLAS_COMPUTE_32F_PEDANTIC = 69, /* float - pedantic */
  ACBLAS_COMPUTE_32F_FAST_16F =
      74, /* float - fast, allows down-converting inputs to half or TF32 */
  ACBLAS_COMPUTE_32F_FAST_16BF =
      75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
  ACBLAS_COMPUTE_32F_FAST_TF32 =
      77, /* float - fast, allows down-converting inputs to TF32 */
  ACBLAS_COMPUTE_64F = 70,          /* double - default */
  ACBLAS_COMPUTE_64F_PEDANTIC = 71, /* double - pedantic */
  ACBLAS_COMPUTE_32I = 72,          /* signed 32-bit int - default */
  ACBLAS_COMPUTE_32I_PEDANTIC = 73, /* signed 32-bit int - pedantic */
} acblasComputeType_t;

typedef enum {
  ACBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
  ACBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
  ACBLASLT_MATMUL_DESC_POINTER_MODE = 2,
  ACBLASLT_MATMUL_DESC_TRANSA = 3,
  ACBLASLT_MATMUL_DESC_TRANSB = 4,
  ACBLASLT_MATMUL_DESC_TRANSC = 5,
  ACBLASLT_MATMUL_DESC_FILL_MODE = 6,
  ACBLASLT_MATMUL_DESC_EPILOGUE = 7,
  ACBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
  ACBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,
  ACBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
  ACBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,
  ACBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,
  ACBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,
  ACBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 15,
  ACBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
  ACBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
  ACBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
  ACBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
  ACBLASLT_MATMUL_DESC_AMAX_D_POINTER = 21,
  ACBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
  ACBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 23,
  ACBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = 24,
  ACBLASLT_MATMUL_DESC_FAST_ACCUM = 25,
  ACBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 26,
  ACBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = 27,
  ACBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = 28,
  ACBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = 29,
  ACBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = 30,
} acblasLtMatmulDescAttributes_t;

typedef enum {
  ACBLAS_OP_N = 0,
  ACBLAS_OP_T = 1,
  ACBLAS_OP_C = 2,
  ACBLAS_OP_HERMITAN = 2, /* synonym if ACBLAS_OP_C */
  ACBLAS_OP_CONJG =
      3 /* conjugate, placeholder - not supported in the current release */
} acblasOperation_t;

typedef enum {
  ACBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
  ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
  ACBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,
  ACBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,
  ACBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,
  ACBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,
  ACBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,
  ACBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,
  ACBLASLT_MATMUL_PREF_IMPL_MASK = 12,
} acblasLtMatmulPreferenceAttributes_t;
typedef struct {
  uint64_t data[8];
} acblasLtMatrixLayoutOpaque_t;
typedef acblasLtMatrixLayoutOpaque_t *acblasLtMatrixLayout_t;

typedef struct {
  uint64_t data[8];
} acblasLtMatmulPreferenceOpaque_t;
typedef acblasLtMatmulPreferenceOpaque_t *acblasLtMatmulPreference_t;

typedef struct {
  uint64_t data[8];
} acblasLtMatmulAlgo_t;

typedef struct {
  acblasLtMatmulAlgo_t algo;
  size_t workspaceSize;
  acblasStatus_t state;
  float wavesCount;
  int reserved[4];
} acblasLtMatmulHeuristicResult_t;

typedef enum hggcDataType_t {
  HGGC_R_16F = 2,      /* real as a half */
  HGGC_C_16F = 6,      /* complex as a pair of half numbers */
  HGGC_R_16BF = 14,    /* real as a bfloat16 */
  HGGC_C_16BF = 15,    /* complex as a pair of bfloat16 numbers */
  HGGC_R_32F = 0,      /* real as a float */
  HGGC_C_32F = 4,      /* complex as a pair of float numbers */
  HGGC_R_64F = 1,      /* real as a double */
  HGGC_C_64F = 5,      /* complex as a pair of double numbers */
  HGGC_R_4I = 16,      /* real as a signed 4-bit int */
  HGGC_C_4I = 17,      /* complex as a pair of signed 4-bit int numbers */
  HGGC_R_4U = 18,      /* real as a unsigned 4-bit int */
  HGGC_C_4U = 19,      /* complex as a pair of unsigned 4-bit int numbers */
  HGGC_R_8I = 3,       /* real as a signed 8-bit int */
  HGGC_C_8I = 7,       /* complex as a pair of signed 8-bit int numbers */
  HGGC_R_8U = 8,       /* real as a unsigned 8-bit int */
  HGGC_C_8U = 9,       /* complex as a pair of unsigned 8-bit int numbers */
  HGGC_R_16I = 20,     /* real as a signed 16-bit int */
  HGGC_C_16I = 21,     /* complex as a pair of signed 16-bit int numbers */
  HGGC_R_16U = 22,     /* real as a unsigned 16-bit int */
  HGGC_C_16U = 23,     /* complex as a pair of unsigned 16-bit int numbers */
  HGGC_R_32I = 10,     /* real as a signed 32-bit int */
  HGGC_C_32I = 11,     /* complex as a pair of signed 32-bit int numbers */
  HGGC_R_32U = 12,     /* real as a unsigned 32-bit int */
  HGGC_C_32U = 13,     /* complex as a pair of unsigned 32-bit int numbers */
  HGGC_R_64I = 24,     /* real as a signed 64-bit int */
  HGGC_C_64I = 25,     /* complex as a pair of signed 64-bit int numbers */
  HGGC_R_64U = 26,     /* real as a unsigned 64-bit int */
  HGGC_C_64U = 27,     /* complex as a pair of unsigned 64-bit int numbers */
  HGGC_R_8F_E4M3 = 28, /* real as a fp8_e4m3 */
  HGGC_R_8F_E5M2 = 29, /* real as a fp8_e5m2 */
} hggcDataType;

struct acblasContext;
typedef struct acblasLtContext *acblasLtHandle_t;
struct acblasLtMatmulDescOpaque_t;
typedef acblasLtMatmulDescOpaque_t *acblasLtMatmulDesc_t;
struct HGstream_st;
typedef struct HGstream_st *hggcStream_t;

#endif // TRITON_ACBLAS_TYPES_H
