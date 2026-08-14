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

#ifndef TRITON_ACBLAS_INSTANCE_H
#define TRITON_ACBLAS_INSTANCE_H

#include "acblas_types.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>

class AcblasLtInstance {
  // Typedefs for acblas functions
  typedef acblasStatus_t (*acblasLtCreate_t)(acblasLtHandle_t *);
  typedef acblasStatus_t (*acblasLtDestroy_t)(acblasLtHandle_t);
  typedef acblasStatus_t (*acblasLtMatmulDescCreate_t)(acblasLtMatmulDesc_t *,
                                                       acblasComputeType_t,
                                                       hggcDataType_t);
  typedef acblasStatus_t (*acblasLtMatmulDescDestroy_t)(acblasLtMatmulDesc_t);
  typedef acblasStatus_t (*acblasLtMatmulDescSetAttribute_t)(
      acblasLtMatmulDesc_t, acblasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef acblasStatus_t (*acblasLtMatrixLayoutCreate_t)(
      acblasLtMatrixLayout_t *, hggcDataType_t, uint64_t, uint64_t, int64_t);
  typedef acblasStatus_t (*acblasLtMatrixLayoutDestroy_t)(
      acblasLtMatrixLayout_t);
  typedef acblasStatus_t (*acblasLtMatmulPreferenceCreate_t)(
      acblasLtMatmulPreference_t *);
  typedef acblasStatus_t (*acblasLtMatmulPreferenceDestroy_t)(
      acblasLtMatmulPreference_t);
  typedef acblasStatus_t (*acblasLtMatmulPreferenceSetAttribute_t)(
      acblasLtMatmulPreference_t, acblasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef acblasStatus_t (*acblasLtMatmulAlgoGetHeuristic_t)(
      acblasLtHandle_t, acblasLtMatmulDesc_t, acblasLtMatrixLayout_t,
      acblasLtMatrixLayout_t, acblasLtMatrixLayout_t, acblasLtMatrixLayout_t,
      acblasLtMatmulPreference_t, int, acblasLtMatmulHeuristicResult_t *,
      int *);
  typedef acblasStatus_t (*acblasLtMatmul_t)(
      acblasLtHandle_t, acblasLtMatmulDesc_t, const void *, const void *,
      const acblasLtMatrixLayout_t, const void *, const acblasLtMatrixLayout_t,
      const void *, const void *, const acblasLtMatrixLayout_t, void *,
      const acblasLtMatrixLayout_t, const acblasLtMatmulAlgo_t *, void *,
      size_t, hggcStream_t);

  static constexpr const char *name = "libacblas.so";

  acblasLtCreate_t acblasLtCreate;
  acblasLtDestroy_t acblasLtDestroy;
  acblasLtMatmulDescCreate_t acblasLtMatmulDescCreate;
  acblasLtMatmulDescDestroy_t acblasLtMatmulDescDestroy;
  acblasLtMatmulDescSetAttribute_t acblasLtMatmulDescSetAttribute;
  acblasLtMatrixLayoutCreate_t acblasLtMatrixLayoutCreate;
  acblasLtMatrixLayoutDestroy_t acblasLtMatrixLayoutDestroy;
  acblasLtMatmulPreferenceCreate_t acblasLtMatmulPreferenceCreate;
  acblasLtMatmulPreferenceDestroy_t acblasLtMatmulPreferenceDestroy;
  acblasLtMatmulPreferenceSetAttribute_t acblasLtMatmulPreferenceSetAttribute;
  acblasLtMatmulAlgoGetHeuristic_t acblasLtMatmulAlgoGetHeuristic;
  acblasLtMatmul_t acblasLtMatmul;

  void *dylibHandle = nullptr;
  acblasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  acblasLtMatmulPreference_t preference = NULL;

  void loadAcblasDylib() {
    if (dylibHandle == nullptr) {
      // First reuse the existing handle
      dylibHandle = dlopen(name, RTLD_NOLOAD);
    }
    if (dylibHandle == nullptr) {
      // If not found, try to load it
      dylibHandle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (dylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Make sure it is in your "
                               "LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    acblasLtCreate = (acblasLtCreate_t)dlsym(dylibHandle, "acblasLtCreate");
    acblasLtDestroy = (acblasLtDestroy_t)dlsym(dylibHandle, "acblasLtDestroy");
    acblasLtMatmulDescCreate = (acblasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "acblasLtMatmulDescCreate");
    acblasLtMatmulDescDestroy = (acblasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "acblasLtMatmulDescDestroy");
    acblasLtMatmulDescSetAttribute = (acblasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "acblasLtMatmulDescSetAttribute");
    acblasLtMatrixLayoutCreate = (acblasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "acblasLtMatrixLayoutCreate");
    acblasLtMatrixLayoutDestroy = (acblasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "acblasLtMatrixLayoutDestroy");
    acblasLtMatmulPreferenceCreate = (acblasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "acblasLtMatmulPreferenceCreate");
    acblasLtMatmulPreferenceDestroy = (acblasLtMatmulPreferenceDestroy_t)dlsym(
        dylibHandle, "acblasLtMatmulPreferenceDestroy");
    acblasLtMatmulPreferenceSetAttribute =
        (acblasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "acblasLtMatmulPreferenceSetAttribute");
    acblasLtMatmulAlgoGetHeuristic = (acblasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "acblasLtMatmulAlgoGetHeuristic");
    acblasLtMatmul = (acblasLtMatmul_t)dlsym(dylibHandle, "acblasLtMatmul");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadAcblasDylib() { dlclose(dylibHandle); }

  void successOrExit(acblasStatus_t status) {
    if (status != ACBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("acBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  // Simple wrapper around the acblasLtMatmul function
  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, hggcDataType_t dtype, float alpha, float beta) {
    acblasLtMatmulDesc_t matmulDesc = NULL;

    acblasOperation_t transa = ACBLAS_OP_T;
    acblasOperation_t transb = ACBLAS_OP_N;

    int8_t fastAccum = 1;

    acblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    int returnedResults = 0;
    acblasLtMatmulHeuristicResult_t heuristicResult = {};

    // Select compute type. Use TF32 when inputs are FP32, otherwise default
    // FP32 accumulation.
    acblasComputeType_t computeType = (dtype == HGGC_R_32F)
                                          ? ACBLAS_COMPUTE_32F_FAST_TF32
                                          : ACBLAS_COMPUTE_32F;
    successOrExit(
        acblasLtMatmulDescCreate(&matmulDesc, computeType, HGGC_R_32F));
    successOrExit(acblasLtMatmulDescSetAttribute(
        matmulDesc, ACBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(acblasLtMatmulDescSetAttribute(
        matmulDesc, ACBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == HGGC_R_8F_E4M3) {
      successOrExit(acblasLtMatmulDescSetAttribute(
          matmulDesc, ACBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == HGGC_R_8F_E4M3 ? HGGC_R_16F : dtype;
    successOrExit(acblasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(acblasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(acblasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(acblasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    successOrExit(acblasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by acblasLtMatmulAlgoGetHeuristic");
    }

    successOrExit(acblasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, (void *)C, Cdesc,
                                 (void *)D, Ddesc, &heuristicResult.algo,
                                 (void *)workspace, workspaceSize, 0));
    if (Ddesc)
      successOrExit(acblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(acblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(acblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(acblasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(acblasLtMatmulDescDestroy(matmulDesc));
  }

public:
  AcblasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadAcblasDylib();
    acblasLtCreate(&ltHandle);

    successOrExit(acblasLtMatmulPreferenceCreate(&preference));
    successOrExit(acblasLtMatmulPreferenceSetAttribute(
        preference, ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~AcblasLtInstance() {
    if (preference)
      successOrExit(acblasLtMatmulPreferenceDestroy(preference));

    acblasLtDestroy(ltHandle);
    unloadAcblasDylib();
  }

  // C = A * B
  // Matrix B needs to be transposed, while matrix A does not. The function
  // *will-not* transpose the matrices, so the caller is responsible for
  // ensuring that the matrices are in the correct format and have the correct
  // dimensions.
  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              hggcDataType_t dtype) {
    // HGGC is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    gemm_impl(n, m, k, B, A, 0, C, dtype, 1.0f, 0.0f);
  }

  void gemm(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
            hggcDataType_t dtype, float alpha, float beta) {
    gemm_impl(n, m, k, B, A, C, D, dtype, alpha, beta);
  }
};

#endif // TRITON_ACBLAS_INSTANCE_H
