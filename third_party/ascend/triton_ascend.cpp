/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/TritonToHFusion/TritonToHFusion.h"
#include "triton-shared/TritonToHIVM/TritonToHIVM.h"
#include "triton-shared/TritonToLLVM/TritonToLLVM.h"
#include "triton-shared/TritonLinearize/TritonLinearize.h"

#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_triton_ascend_passes_convert(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_triton_to_linalg_pipeline",
                     mlir::triton::createTritonToLinalgExperimentalPass);
  ADD_PASS_WRAPPER_0("add_triton_to_llvm",
                     mlir::triton::createTritonToLLVMPass);
  ADD_PASS_WRAPPER_0("add_triton_to_hfusion",
                     mlir::triton::createTritonToHFusionPass);
  ADD_PASS_WRAPPER_0("add_triton_to_hivm",
                     mlir::triton::createTritonToHIVMPass);
  ADD_PASS_WRAPPER_0("add_triton_linearize",
                     mlir::triton::createTritonLinearizePass);
}

// register ascend passes to triton
void init_triton_ascend(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_ascend_passes_convert(passes.def_submodule("convert"));
}
