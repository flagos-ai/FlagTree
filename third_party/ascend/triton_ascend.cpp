/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"

#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_triton_ascend_passes_convert(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_triton_to_linalg_pipeline",
                     mlir::triton::createTritonToLinalgExperimentalPass);
}

// register ascend passes to triton
void init_triton_ascend(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_ascend_passes_convert(passes.def_submodule("convert"));
}