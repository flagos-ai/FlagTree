/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton-shared/DiscreteMaskAccessConversion/DiscreteMaskAccessConversionPass.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/TritonToHFusion/TritonToHFusion.h"
#include "triton-shared/TritonLinearize/TritonLinearize.h"
#include "triton-shared/TritonToLinalgIncubated/TritonToLinalgIncubatedPass.h"
#include "triton-shared/TritonToHIVM/TritonToHIVM.h"
#include "triton-shared/TritonToLLVM/TritonToLLVM.h"
#include "triton-shared/TritonToUnstructureIncubated/UnstructureConversionPass.h"
#include "triton-shared/BubbleUpOperation/BubbleUpOperation.h"
#include "triton-shared/TritonToAnnotation/TritonToAnnotation.h"

#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_triton_ascend_passes_convert(py::module &&m) {

  ADD_PASS_WRAPPER_0("add_triton_to_linalg_pipeline",
                     mlir::triton::createTritonToLinalgExperimentalPass);
  ADD_PASS_WRAPPER_0("add_triton_linearize",
                     mlir::triton::createTritonLinearizePass);
  ADD_PASS_WRAPPER_0("add_triton_to_annotation",
                     mlir::triton::createTritonToAnnotationPass);
  ADD_PASS_WRAPPER_0("add_triton_to_hivm",
                     mlir::triton::createTritonToHIVMPass);
  ADD_PASS_WRAPPER_0("add_triton_to_hfusion",
                     mlir::triton::createTritonToHFusionPass);
  ADD_PASS_WRAPPER_0("add_triton_to_llvm",
                     mlir::triton::createTritonToLLVMPass);
  ADD_PASS_WRAPPER_0("add_bubble_up_operation",
                     mlir::triton::createBubbleUpOperationPass);
  m.def(
      "add_triton_discretemaskaccessconversion",
      [](mlir::PassManager &pm,
         bool compile_on_910_95,
         bool force_simt_template) {
       DiscreteMaskAccessConversionOptions options;
       options.compileOn91095 = compile_on_910_95;
       options.forceSimtTemplate = force_simt_template;
       pm.addPass(
         mlir::triton::createDiscreteMaskAccessConversionPass(options));
      },
      py::arg("pm"),
      py::arg("compile_on_910_95"),
      py::arg("force_simt_template")
);
  m.def(
      "add_triton_to_unstructure_incubated",
      [](mlir::PassManager &pm,
         bool compile_on_910_95,
         bool force_simt_template) {
       TritonToUnstructureIncubatedOptions options;
       options.compileOn91095 = compile_on_910_95;
       options.forceSimtTemplate = force_simt_template;
       pm.addPass(
         mlir::triton::createTritonToUnstructureIncubatedPass(options));
      },
      py::arg("pm"),
      py::arg("compile_on_910_95"),
      py::arg("force_simt_template")
);
  m.def(
      "add_triton_to_linalg_incubated",
      [](mlir::PassManager &pm,
         bool global_kernel,
         bool named_ops,
         bool enable_nd2nz_on_vector,
	 bool enable_select_analysis,
         bool compile_on_910_95) {
        pm.addPass(mlir::triton::Incubated::createTritonToLinalgIncubatedPass(
            global_kernel, named_ops, enable_nd2nz_on_vector, enable_select_analysis, compile_on_910_95));
      },
      py::arg("pm"),
      py::arg("global_kernel"),
      py::arg("named_ops"),
      py::arg("enable_nd2nz_on_vector"),
      py::arg("enable_select_analysis"),
      py::arg("compile_on_910_95") 
);
}

// register ascend passes to triton
void init_triton_ascend(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_ascend_passes_convert(passes.def_submodule("convert"));
}
