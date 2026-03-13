#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "passes.h"
#include "triton-shared/Conversion/TritonToLinalg/TritonToLinalg.h"
// #include
// "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToCoreDialects/TritonToCoreDialects.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace mlir;

void init_triton_tsingmicro_passes_convert(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_linalg_to_std", createConvertLinalgToStandardPass);
  ADD_PASS_WRAPPER_0("add_one_shot_bufferize",
                     bufferization::createOneShotBufferizePass);
  ADD_PASS_WRAPPER_0("add_triton_to_linalg", triton::createTritonToLinalgPass);
  ADD_PASS_WRAPPER_0("add_affine_to_std", createLowerAffinePass);
  // ADD_PASS_WRAPPER_0("add_triton_to_linalg_pipeline",
  //                    triton::createTritonToLinalgExperimentalPass);
  ADD_PASS_WRAPPER_0("add_triton_to_core",
                     triton::createTritonToCoreDialectsPass);

  ADD_PASS_WRAPPER_0("add_linalg_to_loops", createConvertLinalgToLoopsPass);
  ADD_PASS_WRAPPER_0("add_linalg_to_affine_loops",
                     createConvertLinalgToAffineLoopsPass);
  ADD_PASS_WRAPPER_0("add_lower_affine", createLowerAffinePass);

  m.def("add_affine_vectorize", [](mlir::PassManager &pm, int64_t vecsize) {
    affine::AffineVectorizeOptions vectorize_options;
    vectorize_options.vectorSizes.push_back(vecsize);
    pm.addNestedPass<func::FuncOp>(
        affine::createAffineVectorize(vectorize_options));
  });
}

void init_triton_tsingmicro_common(py::module &&m) {
  m.def("generic_print", [](ModuleOp mod) -> std::string {
    std::string str;
    llvm::raw_string_ostream os(str);
    auto printingFlags = OpPrintingFlags();
    printingFlags.enableDebugInfo();
    printingFlags.printGenericOpForm();
    mod.print(os, printingFlags);
    return str;
  });
}

void init_triton_tsingmicro(py::module &&m) {
  init_triton_tsingmicro_common(m.def_submodule("common"));
  auto passes = m.def_submodule("passes");
  init_triton_tsingmicro_passes_convert(passes.def_submodule("convert"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    using namespace mlir;
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, vector::VectorDialect,
                    func::FuncDialect>();

    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerAllDialectInterfaceImplementations(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    func::registerAllExtensions(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  // register passes here
}
