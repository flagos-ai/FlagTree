#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

#include "Transform/Passes.h"
#include "TritonToTileIR/Passes.h"
#include "Utils/Utils.h"
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"
#include "passes.h"
#include "ir.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "triton/Dialect/Triton/IR/Types.h"

namespace py = pybind11;
using namespace mlir;
using namespace triton;

void init_triton_to_cudatile_passes(py::module &&m) {
  using namespace mlir::triton;
  // TODO: it is weird to pass mlir::triton::NVVM here since the conversion is
  // nvidia-specificontext
  m.def("add_triton_to_cudatile", [](mlir::PassManager &pm, bool approx,
                                    bool ftz, int capability, int num_ctas,
                                    int simt_num_warps, int occupancy, std::optional<int> num_stages) {
    pm.addPass(mlir::triton::createConvertTritonToCudaTilePass(
        approx, ftz, capability, num_ctas, simt_num_warps, occupancy, num_stages));
  });
  m.def("add_fma_fusion", [](mlir::PassManager &pm) {
    // Add FMA fusion pass to cuda tile entry operations
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    auto &epm = mpm.nest<cuda_tile::EntryOp>();
    epm.addPass(cuda_tile::createFuseFMAPass());
  });
  m.def("add_loop_split", [](mlir::PassManager &pm, int threshold = 1) {
    // Add Loop Split pass to cuda tile entry operations
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    auto &epm = mpm.nest<cuda_tile::EntryOp>();
    epm.addPass(cuda_tile::createLoopSplitPass({threshold}));
  });
  m.def("add_lift_tt_cf_to_scf", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createLiftTTCFToSCFPass());
  });
  m.def("add_strip_debuginfo", [](mlir::PassManager &pm) {
    // Strip debug info
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    mpm.addPass(mlir::createStripDebugInfoPass());
  });
  m.def("add_synthesize_debug_info_scopes", [](mlir::PassManager &pm) {
    // Synthesize scoped debug info
    auto &mpm = pm.nest<cuda_tile::ModuleOp>();
    mpm.addPass(cuda_tile::createSynthesizeDebugInfoScopesPass());
  });
  m.def("add_rewrite_tensor_pointers_to_ldst", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonRewriteTensorPointer());
  });
  m.def("add_assume_to_tileir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createRewriteAssumeWithCudaTilePass());
  });
  m.def("add_auto_gen_memtoken", [](mlir::PassManager &pm,
                                    bool enable_autogen_alias_mem_token
    ) {
    pm.addPass(mlir::triton::createAutoGenMemoryTokenPass(enable_autogen_alias_mem_token));
  });
}

void init_triton_tileir(py::module &&m) {
  init_triton_to_cudatile_passes(m.def_submodule("passes"));
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::cuda_tile::CudaTileDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    // Register cuda_tile passes to enable nested pass manager parsing
    cuda_tile::registerCudaTilePasses();
  });
  m.def("only_contain_legal_dialects", [](mlir::ModuleOp mod) {
    bool only_contain_legal_dialects = true;
    mod->walk([&](mlir::Operation *op) {
      if (!llvm::isa<mlir::ModuleOp>(op) &&
          (op->getName().getDialectNamespace() !=
              mlir::cuda_tile::CudaTileDialect::getDialectNamespace()
          )) {
        only_contain_legal_dialects = false;
      }
    });
    return only_contain_legal_dialects;
  });
  m.def("write_bytecode", [](mlir::ModuleOp mod) {
    // Find the cuda_tile::ModuleOp within the mlir::ModuleOp.
    cuda_tile::ModuleOp cudaTileModule;
    if (!mod.getBody()->empty())
      if (auto nestedCudaTileModule =
              dyn_cast<cuda_tile::ModuleOp>(&mod.getBody()->front()))
        cudaTileModule = nestedCudaTileModule;

    if (!cudaTileModule)
      throw std::runtime_error(
          "No cuda_tile::ModuleOp found in the input module");

    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    if (failed(cuda_tile::writeBytecode(
            ostream, cudaTileModule,
            cuda_tile::BytecodeVersion::kCurrentVersion)))
      throw std::runtime_error("Failed to write cuda_tile bytecode");
    py::bytes bytes(buffer.data(), buffer.size());
    return bytes;
  });
}
