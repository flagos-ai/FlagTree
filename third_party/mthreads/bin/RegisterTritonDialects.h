#pragma once
#ifndef TRITON_ENABLE_NVIDIA
#define TRITON_ENABLE_NVIDIA 1
#endif
#ifndef TRITON_ENABLE_MUSA
#define TRITON_ENABLE_MUSA 1
#endif

#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#ifdef __TLE__
#include "Dialect/MUSATLE/IR/Dialect.h"
#endif
#include "MTGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"

#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestBufferRegionPass();
void registerTestMembarPass();
void registerTestLoopPeelingPass();
namespace proton {
void registerTestScopeIdAllocationPass();
} // namespace proton
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::instrument::registerTritonInstrumentPasses();
  mlir::triton::gluon::registerGluonPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerRelayoutTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::registerLLVMDIScope();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerInlinerInterface(registry);
  mlir::registerLLVMDILocalVariable();
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);

  mlir::triton::registerTritonMUSAGPUToLLVMPasses();
  mlir::triton::registerMTGPUToLLVMPasses();
  mlir::registerTritonMUSAGPUPasses();

  // Plugin passes
  if (std::string filename =
          mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
      !filename.empty()) {

    TritonPlugin TP(filename);
    std::vector<const char *> passNames;
    if (auto result = TP.getPassHandles(passNames); !result)
      llvm::report_fatal_error(result.takeError());

    for (const char *passName : passNames)
      if (auto result = TP.registerPass(passName); !result)
        llvm::report_fatal_error(result.takeError());

    std::vector<const char *> dialectNames;
    if (auto result = TP.getDialectHandles(dialectNames); !result)
      llvm::report_fatal_error(result.takeError());

    for (unsigned i = 0; i < dialectNames.size(); ++i) {
      const char *dialectName = dialectNames.data()[i];
      auto result = TP.getDialectPluginInfo(dialectName);
      if (!result)
        llvm::report_fatal_error(result.takeError());
      ::mlir::DialectPluginLibraryInfo dialectPluginInfo = *result;
      dialectPluginInfo.registerDialectRegistryCallbacks(&registry);
    }
  }

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::triton::musa::MUSADialect, mlir::triton::mtgpu::MTGPUDialect,
#ifdef __TLE__
      mlir::triton::musa_tle::MUSATLEDialect,
#endif
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
      mlir::triton::gluon::GluonDialect>();
}
