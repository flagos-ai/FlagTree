#pragma once
#include "hcu/include/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "hcu/include/TritonHCUGPUTransforms/Passes.h"
#ifdef __NVIDIA__
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#endif
#ifdef __PROTON__
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonHCUGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
#endif
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h" // flagtree tle raw
#include "tle/dialect/include/Transforms/Passes.h"
#endif
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef __PROTON__
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#endif
#ifdef __NVIDIA__
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#endif

// Below headers will allow registration to ROCm passes
#include "TritonHCUGPUToLLVM/Passes.h"
#include "TritonHCUGPUTransforms/Passes.h"
#include "TritonHCUGPUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#ifdef __PROTON__
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#endif
#ifdef __NVIDIA__
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#endif
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

namespace mlir {
namespace test {
#ifndef __HCU__
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
void registerTestLoopPeelingPass();
#endif
void registerHCUTestAlignmentPass();
void registerTestHCUGPUMembarPass();
void registerTestTritonHCUGPURangeAnalysis();
#ifdef __PROTON__
namespace proton {
void registerTestScopeIdAllocationPass();
} // namespace proton
#endif
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
#ifdef __NVIDIA__
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
#endif
#ifdef __PROTON__
  mlir::triton::instrument::registerTritonInstrumentPasses();
#endif
  mlir::triton::gluon::registerGluonPasses();
#ifdef __TLE__
  mlir::triton::tle::registerPasses(); // flagtree tle
#endif
#ifndef __HCU__
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::test::registerTestLoopPeelingPass();
#endif
  mlir::test::registerHCUTestAlignmentPass();
  mlir::test::registerTestHCUGPUMembarPass();
  mlir::test::registerTestTritonHCUGPURangeAnalysis();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerRelayoutTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
#ifdef __NVIDIA__
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerAllocateSharedMemoryNvPass();
#endif
  mlir::registerLLVMDIScope();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerInlinerInterface(registry);
  mlir::registerLLVMDILocalVariable();

  // TritonHCUGPUToLLVM passes
  mlir::triton::registerAllocateHCUGPUSharedMemory();
  mlir::triton::registerConvertTritonHCUGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerOptimizeHCULDSUsage();

  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);

  // TritonHCUGPUTransforms passes
  mlir::registerTritonHCUGPUAccelerateMatmul();
  mlir::registerTritonHCUGPUOptimizeEpilogue();
  mlir::registerTritonHCUGPUHoistLayoutConversions();
  mlir::registerTritonHCUGPUReorderInstructions();
  mlir::registerTritonHCUGPUBlockPingpong();
  mlir::registerTritonHCUGPUPipeline();
  mlir::registerTritonHCUGPUScheduleLoops();
  mlir::registerTritonHCUGPUCanonicalizePointers();
  mlir::registerTritonHCUGPUConvertToBufferOps();
  mlir::registerTritonHCUGPUInThreadTranspose();
  mlir::registerTritonHCUGPUCoalesceAsyncCopy();
  mlir::registerTritonHCUGPUUpdateAsyncWaitCount();
  mlir::triton::registerTritonHCUGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonHCUGPULowerInstructionSchedHints();
  mlir::registerTritonHCUFoldTrueCmpI();
  mlir::triton::hcugpu::registerTritonHCUGPUOptimizeDotOperands();

#ifdef __NVIDIA__
  // NVWS passes
  mlir::triton::registerNVWSTransformsPasses();

  // NVGPU transform passes
  mlir::registerNVHopperTransformsPasses();
#endif

  // Proton passes
#ifdef __PROTON__
  mlir::test::proton::registerTestScopeIdAllocationPass();
  mlir::triton::proton::registerConvertProtonToProtonGPU();
  mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
  mlir::triton::proton::gpu::registerConvertProtonHCUGPUToLLVM();
  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
  mlir::triton::proton::gpu::registerScheduleBufferStorePass();
  mlir::triton::proton::gpu::registerAddSchedBarriersPass();
#endif

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
#ifdef __NVIDIA__
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
#endif
      mlir::triton::gpu::TritonGPUDialect,
#ifdef __PROTON__
      mlir::triton::instrument::TritonInstrumentDialect,
#endif
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
#ifdef __NVIDIA__
      mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
#endif
      mlir::triton::hcugpu::TritonHCUGPUDialect,
#ifdef __PROTON__
      mlir::triton::proton::ProtonDialect,
      mlir::triton::proton::gpu::ProtonGPUDialect,
#endif
      mlir::ROCDL::ROCDLDialect,
#ifdef __TLE__
      mlir::triton::tle::TleDialect, // flagtree tle raw
#endif
      mlir::triton::gluon::GluonDialect>();
}
