#pragma once
#ifdef __AMD__
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#endif
#ifdef __NVIDIA__
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#endif
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
#ifdef __TLE__
#include "third_party/tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/IR/Dialect.h" // flagtree tle raw
#endif
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#ifdef __AMD__
// Below headers will allow registration to ROCm passes
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"
#endif

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#ifdef __NVIDIA__
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
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
#ifdef __AMD__
void registerAMDTestAlignmentPass();
#endif
void registerTestAllocationPass();
void registerTestMembarPass();
#ifdef __AMD__
void registerTestAMDGPUMembarPass();
#endif
void registerTestTritonAMDGPURangeAnalysis();
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
#ifdef __NVIDIA__
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
#endif
  mlir::triton::instrument::registerTritonInstrumentPasses();
  mlir::triton::gluon::registerGluonPasses();
#ifdef __TLE__
  mlir::triton::tle::registerPasses(); // flagtree tle
#endif
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
#ifdef __AMD__
  mlir::test::registerAMDTestAlignmentPass();
#endif
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::test::registerTestLoopPeelingPass();
#ifdef __AMD__
  mlir::test::registerTestAMDGPUMembarPass();
  mlir::test::registerTestTritonAMDGPURangeAnalysis();
#endif
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

#ifdef __AMD__
  // TritonAMDGPUToLLVM passes
  mlir::triton::registerAllocateAMDGPUSharedMemory();
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerOptimizeAMDLDSUsage();
#endif

#ifdef __AMD__
  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUHoistLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUStreamPipeline();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::registerTritonAMDGPUInThreadTranspose();
  mlir::registerTritonAMDGPUCoalesceAsyncCopy();
  mlir::registerTritonAMDGPUUpdateAsyncWaitCount();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();
  mlir::registerTritonAMDFoldTrueCmpI();
#endif

#ifdef __NVIDIA__
  // NVWS passes
  mlir::triton::registerNVWSTransformsPasses();

  // NVGPU transform passes
  mlir::registerNVHopperTransformsPasses();
#endif

  // Proton passes
  mlir::test::proton::registerTestScopeIdAllocationPass();
  mlir::triton::proton::registerConvertProtonToProtonGPU();
  mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
  mlir::triton::proton::gpu::registerConvertProtonAMDGPUToLLVM();
  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
  mlir::triton::proton::gpu::registerScheduleBufferStorePass();
  mlir::triton::proton::gpu::registerAddSchedBarriersPass();

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
#ifdef __NVIDIA__
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
#endif
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
#ifdef __NVIDIA__
      mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
#endif
#ifdef __AMD__
      mlir::triton::amdgpu::TritonAMDGPUDialect,
#endif
      mlir::triton::proton::ProtonDialect,
      mlir::triton::proton::gpu::ProtonGPUDialect,
#ifdef __AMD__
      mlir::ROCDL::ROCDLDialect,
#endif
      mlir::triton::gluon::GluonDialect,
#ifdef __TLE__
      mlir::triton::tle::TleDialect // flagtree tle raw
#endif
      >();
}
