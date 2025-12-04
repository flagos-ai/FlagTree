#pragma once
#include "Address/Dialect/IR/AddressDialect.h"
#include "Address/Transforms/Passes.h"
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

// Below headers will allow registration to ROCm passes
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToCoreDialects/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "tsingmicro-tx81/Conversion/LinalgTiling/Passes.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"

#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h"
#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h"
#include "magic-kernel/Conversion/LinalgToMK/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h"
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h"
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h"
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h"

#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {
void registerTestMathPolynomialApproximationPass();
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
void registerTestTritonAMDGPURangeAnalysis();
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::registerLinalgPasses();
  mlir::registerTritonNvidiaGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::test::registerTestTritonAMDGPURangeAnalysis();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonToUnstructuredPass();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerStructuredToMemrefPasses();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerTritonPtrToAddress();
  mlir::triton::registerReconcilePtrCasts();
  mlir::triton::registerTritonToCoreDialectsPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::registerLLVMDIScope();

  // Core dialects to MK layer conversion passes
  mlir::triton::registerTx81MemrefToLLVMPass();
  mlir::triton::registerLinalgToMKPass();
  mlir::triton::registerCoreDialectsToMKPass();
  mlir::triton::registerLegalizeTensorFormLoopsPass();
  mlir::addr::registerAddrToLLVMPass();
  mlir::triton::registerLinalgTilingPass();

  // TsingMicro specific conversion passes
  mlir::triton::registerMKToTx81Pass();
  mlir::triton::registerTx81ToLLVMPass();
  mlir::triton::registerKernelArgBufferPass();

  // TritonAMDGPUToLLVM passes
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerDecomposeUnsupportedAMDConversions();
  mlir::triton::registerOptimizeAMDLDSUsage();

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUHoistLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUStreamPipeline();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();

  // Math dialect passes
  mlir::test::registerTestMathPolynomialApproximationPass();

  // FIXME: May not need all of these
  // mlir::registerAllDialects(registry);
  // Register all external models.
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::arith::registerShardingInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::builtin::registerCastOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerInlinerInterface(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);
  mlir::linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::memref::registerMemorySlotExternalModels(registry);

  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::shape::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerFindPayloadReplacementOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);

  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
  mlir::vector::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);

  // This is need for the Bufferization pass(one-shot bufferization)
  mlir::registerAllExtensions(registry);
  mlir::mk::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect,
      mlir::triton::amdgpu::TritonAMDGPUDialect,
      mlir::triton::proton::ProtonDialect, mlir::ROCDL::ROCDLDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect, mlir::bufferization::BufferizationDialect,
      mlir::mk::MagicKernelDialect, mlir::tx::Tx81Dialect,
      mlir::addr::AddressDialect>();
}
