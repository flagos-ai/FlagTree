#pragma once

// --- Triton core dialects & passes ---
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#endif
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
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

// --- TsingMicro-specific includes ---
#include "Address/Dialect/IR/AddressDialect.h"
#include "Address/Transforms/Passes.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "magic-kernel/Conversion/TLEToMK/Passes.h"
#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h"
#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h"
#include "magic-kernel/Conversion/LinalgToMK/Passes.h"
#include "magic-kernel/Conversion/MKPipeline/Passes.h"
#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"
#include "magic-kernel/Transforms/Passes.h"
#include "third_party/tle/include/tle-dsa/Conversion/DsaToCore/DsaToCore.h"
#include "third_party/tle/include/tle-dsa/Dialect/IR/DsaDialect.h"
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h"
#include "triton-shared/Conversion/ReconcilePtrCasts/Passes.h"
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
#include "tsingmicro-tx81/Conversion/AllocateSharedMemory/Passes.h"
#include "tsingmicro-tx81/Conversion/ExportKernelSymbols/Passes.h"
#include "tsingmicro-tx81/Conversion/LinalgFusion/Passes.h"
#include "tsingmicro-tx81/Conversion/LinalgTiling/Passes.h"
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h"
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h"
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h"
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "tsingmicro-tx81/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
void registerTestTritonAMDGPURangeAnalysis();
void registerTestLoopPeelingPass();
namespace proton {
void registerTestScopeIdAllocationPass();
} // namespace proton
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
    // --- Core Triton passes ---
    mlir::registerAllPasses();
    mlir::triton::registerTritonPasses();
    mlir::triton::gpu::registerTritonGPUPasses();
    mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
    mlir::triton::instrument::registerTritonInstrumentPasses();
    mlir::triton::gluon::registerGluonPasses();
#ifdef __TLE__
    mlir::triton::tle::registerPasses();
#endif

    // Test passes
    mlir::test::registerTestAliasPass();
    mlir::test::registerTestAlignmentPass();
    mlir::test::registerTestAllocationPass();
    mlir::test::registerTestMembarPass();
    mlir::test::registerTestLoopPeelingPass();
    mlir::test::registerTestTritonAMDGPURangeAnalysis();

    // Core GPU passes
    mlir::triton::registerConvertTritonToTritonGPUPass();
    mlir::triton::registerRelayoutTritonGPUPass();
    mlir::triton::gpu::registerAllocateSharedMemoryPass();
    mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
    mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
    mlir::triton::registerConvertWarpSpecializeToLLVM();
    mlir::triton::registerConvertTritonGPUToLLVMPass();
    mlir::triton::registerConvertNVGPUToLLVMPass();
    mlir::triton::registerAllocateSharedMemoryNvPass();
    mlir::registerLLVMDIScope();
    mlir::LLVM::registerInlinerInterface(registry);
    mlir::NVVM::registerInlinerInterface(registry);
    mlir::registerLLVMDILocalVariable();

    // TritonAMDGPUToLLVM passes
    mlir::triton::registerAllocateAMDGPUSharedMemory();
    mlir::triton::registerConvertTritonAMDGPUToLLVM();
    mlir::triton::registerConvertBuiltinFuncToLLVM();
    mlir::triton::registerOptimizeAMDLDSUsage();

    mlir::ub::registerConvertUBToLLVMInterface(registry);
    mlir::registerConvertNVVMToLLVMInterface(registry);
    mlir::registerConvertMathToLLVMInterface(registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
    mlir::arith::registerConvertArithToLLVMInterface(registry);

    // TritonAMDGPUTransforms passes
    mlir::registerTritonAMDGPUAccelerateMatmul();
    mlir::registerTritonAMDGPUOptimizeEpilogue();
    mlir::registerTritonAMDGPUHoistLayoutConversions();
    mlir::registerTritonAMDGPUReorderInstructions();
    mlir::registerTritonAMDGPUBlockPingpong();
    mlir::registerTritonAMDGPUPipeline();
    mlir::registerTritonAMDGPUScheduleLoops();
    mlir::registerTritonAMDGPUCanonicalizePointers();
    mlir::registerTritonAMDGPUConvertToBufferOps();
    mlir::registerTritonAMDGPUInThreadTranspose();
    mlir::registerTritonAMDGPUCoalesceAsyncCopy();
    mlir::registerTritonAMDGPUUpdateAsyncWaitCount();
    mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
    mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();
    mlir::registerTritonAMDFoldTrueCmpI();
    mlir::triton::amdgpu::registerTritonAMDGPUOptimizeDotOperands();

    // NVWS passes
    mlir::triton::registerNVWSTransformsPasses();

    // NVGPU transform passes
    mlir::registerNVHopperTransformsPasses();

    // Proton passes
    mlir::test::proton::registerTestScopeIdAllocationPass();
    mlir::triton::proton::registerConvertProtonToProtonGPU();
    mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
    mlir::triton::proton::gpu::registerConvertProtonAMDGPUToLLVM();
    mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
    mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
    mlir::triton::proton::gpu::registerScheduleBufferStorePass();
    mlir::triton::proton::gpu::registerAddSchedBarriersPass();

    // --- TsingMicro-specific passes ---
    mlir::registerLinalgPasses();
    mlir::dsa::registerDsaMemoryToCorePass();
    mlir::triton::registerTLEToMKPass();

    // triton-shared passes
    mlir::triton::registerTritonToLinalgPass();
    mlir::triton::registerTritonToStructuredPass();
    mlir::triton::registerTritonToUnstructuredPass();
    mlir::triton::registerTritonArithToLinalgPasses();
    mlir::triton::registerStructuredToMemrefPasses();
    mlir::triton::registerUnstructuredToMemref();
    mlir::triton::registerTritonPtrToMemref();
    mlir::triton::registerTritonToCoreDialectsPass();
    mlir::triton::registerReconcilePtrCasts();

    // Core dialects to MK layer conversion passes
    mlir::triton::registerTx81MemrefToLLVMPass();
    mlir::triton::registerLinalgToMKPass();
    mlir::triton::registerCoreDialectsToMKPass();
    mlir::triton::registerLegalizeTensorFormLoopsPass();
    mlir::addr::registerAddrToLLVMPass();
    mlir::triton::registerLinalgTilingPass();
    mlir::triton::registerLinalgFusionPass();

    mlir::triton::registerMaterializeStridedLinalgInputsPass();
    // TsingMicro specific conversion passes
    mlir::triton::registerMKToTx81Pass();
    mlir::triton::alloc::registerAllocateSharedMemoryPass();
    mlir::triton::registerTx81ToLLVMPass();
    mlir::triton::registerExportKernelSymbols();
    mlir::triton::registerKernelArgBufferPass();
    mlir::triton::registerMKPipelinePass();
    mlir::triton::registerMKLoopBoundCanonicalizePass();

    // TsingMicroTx81Transforms passes
    mlir::triton::registerInsertBarrierPass();
    mlir::triton::registerTx81ResolveDmaBaseAddrPass();

    // Register all MLIR extensions (bufferization, etc.)
    mlir::registerAllExtensions(registry);

    // Register bufferizable op interface external models needed by
    // --one-shot-bufferize. These live in registerAllDialects() upstream
    // but are not covered by registerAllExtensions().
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerAllDialectInterfaceImplementations(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);
    mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::mk::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::math::MathDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
      mlir::triton::amdgpu::TritonAMDGPUDialect,
      mlir::triton::proton::ProtonDialect,
      mlir::triton::proton::gpu::ProtonGPUDialect, mlir::ROCDL::ROCDLDialect,
#ifdef __TLE__
      mlir::triton::tle::TleDialect,
#endif
      mlir::triton::gluon::GluonDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect, mlir::bufferization::BufferizationDialect,
      mlir::mk::MagicKernelDialect, mlir::tx::Tx81Dialect,
      mlir::addr::AddressDialect, mlir::dsa::DsaDialect>();
}
