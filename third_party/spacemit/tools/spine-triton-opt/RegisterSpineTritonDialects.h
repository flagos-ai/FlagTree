#pragma once
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Conversion/AddTargetDescription/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TLEToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h"
#include "triton-shared/Dialect/TLE/IR/TLEDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h"

#include "mlir/InitAllPasses.h"

inline void registerSpineTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerLinalgPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToLinalgExperimentalPasses();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerReconcilePtrCasts();
  mlir::triton::registerReconcileLlvmPtrCasts();
  mlir::triton::registerTritonToPtr();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonToUnstructuredPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerStructuredToMemrefPasses();
  mlir::triton::registerAddTargetDescriptionPasses();
  mlir::triton::registerScfbufferStandardized();
  mlir::triton::registerConvertScanOp();
  mlir::triton::registerXSMTToLinalgPass();
  mlir::triton::registerTLEToLinalgPass();
  mlir::triton::registerAddLLVMDebugInfoPass();

  // TODO: register Triton & TritonGPU passes
  registry.insert<
      mlir::LLVM::LLVMDialect, mlir::ptr::PtrDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::linalg::LinalgDialect,
      mlir::func::FuncDialect, mlir::tensor::TensorDialect,
      mlir::memref::MemRefDialect, mlir::bufferization::BufferizationDialect,
      mlir::DLTIDialect, mlir::vector::VectorDialect, mlir::xsmt::XSMTDialect,
      mlir::xsmt_async::XSMTAsyncDialect, mlir::tle::TLEDialect,
      mlir::triton::proton::ProtonDialect>();
}
