//===------------------- Tx81MemrefToLLVM.h -------------------------*- C++
//-*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering memref.copy, memref.alloc to mk.load, mk.alloc etc.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_MEMREF_TO_MK_H
#define ZTC_CONVERSION_MEMREF_TO_MK_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h.inc"

void populateTx81MemrefToLLVMCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTx81MemrefToLLVMConversionPatterns(RewritePatternSet &patterns,
                                                LLVMTypeConverter &converter);

std::unique_ptr<OperationPass<ModuleOp>> createTx81MemrefToLLVMPass();

} // namespace triton
} // namespace mlir

#endif // ZTC_CONVERSION_MEMREF_TO_MAGICKERNEL_H