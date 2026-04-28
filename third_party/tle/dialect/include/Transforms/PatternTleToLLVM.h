#pragma once
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace mlir::triton::tle {

/// Populate patterns to convert tle.extract_tile to LLVM
void populateExtractTileOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    const mlir::triton::TargetInfoBase &targetInfo, unsigned benefit = 1);

void populateInsertTileOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    const mlir::triton::TargetInfoBase &targetInfo, unsigned benefit = 1);

void populateMemDescWGMMAViewOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    unsigned benefit = 1);

void populateWGMMASharedOperandFenceOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    unsigned benefit = 1);

} // namespace mlir::triton::tle
