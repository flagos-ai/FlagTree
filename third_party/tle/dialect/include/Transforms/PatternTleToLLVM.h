#pragma once

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace mlir::triton::tle {

/// Populate patterns to convert tle.extract_tile to LLVM
void populateExtractTileOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns,
                                         unsigned benefit = 1);

void populateInsertTileOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns,
                                        unsigned benefit = 1);

} // namespace mlir::triton::tle
