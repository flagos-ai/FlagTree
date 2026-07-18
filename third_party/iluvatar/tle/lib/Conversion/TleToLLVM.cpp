#include "Conversion/TleToLLVM.h"

#include "Conversion/TleToLLVM/LocalPointersOpToLLVM.h"

namespace mlir::triton::iluvatar_tle {

void populateTleToLLVMPatterns(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo,
                               RewritePatternSet &patterns,
                               PatternBenefit benefit) {
  mlir::triton::iluvatar_tle::populateExtractTileOpToLLVMPatterns(
      typeConverter, patterns, targetInfo, benefit);
  mlir::triton::iluvatar_tle::populateInsertTileOpToLLVMPatterns(
      typeConverter, patterns, targetInfo, benefit);
  mlir::triton::iluvatar_tle::populateLocalPointersOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
}

} // namespace mlir::triton::iluvatar_tle
