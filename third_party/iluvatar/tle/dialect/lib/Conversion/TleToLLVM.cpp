#include "Conversion/TleToLLVM.h"

#include "Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"
#include "Conversion/TleToLLVM/ExtractOpToLLVM.h"
#include "Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
#include "Conversion/TleToLLVM/PackOpToLLVM.h"

#ifdef FLAGCX_ENABLED
#include "Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxOpToLLVM.h"
#endif

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
  mlir::triton::iluvatar_tle::populateRemotePointersOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, benefit);
  mlir::triton::iluvatar_tle::populateDistributedBarrierOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  // TLE-Raw: the descriptor plumbing that survives dsl_region inlining.
  // `dsl_region` / `yield` themselves are gone by this point, see
  // TritonIluvatarTleDSLRegionInline.
  mlir::triton::iluvatar_tle::populateExtractOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  mlir::triton::iluvatar_tle::populatePackOpToLLVMPatterns(typeConverter,
                                                           patterns, benefit);
  // FlagCX ops are lowered to LLVM.
#ifdef FLAGCX_ENABLED
  mlir::triton::iluvatar_tle::populateFlagCxOpToLLVMPatterns(typeConverter,
                                                             patterns, benefit);
#endif
}

} // namespace mlir::triton::iluvatar_tle
