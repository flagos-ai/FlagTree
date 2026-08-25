/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxBarrierOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/GetWorldRankOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/GetDeviceIdToFlagCX.h"

namespace mlir::triton::tle {
void populateFlagCxOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
#ifdef FLAGCX_ENABLED
  mlir::triton::tle::populateGetDeviceIdOpToFlagCxPatterns(typeConverter,
                                                           patterns, benefit);
  mlir::triton::tle::populateGetLocalRankOpToLLVMPatterns(typeConverter,
                                                          patterns, benefit);
  mlir::triton::tle::populateGetNumPesOpToLLVMPatterns(typeConverter, patterns,
                                                       benefit);
  mlir::triton::tle::populateGetWorldRankOpToLLVMPatterns(typeConverter,
                                                          patterns, benefit);
  mlir::triton::tle::populateFlagCxBarrierOpToLLVMPatterns(typeConverter,
                                                           patterns, benefit);
#endif
}

} // namespace mlir::triton::tle
