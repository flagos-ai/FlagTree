// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PATTERNTRITONHCUGPUTOLLVM_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PATTERNTRITONHCUGPUTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::HCU {

void populateExtractSliceOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit);

void populateInThreadTransposeOpToTTGPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::PatternBenefit benefit);
void populateConcatOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit);

void populateScaledUpcastOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit);

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_PATTERNTRITONHCUGPUTOLLVM_H_
