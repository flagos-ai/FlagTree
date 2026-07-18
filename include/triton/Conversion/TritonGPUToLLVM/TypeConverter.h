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

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToLLVMTypeConverter(MLIRContext *ctx,
                               const LowerToLLVMOptions &option,
                               const TargetInfoBase &targetInfo,
                               const DataLayoutAnalysis *analysis = nullptr);
  TritonGPUToLLVMTypeConverter(MLIRContext *ctx,
                               const TargetInfoBase &targetInfo,
                               const DataLayoutAnalysis *analysis = nullptr);

  Type convertTritonTensorType(RankedTensorType type,
                               const TargetInfoBase &targetInfo);
  Type convertMemDescType(triton::gpu::MemDescType type,
                          const TargetInfoBase &targetInfo);
  Type convertAsyncTokenType(triton::gpu::AsyncTokenType type);

  template <typename... T> void convertFP8Type() {
    (addConversion([&](T type) -> std::optional<Type> {
       return IntegerType::get(type.getContext(), 8);
     }),
     ...);
  }
};

#endif
