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

//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonXPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONXPU_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONXPU_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h" // TypeConverter

namespace mlir {

class TritonXPUTypeConverter : public TypeConverter {
public:
  TritonXPUTypeConverter(MLIRContext *context, uint32_t buffer_size,
                         uint32_t core_num);
  uint32_t getBufferSize() const { return buffer_size; }
  uint32_t getCoreNum() const { return core_num; }

private:
  MLIRContext *context;
  uint32_t buffer_size;
  uint32_t core_num;
};

class TritonXPUConversionTarget : public ConversionTarget {
public:
  explicit TritonXPUConversionTarget(MLIRContext &ctx,
                                     TritonXPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONXPU_TRANSFORMS_TRITONGPUCONVERSION_H_
