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

#include "triton/Dialect/TritonXPU/Transforms/TritonXPUConversion.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include <cstdint>

using namespace mlir;

//
// TypeConverter
//
TritonXPUTypeConverter::TritonXPUTypeConverter(MLIRContext *context,
                                               uint32_t buffer_size,
                                               uint32_t core_num)
    : context(context), buffer_size(buffer_size), core_num(core_num) {

  addConversion([](Type type) { return type; });

  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    if (tensorType.getEncoding())
      return tensorType;

    ArrayRef<int64_t> shape = tensorType.getShape();
    triton::xpu::ClusterLayoutAttr encoding =
        triton::xpu::getDefaultClusterEncoding(
            this->context, shape, this->buffer_size, this->core_num);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  });

  // TODO[dyq]: check addConversion for triton::PointerType

  //
  // Materializations
  //
  // Note: addArgumentMaterialization was removed in newer MLIR. Argument
  // remats now go through addSourceMaterialization.
  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) -> Value {
    llvm_unreachable("Source rematerialization should not happen in Triton -> "
                     "TritonXPU Conversion");
    return Value();
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) -> Value {
    auto cast =
        builder.create<triton::xpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return cast.getResult();
  });
}

//
// TritonXPUConversion
//
TritonXPUConversionTarget::TritonXPUConversionTarget(
    MLIRContext &context, TritonXPUTypeConverter &typeConverter)
    : ConversionTarget(context) {

  addLegalDialect<triton::xpu::TritonXPUDialect>();

  // Some ops from SCF are illegal
  // TODO[dyq]: addIllegalOp necessary?
  //   addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
  //                scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
      hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }
    if (hasLegalRegions && typeConverter.isLegal(op)) {
      return true;
    }
    return false;
  });

  // TODO[dyq]: XPUSDNN-CHECK check addDynamicallyLegalDialect for triton::DotOp
}
