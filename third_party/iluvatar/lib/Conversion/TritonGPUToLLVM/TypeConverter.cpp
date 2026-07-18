// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MemDescType;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, const TargetInfoBase &targetInfo,
    const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, LowerToLLVMOptions(ctx), targetInfo,
                                   analysis) {}

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, const LowerToLLVMOptions &options,
    const TargetInfoBase &targetInfo, const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, options, analysis) {
  addConversion([ctx](triton::PointerType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
  });
  addConversion([ctx](TensorDescType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, 0);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type, targetInfo);
  });
  addConversion([&](MemDescType type) -> std::optional<Type> {
    return convertMemDescType(type, targetInfo);
  });
  addConversion([&](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
    return convertAsyncTokenType(type);
  });

  convertFP8Type<mlir::Float8E4M3FNUZType, mlir::Float8E4M3FNType,
                 mlir::Float8E5M2Type, mlir::Float8E5M2FNUZType>();
}

Type TritonGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  Type eltType = convertType(type.getElementType());
  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertMemDescType(
    MemDescType type, const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  // base ptr
  auto ptrType = LLVM::LLVMPointerType::get(
      ctx, targetInfo.getAddressSpace(type.getMemorySpace()));

  if (isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
          triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
          type.getEncoding())) {
    return ptrType;
  }

  SmallVector<Type, 4> types;
  types.push_back(ptrType);
  auto rank = type.getRank();
  // offsets
  for (auto i = 0; i < rank; i++) {
    types.push_back(IntegerType::get(ctx, 32));
  }
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type TritonGPUToLLVMTypeConverter::convertAsyncTokenType(
    triton::gpu::AsyncTokenType type) {
  return IntegerType::get(type.getContext(), 32);
}
