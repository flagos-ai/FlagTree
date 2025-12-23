#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  addConversion([&](MemDescType type) -> std::optional<Type> {
    return convertMemDescType(type);
  });
  addConversion([&](CorexDescType type) -> std::optional<Type> {
    return convertCorexDescType(type);
  });
  addConversion([&](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
    return convertAsyncToken(type);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
}

Type TritonGPUToLLVMTypeConverter::getElementTypeForStruct(
    TensorOrMemDesc type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  auto dotOpLayout = mlir::dyn_cast<DotOperandEncodingAttr>(layout);
  if (!dotOpLayout)
    return elemTy;
  if (auto iluvatarmmaParent =
          mlir::dyn_cast<IluvatarMmaEncodingAttr>(dotOpLayout.getParent())) {
    if (iluvatarmmaParent.isVolta()) {
      int bitwidth = elemTy.getIntOrFloatBitWidth();
      if (bitwidth == 8)
        return vec_ty(elemTy, 8);
      return vec_ty(elemTy, 4);
    }
  }
  int bitwidth = elemTy.getIntOrFloatBitWidth();
  assert(bitwidth <= 32);
  return IntegerType::get(ctx, 32);
}

Type TritonGPUToLLVMTypeConverter::convertCorexDescType(CorexDescType type) {
  auto ctx = type.getContext();
  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  auto ptrType = LLVM::LLVMPointerType::get(ctx, 1);
  Type elementType;
  if (auto pointTy = dyn_cast<PointerType>(type.getElementType()))
    elementType = convertType(pointTy.getPointeeType());
  else
    elementType = convertType(type.getElementType());
  SmallVector<Type, 4> types(numElementsPerThread, ptrType);
  for (auto i = 0; i < numElementsPerThread; i++)
    types.push_back(IntegerType::get(ctx, 1));
  for (auto i = 0; i < numElementsPerThread; i++)
    types.push_back(elementType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
