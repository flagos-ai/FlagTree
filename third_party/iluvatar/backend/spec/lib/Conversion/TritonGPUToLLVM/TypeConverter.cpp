#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

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
  addConversion([&](triton::gpu::AsyncTokenType type) -> std::optional<Type> {
    return convertAsyncToken(type);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
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
  auto mmaParent =
      mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpLayout.getParent());
  if (!mmaParent || mmaParent.isHopper())
    return elemTy;
  int bitwidth = elemTy.getIntOrFloatBitWidth();
  assert(bitwidth <= 32);
  return IntegerType::get(ctx, 32);
}
