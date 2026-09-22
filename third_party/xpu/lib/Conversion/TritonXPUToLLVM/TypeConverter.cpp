#include "triton/Conversion/TritonXPUToLLVM/TypeConverter.h" // TritonXPUToLLVMTypeConverter
#include "triton/Dialect/TritonGPU/IR/Types.h"               // MemDescType

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::xpu::getTotalElemsPerThread;

TritonXPUToLLVMTypeConverter::TritonXPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  // deal RankedTensorType to calculate elemNum
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  // TLE: TensorDescType → GM pointer (addrspace 1, 64-bit on XPU).
  // The frontend already decomposes each TensorDesc arg into
  // (tensordesc, offset0:i32, offset1:i32, stride0:i64, stride1:i64).
  // The tensordesc part is just the base GM pointer.
  addConversion([ctx](triton::TensorDescType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, 1);
  });
  // TLE: MemDescType → opaque pointer (LM base pointer)
  addConversion([ctx](triton::gpu::MemDescType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(ctx, 0);
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

Type TritonXPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  // On XPU, addrspace 3 (shared/local memory) is not supported by the target;
  // local memory lives in the flat address space (0).  Map addrspace 3 → 0.
  unsigned addrSpace = type.getAddressSpace();
  if (addrSpace == 3)
    addrSpace = 0;
  if (isa<RankedTensorType>(pointeeType)) {
    auto rankedTensorType = cast<RankedTensorType>(pointeeType);
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto eleType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();
    SmallVector<Type, 4> types;
    // offsets
    for (size_t i = 0; i < shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 32));
    // shapes, strides
    for (size_t i = 0; i < 2 * shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 64));

    types.push_back(LLVM::LLVMPointerType::get(ctx, addrSpace));

    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx, addrSpace);
}

Type TritonXPUToLLVMTypeConverter::getElementTypeForStruct(
    triton::gpu::TensorOrMemDesc type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  // If the element type is a pointer in addrspace 3, remap to addrspace 0
  // (XPU has no addrspace 3 — local memory uses the flat address space).
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(elemTy)) {
    if (ptrTy.getAddressSpace() == 3)
      elemTy = LLVM::LLVMPointerType::get(ctx, 0);
  }
  return elemTy;
}

Type TritonXPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType =
      getElementTypeForStruct(cast<triton::gpu::TensorOrMemDesc>(type));

  unsigned numElementsPerThread;
  if (!layout) {
    // Unencoded tensors in TLE path (e.g. tt.load results, tle_local_ptr
    // results). Compute numElems as if they had the default ClusterLayout,
    // since the lowering patterns will produce/consume that many elements.
    auto defaultEncoding =
        triton::xpu::getDefaultClusterEncoding(ctx, shape, /*buffer_size=*/128,
                                               /*core_num=*/64);
    auto encodedType =
        RankedTensorType::get(shape, type.getElementType(), defaultEncoding);
    numElementsPerThread = getTotalElemsPerThread(encodedType);
  } else {
    numElementsPerThread = getTotalElemsPerThread(type);
  }
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
