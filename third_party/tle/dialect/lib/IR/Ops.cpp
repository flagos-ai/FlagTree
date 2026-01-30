#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::tle {

namespace {
// Triton shared-memory pointers map to LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
} // namespace

LogicalResult DSLRegionOp::verify() {
  Region &body = getBody();
  const uint32_t numArguments = body.getNumArguments(),
                 numOperands = getNumOperands();
  if (numArguments != numOperands) {
    return emitOpError() << "expects number of operands (" << numArguments
                         << ") to match number of region arguments ("
                         << numOperands << ")";
  }
  for (auto [arg, operand] : llvm::zip(body.getArguments(), getOperands())) {
    if (arg.getType() != operand.getType()) {
      return emitOpError() << "expects region argument type (" << arg.getType()
                           << ") to match operand type (" << operand.getType()
                           << ")";
    }
  }
  return success();
}

void ExtractSizesOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, size_t num,
                           Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

void ExtractStridesOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState, size_t num,
                             Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

void LocalPointersOp::build(::mlir::OpBuilder &odsBuilder,
                            ::mlir::OperationState &odsState, Type resultTy,
                            Value src) {
  build(odsBuilder, odsState, resultTy, src, Value());
}

LogicalResult PackOp::verify() {
  TypedValue<LLVM::LLVMStructType> input = getInput();
  ArrayRef<Type> body = input.getType().getBody();
  if (body.size() < 3 || body.size() % 2 != 1 ||
      !isa<LLVM::LLVMPointerType>(body[0]) ||
      !isa<LLVM::LLVMPointerType>(body[1])) {
    return emitOpError() << "expects input struct to have at least 3 elements, "
                            "with the first two being pointer types.";
  }
  return success();
}

LogicalResult LocalPointersOp::verify() {
  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(getSrc().getType());
  if (!memDescTy)
    return emitOpError() << "expects src operand to be a ttg.memdesc";

  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultTy)
    return emitOpError() << "expects result to be a ranked tensor";

  auto ptrTy = dyn_cast<triton::PointerType>(resultTy.getElementType());
  if (!ptrTy)
    return emitOpError() << "expects result element type to be tt.ptr";

  if (ptrTy.getPointeeType() != memDescTy.getElementType())
    return emitOpError()
           << "expects pointer pointee type " << ptrTy.getPointeeType()
           << " to match memdesc element type "
           << memDescTy.getElementType();

  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  auto resultShape = resultTy.getShape();
  Attribute resultEncoding = resultTy.getEncoding();

  if (Value offsets = getOffsets()) {
    auto tensorTy = dyn_cast<RankedTensorType>(offsets.getType());
    if (!tensorTy)
      return emitOpError() << "expects offsets to be a ranked tensor";
    if (!tensorTy.getElementType().isInteger(32))
      return emitOpError() << "expects offsets tensor to have i32 element type";
    auto tensorShape = tensorTy.getShape();
    if (tensorShape.size() != memDescTy.getShape().size())
      return emitOpError() << "expects offsets tensor rank to match buffer rank";
    if (tensorShape != resultShape)
      return emitOpError()
             << "expects offsets tensor shape to match result shape";
  }

  return success();
}

} // namespace mlir::triton::tle
