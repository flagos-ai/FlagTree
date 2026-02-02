#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
    return emitOpError() << "expects pointer pointee type "
                         << ptrTy.getPointeeType()
                         << " to match memdesc element type "
                         << memDescTy.getElementType();

  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  auto resultShape = resultTy.getShape();
  Attribute resultEncoding = resultTy.getEncoding();

  auto &region = getIndices();
  if (region.empty())
    return emitOpError() << "expects a non-empty indices region";

  if (!region.hasOneBlock())
    return emitOpError() << "expects indices region to have a single block";

  auto &block = region.front();
  if (block.getNumArguments() != resultShape.size())
    return emitOpError() << "expects indices region args to match result rank";

  for (BlockArgument arg : block.getArguments()) {
    if (!arg.getType().isInteger())
      return emitOpError() << "expects indices region args to be integer";
  }

  auto *terminator = block.getTerminator();
  auto yield = dyn_cast<LocalPointersReturnOp>(terminator);
  if (!yield)
    return emitOpError() << "expects indices region to terminate with "
                            "tle.local_pointers.return";

  if (yield.getNumOperands() != memDescTy.getShape().size())
    return emitOpError()
           << "expects indices return to match buffer rank";

  for (Value val : yield.getOperands()) {
    if (!val.getType().isInteger())
      return emitOpError()
             << "expects indices return values to be integer";
  }

  return success();
}

} // namespace mlir::triton::tle
