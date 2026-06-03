#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// clang-format off
#include "Dialect/MUSATLE/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::musa_tle {

void MUSATLEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/MUSATLE/IR/Ops.cpp.inc"
      >();
}

} // namespace mlir::triton::musa_tle

#define GET_OP_CLASSES
#include "Dialect/MUSATLE/IR/Ops.cpp.inc"

namespace mlir::triton::musa_tle {
namespace {
constexpr int kSharedMemoryAddressSpace = 3;

static bool isRank0BackingMemDesc(ttg::MemDescType memDescTy) {
  return memDescTy.getShape().size() == 1 && memDescTy.getShape().front() == 1;
}
} // namespace

LogicalResult LocalPointersOp::verify() {
  auto memDescTy = dyn_cast<ttg::MemDescType>(getSrc().getType());
  if (!memDescTy)
    return emitOpError() << "expects src operand to be a ttg.memdesc";
  if (!isa<ttg::SharedMemorySpaceAttr>(memDescTy.getMemorySpace()))
    return emitOpError() << "expects src memdesc to live in shared memory";
  if (!isa<ttg::SharedEncodingTrait>(memDescTy.getEncoding()))
    return emitOpError() << "expects src memdesc to use a shared encoding";

  auto resultTensorTy = dyn_cast<RankedTensorType>(getResult().getType());
  auto resultPtrTy = dyn_cast<triton::PointerType>(getResult().getType());
  if (!resultTensorTy && !resultPtrTy)
    return emitOpError()
           << "expects result to be either tensor<tt.ptr<...>> or tt.ptr";

  auto ptrTy =
      resultTensorTy
          ? dyn_cast<triton::PointerType>(resultTensorTy.getElementType())
          : resultPtrTy;
  if (!ptrTy)
    return emitOpError() << "expects result element type to be tt.ptr";

  if (ptrTy.getPointeeType() != memDescTy.getElementType())
    return emitOpError() << "expects pointer pointee type "
                         << ptrTy.getPointeeType()
                         << " to match memdesc element type "
                         << memDescTy.getElementType();

  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  auto indices = getIndices();
  if (indices.empty()) {
    if (resultTensorTy) {
      if (resultTensorTy.getShape() != memDescTy.getShape())
        return emitOpError()
               << "zero-index local_pointers expects tensor result shape to "
                  "match buffer shape";
      return success();
    }
    if (!memDescTy.getShape().empty() && !isRank0BackingMemDesc(memDescTy))
      return emitOpError()
             << "zero-index scalar local_pointers is only valid for rank-0 "
                "buffers";
    return success();
  }

  if (indices.size() != memDescTy.getShape().size())
    return emitOpError() << "expects indices count to match buffer rank";

  if (resultTensorTy) {
    auto resultShape = resultTensorTy.getShape();
    Attribute resultEncoding = resultTensorTy.getEncoding();

    ArrayRef<int64_t> indexShape;
    for (Value val : indices) {
      auto indexTy = dyn_cast<RankedTensorType>(val.getType());
      if (!indexTy)
        return emitOpError()
               << "tensor result expects indices to be ranked tensors";
      if (!indexTy.getElementType().isInteger())
        return emitOpError() << "expects indices return tensors to have "
                                "integer element types";
      if (indexShape.empty())
        indexShape = indexTy.getShape();
      else if (indexTy.getShape() != indexShape)
        return emitOpError()
               << "expects indices return tensors to have identical shapes";
      if (resultEncoding && indexTy.getEncoding() &&
          resultEncoding != indexTy.getEncoding())
        return emitOpError()
               << "expects indices return tensors to match result encoding";
    }

    if (indexShape != resultShape)
      return emitOpError()
             << "expects indices return tensor shape to match result shape";
    return success();
  }

  for (Value val : indices) {
    if (auto indexTy = dyn_cast<IntegerType>(val.getType())) {
      if (!indexTy.isSignlessInteger())
        return emitOpError()
               << "expects scalar indices to be signless integers";
      continue;
    }
    return emitOpError() << "scalar result expects scalar integer indices";
  }

  return success();
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
