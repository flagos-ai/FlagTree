#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <limits>
#include <optional>

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

static SmallVector<int64_t> getDenseI64Array(Attribute attr) {
  SmallVector<int64_t> values;
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attr))
    llvm::append_range(values, dense.asArrayRef());
  return values;
}

static LogicalResult verifyTileShape(Operation *op, ArrayRef<int64_t> srcShape,
                                     ArrayRef<int64_t> tileShape,
                                     StringRef attrName) {
  if (tileShape.size() != srcShape.size())
    return op->emitOpError() << attrName << " rank must match source rank";
  for (size_t idx = 0; idx < srcShape.size(); ++idx) {
    int64_t srcDim = srcShape[idx];
    int64_t tileDim = tileShape[idx];
    if (tileDim <= 0)
      return op->emitOpError()
             << attrName << " must be positive at dimension " << idx;
    if (srcDim % tileDim != 0)
      return op->emitOpError()
             << "source shape must be divisible by " << attrName
             << " at dimension " << idx << " (source=" << srcDim
             << ", tile=" << tileDim << ")";
  }
  return success();
}

static std::optional<int64_t> getConstantIndex(Value index) {
  auto constOp = index.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

static LogicalResult verifyStaticTileIndex(Operation *op, Value index,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> tileShape) {
  std::optional<int64_t> staticIndex = getConstantIndex(index);
  if (!staticIndex)
    return success();

  int64_t totalTiles = 1;
  for (auto [srcDim, tileDim] : llvm::zip_equal(srcShape, tileShape))
    totalTiles *= srcDim / tileDim;
  if (*staticIndex < 0 || *staticIndex >= totalTiles)
    return op->emitOpError("index out of bounds for tile grid: index=")
           << *staticIndex << ", total_tiles=" << totalTiles;
  return success();
}
} // namespace

void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto resultTy = RankedTensorType::get(tileShape, srcTy.getElementType(),
                                        srcTy.getEncoding());
  state.addOperands({src, index});
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultTy);
}

LogicalResult ExtractTileOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !resultTy)
    return emitOpError("expects source and result to be ranked tensors");

  SmallVector<int64_t> tileShape =
      getDenseI64Array(getOperation()->getAttr("tile_shape"));
  if (failed(verifyTileShape(getOperation(), srcTy.getShape(), tileShape,
                             "tile_shape")))
    return failure();
  if (srcTy.getElementType() != resultTy.getElementType())
    return emitOpError("result element type must match source element type");
  if (srcTy.getRank() != resultTy.getRank())
    return emitOpError("result rank must match source rank");
  if (resultTy.getShape() != ArrayRef<int64_t>(tileShape))
    return emitOpError("result shape must equal tile_shape");
  if (srcTy.getEncoding() && resultTy.getEncoding() &&
      srcTy.getEncoding() != resultTy.getEncoding())
    return emitOpError("result encoding must match source encoding");
  return verifyStaticTileIndex(getOperation(), getIndex(), srcTy.getShape(),
                               tileShape);
}

void InsertTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                         Value tile, Value index) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto tileTy = cast<RankedTensorType>(tile.getType());
  SmallVector<int64_t> tileShape(tileTy.getShape());
  state.addOperands({src, tile, index});
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(srcTy);
}

LogicalResult InsertTileOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.size() < 3)
    return failure();
  auto srcTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto tileTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!srcTy || !tileTy)
    return failure();
  if (srcTy.getElementType() != tileTy.getElementType() ||
      srcTy.getRank() != tileTy.getRank())
    return failure();
  inferredReturnTypes.clear();
  inferredReturnTypes.push_back(srcTy);
  return success();
}

LogicalResult InsertTileOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto tileTy = dyn_cast<RankedTensorType>(getTile().getType());
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !tileTy || !resultTy)
    return emitOpError("expects source, tile, and result to be ranked tensors");

  SmallVector<int64_t> tileShape =
      getDenseI64Array(getOperation()->getAttr("tile_shape"));
  if (failed(verifyTileShape(getOperation(), srcTy.getShape(), tileShape,
                             "tile_shape")))
    return failure();
  if (tileTy.getShape() != ArrayRef<int64_t>(tileShape))
    return emitOpError("tile shape must equal tile_shape");
  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != resultTy.getElementType())
    return emitOpError("result element type must match source element type");
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must match source rank");
  if (srcTy.getRank() != resultTy.getRank())
    return emitOpError("result rank must match source rank");
  if (resultTy.getShape() != srcTy.getShape())
    return emitOpError("result shape must equal source shape");
  if (srcTy.getEncoding() && resultTy.getEncoding() &&
      srcTy.getEncoding() != resultTy.getEncoding())
    return emitOpError("result encoding must match source encoding");
  return verifyStaticTileIndex(getOperation(), getIndex(), srcTy.getShape(),
                               tileShape);
}

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

LogicalResult ExclusiveCumsumOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  if (!srcTy)
    return emitOpError() << "expects src to be a ranked tensor";

  auto exclusiveTy = dyn_cast<RankedTensorType>(getExclusive().getType());
  if (!exclusiveTy)
    return emitOpError() << "expects exclusive result to be a ranked tensor";
  if (exclusiveTy != srcTy)
    return emitOpError() << "expects exclusive result type to match src type";

  if (srcTy.getRank() != 1)
    return emitOpError() << "currently only rank-1 tensors are supported";
  int64_t axisExtent = srcTy.getShape()[0];
  if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0)
    return emitOpError() << "currently only static, positive axis extent is "
                            "supported";
  if (axisExtent > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return emitOpError() << "axis extent is too large";

  const int64_t rank = srcTy.getRank();
  int64_t axis = static_cast<int64_t>(getAxis());
  if (axis < 0)
    axis += rank;
  if (axis != 0)
    return emitOpError() << "currently only axis=0 is supported";

  if (getTotal().getType() != srcTy.getElementType())
    return emitOpError() << "expects total result type to match src element "
                            "type";

  return success();
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
