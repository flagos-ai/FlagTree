#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::tle {

namespace {
// Triton shared-memory pointers map to LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
} // namespace

// ============================================================================
// ExtractTileOp Builder
// ============================================================================
void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto resultType = RankedTensorType::get(tileShape, srcType.getElementType(),
                                          srcType.getEncoding());
  state.addOperands(src);
  state.addOperands(index);
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultType);
}

// ============================================================================
// ExtractTileOp Verification
//
// For dynamic index (index operand is not arith.constant):
//   - Only check constraints that are known at compile time: tile_shape
//   positivity, divisibility, element type, rank match
//   - Skip out-of-bounds and CTA tile alignment checks (only known at runtime)
//
// For static index: perform full checks (same as original implementation)
// ============================================================================
LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();

  // ---- Get tile_shape attribute ----
  auto tileShapeRawAttr = getOperation()->getAttr("tile_shape");
  SmallVector<int64_t> tileShape;
  if (auto denseArray64 =
          mlir::dyn_cast<mlir::DenseI64ArrayAttr>(tileShapeRawAttr)) {
    for (auto v : denseArray64.asArrayRef())
      tileShape.push_back(v);
  }

  // ---- Basic checks required for both static and dynamic index ----

  // Check 1: element types must match
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must equal source rank");

  // Check 3: tile_shape rank must match source rank
  if (tileShape.size() != srcShape.size())
    return emitOpError("tile_shape rank must match source rank");

  // Check 4: tile_shape must be positive in each dimension, divisible, and dst
  // shape must equal tile_shape
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile_shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile_shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    if (dstShape[i] != tileShape[i])
      return emitOpError("result shape must equal tile_shape at dimension ")
             << i;
  }

  // ---- Determine if index is a static constant ----
  // getDefiningOp<arith::ConstantOp>() returns nullptr for dynamic Value
  auto indexConstOp =
      getOperation()->getOperand(1).getDefiningOp<arith::ConstantOp>();

  if (!indexConstOp) {
    // Dynamic index: skip out-of-bounds and offset alignment checks, handled at
    // lowering stage
    return success();
  }

  // ---- Full checks for static index ----
  int64_t index =
      mlir::cast<mlir::IntegerAttr>(indexConstOp.getValue()).getInt();

  // Compute logical grid shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Out-of-bounds check
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // Delinearize to per-dimension tile indices (row-major order)
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile indices -> coordinate-level offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // Boundary check
  if (offsets.size() != static_cast<size_t>(srcTy.getRank()))
    return emitError("offsets size must match tensor rank");

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] > srcShape[i])
      return emitOpError(
                 "result shape cannot exceed source shape at dimension ")
             << i;
    if (offsets[i] + dstShape[i] > srcShape[i])
      return emitOpError("invalid offset at dimension ")
             << i << ": offset(" << offsets[i] << ") + shape(" << dstShape[i]
             << ") > source(" << srcShape[i] << ")";
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
  }

  auto encoding = srcTy.getEncoding();
  if (!encoding)
    return success();
  return success();
}

// ============================================================================
// InsertTileOp Type Inference + Verification
// ============================================================================
LogicalResult InsertTileOp::inferReturnTypes(
    [[maybe_unused]] MLIRContext *context,
    [[maybe_unused]] std::optional<Location> location, ValueRange operands,
    [[maybe_unused]] DictionaryAttr attributes,
    [[maybe_unused]] OpaqueProperties properties,
    [[maybe_unused]] RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // insert_tile(src, tile, index) -> result has the same type as src.
  if (operands.size() < 3)
    return failure();

  auto srcTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto tileTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!srcTy || !tileTy)
    return failure();

  // Keep conservative checks here; full diagnostics are handled in verify().
  if (srcTy.getElementType() != tileTy.getElementType() ||
      srcTy.getRank() != tileTy.getRank())
    return failure();

  inferredReturnTypes.clear();
  inferredReturnTypes.push_back(srcTy);
  return success();
}

// ============================================================================
// InsertTileOp Verification
//
// For dynamic index (index operand is not arith.constant):
//   - Only check constraints that are known at compile time: tile_shape
//   positivity, divisibility, element type, rank/result shape match
//   - Skip out-of-bounds and insertion region boundary checks (only known at
//   runtime)
//
// For static index: perform full checks (same as original implementation)
// ============================================================================
LogicalResult InsertTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto tileTy = cast<RankedTensorType>(getTile().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  auto dstShape = dstTy.getShape();

  // --- Basic checks required for both static and dynamic index ---

  // Check 1: element types must match
  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must equal source rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("result rank must equal source rank");

  // Check 3: result shape must equal source shape
  if (dstShape != srcShape)
    return emitOpError("result shape must equal source shape");

  // Check 4: tile_shape must be positive in each dimension and divide source
  // shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Check 5: insert_tile updates values but does not change global layout,
  // result encoding must match source encoding
  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (srcEnc && dstEnc && srcEnc != dstEnc)
    return emitOpError("result encoding must match source encoding");

  // --- Determine if index is a static constant ---
  // insert_tile index is the 3rd operand: (src, tile, index).
  auto idxDef =
      getOperation()->getOperand(2).getDefiningOp<arith::ConstantOp>();
  if (!idxDef) {
    // Dynamic index: skip out-of-bounds and insertion region boundary checks,
    // handled at lowering stage
    return success();
  }

  // --- Full checks for static index ---
  int64_t index = mlir::cast<mlir::IntegerAttr>(idxDef.getValue()).getInt();
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // Delinearize to per-dimension tile indices (row-major order)
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile indices -> coordinate-level offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // Boundary check: the full insertion region must be within the source
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
    else if (offsets[i] + tileShape[i] > srcShape[i])
      return emitOpError("invalid insertion region at dimension ")
             << i << ": offset(" << offsets[i] << ") + tile(" << tileShape[i]
             << ") > source(" << srcShape[i] << ")";
  }

  return success();
}

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

LogicalResult DistributedBarrierOp::verify() {
  auto *op = getOperation();
  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto rankAttr = op->getAttrOfType<IntegerAttr>("group_rank");
  auto shapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_shape");
  auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  auto maskAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_mask");

  const bool hasAnyGroupMeta =
      rankAttr || shapeAttr || axesAttr || maskAttr || kindAttr;
  if (!hasAnyGroupMeta)
    return success();

  if (!kindAttr) {
    return emitOpError()
           << "group_kind is required when distributed barrier group metadata "
              "is provided";
  }

  StringRef kind = kindAttr.getValue();
  if (kind != "cluster" && kind != "submesh" && kind != "grid") {
    return emitOpError()
           << "group_kind must be 'cluster', 'submesh', or 'grid', got '"
           << kind << "'";
  }

  if (kind == "cluster" || kind == "grid") {
    if (rankAttr || shapeAttr || axesAttr || maskAttr) {
      return emitOpError()
             << kind
             << " group_kind does not accept "
                "group_rank/group_shape/group_axes/group_mask attrs";
    }
    return success();
  }

  if (!rankAttr || !shapeAttr || !axesAttr) {
    return emitOpError()
           << "submesh group_kind requires group_rank/group_shape/group_axes";
  }
  if (!rankAttr.getType().isInteger(32)) {
    return emitOpError() << "group_rank must be i32";
  }

  int32_t rank = static_cast<int32_t>(rankAttr.getInt());
  if (rank <= 0) {
    return emitOpError() << "group_rank must be > 0";
  }
  if (static_cast<int32_t>(shapeAttr.size()) != rank) {
    return emitOpError() << "group_shape length (" << shapeAttr.size()
                         << ") must match group_rank (" << rank << ")";
  }
  if (static_cast<int32_t>(axesAttr.size()) != rank) {
    return emitOpError() << "group_axes length (" << axesAttr.size()
                         << ") must match group_rank (" << rank << ")";
  }

  llvm::SmallSet<int32_t, 8> seenAxes;
  for (int32_t dim : shapeAttr.asArrayRef()) {
    if (dim <= 0)
      return emitOpError() << "group_shape entries must be > 0";
  }
  for (int32_t axis : axesAttr.asArrayRef()) {
    if (axis < 0)
      return emitOpError() << "group_axes entries must be >= 0";
    if (!seenAxes.insert(axis).second) {
      return emitOpError() << "group_axes entries must be unique";
    }
  }
  if (maskAttr) {
    if (maskAttr.asArrayRef().empty())
      return emitOpError() << "group_mask cannot be empty";
    for (int32_t id : maskAttr.asArrayRef()) {
      if (id < 0)
        return emitOpError() << "group_mask entries must be >= 0";
    }
  }

  return success();
}

LogicalResult RemotePointersOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  if (!srcTy)
    return emitOpError() << "expects src operand to be a ranked tensor";
  auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultTy)
    return emitOpError() << "expects result to be a ranked tensor";
  if (srcTy != resultTy)
    return emitOpError() << "expects result type to match src type";

  auto ptrTy = dyn_cast<triton::PointerType>(srcTy.getElementType());
  if (!ptrTy)
    return emitOpError() << "expects src/result element type to be tt.ptr";
  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  if (!getShardId().getType().isInteger(32))
    return emitOpError() << "expects shard_id to be i32";

  return success();
}

} // namespace mlir::triton::tle
