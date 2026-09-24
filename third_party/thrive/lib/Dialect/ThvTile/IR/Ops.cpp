#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/ThvTile/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GET_OP_CLASSES
#include "Dialect/ThvTile/IR/Ops.cpp.inc"

namespace mlir::thvtile {

//===----------------------------------------------------------------------===//
// MakeTensorStructureOp
//===----------------------------------------------------------------------===//

LogicalResult MakeTensorStructureOp::verify() {
  auto resultType = getResult().getType();
  auto blockType = cast<TensorStructureType>(resultType).getBlockType();
  int64_t rank = blockType.getRank();

  if (rank < 1 || rank > 5) {
    return emitOpError()
           << "tensor rank=" << rank << " is out of range [1, 5].\n"
           << "note: ThvTile supports up to 5D tensors.\n"
           << "hint: reshape your data to fit within 5 dimensions.";
  }

  auto order = getOrder();
  if (static_cast<int64_t>(order.size()) != rank) {
    return emitOpError() << "order array size (" << order.size()
                         << ") must match tensor rank (" << rank << ").\n"
                         << "hint: order should be a permutation of [0, "
                         << (rank - 1) << "].";
  }

  // Check order is a valid permutation.
  llvm::SmallVector<bool> seen(rank, false);
  for (int32_t dim : order) {
    if (dim < 0 || dim >= rank) {
      return emitOpError()
             << "order element " << dim << " is out of range [0, " << (rank - 1)
             << "].\n"
             << "hint: order must be a permutation of dimensions.";
    }
    if (seen[dim]) {
      return emitOpError()
             << "order contains duplicate dimension " << dim << ".\n"
             << "hint: each dimension must appear exactly once in order.";
    }
    seen[dim] = true;
  }

  // Check layout metadata operand counts match tensor rank.
  if (static_cast<int64_t>(getShape().size()) != rank) {
    return emitOpError() << "number of shape operands (" << getShape().size()
                         << ") must match tensor rank (" << rank << ").";
  }
  if (!getSourceShape().empty() &&
      static_cast<int64_t>(getSourceShape().size()) != rank) {
    return emitOpError() << "number of source_shape operands ("
                         << getSourceShape().size()
                         << ") must be empty or match tensor rank (" << rank
                         << ").";
  }
  if (static_cast<int64_t>(getStrides().size()) != rank) {
    return emitOpError() << "number of stride operands (" << getStrides().size()
                         << ") must match tensor rank (" << rank << ").";
  }
  if (static_cast<int64_t>(getOffsets().size()) != rank) {
    return emitOpError() << "number of offset operands (" << getOffsets().size()
                         << ") must match tensor rank (" << rank << ").";
  }

  // Check segment metadata matches the selected mask kind.
  auto maskKind = getMaskKind();
  int64_t validSegSizesRank = getValidSegSizes().size();
  int64_t validNumSegsRank = getValidNumSegs().size();
  int64_t validSegStridesRank = getValidSegStrides().size();
  auto allSegmentMetadataEmpty = [&]() {
    return validSegSizesRank == 0 && validNumSegsRank == 0 &&
           validSegStridesRank == 0;
  };

  if (maskKind == MaskKind::NONE) {
    if (!allSegmentMetadataEmpty()) {
      return emitOpError() << "valid segment metadata must be empty when "
                              "mask_kind is 'none'.";
    }
  } else if (maskKind == MaskKind::ARBITRARY) {
    if (!allSegmentMetadataEmpty()) {
      return emitOpError()
             << "valid segment metadata must be empty for arbitrary masks.\n"
             << "hint: arbitrary raw masks cannot be represented as structured "
                "DMA metadata; use iterative_load/store fallback instead.";
    }
  } else if (maskKind == MaskKind::BOUNDARY) {
    if (validSegSizesRank != rank) {
      return emitOpError() << "number of valid_seg_sizes operands ("
                           << validSegSizesRank << ") must match tensor rank ("
                           << rank << ") for boundary mask metadata.";
    }
    if (validNumSegsRank != 0 || validSegStridesRank != 0) {
      return emitOpError()
             << "valid_num_segs and valid_seg_strides must be empty for "
                "boundary masks.";
    }
  } else if (maskKind == MaskKind::STRIDED) {
    if (validSegSizesRank == 0 || validNumSegsRank == 0 ||
        validSegStridesRank == 0) {
      return emitOpError()
             << "strided masks require valid_seg_sizes, valid_num_segs, and "
                "valid_seg_strides metadata.";
    }
    if (validSegSizesRank != validNumSegsRank ||
        validSegSizesRank != validSegStridesRank) {
      return emitOpError()
             << "valid_seg_sizes, valid_num_segs, and valid_seg_strides must "
                "have the same number of operands for strided masks.";
    }
    if (validSegSizesRank != 1 && validSegSizesRank != rank) {
      return emitOpError() << "strided mask metadata rank ("
                           << validSegSizesRank
                           << ") must be either 1 or match tensor rank ("
                           << rank << ").";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  auto metaType = getStructure().getType();
  auto resultType = cast<RankedTensorType>(getResult().getType());
  auto expectedType = cast<TensorStructureType>(metaType).getBlockType();

  // Result type must match the tensor_structure block type.
  if (resultType != expectedType) {
    return emitOpError()
           << "result type (" << resultType
           << ") does not match structure block type (" << expectedType
           << ").\n"
           << "hint: the result tensor shape and element type must exactly "
              "match the tensor_structure's block type.";
  }

  // Optional fallback value must match the loaded element type.
  if (getOther() && getOther().getType() != resultType.getElementType()) {
    return emitOpError() << "other type (" << getOther().getType()
                         << ") must match result element type ("
                         << resultType.getElementType() << ").";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  auto metaType = getStructure().getType();
  auto valueType = cast<RankedTensorType>(getValue().getType());
  auto expectedType = cast<TensorStructureType>(metaType).getBlockType();

  // Stored value type must match the tensor_structure block type.
  if (valueType != expectedType) {
    return emitOpError()
           << "value type (" << valueType
           << ") does not match structure block type (" << expectedType
           << ").\n"
           << "hint: the stored tensor shape and element type must exactly "
              "match the tensor_structure's block type.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IterativeLoadOp
//===----------------------------------------------------------------------===//

LogicalResult IterativeLoadOp::verify() {
  auto ptrsType = dyn_cast<RankedTensorType>(getPtrs().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  // Scalar mode: ptrs is a scalar pointer and result is a scalar.
  if (!ptrsType && !resultType) {
    if (getMask() && isa<RankedTensorType>(getMask().getType())) {
      return emitOpError() << "scalar iterative_load requires a scalar (i1) "
                              "mask, not a tensor.";
    }
    if (getOther() && getOther().getType() != getResult().getType()) {
      return emitOpError() << "scalar iterative_load requires other type ("
                           << getOther().getType() << ") to match result type ("
                           << getResult().getType() << ").";
    }
    return success();
  }

  if (!ptrsType || !resultType) {
    return emitOpError()
           << "ptrs and result must both be scalar or both be tensors.";
  }

  // Tensor mode: ptrs tensor shape must match result tensor shape.
  if (ptrsType.getShape() != resultType.getShape()) {
    return emitOpError() << "pointer tensor shape (" << ptrsType.getShape()
                         << ") does not match result tensor shape ("
                         << resultType.getShape() << ").\n"
                         << "hint: iterative_load requires ptrs and result to "
                            "have the same shape.";
  }

  // Mask shape and fill value type must match the tensor result.
  if (getMask()) {
    auto maskType = cast<RankedTensorType>(getMask().getType());
    if (maskType.getShape() != resultType.getShape()) {
      return emitOpError() << "mask shape (" << maskType.getShape()
                           << ") does not match result shape ("
                           << resultType.getShape() << ").";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IterativeStoreOp
//===----------------------------------------------------------------------===//

LogicalResult IterativeStoreOp::verify() {
  auto valueType = dyn_cast<RankedTensorType>(getValue().getType());
  auto ptrsType = dyn_cast<RankedTensorType>(getPtrs().getType());

  // Scalar mode: value is a scalar and ptrs is a scalar pointer.
  if (!valueType && !ptrsType) {
    if (getMask() && isa<RankedTensorType>(getMask().getType())) {
      return emitOpError() << "scalar iterative_store requires a scalar (i1) "
                              "mask, not a tensor.";
    }
    return success();
  }

  if (!valueType || !ptrsType) {
    return emitOpError()
           << "value and ptrs must both be scalar or both be tensors.";
  }

  // Tensor mode: value tensor shape must match ptrs tensor shape.
  if (valueType.getShape() != ptrsType.getShape()) {
    return emitOpError() << "value tensor shape (" << valueType.getShape()
                         << ") does not match pointer tensor shape ("
                         << ptrsType.getShape() << ").\n"
                         << "hint: iterative_store requires value and ptrs to "
                            "have the same shape.";
  }

  // Mask shape must match the stored tensor value.
  if (getMask()) {
    auto maskType = cast<RankedTensorType>(getMask().getType());
    if (maskType.getShape() != valueType.getShape()) {
      return emitOpError() << "mask shape (" << maskType.getShape()
                           << ") does not match value shape ("
                           << valueType.getShape() << ").";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

LogicalResult EmptyOp::verify() {
  auto resultType = cast<RankedTensorType>(getResult().getType());
  int64_t numDynamic = resultType.getNumDynamicDims();

  // Dynamic size operand count must match dynamic result dimensions.
  if (static_cast<int64_t>(getDynamicSizes().size()) != numDynamic) {
    return emitOpError()
           << "number of dynamic size operands (" << getDynamicSizes().size()
           << ") must match the number of dynamic dimensions (" << numDynamic
           << ") in result type " << resultType << ".\n"
           << "hint: provide one Index operand per '?' dimension, in order.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());
  Type valueType = getValue().getType();
  int64_t rank = srcType.getRank();

  // Padding value must match the tensor element type.
  if (valueType != srcType.getElementType()) {
    return emitOpError() << "padding value type (" << valueType
                         << ") must match source element type ("
                         << srcType.getElementType() << ").";
  }

  // Padding arrays must provide one low/high value per dimension.
  if (static_cast<int64_t>(getLow().size()) != rank ||
      static_cast<int64_t>(getHigh().size()) != rank) {
    return emitOpError() << "low/high padding arrays must each have " << rank
                         << " elements (matching input rank).";
  }

  // Result shape is computed as src + low + high per dimension.
  for (int64_t i = 0; i < rank; ++i) {
    int64_t expected = srcType.getDimSize(i) + getLow()[i] + getHigh()[i];
    if (resType.getDimSize(i) != expected) {
      return emitOpError() << "result dimension " << i << " should be "
                           << expected << " (src=" << srcType.getDimSize(i)
                           << " + low=" << getLow()[i]
                           << " + high=" << getHigh()[i] << ") but got "
                           << resType.getDimSize(i) << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

LogicalResult PermuteOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());
  int64_t rank = srcType.getRank();

  // ThvTile tensor operations are limited to rank <= 5.
  if (rank > 5) {
    return emitOpError() << "tensor rank=" << rank
                         << " exceeds maximum of 5 for ThvTile.\n"
                         << "hint: ThvTile supports up to 5D tensors.";
  }

  auto permutation = getPermutation();
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return emitOpError() << "permutation size (" << permutation.size()
                         << ") must equal input rank (" << rank << ").";
  }

  uint8_t seen[5] = {0, 0, 0, 0, 0};
  for (int32_t dim : permutation) {
    if (dim < 0 || dim >= rank) {
      return emitOpError() << "permutation element " << dim
                           << " is out of range [0, " << (rank - 1) << "].";
    }
    if (seen[dim]) {
      return emitOpError() << "duplicate dimension " << dim
                           << " in permutation.";
    }
    seen[dim] = 1;
  }

  // Result shape must follow the requested permutation.
  for (int64_t i = 0; i < rank; ++i) {
    int64_t expectedDim = srcType.getDimSize(permutation[i]);
    if (resType.getDimSize(i) != expectedDim) {
      return emitOpError() << "result dimension " << i << " should be "
                           << expectedDim << " (from source dimension "
                           << permutation[i] << ") but got "
                           << resType.getDimSize(i) << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FlipOp
//===----------------------------------------------------------------------===//

LogicalResult FlipOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  int64_t rank = srcType.getRank();
  int32_t axis = getAxis();

  if (axis < 0 || axis >= rank) {
    return emitOpError() << "axis=" << axis << " is out of range for rank-"
                         << rank << " input.\n"
                         << "note: valid axis values are 0 to " << (rank - 1)
                         << ".";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());

  // Cast changes element type but preserves shape.
  if (srcType.getShape() != resType.getShape()) {
    return emitOpError() << "source and result shapes must match.\n"
                         << "note: cast only changes element type, not shape.";
  }

  // Reject no-op casts.
  if (srcType.getElementType() == resType.getElementType()) {
    return emitOpError()
           << "source and result have the same element type ("
           << srcType.getElementType() << "). Cast is unnecessary.\n"
           << "hint: remove this cast or check the intended types.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());

  // Bitcast requires equal element bit-widths.
  unsigned srcBits = srcType.getElementTypeBitWidth();
  unsigned resBits = resType.getElementTypeBitWidth();
  if (srcBits != resBits) {
    return emitOpError()
           << "source element bit-width (" << srcBits
           << ") must equal result element bit-width (" << resBits << ").\n"
           << "hint: bitcast requires identical bit-widths. Use thvtile.cast "
              "for type conversions.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExtractSliceOp / InsertSliceOp shared verification
//===----------------------------------------------------------------------===//

static bool allEqual(ArrayRef<int64_t> values, int64_t expected) {
  for (int64_t value : values) {
    if (value != expected)
      return false;
  }
  return true;
}

static bool allSizesMatchShape(ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> shape) {
  if (sizes.size() != shape.size())
    return false;
  for (size_t index = 0; index < sizes.size(); ++index) {
    int64_t size = sizes[index];
    int64_t dim = shape[index];
    if (ShapedType::isDynamic(dim) || size != dim)
      return false;
  }
  return true;
}

static LogicalResult verifySliceAccessPattern(Operation *op,
                                              ArrayRef<int64_t> sizes,
                                              ArrayRef<int64_t> strides,
                                              ArrayRef<int64_t> fullTileShape,
                                              llvm::StringRef accessKind) {
  bool hasUniqueNonUnitStride = false;
  int64_t uniqueNonUnitStride = -1;

  for (size_t index = 0; index < sizes.size(); ++index) {
    int64_t size = sizes[index];
    if (size <= 0) {
      return op->emitOpError() << "static_sizes[" << index
                               << "] must be positive, got " << size << ".";
    }
  }

  for (size_t index = 0; index < strides.size(); ++index) {
    int64_t stride = strides[index];
    if (stride <= 0) {
      return op->emitOpError() << "static_strides[" << index
                               << "] must be positive, got " << stride << ".";
    }

    if (stride == 1)
      continue;

    if (uniqueNonUnitStride < 0) {
      uniqueNonUnitStride = stride;
      hasUniqueNonUnitStride = true;
      continue;
    }

    if (stride != uniqueNonUnitStride) {
      return op->emitOpError()
             << accessKind
             << " access requires a regular strided pattern with one "
             << "uniform non-unit gap, but got gap periods "
             << uniqueNonUnitStride << " and " << stride << ".";
    }
  }

  // Static size/stride coverage must stay within the source/dest tile.
  for (size_t index = 0; index < sizes.size(); ++index) {
    int64_t size = sizes[index];
    int64_t stride = strides[index];
    int64_t dim = fullTileShape[index];
    if (ShapedType::isDynamic(dim))
      continue;
    int64_t coveredSpan = (size - 1) * stride + 1;
    if (coveredSpan > dim) {
      return op->emitOpError()
             << "static size/stride on dimension " << index << " covers "
             << coveredSpan << " elements, exceeding tile dim " << dim << ".";
    }
  }

  // Allow full-tile, scalar-like, or regular strided access patterns.
  bool isFullTileAccess =
      allEqual(strides, 1) && allSizesMatchShape(sizes, fullTileShape);
  bool isScalarLikeAccess = allEqual(sizes, 1) && allEqual(strides, 1);
  bool isGatherScatterLikeAccess = hasUniqueNonUnitStride;

  if (isFullTileAccess || isScalarLikeAccess || isGatherScatterLikeAccess)
    return success();

  return op->emitOpError()
         << "unsupported slice access pattern: expected full-tile access "
         << "(sizes match tile shape and strides are all 1), scalar-like "
         << "access (sizes and strides are all 1), or " << accessKind
         << "-like access with a regular strided pattern and one uniform "
         << "non-unit gap.";
}

//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractSliceOp::verify() {
  auto srcType = cast<RankedTensorType>(getSource().getType());
  auto destType = cast<RankedTensorType>(getDest().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  int64_t rank = srcType.getRank();

  if (destType != resultType) {
    return emitOpError() << "result type must match dest type.\n"
                         << "note: result=" << resultType
                         << ", dest=" << destType;
  }

  // Source and destination element types must match.
  if (srcType.getElementType() != destType.getElementType()) {
    return emitOpError() << "source element type (" << srcType.getElementType()
                         << ") does not match dest element type ("
                         << destType.getElementType() << ").";
  }

  // Offsets, sizes, and strides are specified per source dimension.
  if (static_cast<int64_t>(getOffsets().size()) != rank ||
      static_cast<int64_t>(getStaticSizes().size()) != rank ||
      static_cast<int64_t>(getStaticStrides().size()) != rank) {
    return emitOpError() << "offsets/sizes/strides must each have " << rank
                         << " elements (matching source rank).\n"
                         << "note: source tensor has rank " << rank << ".";
  }

  if (failed(verifySliceAccessPattern(*this, getStaticSizes(),
                                      getStaticStrides(), srcType.getShape(),
                                      "gather")))
    return failure();

  // Rank-0 dest/result is allowed only for scalar-like extracts.
  if (destType.getRank() == 0) {
    for (int64_t size : getStaticSizes()) {
      if (size != 1) {
        return emitOpError()
               << "rank-0 DPS extract requires all static_sizes to be 1.";
      }
    }
    return success();
  }

  if (destType.getRank() != rank) {
    return emitOpError()
           << "dest/result rank (" << destType.getRank()
           << ") must match source rank (" << rank
           << "), unless using rank-0 tensor for scalar-like extract.";
  }

  // Non-scalar extract result shape must match static_sizes.
  for (int64_t i = 0; i < rank; ++i) {
    int64_t expectedDim = getStaticSizes()[i];
    if (destType.getDimSize(i) != expectedDim) {
      return emitOpError() << "dest/result dimension " << i << " is "
                           << destType.getDimSize(i)
                           << " but static_sizes specifies " << expectedDim
                           << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InsertSliceOp
//===----------------------------------------------------------------------===//

LogicalResult InsertSliceOp::verify() {
  auto destType = cast<RankedTensorType>(getDest().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  auto sourceTensorType = dyn_cast<RankedTensorType>(getSource().getType());

  // DPS result type must match dest type.
  if (destType != resultType) {
    return emitOpError() << "result type must match dest type.\n"
                         << "note: result=" << resultType
                         << ", dest=" << destType;
  }

  // Source and destination element types must match.
  Type sourceElementType = getElementTypeOrSelf(getSource().getType());
  if (sourceElementType != destType.getElementType()) {
    return emitOpError() << "source element type (" << sourceElementType
                         << ") does not match dest element type ("
                         << destType.getElementType() << ").";
  }

  int64_t rank = destType.getRank();

  // Offsets, sizes, and strides are specified per destination dimension.
  if (static_cast<int64_t>(getOffsets().size()) != rank ||
      static_cast<int64_t>(getStaticSizes().size()) != rank ||
      static_cast<int64_t>(getStaticStrides().size()) != rank) {
    return emitOpError() << "offsets/sizes/strides must each have " << rank
                         << " elements (matching result rank).";
  }

  if (failed(verifySliceAccessPattern(*this, getStaticSizes(),
                                      getStaticStrides(), destType.getShape(),
                                      "scatter")))
    return failure();

  // Scalar or rank-0 source is only valid for scalar-like inserts.
  if (!sourceTensorType || sourceTensorType.getRank() == 0) {
    for (int64_t size : getStaticSizes()) {
      if (size != 1) {
        return emitOpError()
               << "scalar or rank-0 tensor insert requires all static_sizes "
                  "to be 1.";
      }
    }
    return success();
  }

  if (sourceTensorType.getRank() != rank) {
    return emitOpError()
           << "tensor source rank (" << sourceTensorType.getRank()
           << ") must match result rank (" << rank
           << "), unless using rank-0 tensor for scalar-like insert.";
  }

  // Tensor source shape must match static_sizes.
  for (int64_t i = 0; i < rank; ++i) {
    int64_t expectedDim = getStaticSizes()[i];
    if (sourceTensorType.getDimSize(i) != expectedDim) {
      return emitOpError() << "source dimension " << i << " is "
                           << sourceTensorType.getDimSize(i)
                           << " but static_sizes specifies " << expectedDim
                           << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExpandShapeOp / CollapseShapeOp
//===----------------------------------------------------------------------===//

LogicalResult ExpandShapeOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());
  // Reshape ops must preserve the total element count.
  if (srcType.getNumElements() != resType.getNumElements()) {
    return emitOpError() << "element count mismatch between source ("
                         << srcType.getNumElements() << ") and result ("
                         << resType.getNumElements() << ").";
  }
  return success();
}

LogicalResult CollapseShapeOp::verify() {
  auto srcType = cast<RankedTensorType>(getSrc().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());
  // Reshape ops must preserve the total element count.
  if (srcType.getNumElements() != resType.getNumElements()) {
    return emitOpError() << "element count mismatch between source ("
                         << srcType.getNumElements() << ") and result ("
                         << resType.getNumElements() << ").";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto aType = cast<RankedTensorType>(getA().getType());
  auto bType = cast<RankedTensorType>(getB().getType());

  if (aType.getRank() != 2 || bType.getRank() != 2) {
    return emitOpError() << "matmul requires 2D input tensors.\n"
                         << "note: A has rank " << aType.getRank()
                         << ", B has rank " << bType.getRank() << ".";
  }

  // Inputs are A[M,K] and B[K,N].
  int64_t m = aType.getDimSize(0);
  int64_t aK = aType.getDimSize(1);
  int64_t bK = bType.getDimSize(0);
  int64_t n = bType.getDimSize(1);
  if (aK != bK) {
    return emitOpError()
           << "dimension mismatch: A's K-dim (" << aK << ") != B's K-dim ("
           << bK << ").\n"
           << "note: for A=[M,K] and B=[K,N], the inner dimensions must match.";
  }

  // Result D must have shape [M,N].
  auto dType = cast<RankedTensorType>(getD().getType());
  if (dType.getRank() != 2 || dType.getDimSize(0) != m ||
      dType.getDimSize(1) != n) {
    return emitOpError() << "result D must have shape [M,N] = [" << m << ","
                         << n << "] but got " << dType << ".";
  }

  // Optional C may be a 2D accumulator or a 1D bias.
  if (getC()) {
    auto cType = cast<RankedTensorType>(getC().getType());
    if (cType.getRank() == 2) {
      if (cType.getDimSize(0) != m || cType.getDimSize(1) != n) {
        return emitOpError() << "2D C accumulator must have shape [M,N] = ["
                             << m << "," << n << "] but got " << cType << ".";
      }
    } else if (cType.getRank() == 1) {
      if (cType.getDimSize(0) != n) {
        return emitOpError() << "1D C bias must have shape [N] = [" << n
                             << "] but got " << cType << ".";
      }
    } else {
      return emitOpError()
             << "C must be either a 2D accumulator [M,N] or a 1D bias [N], "
             << "but got " << cType << ".";
    }
  }

  // Check quantization format compatibility and required scales.
  auto aFormat = getAFormat();
  auto bFormat = getBFormat();
  bool hasAScale = static_cast<bool>(getAScale());
  bool hasBScale = static_cast<bool>(getBScale());

  if (aFormat == QuantFormat::NONE && hasAScale) {
    return emitOpError() << "a_scale is provided but a_format is 'none'.\n"
                         << "hint: remove a_scale or set a scaled a_format.";
  }
  if (bFormat == QuantFormat::NONE && hasBScale) {
    return emitOpError() << "b_scale is provided but b_format is 'none'.\n"
                         << "hint: remove b_scale or set a scaled b_format.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp / AtomicCASOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyAtomicValueType(Operation *op, Value value,
                                           llvm::StringRef valueName,
                                           Type expectedElementType,
                                           ArrayRef<int64_t> expectedShape,
                                           bool tensorMode) {
  auto valueTensorType = dyn_cast<RankedTensorType>(value.getType());

  // Tensor atomic operands must match the pointer tensor shape and pointee
  // type.
  if (tensorMode) {
    if (!valueTensorType) {
      return op->emitOpError()
             << valueName
             << " must be an integer tensor when ptr is a tensor of pointers.";
    }
    if (valueTensorType.getShape() != expectedShape) {
      return op->emitOpError()
             << valueName << " shape (" << valueTensorType.getShape()
             << ") does not match pointer tensor shape (" << expectedShape
             << ").";
    }
    if (valueTensorType.getElementType() != expectedElementType) {
      return op->emitOpError() << valueName << " element type ("
                               << valueTensorType.getElementType()
                               << ") does not match pointer pointee type ("
                               << expectedElementType << ").";
    }
    return success();
  }

  // Scalar atomic operands must match the scalar pointer pointee type.
  if (valueTensorType) {
    return op->emitOpError()
           << valueName
           << " must be an integer scalar when ptr is a scalar pointer.";
  }
  if (value.getType() != expectedElementType) {
    return op->emitOpError() << valueName << " type (" << value.getType()
                             << ") does not match pointer pointee type ("
                             << expectedElementType << ").";
  }
  return success();
}

// TODO: remove triton::PointerType after all the ops are refactored.
static std::optional<Type> getPointeeTypeOfPtr(Type type) {
  if (auto ttPtr = dyn_cast<triton::PointerType>(type))
    return ttPtr.getPointeeType();
  if (auto thvPtr = dyn_cast<PtrType>(type))
    return thvPtr.getPointeeType();
  return std::nullopt;
}

// Resolve scalar pointer or tensor-of-pointers to its pointee type.
static LogicalResult getAtomicPointeeType(Operation *op, Value ptr,
                                          Type &pointeeType,
                                          ArrayRef<int64_t> &ptrShape,
                                          bool &tensorMode) {
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    auto pointee = getPointeeTypeOfPtr(ptrTensorType.getElementType());
    if (!pointee) {
      return op->emitOpError()
             << "ptr tensor element type must be a pointer type, got "
             << ptrTensorType.getElementType() << ".";
    }
    pointeeType = *pointee;
    ptrShape = ptrTensorType.getShape();
    tensorMode = true;
    return success();
  }

  auto pointee = getPointeeTypeOfPtr(ptr.getType());
  if (!pointee) {
    return op->emitOpError() << "ptr must be a pointer or tensor of pointers.";
  }
  pointeeType = *pointee;
  ptrShape = {};
  tensorMode = false;
  return success();
}

LogicalResult AtomicRMWOp::verify() {
  Type pointeeType;
  ArrayRef<int64_t> ptrShape;
  bool tensorMode = false;
  if (failed(getAtomicPointeeType(*this, getPtr(), pointeeType, ptrShape,
                                  tensorMode)))
    return failure();

  // Atomic RMW is restricted to integer pointee types.
  if (!isa<IntegerType>(pointeeType)) {
    return emitOpError() << "pointer pointee type must be integer, got "
                         << pointeeType << ".";
  }

  if (failed(verifyAtomicValueType(*this, getValue(), "value", pointeeType,
                                   ptrShape, tensorMode)))
    return failure();
  if (failed(verifyAtomicValueType(*this, getResult(), "result", pointeeType,
                                   ptrShape, tensorMode)))
    return failure();

  // Optional mask must follow the pointer tensor shape.
  if (getMask()) {
    if (!tensorMode) {
      return emitOpError() << "mask is only supported for tensor atomic_rmw.";
    }
    auto maskType = cast<RankedTensorType>(getMask().getType());
    if (maskType.getShape() != ptrShape) {
      return emitOpError() << "mask shape (" << maskType.getShape()
                           << ") does not match pointer tensor shape ("
                           << ptrShape << ").";
    }
  }

  return success();
}

LogicalResult AtomicCASOp::verify() {
  Type pointeeType;
  ArrayRef<int64_t> ptrShape;
  bool tensorMode = false;
  if (failed(getAtomicPointeeType(*this, getPtr(), pointeeType, ptrShape,
                                  tensorMode)))
    return failure();

  // Atomic CAS is restricted to integer pointee types.
  if (!isa<IntegerType>(pointeeType)) {
    return emitOpError() << "pointer pointee type must be integer, got "
                         << pointeeType << ".";
  }

  if (failed(verifyAtomicValueType(*this, getCmp(), "cmp", pointeeType,
                                   ptrShape, tensorMode)))
    return failure();
  if (failed(verifyAtomicValueType(*this, getVal(), "val", pointeeType,
                                   ptrShape, tensorMode)))
    return failure();
  if (failed(verifyAtomicValueType(*this, getResult(), "result", pointeeType,
                                   ptrShape, tensorMode)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// AddPtrOp
//===----------------------------------------------------------------------===//

static bool isPtrOrTensorOfPtr(Type type) {
  if (isa<triton::PointerType, PtrType>(type))
    return true;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<triton::PointerType, PtrType>(tensorTy.getElementType());
  return false;
}

LogicalResult AddPtrOp::verify() {
  if (!isPtrOrTensorOfPtr(getPtr().getType()))
    return emitOpError()
           << "ptr operand must be a pointer or tensor of pointers";
  if (!isPtrOrTensorOfPtr(getResult().getType()))
    return emitOpError() << "result must be a pointer or tensor of pointers";
  if (getPtr().getType() != getResult().getType())
    return emitOpError() << "result type must match ptr operand type";
  return success();
}

OpFoldResult AddPtrOp::fold(FoldAdaptor adaptor) {
  // addptr(ptr, 0) -> ptr
  if (matchPattern(adaptor.getOffset(), m_Zero())) {
    return getPtr();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Memory effects
//===----------------------------------------------------------------------===//

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

void IterativeLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

void IterativeStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

void LibdeviceCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

} // namespace mlir::thvtile
