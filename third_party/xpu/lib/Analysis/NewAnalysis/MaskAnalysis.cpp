#include "triton/Analysis/NewAnalysis/MaskAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/NewAnalysis/OpFoldResultUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace triton {

LogicalResult MaskState::parse(Value operand, const Location loc,
                               OpBuilder &builder) {
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, builder);
  } else if (isa<IntegerType>(operand.getType())) {
    return this->parseIntScalar(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpFOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return this->parseExtSI(op, loc, builder);
  } else {
    return failure();
  }
}

tensor::ExtractSliceOp MaskState::getExtractSlice(Value source,
                                                  const Location loc,
                                                  OpBuilder &builder) const {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> offsets = getWindowOffsets(builder);
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  // XTDK MLIR had a 5-arg inferResultType(source, offsets, sizes, strides);
  // public MLIR 22 only exposes the sizes-based overload.  The result shape is
  // exactly `dims` (offsets only move the window, they do not resize it), so
  // build the type directly.
  SmallVector<int64_t> shape;
  for (auto d : dims) {
    std::optional<int64_t> c = mlir::getConstantIntValue(d);
    // Dynamic dims stay dynamic: the XTDK inferResultType overload
    // this replaces could model them, and tiled IR (e.g. TritonKLX
    // irtest tiled_matmul_kind4) does reach here with a runtime extent.
    shape.push_back(c.has_value() ? *c : ShapedType::kDynamic);
  }
  auto dstType = RankedTensorType::get(shape, sourceType.getElementType());

  return builder.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets,
                                                dims, strides);
}

memref::SubViewOp MaskState::getSubview(Value source, const Location loc,
                                        OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets = getWindowOffsets(builder);
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return builder.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType),
                                           source, offsets, dims, strides);
}

SmallVector<OpFoldResult>
MaskState::getWindowOffsets(OpBuilder &builder) const {
  SmallVector<OpFoldResult> result(offsets.begin(), offsets.end());
  result.resize(getRank(), builder.getIndexAttr(0));
  return result;
}

static memref::SubViewOp createSubview(Value src, Location loc, OpBuilder &b,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstType), src,
                                     offsets, sizes, strides);
}

// Assume block1 wraps around and the remainder is block2.
//
// |----------------------|
// |         |            |
// | block2  |  block1    |
// |         |            |
// |----------------------|
//
// Once we copy the chunks in order, the end result is block1 followed by
// block2.
//
//   buffer_tmp:
//
// |----------------------|
// |             |        |
// | block1      | block2 |
// |             |        |
// |----------------------|
//
// Assume we have the following subview:
//
// +++++++++++++++++-------
// +               +      |
// + subview       +      |
// +               +      |
// +++++++++++++++++-------
//
// If we simply take the subview of `buffer_tmp`, this requires an extra buffer
// to just hold the temporary result.
//
// So we can subview into block1 and block2 directly. There are 2 cases:
//   + subview only spans block1
//   + subview spans both block1 and block2, creating sv1 and sv2 (illustrated
//     below for case when we wrap around side-by-side)
//
// |----------------------------------------|
// |                                        |
// |    col2                       col1     |
// |++++++--------|          |+++++++++++++++
// | sv2 + block2 |          | block1 & sv1 +
// |++++++--------|          |+++++++++++++++
// |                                        |
// |----------------------------------------|
//
// For simplicity, assume we only wrap around side-by-side.
//
// Let (row, col1) and (row, col2) be the dimensions of block1 and block2,
// respectively.
//
// Let (rowFull, colFull), (rowView1, colView1) and (rowView2, colView2) be the
// dimensions of the full subview, sv1, and sv2, respectively.
//
// + colView1 = min(colFull, col1)
// + colView2 = colFull - colView1
// + rowView1 = rowView2 = row = rowFull
std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getSideBySideSubviews(Value block1, Value block2, const Location loc,
                                 OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult col1 = builder.create<memref::DimOp>(loc, block1, 1).getResult();
  OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, builder);
  OpFoldResult subviewCol2 = subOFRs(subviewColFull, subviewCol1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRowFull, subviewCol1}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRowFull, subviewCol2}, strides);

  return {sv1, sv2};
}

std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getStackedSubviews(Value block1, Value block2, const Location loc,
                              OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult row1 = builder.create<memref::DimOp>(loc, block1, 0).getResult();
  OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, builder);
  OpFoldResult subviewRow2 = subOFRs(subviewRowFull, subviewRow1, loc, builder);

  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto sv1 = createSubview(block1, loc, builder, offsets,
                           {subviewRow1, subviewColFull}, strides);
  auto sv2 = createSubview(block2, loc, builder, offsets,
                           {subviewRow2, subviewColFull}, strides);
  return {sv1, sv2};
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  start = addOFRs(state.start, scalar, loc, builder);
  end = addOFRs(state.end, scalar, loc, builder);
  dims = state.dims;
  return success();
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskState::minStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.getRank() != rhsState.getRank()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unexpected case where lhs and rhs have different ranks";
    return failure();
  }

  // `and` of two masks is the intersection of the two windows, per dimension:
  //
  //   off = max(off_a, off_b)
  //   end = min(off_a + dim_a, off_b + dim_b)
  //   dim = max(0, end - off)
  //
  // With both offsets zero this is the plain `min(dim_a, dim_b)` it used to be,
  // and the arithmetic folds back to it.
  SmallVector<OpFoldResult> lhsOffsets = lhsState.getWindowOffsets(builder);
  SmallVector<OpFoldResult> rhsOffsets = rhsState.getWindowOffsets(builder);
  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    if (hasConstZero(lhsOffsets[i]) && hasConstZero(rhsOffsets[i])) {
      offsets.push_back(builder.getIndexAttr(0));
      dims.push_back(minOFRs(lhsDim, rhsDim, loc, builder));
      continue;
    }
    auto newOffset = maxOFRs(lhsOffsets[i], rhsOffsets[i], loc, builder);
    auto lhsEnd = addOFRs(lhsOffsets[i], lhsDim, loc, builder);
    auto rhsEnd = addOFRs(rhsOffsets[i], rhsDim, loc, builder);
    auto newEnd = minOFRs(lhsEnd, rhsEnd, loc, builder);
    // Disjoint windows would give a negative length; clamp instead.
    newEnd = maxOFRs(newEnd, newOffset, loc, builder);
    offsets.push_back(newOffset);
    dims.push_back(subOFRs(newEnd, newOffset, loc, builder));
  }
  return success();
}

LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc, OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }

  return success();
}

LogicalResult MaskState::parseIntScalar(Value scalar, const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto castOp =
      builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder)))
    return failure();

  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)))
    return failure();
  if (auto splatOp = andOp.getLhs().getDefiningOp<triton::SplatOp>())
    if (!splatOp && !lhsState.isMask())
      return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)))
    return failure();
  if (auto splatOp = andOp.getRhs().getDefiningOp<triton::SplatOp>())
    if (!splatOp && !rhsState.isMask())
      return failure();

  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseExtSI(arith::ExtSIOp op, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());
  return parse(op.getIn(), loc, builder);
}

// Map an ordered comparison onto the two forms the window model understands:
// a scalar bound above the range, or one below it.
//   slt/ult/olt        range <  bound    upper bound at bound
//   sle/ule/ole        range <= bound    upper bound at bound + 1
//   sgt/ugt/ogt        range >  bound    lower bound at bound + 1
//   sge/uge/oge        range >= bound    lower bound at bound
// `upperBound` reports the first two forms, `adjustBound` the strict ones whose
// window edge sits one past the bound.  Signed and unsigned spellings normalise
// identically: the window arithmetic below is sign-agnostic, and only the
// predicate name differs.
// eq/ne and the unordered float predicates do not describe a contiguous window
// at all and are rejected.
//
// The predicate decides only which SIDE of the range the bound is on.  It never
// decides which operand is the range: `idx > N` and `idx >= N` put the range on
// the left exactly like `idx < N` does, and exchanging the operands by predicate
// -- which this used to do -- is what made those legal forms fail the
// scalar-bound check in parseOrderedCmp.  triton_shared's parseCmp selects the
// range side by the operand for the same reason.
static bool normalizeOrderedCmp(arith::CmpIPredicate pred, bool &upperBound,
                                bool &adjustBound) {
  switch (pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    upperBound = true;
    return true;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    upperBound = true;
    adjustBound = true;
    return true;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    upperBound = false;
    adjustBound = true;
    return true;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    upperBound = false;
    return true;
  default:
    return false;
  }
}
static bool normalizeOrderedCmp(arith::CmpFPredicate pred, bool &upperBound,
                                bool &adjustBound) {
  switch (pred) {
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    upperBound = true;
    return true;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    upperBound = true;
    adjustBound = true;
    return true;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    upperBound = false;
    adjustBound = true;
    return true;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    upperBound = false;
    return true;
  default:
    return false;
  }
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  bool upperBound = false, adjustBound = false;
  if (!normalizeOrderedCmp(cmpOp.getPredicate(), upperBound, adjustBound)) {
    InFlightDiagnostic diag = emitError(loc) << "Unsupported cmpi";
    return failure();
  }
  return parseOrderedCmp(cmpOp.getLhs(), cmpOp.getRhs(), upperBound, adjustBound,
                         loc, builder);
}

LogicalResult MaskState::parseCmp(arith::CmpFOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  bool upperBound = false, adjustBound = false;
  if (!normalizeOrderedCmp(cmpOp.getPredicate(), upperBound, adjustBound)) {
    InFlightDiagnostic diag = emitError(loc) << "Unsupported cmpf";
    return failure();
  }
  return parseOrderedCmp(cmpOp.getLhs(), cmpOp.getRhs(), upperBound, adjustBound,
                         loc, builder);
}
LogicalResult MaskState::parseOrderedCmp(Value lowerValue, Value upperValue,
                                         bool upperBound, bool adjustBound,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  MaskState lhsState;
  if (failed(lhsState.parse(lowerValue, loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(upperValue, loc, builder)))
    return failure();

  // The range side is the operand that carries a range, and that is decided by
  // the operand itself rather than by the predicate.  `idx > N` puts the range
  // on the left exactly like `idx < N` does; only the side the bound sits on
  // differs, and that is already carried by upperBound/adjustBound.
  //
  // MaskState holds a const member, so the two sides cannot be assigned; the
  // range side is selected by pointer instead.
  MaskState *rangeState = &lhsState;
  MaskState *boundState = &rhsState;
  if (lhsState.scalar && !rhsState.scalar) {
    // The window is on the right (`N < idx` is `idx > N`): exchange the roles
    // and mirror the bound's sense.
    rangeState = &rhsState;
    boundState = &lhsState;
    upperBound = !upperBound;
    adjustBound = !adjustBound;
  }
  // The window model carries exactly one bound and it must be a scalar: a
  // comparison whose bound is itself a range (two vectors compared with each
  // other), or one with no range at all (two scalars), has no representation
  // here.  Reject it explicitly instead of feeding a null OpFoldResult into the
  // min/max/sub below, which used to corrupt the state or trip the assertion on
  // the next line.
  if (rangeState->scalar || !boundState->scalar) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported cmpi: the compared range must be bounded by a scalar";
    return failure();
  }

  // The range side must actually carry a window: an operand that is itself a
  // mask (`(a < b) < c`) has dims but no start/end, and a null OpFoldResult
  // reaching the min/max/sub below would corrupt the state rather than fail.
  if (!rangeState->start || !rangeState->end) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported cmpi: the compared range is not a make_range chain";
    return failure();
  }

  if (adjustBound)
    boundState->scalar =
        addOFRs(boundState->scalar, builder.getIndexAttr(1), loc, builder);

  int32_t cmpDim = -1;
  for (int32_t i = 0; i < rangeState->getRank(); i++) {
    auto dimIntAttr = getIntAttr(rangeState->dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = emitError(loc)
                                  << "Unsupported cmpi with more than one "
                                     "dimension with size larger than 1";
        return failure();
      }
      cmpDim = i;
    }
  }
  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");
  OpFoldResult newDim;
  OpFoldResult newOffset = builder.getIndexAttr(0);
  if (upperBound) {
    // An upper bound pulls the window end down to the scalar.  The clamp keeps
    // a bound below the range start from turning into a negative length:
    //   newEnd = min(end, bound), then max(newEnd, start)
    auto newEnd = minOFRs(rangeState->end, boundState->scalar, loc, builder);
    newEnd = maxOFRs(newEnd, rangeState->start, loc, builder);
    newDim = subOFRs(newEnd, rangeState->start, loc, builder);
  } else {
    // Mirror image: a lower bound pushes the window start up to the scalar,
    // clamped to the range end so a bound past the end collapses to an empty
    // window rather than a negative length.
    auto newStart =
        maxOFRs(rangeState->start, boundState->scalar, loc, builder);
    newStart = minOFRs(newStart, rangeState->end, loc, builder);
    newOffset = subOFRs(newStart, rangeState->start, loc, builder);
    newDim = subOFRs(rangeState->end, newStart, loc, builder);
  }

  SmallVector<OpFoldResult> rangeOffsets = rangeState->getWindowOffsets(builder);
  for (int32_t i = 0; i < rangeState->getRank(); i++) {
    if (i == cmpDim) {
      this->dims.push_back(newDim);
      this->offsets.push_back(newOffset);
    } else {
      this->dims.push_back(rangeState->dims[i]);
      this->offsets.push_back(rangeOffsets[i]);
    }
  }

  return success();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "stride must be 1 for make_range whose result is used "
           "as load or store masks";
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));

  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder)))
    return failure();

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      this->dims[i] = builder.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }

  return success();
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "splat source must be an integer scalar for load/store masks";
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  for (auto s : dstShape)
    this->dims.push_back(builder.getIndexAttr(s));

  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder)))
    return failure();

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));

  return success();
}

} // namespace triton
} // namespace mlir
