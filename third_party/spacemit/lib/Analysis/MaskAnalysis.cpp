//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/MaskAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>

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
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else if (!operand.getDefiningOp()) {
    return this->parseLoopIterArg(operand, loc, builder);
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
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));

  return tensor::ExtractSliceOp::create(builder, loc, source, offsets, dims,
                                        strides);
}

memref::SubViewOp MaskState::getSubview(Value source, const Location loc,
                                        OpBuilder &builder) const {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(getRank(), builder.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return memref::SubViewOp::create(builder, loc, cast<MemRefType>(dstType),
                                   source, offsets, dims, strides);
}

static memref::SubViewOp createSubview(Value src, Location loc, OpBuilder &b,
                                       ArrayRef<OpFoldResult> offsets,
                                       ArrayRef<OpFoldResult> sizes,
                                       ArrayRef<OpFoldResult> strides) {
  auto srcType = cast<MemRefType>(src.getType());
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return memref::SubViewOp::create(b, loc, cast<MemRefType>(dstType), src,
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
// If we simply take the subview of `buffer_tmp`, this requires an extra
// buffer to just hold the temporary result.
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
// Let (rowFull, colFull), (rowView1, colView1) and (rowView2, colView2) be
// the dimensions of the full subview, sv1, and sv2, respectively.
//
// + colView1 = min(colFull, col1)
// + colView2 = colFull - colView1
// + rowView1 = rowView2 = row = rowFull
std::pair<memref::SubViewOp, memref::SubViewOp>
MaskState::getSideBySideSubviews(Value block1, Value block2, const Location loc,
                                 OpBuilder &builder) const {
  OpFoldResult subviewRowFull = dims[0];
  OpFoldResult subviewColFull = dims[1];
  OpFoldResult col1 =
      memref::DimOp::create(builder, loc, block1, 1).getResult();
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
  OpFoldResult row1 =
      memref::DimOp::create(builder, loc, block1, 0).getResult();
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

LogicalResult MaskState::minStateScalar(const MaskState &lhsState,
                                        const MaskState &rhsState, Location loc,
                                        OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    dims.push_back(minOFRs(lhsState.dims[0], rhsState.dims[0], loc, builder));
  } else if (lhsState.scalar) {
    for (uint32_t i = 0; i < rhsState.getRank(); i++) {
      auto lhsDim = lhsState.dims[0];
      auto rhsDim = rhsState.dims[i];
      dims.push_back(minOFRs(lhsDim, rhsDim, loc, builder));
    }
  } else if (rhsState.scalar) {
    for (uint32_t i = 0; i < lhsState.getRank(); i++) {
      auto lhsDim = lhsState.dims[i];
      auto rhsDim = rhsState.dims[0];
      dims.push_back(minOFRs(lhsDim, rhsDim, loc, builder));
    }
  } else {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unexpected case where both lhs and rhs are not scalars";
    return failure();
  }
  return success();
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

  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    dims.push_back(minOFRs(lhsDim, rhsDim, loc, builder));
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
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

void MaskState::dump() const {
  llvm::dbgs() << "start: " << start << "\n";
  llvm::dbgs() << "end: " << end << "\n";
  llvm::dbgs() << "scalar: " << scalar << "\n";
  llvm::dbgs() << "useUnsafeMask: " << useUnsafeMask << "\n";
  llvm::dbgs() << "dims: ";
  for (auto dim : dims)
    llvm::dbgs() << "\t" << dim << "\n";
  llvm::dbgs() << "\n";
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

  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)))
    return failure();

  if (!lhsState.isMask() || !rhsState.isMask()) {
    return this->minStateScalar(lhsState, rhsState, loc, builder);
  }
  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseExtSI(arith::ExtSIOp op, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());
  return parse(op.getIn(), loc, builder);
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt &&
      cmpOp.getPredicate() != arith::CmpIPredicate::ult &&
      cmpOp.getPredicate() != arith::CmpIPredicate::sge) {
    InFlightDiagnostic diag = emitError(loc) << "Unsupported cmpi";
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder)))
    return failure();

  // We only support sge against 0 for lower bounds. Dims already has an
  // implicit assumption that the lower bound is 0, so if we see this, assume
  // the comparison evaluates to true.
  if (cmpOp.getPredicate() == arith::CmpIPredicate::sge &&
      !(rhsState.scalar && hasConstZero(rhsState.scalar))) {
    InFlightDiagnostic diag = emitError(loc)
                              << "Unsupported cmpi with rhs not equal to 0";
    return failure();
  }

  int32_t cmpDim = lhsState.scalar && rhsState.scalar ? 0 : -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
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
  assert(
      cmpDim != -1 ||
      (!lhsState.scalar && cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
       cmpOp.getPredicate() == arith::CmpIPredicate::ult) &&
          "Unexpected case where no dimension has size larger than 1");

  OpFoldResult newDim;
  if (lhsState.scalar) {
    assert(rhsState.scalar && "Unexpected case where rhs is not a scalar");
    // If both lhs and rhs are scalars, we can't just derive the dimension of
    // the mask as the minimum value: lhs/rhs could be 0 and then we don't
    // load/store anything.
    //
    // Instead treat the comparison as a scalar that determines if anything
    // should be loaded/stored by inserting a comparison + select:
    //    dim = lhs < rhs ? lhs.dim : 0
    newDim = compareOFRs(lhsState.scalar, rhsState.scalar, cmpOp.getPredicate(),
                         lhsState.dims[cmpDim], builder.getIndexAttr(0), loc,
                         builder);
  } else if (cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
             cmpOp.getPredicate() == arith::CmpIPredicate::ult) {
    // Important:
    // In the case where the values we are loading are entirely masked off like
    // the following:
    //
    // ---|-------|-----------|
    //    ^       ^           ^
    //   scalar  start       end
    //
    // newEnd = min(end, scalar) = scalar
    // Now scalar < start, so simply doing dim = newEnd - start is incorrect.
    //
    // The correct formula is to optionally move `newDim` back to `start` using
    // max(newEnd, start).
    auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
    newEnd = maxOFRs(newEnd, lhsState.start, loc, builder);
    newDim = subOFRs(newEnd, lhsState.start, loc, builder);
  } else {
    assert(cmpOp.getPredicate() == arith::CmpIPredicate::sge &&
           rhsState.scalar && hasConstZero(rhsState.scalar));
    newDim = lhsState.dims[cmpDim];
  }

  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    if (i == cmpDim)
      this->dims.push_back(newDim);
    else
      this->dims.push_back(lhsState.dims[i]);
  }

  return success();
}

LogicalResult MaskState::parseLoopIterArg(Value v, const Location loc,
                                          OpBuilder &builder) {
  assert(!v.getDefiningOp());

  auto forOp = llvm::dyn_cast<scf::ForOp>(v.getParentRegion()->getParentOp());

  if (!forOp) {
    return failure();
  }

  // TODO: This implementation does not work with nested loops
  if (forOp->getParentOfType<scf::ForOp>()) {
    return failure();
  }

  auto it = llvm::find(forOp.getRegionIterArgs(), v);
  if (it == forOp.getRegionIterArgs().end()) {
    return failure();
  }

  // This is a bit of a hack!!
  //
  // The offset (MaskState::start) of a mask can now depend on a loop's
  // iter-arg like the following example:
  //
  // idx = offset + tl.arange(0, 4)
  // for it in range(n):
  //   mask = idx < size
  //   x = tl.load(x_ptr + idx, mask=mask)
  //   tl.store(y_ptr + idx, x, mask=mask)
  //   idx += 4
  //
  // See
  // test/Conversion/TritonToStructured/mask_loop_iter_arg.mlir and
  // and
  // python/examples/test_mask_loop_iter_arg.py
  // for IR and full triton code.
  //
  // To support this case, we first make the following assumptions:
  //  - MaskAnalysis is runs after PtrAnalysis's prepass finishes, which means
  //    the offset for the load and store pointers have already been set up
  //    at `argIndex + 1`
  //  - The tensor of indices used by the load / store and the mask are the same
  //    (see above where `idx` appears in both the mask and the pointer
  //    arithmetic). This allows us to use the offset at `argIndex + 1` in the
  //    above assumption. In the future, to make this more robust, we need to
  //    verify that the offsets are indeed the same. Or alternatively, make sure
  //    to generate a separate start and end offset for each mask that is being
  //    updated in loops.
  //
  // Now to generate the mask state in each loop iteration, we first construct
  // the mask state *before* coming into the loop by parsing the init-arg. A
  // mask dimensions stay consistent throughout each loop iteration, but its
  // starting offset (`MaskState::start`) will change. So to construct the mask
  // state for each iteration, we need to make MaskState::state be the offset
  // iter-arg at `argIndex + 1`. Now for `MaskState::end`, we can first compute
  // the distance between `start` and `end` before coming into the loop, then
  // use this distance to compute the actual `end` in each loop.
  auto argIndex = std::distance(forOp.getRegionIterArgs().begin(), it);
  auto initArg = forOp.getInitArgs()[argIndex];
  if (auto getStateOp = initArg.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto tritonValue = getStateOp->getOperand(0);
    MaskState lhsState;

    {
      OpBuilder::InsertionGuard guard(builder);
      // Make sure all ops generated for the mask state are inserted before
      // the current loop
      builder.setInsertionPoint(forOp);
      if (failed(lhsState.parse(tritonValue, loc, builder))) {
        return failure();
      }
    }

    if (!lhsState.start && !lhsState.end) {
      assert(lhsState.scalar && "MaskState must have a scalar");
      lhsState.start = builder.getIndexAttr(0);
      lhsState.end = lhsState.scalar;
    }

    auto dist = subOFRs(lhsState.end, lhsState.start, loc, builder);
    this->start = forOp.getRegionIterArg(argIndex + 1);
    this->end = addOFRs(this->start, dist, loc, builder);
    this->dims = lhsState.dims;

    return success();
  }

  return failure();
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

  // Special handling for i1 (bool) splat: the mask represents an "all or
  // nothing" condition. Generate: dim[i] = select(bool, shape[i], 0)
  //
  // This ensures correct semantics when combined with another mask via AND:
  //   result[i] = min(select(bool, shape[i], 0), other[i])
  //
  // When bool=true:  min(shape[i], other[i]) = other[i] (since other[i] <=
  // shape[i]) When bool=false: min(0, other[i]) = 0
  //
  // This is the correct "all or nothing" behavior for row masks.
  auto srcType = cast<IntegerType>(src.getType());
  if (srcType.getWidth() == 1) {
    auto zeroAttr = builder.getIndexAttr(0);
    for (auto s : dstShape) {
      auto shapeAttr = builder.getIndexAttr(s);
      // Create: select(src, shape[i], 0)
      auto selectOp = arith::SelectOp::create(
          builder, loc, src, ofrToIndexValue(shapeAttr, loc, builder),
          ofrToIndexValue(zeroAttr, loc, builder));
      this->dims.push_back(selectOp.getResult());
    }
    return success();
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
