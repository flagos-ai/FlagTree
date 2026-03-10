/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string>

#include "Analysis/MaskAnalysis.h"

#include "Analysis/OpFoldResultUtils.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "mask-analysis"

namespace mlir {
namespace triton {
namespace gcu {

static int64_t kIndentSpaceNum = 0;

static void printBeforeVisit(Operation *op) {
  auto spaces = std::string(kIndentSpaceNum, ' ');
  kIndentSpaceNum += 3;

  LLVM_DEBUG({
    llvm::dbgs() << spaces << "=== visit operand of " << op->getName()
                 << " ENTER === \n";
    llvm::dbgs() << spaces;
    op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

static void printAfterVisit(Operation *op) {
  kIndentSpaceNum -= 3;
  auto spaces = std::string(kIndentSpaceNum, ' ');

  LLVM_DEBUG({
    llvm::dbgs() << spaces << "=== visit operand of " << op->getName()
                 << " EXIT ===\n";
  });

  if (kIndentSpaceNum == 0) {
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
}

void MaskState::addStateScalar(OpBuilder &builder, Location loc,
                               const MaskState &state,
                               const OpFoldResult scalar) {
  this->start = addOFRs(builder, loc, state.start, scalar);
  this->end = addOFRs(builder, loc, state.end, scalar);
  this->dims = state.dims;
}

void MaskState::addStates(OpBuilder &builder, Location loc,
                          const MaskState &lhsState,
                          const MaskState &rhsState) {
  assert(((lhsState.scalar && !rhsState.scalar) ||
          (!lhsState.scalar && rhsState.scalar) ||
          (lhsState.scalar && rhsState.scalar)) &&
         "unsupported scenario where neither lhs nor rhs is a scalar\n");

  if (lhsState.scalar && rhsState.scalar) {
    this->scalar = addOFRs(builder, loc, lhsState.scalar, rhsState.scalar);
    this->dims = lhsState.getRank() != 0 ? lhsState.dims : rhsState.dims;
    return;
  }

  if (lhsState.scalar)
    addStateScalar(builder, loc, rhsState, lhsState.scalar);
  else
    addStateScalar(builder, loc, lhsState, rhsState.scalar);
}

void MaskState::minStates(OpBuilder &builder, Location loc,
                          const MaskState &lhsState,
                          const MaskState &rhsState) {
  assert((lhsState.getRank() == rhsState.getRank()) &&
         "unexpected case where lhs and rhs have different ranks");

  for (int64_t i = 0; i < lhsState.getRank(); ++i) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    this->dims.push_back(minOFRs(builder, loc, lhsDim, rhsDim));
  }
}

void MaskState::setStates(OpBuilder &builder, Location loc,
                          const MaskState &srcState) {
  if (srcState.start)
    this->start = srcState.start;
  if (srcState.end)
    this->end = srcState.end;
  if (srcState.scalar)
    this->scalar = srcState.scalar;

  for (int64_t i = 0; i < srcState.getRank(); ++i) {
    this->dims.push_back(srcState.dims[i]);
  }
}

void MaskAnalysis::parse(OpBuilder &builder, Location loc, Value operand,
                         MaskState &state,
                         llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ') << "enter parse "
                          << operand << "\n");
  if (knownMasks.find(operand) != knownMasks.end()) {
    state = knownMasks.lookup(operand);
    LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                            << "operand is a known mask " << operand << "\n");
    return;
  }

  if (isa<IntegerType>(operand.getType())) {
    LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                            << "operand is an integer " << operand << "\n");
    parseIntScalar(builder, loc, operand, state, knownMasks);
    return;
  }

  if (auto arg = dyn_cast<BlockArgument>(operand)) {
    LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                            << "operand is block argument " << arg << "\n");
    parseBlockArgument(builder, loc, arg, state, knownMasks);
    return;
  }

  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    printBeforeVisit(op);
    parseConstant(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    printBeforeVisit(op);
    parseAdd(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    printBeforeVisit(op);
    parseAnd(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    printBeforeVisit(op);
    parseCmp(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    printBeforeVisit(op);
    parseMakeRange(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    printBeforeVisit(op);
    parseBroadcast(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    printBeforeVisit(op);
    parseSplat(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    printBeforeVisit(op);
    parseExpandDims(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::DotOp>()) {
    printBeforeVisit(op);
    parseDot(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    printBeforeVisit(op);
    parseRemsi(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::SelectOp>()) {
    printBeforeVisit(op);
    parseSelect(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::ReduceOp>()) {
    printBeforeVisit(op);
    parseReduce(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::LoadOp>()) {
    printBeforeVisit(op);
    parseLoad(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    printBeforeVisit(op);
    parseExtsi(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::ExtUIOp>()) {
    printBeforeVisit(op);
    parseExtui(builder, loc, op, state, knownMasks);
    printAfterVisit(op);
  } else {
    operand.dump();
    llvm_unreachable("unexpected to parse the operand\n");
  }
}

void MaskAnalysis::parseBlockArgument(
    OpBuilder &builder, Location loc, BlockArgument blockArg, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());
  assert(isa<scf::ForOp>(blockArg.getOwner()->getParentOp()));

  auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());

  if (blockArg.getArgNumber() == 0) {
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), forOp.getInductionVar());
    state.scalar = castOp.getResult();
  } else {
    auto regionIterIndex =
        blockArg.getArgNumber() - forOp.getNumInductionVars();
    Value regionArg = forOp.getRegionIterArgs()[regionIterIndex];
    assert(knownMasks.count(regionArg) != 0 &&
           "can't find value in knownMasks");
    state = knownMasks.lookup(regionArg);
  }
}

void MaskAnalysis::parseConstant(
    OpBuilder &builder, Location loc, arith::ConstantOp constOp,
    MaskState &state, llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    (void)elementType;
    assert(attr.isSplat() && isa<IntegerType>(elementType) &&
           "all elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    state.scalar = builder.getIndexAttr(value.getSExtValue());

    auto dst = constOp.getResult();
    auto dstShape = cast<ShapedType>(dst.getType()).getShape();
    for (auto s : dstShape)
      state.dims.push_back(builder.getIndexAttr(s));
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    state.scalar = builder.getIndexAttr(value);
  }
}

void MaskAnalysis::parseIntScalar(
    OpBuilder &builder, Location loc, Value scalar, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());
  // scalar is the first argument of ForOp, which means it is an induction var
  // if (auto arg = dyn_cast<BlockArgument>(scalar)) {
  //   if (isa<scf::ForOp>(arg.getOwner()->getParentOp()) &&
  //       arg.getArgNumber() == 0) {
  //     auto forOp = cast<scf::ForOp>(arg.getOwner()->getParentOp());
  //     auto castOp = builder.create<arith::IndexCastOp>(
  //         loc, builder.getIndexType(), forOp.getInductionVar());
  //     state.scalar = castOp.getResult();
  //     return;
  //   }
  // }
  auto castOp =
      builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
  state.scalar = castOp.getResult();
}

void MaskAnalysis::parseAdd(OpBuilder &builder, Location loc,
                            arith::AddIOp addOp, MaskState &state,
                            llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  MaskState lhsState;
  parse(builder, loc, addOp.getLhs(), lhsState, knownMasks);
  assert(!lhsState.isEmpty());

  MaskState rhsState;
  parse(builder, loc, addOp.getRhs(), rhsState, knownMasks);
  assert(!rhsState.isEmpty());

  state.addStates(builder, loc, lhsState, rhsState);
}

void MaskAnalysis::parseAnd(OpBuilder &builder, Location loc,
                            arith::AndIOp andOp, MaskState &state,
                            llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  MaskState lhsState;
  parse(builder, loc, andOp.getLhs(), lhsState, knownMasks);
  assert(!lhsState.isEmpty());

  MaskState rhsState;
  parse(builder, loc, andOp.getRhs(), rhsState, knownMasks);
  assert(!rhsState.isEmpty());

  state.minStates(builder, loc, lhsState, rhsState);
  if (lhsState.start && rhsState.start)
    state.start = maxOFRs(builder, loc, lhsState.start, rhsState.start);
  else if (lhsState.start)
    state.start = lhsState.start;
  else if (rhsState.start)
    state.start = rhsState.start;
}

void MaskAnalysis::parseCmp(OpBuilder &builder, Location loc,
                            arith::CmpIOp cmpOp, MaskState &state,
                            llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  assert(cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
         cmpOp.getPredicate() == arith::CmpIPredicate::ult ||
         cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
         cmpOp.getPredicate() == arith::CmpIPredicate::uge);

  MaskState lhsState;
  parse(builder, loc, cmpOp.getLhs(), lhsState, knownMasks);
  assert(!lhsState.isEmpty());

  MaskState rhsState;
  parse(builder, loc, cmpOp.getRhs(), rhsState, knownMasks);
  assert(!rhsState.isEmpty());

  // Process 1x1 tensor
  bool allOnes = true;
  for (int64_t i = 0; i < lhsState.getRank(); ++i) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    allOnes &= (dimIntAttr && (dimIntAttr.value() == 1));
  }
  if (allOnes) {
    assert((lhsState.scalar && rhsState.scalar) && "unsupported cmpi scenario");

    arith::CmpIOp cmpiOp = builder.create<arith::CmpIOp>(
        loc, cmpOp.getPredicate(), getValue(builder, loc, lhsState.scalar),
        getValue(builder, loc, rhsState.scalar));

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value result = builder.create<arith::SelectOp>(loc, cmpiOp, one, zero);
    for (int64_t i = 0; i < lhsState.getRank(); ++i) {
      state.dims.push_back(result);
    }
    return;
  }

  assert((!lhsState.scalar && rhsState.scalar) && "unsupported cmpi scenario");
  int64_t cmpDim = -1;
  for (int64_t i = 0; i < lhsState.getRank(); ++i) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      assert((cmpDim == -1) && "unsupported cmpi with more than one "
                               "dimension with size larger than 1");
      cmpDim = i;
    }
  }
  assert(cmpDim != -1 &&
         "unexpected case where no dimension has size larger than 1");

  auto newDim = lhsState.dims[cmpDim];
  if (cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
      cmpOp.getPredicate() == arith::CmpIPredicate::ult) {
    auto newEnd = minOFRs(builder, loc, lhsState.end, rhsState.scalar);
    newDim = subOFRs(builder, loc, newEnd, lhsState.start);
  } else {
    auto newstart = maxOFRs(builder, loc, lhsState.start, rhsState.scalar);
    state.start = newstart;
  }

  for (int64_t i = 0; i < lhsState.getRank(); ++i) {
    if (i == cmpDim)
      state.dims.push_back(newDim);
    else
      state.dims.push_back(lhsState.dims[i]);
  }
}

void MaskAnalysis::parseMakeRange(
    OpBuilder &builder, Location loc, triton::MakeRangeOp rangeOp,
    MaskState &state, llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  (void)stride;

  assert((stride == 1) &&
         "stride must be 1 for make_range whose result is used "
         "as load or store masks");

  state.start = builder.getIndexAttr(start);
  state.end = builder.getIndexAttr(end);
  state.dims.push_back(builder.getIndexAttr(shape[0]));
}

void MaskAnalysis::parseBroadcast(
    OpBuilder &builder, Location loc, triton::BroadcastOp broadcastOp,
    MaskState &state, llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  parse(builder, loc, src, state, knownMasks);

  for (uint64_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      state.dims[i] = builder.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast\n");
  }
}

void MaskAnalysis::parseSplat(
    OpBuilder &builder, Location loc, triton::SplatOp splatOp, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  assert(isa<IntegerType>(splatOp.getSrc().getType()) &&
         "splat source must be an integer scalar for load/store masks");

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  // splat bool means either all or none are masked
  if (src.getType().getIntOrFloatBitWidth() == 1) {
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (auto s : dstShape) {
      Value dim = builder.create<arith::ConstantIndexOp>(loc, s);
      Value result = builder.create<arith::SelectOp>(loc, src, dim, zero);
      state.dims.push_back(result);
    }
  } else {
    parse(builder, loc, src, state, knownMasks);
    for (auto s : dstShape)
      state.dims.push_back(builder.getIndexAttr(s));
  }
}

void MaskAnalysis::parseExpandDims(
    OpBuilder &builder, Location loc, triton::ExpandDimsOp expandDimsOp,
    MaskState &state, llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  parse(builder, loc, expandDimsOp.getSrc(), state, knownMasks);
  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  (void)dstShape;
  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  state.dims.insert(state.dims.begin() + axis, builder.getIndexAttr(1));
}

void MaskAnalysis::parseDot(OpBuilder &builder, Location loc,
                            triton::DotOp dotOp, MaskState &state,
                            llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  MaskState srcState;
  parse(builder, loc, dotOp.getC(), srcState, knownMasks);
  assert(!srcState.isEmpty());

  state.start = srcState.start;
  state.end = srcState.end;
  state.scalar = srcState.scalar;
  for (int64_t i = 0; i < srcState.getRank(); ++i)
    state.dims.push_back(srcState.dims[i]);
}

void MaskAnalysis::parseRemsi(
    OpBuilder &builder, Location loc, arith::RemSIOp RemSIOp, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  MaskState lhsState;
  parse(builder, loc, RemSIOp.getLhs(), lhsState, knownMasks);
  assert(!lhsState.isEmpty());

  MaskState rhsState;
  parse(builder, loc, RemSIOp.getRhs(), rhsState, knownMasks);
  assert(!rhsState.isEmpty());

  state.addStates(builder, loc, lhsState, rhsState);
}

void MaskAnalysis::parseSelect(
    OpBuilder &builder, Location loc, arith::SelectOp SelectOp,
    MaskState &state, llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());
  MaskState trueState;
  parse(builder, loc, SelectOp.getTrueValue(), trueState, knownMasks);
  assert(!trueState.isEmpty());

  state.setStates(builder, loc, trueState);
}

void MaskAnalysis::parseReduce(
    OpBuilder &builder, Location loc, triton::ReduceOp ReduceOp,
    MaskState &state, llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());
  auto src = ReduceOp.getSrcs()[0];
  auto axis = ReduceOp.getAxis();

  MaskState srcState;
  parse(builder, loc, src, srcState, knownMasks);

  if (srcState.start)
    state.start = srcState.start;
  if (srcState.end)
    state.end = srcState.end;
  if (srcState.scalar)
    state.scalar = srcState.scalar;
  for (uint32_t i = 0; i < srcState.dims.size(); ++i) {
    if (i != axis) {
      state.dims.push_back(srcState.dims[i]);
    }
  }
}

void MaskAnalysis::parseLoad(
    OpBuilder &builder, Location loc, triton::LoadOp LoadOp, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());

  MaskState srcState;
  parse(builder, loc, LoadOp.getPtr(), srcState, knownMasks);
  assert(!srcState.isEmpty());

  state.setStates(builder, loc, srcState);
}

void MaskAnalysis::parseExtsi(
    OpBuilder &builder, Location loc, arith::ExtSIOp ExtSIOp, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());
  MaskState inState;

  parse(builder, loc, ExtSIOp.getIn(), inState, knownMasks);
  assert(!inState.isEmpty());

  state.setStates(builder, loc, inState);
}

void MaskAnalysis::parseExtui(
    OpBuilder &builder, Location loc, arith::ExtUIOp ExtUIOp, MaskState &state,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  assert(state.isEmpty());
  MaskState inState;

  parse(builder, loc, ExtUIOp.getIn(), inState, knownMasks);
  assert(!inState.isEmpty());

  state.setStates(builder, loc, inState);
}

} // namespace gcu
} // namespace triton
} // namespace mlir
