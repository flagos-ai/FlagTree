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
#include <utility>

#include "Analysis/PtrAnalysis.h"

#include "Analysis/AxisInfoEx.h"
#include "Analysis/MaskAnalysis.h"
#include "Analysis/OpFoldResultUtils.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "ptr-analysis"

namespace mlir {
namespace triton {
namespace gcu {

static llvm::DenseSet<Operation *> addedAssertOps;
static int64_t kIndentSpaceNum = 0;

static void printBeforeVisit(Operation *op) {
  auto spaces = std::string(kIndentSpaceNum, ' ');
  kIndentSpaceNum += 3;

  LLVM_DEBUG({
    llvm::dbgs() << spaces << "=== visit operand of " << op->getName()
                 << " ENTER ===\n";
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

int64_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

void PtrState::addState(OpBuilder &builder, Location loc,
                        const PtrState &lhsState, const PtrState &rhsState) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  // at most one of lhs and rhs should have valid source, since otherwise we
  // will be losing information
  assert(!(lhsState.source && rhsState.source));
  this->source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    this->scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) {
    this->scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  for (uint64_t i = 0; i < lhsState.sizes.size(); ++i) {
    auto newOffset =
        addOFRs(builder, loc, lhsState.offsets[i], rhsState.offsets[i]);
    this->offsets.push_back(newOffset);

    auto newStride =
        addOFRs(builder, loc, lhsState.strides[i], rhsState.strides[i]);
    this->strides.push_back(newStride);

    this->sizes.push_back(lhsState.sizes[i]);
  }
}

void PtrState::mulState(OpBuilder &builder, Location loc,
                        const PtrState &lhsState, const PtrState &rhsState) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  if (lhsState.scalar && rhsState.scalar) {
    LLVM_DEBUG(llvm::dbgs() << "both PtrStates are scalars\n");
    for (uint64_t i = 0; i < lhsState.sizes.size(); ++i) {
      this->offsets.push_back(
          mulOFRValue(builder, loc, lhsState.offsets[i], rhsState.scalar));
      this->strides.push_back(
          mulOFRValue(builder, loc, lhsState.strides[i], rhsState.scalar));
      this->sizes.push_back(lhsState.sizes[i]);
    }
    return;
  }

  bool rhsScalar = true;
  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  assert(!(lhsState.source && rhsState.source));
  this->source = lhsState.source ? lhsState.source : rhsState.source;

  assert((lhsState.scalar || rhsState.scalar) &&
         !(lhsState.scalar && rhsState.scalar) &&
         "currently does not support both tensors are effectively non-scalar");
  if (!rhsState.scalar && lhsState.scalar)
    rhsScalar = false;

  for (uint64_t i = 0; i < lhsState.sizes.size(); ++i) {
    OpFoldResult newOffset;
    OpFoldResult newStride;
    if (rhsScalar) {
      newOffset =
          mulOFRValue(builder, loc, lhsState.offsets[i], rhsState.scalar);
      newStride =
          mulOFRValue(builder, loc, lhsState.strides[i], rhsState.scalar);
    } else {
      newOffset =
          mulOFRValue(builder, loc, rhsState.offsets[i], lhsState.scalar);
      newStride =
          mulOFRValue(builder, loc, rhsState.strides[i], lhsState.scalar);
    }
    this->offsets.push_back(newOffset);
    this->strides.push_back(newStride);
    this->sizes.push_back(lhsState.sizes[i]);
  }
}

void PtrState::remState(OpBuilder &builder, Location loc,
                        const PtrState &lhsState, const PtrState &rhsState) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  assert(!lhsState.source && !rhsState.source);

  this->source = lhsState.source;
  for (uint64_t i = 0; i < lhsState.sizes.size(); ++i) {
    auto newOffset =
        remOFRs(builder, loc, lhsState.offsets[i], rhsState.offsets[i]);
    this->offsets.push_back(newOffset);

    this->strides.push_back(lhsState.strides[i]);
    this->sizes.push_back(lhsState.sizes[i]);
  }
}

void PtrState::divState(OpBuilder &builder, Location loc,
                        const PtrState &lhsState, const PtrState &rhsState) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  assert(!lhsState.source && !rhsState.source);

  this->source = lhsState.source;
  for (uint64_t i = 0; i < lhsState.sizes.size(); ++i) {
    auto newOffset =
        divOFRs(builder, loc, lhsState.offsets[i], rhsState.offsets[i]);
    this->offsets.push_back(newOffset);

    this->strides.push_back(lhsState.strides[i]);
    this->sizes.push_back(lhsState.sizes[i]);
  }
}

void PtrState::setState(OpBuilder &builder, Location loc,
                        const PtrState &srcState) {
  if (srcState.source)
    this->source = srcState.source;
  if (srcState.scalar)
    this->scalar = srcState.scalar;
  for (uint64_t i = 0; i < srcState.sizes.size(); ++i) {
    this->offsets.push_back(srcState.offsets[i]);
    this->strides.push_back(srcState.strides[i]);
    this->sizes.push_back(srcState.sizes[i]);
  }
}

bool isZeroStride(OpBuilder &builder, Location loc, const OpFoldResult ofr) {
  if (auto attr = getIntAttr(ofr)) {
    return attr.value() == 0;
  }

  // Assume stride can not be changed, if it is an argument of ForOp and it is
  // value is 0, then take it as zero stride.
  auto value = getValue(builder, loc, ofr);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
      auto argIndex = arg.getArgNumber() - forOp.getNumInductionVars();
      assert(argIndex < forOp.getInitArgs().size());

      auto initArg = forOp.getInitArgs()[argIndex];
      if (auto constOp =
              dyn_cast<arith::ConstantIndexOp>(initArg.getDefiningOp())) {
        return constOp.value() == 0;
      }
    }
  }

  return false;
}

PtrInfo PtrState::getPtrInfo(OpBuilder &builder, Location loc,
                             const MaskState &mstate) {
  PtrInfo ptrInfo;

  assert(isa<triton::PointerType>(this->source.getType()));
  auto elemType =
      cast<triton::PointerType>(this->source.getType()).getPointeeType();
  auto bpe = elemType.getIntOrFloatBitWidth() / 8;

  auto offsets = getValues(builder, loc, this->offsets);
  auto strides = getValues(builder, loc, this->strides);
  auto sizes = getValues(builder, loc, this->sizes);

  auto rank = getRank();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto addr = builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(),
                                                 this->source);
  auto base = builder.create<arith::AddIOp>(
      loc, addr,
      builder.create<arith::MulIOp>(
          loc, builder.create<arith::ConstantIntOp>(loc, bpe, /*width=*/64),
          builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                             offsets[0])));
  if (rank == 1) {
    ptrInfo.base = builder.create<IntToPtrOp>(
        loc, PtrType::get(builder.getContext(), elemType), base.getResult());
    ptrInfo.shape.push_back(
        mstate.isEmpty() ? sizes[0] : getValues(builder, loc, mstate.dims)[0]);

    ptrInfo.offsets.push_back(zero);
    if (!isZeroStride(builder, loc, this->strides[0])) {
      ptrInfo.strides.push_back(strides[0]);
    } else {
      ptrInfo.strides.push_back(one);
      ptrInfo.broadcastDims.insert(0);
    }
  } else if (rank >= 2 && rank <= 4) {
    for (int i = 1; i < rank; ++i) {
      base = builder.create<arith::AddIOp>(
          loc, base,
          builder.create<arith::MulIOp>(
              loc, builder.create<arith::ConstantIntOp>(loc, bpe, /*width=*/64),
              builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                 offsets[i])));
    }
    ptrInfo.base = builder.create<IntToPtrOp>(
        loc, PtrType::get(builder.getContext(), elemType), base.getResult());
    for (int i = rank - 1; i >= 0; --i) {
      ptrInfo.offsets.push_back(zero);

      auto shapeBegin = ptrInfo.shape.begin();
      if (!mstate.isEmpty()) {
        ptrInfo.shape.insert(shapeBegin,
                             getValues(builder, loc, mstate.dims)[i]);
      } else {
        ptrInfo.shape.insert(shapeBegin, sizes[i]);
      }

      auto strideBegin = ptrInfo.strides.begin();
      if (!isZeroStride(builder, loc, this->strides[i])) {
        ptrInfo.strides.insert(strideBegin, strides[i]);
      } else {
        ptrInfo.broadcastDims.insert(i);
        ptrInfo.strides.insert(strideBegin, zero);
      }
    }
  } else {
    // not support
    assert(false && "not support rank >= 5");
  }
  return ptrInfo;
}

void PtrAnalysis::visitOperand(
    PatternRewriter &rewriter, Location loc, Value operand, PtrState &state,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                          << "enter visitOperand" << operand << "\n");
  if (knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                            << "operand is a known ptr " << operand << "\n");
    return;
  }
  if (isa<IntegerType>(operand.getType())) {
    auto castOp = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), operand);
    state.scalar = castOp.getResult();
    LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                            << "operand is an integer " << operand << "\n");
    return;
  }

  if (auto arg = dyn_cast<BlockArgument>(operand)) {
    LLVM_DEBUG(llvm::dbgs() << std::string(kIndentSpaceNum, ' ')
                            << "operand is block argument " << arg << "\n");
    visitBlockArgument(rewriter, loc, arg, state, knownPtrs);
    return;
  }
  // Supported ops
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    printBeforeVisit(op);
    visitOperandConstSplat(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    printBeforeVisit(op);
    visitOperandAdd(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    printBeforeVisit(op);
    visitOperandMul(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    printBeforeVisit(op);
    visitOperandRem(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    printBeforeVisit(op);
    visitOperandDiv(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::SelectOp>()) {
    printBeforeVisit(op);
    visitOperandSelect(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    printBeforeVisit(op);
    visitOperandExtsi(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<arith::ExtUIOp>()) {
    printBeforeVisit(op);
    visitOperandExtui(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    printBeforeVisit(op);
    visitOperandMakeRange(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    printBeforeVisit(op);
    visitOperandBroadcast(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    printBeforeVisit(op);
    visitOperandSplat(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    printBeforeVisit(op);
    visitOperandExpandDims(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    printBeforeVisit(op);
    visitOperandAddptr(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::BitcastOp>()) {
    printBeforeVisit(op);
    visitOperandBitcast(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::TransOp>()) {
    printBeforeVisit(op);
    visitOperandTrans(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::DotOp>()) {
    printBeforeVisit(op);
    visitOperandDot(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::ReduceOp>()) {
    printBeforeVisit(op);
    visitOperandReduce(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else if (auto op = operand.getDefiningOp<triton::LoadOp>()) {
    printBeforeVisit(op);
    visitOperandLoad(rewriter, loc, op, state, knownPtrs);
    printAfterVisit(op);
  } else {
    operand.dump();
    llvm_unreachable("unexpected to visit the operand\n");
  }
}

void PtrAnalysis::visitBlockArgument(
    PatternRewriter &rewriter, Location loc, BlockArgument blockArg,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  assert(!isa<scf::ForOp>(blockArg.getOwner()->getParentOp()));
  state.source = blockArg;
}

void PtrAnalysis::visitOperandConstSplat(
    PatternRewriter &rewriter, Location loc, arith::ConstantOp op,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  assert(attr.isSplat() && isa<IntegerType>(attr.getElementType()));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = rewriter.getIndexAttr(value.getSExtValue());
  auto constOp = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(), constAttr);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (uint64_t i = 0; i < resultType.getShape().size(); ++i) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(rewriter.getIndexAttr(0));
    }

    state.sizes.push_back(rewriter.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(rewriter.getIndexAttr(0));
  }
}

void PtrAnalysis::visitOperandAdd(
    PatternRewriter &rewriter, Location loc, arith::AddIOp addOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(rewriter, loc, addOp.getLhs(), lhsState, knownPtrs);

  PtrState rhsState;
  visitOperand(rewriter, loc, addOp.getRhs(), rhsState, knownPtrs);

  state.addState(rewriter, loc, lhsState, rhsState);
}

void PtrAnalysis::visitOperandMul(
    PatternRewriter &rewriter, Location loc, arith::MulIOp mulOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(rewriter, loc, mulOp.getLhs(), lhsState, knownPtrs);

  PtrState rhsState;
  visitOperand(rewriter, loc, mulOp.getRhs(), rhsState, knownPtrs);

  state.mulState(rewriter, loc, lhsState, rhsState);
}

void PtrAnalysis::visitOperandRem(
    PatternRewriter &rewriter, Location loc, arith::RemSIOp remOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(rewriter, loc, remOp.getLhs(), lhsState, knownPtrs);

  PtrState rhsState;
  visitOperand(rewriter, loc, remOp.getRhs(), rhsState, knownPtrs);

  state.remState(rewriter, loc, lhsState, rhsState);
}

void PtrAnalysis::visitOperandDiv(
    PatternRewriter &rewriter, Location loc, arith::DivSIOp divOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  PtrState lhsState;
  visitOperand(rewriter, loc, divOp.getLhs(), lhsState, knownPtrs);

  PtrState rhsState;
  visitOperand(rewriter, loc, divOp.getRhs(), rhsState, knownPtrs);

  state.divState(rewriter, loc, lhsState, rhsState);
}

void PtrAnalysis::visitOperandSelect(
    PatternRewriter &rewriter, Location loc, arith::SelectOp selectOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  PtrState trueState;
  visitOperand(rewriter, loc, selectOp.getTrueValue(), trueState, knownPtrs);

  PtrState falseState;
  visitOperand(rewriter, loc, selectOp.getFalseValue(), falseState, knownPtrs);

  // now selectop is bypass, the state is unuse; In the future, we will analyze
  // it under certain constraints.
  state.setState(rewriter, loc, trueState);
}

void PtrAnalysis::visitOperandMakeRange(
    PatternRewriter &rewriter, Location loc, triton::MakeRangeOp rangeOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  state.offsets.push_back(rewriter.getIndexAttr(start));
  state.sizes.push_back(rewriter.getIndexAttr(shape[0]));
  state.strides.push_back(rewriter.getIndexAttr(stride));
}

void PtrAnalysis::visitOperandExpandDims(
    PatternRewriter &rewriter, Location loc, triton::ExpandDimsOp expandDimsOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  visitOperand(rewriter, loc, expandDimsOp.getSrc(), state, knownPtrs);

  auto axis = expandDimsOp.getAxis();

  assert(
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape()[axis] ==
          1 &&
      "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  state.offsets.insert(state.offsets.begin() + axis, rewriter.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, rewriter.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, rewriter.getIndexAttr(0));
}

void PtrAnalysis::visitOperandBroadcast(
    PatternRewriter &rewriter, Location loc, triton::BroadcastOp broadcastOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  visitOperand(rewriter, loc, src, state, knownPtrs);

  for (uint64_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      state.sizes[i] = rewriter.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }
}

void PtrAnalysis::visitOperandSplat(
    PatternRewriter &rewriter, Location loc, triton::SplatOp splatOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  visitOperand(rewriter, loc, src, state, knownPtrs);

  if (isa<IntegerType>(src.getType()) ||
      isa<triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.offsets.push_back(rewriter.getIndexAttr(0));
      state.sizes.push_back(rewriter.getIndexAttr(s));
      state.strides.push_back(rewriter.getIndexAttr(0));
    }
  } else {
    llvm_unreachable("unexpected src type used in splat");
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar) {
    state.offsets[0] = state.scalar;
  }
}

void PtrAnalysis::visitOperandAddptr(
    PatternRewriter &rewriter, Location loc, triton::AddPtrOp addptrOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());

  PtrState ptrState;
  visitOperand(rewriter, loc, addptrOp.getPtr(), ptrState, knownPtrs);

  PtrState offsetState;
  visitOperand(rewriter, loc, addptrOp.getOffset(), offsetState, knownPtrs);

  assert(ptrState.source && "ptr field should provide source / base pointer");
  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  state.addState(rewriter, loc, ptrState, offsetState);
}

void PtrAnalysis::visitOperandBitcast(
    PatternRewriter &rewriter, Location loc, triton::BitcastOp bitcastOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  auto src = bitcastOp.getSrc();
  auto dst = bitcastOp.getResult();
  if ((!state.source) && isa<triton::PointerType>(src.getType())) {
    assert(isa<triton::PointerType>(dst.getType()) &&
           "input is pointer output should also a pointer");
    PtrState srcState;
    visitOperand(rewriter, loc, src, srcState, knownPtrs);
    for (uint64_t i = 0; i < srcState.sizes.size(); ++i) {
      state.offsets.push_back(srcState.offsets[i]);
      state.sizes.push_back(srcState.sizes[i]);
      state.strides.push_back(srcState.strides[i]);
    }
    state.scalar = srcState.scalar;

    auto newBitcastOp =
        rewriter.create<triton::BitcastOp>(loc, dst.getType(), srcState.source);
    state.source = newBitcastOp.getResult();
    return;
  }
  assert(isa<ShapedType>(src.getType()) &&
         "input to tt.bitcast should be a tensor");

  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);

  auto dstElemType = cast<ShapedType>(dst.getType()).getElementType();
  assert(isa<triton::PointerType>(
      cast<ShapedType>(src.getType()).getElementType()));

  for (uint64_t i = 0; i < srcState.sizes.size(); ++i) {
    state.offsets.push_back(srcState.offsets[i]);
    state.sizes.push_back(srcState.sizes[i]);
    state.strides.push_back(srcState.strides[i]);
  }
  state.scalar = srcState.scalar;

  auto newBitcastOp =
      rewriter.create<triton::BitcastOp>(loc, dstElemType, srcState.source);
  state.source = newBitcastOp.getResult();
}

void PtrAnalysis::visitOperandTrans(
    PatternRewriter &rewriter, Location loc, triton::TransOp transOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  auto src = transOp.getSrc();

  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);
  llvm::ArrayRef<int32_t> transOrder = transOp.getOrder();

  for (uint64_t i = 0; i < transOrder.size(); ++i) {
    state.offsets.push_back(srcState.offsets[transOrder[i]]);
    state.sizes.push_back(srcState.sizes[transOrder[i]]);
    state.strides.push_back(srcState.strides[transOrder[i]]);
  }
  state.scalar = srcState.scalar;
  state.source = srcState.source;
}

void PtrAnalysis::visitOperandDot(
    PatternRewriter &rewriter, Location loc, triton::DotOp dotOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  auto src = dotOp.getC();

  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);

  state.setState(rewriter, loc, srcState);
}

void PtrAnalysis::visitOperandReduce(
    PatternRewriter &rewriter, Location loc, triton::ReduceOp reduceOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  auto src = reduceOp.getSrcs()[0];
  auto axis = reduceOp.getAxis();

  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);

  state.scalar = srcState.scalar;
  state.source = srcState.source;
  for (uint32_t i = 0; i < srcState.offsets.size(); ++i) {
    if (i != axis) {
      state.offsets.push_back(srcState.offsets[i]);
      state.sizes.push_back(srcState.sizes[i]);
      state.strides.push_back(srcState.strides[i]);
    }
  }
}

void PtrAnalysis::visitOperandLoad(
    PatternRewriter &rewriter, Location loc, triton::LoadOp loadOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {

  auto src = loadOp.getPtr();
  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);

  state.setState(rewriter, loc, srcState);
}

void PtrAnalysis::visitOperandExtsi(
    PatternRewriter &rewriter, Location loc, arith::ExtSIOp extsiOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  auto src = extsiOp.getIn();

  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);

  state.setState(rewriter, loc, srcState);
}

void PtrAnalysis::visitOperandExtui(
    PatternRewriter &rewriter, Location loc, arith::ExtUIOp extuiOp,
    PtrState &state, llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  assert(state.isEmpty());
  auto src = extuiOp.getIn();

  PtrState srcState;
  visitOperand(rewriter, loc, src, srcState, knownPtrs);

  state.setState(rewriter, loc, srcState);
}

bool isPtrFromLoad(Value v, llvm::DenseMap<Value, bool> &valueFromLoads);
bool isMaskCandidate(Value v, llvm::DenseMap<Value, bool> &valueToCandiates);

void PtrAnalysis::rewriteYieldOp(
    PatternRewriter &rewriter, scf::YieldOp op,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs,
    llvm::SmallDenseMap<Value, MaskState> &knownMasks) {
  // any inserted instruction should be before this yield
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto adaptor = scf::YieldOp::Adaptor(op);
  llvm::SmallVector<PtrState> yieldArgState;
  llvm::SmallVector<MaskState> yieldArgMaskState;
  llvm::SmallVector<Value> operands(adaptor.getOperands());

  LLVM_DEBUG(llvm::dbgs() << "ptr rewriteYieldOp start: \n");
  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // PtrState for those values.
  for (auto [i, v] : llvm::enumerate(operands)) {
    (void)i;
    auto tType = dyn_cast<TensorType>(v.getType());
    if (tType && ((tType.getElementType().isIntOrIndex() &&
                   !tType.getElementType().isInteger(1)) ||
                  isa<triton::PointerType>(tType.getElementType()))) {
      llvm::DenseMap<Value, bool> valueFromLoads;
      if (!isPtrFromLoad(v, valueFromLoads)) {
        PtrState state;
        visitOperand(rewriter, op.getLoc(), v, state, knownPtrs);
        yieldArgState.push_back(state);
        LLVM_DEBUG(llvm::dbgs() << "ptr yieldArgState size:"
                                << yieldArgState.size() << "\n");
      }
    }
  }

  // mask info process
  for (auto [i, v] : llvm::enumerate(operands)) {
    auto tType = dyn_cast<TensorType>(v.getType());
    if (tType && ((tType.getElementType().isIntOrIndex() &&
                   !tType.getElementType().isInteger(1)))) {
      llvm::DenseMap<Value, bool> valueToCandiates;
      if (isMaskCandidate(v, valueToCandiates)) {
        MaskState state;
        gcu::MaskAnalysis::parse(rewriter, op.getLoc(), v, state, knownMasks);
        yieldArgMaskState.push_back(state);
        LLVM_DEBUG(llvm::dbgs() << "ptr yieldArgMaskState size:"
                                << yieldArgMaskState.size() << "\n");
      }
    }
    (void)i;
  }

  // For each of the PtrState recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : yieldArgState) {
    if (state.scalar) {
      operands.push_back(state.scalar);
    }
    for (auto s : state.offsets) {
      // offsets can be IntAttr zeroes, since reinterpret_cast collapses them
      // for the input memref, and the for loop may not update offsets other
      // than offsets[0]. Create constants Values for those zeroes.
      if (auto sIntAttr = getIntAttr(s)) {
        assert(sIntAttr.value() == 0 && "attribute offsets should be zeroes");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(s.get<Value>());
      }
    }

    for (auto s : state.strides) {
      assert(!getIntAttr(s) &&
             "PtrState strides for yield within for loop not expected to be "
             "attribute.");
      operands.push_back(s.get<Value>());
    }
  }

  // mask info process
  for (auto state : yieldArgMaskState) {
    if (state.start) {
      auto sIntAttr = getIntAttr(state.start);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        operands.push_back(constOp.getResult());
        state.start = constOp.getResult();
      } else {
        operands.push_back(state.start.get<Value>());
      }
    }
    if (state.end) {
      auto sIntAttr = getIntAttr(state.end);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        operands.push_back(constOp.getResult());
        state.end = constOp.getResult();
      } else {
        operands.push_back(state.end.get<Value>());
      }
    }
  }

  // Yield is a terminator op that must be at the end of the function
  rewriter.setInsertionPointAfter(op);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
  assert(op->getNumResults() == 0);
}

bool PtrAnalysis::byPassForOp(PatternRewriter &rewriter, scf::ForOp op,
                              const SmallVector<Operation *, 8> &candidateOps) {
  bool bypass = true;

  op.walk<WalkOrder::PreOrder>([&](mlir::Operation *_op) {
    bypass = mlir::TypeSwitch<mlir::Operation *, bool>(_op)
                 .Case<triton::LoadOp, triton::StoreOp>([&](auto loadstoreOp) {
                   auto iter =
                       std::find(candidateOps.begin(), candidateOps.end(),
                                 loadstoreOp.getOperation());
                   return iter == candidateOps.end();
                 })
                 .Default([&](auto op) { return true; });
    return !bypass ? WalkResult::interrupt() : WalkResult::advance();
  });

  return bypass;
}

LogicalResult PtrAnalysis::rewriteForOp(
    PatternRewriter &rewriter, scf::ForOp op,
    SmallDenseMap<Value, PtrState> &knownPtrs,
    SmallDenseMap<Value, MaskState> &knownMasks,
    SmallVector<Operation *, 8> &candidateOps,
    SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints) {
  llvm::SmallVector<Value> newInitArgs;
  llvm::SmallVector<std::pair<int, PtrState>> initArgIndexState;
  llvm::SmallVector<std::pair<int, MaskState>> initmaskIndexState;

  LLVM_DEBUG(llvm::dbgs() << "rewriteForOp: \n");
  // Create a new list of init args
  for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
    newInitArgs.push_back(arg);
    // Only parse those args whose type is ptr or int tensor, since they will
    // be possible of operands of triton::AddPtrOp
    LLVM_DEBUG(llvm::dbgs() << "i: " << i << "\n");
    auto tType = dyn_cast<TensorType>(arg.getType());
    if (tType && ((tType.getElementType().isIntOrIndex() &&
                   !tType.getElementType().isInteger(1)) ||
                  isa<triton::PointerType>(tType.getElementType()))) {
      llvm::DenseMap<Value, bool> valueFromLoads;
      if (!isPtrFromLoad(op.getRegionIterArg(i), valueFromLoads)) {
        PtrState state;
        visitOperand(rewriter, op.getLoc(), arg, state, knownPtrs);
        // Record the PtrState for later processing
        initArgIndexState.push_back(std::make_pair(i, state));
      }
    }
  }
  // mask info process
  for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
    auto tType = dyn_cast<TensorType>(arg.getType());
    if (tType && ((tType.getElementType().isIntOrIndex() &&
                   !tType.getElementType().isInteger(1)))) {
      llvm::DenseMap<Value, bool> valueToCandiates;
      if (isMaskCandidate(op.getRegionIterArg(i), valueToCandiates)) {
        MaskState state;
        gcu::MaskAnalysis::parse(rewriter, op.getLoc(), arg, state, knownMasks);
        initmaskIndexState.push_back(std::make_pair(i, state));
      }
    }
  }

  if (initmaskIndexState.size() == 0 && initArgIndexState.size() == 0)
    return failure();

  // Set insertion point to be before the for loop for new variables passed
  // into the new loop.
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  // For each of the PtrState recorded in the last step, insert new
  // instructions to describe offset and stride for each dimension and append
  // them to init args
  for (auto [i, state] : initArgIndexState) {
    // For each dimension, if the corresponding offset and stride is an
    // integer attribute, create a constant value and append them at the end
    // of init arg list.
    (void)i;
    if (state.scalar) {
      newInitArgs.push_back(state.scalar);
    }
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.offsets[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(s.get<Value>());
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.strides[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(s.get<Value>());
      }
    }
  }

  // mask info process
  for (auto [i, state] : initmaskIndexState) {
    if (state.start) {
      auto sIntAttr = getIntAttr(state.start);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.start = constOp.getResult();
      } else {
        newInitArgs.push_back(state.start.get<Value>());
      }
    }

    if (state.end) {
      auto sIntAttr = getIntAttr(state.end);
      if (sIntAttr) {
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.end = constOp.getResult();
      } else {
        newInitArgs.push_back(state.end.get<Value>());
      }
    }
    (void)i;
  }
  rewriter.restoreInsertionPoint(origIp);

  // Create a new scf::ForOp that uses updated init args and same loop body
  auto newOp = rewriter.create<scf::ForOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
      newInitArgs,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {
        IRMapping mapping;
        mapping.map(op.getInductionVar(), iv);
        mapping.map(op.getInitArgs(), newInitArgs);
        mapping.map(op.getRegionIterArgs(), args);
        for (Operation &bodyOp : op.getBody()->getOperations()) {
          Operation *newOp = builder.clone(bodyOp, mapping);
          if (candidateHints.contains(&bodyOp)) {
            auto strideHint = candidateHints[&bodyOp];
            candidateHints.erase(&bodyOp);
            candidateHints.insert(std::make_pair(newOp, strideHint));

            auto it =
                std::find(candidateOps.begin(), candidateOps.end(), &bodyOp);
            assert(it != candidateOps.end());

            candidateOps.erase(it);
            candidateOps.push_back(newOp);
          }
        }
      });
  newOp->setAttrs(op->getAttrs());
  // Convert the book-keeping data structure to use the correct key and value.
  // Key is converted from init arg index to newly created block arg, and
  // Value's PtrState fields are converted from init arg to newly created block
  // arg
  int cnt = op.getRegionIterArgs().size();
  LLVM_DEBUG(llvm::dbgs() << "rewriteForOp RegionIterArgs init size: " << cnt
                          << "\n");

  for (auto [i, state] : initArgIndexState) {
    if (state.scalar) {
      state.scalar = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }
    for (auto it = state.offsets.begin(); it != state.offsets.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    for (auto it = state.strides.begin(); it != state.strides.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "rewriteForOp RegionIterArgs loop size: " << cnt << "\n");
    auto key = newOp.getRegionIterArgs()[i];
    knownPtrs.insert(std::make_pair(key, state));
  }

  // mask info process
  for (auto [i, state] : initmaskIndexState) {
    if (state.start) {
      state.start = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }
    if (state.end) {
      state.end = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    auto key = newOp.getRegionIterArgs()[i];
    knownMasks.insert(std::make_pair(key, state));
  }

  assert(static_cast<size_t>(cnt) == newOp.getRegionIterArgs().size() &&
         "expect to remap all new block args");
  LLVM_DEBUG(llvm::dbgs() << "rewriteForOp getNumResults size: "
                          << op.getNumResults() << "\n");
  // Replace only the results that correspond to the original scf.for
  auto resultsToReplaceWith = ResultRange(
      newOp.result_begin(), newOp.result_begin() + op.getNumResults());
  rewriter.replaceOp(op, resultsToReplaceWith);
  if (newOp.getNumRegionIterArgs()) {
    LLVM_DEBUG(llvm::dbgs() << "newOp getNumRegionIterArgs size: "
                            << newOp.getNumRegionIterArgs() << "\n");
    auto yieldOp = cast<scf::YieldOp>(newOp.getBody()->getTerminator());
    rewriteYieldOp(rewriter, yieldOp, knownPtrs, knownMasks);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "ptr analysis create new for\n";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
  return success();
}

void PtrAnalysis::foldAwayForOp(
    PatternRewriter &rewriter, scf::ForOp forOp,
    llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
  LLVM_DEBUG(llvm::dbgs() << "foldAwayForOp: \n");
  for (auto it : llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs(),
                           forOp.getResults(), forOp.getYieldedValues())) {
    // mlir's std canonicalize pass will handle this case
    bool forwarded =
        ((std::get<1>(it) == std::get<3>(it)) ||
         (std::get<1>(it).use_empty() &&
          (std::get<0>(it) == std::get<3>(it) || std::get<2>(it).use_empty())));
    if (forwarded || !std::get<1>(it).hasOneUse())
      continue;

    // Note: try to support more patterns
    for (auto op : std::get<1>(it).getUsers()) {
      size_t totalUsers = 0;
      for (OpResult result : op->getResults()) {
        auto userRange = result.getUsers();
        totalUsers += std::distance(userRange.begin(), userRange.end());
      }

      if (totalUsers == 1 && op->getResult(0) == std::get<3>(it) &&
          std::get<2>(it).use_empty()) {
        op->getResult(0).replaceAllUsesWith(std::get<1>(it));
      }
    }
  }
}

bool checkElemType(Type t) {
  if (!isa<TensorType>(t))
    return false;

  auto tensorType = dyn_cast<TensorType>(t);
  if (!tensorType.getElementType().isIntOrFloat() ||
      tensorType.getElementType().getIntOrFloatBitWidth() > 32)
    return false;

  // Note: add other limits if needed
  return true;
}

bool checkNoScalar(Type t) {
  if (!isa<TensorType>(t))
    return false;

  auto tensorType = dyn_cast<TensorType>(t);
  auto shape = tensorType.getShape();
  if (std::all_of(shape.begin(), shape.end(), [](int i) { return i == 1; })) {
    return false;
  }

  // Note: add other limits if needed
  return true;
}

bool checkPtrType(Type t) {
  if (!isa<TensorType>(t))
    return false;

  if (isa<triton::PointerType>(t) &&
      isa<RankedTensorType>(
          dyn_cast<triton::PointerType>(t).getPointeeType())) {
    return false;
  }

  // Note: add other limits if needed
  return true;
}

// If load/store's ptr operand (actually the offsets) is from other load op,
// then bypass this load/store op. Since the offsets are dynamic, there is no
// way to check whether offsets are continuous
bool isPtrFromLoad(Value v, llvm::DenseMap<Value, bool> &valueFromLoads) {
  if (valueFromLoads.contains(v)) {
    return valueFromLoads.at(v);
  }

  if (isa<IntegerType>(v.getType())) {
    valueFromLoads.insert(std::make_pair(v, false));
    return false;
  }

  bool bypass = false;
  // need more check if it is the block argument of ForOp
  if (!v.getDefiningOp()) {
    auto blockArgOp = dyn_cast_or_null<mlir::BlockArgument>(v);
    if (blockArgOp && isa<scf::ForOp>(blockArgOp.getOwner()->getParentOp())) {
      auto forOp = dyn_cast<scf::ForOp>(blockArgOp.getOwner()->getParentOp());
      auto idx = blockArgOp.getArgNumber() - forOp.getNumInductionVars();

      auto initValue = forOp.getInitArgs()[idx];
      bypass = initValue.getDefiningOp()
                   ? isPtrFromLoad(initValue, valueFromLoads)
                   : true;

      /// yieldOp maybe use the block argument which produce infinite loop.
      valueFromLoads.insert(std::make_pair(v, bypass));

      /// if already bypass, no need to analysis yield.
      if (!bypass) {
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        auto yieldValue = yieldOp.getOperands()[idx];
        bool yieldBypass = isPtrFromLoad(yieldValue, valueFromLoads);
        bypass = bypass || yieldBypass;

        valueFromLoads[v] = bypass;
      }
    } else {
      valueFromLoads.insert(std::make_pair(v, bypass));
    }

    return bypass;
  }
  TypeSwitch<Operation *>(v.getDefiningOp())
      .Case<triton::LoadOp>([&](auto op) { bypass = true; })
      .Case<arith::ConstantOp, arith::ConstantIndexOp, triton::MakeRangeOp>(
          [&](auto op) { bypass = false; })
      .Case<triton::AddPtrOp>([&](triton::AddPtrOp op) {
        bypass = isPtrFromLoad(op.getPtr(), valueFromLoads) ||
                 isPtrFromLoad(op.getOffset(), valueFromLoads);
      })
      .Case<triton::SplatOp, triton::ExpandDimsOp, triton::BitcastOp,
            triton::BroadcastOp, triton::gpu::ConvertLayoutOp, triton::TransOp>(
          [&](auto op) { bypass = isPtrFromLoad(op.getSrc(), valueFromLoads); })
      .Case<arith::AddIOp, arith::MulIOp, arith::CmpIOp, arith::OrIOp>(
          [&](auto op) {
            bypass = isPtrFromLoad(op.getLhs(), valueFromLoads) ||
                     isPtrFromLoad(op.getRhs(), valueFromLoads);
          })
      .Case<arith::ExtSIOp, arith::ExtUIOp>(
          [&](auto op) { bypass = isPtrFromLoad(op.getIn(), valueFromLoads); })
      .Case<arith::SelectOp, arith::DivSIOp, arith::SubIOp, arith::RemSIOp,
            arith::RemUIOp, arith::FPToSIOp, arith::FPToUIOp, triton::DotOp,
            triton::ReduceOp, triton::ReshapeOp, triton::gpu::ConvertLayoutOp>(
          [&](auto op) {
            // Now bypass SelectOP, SubIOp, DivSIOp, RemSIOp and RemUIOp.
            // Optimization will be considered in subsequent steps
            LLVM_DEBUG(llvm::dbgs()
                       << "bypass from :" << op->getName().getStringRef().str()
                       << "\n");
            bypass = true;
          })
      .Case<scf::ForOp, scf::IfOp, scf::WhileOp>([&](auto op) {
        // Now bypass ForOp, WhileOp, IfOp op
        LLVM_DEBUG(llvm::dbgs() << "bypass from :"
                                << op->getName().getStringRef().str() << "\n");
        bypass = true;
      })
      .Default([&](auto op) {
        std::string info = std::string("add logic to support op ") +
                           op->getName().getStringRef().str();
        llvm_unreachable(info.c_str());
      });
  valueFromLoads.insert(std::make_pair(v, bypass));
  return bypass;
}

bool isPtrCandidate(Value v, const gcu::AxisInfoEx *axisInfoEx,
                    SmallVector<int32_t> &strideHint) {
  if (!axisInfoEx) {
    LLVM_DEBUG(llvm::dbgs() << "bypass load/store op not get axisInfoEx: \n");
    return false;
  }

  llvm::DenseMap<Value, bool> valueFromLoads;
  if (isPtrFromLoad(v, valueFromLoads)) {
    LLVM_DEBUG(llvm::dbgs() << "bypass load/store op is isPtrFromLoad: \n");
    return false;
  }

  if (!isa<RankedTensorType>(v.getType())) {
    LLVM_DEBUG(llvm::dbgs() << "bypass load/store op is type error: \n");
    return false;
  }

  assert(isa<RankedTensorType>(v.getType()));
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  auto tshape = tensorType.getShape();
  assert(tshape.size() == static_cast<unsigned>(axisInfoEx->getRank()));

  // bool isContiguous = false;
  auto rank = axisInfoEx->getRank();
  if (rank >= 5)
    return false;

  bool bcheckNotZero = false;
  for (int idx = 0; idx < rank; ++idx) {
    if (axisInfoEx->getContinualInterval(idx) != 0) {
      bcheckNotZero = true;
      break;
    }
  }
  // assert((rank == 1 || bcheckNotZero) && "not support all stride is zero");

  for (int i = 0; i < rank - 1; ++i) {
    // skip ptr whose shape including dim size = 1
    if (tshape[i] == 1) {
      LLVM_DEBUG(llvm::dbgs() << "bypass load/store op shape including 1: \n");
      return false;
    }
    if (axisInfoEx->getContinualInterval(i) <= 0)
      continue;
    for (int j = i + 1; j < rank; ++j) {
      if (axisInfoEx->getContinualInterval(j) <= 0)
        continue;
      if ((axisInfoEx->getContinualInterval(i) %
               axisInfoEx->getContinualInterval(j) !=
           0) &&
          (axisInfoEx->getContinualInterval(j) %
               axisInfoEx->getContinualInterval(i) !=
           0)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "bypass load/store op static stride is not ratio: \n");
        return false;
      }
    }
  }

  // check mismatch shape&stride, only check stride=1
  // auto minShapeContinue = tshape[0];
  // for (int i = 1; i < rank; ++i)
  //   if (tshape[i] < minShapeContinue)
  //     minShapeContinue = tshape[i];
  // for (int i = 0; i < rank; ++i) {
  //   if (axisInfoEx->isContinualLowDim(tshape, i)) {
  //     for (int j = 0; j < rank; ++j) {
  //       if (j == i || axisInfoEx->getContinualInterval(j) <= 0)
  //         continue;

  //       if ((axisInfoEx->getContinualInterval(j) == 1) &&
  //           (axisInfoEx->getContinualSize(j) == 1))
  //        continue;

  //       if (axisInfoEx->getContinualInterval(j) < minShapeContinue) {
  //         //std::cout << "check mismatch shape&stride error \n";
  //         LLVM_DEBUG(llvm::dbgs()
  //             << "bypass load op static stride is mismatch shape: \n");
  //         return false;
  //       }
  //     }
  //   }
  // }
  for (int i = 0; i < rank; ++i) {
    int32_t stride = axisInfoEx->getContinualInterval(i);
    strideHint.push_back(stride);
    // if (stride < 0)
    //     bDynamicStride = true;
  }

  if (std::count(strideHint.begin(), strideHint.end(), 1) > 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "bypass load/store op including two dim with stride 1: \n");
    return false;
  }

  for (int i = 0; i < rank; ++i) {
    if (!axisInfoEx->isContinualDim(tshape, i)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "bypass load/store op is not continue shape: \n");
      return false;
    }
  }

  // isContiguous = true;
  // if (bDynamicStride) {
  //   isContiguous = true;
  // } else {
  //   for (int i = 0; i < rank; ++i) {
  //     if (axisInfoEx->isContinualLowDim(tshape, i) ||
  //       axisInfoEx->getContinualInterval(i) == 0) {
  //         isContiguous = true;
  //         break;
  //     }
  //   }
  // }

  LLVM_DEBUG(llvm::dbgs() << "ptr contiguous true:\n");
  for (int k = 0; k < rank; ++k) {
    LLVM_DEBUG(llvm::dbgs() << "dim: " << k << "\n"
                            << "axisInfoEx.divisibility: "
                            << axisInfoEx->getDivisibility(k) << "\n"
                            << "axisInfoEx.continualsize: "
                            << axisInfoEx->getContinualSize(k) << "\n"
                            << "axisInfoEx.continualinterval: "
                            << axisInfoEx->getContinualInterval(k) << "\n"
                            << "tensor shape: " << tshape[k] << "\n"
                            << "stride hint: " << strideHint[k] << "\n");
  }

  return true;
}

bool isMaskCandidate(Value v, llvm::DenseMap<Value, bool> &valueToCandiates) {
  if (valueToCandiates.contains(v)) {
    return valueToCandiates.at(v);
  }

  if (isa<IntegerType>(v.getType())) {
    valueToCandiates.insert(std::make_pair(v, true));
    return true;
  }

  bool candidate = true;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto blockArgOp = dyn_cast_or_null<mlir::BlockArgument>(v);
    if (blockArgOp && isa<scf::ForOp>(blockArgOp.getOwner()->getParentOp())) {
      auto forOp = dyn_cast<scf::ForOp>(blockArgOp.getOwner()->getParentOp());
      auto idx = blockArgOp.getArgNumber() - forOp.getNumInductionVars();

      auto initValue = forOp.getInitArgs()[idx];
      candidate = initValue.getDefiningOp()
                      ? isMaskCandidate(initValue, valueToCandiates)
                      : true;

      /// yieldOp maybe use the block argument which produce infinite loop.
      valueToCandiates.insert(std::make_pair(v, candidate));
      /// if already not be candidate, no need to analysis yield.
      if (candidate) {
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        auto yieldValue = yieldOp.getOperands()[idx];
        bool yieldCandidate = isMaskCandidate(yieldValue, valueToCandiates);
        candidate = candidate && yieldCandidate;

        valueToCandiates[v] = candidate;
      }
    } else {
      valueToCandiates.insert(std::make_pair(v, candidate));
    }
    return candidate;
  }

  candidate = false;
  TypeSwitch<Operation *>(v.getDefiningOp())
      .Case<triton::LoadOp>([&](auto op) { candidate = false; })
      .Case<arith::ConstantOp, arith::ConstantIndexOp, triton::MakeRangeOp>(
          [&](auto op) { candidate = true; })
      .Case<triton::ExpandDimsOp, triton::BitcastOp, triton::BroadcastOp>(
          [&](auto op) {
            candidate = isMaskCandidate(op.getSrc(), valueToCandiates);
          })
      .Case<arith::AddIOp, arith::AndIOp>([&](auto op) {
        candidate = isMaskCandidate(op.getLhs(), valueToCandiates) &&
                    isMaskCandidate(op.getRhs(), valueToCandiates);
      })
      .Case<arith::ExtSIOp, arith::ExtUIOp>([&](auto op) {
        candidate = isMaskCandidate(op.getIn(), valueToCandiates);
      })
      .Case<arith::SelectOp, arith::DivSIOp, arith::SubIOp, arith::RemSIOp,
            arith::MulIOp, arith::RemUIOp, arith::FPToSIOp, arith::FPToUIOp,
            triton::ReduceOp, triton::DotOp, triton::ReshapeOp>([&](auto op) {
        // bypass DivSIOp, which is completely discontiguous index operation,
        // and cannot be converted to dte
        LLVM_DEBUG(llvm::dbgs() << "bypass from :"
                                << op->getName().getStringRef().str() << "\n");
        candidate = false;
      })
      .Case<scf::ForOp, scf::IfOp, scf::WhileOp>([&](auto op) {
        // bypass ForOp, IfOp, WhileOp,
        // which is maybe discontiguous index operation.
        LLVM_DEBUG(llvm::dbgs() << "bypass from :"
                                << op->getName().getStringRef().str() << "\n");
        candidate = false;
      })
      .Case<triton::SplatOp>([&](auto op) {
        assert(isa<IntegerType>(op.getSrc().getType()) &&
               "splat source must be an integer scalar for load/store masks");
        candidate = isMaskCandidate(op.getSrc(), valueToCandiates);
      })
      .Case<arith::CmpIOp>([&](auto op) {
        if (op.getPredicate() == arith::CmpIPredicate::slt ||
            op.getPredicate() == arith::CmpIPredicate::ult ||
            op.getPredicate() == arith::CmpIPredicate::sge ||
            op.getPredicate() == arith::CmpIPredicate::uge) {
          if (auto tensorType = dyn_cast<TensorType>(op.getLhs().getType())) {
            auto shape = tensorType.getShape();
            if (shape.size() >= 2 &&
                std::all_of(shape.begin(), shape.end(),
                            [](int i) { return i != 1; })) {
              candidate = false;
            } else {
              candidate = isMaskCandidate(op.getLhs(), valueToCandiates) &&
                          isMaskCandidate(op.getRhs(), valueToCandiates);
            }
          } else {
            candidate = isMaskCandidate(op.getLhs(), valueToCandiates) &&
                        isMaskCandidate(op.getRhs(), valueToCandiates);
          }
        } else {
          candidate = false;
        }
      })
      .Default([&](auto op) {
        std::string info = std::string("add logic to support op ") +
                           op->getName().getStringRef().str();
        llvm_unreachable(info.c_str());
      });

  valueToCandiates.insert(std::make_pair(v, candidate));
  return candidate;
}

void PtrAnalysis::collectCandidateLoadStoreOps(
    ModuleOp &moduleOp, llvm::SmallVector<Operation *, 8> &candidates,
    llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints) {
  gcu::ModuleAxisInfoExAnalysis axisInfoExAnalysis(moduleOp);

  llvm::SmallVector<Operation *, 8> loadstoreOps;
  moduleOp.walk([&](triton::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Note: try to support nested for loop if needed
      TypeSwitch<Operation *>(op).Case<triton::LoadOp, triton::StoreOp>(
          [&](auto matchOp) {
            loadstoreOps.push_back(matchOp.getOperation());
          });
      // Note: try to support other cases like func call if needed
    });
  });

  for (auto op : loadstoreOps) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto ptr = loadOp.getPtr();
      auto axisInfoEx = axisInfoExAnalysis.getAxisInfoEx(ptr);

      if (!checkNoScalar(loadOp.getType())) {
        LLVM_DEBUG(llvm::dbgs() << "bypass load op due to scalar data type: "
                                << loadOp << "\n");
        continue;
      }

      if (!checkElemType(loadOp.getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "bypass load op due to noncandidate element type: "
                   << loadOp << "\n");
        continue;
      }

      if (!checkPtrType(ptr.getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "bypass load op due to noncandidate ptr type: " << loadOp
                   << "\n");
        continue;
      }
      SmallVector<int32_t> strideHint;
      if (!isPtrCandidate(ptr, axisInfoEx, strideHint)) {
        LLVM_DEBUG(llvm::dbgs() << "bypass load op due to noncandidate ptr: "
                                << loadOp << "\n");
        continue;
      }
      llvm::DenseMap<Value, bool> valueToCandiates;
      if (loadOp.getMask() &&
          !isMaskCandidate(loadOp.getMask(), valueToCandiates)) {
        LLVM_DEBUG(llvm::dbgs() << "bypass load op due to noncandidate mask: "
                                << loadOp << "\n");
        continue;
      }

      // Great to arrive here
      LLVM_DEBUG(llvm::dbgs() << "candidate load op " << loadOp << "\n");
      candidates.push_back(op);
      candidateHints.insert(std::make_pair(op, strideHint));
    } else {
      assert(isa<triton::StoreOp>(op));

      auto storeOp = dyn_cast<triton::StoreOp>(op);
      auto ptr = storeOp.getPtr();
      auto axisInfoEx = axisInfoExAnalysis.getAxisInfoEx(ptr);

      if (!checkNoScalar(storeOp.getValue().getType())) {
        LLVM_DEBUG(llvm::dbgs() << "bypass store op due to scalar data type: "
                                << storeOp << "\n");
        continue;
      }

      if (!checkElemType(storeOp.getValue().getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "bypass store op due to noncandidate element type: "
                   << storeOp << "\n");
        continue;
      }

      if (!checkPtrType(ptr.getType())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "bypass store op due to noncandidate ptr type: "
                   << storeOp << "\n");
        continue;
      }

      SmallVector<int32_t> strideHint;
      if (!isPtrCandidate(ptr, axisInfoEx, strideHint)) {
        LLVM_DEBUG(llvm::dbgs() << "bypass store op due to noncandidate ptr: "
                                << storeOp << "\n");
        continue;
      }
      llvm::DenseMap<Value, bool> valueToCandiates;
      if (storeOp.getMask() &&
          !isMaskCandidate(storeOp.getMask(), valueToCandiates)) {
        LLVM_DEBUG(llvm::dbgs() << "bypass store op due to noncandidate mask: "
                                << storeOp << "\n");
        continue;
      }

      // Great to arrive here
      LLVM_DEBUG(llvm::dbgs() << "candidate store op " << storeOp << "\n");
      candidates.push_back(op);
      candidateHints.insert(std::make_pair(op, strideHint));
    }
  }
}

} // namespace gcu
} // namespace triton
} // namespace mlir
