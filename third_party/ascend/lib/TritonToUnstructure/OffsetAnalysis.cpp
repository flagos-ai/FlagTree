/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToUnstructure/OffsetAnalysis.h"
#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "Utils/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-offset-analysis"

namespace mlir {
namespace triton {

PtrOffsetInfo::PtrOffsetInfo() : ptr(nullptr), offset(nullptr) {}

PtrOffsetInfo::PtrOffsetInfo(const PtrOffsetInfo &other) { *this = other; }

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr) : ptr(ptr) { setZeroOffset(); }

PtrOffsetInfo::PtrOffsetInfo(ArrayRef<AxisInfo> structured)
    : ptr(nullptr), offset(nullptr) {
  setStructured(structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, AxisInfo structured) : ptr(ptr) {
  setZeroOffset();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, ArrayRef<AxisInfo> structured)
    : ptr(ptr) {
  setStructured(structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, const Value &offset,
                             AxisInfo structured)
    : ptr(ptr), offset(offset) {
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, const Value &offset,
                             ArrayRef<AxisInfo> structured)
    : ptr(ptr), offset(offset) {
  setStructured(structured);
}

PtrOffsetInfo &PtrOffsetInfo::operator=(const PtrOffsetInfo &other) {
  setPtr(other.getPtr());
  setOffset(other.getOffset());
  setOffsets(other.getOffsets());
  setStructured(other.getStructured());
  setScalarLike(other.isScalarLike());
  setPointerDescriptorOwned(other.isPointerDescriptorOwned());
  return *this;
}

Value PtrOffsetInfo::getPtr() const { return this->ptr; }
Value PtrOffsetInfo::getOffset() const { return this->offset; }
SmallVector<Value> PtrOffsetInfo::getOffsets() const {
  return this->tptOffsets;
}
SmallVector<Value> &PtrOffsetInfo::getOffsetsRef() { return this->tptOffsets; }

bool PtrOffsetInfo::isScalarLike() const { return this->scalarLike; }
bool PtrOffsetInfo::isPointerDescriptorOwned() const {
  return this->pointerDescriptorOwned;
}

SmallVector<PtrOffsetInfo::AxisInfo> &PtrOffsetInfo::getStructuredRef() {
  return this->structured;
}
const SmallVector<PtrOffsetInfo::AxisInfo> &
PtrOffsetInfo::getStructured() const {
  return this->structured;
}

int PtrOffsetInfo::getRank() const { return structured.size(); }

void PtrOffsetInfo::setPtr(const Value &ptr) { this->ptr = ptr; }
void PtrOffsetInfo::setOffset(const Value &offset) { this->offset = offset; }

void PtrOffsetInfo::setOffsets(ValueRange offsets) {
  tptOffsets.clear();
  for (auto offset : offsets)
    tptOffsets.push_back(offset);
}

void PtrOffsetInfo::setStructured() {
  assert(ptr && "ptr Should be to infer rank");
  this->structured.clear();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), AxisInfo::structured);
}

void PtrOffsetInfo::setStructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, AxisInfo::structured);
}

void PtrOffsetInfo::setStructured(int rank, AxisInfo info) {
  this->structured.clear();
  this->structured.resize(rank, info);
}

void PtrOffsetInfo::setUnstructured() {
  assert(ptr && "ptr Should be to infer rank");
  this->structured.clear();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), AxisInfo::unstructured);
}

void PtrOffsetInfo::setUnstructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, AxisInfo::unstructured);
}

void PtrOffsetInfo::setStructured(ArrayRef<AxisInfo> structured) {
  this->structured.resize(structured.size());
  for (size_t i = 0; i < structured.size(); i++)
    this->structured[i] = structured[i];
}

void PtrOffsetInfo::setStructured(const PtrOffsetInfo &other) {
  this->setStructured(other.getStructured());
}

void PtrOffsetInfo::setScalarLike(bool scalarLike) {
  this->scalarLike = scalarLike;
}

void PtrOffsetInfo::setPointerDescriptorOwned(bool pointerDescriptorOwned) {
  this->pointerDescriptorOwned = pointerDescriptorOwned;
}

bool PtrOffsetInfo::isStructured(int dim) const {
  return this->scalarLike || structured[dim] == AxisInfo::structured ||
         structured[dim] == AxisInfo::scalar;
}

bool PtrOffsetInfo::isStructured() const {
  return this->scalarLike || llvm::all_of(structured, [](auto dim) {
           return dim == AxisInfo::structured || dim == AxisInfo::scalar;
         });
}

bool PtrOffsetInfo::isUnstructured() const {
  return llvm::all_of(structured,
                      [](auto dim) { return dim == AxisInfo::unstructured; });
}

bool PtrOffsetInfo::isUnstructuredOrScalarlike() const {
  return llvm::all_of(structured, [](auto dim) {
    return dim == AxisInfo::unstructured || dim == AxisInfo::scalarlike ||
           dim == AxisInfo::scalar;
  });
}

void PtrOffsetInfo::setZeroOffset() {
  if (!ptr)
    return;
  Value offset;
  OpBuilder builder(ptr.getContext());
  builder.setInsertionPointToStart(ptr.getParentBlock());
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    offset = builder.create<arith::ConstantOp>(
        ptr.getLoc(), DenseElementsAttr::get(
                          RankedTensorType::get(tensorType.getShape(),
                                                builder.getIntegerType(64),
                                                tensorType.getEncoding()),
                          builder.getZeroAttr(builder.getIntegerType(64))));
  } else {
    offset = builder.create<arith::ConstantOp>(ptr.getLoc(),
                                               builder.getI64IntegerAttr(0));
  }
  setOffset(offset);
}

PtrOffsetInfo combineInfo(const PtrOffsetInfo &lhs, const PtrOffsetInfo &rhs) {
  PtrOffsetInfo info;
  assert(lhs.getRank() == rhs.getRank() && "Rank must be same to be combined");

  info.setScalarLike(lhs.isScalarLike() && rhs.isScalarLike());
  auto &structuredRef = info.getStructuredRef();
  auto lhsStructured = lhs.getStructured();
  auto rhsStructured = rhs.getStructured();
  structuredRef.resize(lhs.getRank());
  for (size_t i = 0; i < structuredRef.size(); i++)
    structuredRef[i] = std::min(lhsStructured[i], rhsStructured[i]);
  return info;
}

namespace {

bool isScalarPointer(Value value) {
  auto pointerType = dyn_cast<triton::PointerType>(value.getType());
  return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
}

bool isTensorPointer(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

// A scalar pointer selected or carried by structured control flow is already a
// complete runtime address. Recording the SSA value itself as the source avoids
// choosing one incoming source and losing the other branch or loop iteration.
// Later addptr users can still accumulate a separate offset from this address.
void recordOpaqueScalarPointer(
    Value pointer, llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  offsetMap[pointer] = PtrOffsetInfo();
  offsetMap[pointer].setPtr(pointer);
  offsetMap[pointer].setZeroOffset();
  offsetMap[pointer].setScalarLike(true);
}

// A tensor pointer without a proven common scalar base may choose a different
// source for every element. This includes opaque function arguments,
// per-lane selects, and loop phi values whose incoming provenance has not been
// merged. Preserve the pointer tensor itself as the complete opaque base and
// attach a zero displacement. Later addptr parsing can accumulate an additional
// offset without discarding the runtime lane-wise source choice.
void recordOpaqueTensorPointer(
    Value pointerTensor, llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto tensorType = cast<RankedTensorType>(pointerTensor.getType());
  offsetMap[pointerTensor] = PtrOffsetInfo();
  offsetMap[pointerTensor].setPtr(pointerTensor);
  offsetMap[pointerTensor].setZeroOffset();
  offsetMap[pointerTensor].setUnstructured(tensorType.getRank());
}

// PointerDescriptorBoundary stores loop-carried slot indices in the common
// init/region-argument/result/terminator coordinate space. Only those slots
// belong to the CFO descriptor protocol; unmarked tensor-pointer loops retain
// the established main-dev provenance analysis.
bool isPointerDescriptorBoundarySlot(Operation *loop, unsigned slot) {
  auto slots = dyn_cast_or_null<DenseI32ArrayAttr>(
      loop->getAttr(controlflow::kPointerDescriptorBoundaryAttr));
  if (!slots)
    return false;
  return llvm::is_contained(slots.asArrayRef(), static_cast<int32_t>(slot));
}

bool isScalarPointerOffsetBoundarySlot(Operation *loop, unsigned slot) {
  auto slots = dyn_cast_or_null<DenseI32ArrayAttr>(
      loop->getAttr(kScalarPointerOffsetBoundaryAttr));
  return slots &&
         llvm::is_contained(slots.asArrayRef(), static_cast<int32_t>(slot));
}

bool isScalarPointerOffsetBoundaryRegionArgument(LoopLikeOpInterface loopOp,
                                                 BlockArgument regionIterArg) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    if (regionIterArg.getOwner() != whileOp.getBeforeBody() &&
        regionIterArg.getOwner() != whileOp.getAfterBody())
      return false;
    return isScalarPointerOffsetBoundarySlot(loopOp.getOperation(),
                                             regionIterArg.getArgNumber());
  }

  if (auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation())) {
    if (regionIterArg.getOwner() != forOp.getBody() ||
        regionIterArg.getArgNumber() == 0)
      return false;
    return isScalarPointerOffsetBoundarySlot(loopOp.getOperation(),
                                             regionIterArg.getArgNumber() - 1);
  }
  return false;
}

bool isPointerDescriptorBoundaryRegionArgument(LoopLikeOpInterface loopOp,
                                               BlockArgument regionIterArg) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    if (regionIterArg.getOwner() != whileOp.getBeforeBody() &&
        regionIterArg.getOwner() != whileOp.getAfterBody())
      return false;
    return isPointerDescriptorBoundarySlot(whileOp,
                                           regionIterArg.getArgNumber());
  }

  for (auto [slot, argument] : llvm::enumerate(loopOp.getRegionIterArgs())) {
    if (argument == regionIterArg)
      return isPointerDescriptorBoundarySlot(loopOp.getOperation(), slot);
  }
  return false;
}

bool isPointerDescriptorBoundaryResult(LoopLikeOpInterface loopOp,
                                       Value result) {
  auto opResult = dyn_cast<OpResult>(result);
  return opResult && opResult.getOwner() == loopOp.getOperation() &&
         isPointerDescriptorBoundarySlot(loopOp.getOperation(),
                                         opResult.getResultNumber());
}

// Materialize the current value of an ordinary tensor offset carried beside a
// CFO pointer descriptor. T2L cannot recover layout from an scf.for block
// argument, but it can analyze the equivalent affine expression. Keep this
// deliberately narrow: the backedge must be an additive, constant-splat
// recurrence, and the carrier slot itself must not belong to the descriptor.
Value materializeAffineForOffsetCarrier(Value value, triton::AddPtrOp addPtr,
                                        RewriterBase &rewriter) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getArgNumber() == 0)
    return nullptr;
  auto forOp = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp || blockArg.getOwner() != forOp.getBody())
    return nullptr;

  unsigned slot = blockArg.getArgNumber() - 1;
  if (!forOp->hasAttr(controlflow::kPointerDescriptorBoundaryAttr) ||
      isPointerDescriptorBoundarySlot(forOp, slot) ||
      slot >= forOp.getInitArgs().size() ||
      slot >= forOp.getYieldedValues().size())
    return nullptr;

  auto carrierType = dyn_cast<RankedTensorType>(value.getType());
  auto elementType = carrierType
                         ? dyn_cast<IntegerType>(carrierType.getElementType())
                         : IntegerType();
  if (!carrierType || !elementType || elementType.getWidth() > 64 ||
      forOp.getInitArgs()[slot].getType() != carrierType)
    return nullptr;

  auto backedgeAdd =
      forOp.getYieldedValues()[slot].getDefiningOp<arith::AddIOp>();
  if (!backedgeAdd)
    return nullptr;
  Value carrierStep;
  if (backedgeAdd.getLhs() == value)
    carrierStep = backedgeAdd.getRhs();
  else if (backedgeAdd.getRhs() == value)
    carrierStep = backedgeAdd.getLhs();
  else
    return nullptr;

  auto constant = carrierStep.getDefiningOp<arith::ConstantOp>();
  auto elements = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                           : DenseElementsAttr();
  if (!elements || !elements.isSplat() ||
      !isa<IntegerType>(elements.getElementType()) ||
      carrierStep.getType() != carrierType)
    return nullptr;

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(addPtr);
  Value iterationDistance = rewriter.create<arith::SubIOp>(
      addPtr.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());
  Value iteration = rewriter.create<arith::DivUIOp>(
      addPtr.getLoc(), iterationDistance, forOp.getStep());
  Value typedIteration = rewriter.create<arith::IndexCastOp>(
      addPtr.getLoc(), elementType, iteration);
  Value splatIteration = rewriter.create<triton::SplatOp>(
      addPtr.getLoc(), carrierType, typedIteration);
  Value materializedStep =
      rewriter.create<arith::ConstantOp>(addPtr.getLoc(), elements);
  Value displacement = rewriter.create<arith::MulIOp>(
      addPtr.getLoc(), splatIteration, materializedStep);
  return rewriter.create<arith::AddIOp>(
      addPtr.getLoc(), forOp.getInitArgs()[slot], displacement);
}

// Recover structured axes only when the carrier's own provenance proves that
// every lane is either affine, uniform, or a singleton axis. Complete
// carriers without that proof remain opaque and keep the existing indirect
// access fallback.
bool getRecoverableCarrierAxes(const PtrOffsetInfo &carrierInfo, unsigned rank,
                               SmallVectorImpl<PtrOffsetInfo::AxisInfo> &axes) {
  if (carrierInfo.getRank() != static_cast<int>(rank) ||
      !carrierInfo.isStructured())
    return false;

  axes.clear();
  axes.reserve(rank);
  for (PtrOffsetInfo::AxisInfo axis : carrierInfo.getStructured()) {
    if (axis == PtrOffsetInfo::AxisInfo::unstructured)
      return false;
    axes.push_back(axis == PtrOffsetInfo::AxisInfo::scalarlike
                       ? PtrOffsetInfo::AxisInfo::structured
                       : axis);
  }
  return true;
}

} // namespace

void parse(Value operand, const Location &loc, RewriterBase &rewriter,
           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (offsetMap.contains(operand)) {
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "found\n" << operand << '\n';
    });
    return;
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "parse\n" << operand << '\n';
  });

  if (auto *defOp = operand.getDefiningOp()) {
    if (isa<arith::ArithDialect>(defOp->getDialect())) {
      parseArithOp(defOp, loc, rewriter, offsetMap);
    } else if (isa<triton::TritonDialect>(defOp->getDialect())) {
      parseTritonOp(defOp, loc, rewriter, offsetMap);
    } else {
      if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
        parseIf(ifOp, loc, rewriter, offsetMap, operand);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(defOp)) {
        parseYield(yieldOp, loc, rewriter, offsetMap);
      } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(defOp)) {
        parseLoopOp(loopOp, loc, rewriter, offsetMap, operand);
      } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(defOp)) {
        parseExtract(extractOp, loc, rewriter, offsetMap);
      } else if (auto insertOp = dyn_cast<tensor::InsertOp>(defOp)) {
        parseInsert(insertOp, loc, rewriter, offsetMap);
      } else if (auto extractSliceOp =
                     dyn_cast<tensor::ExtractSliceOp>(defOp)) {
        parseExtractSlice(extractSliceOp, loc, rewriter, offsetMap);
      } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(defOp)) {
        parseInsertSlice(insertSliceOp, loc, rewriter, offsetMap);
      } else if (isDistributedTypeCustomOp(defOp)) {
        auto opResult = dyn_cast<OpResult>(operand);
        assert(opResult && "Expected operand to be an OpResult");
        parseStructuredCustomOp(defOp, loc, rewriter, offsetMap,
                                opResult.getResultNumber());
      }
    }
  } else if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
    auto parentOp = blockArgument.getOwner()->getParentOp();
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Handling block argument\n" << *blockArgument.getOwner() << '\n';
    });
    if (isa<FunctionOpInterface>(parentOp)) {
      if (auto ptrType = dyn_cast<triton::PointerType>(operand.getType())) {
        offsetMap[operand] =
            PtrOffsetInfo(operand, PtrOffsetInfo::AxisInfo::scalar);
      } else if (isTensorPointer(operand)) {
        // CFO may rebuild a tensor pointer from one invariant opaque function
        // argument plus a separately carried displacement. The argument itself
        // is the complete per-lane base: it has no common scalar provenance and
        // must remain unstructured even when the added displacement is affine.
        recordOpaqueTensorPointer(operand, offsetMap);
      } else {
        offsetMap[operand] = PtrOffsetInfo();
        if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType()))
          offsetMap[operand].setUnstructured(tensorType.getRank());
      }
    } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      parseLoopRegionIterArg(loopOp, loc, rewriter, offsetMap, blockArgument);
    }
  } else {
    llvm_unreachable("Unreachable");
  }

  if (!offsetMap.contains(operand)) {
    offsetMap[operand] = PtrOffsetInfo();
    if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType()))
      offsetMap[operand].setUnstructured(tensorType.getRank());
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "finish parse\n" << operand << '\n';
    auto data = offsetMap.at(operand);
    for (auto s : data.getStructuredRef())
      os << static_cast<int>(s);
    os << "\n";
  });

  if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      tensorType && isa<triton::PointerType>(tensorType.getElementType())) {
    auto data = offsetMap.at(operand);
    assert(data.getPtr() && "pointer type should be parsed");
  }
}

void parseLoopRegionIterArg(LoopLikeOpInterface loopOp, const Location &loc,
                            RewriterBase &rewriter,
                            llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                            BlockArgument regionIterArg) {
  // This argument is a dynamic relative offset created by T2U.  Do not copy
  // the loop init provenance here: doing so makes every backedge restart from
  // the initial offset and drops the accumulated delta.  Record the argument
  // itself as the offset identity so every later addptr keeps the live carrier
  // (`offset(region_arg) = region_arg`) when adding its backedge delta.
  if (isScalarPointerOffsetBoundaryRegionArgument(loopOp, regionIterArg)) {
    offsetMap.erase(regionIterArg);
    offsetMap[regionIterArg] = PtrOffsetInfo();
    offsetMap[regionIterArg].setOffset(regionIterArg);
    offsetMap[regionIterArg].setScalarLike(true);
    return;
  }

  if (isScalarPointer(regionIterArg) &&
      isPointerDescriptorBoundaryRegionArgument(loopOp, regionIterArg)) {
    recordOpaqueScalarPointer(regionIterArg, offsetMap);
    return;
  }

  // CFO descriptor slots may merge different incoming bases. Without a fixed
  // point, preserve the runtime phi itself rather than choosing one edge.
  // Ordinary unmarked loops stay on the established init/condition analysis.
  if (isTensorPointer(regionIterArg) &&
      isPointerDescriptorBoundaryRegionArgument(loopOp, regionIterArg)) {
    recordOpaqueTensorPointer(regionIterArg, offsetMap);
    return;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation());
      whileOp && whileOp.getAfterBody() == regionIterArg.getOwner()) {
    auto argNum = regionIterArg.getArgNumber();
    auto conditionArg = whileOp.getConditionOp().getArgs()[argNum];
    parse(conditionArg, loc, rewriter, offsetMap);
    auto tmp = offsetMap[conditionArg];
    offsetMap[regionIterArg] = tmp;
    return;
  }
  OpOperand *initArgOperand = loopOp.getTiedLoopInit(regionIterArg);
  if (!initArgOperand)
    return;
  Value initArg = initArgOperand->get();
  parse(initArg, loc, rewriter, offsetMap);
  auto tmp = offsetMap[initArg];
  offsetMap[regionIterArg] = tmp;
}

void parseArithOp(Operation *arithOp, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<arith::ArithDialect>(arithOp->getDialect()));
  if (auto addIOp = dyn_cast<arith::AddIOp>(arithOp)) {
    parseAddI(addIOp, loc, rewriter, offsetMap);
  } else if (auto subIOp = dyn_cast<arith::SubIOp>(arithOp)) {
    parseSubI(subIOp, loc, rewriter, offsetMap);
  } else if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(arithOp)) {
    parseIndexCast(indexCastOp, loc, rewriter, offsetMap);
  } else if (auto constantFloatOp = dyn_cast<arith::ConstantFloatOp>(arithOp)) {
    parseConstantOp(constantFloatOp, loc, rewriter, offsetMap);
  } else if (auto constantIntOp = dyn_cast<arith::ConstantIntOp>(arithOp)) {
    parseConstantOp(constantIntOp, loc, rewriter, offsetMap);
  } else if (auto constantOp = dyn_cast<arith::ConstantOp>(arithOp)) {
    parseConstantOp(constantOp, loc, rewriter, offsetMap);
  } else if (auto extSIOp = dyn_cast<arith::ExtSIOp>(arithOp)) {
    parseExtSI(extSIOp, loc, rewriter, offsetMap);
  } else if (auto mulIOp = dyn_cast<arith::MulIOp>(arithOp)) {
    parseMulI(mulIOp, loc, rewriter, offsetMap);
  } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(arithOp)) {
    parseBinaryOp(remSIOp, loc, rewriter, offsetMap);
  } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(arithOp)) {
    parseBinaryOp(divSIOp, loc, rewriter, offsetMap);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(arithOp)) {
    parseSelect(selectOp, loc, rewriter, offsetMap);
  } else if (auto fPToSIOp = dyn_cast<arith::FPToSIOp>(arithOp)) {
    parseFPToSI(fPToSIOp, loc, rewriter, offsetMap);
  } else if (auto sIToFPOp = dyn_cast<arith::SIToFPOp>(arithOp)) {
    parseSIToFP(sIToFPOp, loc, rewriter, offsetMap);
  } else if (auto mulFOp = dyn_cast<arith::MulFOp>(arithOp)) {
    parseBinaryOp(mulFOp, loc, rewriter, offsetMap);
  } else if (auto divFOp = dyn_cast<arith::DivFOp>(arithOp)) {
    parseBinaryOp(divFOp, loc, rewriter, offsetMap);
  } else if (auto addFOp = dyn_cast<arith::AddFOp>(arithOp)) {
    parseBinaryOp(addFOp, loc, rewriter, offsetMap);
  } else if (auto subFOp = dyn_cast<arith::SubFOp>(arithOp)) {
    parseBinaryOp(subFOp, loc, rewriter, offsetMap);
  } else if (auto minNumFOp = dyn_cast<arith::MinNumFOp>(arithOp)) {
    parseBinaryOp(minNumFOp, loc, rewriter, offsetMap);
  } else if (auto maxNumFOp = dyn_cast<arith::MaxNumFOp>(arithOp)) {
    parseBinaryOp(maxNumFOp, loc, rewriter, offsetMap);
  } else if (auto maxSIOp = dyn_cast<arith::MaxSIOp>(arithOp)) {
    parseBinaryOp(maxSIOp, loc, rewriter, offsetMap);
  } else if (auto minSIOp = dyn_cast<arith::MinSIOp>(arithOp)) {
    parseBinaryOp(minSIOp, loc, rewriter, offsetMap);
  } else if (auto cmpIOp = dyn_cast<arith::CmpIOp>(arithOp)) {
    parseBinaryOp(cmpIOp, loc, rewriter, offsetMap);
  } else if (auto andIOp = dyn_cast<arith::AndIOp>(arithOp)) {
    parseBinaryOp(andIOp, loc, rewriter, offsetMap);
  } else if (auto orIOp = dyn_cast<arith::OrIOp>(arithOp)) {
    parseBinaryOp(orIOp, loc, rewriter, offsetMap);
  }
}

void parseTritonOp(Operation *tritonOp, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<triton::TritonDialect>(tritonOp->getDialect()));
  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(tritonOp)) {
    parseAddPtr(addPtrOp, loc, rewriter, offsetMap);
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(tritonOp)) {
    parseSplat(splatOp, loc, rewriter, offsetMap);
  } else if (auto getProgramIdOp = dyn_cast<triton::GetProgramIdOp>(tritonOp)) {
    parseConstantOp(getProgramIdOp, loc, rewriter, offsetMap);
  } else if (auto getNumProgramsOp =
                 dyn_cast<triton::GetNumProgramsOp>(tritonOp)) {
    parseConstantOp(getNumProgramsOp, loc, rewriter, offsetMap);
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(tritonOp)) {
    parseMakeRange(makeRangeOp, loc, rewriter, offsetMap);
  } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(tritonOp)) {
    parseBitcast(bitcastOp, loc, rewriter, offsetMap);
  } else if (auto loadOp = dyn_cast<triton::LoadOp>(tritonOp)) {
    parseLoad(loadOp, loc, rewriter, offsetMap);
  } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(tritonOp)) {
    parseBroadcast(broadcastOp, loc, rewriter, offsetMap);
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(tritonOp)) {
    parseExpandDims(expandDimsOp, loc, rewriter, offsetMap);
  } else if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(tritonOp)) {
    parseReshape(reshapeOp, loc, rewriter, offsetMap);
  } else if (auto clampFOp = dyn_cast<triton::ClampFOp>(tritonOp)) {
    parseClampF(clampFOp, loc, rewriter, offsetMap);
  }
  // FIXME:Z|wait triton version upgrade to 3.4
  // else if (auto makeTensorDescOp =
  //                dyn_cast<triton::MakeTensorDescOp>(tritonOp)) {
  //   parseMakeTensorDesc(makeTensorDescOp, loc, rewriter, offsetMap);
  // }
  else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(tritonOp)) {
    parseMakeTensorPtr(makeTensorPtrOp, loc, rewriter, offsetMap);
  } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(tritonOp)) {
    parseReduce(reduceOp, loc, rewriter, offsetMap);
  } else if (auto reduceReturnOp = dyn_cast<triton::ReduceReturnOp>(tritonOp)) {
    parseReduceReturn(reduceReturnOp, loc, rewriter, offsetMap);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(tritonOp)) {
    parseAdvance(advanceOp, loc, rewriter, offsetMap);
  } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(tritonOp)) {
    parseIntToPtr(intToPtrOp, loc, rewriter, offsetMap);
  }
}

void parseAddPtr(triton::AddPtrOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addPtr base_ptr
  Value ptr = op.getPtr();
  parse(ptr, op.getLoc(), rewriter, offsetMap);
  auto ptrInfo = offsetMap.find(ptr);
  if (ptrInfo == offsetMap.end()) {
    op.emitOpError("could not analyze the pointer base");
    return;
  }
  // Get addPtr offset
  Value offsetValue = op.getOffset();
  PtrOffsetInfo ptrOffsetInfo = ptrInfo->second;
  bool isRebuild = op->hasAttr(controlflow::kPointerDescriptorRebuildAttr);
  Attribute offsetForm =
      op->getAttr(controlflow::kPointerDescriptorOffsetFormAttr);
  auto structuredAxes = dyn_cast_or_null<DenseI32ArrayAttr>(
      op->getAttr(controlflow::kPointerDescriptorStructuredAxesAttr));
  auto resultType = dyn_cast<RankedTensorType>(op.getType());
  auto baseSplat = ptr.getDefiningOp<triton::SplatOp>();
  bool hasScalarPointerSplatBase =
      baseSplat && isa<triton::PointerType>(baseSplat.getSrc().getType());
  SmallVector<PtrOffsetInfo::AxisInfo> descriptorAxes;
  if (structuredAxes) {
    if (!isRebuild || !resultType ||
        structuredAxes.asArrayRef().size() !=
            static_cast<size_t>(resultType.getRank())) {
      op.emitOpError("invalid pointer descriptor structured-axis metadata");
      return;
    }
    for (int32_t axis : structuredAxes.asArrayRef()) {
      if (axis == 1)
        descriptorAxes.push_back(PtrOffsetInfo::AxisInfo::structured);
      else if (axis == 0)
        descriptorAxes.push_back(PtrOffsetInfo::AxisInfo::unstructured);
      else {
        op.emitOpError(
            "pointer descriptor structured axes must contain only 0 or 1");
        return;
      }
    }
  }
  bool isStridedRankOne = false;
  if (offsetForm) {
    auto form = dyn_cast<StringAttr>(offsetForm);
    if (!isRebuild || !form ||
        form.getValue() != controlflow::kStrided1DOffsetForm) {
      op.emitOpError("invalid pointer descriptor offset form");
      return;
    }
    auto offsetType = dyn_cast<RankedTensorType>(offsetValue.getType());
    auto offsetElementType =
        offsetType ? dyn_cast<IntegerType>(offsetType.getElementType())
                   : IntegerType();
    auto baseSplat = ptr.getDefiningOp<triton::SplatOp>();
    if (!resultType || resultType.getRank() != 1 || !offsetType ||
        !isa<triton::PointerType>(resultType.getElementType()) ||
        !offsetElementType || offsetElementType.getWidth() > 64 ||
        resultType.getShape() != offsetType.getShape() ||
        resultType.getEncoding() != offsetType.getEncoding() || !baseSplat ||
        !isa<triton::PointerType>(baseSplat.getSrc().getType())) {
      op.emitOpError(
          "expected strided_1d on a rank-1 tensor pointer rebuilt from a "
          "scalar pointer base and a compatible ranked integer offset");
      return;
    }
    if (descriptorAxes.size() != 1 ||
        descriptorAxes.front() != PtrOffsetInfo::AxisInfo::structured) {
      op.emitOpError(
          "strided_1d requires PointerDescriptorStructuredAxes = [1]");
      return;
    }
    isStridedRankOne = true;
  }
  bool isCompleteOffsetCarrier = isRebuild && !isStridedRankOne;
  bool isDescriptorOwned =
      isRebuild || ptrOffsetInfo.isPointerDescriptorOwned();
  bool descriptorIsOpaque =
      !descriptorAxes.empty() &&
      llvm::all_of(descriptorAxes, [](PtrOffsetInfo::AxisInfo axis) {
        return axis == PtrOffsetInfo::AxisInfo::unstructured;
      });

  if (isCompleteOffsetCarrier) {
    // The carrier is complete relative to the descriptor base, but parsing
    // that base may expose an additional displacement from its scalar source.
    // Preserve both parts when reconstructing the complete pointer offset.
    auto offsetType = dyn_cast<RankedTensorType>(offsetValue.getType());
    auto offsetElementType =
        offsetType ? dyn_cast<IntegerType>(offsetType.getElementType())
                   : IntegerType();
    if (!resultType || !offsetType || !offsetElementType ||
        resultType.getShape() != offsetType.getShape() ||
        resultType.getEncoding() != offsetType.getEncoding()) {
      op.emitOpError("expected a shape- and encoding-compatible ranked integer "
                     "complete-offset carrier");
      return;
    }
    if (offsetElementType.getWidth() > 64) {
      op.emitOpError("complete-offset carrier wider than i64 is unsupported");
      return;
    }

    SmallVector<PtrOffsetInfo::AxisInfo> recoveredAxes;
    Value materializedOffset;
    if (descriptorIsOpaque && hasScalarPointerSplatBase) {
      parse(offsetValue, op.getLoc(), rewriter, offsetMap);
      auto carrierInfo = offsetMap.find(offsetValue);
      if (carrierInfo != offsetMap.end() &&
          getRecoverableCarrierAxes(carrierInfo->second, resultType.getRank(),
                                    recoveredAxes)) {
        materializedOffset =
            materializeAffineForOffsetCarrier(offsetValue, op, rewriter);
        if (!materializedOffset)
          recoveredAxes.clear();
      }
    }

    if (materializedOffset) {
      offsetValue = materializedOffset;
      op->setOperand(1, offsetValue);
      SmallVector<int32_t> structuredAxes(recoveredAxes.size(), 1);
      op->setAttr(controlflow::kPointerDescriptorStructuredAxesAttr,
                  rewriter.getDenseI32ArrayAttr(structuredAxes));
    }

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (offsetElementType.getWidth() != 64) {
      auto carrierType =
          RankedTensorType::get(offsetType.getShape(), rewriter.getI64Type(),
                                offsetType.getEncoding());
      offsetValue = rewriter.create<arith::ExtSIOp>(op.getLoc(), carrierType,
                                                    offsetValue);
    }
    Value baseDisplacement = ptrOffsetInfo.getOffset();
    if (!baseDisplacement ||
        baseDisplacement.getType() != offsetValue.getType()) {
      op.emitOpError(
          "expected pointer-base displacement compatible with complete "
          "carrier");
      return;
    }
    Value completeOffset = rewriter.create<arith::AddIOp>(
        op.getLoc(), baseDisplacement, offsetValue);
    ptrOffsetInfo.setOffset(completeOffset);
    // setUnstructured() updates only the per-axis classification; it does not
    // clear the independent scalar-like property inherited from a splatted
    // base. A complete offset carrier is intentionally opaque here, so there
    // is no proof that all lanes address the same element. Keep this state
    // conservative and force the lane-wise memory-access path.
    ptrOffsetInfo.setScalarLike(false);
    if (!recoveredAxes.empty())
      ptrOffsetInfo.setStructured(recoveredAxes);
    else if (!descriptorAxes.empty())
      ptrOffsetInfo.setStructured(descriptorAxes);
    else
      ptrOffsetInfo.setUnstructured(resultType.getRank());
    ptrOffsetInfo.setPointerDescriptorOwned(true);
    offsetMap[op.getResult()] = ptrOffsetInfo;
    return;
  }

  parse(offsetValue, op.getLoc(), rewriter, offsetMap);
  // Modify IR

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto offsetType = dyn_cast<RankedTensorType>(offsetValue.getType())) {
    auto offsetElementType = cast<IntegerType>(offsetType.getElementType());
    if (offsetElementType.getWidth() != 64) {
      auto newOffsetType = RankedTensorType::get(offsetType.getShape(),
                                                 rewriter.getIntegerType(64),
                                                 offsetType.getEncoding());
      offsetValue = rewriter.create<arith::ExtSIOp>(op.getLoc(), newOffsetType,
                                                    offsetValue);
    }
  } else {
    auto offsetIntType = cast<IntegerType>(offsetValue.getType());
    if (offsetIntType.getWidth() != 64) {
      offsetValue = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getIntegerType(64), offsetValue);
    }
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] Adding offset\n";
    os << ptrOffsetInfo.getOffset() << '\n' << offsetValue << '\n';
  });
  Value offset = rewriter.create<arith::AddIOp>(
      op.getLoc(), ptrOffsetInfo.getOffset(), offsetValue);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr] offset is\n" << offset << '\n';
  });
  // Set addPtr offset map
  auto dst = op.getResult();
  PtrOffsetInfo offsetOffsetInfo = offsetMap.at(op.getOffset());
  auto dstOffsetInfo = combineInfo(ptrOffsetInfo, offsetOffsetInfo);
  dstOffsetInfo.setPtr(ptrOffsetInfo.getPtr());
  dstOffsetInfo.setOffset(offset);
  dstOffsetInfo.setPointerDescriptorOwned(isDescriptorOwned);
  // CFO's strided_1d descriptor is an explicit proof that a scalar pointer
  // base plus this rank-1 lane offset remains a contiguous SIMD access. The
  // generic offset join above cannot retain that proof when the lane offset
  // was built through several arithmetic operations, so restore the marker's
  // classification after the join. Complete opaque carriers intentionally do
  // not enter this branch and keep their conservative unstructured state.
  if (isStridedRankOne) {
    dstOffsetInfo.setStructured(descriptorAxes);
    // The descriptor describes a lane-varying, structured offset.  It is
    // contiguous, but it is not a scalar-like pointer: scalarLike would
    // route the load/store converter through splatAndLoadScenario(), which
    // loads lane zero once and fills every lane with that value.
    dstOffsetInfo.setScalarLike(false);
  }
  offsetMap[dst] = dstOffsetInfo;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    auto &ptrStructured = ptrOffsetInfo.getStructuredRef();
    auto &offsetStructured = offsetOffsetInfo.getStructuredRef();
    os << "[parseAddPtr] ptrStructured: ";
    for (size_t i = 0; i < ptrStructured.size(); i++)
      os << static_cast<int>(ptrStructured[i]);
    os << "\n";
    os << "[parseAddPtr] offsetStructured: ";
    for (size_t i = 0; i < offsetStructured.size(); i++)
      os << static_cast<int>(offsetStructured[i]);
    os << "\n";
  });
}

void parseSplat(triton::SplatOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get splat src
  auto src = op.getSrc();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto dst = op.getResult();
  auto dstType = cast<RankedTensorType>(dst.getType());
  PtrOffsetInfo dstOffsetInfo(srcOffsetInfo.getPtr());
  // Modify IR
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseSplat] dst is\n" << dst << '\n';
  });
  if (isa<triton::PointerType>(dstType.getElementType())) {
    RewriterBase::InsertionGuard guard(rewriter);
    auto dstShape = dstType.getShape();
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::SplatOp>(
        loc,
        RankedTensorType::get(dstShape, rewriter.getIntegerType(64),
                              dstType.getEncoding()),
        valueOffset);
    dstOffsetInfo.setOffset(offset);
  }
  // Set addPtr offset map
  auto &dstStructured = dstOffsetInfo.getStructuredRef();
  for (auto dim : dstType.getShape())
    dstStructured.push_back(dim == 1 ? PtrOffsetInfo::AxisInfo::scalar
                                     : PtrOffsetInfo::AxisInfo::scalarlike);
  dstOffsetInfo.setScalarLike(true);
  dstOffsetInfo.setPointerDescriptorOwned(
      srcOffsetInfo.isPointerDescriptorOwned());
  offsetMap[dst] = dstOffsetInfo;
}

template <typename BinOpTy>
void parseBinaryOp(BinOpTy op, const Location &loc, RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  auto &lhsStructured = lhsOffsetInfo.getStructuredRef();
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  auto &rhsStructured = rhsOffsetInfo.getStructuredRef();
  auto dst = op->getResult(0);
  PtrOffsetInfo dstOffsetInfo;
  dstOffsetInfo.setScalarLike(lhsOffsetInfo.isScalarLike() &&
                              rhsOffsetInfo.isScalarLike());
  if (dstOffsetInfo.isScalarLike())
    dstOffsetInfo.setStructured(lhsStructured.size(),
                                PtrOffsetInfo::AxisInfo::scalarlike);
  else
    dstOffsetInfo.setUnstructured(lhsStructured.size());
  offsetMap[dst] = dstOffsetInfo;
}

void parseAddI(arith::AddIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addi lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  // Get addi rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  // Set addi offset map
  auto dst = op.getResult();
  offsetMap[dst] = combineInfo(lhsOffsetInfo, rhsOffsetInfo);
}

void parseSubI(arith::SubIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get addi lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  // Get addi rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  // Set addi offset map
  auto dst = op.getResult();
  offsetMap[dst] = combineInfo(lhsOffsetInfo, rhsOffsetInfo);
  if (!(lhsOffsetInfo.isStructured() && rhsOffsetInfo.isScalarLike())) {
    offsetMap[dst].setUnstructured(offsetMap[dst].getRank());
  }
}

void parseIndexCast(arith::IndexCastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get indexCast input
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  // Set indexCast offset map
  auto dst = op.getOut();
  auto srcOffsetInfo = offsetMap.at(src);
  offsetMap[dst] = srcOffsetInfo;
}

template <typename ConstOpTy>
void parseConstantOp(ConstOpTy dst, const Location &loc, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set constant offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(true);
  if (auto tensorType =
          dyn_cast<RankedTensorType>(dst->getResult(0).getType())) {
    auto &dstStructured = offsetMap[dst].getStructuredRef();
    for (auto dim : tensorType.getShape())
      dstStructured.push_back(dim == 1 ? PtrOffsetInfo::AxisInfo::scalar
                                       : PtrOffsetInfo::AxisInfo::scalarlike);
  }
}

void parseMakeRange(triton::MakeRangeOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set makeRange offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setStructured(1);
}

void parseExtSI(arith::ExtSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extSI input
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  // Set extSI offset map
  auto dst = op.getOut();
  auto srcOffsetInfo = offsetMap.at(src);
  offsetMap[dst] = srcOffsetInfo;
}

void parseBitcast(triton::BitcastOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get bitcast src
  auto src = op.getSrc();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set extSI offset map
  auto dst = op.getResult();
  if (auto ptr = srcOffsetInfo.getPtr()) {
    Type ptrType = dst.getType();
    if (auto tensorType = dyn_cast<RankedTensorType>(ptrType))
      ptrType = tensorType.getElementType();
    rewriter.setInsertionPoint(op);
    ptr = rewriter.create<triton::BitcastOp>(loc, ptrType, ptr);
    offsetMap[dst] =
        PtrOffsetInfo(ptr, srcOffsetInfo.getOffset(), srcStructured);
  } else {
    offsetMap[dst] = PtrOffsetInfo(srcStructured);
  }
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setPointerDescriptorOwned(
      srcOffsetInfo.isPointerDescriptorOwned());
}

void parseLoad(triton::LoadOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get load ptr
  auto ptr = op.getPtr();
  parse(ptr, op.getLoc(), rewriter, offsetMap);
  // Set load offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(offsetMap[ptr].isScalarLike());
  auto tensorType = dyn_cast<RankedTensorType>(dst.getType());
  if (!tensorType)
    return;
  offsetMap[dst].setUnstructured(tensorType.getRank());
}

void parseMulI(arith::MulIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get muli lhs
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  auto &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get muli rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  auto &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set muli offset map
  size_t maxSize = std::max(lhsStructured.size(), rhsStructured.size());
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(maxSize);
  for (size_t i = 0; i < maxSize; i++)
    if (lhsScalarLike)
      dstStructured[i] = rhsStructured[i];
    else if (rhsScalarLike)
      dstStructured[i] = lhsStructured[i];
    else
      dstStructured[i] = PtrOffsetInfo::AxisInfo::unstructured;
}

void parseBroadcast(triton::BroadcastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get broadcast src
  auto src = op.getSrcMutable().get();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Get broadcast dim
  auto dst = op.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "tt.broadcast's input should be a tensor");
  auto srcType = cast<RankedTensorType>(src.getType());
  auto dstType = cast<RankedTensorType>(dst.getType());
  assert(srcType.getRank() == dstType.getRank() &&
         "rank of source shoule be equal to destnation");
  auto broadcastDim = ConverterUtils::getBroadcastDims(srcType, dstType);
  // Set broadcast offset map
  offsetMap[dst] = PtrOffsetInfo(srcOffsetInfo.getPtr());
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setPointerDescriptorOwned(
      srcOffsetInfo.isPointerDescriptorOwned());

  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get(dstType.getShape(), rewriter.getIntegerType(64),
                              dstType.getEncoding()),
        valueOffset);

    offsetMap[dst].setOffset(offset);
  }

  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (llvm::find(broadcastDim, i) != broadcastDim.end() && dstShape[i] != 1) {
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalarlike;
    } else {
      dstStructured[i] = srcStructured[i];
    }
}

void parseExpandDims(triton::ExpandDimsOp op, const Location &loc,
                     RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get expandDims src
  auto src = op.getSrc();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set expandDims offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(srcOffsetInfo.getPtr());
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setPointerDescriptorOwned(
      srcOffsetInfo.isPointerDescriptorOwned());
  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset =
        rewriter.create<triton::ExpandDimsOp>(loc, valueOffset, op.getAxis());

    offsetMap[dst].setOffset(offset);
  }
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size() + 1);
  size_t j = 0;
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (i == op.getAxis()) {
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalar;
    } else {
      dstStructured[i] = srcStructured[j];
      j++;
    }
}

void parseReshape(triton::ReshapeOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto dst = op.getResult();
  auto dstType = cast<RankedTensorType>(dst.getType());
  // Numeric reshapes retain the existing conservative classification.
  if (!isa<triton::PointerType>(dstType.getElementType()))
    return;

  parse(op.getSrc(), op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo info = offsetMap.at(op.getSrc());
  if (!info.getPtr() || !isScalarPointer(info.getPtr()) ||
      (op.getAllowReorder() && dstType.getEncoding())) {
    // Without a common scalar base (or a fixed lane mapping), keep the
    // actual result pointers. Never choose one lane's base for every lane.
    recordOpaqueTensorPointer(dst, offsetMap);
    return;
  }

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (op.getAllowReorder()) {
    // Choose the order-preserving realization permitted by allow_reorder.
    // Remaining pointer users and the derived offset must share this mapping.
    rewriter.modifyOpInPlace(op, [&] { op->removeAttr("allow_reorder"); });
  }
  auto offsetType = cast<RankedTensorType>(info.getOffset().getType());
  auto reshapedOffsetType = RankedTensorType::get(
      dstType.getShape(), offsetType.getElementType(), dstType.getEncoding());
  Value offset = rewriter.create<triton::ReshapeOp>(
      op.getLoc(), reshapedOffsetType, info.getOffset(),
      /*allowReorder=*/false, op.getEfficientLayout());
  info.setOffset(offset);
  // Reshaping complete offsets preserves addresses, but the old per-axis
  // structure does not describe the new shape. Scalar-like state must also
  // be cleared: setUnstructured alone does not force the lane-wise path.
  info.setUnstructured(dstType.getRank());
  info.setScalarLike(false);
  offsetMap[dst] = info;
}

void parseClampF(triton::ClampFOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get clampF src
  auto src = op.getX();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Get clampF min
  auto clampMin = op.getMin();
  parse(clampMin, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo minOffsetInfo = offsetMap.at(clampMin);
  // Get clampF max
  auto clampMax = op.getMax();
  parse(clampMax, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo maxOffsetInfo = offsetMap.at(clampMax);
  // Set clampF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike() &&
                               minOffsetInfo.isScalarLike() &&
                               maxOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseSelect(arith::SelectOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  if (isScalarPointer(op.getResult())) {
    recordOpaqueScalarPointer(op.getResult(), offsetMap);
    return;
  }

  if (isTensorPointer(op.getResult())) {
    recordOpaqueTensorPointer(op.getResult(), offsetMap);
    return;
  }

  // Get select condition
  auto condition = op.getCondition();
  parse(condition, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo conditionOffsetInfo = offsetMap.at(condition);
  bool conditionScalarLike = conditionOffsetInfo.isScalarLike();
  // Get select trueValue
  auto trueValue = op.getTrueValue();
  parse(trueValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo trueValueOffsetInfo = offsetMap.at(trueValue);
  auto &trueValueStructured = trueValueOffsetInfo.getStructuredRef();
  bool trueValueScalarLike = trueValueOffsetInfo.isScalarLike();
  // Get select falseValue
  auto falseValue = op.getFalseValue();
  parse(falseValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo falseValueOffsetInfo = offsetMap.at(falseValue);
  auto &falseValueStructured = falseValueOffsetInfo.getStructuredRef();
  bool falseValueScalarLike = falseValueOffsetInfo.isScalarLike();
  // Set select offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;

  auto dstIsScalar =
      trueValueScalarLike && falseValueScalarLike && conditionScalarLike;
  offsetMap[dst].setScalarLike(dstIsScalar);

  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(trueValueStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    dstStructured[i] = (dstIsScalar) ? PtrOffsetInfo::AxisInfo::scalarlike
                                     : PtrOffsetInfo::AxisInfo::unstructured;
}

void parseFPToSI(arith::FPToSIOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get FPToSI src
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set FPToSI offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank(),
                                 PtrOffsetInfo::AxisInfo::scalarlike);
  else
    offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseSIToFP(arith::SIToFPOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get SIToFP src
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Set SIToFP offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank(),
                                 PtrOffsetInfo::AxisInfo::scalarlike);
  else
    offsetMap[dst].setUnstructured(dstType.getRank());
}

// FIXME:Z|wait triton version upgrade to 3.4
// void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
//                          RewriterBase &rewriter,
//                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
//   // Set MakeTensorDesc offset map
//   auto dst = op.getResult();
//   offsetMap[dst] = PtrOffsetInfo();
//   auto dstType = dyn_cast<ShapedType>(dst.getType());
//   if (!dstType)
//     return;
//   offsetMap[dst].setStructured(dstType.getRank());
// }

void parseMakeTensorPtr(triton::MakeTensorPtrOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set MakeTensorPtr offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(dst);
  auto dstType = dyn_cast<ShapedType>(
      cast<triton::PointerType>(dst.getType()).getPointeeType());
  if (!dstType)
    return;
  offsetMap[dst].setStructured(dstType.getRank());
  offsetMap[dst].setOffsets(op.getOffsets());
}

void parseAdvance(triton::AdvanceOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set Advance offset map
  auto ptr = op.getPtr();
  parse(ptr, op.getLoc(), rewriter, offsetMap);
  auto dst = op.getResult();
  auto ptrOffsetInfo = offsetMap.at(ptr);
  offsetMap[dst] = ptrOffsetInfo;
  auto dstType = dyn_cast<ShapedType>(
      cast<triton::PointerType>(dst.getType()).getPointeeType());
  if (!dstType)
    return;
  offsetMap[dst].setStructured(dstType.getRank());
  auto &offsets = offsetMap[dst].getOffsetsRef();

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  for (auto [curOffset, opOffset] : llvm::zip(offsets, op.getOffsets())) {
    curOffset =
        rewriter.create<arith::AddIOp>(op.getLoc(), curOffset, opOffset);
  }
}

void parseReduce(triton::ReduceOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (!dstType)
    return;
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalar;
    else
      dstStructured[i] = srcStructured[i];
}

void parseReduceReturn(triton::ReduceReturnOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  auto &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  if (!dstType)
    return;
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = PtrOffsetInfo::AxisInfo::scalar;
    else
      dstStructured[i] = srcStructured[i];
}

void parseIf(scf::IfOp op, const Location &loc, RewriterBase &rewriter,
             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  if (isScalarPointer(dst)) {
    recordOpaqueScalarPointer(dst, offsetMap);
    return;
  }

  const unsigned int index = cast<OpResult>(dst).getResultNumber();
  // Get if then region
  Block &thenBlock = op.getThenRegion().front();
  Value thenYieldedValue = thenBlock.getTerminator()->getOperand(index);
  parse(thenYieldedValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo thenOffsetInfo = offsetMap.at(thenYieldedValue);
  auto &thenStructured = thenOffsetInfo.getStructuredRef();
  auto thenSrcPtr = thenOffsetInfo.getPtr();
  // Get if else region
  bool dstIsScalar = thenOffsetInfo.isScalarLike();
  bool descriptorOwned = thenOffsetInfo.isPointerDescriptorOwned();
  SmallVector<PtrOffsetInfo::AxisInfo> elseStructured;
  if (op.elseBlock()) {
    Block &elseBlock = op.getElseRegion().front();
    Value elseYieldedValue = elseBlock.getTerminator()->getOperand(index);
    parse(elseYieldedValue, op.getLoc(), rewriter, offsetMap);
    PtrOffsetInfo elseOffsetInfo = offsetMap.at(elseYieldedValue);
    elseStructured = elseOffsetInfo.getStructuredRef();
    dstIsScalar = dstIsScalar && elseOffsetInfo.isScalarLike();
    descriptorOwned =
        descriptorOwned && elseOffsetInfo.isPointerDescriptorOwned();
    if (thenSrcPtr != elseOffsetInfo.getPtr()) {
      emitError(loc)
          << "Currently ptr type from different source not supported";
    }
  }

  // Set if offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setPtr(thenSrcPtr);
  offsetMap[dst].setScalarLike(dstIsScalar);
  offsetMap[dst].setPointerDescriptorOwned(descriptorOwned);
  auto &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(thenStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (op.elseBlock())
      dstStructured[i] = (dstIsScalar) ? PtrOffsetInfo::AxisInfo::scalarlike
                                       : PtrOffsetInfo::AxisInfo::unstructured;
    else
      dstStructured[i] = thenStructured[i];
  SmallVector<Value> dstOffsets(thenOffsetInfo.getOffsetsRef().size());
  if (!dstOffsets.empty()) {
    // Assumes ifOp is already rewritten
    for (size_t i = 0; i < dstOffsets.size(); i++)
      dstOffsets[i] = op->getResult(index + i);
    offsetMap[dst].setOffsets(dstOffsets);
  }
}

void parseYield(scf::YieldOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get yield src
  for (auto src : op->getOperands())
    parse(src, op.getLoc(), rewriter, offsetMap);
}

void parseLoopOp(LoopLikeOpInterface op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  if (isScalarPointer(dst) && isPointerDescriptorBoundaryResult(op, dst)) {
    recordOpaqueScalarPointer(dst, offsetMap);
    return;
  }

  // Apply the same boundary-slot rule to results so entry, region arguments,
  // backedges and zero-trip results cannot disagree about the representation.
  if (isTensorPointer(dst) && isPointerDescriptorBoundaryResult(op, dst)) {
    recordOpaqueTensorPointer(dst, offsetMap);
    return;
  }

  auto resNum = cast<OpResult>(dst).getResultNumber();
  Value yieldedValue = nullptr;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    yieldedValue = whileOp.getConditionOp().getArgs()[resNum];
  } else {
    yieldedValue = op.getYieldedValues()[resNum];
  }
  parse(yieldedValue, op.getLoc(), rewriter, offsetMap);
  auto yieldOffsetInfo = offsetMap.at(yieldedValue);
  offsetMap[dst] = yieldOffsetInfo;
}

void parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extractSlice src
  auto src = op.getSource();
  parse(src, op.getLoc(), rewriter, offsetMap);
  // Set extractSlice offset map
  auto dst = op.getResult();
  auto srcPtrInfo = offsetMap.at(src);
  auto srcPtr = srcPtrInfo.getPtr();
  auto srcOffset = srcPtrInfo.getOffset();
  auto srcStructured = srcPtrInfo.getStructured();
  auto droppedDims = op.getDroppedDims();
  if (srcOffset) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto offsetType = getExtractSlicedType(op.getMixedSizes(), droppedDims,
                                           getElementTypeOrSelf(srcOffset));
    srcOffset = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), offsetType, srcOffset, op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
  }
  SmallVector<PtrOffsetInfo::AxisInfo> dstStructured;
  for (size_t i = 0; i < srcStructured.size(); i++) {
    if (!droppedDims[i])
      dstStructured.push_back(srcStructured[i]);
  }
  offsetMap[dst] = PtrOffsetInfo(srcPtr, srcOffset, dstStructured);
  offsetMap[dst].setPointerDescriptorOwned(
      srcPtrInfo.isPointerDescriptorOwned());
}

void parseInsertSlice(tensor::InsertSliceOp op, const Location &loc,
                      RewriterBase &rewriter,
                      llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get insertSlice src and dst
  auto src = op.getSource();
  parse(src, op.getLoc(), rewriter, offsetMap);
  auto dst = op.getDest();
  parse(dst, op.getLoc(), rewriter, offsetMap);
  // Set insertSlice offset map
  auto res = op.getResult();
  auto srcPtrInfo = offsetMap.at(src);
  auto dstPtrInfo = offsetMap.at(dst);
  PtrOffsetInfo resPtrInfo;
  if (auto srcOffset = srcPtrInfo.getOffset()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto resOffset = rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), srcOffset, dstPtrInfo.getOffset(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    auto srcPtr = srcPtrInfo.getPtr();
    auto dstPtr = dstPtrInfo.getPtr();
    assert(srcPtr == dstPtr && "ptrInfo for insert slice should be consistent");
    resPtrInfo.setPtr(srcPtr);
    resPtrInfo.setOffset(resOffset);
  }
  auto droppedDims = op.getDroppedDims();
  auto srcStructuredIter = srcPtrInfo.getStructured().begin();
  SmallVector<PtrOffsetInfo::AxisInfo> resStructured;
  auto srcShape = op.getStaticSizes();
  auto dstShape = cast<RankedTensorType>(dst.getType()).getShape();
  for (size_t i = 0; i < dstShape.size(); i++) {
    if (!ShapedType::isDynamic(srcShape[i]) && srcShape[i] == dstShape[i]) {
      resStructured.push_back(*srcStructuredIter);
    } else {
      resStructured.push_back(PtrOffsetInfo::AxisInfo::unstructured);
    }
    if (!droppedDims[i])
      ++srcStructuredIter;
  }
  resPtrInfo.setStructured(resStructured);
  resPtrInfo.setPointerDescriptorOwned(srcPtrInfo.isPointerDescriptorOwned() &&
                                       dstPtrInfo.isPointerDescriptorOwned());
  offsetMap[res] = resPtrInfo;
}

void parseExtract(tensor::ExtractOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto parentValue = op.getTensor();
  parse(parentValue, op.getLoc(), rewriter, offsetMap);
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  if (isa<triton::PointerType>(dst.getType())) {
    offsetMap[dst].setPtr(dst);
    offsetMap[dst].setZeroOffset();
  }
  offsetMap[dst].setScalarLike(true);
}

void parseInsert(tensor::InsertOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto src = op.getScalar();
  parse(src, op.getLoc(), rewriter, offsetMap);
  auto dst = op.getDest();
  parse(dst, op.getLoc(), rewriter, offsetMap);

  auto res = op.getResult();
  auto srcPtrInfo = offsetMap.at(src);
  auto dstPtrInfo = offsetMap.at(dst);

  PtrOffsetInfo resPtrInfo;
  if (auto srcOffset = srcPtrInfo.getOffset()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto resOffset = rewriter.create<tensor::InsertOp>(
        op.getLoc(), srcOffset, dstPtrInfo.getOffset(), op.getIndices());
    auto srcPtr = srcPtrInfo.getPtr();
    auto dstPtr = dstPtrInfo.getPtr();
    resPtrInfo.setPtr(srcPtr);
    resPtrInfo.setOffset(resOffset);
  }
  resPtrInfo.setUnstructured(dstPtrInfo.getRank());
  resPtrInfo.setPointerDescriptorOwned(srcPtrInfo.isPointerDescriptorOwned() &&
                                       dstPtrInfo.isPointerDescriptorOwned());
  offsetMap[res] = resPtrInfo;
}

void parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(dst);
  offsetMap[dst].setScalarLike(true);
}

namespace {
template <typename CustomOpT>
void parseStructuredCustomOpImpl(
    CustomOpT op, const Location &loc, RewriterBase &rewriter,
    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, unsigned int resultIdx) {
  for (auto operand : op.getInputs()) {
    parse(operand, op->getLoc(), rewriter, offsetMap);
  }
  auto dst = op->getResult(resultIdx);
  offsetMap[dst] = PtrOffsetInfo();
  auto tensorType = dyn_cast<RankedTensorType>(dst.getType());
  if (!tensorType) {
    if (isa<triton::PointerType>(dst.getType())) {
      offsetMap[dst].setPtr(dst);
      offsetMap[dst].setZeroOffset();
    } else if (isa<IntegerType>(dst.getType())) {
      offsetMap[dst].setOffset(dst);
    } else {
      emitError(loc) << "Unsupported return type for hivm custom op: "
                     << dst.getType();
    }
    return;
  }
  if (llvm::isa<triton::PointerType>(tensorType.getElementType())) {
    if (checkStructureAnnotated(op, rewriter)) {
      auto srcValArrayAttr = op->template getAttrOfType<DenseI32ArrayAttr>(
          ConverterUtils::customSrcPtrIndexAttrName);
      assert(srcValArrayAttr &&
             "structure hivm custom op should present src tensor<tt.ptr>");
      auto srcValArray = srcValArrayAttr.asArrayRef();
      assert(srcValArray[resultIdx] != -1 &&
             "tensor<tt.ptr> result should map to src tensor<tt.ptr>");
      auto srcOffsetInfo = offsetMap[op->getOperand(srcValArray[resultIdx])];
      offsetMap[dst] = srcOffsetInfo;
      return;
    }
    emitError(loc) << "Unsupported return unstructure RankedTensor of tt.ptr "
                      "for hivm custom op: "
                   << dst;
  }
  offsetMap[dst].setUnstructured(tensorType.getRank());
}
} // namespace

void parseStructuredCustomOp(Operation *op, const Location &loc,
                             RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                             unsigned int resultIdx) {
  if (auto customOp = dyn_cast<hivm::CustomOp>(op)) {
    parseStructuredCustomOpImpl(customOp, loc, rewriter, offsetMap, resultIdx);
  } else if (auto macroOp = dyn_cast<hivm::CustomMacroOp>(op)) {
    parseStructuredCustomOpImpl(macroOp, loc, rewriter, offsetMap, resultIdx);
  } else {
    llvm_unreachable("expected hivm custom op");
  }
}

} // namespace triton
} // namespace mlir
