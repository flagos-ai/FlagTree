#include "TritonToUnstructure/OffsetAnalysis.h"
#include "Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-offset-analysis"

namespace mlir {
namespace triton {

PtrOffsetInfo::PtrOffsetInfo() : ptr(nullptr), offset(nullptr) {}

PtrOffsetInfo::PtrOffsetInfo(const PtrOffsetInfo &other) {
  *this = other;
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr) : ptr(ptr) {
  setZeroOffset();
}

PtrOffsetInfo::PtrOffsetInfo(ArrayRef<bool> structured) : ptr(nullptr), offset(nullptr) {
  setStructured(structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, bool structured) : ptr(ptr) {
  setZeroOffset();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, ArrayRef<bool> structured) : ptr(ptr) {
  setStructured(structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, const Value &offset, bool structured) : ptr(ptr), offset(offset) {
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), structured);
}

PtrOffsetInfo::PtrOffsetInfo(const Value &ptr, const Value &offset, ArrayRef<bool> structured) : ptr(ptr), offset(offset) {
  setStructured(structured);
}

PtrOffsetInfo &PtrOffsetInfo::operator=(const PtrOffsetInfo &other) {
  setPtr(other.getPtr());
  setOffset(other.getOffset());
  setStructured(other.getStructured());
  setScalarLike(other.isScalarLike());
  setNegativeFlag(other.isNegativeFlag());
  return *this;
}

Value PtrOffsetInfo::getPtr() const { return this->ptr; }
Value PtrOffsetInfo::getOffset() const { return this->offset; }
bool PtrOffsetInfo::isScalarLike() const { return this->scalarLike; }
bool PtrOffsetInfo::isNegativeFlag() const { return this->negativeFlag; }

SmallVector<bool> &PtrOffsetInfo::getStructuredRef() { return this->structured; }
const SmallVector<bool> &PtrOffsetInfo::getStructured() const {
  return this->structured;
}

int PtrOffsetInfo::getRank() const {
  return structured.size();
}

void PtrOffsetInfo::setPtr(const Value &ptr) { this->ptr = ptr; }
void PtrOffsetInfo::setOffset(const Value &offset) { this->offset = offset; }

void PtrOffsetInfo::setStructured() {
  assert(ptr && "ptr Should be to infer rank");
  this->structured.clear();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), true);
}

void PtrOffsetInfo::setStructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, true);
}

void PtrOffsetInfo::setUnstructured() {
  assert(ptr && "ptr Should be to infer rank");
  this->structured.clear();
  if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
    this->structured.resize(tensorType.getRank(), false);
}

void PtrOffsetInfo::setUnstructured(int rank) {
  this->structured.clear();
  this->structured.resize(rank, false);
}

void PtrOffsetInfo::setStructured(ArrayRef<bool> structured) {
  this->structured.resize(structured.size());
  for (size_t i = 0; i < structured.size(); i++)
    this->structured[i] = structured[i];
}

void PtrOffsetInfo::setStructured(const PtrOffsetInfo &other) {
  this->setStructured(other.getStructured());
}

void PtrOffsetInfo::setNegativeFlag(bool negativeFlag) {
  this->negativeFlag = negativeFlag;
}

void PtrOffsetInfo::setScalarLike(bool scalarLike) {
  this->scalarLike = scalarLike;
}

bool PtrOffsetInfo::isStructured(int dim) const {
  return this->scalarLike || structured[dim];
}

bool PtrOffsetInfo::isStructured() const {
  return this->scalarLike ||
         llvm::all_of(structured, [](auto dim) { return dim; });
}

bool PtrOffsetInfo::isUnstructured() const {
  return llvm::all_of(structured, [](auto dim) { return !dim; });
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
                                         builder.getIntegerType(64)),
                   builder.getZeroAttr(builder.getIntegerType(64))));
  } else {
    offset = builder.create<arith::ConstantOp>(
        ptr.getLoc(), builder.getI64IntegerAttr(0));
  }
  setOffset(offset);
}

PtrOffsetInfo combineInfo(const PtrOffsetInfo &lhs, const PtrOffsetInfo &rhs) {
  PtrOffsetInfo info;
  assert(lhs.getRank() == rhs.getRank() &&
         "Rank must be same to be combined");

  info.setScalarLike(lhs.isScalarLike() &&
                     rhs.isScalarLike());
  SmallVector<bool> &structuredRef = info.getStructuredRef();
  structuredRef.resize(lhs.getRank());
  for (size_t i = 0; i < structuredRef.size(); i++)
    structuredRef[i] = lhs.isStructured(i) && rhs.isStructured(i);
  info.setNegativeFlag(lhs.isNegativeFlag() || rhs.isNegativeFlag());
  return info;
}

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
      }
    } 
  } else if (auto ptrType = dyn_cast<triton::PointerType>(operand.getType())) {
    offsetMap[operand] = PtrOffsetInfo(operand, true);
  } else if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
    auto parentOp = blockArgument.getOwner()->getParentOp();
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Handling block argument\n" << *parentOp << '\n';
    });
    if (isa<FunctionOpInterface>(parentOp)) {
      offsetMap[operand] = PtrOffsetInfo();
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
      os << s;
    os << "\n";
    os <<  "FNparse: " << operand << " ,isNegativeFlag: " << data.isNegativeFlag() << "\n";
  });
}

void parseLoopRegionIterArg(LoopLikeOpInterface loopOp, const Location &loc,
                            RewriterBase &rewriter,
                            llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                            BlockArgument regionIterArg) {
  auto regionIterArgInfo = PtrOffsetInfo(regionIterArg);
  OpOperand *initArgOperand = loopOp.getTiedLoopInit(regionIterArg);
  if (!initArgOperand)
    return;
  Value initArg = initArgOperand->get();
  parse(initArg, loc, rewriter, offsetMap);
  regionIterArgInfo.setStructured(offsetMap[initArg]);
  offsetMap[regionIterArg] = regionIterArgInfo;
}

void parseArithOp(Operation *arithOp, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  assert(isa<arith::ArithDialect>(arithOp->getDialect()));
  if (auto addIOp = dyn_cast<arith::AddIOp>(arithOp)) {
    parseAddI(addIOp, loc, rewriter, offsetMap);
  } else if (auto subIOp = dyn_cast<arith::SubIOp>(arithOp)) {
    parseBinaryOp(subIOp, loc, rewriter, offsetMap);
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
  } else if (auto clampFOp = dyn_cast<triton::ClampFOp>(tritonOp)) {
    parseClampF(clampFOp, loc, rewriter, offsetMap);
  } else if (auto makeTensorDescOp =
                 dyn_cast<triton::MakeTensorDescOp>(tritonOp)) {
    parseMakeTensorDesc(makeTensorDescOp, loc, rewriter, offsetMap);
  } else if (auto makeTensorPtrOp =
                 dyn_cast<triton::MakeTensorPtrOp>(tritonOp)) {
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
  // Get addPtr offset
  Value offsetValue = op.getOffset();
  parse(offsetValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo ptrOffsetInfo = offsetMap.at(ptr);
  PtrOffsetInfo offsetOffsetInfo = offsetMap.at(offsetValue);
  // Modify IR

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto offsetType = dyn_cast<RankedTensorType>(offsetValue.getType())) {
    auto offsetElementType = cast<IntegerType>(offsetType.getElementType());
    if (offsetElementType.getWidth() != 64) {
      auto newOffsetType = RankedTensorType::get(offsetType.getShape(),
                                                 rewriter.getIntegerType(64));
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
  auto dstOffsetInfo = combineInfo(ptrOffsetInfo, offsetOffsetInfo);
  dstOffsetInfo.setPtr(ptrOffsetInfo.getPtr());
  dstOffsetInfo.setOffset(offset);
  offsetMap[dst] = dstOffsetInfo;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    SmallVector<bool> &ptrStructured = ptrOffsetInfo.getStructuredRef();
    SmallVector<bool> &offsetStructured = offsetOffsetInfo.getStructuredRef();
    os << "[parseAddPtr] ptrStructured: ";
    for (size_t i = 0; i < ptrStructured.size(); i++)
      os << ptrStructured[i];
    os << "\n";
    os << "[parseAddPtr] offsetStructured: ";
    for (size_t i = 0; i < offsetStructured.size(); i++)
      os << offsetStructured[i];
    os << "\n";
    os << "[parseAddPtr] offsetOffsetInfo.isNegativeFlag(): ";
      os << offsetOffsetInfo.isNegativeFlag();
    os << "\n";
    os << "[parseAddPtr] ptrOffsetInfo.isNegativeFlag(): ";
      os << ptrOffsetInfo.isNegativeFlag();
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
        loc, RankedTensorType::get(dstShape, rewriter.getIntegerType(64)),
        valueOffset);
    dstOffsetInfo.setOffset(offset);
  }
  // Set addPtr offset map

  dstOffsetInfo.setStructured(dstType.getRank());
  dstOffsetInfo.setScalarLike(true);
  dstOffsetInfo.setNegativeFlag(srcOffsetInfo.isNegativeFlag());
  offsetMap[dst] = dstOffsetInfo;
}

template <typename BinOpTy>
void parseBinaryOp(BinOpTy op, const Location &loc, RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto lhs = op.getLhs();
  parse(lhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo lhsOffsetInfo = offsetMap.at(lhs);
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  auto dst = op->getResult(0);
  PtrOffsetInfo dstOffsetInfo;
  dstOffsetInfo.setScalarLike(lhsOffsetInfo.isScalarLike() &&
                               rhsOffsetInfo.isScalarLike());
  if (dstOffsetInfo.isScalarLike())
    dstOffsetInfo.setStructured(lhsStructured.size());
  else
    dstOffsetInfo.setUnstructured(lhsStructured.size());

  if (isa<arith::SubIOp, arith::SubFOp>(op.getOperation())) {
    dstOffsetInfo.setNegativeFlag(true);
  } else {
    dstOffsetInfo.setNegativeFlag(lhsOffsetInfo.isNegativeFlag() ||
                                     rhsOffsetInfo.isNegativeFlag());
  }
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

void parseIndexCast(arith::IndexCastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get indexCast input
  auto src = op.getIn();
  parse(src, op.getLoc(), rewriter, offsetMap);
  // Set indexCast offset map
  auto dst = op.getOut();
  offsetMap[dst] = offsetMap.at(src);
}

template <typename AttrTy, typename TypeTy>
bool isConstantNegative(AttrTy attr, TypeTy type) {
  if constexpr (std::is_same_v<AttrTy, mlir::IntegerAttr> &&
                std::is_same_v<TypeTy, mlir::IntegerType>) {
    return attr.getInt() < 0;
  } else if constexpr (std::is_same_v<AttrTy, mlir::FloatAttr> &&
                     std::is_same_v<TypeTy, mlir::FloatType>) {
    return attr.getValueAsDouble() < 0.0;
  } else if constexpr(std::is_same_v<AttrTy, mlir::IntegerAttr> &&
                    std::is_same_v<TypeTy, mlir::IndexType>) {
      return attr.getInt() < 0;
  }  else if constexpr (std::is_base_of_v<mlir::DenseElementsAttr, AttrTy> &&
                     std::is_base_of_v<mlir::RankedTensorType, TypeTy>) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
    auto elemType = tensorType.getElementType();

    if (auto denseIntAttr = dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      if (auto intElemType = dyn_cast<mlir::IntegerType>(elemType)) {
        for (auto elemVal : denseIntAttr.template getValues<mlir::APInt>()) {
          auto elemAttr = mlir::IntegerAttr::get(intElemType, elemVal);
          if (isConstantNegative(elemAttr, intElemType)) {
            LLVM_DEBUG({
              llvm::dbgs() << DEBUG_TYPE << " PCO: Tensor has negative element: " << elemAttr << "\n";
            });
            return true;
          }
        }
        return false;
      }
    }

    else if (auto denseFloatAttr = dyn_cast<mlir::DenseFPElementsAttr>(attr)) {
      if (auto floatElemType = dyn_cast<mlir::FloatType>(elemType)) {
        for (auto elemVal : denseFloatAttr.template getValues<mlir::APFloat>()) {
          auto elemAttr = mlir::FloatAttr::get(floatElemType, elemVal);
          if (isConstantNegative(elemAttr, floatElemType)) {
            LLVM_DEBUG({
              llvm::dbgs() << DEBUG_TYPE << " PCO: Tensor has negative element: " << elemAttr << "\n";
            });
            return true;
          }
        }
        return false;
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " PCO: Unsupported tensor elemType: " << elemType
                  << ",tensorType:" << tensorType << "\n";
    });
    return false;
  } else {
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " PCO, Unsupported: attr: " << attr
                  << ", type: " << type << " \n";
    });
    return false;
  }
}

template <typename ConstOpTy>
void parseConstantOp(ConstOpTy dst, const Location &loc,
                     RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  mlir::Operation *opPtr = nullptr;
  if constexpr (std::is_pointer_v<ConstOpTy>) {
    if (dst != nullptr) {
      opPtr = dst->getOperation();
    }
  } else {
    opPtr = dst.getOperation();
  }

  mlir::Value opResult = opPtr->getResult(0);

  offsetMap[opResult] = PtrOffsetInfo();
  offsetMap[opResult].setScalarLike(true);
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(opResult.getType())) {
    offsetMap[opResult].setStructured(tensorType.getRank());
  }

  auto constantOp = mlir::dyn_cast<mlir::arith::ConstantOp>(opPtr);
  if (!constantOp) {
    LLVM_DEBUG({
      llvm::dbgs() << "Warning: Non-ConstantOp (" << opPtr->getName()
                   << ") passed to parseConstantOp\n";
    });
    return;
  }

  mlir::Attribute constAttr = constantOp.getValue();
  mlir::Type resultType = opResult.getType();

  if (auto intType = dyn_cast<mlir::IntegerType>(resultType)) {
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constAttr)) {
      offsetMap[opResult].setNegativeFlag(isConstantNegative(intAttr, intType));
    }
  } else if (auto floatType = dyn_cast<mlir::FloatType>(resultType)) {
    if (auto floatAttr = dyn_cast<mlir::FloatAttr>(constAttr)) {
      offsetMap[opResult].setNegativeFlag(isConstantNegative(floatAttr, floatType));
    }
  } else if (auto indexType = dyn_cast<mlir::IndexType>(resultType)) {
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constAttr)) {
      offsetMap[opResult].setNegativeFlag(isConstantNegative(intAttr, indexType));
    }
  } else if (auto indexType = dyn_cast<mlir::RankedTensorType>(resultType)) {
    if (auto intAttr = dyn_cast<mlir::DenseIntElementsAttr>(constAttr)) {
      offsetMap[opResult].setNegativeFlag(isConstantNegative(intAttr, indexType));
    }
  } else {
    llvm_unreachable("PCO: Non-ConstantOp passed to parseConstantOp \n");
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
  offsetMap[dst] = offsetMap.at(src);
}

void parseBitcast(triton::BitcastOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get bitcast src
  auto src = op.getSrc();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set extSI offset map
  auto dst = op.getResult();
  if (auto ptr = srcOffsetInfo.getPtr()) {
    Type ptrType = dst.getType();
    if (auto tensorType = dyn_cast<RankedTensorType>(ptrType))
      ptrType = tensorType.getElementType();
    rewriter.setInsertionPoint(op);
    ptr = rewriter.create<triton::BitcastOp>(loc, ptrType, ptr);
    offsetMap[dst] = PtrOffsetInfo(ptr, srcOffsetInfo.getOffset(), srcStructured);
  } else {
    offsetMap[dst] = PtrOffsetInfo(srcStructured);
  }
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
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
  offsetMap[dst].setNegativeFlag(offsetMap[ptr].isNegativeFlag());
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
  SmallVector<bool> &lhsStructured = lhsOffsetInfo.getStructuredRef();
  bool lhsScalarLike = lhsOffsetInfo.isScalarLike();
  // Get muli rhs
  auto rhs = op.getRhs();
  parse(rhs, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo rhsOffsetInfo = offsetMap.at(rhs);
  SmallVector<bool> &rhsStructured = rhsOffsetInfo.getStructuredRef();
  bool rhsScalarLike = rhsOffsetInfo.isScalarLike();
  // Set muli offset map
  size_t maxSize = std::max(lhsStructured.size(), rhsStructured.size());
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(lhsScalarLike && rhsScalarLike);
  offsetMap[dst].setNegativeFlag(lhsOffsetInfo.isNegativeFlag()
                                  || rhsOffsetInfo.isNegativeFlag());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(maxSize);
  for (size_t i = 0; i < maxSize; i++)
    if (lhsScalarLike)
      dstStructured[i] = rhsStructured[i];
    else if (rhsScalarLike)
      dstStructured[i] = lhsStructured[i];
    else
      dstStructured[i] = false;
}

void parseBroadcast(triton::BroadcastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get broadcast src
  auto src = op.getSrcMutable().get();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
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
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());

  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::BroadcastOp>(
        loc,
        RankedTensorType::get(dstType.getShape(), rewriter.getIntegerType(64)),
        valueOffset);

    offsetMap[dst].setOffset(offset);
  }

  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (llvm::find(broadcastDim, i) != broadcastDim.end())
      dstStructured[i] = true;
    else
      dstStructured[i] = srcStructured[i];
}

void parseExpandDims(triton::ExpandDimsOp op, const Location &loc,
                     RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get expandDims src
  auto src = op.getSrcMutable().get();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set expandDims offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(srcOffsetInfo.getPtr());
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
  if (srcOffsetInfo.getPtr()) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value valueOffset = srcOffsetInfo.getOffset();
    Value offset = rewriter.create<triton::ExpandDimsOp>(loc, valueOffset,
                                                         op.getAxisAttr());

    offsetMap[dst].setOffset(offset);
  }
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(srcStructured.size() + 1);
  size_t j = 0;
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (i == op.getAxis()) {
      dstStructured[i] = true;
    } else {
      dstStructured[i] = srcStructured[j];
      j++;
    }
}

void parseClampF(triton::ClampFOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get clampF src
  auto src = op.getX();
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  // Get clampF min
  auto clampMin = op.getX();
  parse(clampMin, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo minOffsetInfo = offsetMap.at(clampMin);
  // Get clampF max
  auto clampMax = op.getX();
  parse(clampMax, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo maxOffsetInfo = offsetMap.at(clampMax);
  // Set clampF offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike() &&
                               minOffsetInfo.isScalarLike() &&
                               maxOffsetInfo.isScalarLike());
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag() ||
                               minOffsetInfo.isNegativeFlag() ||
                               maxOffsetInfo.isNegativeFlag());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseSelect(arith::SelectOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get select condition
  auto condition = op.getCondition();
  parse(condition, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo conditionOffsetInfo = offsetMap.at(condition);
  bool conditionScalarLike = conditionOffsetInfo.isScalarLike();
  // Get select trueValue
  auto trueValue = op.getTrueValue();
  parse(trueValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo trueValueOffsetInfo = offsetMap.at(trueValue);
  SmallVector<bool> &trueValueStructured = trueValueOffsetInfo.getStructuredRef();
  bool trueValueScalarLike = trueValueOffsetInfo.isScalarLike();
  // Get select falseValue
  auto falseValue = op.getFalseValue();
  parse(falseValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo falseValueOffsetInfo = offsetMap.at(falseValue);
  SmallVector<bool> &falseValueStructured = falseValueOffsetInfo.getStructuredRef();
  bool falseValueScalarLike = falseValueOffsetInfo.isScalarLike();
  // Set select offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(conditionScalarLike && trueValueScalarLike &&
                               falseValueScalarLike);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst].setNegativeFlag(trueValueOffsetInfo.isNegativeFlag() || 
                                   falseValueOffsetInfo.isNegativeFlag());
  if (!dstType)
    return;
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank());
  else
    offsetMap[dst].setUnstructured(dstType.getRank());
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
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank());
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
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  if (offsetMap[dst].isScalarLike())
    offsetMap[dst].setStructured(dstType.getRank());
  else
    offsetMap[dst].setUnstructured(dstType.getRank());
}

void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set MakeTensorDesc offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setStructured(dstType.getRank());
}

void parseMakeTensorPtr(triton::MakeTensorPtrOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set MakeTensorPtr offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setStructured(dstType.getRank());
}

void parseAdvance(triton::AdvanceOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Set Advance offset map
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo();
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!dstType)
    return;
  offsetMap[dst].setStructured(dstType.getRank());
}

void parseReduce(triton::ReduceOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get reduce src
  Value src = op->getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo srcOffsetInfo = offsetMap.at(src);
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = true;
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
  SmallVector<bool> &srcStructured = srcOffsetInfo.getStructuredRef();
  // Set reduce offset map
  Value dst = op->getResult(0);
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(srcOffsetInfo.isScalarLike());
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
  if (!dstType)
    return;
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  auto dstShape = dstType.getShape();
  dstStructured.resize(dstShape.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (dstShape[i] == 1)
      dstStructured[i] = true;
    else
      dstStructured[i] = srcStructured[i];
}

void parseIf(scf::IfOp op, const Location &loc, RewriterBase &rewriter,
             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  const unsigned int index = cast<OpResult>(dst).getResultNumber();
  // Get if then region
  Block &thenBlock = op.getThenRegion().front();
  Value thenYieldedValue = thenBlock.getTerminator()->getOperand(index);
  parse(thenYieldedValue, op.getLoc(), rewriter, offsetMap);
  PtrOffsetInfo thenOffsetInfo = offsetMap.at(thenYieldedValue);
  SmallVector<bool> &thenStructured = thenOffsetInfo.getStructuredRef();
  // Get if else region
  bool dstIsScalar = thenOffsetInfo.isScalarLike();
  SmallVector<bool> elseStructured;
  if (op.elseBlock()) {
    Block &elseBlock = op.getElseRegion().front();
    Value elseYieldedValue = elseBlock.getTerminator()->getOperand(index);
    parse(elseYieldedValue, op.getLoc(), rewriter, offsetMap);
    PtrOffsetInfo elseOffsetInfo = offsetMap.at(elseYieldedValue);
    elseStructured = elseOffsetInfo.getStructuredRef();
    dstIsScalar = dstIsScalar && elseOffsetInfo.isScalarLike();
  }
  // Set if offset map
  offsetMap[dst] = PtrOffsetInfo();
  offsetMap[dst].setScalarLike(dstIsScalar);
  offsetMap[dst].setNegativeFlag(thenOffsetInfo.isNegativeFlag());
  SmallVector<bool> &dstStructured = offsetMap[dst].getStructuredRef();
  dstStructured.resize(thenStructured.size());
  for (size_t i = 0; i < dstStructured.size(); i++)
    if (op.elseBlock())
      dstStructured[i] = thenStructured[i] && elseStructured[i];
    else
      dstStructured[i] = thenStructured[i];
}

void parseYield(scf::YieldOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get yield src
  for (auto src : op->getOperands())
    parse(src, op.getLoc(), rewriter, offsetMap);
}

void parseLoopOp(LoopLikeOpInterface op, const Location &loc, RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst) {
  auto resNum = cast<OpResult>(dst).getResultNumber();
  Value yieldedValue = op.getYieldedValues()[resNum];
  parse(yieldedValue, op.getLoc(), rewriter, offsetMap);
  offsetMap[dst] = PtrOffsetInfo() = offsetMap.at(yieldedValue);
}

void parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  // Get extractSlice src
  auto src = op.getOperand(0);
  parse(src, op.getLoc(), rewriter, offsetMap);
  // Set extractSlice offset map
  auto dst = op.getResult();
  offsetMap[dst] = offsetMap.at(src);
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
  }
  offsetMap[dst].setScalarLike(true);
}

void parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  auto dst = op.getResult();
  offsetMap[dst] = PtrOffsetInfo(dst);
  offsetMap[dst].setScalarLike(true);

  parse(op.getSrc(), op.getLoc(), rewriter, offsetMap);
  auto srcOffsetInfo = offsetMap.at(op.getSrc());
  offsetMap[dst].setNegativeFlag(srcOffsetInfo.isNegativeFlag());
}

} // namespace triton
} // namespace mlir