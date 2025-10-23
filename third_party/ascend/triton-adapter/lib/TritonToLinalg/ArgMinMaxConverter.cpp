#include "TritonToLinalg/ArgMinMaxConverter.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

// ArgMinConverter functions
LogicalResult ArgMinConverter::matchComparisonResult(
    Value currValue, Value currIndex, Value reduceValue, Value reduceIndex,
    mlir::Block::iterator &it, Value &comparisonResult) {
  LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");

  auto cmpOp = dyn_cast<arith::CmpFOp>(*it);
  auto cmpIOp = dyn_cast<arith::CmpIOp>(*it++);
  if (!cmpOp && !cmpIOp)
    return failure();

  if (cmpOp) {
    if (cmpOp.getPredicate() != arith::CmpFPredicate::OLT ||
        currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpOp;
  }

  if (cmpIOp) {
    if ((cmpIOp.getPredicate() != arith::CmpIPredicate::slt &&
         cmpIOp.getPredicate() != arith::CmpIPredicate::ult) ||
        currValue != cmpIOp.getLhs() || reduceValue != cmpIOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpIOp;
  }

  return success();
}

float ArgMinConverter::getBaseReductionValue() {
  return std::numeric_limits<float>::infinity();
}

int8_t ArgMinConverter::getBaseReductionIntValue() { return 127; }

// ArgMaxConverter functions
LogicalResult ArgMaxConverter::matchComparisonResult(
    Value currValue, Value currIndex, Value reduceValue, Value reduceIndex,
    mlir::Block::iterator &it, Value &comparisonResult) {
  auto cmpOp = dyn_cast<arith::CmpFOp>(*it);
  auto cmpIOp = dyn_cast<arith::CmpIOp>(*it++);
  if (!cmpOp && !cmpIOp)
    return failure();

  if (cmpOp) {
    if (cmpOp.getPredicate() != arith::CmpFPredicate::OGT ||
        currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpOp;
  }

  if (cmpIOp) {
    if ((cmpIOp.getPredicate() != arith::CmpIPredicate::sgt &&
         cmpIOp.getPredicate() != arith::CmpIPredicate::ugt) ||
        currValue != cmpIOp.getLhs() || reduceValue != cmpIOp.getRhs()) {
      return failure();
    }
    comparisonResult = cmpIOp;
  }

  return success();
}

float ArgMaxConverter::getBaseReductionValue() {
  return -std::numeric_limits<float>::infinity();
}

int8_t ArgMaxConverter::getBaseReductionIntValue() { return -128; }

} // namespace TTOpConverters