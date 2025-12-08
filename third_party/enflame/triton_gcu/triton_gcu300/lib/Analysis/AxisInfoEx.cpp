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
#include "Analysis/AxisInfoEx.h"
#include <memory>
#include <string>
#include <vector>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "axis-info-ex"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gcu {
namespace {

int64_t gcdImpl(int64_t a, int64_t b, int64_t *x, int64_t *y) {
  // Base Case
  if (a == 0) {
    *x = 0;
    *y = 1;
    return b;
  }
  int64_t x1, y1; // To store results of recursive call
  int64_t gcd = gcdImpl(b % a, a, &x1, &y1);
  // Update x and y using results of
  // recursive call
  *x = y1 - (b / a) * x1;
  *y = x1;
  return gcd;
}

int64_t gcd(int64_t a, int64_t b) {
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  int64_t x, y;
  return gcdImpl(a, b, &x, &y);
}

constexpr int log2Int(int64_t num) {
  return (num > 1) ? 1 + log2Int(num / 2) : 0;
}

// If lhs * rhs overflows, return max value possible value for the type
int64_t multiplyDivisor(int64_t lhs, int64_t rhs) {
  int64_t maxDivisor = highestPowOf2Divisor<int64_t>(0);
  if (lhs > maxDivisor / rhs)
    return maxDivisor;
  return lhs * rhs;
}

class AxisInfoExVisitor {
public:
  AxisInfoExVisitor() = default;
  virtual ~AxisInfoExVisitor() = default;

  static bool isContiguousDim(const AxisInfoEx &info, ArrayRef<int64_t> shape,
                              int dim) {
    return info.getContiguity(dim) == shape[dim];
  }

  static bool isConstantDim(const AxisInfoEx &info, ArrayRef<int64_t> shape,
                            int dim) {
    return info.getConstancy(dim) == shape[dim];
  }

  virtual AxisInfoEx
  getAxisInfoEx(Operation *op,
                ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) = 0;

  virtual bool match(Operation *op) = 0;
};

// Base class for all operations
template <typename OpTy>
class AxisInfoExVisitorImpl : public AxisInfoExVisitor {
public:
  using AxisInfoExVisitor::AxisInfoExVisitor;

  AxisInfoEx getAxisInfoEx(
      Operation *op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) final {
    return getAxisInfoEx(cast<OpTy>(op), operands);
  }

  bool match(Operation *op) final { return isa<OpTy>(op); }

  virtual AxisInfoEx
  getAxisInfoEx(OpTy op,
                ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) = 0;
};

// Binary operations
template <typename OpTy>
class BinaryOpVisitorImpl : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    assert(operands.size() == 2 && "Expected two operands");
    AxisInfoEx::DimVectorT divisibility;
    AxisInfoEx::DimVectorT continualSize;
    AxisInfoEx::DimVectorT continualInterval;
    auto constantValue = getConstantValue(op, lhsInfo, rhsInfo);
    for (auto i = 0; i < rank; ++i) {
      if (constantValue.has_value()) {
        divisibility.push_back(
            highestPowOf2Divisor<int64_t>(constantValue.value()));
        continualSize.push_back(
            std::max(lhsInfo.getContinualSize(i), rhsInfo.getContinualSize(i)));
        continualInterval.push_back(0);
      } else {
        divisibility.push_back(getDivisibility(op, lhsInfo, rhsInfo, i));
        continualSize.push_back(getContinualSize(op, lhsInfo, rhsInfo, i));
        continualInterval.push_back(
            getContinualInterval(op, lhsInfo, rhsInfo, i));
      }
    }
    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }

protected:
  virtual int64_t getDivisibility(OpTy op, const AxisInfoEx &lhs,
                                  const AxisInfoEx &rhs, int dim) {
    return 1;
  }

  virtual int64_t getContinualSize(OpTy op, const AxisInfoEx &lhs,
                                   const AxisInfoEx &rhs, int dim) {
    return 1;
  }

  virtual int64_t getContinualInterval(OpTy op, const AxisInfoEx &lhs,
                                       const AxisInfoEx &rhs, int dim) {
    return 1;
  }

  virtual std::optional<int64_t>
  getConstantValue(OpTy op, const AxisInfoEx &lhs, const AxisInfoEx &rhs) {
    return std::nullopt;
  }
};

class AxisInfoExVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  AxisInfoEx apply(Operation *op,
                   ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) {
    for (auto &visitor : visitors)
      if (visitor->match(op))
        return visitor->getAxisInfoEx(op, operands);
    return AxisInfoEx();
  }

private:
  std::vector<std::unique_ptr<AxisInfoExVisitor>> visitors;
};

class AxisInfoExAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                               dataflow::Lattice<AxisInfoEx>> {
private:
  AxisInfoExVisitorList visitors;

  void setToEntryState(dataflow::Lattice<AxisInfoEx> *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(AxisInfoEx::getPessimisticValueState(
                           lattice->getAnchor())));
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<dataflow::Lattice<AxisInfoEx> *> argLattices,
      unsigned firstIndex) override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      visitForOpInductionVar(forOp, argLattices);
    } else {
      setAllToEntryStates(argLattices.take_front(firstIndex));
      setAllToEntryStates(argLattices.drop_front(
          firstIndex + successor.getSuccessorInputs().size()));
    }
  }

public:
  explicit AxisInfoExAnalysis(DataFlowSolver &solver);
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AxisInfoEx>>::getLatticeElement;
  using FuncAxisInfoMapT = DenseMap<FunctionOpInterface, AxisInfoEx>;

  llvm::LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands,
                 ArrayRef<dataflow::Lattice<AxisInfoEx> *> results) override;
  void
  visitForOpInductionVar(scf::ForOp op,
                         ArrayRef<dataflow::Lattice<AxisInfoEx> *> argLattices);
};

template <typename OpTy>
class CastOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    return operands[0]->getValue();
  }
};

class MakeRangeOpAxisInfoExVisitor final
    : public AxisInfoExVisitorImpl<triton::MakeRangeOp> {
public:
  using AxisInfoExVisitorImpl<triton::MakeRangeOp>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      triton::MakeRangeOp op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    auto start = op.getStart();
    auto end = op.getEnd();
    return AxisInfoEx(/*divisibility=*/{highestPowOf2Divisor(start)},
                      /*continualSize=*/{end - start},
                      /*continualInterval=*/{1});
  }
};

template <typename OpTy>
class ConstantOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (intAttr || boolAttr) {
      int64_t value{};
      if (intAttr)
        value = intAttr.getValue().getZExtValue();
      else
        value = boolAttr.getValue() ? 1 : 0;
      return AxisInfoEx(/*divisibility=*/{highestPowOf2Divisor(value)},
                        /*continualSize=*/{AxisInfoEx::kDefaultContinueSize},
                        /*continualInterval=*/{0},
                        /*knownConstantValue=*/{value});
    }

    auto splatAttr = dyn_cast<SplatElementsAttr>(op.getValue());
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      int64_t value = splatAttr.template getSplatValue<APInt>().getZExtValue();
      TensorType ty = cast<TensorType>(splatAttr.getType());
      return AxisInfoEx(
          /*divisibility=*/
          AxisInfoEx::DimVectorT(ty.getRank(), highestPowOf2Divisor(value)),
          /*continualSize=*/
          AxisInfoEx::DimVectorT(ty.getShape().begin(), ty.getShape().end()),
          /*continualInterval=*/
          AxisInfoEx::DimVectorT(ty.getRank(), 0),
          /*knownConstantValue=*/{value});
    }
    return AxisInfoEx();
  }
};

template <typename OpTy>
class AddSubOpAxisInfoExVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getDivisibility(OpTy op, const AxisInfoEx &lhs, const AxisInfoEx &rhs,
                          int dim) override {
    // lhs = k * d_lhs = k * k' * gcd(d_lhs, d_rhs)
    // rhs = p * d_rhs = p * p' * gcd(d_lhs, d_rhs)
    // lhs + rhs = k * d_lhs + p * d_rhs = (k * d_lhs + p * d_rhs) *
    // gcd(d_lhs, d_rhs)
    auto rhsDivisibility = rhs.getDivisibility(dim);
    return gcd(lhs.getDivisibility(dim), rhsDivisibility);
  }

  int64_t getContinualSize(OpTy op, const AxisInfoEx &lhs,
                           const AxisInfoEx &rhs, int dim) override {
    return gcd(lhs.getContinualSize(dim), rhs.getContinualSize(dim));
  }

  int64_t getContinualInterval(OpTy op, const AxisInfoEx &lhs,
                               const AxisInfoEx &rhs, int dim) override {
    if (lhs.getContinualInterval(dim) ==
            AxisInfoEx::kDefaultContinualInterval ||
        rhs.getContinualInterval(dim) == AxisInfoEx::kDefaultContinualInterval)
      return AxisInfoEx::kDefaultContinualInterval;
    return std::abs(
        applyOp(lhs.getContinualInterval(dim), rhs.getContinualInterval(dim)));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfoEx &lhs,
                                          const AxisInfoEx &rhs) override {
    if (!lhs.getConstantValue().has_value() ||
        !rhs.getConstantValue().has_value()) {
      return std::nullopt;
    }

    return {applyOp(lhs.getConstantValue().value(),
                    rhs.getConstantValue().value())};
  }

private:
  static int64_t applyOp(int64_t lhs, int64_t rhs) {
    static_assert(std::is_same_v<OpTy, arith::SubIOp> ||
                  std::is_same_v<OpTy, arith::AddIOp> ||
                  std::is_same_v<OpTy, LLVM::AddOp> ||
                  std::is_same_v<OpTy, triton::AddPtrOp>);
    if constexpr (std::is_same_v<OpTy, arith::SubIOp>) {
      return lhs - rhs;
    }
    return lhs + rhs;
  }
};

class MulIOpAxisInfoExVisitor final
    : public BinaryOpVisitorImpl<arith::MulIOp> {
public:
  using BinaryOpVisitorImpl<arith::MulIOp>::BinaryOpVisitorImpl;

private:
  int64_t getDivisibility(arith::MulIOp op, const AxisInfoEx &lhs,
                          const AxisInfoEx &rhs, int dim) override {
    auto lhsDivisibility = lhs.getDivisibility(dim);
    auto rhsDivisibility = rhs.getDivisibility(dim);
    return multiplyDivisor(lhsDivisibility, rhsDivisibility);
  }

  int64_t getContinualSize(arith::MulIOp op, const AxisInfoEx &lhs,
                           const AxisInfoEx &rhs, int dim) override {
    return std::max(gcd(lhs.getConstancy(dim), rhs.getContinualSize(dim)),
                    gcd(lhs.getContinualSize(dim), rhs.getConstancy(dim)));
  }

  int64_t getContinualInterval(arith::MulIOp op, const AxisInfoEx &lhs,
                               const AxisInfoEx &rhs, int dim) override {
    if (lhs.getContinualInterval(dim) ==
            AxisInfoEx::kDefaultContinualInterval ||
        rhs.getContinualInterval(dim) == AxisInfoEx::kDefaultContinualInterval)
      return AxisInfoEx::kDefaultContinualInterval;

    // lhs * cst
    auto lhsStrideValue =
        rhs.getConstantValue().has_value()
            ? lhs.getContinualInterval(dim) * rhs.getConstantValue().value()
            : AxisInfoEx::kDefaultContinualInterval;
    // cst * rhs
    auto rhsStrideValue =
        lhs.getConstantValue().has_value()
            ? rhs.getContinualInterval(dim) * lhs.getConstantValue().value()
            : AxisInfoEx::kDefaultContinualInterval;
    return std::max(lhsStrideValue, rhsStrideValue);
  }

  std::optional<int64_t> getConstantValue(arith::MulIOp op,
                                          const AxisInfoEx &lhs,
                                          const AxisInfoEx &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() * rhs.getConstantValue().value()};
    return std::nullopt;
  }
};

template <typename OpTy>
class DivOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    assert(operands.size() == 2 && "Expected two operands");
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resTy)
      return AxisInfoEx{};

    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto &lhs = operands[0]->getValue();
    auto &rhs = operands[1]->getValue();

    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    std::optional<int64_t> constantValue;
    for (short i = 0; i < rank; ++i) {
      if ((rhs.getConstantValue().has_value() &&
           rhs.getConstantValue().value() == 1) ||
          (lhs.getConstantValue().has_value() &&
           lhs.getConstantValue().value() == 0)) {
        // Case1: lhs / 1 or 0 / rhs, the result both equal to lhs.
        divisibility.push_back(lhs.getDivisibility(i));
        continualSize.push_back(lhs.getContinualSize(i));
        continualInterval.push_back(lhs.getContinualInterval(i));
        constantValue = {lhs.getConstantValue()};
      } else if (lhs.getConstantValue().has_value() &&
                 rhs.getConstantValue().has_value()) {
        // Case2: cst1 / cst2.
        continualSize.push_back(lhs.getConstancy(i));
        continualInterval.push_back(0);
        constantValue = {lhs.getConstantValue().value() /
                         rhs.getConstantValue().value()};
        divisibility.push_back(highestPowOf2Divisor(constantValue.value()));
      } else if (!lhs.isConstantDim(shape, i) && lhs.isContinualDim(shape, i) &&
                 rhs.isConstantDim(shape, i) &&
                 rhs.getConstantValue().has_value() &&
                 llvm::isPowerOf2_64(lhs.getContinualInterval(i))) {
        // Case 3: lhs stride(stride_val is power of 2), rhs constant.
        // lhs: d_lhs * k, d_lhs * k + s, ..., d_lhs * k + n * s
        // rhs: d_rhs * p, d_rhs * p, ..., d_rhs * p
        // lhs / rhs = d_lhs * k / (d_rhs * p), (d_lhs * k + s) / (d_rhs * p),
        // ..., (d_lhs * k + n*s) / (d_rhs * p)
        // Because d_lhs % d_rhs = 0 || d_rhs % d_lhs = 0,
        // the minimal stride is
        // minStride = max(gcd(d_lhs, d_rhs) / strideVal, 1).
        // Since minStride maybe > len(lhs),
        // we need to use another gcd to get the actual constancy.
        int64_t divisibilityGCD =
            gcd(lhs.getDivisibility(i), rhs.getDivisibility(i));
        bool isContinual =
            lhs.getContinualInterval(i) % rhs.getConstantValue().value() == 0;
        int64_t newContinualSize =
            isContinual ? lhs.getContinualSize(i)
                        : std::max<int64_t>(
                              divisibilityGCD / lhs.getContinualInterval(i), 1);
        continualSize.push_back(gcd(lhs.getContinualSize(i), newContinualSize));
        continualInterval.push_back(lhs.getContinualInterval(i) /
                                    rhs.getConstantValue().value());
        divisibility.push_back(std::max<int64_t>(
            lhs.getDivisibility(i) / rhs.getConstantValue().value(), 1));
      } else if (lhs.isStridedConstantDim(shape, i) &&
                 rhs.getConstantValue().has_value()) {
        divisibility.push_back(std::max<int64_t>(
            lhs.getDivisibility(i) / rhs.getConstantValue().value(), 1));
        continualSize.push_back(
            gcd(lhs.getContinualSize(i), rhs.getContinualSize(i)));
        continualInterval.push_back(0);
      } else {
        divisibility.push_back(AxisInfoEx::kInitDivisibility);
        continualSize.push_back(AxisInfoEx::kDefaultContinueSize);
        continualInterval.push_back(AxisInfoEx::kDefaultContinualInterval);
      }
    }
    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }
};

template <typename OpTy>
class RemOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    assert(operands.size() == 2 && "Expected two operands");
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resTy)
      return AxisInfoEx{};

    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto &lhs = operands[0]->getValue();
    auto &rhs = operands[1]->getValue();

    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    std::optional<int64_t> constantValue;
    for (short i = 0; i < rank; ++i) {
      if (rhs.getConstantValue().has_value() &&
          rhs.getConstantValue().value() == 1) {
        // Case1: lhs % 1.
        divisibility.push_back(highestPowOf2Divisor<int64_t>(0));
        continualSize.push_back(shape[i]);
        continualInterval.push_back(0);
        constantValue = {0};
      } else if (lhs.getConstantValue().has_value() &&
                 rhs.getConstantValue().has_value()) {
        // Case2: cst1 % cst2.
        constantValue = {lhs.getConstantValue().value() %
                         rhs.getConstantValue().value()};
        divisibility.push_back(highestPowOf2Divisor(constantValue.value()));
        continualSize.push_back(lhs.getConstancy(i));
        continualInterval.push_back(0);
      } else if (lhs.isContinualLowDim(shape, i) &&
                 rhs.isConstantDim(shape, i)) {
        // Case3: lhs contiguous, rhs constant.
        // lhs: d_lhs * k = gcd(d_lhs, d_rhs) * k' * k = gcd(d_lhs, d_rhs) * k''
        // rhs: d_rhs * p = gcd(d_lhs, d_rhs) * p' * p = gcd(d_lhs, d_rhs) * p''
        // lhs = gcd(d_lhs, d_rhs) * k'' = gcd(d_lhs, d_rhs) * d + r
        // r must be divisible by gcd(d_lhs, d_rhs)
        divisibility.push_back(
            gcd(lhs.getDivisibility(i), rhs.getDivisibility(i)));

        // lhs: d_lhs * k, d_lhs * k + 1, ..., d_lhs * k + n
        // rhs: d_rhs * p, d_rhs * p, ..., d_rhs * p
        // lhs % rhs = d_lhs * k % (d_rhs * p), (d_lhs * k + 1) % (d_rhs * p),
        // ..., (d_lhs * k + n) % (d_rhs * p)
        // Because d_lhs % d_rhs = 0 || d_rhs % d_lhs = 0,
        // The minimal contiguity is gcd(d_lhs, d_rhs).
        // Since gcd(d_lhs, d_rhs) maybe > len(lhs),
        // we need to use another gcd to get the actual contiguity.
        continualSize.push_back(
            gcd(lhs.getContiguity(i),
                gcd(lhs.getDivisibility(i), rhs.getDivisibility(i))));
        continualInterval.push_back(1);
      } else if (lhs.isStridedContinualDim(shape, i) &&
                 rhs.getConstantValue().has_value()) {
        // Case4: lhs strided contiguous, rhs constant value.
        divisibility.push_back(
            gcd(lhs.getDivisibility(i), rhs.getDivisibility(i)));
        continualSize.push_back(
            gcd(lhs.getContiguity(i),
                gcd(lhs.getDivisibility(i), rhs.getDivisibility(i))));
        continualInterval.push_back(lhs.getContinualInterval(i) %
                                    rhs.getConstantValue().value());
      } else if (lhs.isStridedConstantDim(shape, i) &&
                 rhs.getConstantValue().has_value()) {
        // Case5: lhs strided constant, rhs constant value.
        divisibility.push_back(
            gcd(lhs.getDivisibility(i), rhs.getDivisibility(i)));
        continualSize.push_back(lhs.getConstancy(i));
        continualInterval.push_back(0);
      } else {
        divisibility.push_back(AxisInfoEx::kInitDivisibility);
        continualSize.push_back(AxisInfoEx::kDefaultContinueSize);
        continualInterval.push_back(AxisInfoEx::kDefaultContinualInterval);
      }
    }

    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }
};

class SplatOpAxisInfoExVisitor final
    : public AxisInfoExVisitorImpl<triton::SplatOp> {
public:
  using AxisInfoExVisitorImpl<triton::SplatOp>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      triton::SplatOp op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    Type _retTy = *op->result_type_begin();
    TensorType retTy = cast<TensorType>(_retTy);
    AxisInfoEx opInfo = operands[0]->getValue();
    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    for (int i = 0; i < retTy.getRank(); ++i) {
      divisibility.push_back(opInfo.getDivisibility(0));
      continualSize.push_back(retTy.getShape()[i]);
      continualInterval.push_back(0);
    }
    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      operands[0]->getValue().getConstantValue());
  }
};

class LoadOpAxisInfoExVisitor final
    : public AxisInfoExVisitorImpl<triton::LoadOp> {
public:
  using AxisInfoExVisitorImpl<triton::LoadOp>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      triton::LoadOp op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    // If pointers and mask both have constancy properties, those properties
    // will also extend to output.
    AxisInfoEx ptrInfo = operands[0]->getValue();
    std::optional<AxisInfoEx> maskInfo;
    if (operands.size() > 1) {
      maskInfo = operands[1]->getValue();
    }
    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;

    for (int i = 0; i < ptrInfo.getRank(); ++i) {
      divisibility.push_back(ptrInfo.getDivisibility(i));
      continualSize.push_back(
          gcd(ptrInfo.getContinualSize(i),
              maskInfo.has_value() ? maskInfo->getConstancy(i) : 0));
      continualInterval.push_back(ptrInfo.getContinualInterval(i));
    }

    return AxisInfoEx(divisibility, continualSize, continualInterval);
  }
};

class ExpandDimsOpAxisInfoExVisitor final
    : public AxisInfoExVisitorImpl<triton::ExpandDimsOp> {
public:
  using AxisInfoExVisitorImpl<triton::ExpandDimsOp>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      triton::ExpandDimsOp op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    AxisInfoEx opInfo = operands[0]->getValue();
    AxisInfoEx::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfoEx::DimVectorT continualSize = opInfo.getContinualSize();
    AxisInfoEx::DimVectorT continualInterval = opInfo.getContinualInterval();

    ArrayRef<int64_t> srcShape = op.getSrc().getType().getShape();
    int64_t expandedDim = std::max(static_cast<int32_t>(op.getAxis()) - 1, 0);
    int64_t expandedDivisibility = opInfo.isConstantDim(srcShape, expandedDim)
                                       ? divisibility[expandedDim]
                                       : AxisInfoEx::kInitDivisibility;
    divisibility.insert(divisibility.begin() + op.getAxis(),
                        expandedDivisibility);
    continualSize.insert(continualSize.begin() + op.getAxis(),
                         AxisInfoEx::kDefaultContinueSize);
    continualInterval.insert(continualInterval.begin() + op.getAxis(), 0);
    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      opInfo.getConstantValue());
  }
};

class BroadcastOpAxisInfoExVisitor final
    : public AxisInfoExVisitorImpl<triton::BroadcastOp> {
public:
  using AxisInfoExVisitorImpl<triton::BroadcastOp>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      triton::BroadcastOp op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    Type _retTy = *op->result_type_begin();
    Type _opTy = *op->operand_type_begin();
    TensorType retTy = cast<TensorType>(_retTy);
    TensorType opTy = cast<TensorType>(_opTy);
    ArrayRef<int64_t> retShape = retTy.getShape();
    ArrayRef<int64_t> opShape = opTy.getShape();
    AxisInfoEx opInfo = operands[0]->getValue();
    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    for (int i = 0; i < retTy.getRank(); ++i) {
      divisibility.push_back(opInfo.getDivisibility(i));
      continualSize.push_back(opShape[i] == 1 ? retShape[i]
                                              : opInfo.getContinualSize(i));
      continualInterval.push_back(
          opShape[i] == 1 ? 0 : opInfo.getContinualInterval(i));
    }
    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      opInfo.getConstantValue());
  }
};

class TransOpAxisInfoExVisitor final
    : public AxisInfoExVisitorImpl<triton::TransOp> {
public:
  using AxisInfoExVisitorImpl<triton::TransOp>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      triton::TransOp op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    ArrayRef<int32_t> trans_order = op.getOrder();
    AxisInfoEx opInfo = operands[0]->getValue();
    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    for (unsigned i = 0; i < trans_order.size(); ++i) {
      divisibility.push_back(opInfo.getDivisibility(trans_order[i]));
      continualSize.push_back(opInfo.getContinualSize(trans_order[i]));
      continualInterval.push_back(opInfo.getContinualInterval(trans_order[i]));
    }
    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      opInfo.getConstantValue());
  }
};

template <typename OpTy>
class CmpOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resTy)
      return AxisInfoEx();
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();

    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      int64_t constancyHint = AxisInfoEx::kDefaultContinueSize;
      int64_t continualIntervalHint = AxisInfoEx::kDefaultContinualInterval;
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value()) {
        constancyHint = lhsInfo.getConstancy(d);
        constantValue =
            compare(getPredicate(op), lhsInfo.getConstantValue().value(),
                    rhsInfo.getConstantValue().value())
                ? 1
                : 0;
        continualIntervalHint = 0;
      } else if (gtPredicate(getPredicate(op)) ||
                 ltPredicate(getPredicate(op))) {
        // Lhs and rhs are both partial constants.
        constancyHint = gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));
        auto commonDivisor =
            gcd(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d));
        if (lhsInfo.isConstantDim(shape, d) &&
            rhsInfo.isContinualLowDim(shape, d)) {
          // Case 2: lhs all constant, rhs all contiguous
          // NOTE:
          // lhs: k0 * d, k0 * d, ...
          // rhs: k1 * d, k1 * d + 1, ...
          // lhs lt rhs: 1, 1, 1, 1 (minimal len: d if k0 <= k1)
          // lhs lt rhs: 0, 0, 0, 0 (minimal len: d if k0 > k1)
          // lhs gt rhs: 0, 0, 0, 0 (minimal len: d if k0 <= k1)
          // lhs gt rhs: 1, 1, 1, 1 (minimal len: d if k0 > k1)
          constancyHint = std::max(
              constancyHint, gcd(rhsInfo.getContiguity(d), commonDivisor));
        } else if (lhsInfo.isContinualLowDim(shape, d) &&
                   rhsInfo.isConstantDim(shape, d)) {
          // Case 3: lhs all contiguous, rhs all constant
          // NOTE
          // lhs: k0 * d, k0 * d + 1, ...
          // rhs: k1 * d, k1 * d, ...
          // lhs gt rhs: 1, 1, 1, 1 (minimal len: d if k0 >= k1)
          // lhs gt rhs: 0, 0, 0, 0 (minimal len: d if k0 < k1)
          // lhs lt rhs: 0, 0, 0, 0 (minimal len: d if k0 >= k1)
          // lhs lt rhs: 1, 1, 1, 1 (minimal len: d if k0 < k1)
          constancyHint = std::max(
              constancyHint, gcd(lhsInfo.getContiguity(d), commonDivisor));
        } else if (lhsInfo.isConstantDim(shape, d) &&
                   rhsInfo.isConstantDim(shape, d)) {
          // Case 4: lhs all constant, rhs all constant
          continualIntervalHint = 0;
        }
      }

      divisibility.push_back(AxisInfoEx::kInitDivisibility);
      continualSize.push_back(constancyHint);
      continualInterval.push_back(continualIntervalHint);
    }

    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }

private:
  static arith::CmpIPredicate getPredicate(arith::CmpIOp op) {
    return op.getPredicate();
  }

  static bool gtPredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sgt ||
           predicate == arith::CmpIPredicate::ugt;
  }

  static bool gePredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sge ||
           predicate == arith::CmpIPredicate::uge;
  }

  static bool ltPredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::slt ||
           predicate == arith::CmpIPredicate::ult;
  }

  static bool lePredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sle ||
           predicate == arith::CmpIPredicate::ule;
  }

  static bool compare(arith::CmpIPredicate predicate, int64_t lhs,
                      int64_t rhs) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return lhs == rhs;
    case arith::CmpIPredicate::ne:
      return lhs != rhs;
    case arith::CmpIPredicate::slt:
      return lhs < rhs;
    case arith::CmpIPredicate::sle:
      return lhs <= rhs;
    case arith::CmpIPredicate::sgt:
      return lhs > rhs;
    case arith::CmpIPredicate::sge:
      return lhs >= rhs;
    case arith::CmpIPredicate::ult:
      return static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
    case arith::CmpIPredicate::ule:
      return static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
    case arith::CmpIPredicate::ugt:
      return static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
    case arith::CmpIPredicate::uge:
      return static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
    default:
      break;
    }
    llvm_unreachable("unknown comparison predicate");
  }
};

template <typename OpTy>
class SelectOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    auto pResultInfo = operands[0]->getValue();
    auto lhsInfo = operands[1]->getValue();
    auto rhsInfo = operands[2]->getValue();
    auto rank = lhsInfo.getRank();

    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    std::optional<int64_t> constantValue;
    if (pResultInfo.getConstantValue().has_value()) {
      if (pResultInfo.getConstantValue() == 0) {
        divisibility = rhsInfo.getDivisibility();
        continualSize = rhsInfo.getContinualSize();
        continualInterval = rhsInfo.getContinualInterval();
        constantValue = rhsInfo.getConstantValue();
      } else {
        divisibility = lhsInfo.getDivisibility();
        continualSize = lhsInfo.getContinualSize();
        continualInterval = lhsInfo.getContinualInterval();
        constantValue = lhsInfo.getConstantValue();
      }
    } else {
      bool i1Cond = isa<IntegerType>(op.getOperand(0).getType());
      for (auto d = 0; d < rank; ++d) {
        if (i1Cond) {
          continualSize.push_back(
              gcd(lhsInfo.getContinualSize(d), rhsInfo.getContinualSize(d)));
        } else {
          continualSize.push_back(gcd(
              gcd(lhsInfo.getContinualSize(d), pResultInfo.getConstancy(d)),
              gcd(rhsInfo.getContinualSize(d), pResultInfo.getConstancy(d))));
        }
        continualInterval.push_back(AxisInfoEx::kDefaultContinualInterval);
        divisibility.push_back(
            std::min(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
      }
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value() &&
          lhsInfo.getConstantValue() == rhsInfo.getConstantValue())
        constantValue = lhsInfo.getConstantValue();
    }

    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }
};

template <typename OpTy>
class LogicalOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    assert((std::is_same_v<OpTy, arith::AndIOp> ||
            std::is_same_v<OpTy, arith::OrIOp> ||
            std::is_same_v<OpTy, arith::XOrIOp>) &&
           "LogicalOp not support");
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    std::optional<int64_t> constantValue;
    for (int d = 0; d < rank; ++d) {
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value()) {
        if constexpr (std::is_same_v<OpTy, arith::AndIOp>) {
          constantValue = {lhsInfo.getConstantValue().value() &
                           rhsInfo.getConstantValue().value()};
        } else if constexpr (std::is_same_v<OpTy, arith::OrIOp>) {
          constantValue = {lhsInfo.getConstantValue().value() |
                           rhsInfo.getConstantValue().value()};
        } else if constexpr (std::is_same_v<OpTy, arith::XOrIOp>) {
          constantValue = {lhsInfo.getConstantValue().value() ^
                           rhsInfo.getConstantValue().value()};
        }
      }
      if (lhsInfo.getContinualInterval(d) == 0 &&
          rhsInfo.getContinualInterval(d) == 0) {
        divisibility.push_back(
            gcd(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
        continualSize.push_back(
            gcd(lhsInfo.getContinualSize(d), rhsInfo.getContinualSize(d)));
        continualInterval.push_back(0);
        continue;
      }

      divisibility.push_back(AxisInfoEx::kInitDivisibility);
      continualSize.push_back(AxisInfoEx::kDefaultContinueSize);
      continualInterval.push_back(AxisInfoEx::kDefaultContinualInterval);
    }

    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }
};

class ShLIOpAxisInfoExVisitor final
    : public BinaryOpVisitorImpl<arith::ShLIOp> {
public:
  using BinaryOpVisitorImpl<arith::ShLIOp>::BinaryOpVisitorImpl;

private:
  int64_t getDivisibility(arith::ShLIOp op, const AxisInfoEx &lhs,
                          const AxisInfoEx &rhs, int dim) override {
    auto shift = rhs.getConstantValue().has_value()
                     ? rhs.getConstantValue().value()
                     : rhs.getDivisibility(dim);
    auto numBits = log2Int(lhs.getDivisibility(dim));
    auto maxBits = log2Int(highestPowOf2Divisor<int64_t>(0));
    // Make sure the return value doesn't exceed
    // highestPowOf2Divisor<int64>(0).
    if (shift + numBits > maxBits)
      return highestPowOf2Divisor<int64_t>(0);
    return lhs.getDivisibility(dim) << shift;
  }

  int64_t getContinualSize(arith::ShLIOp op, const AxisInfoEx &lhs,
                           const AxisInfoEx &rhs, int dim) override {
    int64_t dimContinueSize = AxisInfoEx::kDefaultContinueSize;
    if (rhs.getConstantValue().has_value())
      dimContinueSize = lhs.getContiguity(dim);
    return dimContinueSize;
  }

  int64_t getContinualInterval(arith::ShLIOp op, const AxisInfoEx &lhs,
                               const AxisInfoEx &rhs, int dim) override {
    int64_t dimContinualInterval = AxisInfoEx::kDefaultContinualInterval;
    if (rhs.getConstantValue().has_value()) {
      auto shift = rhs.getConstantValue().value();
      auto numBits = log2Int(shift);
      auto maxBits = log2Int(highestPowOf2Divisor<int64_t>(0));
      if (shift + numBits <= maxBits)
        dimContinualInterval = lhs.getContinualInterval(dim)
                               << rhs.getConstantValue().value();
    }
    return dimContinualInterval;
  }

  std::optional<int64_t> getConstantValue(arith::ShLIOp op,
                                          const AxisInfoEx &lhs,
                                          const AxisInfoEx &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() << rhs.getConstantValue().value()};
    return std::nullopt;
  }
};

template <typename OpTy>
class ShROpAxisInfoExVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getDivisibility(OpTy op, const AxisInfoEx &lhs, const AxisInfoEx &rhs,
                          int dim) override {
    if (rhs.getConstantValue().has_value())
      return std::max<int64_t>(AxisInfoEx::kInitDivisibility,
                               lhs.getDivisibility(dim) /
                                   (1 << rhs.getConstantValue().value()));
    return AxisInfoEx::kInitDivisibility;
  }

  int64_t getContinualSize(OpTy op, const AxisInfoEx &lhs,
                           const AxisInfoEx &rhs, int dim) override {
    int64_t dimContinueSize = AxisInfoEx::kDefaultContinueSize;
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 0)
      dimContinueSize = lhs.getContiguity(dim);
    return dimContinueSize;
  }

  int64_t getContinualInterval(OpTy op, const AxisInfoEx &lhs,
                               const AxisInfoEx &rhs, int dim) override {
    int64_t dimContinualInterval = AxisInfoEx::kDefaultContinualInterval;
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 0)
      dimContinualInterval = lhs.getContinualInterval(dim);
    return dimContinualInterval;
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfoEx &lhs,
                                          const AxisInfoEx &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() >> rhs.getConstantValue().value()};
    return std::nullopt;
  }
};

template <typename OpTy>
class MaxMinOpAxisInfoExVisitor final : public AxisInfoExVisitorImpl<OpTy> {
public:
  using AxisInfoExVisitorImpl<OpTy>::AxisInfoExVisitorImpl;

  AxisInfoEx getAxisInfoEx(
      OpTy op,
      ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands) override {
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    AxisInfoEx::DimVectorT divisibility, continualSize, continualInterval;
    std::optional<int64_t> constantValue;

    for (int d = 0; d < rank; ++d) {
      const AxisInfoEx *resInfo = nullptr;
      if constexpr (std::is_same_v<OpTy, arith::MaxSIOp> ||
                    std::is_same_v<OpTy, arith::MaxUIOp>) {
        if (lhsInfo.getConstantValue().has_value() &&
            rhsInfo.getConstantValue().has_value()) {
          constantValue = {std::max(lhsInfo.getConstantValue().value(),
                                    rhsInfo.getConstantValue().value())};
          if (lhsInfo.getConstantValue().value() >=
              rhsInfo.getConstantValue().value()) {
            resInfo = &lhsInfo;
          } else {
            resInfo = &rhsInfo;
          }
          divisibility.push_back(resInfo->getDivisibility(d));
          continualSize.push_back(resInfo->getContinualSize(d));
          continualInterval.push_back(resInfo->getContinualInterval(d));
          continue;
        }
      } else {
        assert((std::is_same_v<OpTy, arith::MinSIOp> ||
                std::is_same_v<OpTy, arith::MinUIOp>) &&
               "MaxMinOp not support");
        if (lhsInfo.getConstantValue().has_value() &&
            rhsInfo.getConstantValue().has_value()) {
          constantValue = {std::min(lhsInfo.getConstantValue().value(),
                                    rhsInfo.getConstantValue().value())};
          if (lhsInfo.getConstantValue().value() <=
              rhsInfo.getConstantValue().value()) {
            resInfo = &lhsInfo;
          } else {
            resInfo = &rhsInfo;
          }
          divisibility.push_back(resInfo->getDivisibility(d));
          continualSize.push_back(resInfo->getContinualSize(d));
          continualInterval.push_back(resInfo->getContinualInterval(d));
          continue;
        }
      }

      divisibility.push_back(AxisInfoEx::kInitDivisibility);
      continualSize.push_back(AxisInfoEx::kDefaultContinueSize);
      continualInterval.push_back(AxisInfoEx::kDefaultContinualInterval);
    }

    return AxisInfoEx(divisibility, continualSize, continualInterval,
                      constantValue);
  }
};

//===----------------------------------------------------------------------===//
// AxisInfoExAnalysis
//===----------------------------------------------------------------------===//

AxisInfoExAnalysis::AxisInfoExAnalysis(DataFlowSolver &solver)
    : dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<AxisInfoEx>>(
          solver) {
  // UnrealizedConversionCast:
  // This is needed by TritonGPUToLLVM, to get AxisInfoEx when the graph is
  // in the process of a PartialConversion, where UnrealizedConversionCast
  // may exist
  visitors.append<CastOpAxisInfoExVisitor<arith::ExtSIOp>,
                  CastOpAxisInfoExVisitor<arith::ExtUIOp>,
                  CastOpAxisInfoExVisitor<arith::TruncIOp>,
                  CastOpAxisInfoExVisitor<arith::IndexCastOp>,
                  CastOpAxisInfoExVisitor<triton::gpu::ConvertLayoutOp>,
                  CastOpAxisInfoExVisitor<mlir::UnrealizedConversionCastOp>,
                  CastOpAxisInfoExVisitor<triton::BitcastOp>>();

  // when scf.for supports integer induction variables
  visitors.append<MakeRangeOpAxisInfoExVisitor>();
  visitors.append<ConstantOpAxisInfoExVisitor<arith::ConstantOp>,
                  ConstantOpAxisInfoExVisitor<LLVM::ConstantOp>>();
  visitors.append<AddSubOpAxisInfoExVisitor<triton::AddPtrOp>,
                  AddSubOpAxisInfoExVisitor<arith::AddIOp>,
                  AddSubOpAxisInfoExVisitor<arith::SubIOp>,
                  AddSubOpAxisInfoExVisitor<LLVM::AddOp>>();
  visitors.append<MulIOpAxisInfoExVisitor>();
  visitors.append<DivOpAxisInfoExVisitor<arith::DivSIOp>,
                  DivOpAxisInfoExVisitor<arith::DivUIOp>>();
  visitors.append<RemOpAxisInfoExVisitor<arith::RemSIOp>,
                  RemOpAxisInfoExVisitor<arith::RemUIOp>>();
  visitors.append<BroadcastOpAxisInfoExVisitor>();
  visitors.append<TransOpAxisInfoExVisitor>();
  visitors.append<SplatOpAxisInfoExVisitor>();
  visitors.append<ExpandDimsOpAxisInfoExVisitor>();
  visitors.append<CmpOpAxisInfoExVisitor<arith::CmpIOp>>();
  visitors.append<LogicalOpAxisInfoExVisitor<arith::AndIOp>,
                  LogicalOpAxisInfoExVisitor<arith::OrIOp>,
                  LogicalOpAxisInfoExVisitor<arith::XOrIOp>>();
  visitors.append<SelectOpAxisInfoExVisitor<mlir::arith::SelectOp>>();
  visitors
      .append<ShLIOpAxisInfoExVisitor, ShROpAxisInfoExVisitor<arith::ShRUIOp>,
              ShROpAxisInfoExVisitor<arith::ShRSIOp>>();
  visitors.append<MaxMinOpAxisInfoExVisitor<arith::MaxSIOp>,
                  MaxMinOpAxisInfoExVisitor<arith::MaxUIOp>,
                  MaxMinOpAxisInfoExVisitor<arith::MinSIOp>,
                  MaxMinOpAxisInfoExVisitor<arith::MinUIOp>>();
  visitors.append<LoadOpAxisInfoExVisitor>();
}

llvm::LogicalResult AxisInfoExAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AxisInfoEx> *> operands,
    ArrayRef<dataflow::Lattice<AxisInfoEx> *> results) {
  // but why is scf.if not initialized otherwise?
  for (auto op : operands)
    if (op->getValue().getRank() == 0)
      setToEntryState((dataflow::Lattice<AxisInfoEx> *)op);
  AxisInfoEx curr = visitors.apply(op, operands);
  if (curr.getRank() == 0) {
    setAllToEntryStates(results);
    return success();
  }
  // override with hint
  auto newDivisibility = curr.getDivisibility();
  auto continualSize = curr.getContinualSize();
  auto continualInterval = curr.getContinualInterval();
  if (Attribute attr = op->getDiscardableAttr("tt.divisibility")) {
    auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
    newDivisibility = AxisInfoEx::DimVectorT(vals.begin(), vals.end());
  }
  if (Attribute attr = op->getDiscardableAttr("tt.contiguity")) {
    auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
    continualSize = AxisInfoEx::DimVectorT(vals.begin(), vals.end());
    continualInterval = AxisInfoEx::DimVectorT(vals.size(), 1);
  }
  if (Attribute attr = op->getDiscardableAttr("tt.constancy")) {
    assert(!op->getAttr("tt.contiguity") &&
           "Get tt.constancy and tt.contiguity attribute at the same op");
    auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
    continualSize = AxisInfoEx::DimVectorT(vals.begin(), vals.end());
    continualInterval = AxisInfoEx::DimVectorT(vals.size(), 0);
  }
  curr = AxisInfoEx(newDivisibility, continualSize, continualInterval,
                    curr.getConstantValue());
  // join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(curr));

  return success();
}

void AxisInfoExAnalysis::visitForOpInductionVar(
    scf::ForOp op, ArrayRef<dataflow::Lattice<AxisInfoEx> *> argLattices) {
  ProgramPoint *programPoint = getProgramPointAfter(op);
  auto lb = getLatticeElementFor(programPoint, op.getLowerBound())->getValue();
  auto step = getLatticeElementFor(programPoint, op.getStep())->getValue();
  std::optional<Value> iv = op.getSingleInductionVar();
  assert(iv && "visitForOpInduction not support");
  auto rank = 1;
  TensorType ty = dyn_cast<TensorType>(iv.value().getType());
  if (ty)
    rank = ty.getRank();

  auto divValue = AxisInfoEx::kInitDivisibility;
  auto divHint = gcd(lb.getDivisibility(0), step.getDivisibility(0));
  if (divHint != 0)
    divValue = divHint;

  AxisInfoEx::DimVectorT knownDivisibility(rank, divValue);
  AxisInfoEx::DimVectorT knowContinualSize(rank,
                                           AxisInfoEx::kDefaultContinueSize);
  AxisInfoEx::DimVectorT knowContinualInterval(
      rank, AxisInfoEx::kDefaultContinualInterval);
  auto inductionVar =
      AxisInfoEx(knownDivisibility, knowContinualSize, knowContinualInterval);
  (void)argLattices[0]->join(inductionVar);
}
} // anonymous namespace

template <class T>
void AxisInfoEx::initPessimisticStateFromFunc(int argNumber, T funcOp, int rank,
                                              DimVectorT *contiguity,
                                              DimVectorT *divisibility,
                                              DimVectorT *constancy) {
  // liast of attributes that we care about
  SmallVector<std::pair<DimVectorT *, std::string>> retVecs;
  retVecs.push_back({contiguity, "tt.contiguity"});
  retVecs.push_back({divisibility, "tt.divisibility"});
  retVecs.push_back({constancy, "tt.constancy"});
  // initialize attributes one by one
  for (auto [vec, attrName] : retVecs) {
    Attribute attr = funcOp.getArgAttr(argNumber, attrName);
    if (auto int_attr = dyn_cast_or_null<IntegerAttr>(attr))
      *vec = DimVectorT(rank, int_attr.getValue().getZExtValue());
    if (auto dense_attr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
      auto vals = dense_attr.getValues<int>();
      *vec = DimVectorT(vals.begin(), vals.end());
    }
  }
}

/*static*/ AxisInfoEx AxisInfoEx::getPessimisticValueState(Value value) {
  int rank = 1;
  if (TensorType ty = dyn_cast<TensorType>(value.getType()))
    rank = ty.getRank();
  if (triton::PointerType ty = dyn_cast<triton::PointerType>(value.getType()))
    if (TensorType elemTy = dyn_cast<TensorType>(ty.getPointeeType()))
      rank = elemTy.getRank();

  DimVectorT continualSize(rank, kDefaultContinueSize);
  DimVectorT continualInterval(rank, kDefaultContinualInterval);
  DimVectorT knownDivisibility, knownContiguity, knownConstancy;
  BlockArgument blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    Operation *op = blockArg.getOwner()->getParentOp();
    auto fun = dyn_cast<FunctionOpInterface>(op);
    if (!fun)
      fun = dyn_cast<LLVM::LLVMFuncOp>(op);

    if (fun)
      initPessimisticStateFromFunc(blockArg.getArgNumber(), fun, rank,
                                   &knownContiguity, &knownDivisibility,
                                   &knownConstancy);
  } else if (Operation *op = value.getDefiningOp()) {
    if (Attribute attr = op->getDiscardableAttr("tt.divisibility")) {
      auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
      knownDivisibility = DimVectorT(vals.begin(), vals.end());
    }
    if (Attribute attr = op->getDiscardableAttr("tt.contiguity")) {
      auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
      knownContiguity = DimVectorT(vals.begin(), vals.end());
    }
    if (Attribute attr = op->getDiscardableAttr("tt.constancy")) {
      auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
      knownConstancy = DimVectorT(vals.begin(), vals.end());
    }
  }

  if (knownDivisibility.empty()) {
    knownDivisibility = DimVectorT(rank, kInitDivisibility);
  }
  if (!knownConstancy.empty()) {
    assert(knownContiguity.empty() &&
           "Get tt.constancy and tt.contiguity attribute at the same arg");
    continualSize = knownConstancy;
    continualInterval = DimVectorT(rank, 0);
  }
  if (!knownContiguity.empty()) {
    continualSize = knownContiguity;
    continualInterval = DimVectorT(rank, 1);
  }

  return AxisInfoEx(knownDivisibility, continualSize, continualInterval);
}

/*static*/ AxisInfoEx AxisInfoEx::join(const AxisInfoEx &lhs,
                                       const AxisInfoEx &rhs) {
  auto lhsRank = lhs.getRank();
  auto rhsRank = rhs.getRank();
  // If one argument is not initialized, return the other.
  if (lhsRank == 0)
    return rhs;
  if (rhsRank == 0)
    return lhs;
  assert(lhsRank == rhsRank && "lhsRank and rhsRank are mismatch");

  DimVectorT divisibility(lhsRank, kInitDivisibility);
  DimVectorT continualSize(lhsRank, kDefaultContinueSize);
  DimVectorT continualInterval(lhsRank, kDefaultContinualInterval);
  for (auto i = 0; i < lhsRank; ++i) {
    divisibility[i] = (gcd(lhs.getDivisibility(i), rhs.getDivisibility(i)));
    continualSize[i] = (gcd(lhs.getContinualSize(i), rhs.getContinualSize(i)));
    if (lhs.continualInterval[i] == rhs.continualInterval[i])
      continualInterval[i] = lhs.continualInterval[i];
  }
  std::optional<int64_t> constantValue;
  if (lhs.getConstantValue().has_value() &&
      rhs.getConstantValue().has_value() &&
      lhs.getConstantValue() == rhs.getConstantValue())
    constantValue = lhs.getConstantValue();
  return AxisInfoEx(divisibility, continualSize, continualInterval,
                    constantValue);
}

unsigned ModuleAxisInfoExAnalysis::getPtrContiguity(Value ptr) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto layout = tensorTy.getEncoding();

  // Here order should be ordered by contiguous first, so the first element
  // should have the largest contiguous.
  auto order = triton::gpu::getOrder(tensorTy);
  unsigned align = getPtrAlignment(ptr);

  // auto uniqueContigPerThread =
  //     triton::gpu::getUniqueContigPerThread(layout, tensorTy.getShape());
  // assert(order[0] < uniqueContigPerThread.size() &&
  //        "Unexpected uniqueContigPerThread size");
  // unsigned contiguity = uniqueContigPerThread[order[0]];
  // LDBG("getPtrContiguity uniqueContigPerThread = " << contiguity);
  // contiguity = std::min(align, contiguity);

  return 0;
}

unsigned ModuleAxisInfoExAnalysis::getPtrAlignment(Value ptr) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfoEx(ptr);
  if (!axisInfo)
    return 1;
  auto layout = tensorTy.getEncoding();
  auto order = triton::gpu::getOrder(tensorTy);
  auto maxMultipleBytes = axisInfo->getDivisibility(order[0]);
  auto maxContig = axisInfo->getContiguity(order[0]);
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto maxMultiple = std::max<int64_t>(maxMultipleBytes / elemNumBytes, 1);
  unsigned alignment = std::min(maxMultiple, maxContig);
  LDBG("getPtrAlignment order[0] "
       << order[0] << " maxMultipleBytes = " << maxMultipleBytes
       << " maxContig = " << maxContig << " elemNumBits = " << elemNumBits
       << " maxMultiple = " << maxMultiple << " alignment " << alignment);
  LLVM_DEBUG({
    std::string axisStr;
    llvm::raw_string_ostream os(axisStr);
    axisInfo->print(os);
    LDBG("-- " << axisStr);
  });
  return alignment;
}

unsigned ModuleAxisInfoExAnalysis::getMaskAlignment(Value mask) {
  auto tensorTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfoEx(mask);
  if (!axisInfo)
    return 1;
  auto maskOrder = triton::gpu::getOrder(tensorTy);
  auto alignment = std::max<unsigned>(axisInfo->getConstancy(maskOrder[0]), 1);
  LDBG("getMaskAlignment maskOrder[0] " << maskOrder[0] << " alignment "
                                        << alignment);
  LLVM_DEBUG({
    std::string axisStr;
    llvm::raw_string_ostream os(axisStr);
    axisInfo->print(os);
    LDBG("-- " << axisStr);
  });
  return alignment;
}

void ModuleAxisInfoExAnalysis::initialize(FunctionOpInterface funcOp) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AxisInfoExAnalysis *analysis = solver->load<AxisInfoExAnalysis>();
  if (failed(solver->initializeAndRun(funcOp)))
    return;
  auto *axisInfoMap = getFuncData(funcOp);
  auto updateAxisInfoMap = [&](Value value) {
    auto axisInfoEx = analysis->getLatticeElement(value)->getValue();
    AxisInfoEx curAxisInfo;
    if (axisInfoMap->count(value)) {
      curAxisInfo = AxisInfoEx::join(axisInfoEx, axisInfoMap->lookup(value));
    } else {
      curAxisInfo = axisInfoEx;
    }
    (*axisInfoMap)[value] = curAxisInfo;
  };
  funcOp.walk([&](Operation *op) {
    for (auto value : op->getResults()) {
      updateAxisInfoMap(value);
    }
  });
  funcOp.walk([&](Block *block) {
    for (auto value : block->getArguments()) {
      updateAxisInfoMap(value);
    }
  });
}

void ModuleAxisInfoExAnalysis::update(CallOpInterface callOp,
                                      FunctionOpInterface callee) {
  auto caller = callOp->getParentOfType<FunctionOpInterface>();
  auto *axisInfoExMap = getFuncData(caller);
  for (auto entry : llvm::enumerate(callOp->getOperands())) {
    auto index = entry.index();
    auto value = entry.value();
    auto setAttrFn = [&](StringRef attrName, int64_t prevValue) {
      auto curValue = highestPowOf2Divisor<int64_t>(0);
      if (callee.getArgAttrOfType<IntegerAttr>(index, attrName)) {
        curValue =
            callee.getArgAttrOfType<IntegerAttr>(index, attrName).getInt();
      }
      auto attr = IntegerAttr::get(IntegerType::get(callee.getContext(), 64),
                                   gcd(prevValue, curValue));
      callee.setArgAttr(index, attrName, attr);
    };
    auto axisInfoEx = axisInfoExMap->lookup(value);
    assert(axisInfoEx.getRank() == 1 && "only scalar arguments are supported");
    setAttrFn("tt.divisibility", axisInfoEx.getDivisibility(0));
    if (axisInfoEx.getContinualInterval(0) == 0)
      setAttrFn("tt.constancy", axisInfoEx.getContinualSize(0));
    else if (axisInfoEx.getContinualInterval(0) == 1)
      setAttrFn("tt.contiguity", axisInfoEx.getContinualSize(0));
  }
}

} // namespace gcu
} // namespace triton
} // namespace mlir
