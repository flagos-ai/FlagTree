// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <limits>
#include <optional>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEOPTIMIZELOCALPOINTERSTORES
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static Value stripValueWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = cvt.getSrc();
      continue;
    }
    if (auto splat = current.getDefiningOp<tt::SplatOp>()) {
      current = splat.getSrc();
      continue;
    }
    if (auto broadcast = current.getDefiningOp<tt::BroadcastOp>()) {
      current = broadcast.getSrc();
      continue;
    }
    if (auto expand = current.getDefiningOp<tt::ExpandDimsOp>()) {
      current = expand.getSrc();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

struct IntRange {
  int64_t min;
  int64_t max;
};

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedSub(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
#if defined(__SIZEOF_INT128__)
  __int128 product = static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
  if (product > std::numeric_limits<int64_t>::max() ||
      product < std::numeric_limits<int64_t>::min())
    return false;
  result = static_cast<int64_t>(product);
  return true;
#else
  if (lhs == 0 || rhs == 0) {
    result = 0;
    return true;
  }
  if (lhs == -1 && rhs == std::numeric_limits<int64_t>::min())
    return false;
  if (rhs == -1 && lhs == std::numeric_limits<int64_t>::min())
    return false;
  int64_t absLhs = lhs < 0 ? -lhs : lhs;
  int64_t absRhs = rhs < 0 ? -rhs : rhs;
  if (absLhs > std::numeric_limits<int64_t>::max() / absRhs)
    return false;
  result = lhs * rhs;
  return true;
#endif
}

static std::optional<int64_t> getConstantIntLike(Value value) {
  value = stripValueWrappers(value);
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getValue().getSExtValue();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    }
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static std::optional<IntRange> getIntRange(Value value, unsigned depth = 0) {
  if (depth > 16)
    return std::nullopt;

  Value current = stripValueWrappers(value);
  if (std::optional<int64_t> cst = getConstantIntLike(current))
    return IntRange{*cst, *cst};

  if (auto range = current.getDefiningOp<tt::MakeRangeOp>()) {
    int64_t start = range.getStartAttr().getInt();
    int64_t end = range.getEndAttr().getInt();
    if (end <= start)
      return std::nullopt;
    return IntRange{start, end - 1};
  }

  if (current.getDefiningOp<tt::GetProgramIdOp>())
    return IntRange{0, std::numeric_limits<int64_t>::max()};

  if (auto add = current.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getIntRange(add.getLhs(), depth + 1);
    auto rhs = getIntRange(add.getRhs(), depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t min, max;
    if (!checkedAdd(lhs->min, rhs->min, min) ||
        !checkedAdd(lhs->max, rhs->max, max))
      return std::nullopt;
    return IntRange{min, max};
  }

  if (auto sub = current.getDefiningOp<arith::SubIOp>()) {
    auto lhs = getIntRange(sub.getLhs(), depth + 1);
    auto rhs = getIntRange(sub.getRhs(), depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t min, max;
    if (!checkedSub(lhs->min, rhs->max, min) ||
        !checkedSub(lhs->max, rhs->min, max))
      return std::nullopt;
    return IntRange{min, max};
  }

  if (auto mul = current.getDefiningOp<arith::MulIOp>()) {
    auto lhs = getIntRange(mul.getLhs(), depth + 1);
    auto rhs = getIntRange(mul.getRhs(), depth + 1);
    if (!lhs || !rhs)
      return std::nullopt;
    int64_t products[4];
    if (!checkedMul(lhs->min, rhs->min, products[0]) ||
        !checkedMul(lhs->min, rhs->max, products[1]) ||
        !checkedMul(lhs->max, rhs->min, products[2]) ||
        !checkedMul(lhs->max, rhs->max, products[3]))
      return std::nullopt;
    return IntRange{
        *std::min_element(std::begin(products), std::end(products)),
        *std::max_element(std::begin(products), std::end(products))};
  }

  if (auto rem = current.getDefiningOp<arith::RemSIOp>()) {
    auto lhs = getIntRange(rem.getLhs(), depth + 1);
    auto rhs = getConstantIntLike(rem.getRhs());
    if (!lhs || !rhs || *rhs <= 0 || lhs->min < 0)
      return std::nullopt;
    return IntRange{0, *rhs - 1};
  }

  if (auto rem = current.getDefiningOp<arith::RemUIOp>()) {
    auto rhs = getConstantIntLike(rem.getRhs());
    if (!rhs || *rhs <= 0)
      return std::nullopt;
    return IntRange{0, *rhs - 1};
  }

  return std::nullopt;
}

static std::optional<bool> getConstantBoolLike(Value value) {
  value = stripValueWrappers(value);
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto boolAttr = dyn_cast<BoolAttr>(cst.getValue()))
      return boolAttr.getValue();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (dense.isSplat())
        return !dense.getSplatValue<APInt>().isZero();
    }
  }
  return std::nullopt;
}

static bool isComparisonKnownTrue(arith::CmpIOp cmp) {
  auto lhs = getIntRange(cmp.getLhs());
  auto rhs = getIntRange(cmp.getRhs());
  if (!lhs || !rhs)
    return false;

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return lhs->min == lhs->max && rhs->min == rhs->max && lhs->min == rhs->min;
  case arith::CmpIPredicate::ne:
    return lhs->max < rhs->min || rhs->max < lhs->min;
  case arith::CmpIPredicate::slt:
    return lhs->max < rhs->min;
  case arith::CmpIPredicate::sle:
    return lhs->max <= rhs->min;
  case arith::CmpIPredicate::sgt:
    return lhs->min > rhs->max;
  case arith::CmpIPredicate::sge:
    return lhs->min >= rhs->max;
  case arith::CmpIPredicate::ult:
    return lhs->min >= 0 && rhs->min >= 0 && lhs->max < rhs->min;
  case arith::CmpIPredicate::ule:
    return lhs->min >= 0 && rhs->min >= 0 && lhs->max <= rhs->min;
  case arith::CmpIPredicate::ugt:
    return lhs->min >= 0 && rhs->min >= 0 && lhs->min > rhs->max;
  case arith::CmpIPredicate::uge:
    return lhs->min >= 0 && rhs->min >= 0 && lhs->min >= rhs->max;
  }
  return false;
}

static bool isKnownAllTrueMask(Value mask, unsigned depth = 0) {
  if (depth > 16)
    return false;

  if (std::optional<bool> cst = getConstantBoolLike(mask))
    return *cst;

  Value current = stripValueWrappers(mask);
  if (auto andOp = current.getDefiningOp<arith::AndIOp>())
    return isKnownAllTrueMask(andOp.getLhs(), depth + 1) &&
           isKnownAllTrueMask(andOp.getRhs(), depth + 1);

  if (auto cmp = current.getDefiningOp<arith::CmpIOp>())
    return isComparisonKnownTrue(cmp);

  return false;
}

class OptimizeLocalPointerStoresPass
    : public impl::TritonTleOptimizeLocalPointerStoresBase<
          OptimizeLocalPointerStoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<triton::StoreOp> stores;
    module.walk([&](triton::StoreOp store) { stores.push_back(store); });

    for (triton::StoreOp store : stores) {
      if (!store)
        continue;

      Value ptr = stripConvertLayouts(store.getPtr());
      auto localPointers = ptr.getDefiningOp<tle::LocalPointersOp>();
      if (!localPointers)
        continue;

      auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
      auto memDescTy =
          dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
      if (!valueTy || !memDescTy)
        continue;

      if (!store.getBoundaryCheck().empty())
        continue;
      if (valueTy.getShape() != memDescTy.getShape())
        continue;
      if (valueTy.getElementType() != memDescTy.getElementType())
        continue;

      OpBuilder builder(store);
      Value valueToStore = store.getValue();

      if (Value mask = store.getMask(); mask && !isKnownAllTrueMask(mask)) {
        auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
        if (!maskTy || maskTy.getShape() != valueTy.getShape())
          continue;
        if (maskTy.getEncoding() != valueTy.getEncoding()) {
          auto targetMaskTy =
              RankedTensorType::get(maskTy.getShape(), maskTy.getElementType(),
                                    valueTy.getEncoding());
          mask = builder
                     .create<ttg::ConvertLayoutOp>(store.getLoc(), targetMaskTy,
                                                   mask)
                     .getResult();
        }
        Value oldValue = builder.create<ttg::LocalLoadOp>(
            store.getLoc(), valueTy, localPointers.getSrc());
        valueToStore = builder
                           .create<arith::SelectOp>(store.getLoc(), mask,
                                                    valueToStore, oldValue)
                           .getResult();
      }

      builder.create<ttg::LocalStoreOp>(store.getLoc(), valueToStore,
                                        localPointers.getSrc());
      store.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
