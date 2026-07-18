// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tt = mlir::triton;

#define GEN_PASS_DEF_TRITONMUSAGPUMARKINPLACELOADS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

inline constexpr llvm::StringLiteral kInplaceLoadAttr =
    "musa.inplace_load_candidate";

static bool
areEquivalentValues(Value lhs, Value rhs,
                    llvm::DenseMap<std::pair<Value, Value>, bool> &cache);

static bool
areEquivalentOps(Operation *lhs, Operation *rhs,
                 llvm::DenseMap<std::pair<Value, Value>, bool> &cache) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;

  auto checkEquivalent = [&](Value left, Value right) -> LogicalResult {
    return success(areEquivalentValues(left, right, cache));
  };

  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, checkEquivalent, /*markEquivalent=*/nullptr,
      OperationEquivalence::Flags::IgnoreLocations);
}

static bool
areEquivalentValues(Value lhs, Value rhs,
                    llvm::DenseMap<std::pair<Value, Value>, bool> &cache) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs || lhs.getType() != rhs.getType())
    return false;

  auto key = std::make_pair(lhs, rhs);
  auto reverseKey = std::make_pair(rhs, lhs);
  if (auto it = cache.find(key); it != cache.end())
    return it->second;
  if (auto it = cache.find(reverseKey); it != cache.end())
    return it->second;

  cache[key] = false;
  cache[reverseKey] = false;

  if (auto lhsArg = dyn_cast<BlockArgument>(lhs)) {
    auto rhsArg = dyn_cast<BlockArgument>(rhs);
    bool equivalent = rhsArg && lhsArg.getOwner() == rhsArg.getOwner() &&
                      lhsArg.getArgNumber() == rhsArg.getArgNumber();
    cache[key] = equivalent;
    cache[reverseKey] = equivalent;
    return equivalent;
  }

  bool equivalent =
      areEquivalentOps(lhs.getDefiningOp(), rhs.getDefiningOp(), cache);
  cache[key] = equivalent;
  cache[reverseKey] = equivalent;
  return equivalent;
}

static bool hasSameAddressStoreInFunc(tt::LoadOp loadOp,
                                      ArrayRef<tt::StoreOp> storeOps) {
  llvm::DenseMap<std::pair<Value, Value>, bool> cache;
  Value loadPtr = loadOp.getPtr();
  for (tt::StoreOp storeOp : storeOps) {
    if (areEquivalentValues(loadPtr, storeOp.getPtr(), cache))
      return true;
  }
  return false;
}

struct TritonMUSAGPUMarkInplaceLoadsPass
    : impl::TritonMUSAGPUMarkInplaceLoadsBase<
          TritonMUSAGPUMarkInplaceLoadsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    MLIRContext *ctx = &getContext();
    mod.walk([&](tt::FuncOp funcOp) {
      llvm::SmallVector<tt::StoreOp> storeOps;
      funcOp.walk([&](tt::StoreOp storeOp) { storeOps.push_back(storeOp); });
      if (storeOps.empty())
        return;

      funcOp.walk([&](tt::LoadOp loadOp) {
        if (hasSameAddressStoreInFunc(loadOp, storeOps))
          loadOp->setAttr(kInplaceLoadAttr, UnitAttr::get(ctx));
      });
    });
  }
};

} // namespace
} // namespace mlir
