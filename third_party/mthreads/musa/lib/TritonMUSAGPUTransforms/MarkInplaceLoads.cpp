#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <tuple>
#include <utility>

namespace mlir {
namespace tt = mlir::triton;

#define GEN_PASS_DEF_TRITONMUSAGPUMARKINPLACELOADS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

inline constexpr llvm::StringLiteral kInplaceLoadAttr =
    "musa.inplace_load_candidate";
inline constexpr llvm::StringLiteral kInplaceAliasAttr =
    "musa.inplace_alias_pairs";

using ArgAliasPairs = llvm::SmallVector<std::pair<unsigned, unsigned>>;

static ArgAliasPairs parseArgAliasPairs(llvm::StringRef aliasPairs) {
  ArgAliasPairs parsed;
  llvm::SmallVector<llvm::StringRef> pairs;
  aliasPairs.split(pairs, ",", -1, false);
  for (llvm::StringRef pair : pairs) {
    llvm::StringRef lhsStr;
    llvm::StringRef rhsStr;
    std::tie(lhsStr, rhsStr) = pair.split(":");
    unsigned lhs = 0;
    unsigned rhs = 0;
    if (lhsStr.getAsInteger(10, lhs) || rhsStr.getAsInteger(10, rhs))
      continue;
    parsed.push_back({lhs, rhs});
  }
  return parsed;
}

static bool areAliasedArgs(BlockArgument lhs, BlockArgument rhs,
                           ArrayRef<std::pair<unsigned, unsigned>> aliasPairs) {
  if (lhs.getOwner() != rhs.getOwner())
    return false;

  unsigned lhsNo = lhs.getArgNumber();
  unsigned rhsNo = rhs.getArgNumber();
  if (lhsNo == rhsNo)
    return true;

  for (auto [aliasLhs, aliasRhs] : aliasPairs) {
    if ((lhsNo == aliasLhs && rhsNo == aliasRhs) ||
        (lhsNo == aliasRhs && rhsNo == aliasLhs))
      return true;
  }
  return false;
}

static bool
areEquivalentValues(Value lhs, Value rhs,
                    ArrayRef<std::pair<unsigned, unsigned>> aliasPairs,
                    llvm::DenseMap<std::pair<Value, Value>, bool> &cache);

static bool
areEquivalentOps(Operation *lhs, Operation *rhs,
                 ArrayRef<std::pair<unsigned, unsigned>> aliasPairs,
                 llvm::DenseMap<std::pair<Value, Value>, bool> &cache) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;

  auto checkEquivalent = [&](Value left, Value right) -> LogicalResult {
    return success(areEquivalentValues(left, right, aliasPairs, cache));
  };

  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, checkEquivalent, /*markEquivalent=*/nullptr,
      OperationEquivalence::Flags::IgnoreLocations);
}

static bool
areEquivalentValues(Value lhs, Value rhs,
                    ArrayRef<std::pair<unsigned, unsigned>> aliasPairs,
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
    bool equivalent = rhsArg && areAliasedArgs(lhsArg, rhsArg, aliasPairs);
    cache[key] = equivalent;
    cache[reverseKey] = equivalent;
    return equivalent;
  }

  bool equivalent = areEquivalentOps(lhs.getDefiningOp(), rhs.getDefiningOp(),
                                     aliasPairs, cache);
  cache[key] = equivalent;
  cache[reverseKey] = equivalent;
  return equivalent;
}

static bool
hasSameAddressStoreInFunc(tt::LoadOp loadOp, ArrayRef<tt::StoreOp> storeOps,
                          ArrayRef<std::pair<unsigned, unsigned>> aliasPairs) {
  llvm::DenseMap<std::pair<Value, Value>, bool> cache;
  Value loadPtr = loadOp.getPtr();
  for (tt::StoreOp storeOp : storeOps) {
    if (areEquivalentValues(loadPtr, storeOp.getPtr(), aliasPairs, cache))
      return true;
  }
  return false;
}

struct TritonMUSAGPUMarkInplaceLoadsPass
    : impl::TritonMUSAGPUMarkInplaceLoadsBase<
          TritonMUSAGPUMarkInplaceLoadsPass> {
  using Base = impl::TritonMUSAGPUMarkInplaceLoadsBase<
      TritonMUSAGPUMarkInplaceLoadsPass>;

  TritonMUSAGPUMarkInplaceLoadsPass() = default;
  TritonMUSAGPUMarkInplaceLoadsPass(
      TritonMUSAGPUMarkInplaceLoadsOptions options)
      : Base(std::move(options)) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::StringRef aliasSpec = inplaceAliasPairs;
    if (aliasSpec.empty()) {
      if (auto attr = mod->getAttrOfType<StringAttr>(kInplaceAliasAttr))
        aliasSpec = attr.getValue();
    }
    ArgAliasPairs aliasPairs = parseArgAliasPairs(aliasSpec);

    MLIRContext *ctx = &getContext();
    mod.walk([&](tt::FuncOp funcOp) {
      llvm::SmallVector<tt::StoreOp> storeOps;
      funcOp.walk([&](tt::StoreOp storeOp) { storeOps.push_back(storeOp); });
      if (storeOps.empty())
        return;

      funcOp.walk([&](tt::LoadOp loadOp) {
        if (hasSameAddressStoreInFunc(loadOp, storeOps, aliasPairs))
          loadOp->setAttr(kInplaceLoadAttr, UnitAttr::get(ctx));
      });
    });
  }
};

} // namespace
} // namespace mlir
