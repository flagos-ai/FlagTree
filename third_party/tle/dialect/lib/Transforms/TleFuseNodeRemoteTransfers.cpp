/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEFUSENODEREMOTETRANSFERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;

constexpr StringLiteral kNodeBufferBindingsAttr = "tle.node_buffer_bindings";

struct PointerExpr {
  Value root;
  tt::MakeRangeOp range;
  SmallVector<Value> scalarOffsets;
  tle::RemotePointersOp remote;
  BlockArgument localArg;
};

static bool collectOffset(Value value, PointerExpr &expr, unsigned depth = 0) {
  if (!value || depth > 32)
    return false;

  if (auto splat = value.getDefiningOp<tt::SplatOp>())
    return collectOffset(splat.getSrc(), expr, depth + 1);

  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return collectOffset(add.getLhs(), expr, depth + 1) &&
           collectOffset(add.getRhs(), expr, depth + 1);

  if (auto range = value.getDefiningOp<tt::MakeRangeOp>()) {
    if (expr.range || range.getStartAttr().getInt() != 0 ||
        range.getEndAttr().getInt() <= 0)
      return false;
    expr.range = range;
    return true;
  }

  // Tensor integer casts are compiler-introduced width normalization around
  // make_range/addi. Scalar casts are retained as a scalar offset term so
  // their runtime semantics are preserved.
  if (isa<RankedTensorType>(value.getType())) {
    if (auto cast = value.getDefiningOp<arith::ExtSIOp>())
      return collectOffset(cast.getIn(), expr, depth + 1);
    if (auto cast = value.getDefiningOp<arith::ExtUIOp>())
      return collectOffset(cast.getIn(), expr, depth + 1);
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
      return collectOffset(cast.getIn(), expr, depth + 1);
    return false;
  }

  if (!value.getType().isIntOrIndex())
    return false;
  expr.scalarOffsets.push_back(value);
  return true;
}

static bool decomposePointer(Value value, PointerExpr &expr) {
  Value current = value;
  unsigned depth = 0;

  while (current && depth++ <= 32) {
    if (auto addPtr = current.getDefiningOp<tt::AddPtrOp>()) {
      if (!collectOffset(addPtr.getOffset(), expr))
        return false;
      current = addPtr.getPtr();
      continue;
    }

    if (auto splat = current.getDefiningOp<tt::SplatOp>()) {
      current = splat.getSrc();
      continue;
    }
    break;
  }
  if (depth > 33)
    return false;

  expr.root = current;
  if (auto marker = current.getDefiningOp<tle::RemotePointersOp>()) {
    if (marker.getSpace() != "node" || !marker.getResult())
      return false;
    expr.remote = marker;
  } else if (auto arg = dyn_cast<BlockArgument>(current)) {
    auto ptrTy = dyn_cast<tt::PointerType>(arg.getType());
    if (!ptrTy || ptrTy.getAddressSpace() != 1)
      return false;
    expr.localArg = arg;
  } else {
    return false;
  }
  return true;
}

static tle::RemotePointersOp findNodeRemoteMarker(Value value,
                                                  unsigned depth = 0) {
  if (!value || depth > 32)
    return {};

  if (auto marker = value.getDefiningOp<tle::RemotePointersOp>())
    return marker.getSpace() == "node" && marker.getResult() ? marker : nullptr;
  if (auto addPtr = value.getDefiningOp<tt::AddPtrOp>())
    return findNodeRemoteMarker(addPtr.getPtr(), depth + 1);

  if (auto splat = value.getDefiningOp<tt::SplatOp>())
    return findNodeRemoteMarker(splat.getSrc(), depth + 1);
  return {};
}

static tle::RemotePointersOp findNodeRemoteMarkerInValue(Value value,
                                                         unsigned depth = 0) {
  if (!value || depth > 32)
    return {};

  if (auto load = value.getDefiningOp<tt::LoadOp>())
    return findNodeRemoteMarker(load.getPtr());

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};

  for (Value operand : def->getOperands())
    if (auto marker = findNodeRemoteMarkerInValue(operand, depth + 1))
      return marker;

  return {};
}

static Value castOffsetToI64(OpBuilder &builder, Location loc, Value value) {
  Type type = value.getType();
  Type i64Ty = builder.getI64Type();
  if (type == i64Ty)
    return value;
  if (isa<IndexType>(type))
    return builder.create<arith::IndexCastOp>(loc, i64Ty, value);
  auto intTy = dyn_cast<IntegerType>(type);
  if (!intTy)
    return {};

  if (intTy.getWidth() < 64)
    return builder.create<arith::ExtSIOp>(loc, i64Ty, value);
  if (intTy.getWidth() > 64)
    return builder.create<arith::TruncIOp>(loc, i64Ty, value);
  return value;
}

static Value materializeOffset(OpBuilder &builder, Location loc,
                               ArrayRef<Value> terms) {
  Value result = builder.create<arith::ConstantIntOp>(loc, 0, 64);

  for (Value term : terms) {
    Value termI64 = castOffsetToI64(builder, loc, term);
    if (!termI64)
      return {};
    result = builder.create<arith::AddIOp>(loc, result, termI64);
  }
  return result;
}

struct PrefixMaskInfo {
  Value validN;
  std::optional<int64_t> constantValidN;
  bool isUnsigned;
};

static std::optional<int64_t> getConstantI64(Value value);

static Value stripTensorIntegerCasts(Value value) {
  while (isa<RankedTensorType>(value.getType())) {
    if (auto cast = value.getDefiningOp<arith::ExtSIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::ExtUIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
      value = cast.getIn();
      continue;
    }
    break;
  }
  return value;
}

static std::optional<PrefixMaskInfo>
matchPrefixMask(Value mask, tt::MakeRangeOp expectedRange) {
  auto cmp = mask.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;

  bool isUnsigned = false;
  if (cmp.getPredicate() == arith::CmpIPredicate::ult)
    isUnsigned = true;
  else if (cmp.getPredicate() != arith::CmpIPredicate::slt)
    return std::nullopt;

  if (stripTensorIntegerCasts(cmp.getLhs()) != expectedRange.getResult())
    return std::nullopt;

  Value validN;
  std::optional<int64_t> constantValidN;
  if (auto limitSplat = cmp.getRhs().getDefiningOp<tt::SplatOp>()) {
    validN = limitSplat.getSrc();
    if (!validN.getType().isIntOrIndex())
      return std::nullopt;
    if (auto intTy = dyn_cast<IntegerType>(validN.getType());
        intTy && (intTy.getWidth() == 1 || intTy.getWidth() > 64))
      return std::nullopt;
  } else if (auto constant = cmp.getRhs().getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
    auto tensorTy = dyn_cast<RankedTensorType>(cmp.getRhs().getType());
    auto elementTy =
        tensorTy ? dyn_cast<IntegerType>(tensorTy.getElementType()) : nullptr;
    if (!dense || !dense.isSplat() || !elementTy || elementTy.getWidth() == 1 ||
        elementTy.getWidth() > 64)
      return std::nullopt;

    APInt value = dense.getSplatValue<APInt>();
    if (isUnsigned && value.getActiveBits() > 63)
      constantValidN = std::numeric_limits<int64_t>::max();
    else
      constantValidN = isUnsigned ? static_cast<int64_t>(value.getZExtValue())
                                  : value.getSExtValue();
  } else {
    return std::nullopt;
  }

  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskTy || maskTy.getRank() != 1 ||
      maskTy.getDimSize(0) != expectedRange.getEndAttr().getInt())
    return std::nullopt;
  return PrefixMaskInfo{validN, constantValidN, isUnsigned};
}

static Value materializeDynamicPrefixLength(OpBuilder &builder, Location loc,
                                            PrefixMaskInfo mask,
                                            int64_t extent) {
  Value limitI64;
  Type type = mask.validN.getType();
  Type i64Ty = builder.getI64Type();
  if (isa<IndexType>(type)) {
    limitI64 = builder.create<arith::IndexCastOp>(loc, i64Ty, mask.validN);
  } else {
    auto intTy = dyn_cast<IntegerType>(type);
    if (!intTy || intTy.getWidth() > 64)
      return {};
    if (intTy.getWidth() < 64) {
      limitI64 =
          mask.isUnsigned
              ? Value(builder.create<arith::ExtUIOp>(loc, i64Ty, mask.validN))
              : Value(builder.create<arith::ExtSIOp>(loc, i64Ty, mask.validN));
    } else {
      limitI64 = mask.validN;
    }
  }

  Value upper = builder.create<arith::ConstantIntOp>(loc, extent, 64);
  arith::CmpIPredicate predicate =
      mask.isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
  Value withinExtent =
      builder.create<arith::CmpIOp>(loc, predicate, limitI64, upper);
  builder.create<tt::AssertOp>(
      loc, withinExtent,
      builder.getStringAttr("valid_n must not exceed tl.arange extent"));
  return limitI64;
}

static std::optional<std::string>
getUnsupportedLoadOptionsReason(tt::LoadOp load) {
  if (load.getOther() || !load.getBoundaryCheck().empty() ||
      load.getPaddingAttr())
    return "the FlagCX node transfer cannot represent tl.load other, "
           "padding_option, or boundary_check values; omit these options, use "
           "the same contiguous prefix mask (offsets < valid_n) on tl.load "
           "and tl.store, and perform default-value filling separately";

  if (load.getCache() != tt::CacheModifier::NONE ||
      load.getEvict() != tt::EvictionPolicy::NORMAL || load.getIsVolatile())
    return "the FlagCX node transfer has no GPU-local cache, eviction, or "
           "volatile semantics; omit cache_modifier and eviction_policy and "
           "use volatile=False";

  if (!load.getFlagtreeHints().empty())
    return "the FlagCX node transfer does not support load hints; omit the "
           "hints and use a plain tl.load";
  return std::nullopt;
}

static std::optional<std::string>
getUnsupportedStoreOptionsReason(tt::StoreOp store) {
  if (!store.getBoundaryCheck().empty())
    return "the FlagCX node transfer cannot represent tl.store "
           "boundary_check; omit it and use the same contiguous prefix mask "
           "(offsets < valid_n) on tl.load and tl.store";

  if (store.getCache() != tt::CacheModifier::NONE ||
      store.getEvict() != tt::EvictionPolicy::NORMAL)
    return "the FlagCX node transfer has no GPU-local cache or eviction "
           "semantics; omit cache_modifier and eviction_policy";
  return std::nullopt;
}

static bool hasOnlyEffectFreeOpsBetween(tt::LoadOp load, tt::StoreOp store) {
  for (Operation *op = load->getNextNode(); op != store.getOperation();
       op = op->getNextNode()) {
    if (!op || !isMemoryEffectFree(op))
      return false;
  }
  return true;
}

static std::optional<int64_t> getElementBytes(Type type) {
  unsigned bitWidth = 0;
  if (auto intTy = dyn_cast<IntegerType>(type))
    bitWidth = intTy.getWidth();
  else if (auto floatTy = dyn_cast<FloatType>(type))
    bitWidth = floatTy.getWidth();
  else
    return std::nullopt;
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

static bool haveMatchingScalarAccessTypes(tt::LoadOp load, tt::StoreOp store,
                                          int64_t &elemBytes) {
  Type valueTy = load.getType();
  if (isa<RankedTensorType>(valueTy) || store.getValue().getType() != valueTy)
    return false;

  auto loadPtrTy = dyn_cast<tt::PointerType>(load.getPtr().getType());
  auto storePtrTy = dyn_cast<tt::PointerType>(store.getPtr().getType());
  if (!loadPtrTy || !storePtrTy || loadPtrTy.getPointeeType() != valueTy ||
      storePtrTy.getPointeeType() != valueTy)
    return false;

  std::optional<int64_t> bytes = getElementBytes(valueTy);
  if (!bytes)
    return false;
  elemBytes = *bytes;
  return true;
}

static bool haveMatchingAccessTypes(tt::LoadOp load, tt::StoreOp store,
                                    int64_t extent, int64_t &elemBytes) {

  auto valueTy = dyn_cast<RankedTensorType>(load.getType());
  auto loadPtrTy = dyn_cast<RankedTensorType>(load.getPtr().getType());
  auto storePtrTy = dyn_cast<RankedTensorType>(store.getPtr().getType());
  if (!valueTy || !loadPtrTy || !storePtrTy || valueTy.getRank() != 1 ||
      loadPtrTy.getRank() != 1 || storePtrTy.getRank() != 1 ||
      valueTy.getDimSize(0) != extent ||
      loadPtrTy.getShape() != valueTy.getShape() ||
      storePtrTy.getShape() != valueTy.getShape())
    return false;

  auto loadElemPtr = dyn_cast<tt::PointerType>(loadPtrTy.getElementType());
  auto storeElemPtr = dyn_cast<tt::PointerType>(storePtrTy.getElementType());
  if (!loadElemPtr || !storeElemPtr ||
      loadElemPtr.getPointeeType() != valueTy.getElementType() ||
      storeElemPtr.getPointeeType() != valueTy.getElementType())
    return false;

  std::optional<int64_t> bytes = getElementBytes(valueTy.getElementType());
  if (!bytes)
    return false;
  elemBytes = *bytes;
  return true;
}

static bool isFusionAddressOp(Operation *op) {
  if (auto remote = dyn_cast<tle::RemotePointersOp>(op))
    return remote.getSpace() == "node" && remote.getResult();
  return isa<tt::AddPtrOp, tt::SplatOp, tt::MakeRangeOp, arith::CmpIOp,
             arith::AddIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
             arith::IndexCastOp, arith::ConstantOp>(op);
}

static void eraseDeadAddressOps(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (isFusionAddressOp(op) && op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : llvm::reverse(dead)) {
      if (op->use_empty()) {
        op->erase();
        changed = true;
      }
    }
  }
}

static std::optional<int64_t> getConstantI64(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

// Encode role:local runtime argument ordinal:registered-memory handle.
// Role `s` validates a PUT source and `d` validates a GET destination.
static void
setBindingAttr(ModuleOp module,
               SmallVectorImpl<std::tuple<char, unsigned, int64_t>> &bindings) {
  if (bindings.empty()) {
    module->removeAttr(kNodeBufferBindingsAttr);
    return;
  }

  llvm::sort(bindings);
  bindings.erase(std::unique(bindings.begin(), bindings.end()), bindings.end());
  std::string encoded;
  llvm::raw_string_ostream os(encoded);
  llvm::interleaveComma(bindings, os, [&](const auto &binding) {
    os << std::get<0>(binding) << ":" << std::get<1>(binding) << ":"
       << std::get<2>(binding);
  });

  module->setAttr(kNodeBufferBindingsAttr,
                  StringAttr::get(module.getContext(), os.str()));
}

struct TritonTleFuseNodeRemoteTransfers
    : public impl::TritonTleFuseNodeRemoteTransfersBase<
          TritonTleFuseNodeRemoteTransfers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<std::tuple<char, unsigned, int64_t>> localBindings;
    llvm::DenseMap<Operation *, std::string> fusionFailures;
    auto recordFailure = [&](tle::RemotePointersOp marker, StringRef reason) {
      if (marker && !fusionFailures.count(marker.getOperation()))
        fusionFailures.try_emplace(marker.getOperation(), reason.str());
    };

    SmallVector<tt::StoreOp> stores;
    module.walk([&](tt::StoreOp store) { stores.push_back(store); });
    for (tt::StoreOp store : stores) {
      if (!store)
        continue;

      auto dstMarker = findNodeRemoteMarker(store.getPtr());
      auto valueMarker = findNodeRemoteMarkerInValue(store.getValue());

      auto candidateMarker = dstMarker ? dstMarker : valueMarker;
      if (!candidateMarker)
        continue;

      if (auto reason = getUnsupportedStoreOptionsReason(store)) {
        recordFailure(candidateMarker, *reason);
        continue;
      }

      auto load = store.getValue().getDefiningOp<tt::LoadOp>();
      if (!load) {
        recordFailure(candidateMarker,
                      "the loaded value must be stored directly without "
                      "intermediate computation; use a direct tl.load followed "
                      "by its paired tl.store and perform computation in a "
                      "separate path");
        continue;
      }

      if (!load->hasOneUse()) {
        recordFailure(candidateMarker,
                      "the source load must be used only by its paired store; "
                      "if another store needs the data, issue a separate "
                      "tl.load/tl.store outside the node-transfer pair");
        continue;
      }

      if (auto reason = getUnsupportedLoadOptionsReason(load)) {
        recordFailure(candidateMarker, *reason);
        continue;
      }

      if (load->getBlock() != store->getBlock() ||
          !load->isBeforeInBlock(store)) {
        recordFailure(candidateMarker,
                      "the load and store must be ordered in the same basic "
                      "block; keep the paired tl.load and tl.store in the same "
                      "control-flow path with the load before the store");
        continue;
      }

      if (!hasOnlyEffectFreeOpsBetween(load, store)) {
        recordFailure(candidateMarker,
                      "the node transfer is emitted at the paired store, so "
                      "memory accesses, atomics, barriers, and communication "
                      "operations are forbidden between tl.load and tl.store; "
                      "move such operations before the load or after the "
                      "store");
        continue;
      }

      PointerExpr srcExpr;
      PointerExpr dstExpr;

      if (!decomposePointer(load.getPtr(), srcExpr) ||
          !decomposePointer(store.getPtr(), dstExpr)) {
        recordFailure(candidateMarker,
                      "source and destination pointers must use a scalar "
                      "offset plus a contiguous tt.make_range(0, N), and the "
                      "local pointer must be an entry-function argument; pass "
                      "the registered buffer directly to the kernel, define "
                      "offsets = tl.arange(0, N) once, and access both sides "
                      "as base + integer_scalar_offset + offsets");
        continue;
      }

      bool isPut = !srcExpr.remote && dstExpr.remote;
      bool isGet = srcExpr.remote && !dstExpr.remote;
      if (!isPut && !isGet) {
        recordFailure(candidateMarker,
                      "exactly one side of the copy must be node-remote; use "
                      "tl.load(local) followed by tl.store(tle.remote(...), "
                      "value) for PUT, or tl.load(tle.remote(...)) followed by "
                      "tl.store(local, value) for GET; local-to-local and "
                      "remote-to-remote copies are not supported here");
        continue;
      }

      bool srcHasRange = static_cast<bool>(srcExpr.range);
      bool dstHasRange = static_cast<bool>(dstExpr.range);
      bool isScalarCopy = !srcHasRange && !dstHasRange;

      if (isScalarCopy && (load.getMask() || store.getMask())) {
        recordFailure(candidateMarker,
                      "scalar node remote copies do not support masks; omit "
                      "the mask for a single-element transfer, or use shared "
                      "offsets = tl.arange(0, N) and mask = offsets < valid_n "
                      "for a masked tensor transfer");
        continue;
      }

      if (load.getMask() != store.getMask()) {
        recordFailure(candidateMarker,
                      "the FlagCX transfer interface uses one contiguous "
                      "element count, so load and store must use the same "
                      "prefix mask; define mask = offsets < valid_n once and "
                      "pass that mask to both tl.load and tl.store");
        continue;
      }

      if (srcHasRange != dstHasRange ||
          (srcHasRange &&
           srcExpr.range.getOperation() != dstExpr.range.getOperation())) {
        recordFailure(candidateMarker,
                      "the FlagCX transfer interface uses one element count "
                      "for both source and destination, so they must use the "
                      "same tt.make_range(0, N); define offsets = tl.arange(0, "
                      "N) once and reuse offsets in both pointer "
                      "expressions");
        continue;
      }

      if (isScalarCopy && (isa<RankedTensorType>(load.getType()) ||
                           isa<RankedTensorType>(load.getPtr().getType()) ||
                           isa<RankedTensorType>(store.getPtr().getType()))) {
        recordFailure(candidateMarker,
                      "the load/store operates on tensor values, but no "
                      "contiguous transfer range was found; use the same "
                      "offsets = tl.arange(0, N) for both source and "
                      "destination, or use scalar pointers for a "
                      "single-element transfer");
        continue;
      }

      std::optional<PrefixMaskInfo> prefixMask;
      if (load.getMask() && !isScalarCopy) {
        prefixMask = matchPrefixMask(load.getMask(), srcExpr.range);
        if (!prefixMask) {
          recordFailure(candidateMarker,
                        "the FlagCX transfer interface only supports a "
                        "contiguous element count; mask must be a "
                        "one-dimensional contiguous prefix equivalent to "
                        "tt.make_range(0, N) < valid_n (sparse masks such as "
                        "(range % 2) == 0 are unsupported)");
          continue;
        }
      }

      int64_t extent = isScalarCopy ? 1 : srcExpr.range.getEndAttr().getInt();
      int64_t elemBytes = 0;

      bool matchingTypes =
          isScalarCopy
              ? haveMatchingScalarAccessTypes(load, store, elemBytes)
              : haveMatchingAccessTypes(load, store, extent, elemBytes);
      if (!matchingTypes) {
        recordFailure(candidateMarker,
                      "source and destination access types and shapes must "
                      "match the contiguous transfer range; use the same "
                      "element dtype and the same N on both sides, and perform "
                      "any dtype conversion or reshaping outside the direct "
                      "tl.load/tl.store transfer pair");
        continue;
      }

      PointerExpr &localExpr = isPut ? srcExpr : dstExpr;
      PointerExpr &remoteExpr = isPut ? dstExpr : srcExpr;
      auto func = store->getParentOfType<tt::FuncOp>();
      if (!func || !localExpr.localArg ||
          localExpr.localArg.getOwner() != &func.getBody().front()) {
        recordFailure(candidateMarker,
                      "the local buffer root must be an entry-function "
                      "global pointer argument; pass the buffer registered by "
                      "ctx = tle.create_dist_tensor(buffer) directly as a "
                      "kernel argument instead of reconstructing its root "
                      "pointer inside the kernel");
        continue;
      }

      OpBuilder builder(store);
      Location loc = store.getLoc();

      Value srcOffset = materializeOffset(builder, loc, srcExpr.scalarOffsets);
      Value dstOffset = materializeOffset(builder, loc, dstExpr.scalarOffsets);

      if (!srcOffset || !dstOffset) {
        recordFailure(candidateMarker,
                      "source and destination offsets must be integer values; "
                      "use scalar integer src_offset/dst_offset and form each "
                      "pointer as base + scalar_offset, optionally followed by "
                      "the shared tl.arange(0, N)");
        continue;
      }

      std::optional<int64_t> constantValidN;
      if (prefixMask) {
        constantValidN = prefixMask->constantValidN;
        if (!constantValidN)
          constantValidN = getConstantI64(prefixMask->validN);
        if (constantValidN && *constantValidN <= 0) {
          recordFailure(candidateMarker,
                        "nelems must be greater than zero; valid_n for a "
                        "contiguous prefix transfer must be > 0");
          continue;
        }
        if (constantValidN && *constantValidN > extent) {
          recordFailure(candidateMarker,
                        "valid_n must not exceed the tl.arange extent; use "
                        "1 <= valid_n <= N for tl.arange(0, N)");
          continue;
        }
      }

      Value nelems;
      if (!prefixMask)
        nelems = builder.create<arith::ConstantIntOp>(loc, extent, 64);
      else if (constantValidN)
        nelems = builder.create<arith::ConstantIntOp>(loc, *constantValidN, 64);
      else
        nelems =
            materializeDynamicPrefixLength(builder, loc, *prefixMask, extent);
      if (!nelems) {
        recordFailure(candidateMarker,
                      "valid_n must be an integer up to i64; pass a scalar "
                      "integer valid_n and use the same mask = offsets < "
                      "valid_n on tl.load and tl.store");
        continue;
      }
      auto marker = remoteExpr.remote;

      if (isPut) {
        builder.create<tle::NodePutOp>(
            loc, marker.getSrc(), marker.getSrc(), marker.getComm(),
            marker.getShardId(), srcOffset, dstOffset, nelems,
            marker.getNetIdx(), builder.getI64IntegerAttr(elemBytes),
            marker.getCoopKindAttr());
      } else {
        builder.create<tle::NodeGetOp>(
            loc, marker.getSrc(), marker.getSrc(), marker.getComm(),
            marker.getShardId(), srcOffset, dstOffset, nelems,
            marker.getNetIdx(), builder.getI64IntegerAttr(elemBytes),
            marker.getCoopKindAttr());
      }

      unsigned localArgNumber = localExpr.localArg.getArgNumber();
      if (std::optional<int64_t> memHandle = getConstantI64(marker.getSrc()))
        localBindings.emplace_back(isPut ? 's' : 'd', localArgNumber,
                                   *memHandle);
      store.erase();
      if (load->use_empty())
        load.erase();
    }

    eraseDeadAddressOps(module);

    bool hasUnfusedMarker = false;
    module.walk([&](tle::RemotePointersOp op) {
      if (op.getSpace() == "node" && op.getResult()) {

        auto failure = fusionFailures.find(op.getOperation());
        if (failure != fusionFailures.end())
          op.emitOpError() << "could not fuse node remote pointer: "
                           << failure->second;
        else
          op.emitOpError()
              << "could not fuse node remote pointer: expected a direct, "
                 "single-use scalar or contiguous load/store copy; use "
                 "tl.load on one side and immediately tl.store that value "
                 "to the other side, make exactly one side tle.remote(...), "
                 "use unmasked scalar pointers to transfer exactly one "
                 "element, and for tensor copies reuse "
                 "offsets = tl.arange(0, N) with no mask or an optional "
                 "shared mask = offsets < valid_n on both sides";
        hasUnfusedMarker = true;
      }
    });
    if (hasUnfusedMarker) {
      signalPassFailure();
      return;
    }

    setBindingAttr(module, localBindings);
  }
};

} // namespace
} // namespace mlir::triton::tle
