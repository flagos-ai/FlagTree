// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <limits>
#include <optional>

namespace {
// This pass runs after CANN memory planning. It never allocates or moves UB
// buffers: two existing, private index buffers become absolute address tables.
// All admission is completed before changing the IR.
struct PermuteBuffer {
  int64_t address, count, bytes;
  mlir::Attribute space;
};
struct PermuteSource {
  PermuteBuffer first, second;
  mlir::Value condition;
};

static mlir::LogicalResult
getPermuteStrides(mlir::MemRefType type,
                  llvm::SmallVectorImpl<int64_t> &strides, int64_t &offset) {
#ifdef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  return type.getStridesAndOffset(strides, offset);
#else
  return mlir::getStridesAndOffset(type, strides, offset);
#endif
}

static std::optional<PermuteBuffer> resolvePermuteBuffer(mlir::Value value) {
  using namespace mlir;
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return resolvePermuteBuffer(cast.getSource());
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getPermuteStrides(type, strides, offset)) ||
      ShapedType::isDynamic(offset))
    return std::nullopt;
  int64_t count = 1;
  for (int i = type.getRank() - 1; i >= 0; --i) {
    if (type.getDimSize(i) != 1 && strides[i] != count)
      return std::nullopt;
    if (type.getDimSize(i) <= 0 || count > 4096 / type.getDimSize(i))
      return std::nullopt;
    count *= type.getDimSize(i);
  }
  int64_t bytes = type.getElementTypeBitWidth() / 8;
  if (bytes != 2 && bytes != 4)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (op->getName().getStringRef() == "hivm.hir.pointer_cast" &&
      op->getNumOperands() == 1) {
    llvm::APInt address;
    if (!matchPattern(op->getOperand(0), m_ConstantInt(&address)))
      return std::nullopt;
    int64_t base = address.getSExtValue();
    if (base < 0 || base > std::numeric_limits<int32_t>::max() || offset < 0 ||
        offset > (std::numeric_limits<int32_t>::max() - base) / bytes)
      return std::nullopt;
    int64_t start = base + offset * bytes;
    if (start > std::numeric_limits<int32_t>::max() - count * bytes)
      return std::nullopt;
    return PermuteBuffer{start, count, bytes, type.getMemorySpace()};
  }
  if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(op)) {
    auto source = resolvePermuteBuffer(op->getOperand(0));
    if (source && source->count == count && source->bytes == bytes)
      return source;
  }
  if (auto view = dyn_cast<memref::SubViewOp>(op)) {
    auto source = resolvePermuteBuffer(view.getSource());
    auto sourceType = cast<MemRefType>(view.getSource().getType());
    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;
    if (!source ||
        failed(getPermuteStrides(sourceType, sourceStrides, sourceOffset)))
      return std::nullopt;
    int64_t start = source->address;
    for (auto [i, off] : llvm::enumerate(view.getStaticOffsets())) {
      if (ShapedType::isDynamic(off) || off < 0 ||
          off >= sourceType.getDimSize(i))
        return std::nullopt;
      start += off * sourceStrides[i] * bytes;
    }
    if (start < source->address ||
        start + count * bytes > source->address + source->count * bytes)
      return std::nullopt;
    return PermuteBuffer{start, count, bytes, source->space};
  }
  return std::nullopt;
}

static std::optional<PermuteSource>
resolvePermuteSource(mlir::Value value, bool aliasOnly = false) {
  using namespace mlir;
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return resolvePermuteSource(cast.getSource(), aliasOnly);
  if (auto *op = value.getDefiningOp()) {
    if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(op))
      return resolvePermuteSource(op->getOperand(0), aliasOnly);
    // For interference checks, a subview is conservatively covered by the
    // entire source allocation. Actual gather admission still needs an exact
    // packed descriptor; never use this over-approximation to generate indices.
    if (aliasOnly && isa<memref::SubViewOp>(op))
      return resolvePermuteSource(op->getOperand(0), true);
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      auto a = resolvePermuteBuffer(select.getTrueValue());
      auto b = resolvePermuteBuffer(select.getFalseValue());
      if (a && b && a->count == b->count && a->bytes == b->bytes &&
          a->space == b->space)
        return PermuteSource{*a, *b, select.getCondition()};
      return std::nullopt;
    }
  }
  if (auto buffer = resolvePermuteBuffer(value))
    return PermuteSource{*buffer, *buffer, {}};
  return std::nullopt;
}

static bool permuteOverlap(PermuteBuffer a, PermuteBuffer b) {
  return a.space == b.space && a.address < b.address + b.count * b.bytes &&
         b.address < a.address + a.count * a.bytes;
}

struct HoistStructuredPermutationAddressesPass
    : mlir::PassWrapper<HoistStructuredPermutationAddressesPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      HoistStructuredPermutationAddressesPass)
  llvm::StringRef getArgument() const final {
    return "hoist-structured-permutation-addresses";
  }
  llvm::StringRef getDescription() const final {
    return "Reuse private vector permutation address tables across planned UB "
           "loops";
  }
  void runOnOperation() final {
    using namespace mlir;
    SmallVector<func::CallOp> candidates;
    getOperation().walk([&](func::CallOp call) {
      if (call.getCallee().starts_with("custom_flagtree_stride_gather_"))
        candidates.push_back(call);
    });
    for (auto call : candidates)
      optimize(call);
  }
  void optimize(mlir::func::CallOp call) {
    using namespace mlir;
    StringRef suffix = call.getCallee().drop_front(
        StringRef("custom_flagtree_stride_gather_").size());
    if (suffix != "half" && suffix != "bf16" && suffix != "float" &&
        suffix != "int16" && suffix != "int32")
      return;
    auto loop = call->getParentOfType<scf::ForOp>();
    auto function = call->getParentOfType<func::FuncOp>();
    if (!loop || !function || function.getBody().empty() ||
        loop->getBlock() != &function.getBody().front() ||
        call->getBlock() != loop.getBody() || call.getNumOperands() != 4 ||
        call.getNumResults() != 0)
      return;
    llvm::APInt lower, step, upper;
    if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !lower.isZero() ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        step.getSExtValue() <= 0 ||
        step.getSExtValue() > std::numeric_limits<int32_t>::max() / 2)
      return;
    int64_t threshold = 2 * step.getSExtValue();
    bool dynamicTrips =
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper));
    if (!dynamicTrips && upper.getSExtValue() <= threshold)
      return;
    auto source = resolvePermuteSource(call.getOperand(0));
    auto indices = resolvePermuteBuffer(call.getOperand(1));
    auto destination = resolvePermuteBuffer(call.getOperand(2));
    auto scratch = resolvePermuteBuffer(call.getOperand(3));
    auto ub =
        hivm::AddressSpaceAttr::get(&getContext(), hivm::AddressSpace::UB);
    if (!source || !indices || !destination || !scratch ||
        indices->space != ub || indices->bytes != 4 || scratch->bytes != 4 ||
        source->first.bytes != destination->bytes ||
        call.getOperand(1).getType() != call.getOperand(3).getType() ||
        indices->count != source->first.count ||
        indices->count != destination->count ||
        indices->count != scratch->count ||
        indices->space != source->first.space ||
        indices->space != destination->space ||
        indices->space != scratch->space ||
        (indices->address | scratch->address | destination->address) % 32 ||
        permuteOverlap(*indices, *scratch))
      return;
    for (auto buffer : {source->first, source->second, *destination})
      if (permuteOverlap(*indices, buffer) || permuteOverlap(*scratch, buffer))
        return;
    // Source/destination overlap is deliberately not admitted by this path.
    if (permuteOverlap(source->first, *destination) ||
        permuteOverlap(source->second, *destination))
      return;
    // Any additional use of either table after preparation invalidates reuse.
    // Examine physical aliases too, not merely SSA uses of the original views.
    bool safe = true;
    function.walk([&](Operation *op) {
      if (!safe || op == function.getOperation() || op == call.getOperation() ||
          isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::SubViewOp>(op))
        return;
      Operation *top = op;
      while (top->getBlock() != loop->getBlock() && top->getParentOp())
        top = top->getParentOp();
      if (top->getBlock() != loop->getBlock()) {
        safe = false;
        return;
      }
      bool beforeLoop =
          top != loop.getOperation() && top->isBeforeInBlock(loop);
      // Initialization on PIPE_V is drained by prepare_addresses' leading
      // vector barrier. Other preheader accesses must pass the same physical
      // alias checks as loop accesses: an earlier DMA read can still be in
      // flight when the address table is overwritten. Unknown calls can hide
      // these asynchronous accesses behind scalar addresses.
      if (beforeLoop)
        if (auto init = dyn_cast<func::CallOp>(op)) {
          StringRef name = init.getCallee();
          if (name == "arange_1d_int32_t" ||
              name == "broadcast_scalar_int32_t_to_1d" ||
              name == "vand_1d_int32_t" || name == "vmuls_vs_1d_int32_t" ||
              name == "vshrs_vs_1d_int32_t" || name == "vadd_1d_int32_t")
            return;
        }
      // Unknown calls/operations can hide UB accesses behind scalar addresses.
      // Only compiler-owned DMA ABIs and explicit synchronization are trusted;
      // all their memory accesses remain covered by the descriptor checks
      // below.
      StringRef opName = op->getName().getStringRef();
      bool modeled = isMemoryEffectFree(op) ||
                     isa<hivm::PointerCastOp, scf::ForOp, scf::IfOp>(op) ||
                     opName == "hivm.hir.set_flag" ||
                     opName == "hivm.hir.wait_flag" ||
                     opName == "hivm.hir.pipe_barrier" ||
                     opName == "hivm.hir.set_mask_norm";
      if (auto other = dyn_cast<func::CallOp>(op))
        modeled = other.getCallee().starts_with("load_gm_to_ubuf_1d_") ||
                  other.getCallee().starts_with("store_ubuf_to_gm_1d_");
      if (!modeled) {
        safe = false;
        return;
      }
      for (Value value : op->getOperands()) {
        auto type = dyn_cast<MemRefType>(value.getType());
        if (!type || type.getMemorySpace() != indices->space)
          continue;
        auto buffer = resolvePermuteBuffer(value);
        if (buffer) {
          if (permuteOverlap(*buffer, *indices) ||
              permuteOverlap(*buffer, *scratch))
            safe = false;
        } else if (auto alternatives = resolvePermuteSource(value, true)) {
          for (auto b : {alternatives->first, alternatives->second})
            if (permuteOverlap(b, *indices) || permuteOverlap(b, *scratch))
              safe = false;
        } else {
          safe = false;
        }
      }
    });
    if (!safe)
      return;

    OpBuilder builder(loop);
    auto loc = call.getLoc();
    auto module = getOperation();
    auto tableType = cast<MemRefType>(call.getOperand(1).getType());
    StringRef prepare = "custom_flagtree_prepare_addresses";
    std::string name = "custom_flagtree_addressed_gather_" + suffix.str();
    auto compatible = [&](StringRef symbol, TypeRange inputs) {
      auto *existing = module.lookupSymbol(symbol);
      if (!existing)
        return true;
      auto fn = dyn_cast<func::FuncOp>(existing);
      return fn && fn.isPrivate() && fn.getBody().empty() &&
             fn->hasAttr("flagtree.structured_address_table") &&
             fn.getFunctionType() == builder.getFunctionType(inputs, {});
    };
    if (!compatible(prepare, {tableType, tableType, builder.getI32Type(),
                              builder.getI32Type(), builder.getI32Type()}) ||
        !compatible(name, {tableType, call.getOperand(2).getType()}))
      return;
    auto staticTable =
        MemRefType::get({indices->count}, builder.getI32Type(),
                        MemRefLayoutAttrInterface{}, indices->space);
    auto pointer = [&](int64_t address) -> Value {
      Value constant = builder.create<arith::ConstantIntOp>(loc, address, 64);
      Value storage =
          builder.create<hivm::PointerCastOp>(loc, staticTable, constant);
      return builder.create<memref::CastOp>(loc, tableType, storage);
    };
    Value tableA = pointer(indices->address),
          tableB = pointer(scratch->address);
    auto declare = [&](StringRef name, TypeRange types) {
      if (module.lookupSymbol(name))
        return;
      OpBuilder decl(module.getBodyRegion());
      auto fn = decl.create<func::FuncOp>(loc, name,
                                          builder.getFunctionType(types, {}));
      fn.setPrivate();
      fn->setAttr("hacc.always_inline", builder.getUnitAttr());
      fn->setAttr("flagtree.structured_address_table", builder.getUnitAttr());
      fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
      // Inherit the core-kind attribute from the existing private declaration.
      if (auto old = module.lookupSymbol<func::FuncOp>(call.getCallee()))
        if (auto kind = old->getAttr("hivm.func_core_type"))
          fn->setAttr("hivm.func_core_type", kind);
    };
    declare(prepare, {tableType, tableType, builder.getI32Type(),
                      builder.getI32Type(), builder.getI32Type()});
    SmallVector<Value> args{tableA, tableB};
    for (int64_t v :
         {source->first.bytes, source->first.address, source->second.address})
      args.push_back(builder.create<arith::ConstantIntOp>(loc, v, 32));
    Value reuse;
    if (dynamicTrips) {
      Value limit = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(loop.getStep().getType(), threshold));
      reuse = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                            loop.getUpperBound(), limit);
      auto guard = builder.create<scf::IfOp>(loc, reuse, false);
      auto inside = guard.getThenBodyBuilder();
      inside.create<func::CallOp>(loc, prepare, TypeRange{}, args);
    } else {
      builder.create<func::CallOp>(loc, prepare, TypeRange{}, args);
    }
    builder.setInsertionPoint(call);
    Value chosen = tableA;
    if (source->condition)
      chosen = builder.create<arith::SelectOp>(loc, source->condition, tableA,
                                               tableB);
    declare(name, {tableType, call.getOperand(2).getType()});
    if (reuse) {
      auto guard = builder.create<scf::IfOp>(loc, reuse, true);
      auto fast = guard.getThenBodyBuilder();
      fast.create<func::CallOp>(loc, name, TypeRange{},
                                ValueRange{chosen, call.getOperand(2)});
      auto slow = guard.getElseBodyBuilder();
      slow.clone(*call);
    } else {
      builder.create<func::CallOp>(loc, name, TypeRange{},
                                   ValueRange{chosen, call.getOperand(2)});
    }
    call.erase();
  }
};

// A symbolic element offset: only a common SSA base plus a constant is needed
// to prove that two contiguous GM stores are disjoint.
struct StoreOffset {
  mlir::Value base;
  int64_t constant;
};
static std::optional<StoreOffset> addStoreOffsets(StoreOffset a,
                                                  StoreOffset b) {
  if (a.base && b.base)
    return std::nullopt;
  if ((b.constant > 0 &&
       a.constant > std::numeric_limits<int64_t>::max() - b.constant) ||
      (b.constant < 0 &&
       a.constant < std::numeric_limits<int64_t>::min() - b.constant))
    return std::nullopt;
  return StoreOffset{a.base ? a.base : b.base, a.constant + b.constant};
}
static std::optional<StoreOffset> resolveStoreOffset(mlir::Value value);
static std::optional<StoreOffset>
resolveStoreExpr(mlir::AffineExpr expr, mlir::affine::AffineApplyOp apply) {
  using namespace mlir;
  if (auto c = dyn_cast<AffineConstantExpr>(expr))
    return StoreOffset{{}, c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(expr))
    return resolveStoreOffset(apply.getMapOperands()[d.getPosition()]);
  if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
    return resolveStoreOffset(
        apply.getMapOperands()[apply.getAffineMap().getNumDims() +
                               sym.getPosition()]);
  if (auto binary = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (binary.getKind() == AffineExprKind::Add) {
      auto a = resolveStoreExpr(binary.getLHS(), apply);
      auto b = resolveStoreExpr(binary.getRHS(), apply);
      if (a && b)
        return addStoreOffsets(*a, *b);
    }
  }
  return std::nullopt;
}
static std::optional<StoreOffset> resolveStoreOffset(mlir::Value value) {
  using namespace mlir;
  llvm::APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return StoreOffset{{}, constant.getSExtValue()};
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>())
    return resolveStoreExpr(apply.getAffineMap().getResult(0), apply);
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto a = resolveStoreOffset(add.getLhs()),
         b = resolveStoreOffset(add.getRhs());
    if (a && b)
      return addStoreOffsets(*a, *b);
    return std::nullopt;
  }
  return StoreOffset{value, 0};
}
struct PlannedStore {
  mlir::Value allocation;
  StoreOffset offset;
  int64_t count;
  mlir::Type element;
};
static std::optional<PlannedStore> getPlannedStore(mlir::Operation *op) {
  using namespace mlir;
  auto call = dyn_cast<func::CallOp>(op);
  if (!call || !call.getCallee().starts_with("store_ubuf_to_gm_1d_") ||
      call.getNumOperands() != 3 || call.getNumResults())
    return std::nullopt;
  llvm::APInt mode;
  if (!matchPattern(call.getOperand(2), m_ConstantInt(&mode)) || !mode.isZero())
    return std::nullopt;
  auto source = dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto ub =
      hivm::AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::UB);
  auto gm =
      hivm::AddressSpaceAttr::get(op->getContext(), hivm::AddressSpace::GM);
  if (!source || source.getMemorySpace() != ub)
    return std::nullopt;
  Value src = call.getOperand(0);
  while (auto cast = src.getDefiningOp<memref::CastOp>())
    src = cast.getSource();
  auto packedSource = cast<MemRefType>(src.getType());
  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset;
  auto buffers = resolvePermuteSource(src, true);
  int64_t bytes = source.getElementTypeBitWidth() / 8;
  // Admit only the library's direct DMA path: no scalar tail, unaligned UB
  // handling, strided packing, or hidden temporary storage inside the call.
  if (!packedSource.hasStaticShape() || packedSource.getRank() != 1 ||
      failed(getPermuteStrides(packedSource, sourceStrides, sourceOffset)) ||
      sourceStrides[0] != 1 || ShapedType::isDynamic(sourceOffset) || !bytes ||
      sourceOffset < 0 || (sourceOffset % 32) * bytes % 32 ||
      (packedSource.getNumElements() * bytes) % 32 || !buffers ||
      buffers->first.address % 32 || buffers->second.address % 32)
    return std::nullopt;
  Value dest = call.getOperand(1);
  while (auto cast = dest.getDefiningOp<memref::CastOp>())
    dest = cast.getSource();
  auto type = dyn_cast<MemRefType>(dest.getType());
  auto view = dest.getDefiningOp<memref::ReinterpretCastOp>();
  SmallVector<int64_t> strides;
  int64_t ignored;
  if (!type || type.getMemorySpace() != gm || type.getRank() != 1 ||
      type.getElementType() != source.getElementType() ||
      !type.hasStaticShape() || type.getNumElements() <= 0 ||
      type.getNumElements() != packedSource.getNumElements() ||
      failed(getPermuteStrides(type, strides, ignored)) || strides[0] != 1 ||
      !view || view.getMixedOffsets().size() != 1)
    return std::nullopt;
  std::optional<StoreOffset> offset;
  OpFoldResult off = view.getMixedOffsets()[0];
  if (auto attr = dyn_cast<Attribute>(off))
    offset = StoreOffset{{}, cast<IntegerAttr>(attr).getInt()};
  else
    offset = resolveStoreOffset(cast<Value>(off));
  if (!offset || offset->constant > std::numeric_limits<int64_t>::max() -
                                        type.getNumElements())
    return std::nullopt;
  return PlannedStore{view.getSource(), *offset, type.getNumElements(),
                      type.getElementType()};
}
static bool disjointPlannedStores(const PlannedStore &a,
                                  const PlannedStore &b) {
  if (a.allocation != b.allocation || a.element != b.element ||
      a.offset.base != b.offset.base)
    return false;
  return a.offset.constant + a.count <= b.offset.constant ||
         b.offset.constant + b.count <= a.offset.constant;
}
struct CoalesceStructuredStoreSyncPass
    : mlir::PassWrapper<CoalesceStructuredStoreSyncPass,
                        mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoalesceStructuredStoreSyncPass)
  llvm::StringRef getArgument() const final {
    return "coalesce-structured-store-sync";
  }
  llvm::StringRef getDescription() const final {
    return "Coalesce MTE3-only waits between proven disjoint UB-to-GM stores";
  }
  void runOnOperation() final {
    using namespace mlir;
    getOperation().walk([&](func::FuncOp function) {
      bool privatePermutation = false;
      function.walk([&](func::CallOp call) {
        privatePermutation |=
            call.getCallee().starts_with("custom_flagtree_stride_gather_") ||
            call.getCallee().starts_with("custom_flagtree_addressed_gather_");
      });
      if (!privatePermutation)
        return;
      function.walk([&](scf::ForOp loop) {
        SmallVector<PlannedStore> group;
        SmallVector<hivm::PipeBarrierOp> pending;
        for (Operation &op : llvm::make_early_inc_range(*loop.getBody())) {
          if (auto store = getPlannedStore(&op)) {
            bool disjoint = llvm::all_of(group, [&](auto &old) {
              return disjointPlannedStores(old, *store);
            });
            if (disjoint) {
              for (auto barrier : pending)
                barrier.erase();
            } else {
              group.clear();
            }
            pending.clear();
            group.push_back(*store);
            continue;
          }
          if (auto barrier = dyn_cast<hivm::PipeBarrierOp>(&op)) {
            if (barrier.getPipe().getPipe() == hivm::PIPE::PIPE_MTE3) {
              if (!group.empty())
                pending.push_back(barrier);
              continue;
            }
          }
          // A read/write, another pipeline synchronization, control flow, or an
          // unknown operation ends the group. Keep the final wait and every
          // cross-pipeline dependency exactly as CANN scheduled them.
          if (op.getNumRegions() ||
              (!isMemoryEffectFree(&op) && !isa<hivm::PointerCastOp>(&op))) {
            group.clear();
            pending.clear();
          }
        }
      });
    });
  }
};

} // namespace

namespace mlir::triton::ascend {
void registerStructuredPermutationPasses() {
  mlir::PassRegistration<HoistStructuredPermutationAddressesPass>();
  mlir::PassRegistration<CoalesceStructuredStoreSyncPass>();
}
} // namespace mlir::triton::ascend
