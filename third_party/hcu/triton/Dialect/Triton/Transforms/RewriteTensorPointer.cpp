#include <stack>
#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREWRITETENSORPOINTER
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

/// An additional struct to record the meta information of operations
/// with tensor pointers
struct RewritedInfo {
private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tensorShape;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;

public:
  RewritedInfo() = default;

  RewritedInfo(const RewritedInfo &other) = default;

  RewritedInfo &operator=(const RewritedInfo &other) = default;

  RewritedInfo(Value base, const SmallVector<Value> &shape,
               const SmallVector<Value> &strides,
               const SmallVector<Value> &offsets,
               const ArrayRef<int64_t> &tensorShape)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tensorShape(tensorShape) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tensorShape.size());
  }

  Value getBase() const { return base; }
  ArrayRef<Value> getShape() const { return shape; }
  ArrayRef<Value> getStrides() const { return strides; }
  ArrayRef<int64_t> getTensorShape() const { return tensorShape; }
  ArrayRef<Value> getStaticOffsets() const { return offsets; }

  unsigned int length() const { return shape.size(); }

  Value getOffset(unsigned i) { return offsets[i]; }

  SmallVector<Value> getOffsets() { return offsets; }

  void setOffset(unsigned i, Value newOffset) {
    offsets[i] = newOffset;
    cachedOffsetWithRange.clear();
  }

  void setOffsets(const SmallVector<Value> &newOffsets) {
    offsets = newOffsets;
    cachedOffsetWithRange.clear();
  }

  Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                   unsigned i) {
    if (cachedOffsetWithRange.count(i))
      return cachedOffsetWithRange[i];

    // Add range
    auto indexI32RowType =
        RankedTensorType::get({tensorShape[i]}, builder.getI32Type());
    auto indexRowType =
        RankedTensorType::get({tensorShape[i]}, builder.getI64Type());
    Value splatOffset =
        triton::SplatOp::create(builder, loc, indexRowType, offsets[i]);
    Value range = triton::MakeRangeOp::create(builder, loc, indexI32RowType, 0,
                                              tensorShape[i]);
    Value i64Range = arith::ExtSIOp::create(builder, loc, indexRowType, range);

    // Expand dimensions
    Value expandedResult =
        arith::AddIOp::create(builder, loc, splatOffset, i64Range);
    for (size_t j = 0; j < tensorShape.size(); ++j) {
      if (j == i)
        continue;
      expandedResult =
          triton::ExpandDimsOp::create(builder, loc, expandedResult, j);
    }

    return cachedOffsetWithRange[i] = expandedResult;
  }

  Value generatePtr(OpBuilder &builder, const Location &loc) {
    assert(tensorShape.size() == offsets.size() &&
           tensorShape.size() == strides.size());
    auto indexTensorType =
        RankedTensorType::get(tensorShape, builder.getI64Type());
    auto ptrType = cast<triton::PointerType>(base.getType());
    auto ptrTensorType = RankedTensorType::get(tensorShape, ptrType);

    // Generate offsets per dimension
    Value ptr = triton::SplatOp::create(builder, loc, ptrTensorType, base);
    for (unsigned i = 0; i < tensorShape.size(); ++i) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // We must splat strides into the expanded shape not a row for retaining
      // the divisibility information given by strides
      Value splatStride = triton::SplatOp::create(
          builder, loc, offsetWithRange.getType(), strides[i]);
      Value offsetWithStride =
          arith::MulIOp::create(builder, loc, offsetWithRange, splatStride);
      Value broadcasted = triton::BroadcastOp::create(
          builder, loc, indexTensorType, offsetWithStride);

      // Add to the pointer
      ptr = triton::AddPtrOp::create(builder, loc, ptrTensorType, ptr,
                                     broadcasted);
    }

    return ptr;
  }

  Value generateMask(OpBuilder &builder, const Location &loc,
                     const std::optional<ArrayRef<int32_t>> &boundaryCheck) {
    if (!boundaryCheck.has_value())
      return {};

    // Generate mask per dimension
    auto maskTensorType =
        RankedTensorType::get(tensorShape, builder.getI1Type());
    Value mask;
    for (auto i : boundaryCheck.value()) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // Compare with lower bound
      Value lowerBound = mlir::arith::ConstantIntOp::create(
          builder, loc, builder.getI64Type(), 0);
      Value splatLowerBound = triton::SplatOp::create(
          builder, loc, offsetWithRange.getType(), lowerBound);
      Value cmpLower =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                offsetWithRange, splatLowerBound);

      // Compare with upper bound
      Value splatUpperBound = triton::SplatOp::create(
          builder, loc, offsetWithRange.getType(), shape[i]);
      Value cmpUpper =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                offsetWithRange, splatUpperBound);

      // And and broadcast
      Value andResult = arith::AndIOp::create(builder, loc, cmpLower, cmpUpper);
      Value broadcasted =
          triton::BroadcastOp::create(builder, loc, maskTensorType, andResult);

      // And up all results
      if (!mask) {
        mask = broadcasted;
      } else {
        mask = arith::AndIOp::create(builder, loc, mask, broadcasted);
      }
    }

    return mask;
  }

  Value generateOther(OpBuilder &builder, const Location &loc,
                      const std::optional<triton::PaddingOption> &padding) {
    if (!padding.has_value())
      return Value();

    // Create element attribute
    auto elementType =
        cast<triton::PointerType>(base.getType()).getPointeeType();
    auto otherTensorType = RankedTensorType::get(tensorShape, elementType);

    // Set zero padding value
    TypedAttr attr = builder.getZeroAttr(elementType);

    // Float NaN padding case
    if (padding.value() == triton::PaddingOption::PAD_NAN) {
      assert(!elementType.isIntOrIndex());
      auto apNaN = llvm::APFloat::getNaN(
          cast<FloatAttr>(attr).getValue().getSemantics());
      attr = builder.getFloatAttr(elementType, apNaN);
    }

    // Create tensor
    Value constant = arith::ConstantOp::create(builder, loc, attr);
    return triton::SplatOp::create(builder, loc, otherTensorType, constant);
  }
};

} // namespace

// TODO: this pass relies on assumptions of how block pointers are created and
// on pattern matches that walks the SSA links to find the base/strides. This is
// very fragile and to solve we should expose convert Ptr of tensor to a
// structure containins all values and not only offsets.
class RewriteTensorPointerPass
    : public impl::TritonRewriteTensorPointerBase<RewriteTensorPointerPass> {
private:
  DenseMap<Value, RewritedInfo> rewritedInfo;

public:
  static bool needRewrite(Operation *op) {
    return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                       [](Value operand) {
                         return triton::isTensorPointerType(operand.getType());
                       });
  }

  static void generateNewOperands(SmallVector<Value> &oldOperands,
                                  unsigned index, ArrayRef<Value> newValues) {
    size_t size = oldOperands.size();
    assert(index < size);
    SmallVector<Value> operands = oldOperands;
    oldOperands.reserve(size - 1 + newValues.size());
    oldOperands.clear();
    if (index != 0) {
      oldOperands.append(operands.begin(), operands.begin() + index);
    }
    oldOperands.append(newValues.begin(), newValues.end());
    if (index != size - 1) {
      oldOperands.append(operands.begin() + index + 1, operands.end());
    }
  }

  Operation *rewriteMakeTensorPtrOp(OpBuilder &builder,
                                    triton::MakeTensorPtrOp op,
                                    std::stack<Operation *> &eraser) {
    // Save info for later use
    auto ptrType = cast<triton::PointerType>(op.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    // Cast I32 offsets into I64
    SmallVector<Value> i64Offsets;
    for (auto offset : op.getOffsets()) {
      auto i64Offset = arith::ExtSIOp::create(builder, op.getLoc(),
                                              builder.getI64Type(), offset);
      i64Offsets.push_back(i64Offset);
    }

    // Save information
    rewritedInfo[op.getResult()] =
        RewritedInfo(op.getBase(), op.getShape(), op.getStrides(), i64Offsets,
                     tensorType.getShape());

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteAdvanceOp(OpBuilder &builder, triton::AdvanceOp op,
                              std::stack<Operation *> &eraser) {
    // Get info from previous results
    assert(rewritedInfo.count(op.getPtr()));
    auto info = rewritedInfo[op.getPtr()];

    // Calculate new offsets
    assert(info.length() == op.getOffsets().size());
    SmallVector<Value> newOffsets;
    for (size_t i = 0; i < info.length(); ++i) {
      Value i64Offset = arith::ExtSIOp::create(
          builder, op.getLoc(), builder.getI64Type(), op.getOffsets()[i]);
      Value newOffset = arith::AddIOp::create(builder, op.getLoc(),
                                              info.getOffset(i), i64Offset);
      newOffsets.push_back(newOffset);
    }

    // Save info for later use
    info.setOffsets(newOffsets);
    rewritedInfo[op.getResult()] = info;

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

// Bound the entire full BN column tile without trusting launch metadata.
// Only one uniform CTA program-id axis1, unsigned remainder (or exact power2
// mask), then BN scale is accepted. It does not alter any pointer/descriptor.
static bool isLegacyV0BoundedPIDColumn(Value value, int64_t totalN, unsigned bn) {
  if (totalN < bn || totalN % bn)
    return false;
  auto constant = [](Value v) -> std::optional<int64_t> {
    auto op = v.getDefiningOp<arith::ConstantOp>();
    auto a = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
    if (!a || !a.getValue().isSignedIntN(64)) return std::nullopt;
    return a.getValue().getSExtValue();
  };
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
    if (!ext.getIn().getType().isInteger(32) || !value.getType().isInteger(64)) return false;
    value = ext.getIn();
  } else if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
    if (!ext.getIn().getType().isInteger(32) || !value.getType().isInteger(64)) return false;
    value = ext.getIn();
  }
  if (!value.getType().isInteger(32)) return false;
  Value tile;
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    if (constant(mul.getRhs()) != bn) return false;
    tile = mul.getLhs();
  } else if (auto shift = value.getDefiningOp<arith::ShLIOp>()) {
    unsigned log = bn == 128 ? 7 : bn == 256 ? 8 : 32;
    if (log == 32 || constant(shift.getRhs()) != log) return false;
    tile = shift.getLhs();
  } else return false;
  int64_t tiles = totalN / bn;
  Value pid;
  if (auto rem = tile.getDefiningOp<arith::RemUIOp>()) {
    if (constant(rem.getRhs()) != tiles) return false;
    pid = rem.getLhs();
  } else if (auto mask = tile.getDefiningOp<arith::AndIOp>()) {
    if (tiles < 1 || (uint64_t(tiles) & (uint64_t(tiles) - 1)) ||
        constant(mask.getRhs()) != tiles - 1) return false;
    pid = mask.getLhs();
  } else return false;
  auto op = pid.getDefiningOp<triton::GetProgramIdOp>();
  return op && op.getAxisAsInt() == 1;
}

  // Capture-only paired A/B block pointers. The old B-only selector below is
  // unchanged. This is source construction, not opaque runtime eligibility:
  // make_llir must retain its capture hard stop for the stage5 request.
  bool tryPromoteIndexedABCapture(OpBuilder &builder, triton::LoadOp load,
                                  const RewritedInfo &info) {
    auto mod = load->getParentOfType<ModuleOp>();
    auto is = [&](StringRef key, int32_t value) {
      auto a = mod->getAttrOfType<IntegerAttr>(key);
      return a && a.getType().isInteger(32) && a.getInt() == value;
    };
    if (!is("ttg.hcu.v0_indexed_a_packet", 1) ||
        !is("ttg.hcu.v0_blockptr", 1) || !is("ttg.hcu.v0_target936", 1) ||
        !is("ttg.hcu.v0_mls_local_prefetch_capture", 1) ||
        !is("ttg.hcu.v0_warps", 4) || !is("ttg.hcu.mmac_tpw_n", 2) ||
        !is("ttg.hcu.mmac_transposed", 1) ||
        !is("ttg.hcu.canonical_matrix", 0))
      return false;
    for (StringRef key : {"ttg.hcu.alt_matrix", "ttg.hcu.async_slot_war",
                          "ttg.hcu.local_prefetch", "ttg.hcu.v0_swizzled_a_vec8"})
      if (mod->hasAttr(key) && !is(key, 0)) return false;
    auto constant = [](Value v) -> std::optional<int64_t> {
      auto c = v.getDefiningOp<arith::ConstantOp>();
      auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
      if (!a || !a.getValue().isSignedIntN(64)) return std::nullopt;
      return a.getInt();
    };
    auto i32 = [](Value v) -> Value {
      if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
        if (!v.getType().isInteger(64)) return {};
        v = ext.getIn();
      }
      return v.getType().isInteger(32) ? v : Value{};
    };
    auto f = load->getParentOfType<triton::FuncOp>();
    auto loop = load->getParentOfType<scf::ForOp>();
    auto ty = dyn_cast<RankedTensorType>(load.getType());
    if (!f || !llvm::hasSingleElement(f.getBody()) || f.getNumArguments() != 3 ||
        !loop || loop->getParentOp() != f.getOperation() ||
        load->getBlock() != loop.getBody() || !load->hasOneUse() || !ty ||
        ty.getRank() != 2 || !(ty.getElementType().isF16() || ty.getElementType().isBF16()) ||
        info.length() != 2 || constant(loop.getLowerBound()) != 0 ||
        constant(loop.getStep()) != 1 || constant(loop.getUpperBound()) != 32 ||
        !loop.getInductionVar().getType().isInteger(32))
      return false;
    auto dot = dyn_cast<triton::DotOp>(*load->getUsers().begin());
    if (!dot || dot->getBlock() != loop.getBody() ||
        dot.getA().getType().getShape() != ArrayRef<int64_t>({128, 32}) ||
        dot.getB().getType().getShape() != ArrayRef<int64_t>({32, 128}) ||
        dot.getType().getShape() != ArrayRef<int64_t>({128, 128}) ||
        !dot.getType().getElementType().isF32() ||
        dot.getA().getType().getElementType() != ty.getElementType() ||
        dot.getB().getType().getElementType() != ty.getElementType())
      return false;
    unsigned opIdx = dot.getA() == load.getResult() ? 0 : 1;
    if ((opIdx == 1 && dot.getB() != load.getResult()) ||
        ty.getShape() != (opIdx == 0 ? ArrayRef<int64_t>({128, 32})
                                    : ArrayRef<int64_t>({32, 128})))
      return false;
    // A may be visited before B or after it. Count only live loads, including
    // our exact provisional replacement; never treat arbitrary MatrixLoad as
    // a second ordinary load or turn explicit per-op False into True.
    unsigned loads = 0, dots = 0;
    for (auto &op : *loop.getBody()) {
      dots += isa<triton::DotOp>(op);
      if (isa<triton::LoadOp, triton::MatrixLoadOp>(op) &&
          !op.getResult(0).use_empty()) ++loads;
    }
    if (loads != 2 || dots != 1) return false;
    for (Value operand : {dot.getA(), dot.getB()}) {
      auto *producer = operand.getDefiningOp();
      auto ld = dyn_cast_or_null<triton::LoadOp>(producer);
      auto ml = dyn_cast_or_null<triton::MatrixLoadOp>(producer);
      if (!producer || producer->getBlock() != loop.getBody() || !operand.hasOneUse() ||
          (!ld && !(ml && ml->hasAttr("ttg.hcu.v0_blockptr_candidate"))))
        return false;
      auto policy = producer->getAttrOfType<BoolAttr>("tt.load.async");
      if ((producer->hasAttr("tt.load.async") && !policy) ||
          (policy && !policy.getValue()) || (!policy && !is("ttg.hcu.v0_default_async", 1)))
        return false;
      if (ld && (ld.getMask() || ld.getOther() || ld.getPadding() ||
                 !ld.getBoundaryCheck().empty() || ld.getIsVolatile() ||
                 ld.getCache() != triton::CacheModifier::NONE ||
                 ld.getEvict() != triton::EvictionPolicy::NORMAL ||
                 (ld.getFlagtreeHintsAttr() && !ld.getFlagtreeHintsAttr().getValue().empty())))
        return false;
    }
    auto shape = info.getShape(), strides = info.getStrides(), offsets = info.getStaticOffsets();
    auto extent = constant(shape[opIdx]);
    // Fixed K and 16-bit elements: bound before multiplying/narrowing so
    // the entire contiguous source fits the signed I32 byte-offset contract.
    if (!extent || *extent < 128 || *extent % 128 ||
        *extent >= (int64_t(1) << 20) ||
        constant(shape[1 - opIdx]) != 1024 ||
        constant(strides[0]) != (opIdx == 0 ? 1024 : *extent) ||
        constant(strides[1]) != 1)
      return false;
    Value k = i32(offsets[1 - opIdx]);
    auto kmul = k ? k.getDefiningOp<arith::MulIOp>() : arith::MulIOp{};
    if (!kmul || kmul.getLhs() != loop.getInductionVar() || constant(kmul.getRhs()) != 32)
      return false;
    Value nonK = i32(offsets[opIdx]);
    if (!nonK) return false;
    if (opIdx == 0) {
      // The unmasked TL launch contract is grid.x <= M/128.
      // No prefix, negative base adjustment, or invented source padding.
      auto mul = nonK.getDefiningOp<arith::MulIOp>();
      auto pid = mul ? mul.getLhs().getDefiningOp<triton::GetProgramIdOp>()
                     : triton::GetProgramIdOp{};
      if (!mul || constant(mul.getRhs()) != 128 || !pid || pid.getAxisAsInt() != 0)
        return false;
    } else if (!isLegacyV0BoundedPIDColumn(nonK, *extent, 128) && constant(nonK) != 0) {
      return false;
    }
    auto base = dyn_cast<BlockArgument>(info.getBase());
    auto ptr = dyn_cast<triton::PointerType>(info.getBase().getType());
    if (!base || base.getOwner() != &f.getBody().front() || base.getArgNumber() != opIdx ||
        !ptr || ptr.getAddressSpace() != 1 || ptr.getPointeeType() != ty.getElementType())
      return false;
    auto align = f.getArgAttrOfType<IntegerAttr>(opIdx, "tt.divisibility");
    if (!align || !align.getValue().isSignedIntN(64) || align.getInt() < 16 || align.getInt() % 16)
      return false;
    SmallVector<Value> shapeI32{
        arith::ConstantIntOp::create(builder, load.getLoc(), *constant(shape[0]), 32),
        arith::ConstantIntOp::create(builder, load.getLoc(), *constant(shape[1]), 32)};
    SmallVector<Value> indices = opIdx == 0 ? SmallVector<Value>{nonK, k}
                                           : SmallVector<Value>{k, nonK};
    SmallVector<int32_t> tile{int32_t(ty.getShape()[0]), int32_t(ty.getShape()[1])};
    auto ml = builder.create<triton::MatrixLoadOp>(load.getLoc(), ty, info.getBase(),
        shapeI32, strides, tile, indices, ArrayRef<int32_t>{},
        load.getCache(), load.getEvict(), load.getIsVolatile());
    ml->setAttr("ttg.hcu.v0_blockptr_candidate", builder.getUnitAttr());
    mod->setAttr("ttg.hcu.v0_blockptr_loop", builder.getI32IntegerAttr(1));
    if (auto policy = load->getAttrOfType<BoolAttr>("tt.load.async"))
      ml->setAttr("tt.load.async", policy);
    load.getResult().replaceAllUsesWith(ml.getResult());
    return true;
  }

  // A closed one-shot block-pointer experiment. No pointer-array provenance
  // inference. Values narrowed below have been compared against small exact
  // constants. The final DotOperand legality is checked after acceleration;
  // that pass must lower rejected candidates back to an ordinary tt.load.
  bool tryPromoteLegacyV0BlockPtr(OpBuilder &builder, triton::LoadOp load,
                                  const RewritedInfo &info,
                                  const std::stack<Operation *> &eraser) {
    auto mod = load->getParentOfType<ModuleOp>();
    auto capture = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_mls_local_prefetch_capture");
    if (capture && capture.getType().isInteger(32) && capture.getInt() == 1) {
      if (auto hint = mod->getAttr("ttg.hcu.v0_indexed_a_packet")) {
        auto value = dyn_cast<IntegerAttr>(hint);
        if (!value || !value.getType().isInteger(32) || value.getInt() != 0)
          return tryPromoteIndexedABCapture(builder, load, info);
      }
    }
    auto intAttr = [&](StringRef name) -> int64_t {
      auto a = mod->getAttrOfType<IntegerAttr>(name);
      return a ? a.getInt() : 0;
    };
    if (intAttr("ttg.hcu.v0_blockptr") != 1 ||
        intAttr("ttg.hcu.v0_target936") != 1)
      return false;
    auto f = load->getParentOfType<triton::FuncOp>();
    auto loop = load->getParentOfType<scf::ForOp>();
    if (!f || !llvm::hasSingleElement(f.getBody()) ||
        (loop ? loop->getParentOp() != f.getOperation() || load->getBlock() != loop.getBody()
              : load->getBlock() != &f.getBody().front()))
      return false;
    // RewriteFor clones the current loop before erasing its predecessor. Only
    // ignore exact operations already owned by this pass's deferred eraser.
    llvm::SmallPtrSet<Operation *, 8> pending;
    auto pendingErase = eraser;
    while (!pendingErase.empty()) {
      pending.insert(pendingErase.top());
      pendingErase.pop();
    }
    unsigned functions = 0;
    bool simple = true;
    mod.walk([&](triton::FuncOp fn) { ++functions; });
    auto plain = [&](Operation &op) {
      auto name = op.getName().getStringRef();
      return op.getNumRegions() == 0 && op.getNumSuccessors() == 0 &&
          ((loop && isa<triton::GetProgramIdOp>(&op)) || name.starts_with("arith.") || llvm::is_contained(ArrayRef<StringRef>({
             "tt.return", "tt.make_range", "tt.expand_dims", "tt.broadcast",
             "tt.splat", "tt.addptr", "tt.make_tensor_ptr", "tt.advance",
             "tt.load", "tt.store", "tt.dot", "scf.yield"}), name));
    };
    for (auto &op : f.getBody().front()) {
      if (pending.contains(&op))
        continue;
      if (loop && &op == loop.getOperation()) {
        for (auto &bodyOp : *loop.getBody())
          simple &= plain(bodyOp);
      } else {
        simple &= plain(op);
      }
    }
    if (functions != 1 || !simple || !load->hasOneUse() ||
        load.getIsVolatile() || !load.getBoundaryCheck().empty() ||
        load.getPadding() || load.getMask() || load.getOther() ||
        (load.getFlagtreeHintsAttr() && !load.getFlagtreeHintsAttr().getValue().empty()) ||
        load.getCache() != triton::CacheModifier::NONE ||
        load.getEvict() != triton::EvictionPolicy::NORMAL)
      return false;
    auto policy = load->getAttrOfType<BoolAttr>("tt.load.async");
    if ((policy && !policy.getValue()) ||
        (!policy && intAttr("ttg.hcu.v0_default_async") != 1))
      return false;
    auto dot = dyn_cast<triton::DotOp>(*load->getUsers().begin());
    auto ty = dyn_cast<RankedTensorType>(load.getType());
    if (!ty || !dot || dot.getB() != load.getResult() || ty.getRank() != 2 ||
        !(ty.getElementType().isF16() || ty.getElementType().isBF16()) ||
        dot.getA().getType().getElementType() != ty.getElementType() ||
        !dot.getType().getElementType().isF32())
      return false;
    if (loop && (loop.getNumResults() != 1 || loop.getInitArgs().size() != 1 ||
                 loop.getBody()->getNumArguments() != 2))
      return false;
    unsigned dots = 0, globalLoads = 0, globalStores = 0;
    triton::StoreOp store;
    auto count = [&](Operation *op) {
      dots += isa<triton::DotOp>(op);
      if (auto ld = dyn_cast<triton::LoadOp>(op))
        globalLoads += !ld.getResult().use_empty();
      if (auto st = dyn_cast<triton::StoreOp>(op)) { ++globalStores; store = st; }
    };
    for (auto &op : f.getBody().front()) {
      if (pending.contains(&op))
        continue;
      if (loop && &op == loop.getOperation()) {
        for (auto &bodyOp : *loop.getBody())
          count(&bodyOp);
      } else {
        count(&op);
      }
    }
    if (dots != 1 || globalLoads != 2 || globalStores != 1 ||
        info.length() != 2 || f.getNumArguments() != 3 || !store ||
        store.getMask() ||
        !store.getBoundaryCheck().empty() ||
        !dot.getA().getDefiningOp<triton::LoadOp>())
      return false;
    Value outputAccumulator = loop ? Value(loop.getResult(0)) : Value(dot.getD());
    if (store.getValue() != outputAccumulator) {
      // Only the loop exit gains one exact truncation to the input dtype.
      // Final post-epilogue CF verification repeats this with actual layouts.
      auto trunc = store.getValue().getDefiningOp<arith::TruncFOp>();
      auto outTy = dyn_cast<RankedTensorType>(store.getValue().getType());
      auto accTy = dyn_cast<RankedTensorType>(outputAccumulator.getType());
      if (!loop || !trunc || trunc.getIn() != outputAccumulator ||
          !trunc.getOut().hasOneUse() || trunc->getBlock() != store->getBlock() ||
          store->getBlock() != &f.getBody().front() ||
          !trunc->isBeforeInBlock(store.getOperation()) || !accTy || !outTy ||
          !accTy.getElementType().isF32() || outTy.getElementType() != ty.getElementType() ||
          outTy.getShape() != accTy.getShape() || outTy.getEncoding() != accTy.getEncoding())
        return false;
      unsigned truncCount = 0;
      for (auto &op : f.getBody().front()) {
        if (pending.contains(&op)) continue;
        if (&op == loop.getOperation()) {
          for (auto &bodyOp : *loop.getBody()) truncCount += isa<arith::TruncFOp>(&bodyOp);
        } else truncCount += isa<arith::TruncFOp>(&op);
      }
      if (truncCount != 1) return false;
    }
    auto cst = [](Value v) -> std::optional<int64_t> {
      while (auto ext = v.getDefiningOp<arith::ExtSIOp>())
        v = ext.getIn();
      APInt n;
      if (!matchPattern(v, m_ConstantInt(&n)) || n.getBitWidth() > 64)
        return std::nullopt;
      return n.getSExtValue();
    };
    auto is = [&](Value v, int64_t x) { return cst(v) == x; };
    int64_t warps = intAttr("ttg.hcu.v0_warps");
    bool s25 = warps == 4 && ty.getShape() == ArrayRef<int64_t>({32, 128});
    bool s26 = warps == 8 && ty.getShape() == ArrayRef<int64_t>({16, 256});
    if (!s25 && !s26)
      return false;
    int64_t m = s25 ? 32 : 64;
    auto aTy = dot.getA().getType(), cTy = dot.getType();
    // Only the two fullM counted-loop packets; one-shot ownership is unchanged.
    // The joint-phase attribute is injected later at make_llir, so its required
    // mode and real16KiB allocation are checked by the final closed preflight.
    if (loop && aTy.getShape() == ArrayRef<int64_t>({4 * m, ty.getShape()[0]}) &&
        cTy.getShape() == ArrayRef<int64_t>({4 * m, ty.getShape()[1]}))
      m *= 4;
    if (aTy.getShape() != ArrayRef<int64_t>({m, ty.getShape()[0]}) ||
        cTy.getShape() != ArrayRef<int64_t>({m, ty.getShape()[1]}))
      return false;
    int64_t g = s25 ? 1024 : 4096;
    auto shape = info.getShape(), strides = info.getStrides();
    auto offsets = info.getStaticOffsets();
    int64_t sourceK = g, sourceN = g;
    if (loop) {
      auto kt = cst(shape[0]), nt = cst(shape[1]);
      if (!kt || !nt || *kt < ty.getShape()[0] || *nt < ty.getShape()[1] ||
          *kt > 0x7fffffffll || *nt > 0x7fffffffll ||
          *kt % ty.getShape()[0] || *nt % ty.getShape()[1] ||
          !is(strides[0], *nt) || !is(strides[1], 1) ||
          uint64_t(*kt) > 0x7fffffffull / (2ull * uint64_t(*nt)))
        return false;
      // Same rectangular source bounds as the closed static-loop preflight.
      // Check before narrowing shape to I32 or multiplying2*K*N in codegen.
      sourceK = *kt; sourceN = *nt;
    } else if (!is(shape[0], g) || !is(shape[1], g) ||
               !is(strides[0], g) || !is(strides[1], 1)) {
      return false;
    }
    Value loopRow, loopColumn;
    if (loop) {
      auto trips = cst(loop.getUpperBound());
      auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
      auto zero = loop.getInitArgs()[0].getDefiningOp<arith::ConstantOp>();
      auto dense = zero ? dyn_cast<DenseFPElementsAttr>(zero.getValue()) : DenseFPElementsAttr{};
      if (!loop.getInductionVar().getType().isInteger(32) ||
          !is(loop.getLowerBound(), 0) || !is(loop.getStep(), 1) ||
          !trips || *trips < 2 || *trips > sourceK / ty.getShape()[0] ||
          !loop.getResult(0).hasOneUse() || !yield || yield.getNumOperands() != 1 ||
          yield.getOperand(0) != dot.getD() || dot.getC() != loop.getRegionIterArg(0) ||
          dot->getBlock() != loop.getBody() || !dot.getD().hasOneUse() ||
          !dense || !dense.isSplat() || !dense.getSplatValue<llvm::APFloat>().isZero() ||
          loop.getInitArgs()[0].getType() != cTy ||
          (!is(offsets[1], 0) &&
           !isLegacyV0BoundedPIDColumn(offsets[1], sourceN, ty.getShape()[1])))
        return false;
      loopColumn = offsets[1];
      if (auto ext = loopColumn.getDefiningOp<arith::ExtSIOp>())
        loopColumn = ext.getIn();
      else if (auto ext = loopColumn.getDefiningOp<arith::ExtUIOp>())
        loopColumn = ext.getIn();
      if (!loopColumn.getType().isInteger(32)) {
        if (!is(loopColumn, 0)) return false;
        loopColumn = arith::ConstantIntOp::create(builder, load.getLoc(), 0, 32);
      }
      loopRow = offsets[0];
      // Actual inliner+pointer-rewrite source is i32 IV*BK then sign-extended
      // once to i64 by RewritedInfo. No truncation or general affine inference.
      if (auto ext = loopRow.getDefiningOp<arith::ExtSIOp>()) {
        if (!ext.getIn().getType().isInteger(32) || !loopRow.getType().isInteger(64))
          return false;
        loopRow = ext.getIn();
      }
      auto row = loopRow.getDefiningOp<arith::MulIOp>();
      if (!row || row->getBlock() != loop.getBody() ||
          row.getLhs() != loop.getInductionVar() ||
          !is(row.getRhs(), ty.getShape()[0]) || !loopRow.getType().isInteger(32))
        return false;
    } else if (!is(offsets[0], 0) || (cst(offsets[1]) != 0 && cst(offsets[1]) != 8)) {
      return false;
    }
    auto base = dyn_cast<BlockArgument>(info.getBase());
    auto ptr = dyn_cast<triton::PointerType>(info.getBase().getType());
    if (!base || base.getOwner() != &f.getBody().front() || base.getArgNumber() != 1 || !ptr ||
        ptr.getAddressSpace() != 1 || ptr.getPointeeType() != ty.getElementType())
      return false;
    auto align = f.getArgAttrOfType<IntegerAttr>(base.getArgNumber(),
                                                "tt.divisibility");
    if (!align || align.getInt() < 16 || align.getInt() % 16)
      return false;
    SmallVector<Value> shapeI32, indices;
    for (auto v : shape)
      shapeI32.push_back(arith::ConstantIntOp::create(builder, load.getLoc(),
                                                    *cst(v), 32));
    indices.push_back(loop ? loopRow : Value(arith::ConstantIntOp::create(
        builder, load.getLoc(), 0, 32)));
    indices.push_back(loop ? loopColumn : Value(arith::ConstantIntOp::create(
        builder, load.getLoc(), *cst(offsets[1]), 32)));
    SmallVector<int32_t> tile{int32_t(ty.getShape()[0]), int32_t(ty.getShape()[1])};
    auto ml = builder.create<triton::MatrixLoadOp>(
        load.getLoc(), ty, info.getBase(), shapeI32, strides,
        tile, indices, ArrayRef<int32_t>{},
        load.getCache(), load.getEvict(), load.getIsVolatile());
    ml->setAttr("ttg.hcu.v0_blockptr_candidate", builder.getUnitAttr());
    if (loop)
      mod->setAttr("ttg.hcu.v0_blockptr_loop", builder.getI32IntegerAttr(1));
    if (policy)
      ml->setAttr("tt.load.async", policy);
    load.getResult().replaceAllUsesWith(ml.getResult());
    return true;
  }

  Operation *rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                std::stack<Operation *> &eraser) {
    assert(isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op));

    // We only have to rewrite load/stores with tensor pointers
    auto ptr = op->getOperand(0);
    if (!triton::isTensorPointerType(ptr.getType()))
      return nullptr;

    // Get info from previous results
    assert(rewritedInfo.count(ptr));
    auto info = rewritedInfo[ptr];

    // Load/store with tensor pointers implicitly will check the bound while
    // accessing memory, so we should set `mask` and `other` (according to the
    // padding). Also note that load with tensor pointers do not have `mask` and
    // `other` while building IR from Python AST
    std::optional<ArrayRef<int>> boundaryCheck;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      assert(!loadOp.getMask() && !loadOp.getOther());
      boundaryCheck = loadOp.getBoundaryCheck();
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      assert(!storeOp.getMask());
      boundaryCheck = storeOp.getBoundaryCheck();
    }

    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (tryPromoteLegacyV0BlockPtr(builder, load, info, eraser)) {
        eraser.push(op);
        return nullptr;
      }
    }

    // Generate new `ptr`, `mask` and `other`
    auto newPtr = info.generatePtr(builder, op->getLoc());
    auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
    Value newOther;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op))
      newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

    // Create a new operation
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto newResult = triton::LoadOp::create(
          builder, loadOp.getLoc(), newPtr, newMask, newOther,
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile(),
          loadOp.getFlagtreeHintsAttr()); // flagtree hints
      if (auto asyncAttr = loadOp->getAttr("tt.load.async"))
        newResult->setAttr("tt.load.async", asyncAttr);
      op->getResult(0).replaceAllUsesWith(newResult);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      triton::StoreOp::create(builder, storeOp.getLoc(), newPtr,
                              storeOp.getValue(), newMask, storeOp.getCache(),
                              storeOp.getEvict());
    }

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteIfOp(OpBuilder &builder, scf::IfOp op,
                         std::stack<Operation *> &eraser) {
    auto thenYieldOp = op.thenYield();
    assert(op.getNumResults() == thenYieldOp.getNumOperands());
    SmallVector<Value> results = thenYieldOp.getOperands();

    // get new result types
    SmallVector<Type> newRetTypes;
    bool needRewrite = false;
    for (unsigned i = 0; i < results.size(); ++i) {
      if (!triton::isTensorPointerType(results[i].getType())) {
        newRetTypes.push_back(results[i].getType());
        continue;
      }
      needRewrite = true;
      auto makeTensorPtrOp = triton::getMakeTensorPtrOp(results[i]);
      assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
      const auto &info = rewritedInfo[makeTensorPtrOp.getResult()];
      for (unsigned j = 0; j < info.length(); ++j) {
        newRetTypes.push_back(builder.getI64Type());
      }
    }
    if (!needRewrite)
      return op;
    // create and clone new IfOp
    bool hasElse = !op.getElseRegion().empty();
    scf::IfOp newOp = scf::IfOp::create(builder, op.getLoc(), newRetTypes,
                                        op.getCondition(), hasElse);
    IRMapping mapping;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      mapping.map(op->getOperand(i), newOp->getOperand(i));
    }
    auto rematerialize = [&](Block *block) {
      for (Operation &opInIf : block->getOperations()) {
        builder.clone(opInIf, mapping);
      }
    };
    builder.setInsertionPointToStart(newOp.thenBlock());
    rematerialize(op.thenBlock());
    if (hasElse) {
      builder.setInsertionPointToStart(newOp.elseBlock());
      rematerialize(op.elseBlock());
    }

    // update rewritedInfo
    auto opResults = op.getResults();
    unsigned oldResIdx = 0, newResIdx = 0;
    while (oldResIdx < results.size()) {
      if (!triton::isTensorPointerType(results[oldResIdx].getType())) {
        opResults[oldResIdx].replaceAllUsesWith(newOp.getResult(newResIdx));
        oldResIdx++;
        newResIdx++;
      } else {
        auto makeTensorPtrOp = triton::getMakeTensorPtrOp(results[oldResIdx]);
        assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
        auto info = rewritedInfo[makeTensorPtrOp.getResult()];
        for (unsigned j = 0; j < info.length(); ++j) {
          info.setOffset(j, newOp->getResult(newResIdx++));
        }
        rewritedInfo[op.getResult(oldResIdx)] = info;
        oldResIdx++;
      }
    }

    eraser.push(op);
    return newOp;
  }

  Operation *rewriteForOp(OpBuilder &builder, scf::ForOp op,
                          std::stack<Operation *> &eraser) {
    // Generate new iteration operands and set rewritten information
    SmallVector<Value> oldIterOperands = llvm::to_vector(op.getInitArgs());
    SmallVector<Value> newIterOperands = llvm::to_vector(op.getInitArgs());
    for (unsigned i = 0, oldI = 0, size = op.getInitArgs().size(); i < size;
         ++i, ++oldI) {
      if (!triton::isTensorPointerType(newIterOperands[i].getType()))
        continue;

      // Expand the tensor pointer into offsets
      assert(rewritedInfo.count(newIterOperands[i]));
      auto info = rewritedInfo[newIterOperands[i]];
      generateNewOperands(newIterOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }

    // Rebuild the loop type
    auto newForOp =
        scf::ForOp::create(builder, op.getLoc(), op.getLowerBound(),
                           op.getUpperBound(), op.getStep(), newIterOperands);
    newForOp->setAttrs(op->getAttrs());

    // Create value mapping. Note that for tensor pointers, we use identity
    // mapping. It may refer to a value in the old loop, but we will rewrite it
    // later
    IRMapping mapping;
    for (unsigned i = 0, oldI = 0, sz = op.getInitArgs().size(); oldI < sz;
         ++i, ++oldI) {
      auto oldRegionIterArg = op.getRegionIterArg(oldI);
      if (triton::isTensorPointerType(oldRegionIterArg.getType())) {
        // Pass rewritten info inside
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        mapping.map(oldRegionIterArg, oldRegionIterArg);
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getRegionIterArg(i + j));
        rewritedInfo[oldRegionIterArg] = info;
        i += info.length() - 1;
      } else {
        mapping.map(oldRegionIterArg, newForOp.getRegionIterArg(i));
      }
    }
    mapping.map(op.getInductionVar(), newForOp.getInductionVar());

    // Clone body
    builder.setInsertionPointToStart(newForOp.getBody());
    for (auto &opInFor : *op.getBody()) {
      builder.clone(opInFor, mapping);
    }

    // Replace later usages
    assert(op.getNumResults() == op.getInitArgs().size());
    for (unsigned i = 0, oldI = 0; oldI < op.getNumResults(); ++i, ++oldI) {
      auto oldResult = op.getResult(oldI);
      if (triton::isTensorPointerType(oldResult.getType())) {
        // Pack new offsets into rewritten info
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getResult(i + j));
        i += info.length() - 1;
        rewritedInfo[oldResult] = info;
      } else {
        oldResult.replaceAllUsesWith(newForOp.getResult(i));
      }
    }

    // Erase later
    eraser.push(op);
    return newForOp;
  }

  Operation *rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                            std::stack<Operation *> &eraser) {
    // Replace tensor pointers with offsets
    SmallVector<Value> newOperands = op->getOperands();
    for (unsigned i = 0, size = op.getNumOperands(); i < size; ++i) {
      if (!triton::isTensorPointerType(newOperands[i].getType()))
        continue;

      assert(rewritedInfo.count(newOperands[i]));
      auto info = rewritedInfo[newOperands[i]];
      generateNewOperands(newOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }
    op->setOperands(newOperands);

    // No need to erase
    return nullptr;
  }

  Operation *rewriteOp(Operation *op, std::stack<Operation *> &eraser) {
    OpBuilder builder(op);

    // Rewrite `make_tensor_ptr` and `advance` and make a tensor of pointers
    // Rewriting functions return the next operation to visit, if there is no
    // next one, simply return `nullptr`
    if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
      return rewriteMakeTensorPtrOp(builder, makeTensorPtrOp, eraser);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
      return rewriteAdvanceOp(builder, advanceOp, eraser);
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      return rewriteLoadStoreOp(builder, op, eraser);
    } else if (isa<scf::SCFDialect, cf::ControlFlowDialect>(op->getDialect())) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        return rewriteIfOp(builder, ifOp, eraser);
      }
      if (!needRewrite(op))
        return op;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return rewriteForOp(builder, forOp, eraser);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        return rewriteYieldOp(builder, yieldOp, eraser);
      } else {
        llvm_unreachable("Currently we only support tensor pointer usages "
                         "inside a `scf::ForOp` or `scf::IfOp`, others such as "
                         "`scf::WhileOp`, `cf::BranchOp` or `cf::CondBranchOp` "
                         "are not supported yet");
      }
    }

    // Otherwise return the original one
    return op;
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nestedOp : llvm::make_early_inc_range(block)) {
          if (auto newOp = rewriteOp(&nestedOp, eraser)) {
            visitOperation(newOp, eraser);
          }
        }
      }
    }
  }

  void runOnOperation() override {
    // NOTES(Chenggang): we don't use `ConversionPatternRewriter`, because
    // MLIR does not support one-multiple value mapping. For example, if we use
    // `ConversionPatternRewriter`, we can not make a type converter, which
    // converts `ptr<tensor>` into multiple types `ptr<>, int64, int64, ...`
    // (containing the base/offsets/strides...). What we can do is to convert
    // `ptr<tensor>` into a single type `Tuple<ptr<>, int64, int64, ...>`. But
    // in this way, we also have to define `PackTuple` and `UnpackTuple`
    // operations and make a canonicalization pass to optimize, which is much
    // So here we recursively build the IR, to be specific, we have to rewrite
    // `tt.make_tensor_ptr`, `tt.advance`, `tt.load`, `tt.store`,
    // `scf.for` (tensor pointer usages may be in a loop fashion)
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser);

    // The operation could not be erased during visit, because they may have
    // later usages, so we erase after visit
    rewritedInfo.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};

} // namespace mlir::triton
