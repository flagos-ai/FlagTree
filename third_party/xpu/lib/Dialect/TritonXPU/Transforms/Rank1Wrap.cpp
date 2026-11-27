// Preserve P800 rank-1 modulo loads across the prebuilt TritonShared lowering.
// That lowering drops the wrapped dimension and emits a contiguous DMA. Capture
// the supported address contract before lowering, then split the DMA at wrap
// boundaries after SDNN fusion and before LoopGrid. Do not move the repair
// before fusion: the vendor fusion passes do not accept the resulting scf.for
// region.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include <limits>
#include <map>

using namespace mlir;

namespace {
constexpr StringLiteral specsAttr = "flagtree.rank1_wrap_specs";
constexpr StringLiteral tagPrefix = "flagtree.rank1.wrap.";
constexpr int64_t maxIndex = std::numeric_limits<int32_t>::max();

bool named(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

bool containsRemainder(Value value, llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *op = value.getDefiningOp();
  if (!op || !seen.insert(op).second)
    return false;
  // Scalar remainders (for example a wrapped program id) already survive as
  // ordinary scalar offsets. Only tensor remainders become split pointers.
  if ((named(op, "arith.remsi") || named(op, "arith.remui")) &&
      isa<RankedTensorType>(value.getType()))
    return true;
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return containsRemainder(operand, seen);
  });
}

bool isTileBase(Value value, int64_t width) {
  Operation *op = value.getDefiningOp();
  if (!named(op, "arith.muli"))
    return false;
  for (int i = 0; i < 2; ++i) {
    Operation *pid = op->getOperand(i).getDefiningOp();
    Operation *constant = op->getOperand(1 - i).getDefiningOp();
    auto attr = constant ? constant->getAttrOfType<IntegerAttr>("value")
                         : IntegerAttr();
    if (named(pid, "tt.get_program_id") && attr && attr.getInt() == width)
      return true;
  }
  return false;
}

bool isRamp(Value value, int64_t width) {
  Operation *op = value.getDefiningOp();
  if (named(op, "tt.make_range")) {
    auto start = op->getAttrOfType<IntegerAttr>("start");
    auto end = op->getAttrOfType<IntegerAttr>("end");
    return start && end && start.getInt() == 0 && end.getInt() == width;
  }
  if (!named(op, "arith.addi"))
    return false;
  for (int i = 0; i < 2; ++i) {
    Operation *range = op->getOperand(i).getDefiningOp();
    Operation *splat = op->getOperand(1 - i).getDefiningOp();
    if (named(range, "tt.make_range") && named(splat, "tt.splat") &&
        isTileBase(splat->getOperand(0), width) &&
        isRamp(op->getOperand(i), width))
      return true;
  }
  return false;
}

struct CaptureRank1Wrap
    : PassWrapper<CaptureRank1Wrap, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CaptureRank1Wrap)
  StringRef getArgument() const final { return "tritonxpu-capture-rank1-wrap"; }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(specsAttr)) {
      module.emitError("rank-1 wrap capture already applied");
      return signalPassFailure();
    }
    SmallVector<Attribute> specs;
    bool failed = false;
    module.walk([&](Operation *load) {
      if (!named(load, "tt.load"))
        return;
      Value ptr = load->getOperand(0);
      auto type = dyn_cast<RankedTensorType>(ptr.getType());
      if (!type || type.getRank() != 1)
        return;
      llvm::SmallPtrSet<Operation *, 32> seen;
      if (!containsRemainder(ptr, seen))
        return;

      auto reject = [&]() {
        load->emitError("unsupported P800 SDNN rank-1 modulo load: expected "
                        "a direct argument plus (pid * BLOCK + lane) % PERIOD "
                        "with positive constant PERIOD and x < LENGTH mask");
        failed = true;
      };
      int64_t width = type.getDimSize(0);
      // Nested control flow, reused pointers, arbitrary strides, dynamic
      // periods and non-prefix masks need a different lowering. Fail rather
      // than silently letting the vendor path discard their modulo.
      if (!named(load->getParentOp(), "tt.func") || width <= 0 ||
          width > maxIndex || !ptr.hasOneUse())
        return reject();
      Operation *add = ptr.getDefiningOp();
      if (!named(add, "tt.addptr"))
        return reject();
      Operation *base = add->getOperand(0).getDefiningOp();
      Operation *rem = add->getOperand(1).getDefiningOp();
      if (!named(base, "tt.splat") || !named(rem, "arith.remsi"))
        return reject();
      auto arg = dyn_cast<BlockArgument>(base->getOperand(0));
      Operation *constant = rem->getOperand(1).getDefiningOp();
      auto values = constant
                        ? constant->getAttrOfType<DenseIntElementsAttr>("value")
                        : DenseIntElementsAttr();
      if (!arg || arg.getOwner() != load->getBlock() || !values ||
          !values.isSplat() || !isRamp(rem->getOperand(0), width))
        return reject();
      auto rampType = dyn_cast<RankedTensorType>(rem->getOperand(0).getType());
      if (!rampType || !rampType.getElementType().isInteger(32))
        return reject();
      int64_t period = values.getSplatValue<APInt>().getSExtValue();
      if (period <= 0 || period > maxIndex || load->getNumOperands() < 2)
        return reject();
      Operation *mask = load->getOperand(1).getDefiningOp();
      if (!named(mask, "arith.cmpi") ||
          mask->getOperand(0) != rem->getOperand(0))
        return reject();
      auto predicate =
          mask->getAttrOfType<arith::CmpIPredicateAttr>("predicate");
      Operation *limitOp = mask->getOperand(1).getDefiningOp();
      auto limit = limitOp
                       ? limitOp->getAttrOfType<DenseIntElementsAttr>("value")
                       : DenseIntElementsAttr();
      if (!predicate || predicate.getValue() != arith::CmpIPredicate::slt ||
          !limit || !limit.isSplat())
        return reject();
      int64_t length = limit.getSplatValue<APInt>().getSExtValue();
      if (length < 0 || length > maxIndex - width)
        return reject();

      Builder builder(module.getContext());
      int64_t id = specs.size();
      auto tag = builder.getStringAttr(tagPrefix.str() + std::to_string(id));
      Location loc = NameLoc::get(tag, load->getLoc());
      load->setLoc(loc);
      add->setLoc(loc);
      rem->setLoc(loc);
      specs.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("id", builder.getI64IntegerAttr(id)),
           builder.getNamedAttr("period", builder.getI64IntegerAttr(period)),
           builder.getNamedAttr(
               "arg", builder.getI64IntegerAttr(arg.getArgNumber()))}));
    });
    if (failed)
      return signalPassFailure();
    module->setAttr(specsAttr, ArrayAttr::get(module.getContext(), specs));
  }
};

// Fusion may combine locations. Never pick an arbitrary source if two wrapped
// loads have become one operation: the repair must still identify one DMA.
void collectTags(Location loc, llvm::SmallSet<int64_t, 4> &tags) {
  if (auto name = dyn_cast<NameLoc>(loc)) {
    StringRef text = name.getName().getValue();
    int64_t id;
    if (text.consume_front(tagPrefix) && !text.getAsInteger(10, id))
      tags.insert(id);
    collectTags(name.getChildLoc(), tags);
  } else if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations())
      collectTags(child, tags);
  } else if (auto call = dyn_cast<CallSiteLoc>(loc)) {
    collectTags(call.getCallee(), tags);
    collectTags(call.getCaller(), tags);
  }
}

struct RepairRank1Wrap : PassWrapper<RepairRank1Wrap, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RepairRank1Wrap)
  StringRef getArgument() const final { return "tritonxpu-repair-rank1-wrap"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto specs = module->getAttrOfType<ArrayAttr>(specsAttr);
    if (!specs) {
      module.emitError("missing rank-1 wrap capture");
      return signalPassFailure();
    }
    std::map<int64_t, SmallVector<Operation *>> found;
    bool ambiguous = false;
    module.walk([&](Operation *op) {
      if (!named(op, "sdnn.dma"))
        return;
      llvm::SmallSet<int64_t, 4> tags;
      collectTags(op->getLoc(), tags);
      if (tags.size() > 1) {
        op->emitError("ambiguous rank-1 wrap DMA provenance");
        ambiguous = true;
      }
      for (int64_t id : tags)
        found[id].push_back(op);
    });
    if (ambiguous)
      return signalPassFailure();
    for (Attribute attr : specs) {
      auto spec = cast<DictionaryAttr>(attr);
      int64_t id = cast<IntegerAttr>(spec.get("id")).getInt();
      if (found[id].size() != 1) {
        module.emitError("rank-1 wrap requires exactly one DMA for load ")
            << id;
        return signalPassFailure();
      }
    }
    for (Attribute attr : specs) {
      auto spec = cast<DictionaryAttr>(attr);
      int64_t id = cast<IntegerAttr>(spec.get("id")).getInt();
      int64_t period = cast<IntegerAttr>(spec.get("period")).getInt();
      Operation *dma = found[id][0];
      if (dma->getNumOperands() != 2 || dma->getNumResults() != 0) {
        dma->emitError("unexpected rank-1 wrap DMA arity");
        return signalPassFailure();
      }
      auto castOp = dma->getOperand(0).getDefiningOp<memref::CastOp>();
      auto src = castOp ? castOp.getSource().getDefiningOp<memref::SubViewOp>()
                        : memref::SubViewOp();
      auto view =
          src ? src.getSource().getDefiningOp<memref::ReinterpretCastOp>()
              : memref::ReinterpretCastOp();
      auto dst = dma->getOperand(1).getDefiningOp<memref::SubViewOp>();
      auto arg =
          view ? dyn_cast<BlockArgument>(view.getSource()) : BlockArgument();
      if (!view || !dst || src.getType().getRank() != 1 ||
          dst.getType().getRank() != 1 || !arg ||
          arg.getArgNumber() != cast<IntegerAttr>(spec.get("arg")).getInt() ||
          src.getMixedSizes() != dst.getMixedSizes() ||
          !isConstantIntValue(src.getMixedOffsets()[0], 0) ||
          !isConstantIntValue(src.getMixedStrides()[0], 1) ||
          !isConstantIntValue(view.getMixedStrides()[0], 1)) {
        dma->emitError("unexpected rank-1 wrap DMA views; refusing repair");
        return signalPassFailure();
      }

      OpBuilder builder(dma);
      Location loc = dma->getLoc();
      auto materialize = [&](OpFoldResult value) {
        return getValueOrCreateConstantIndexOp(builder, loc, value);
      };
      Value offset = materialize(view.getMixedOffsets()[0]);
      Value count = materialize(src.getMixedSizes()[0]);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value modulus = builder.create<arith::ConstantIndexOp>(loc, period);
      Value start = builder.create<arith::RemSIOp>(loc, offset, modulus);
      Value sum = builder.create<arith::AddIOp>(loc, start, count);
      Value pad = builder.create<arith::ConstantIndexOp>(loc, period - 1);
      Value numerator = builder.create<arith::AddIOp>(loc, sum, pad);
      Value segments = builder.create<arith::DivSIOp>(loc, numerator, modulus);
      Value active = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, count, zero);
      Value upper =
          builder.create<arith::SelectOp>(loc, active, segments, zero);
      // Each iteration copies min(remaining lanes, lanes until the next wrap).
      // A fully masked tile executes no DMA, including when offset % period !=
      // 0.
      auto loop =
          builder.create<scf::ForOp>(loc, zero, upper, one, ValueRange{zero});
      builder.setInsertionPointToStart(loop.getBody());
      Value pos = loop.getRegionIterArgs()[0];
      Value absolute = builder.create<arith::AddIOp>(loc, offset, pos);
      Value wrapped = builder.create<arith::RemSIOp>(loc, absolute, modulus);
      Value tillWrap = builder.create<arith::SubIOp>(loc, modulus, wrapped);
      Value remaining = builder.create<arith::SubIOp>(loc, count, pos);
      Value length = builder.create<arith::MinUIOp>(loc, remaining, tillWrap);
      auto oldType = cast<MemRefType>(castOp.getType());
      auto layout = StridedLayoutAttr::get(module.getContext(),
                                           ShapedType::kDynamic, {1});
      auto type =
          MemRefType::get({ShapedType::kDynamic}, oldType.getElementType(),
                          layout, oldType.getMemorySpace());
      auto source = builder.create<memref::ReinterpretCastOp>(
          loc, type, view.getSource(), OpFoldResult(wrapped),
          ArrayRef<OpFoldResult>{length},
          ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
      auto dynamicSource = builder.create<memref::CastOp>(loc, oldType, source);
      auto dest = builder.create<memref::SubViewOp>(
          loc, dst.getResult(), ArrayRef<OpFoldResult>{pos},
          ArrayRef<OpFoldResult>{length},
          ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
      Operation *copy = dma->clone();
      copy->setOperands({dynamicSource.getResult(), dest.getResult()});
      builder.insert(copy);
      Value next = builder.create<arith::AddIOp>(loc, pos, length);
      builder.create<scf::YieldOp>(loc, next);
      dma->erase();
      if (castOp->use_empty())
        castOp->erase();
      if (src->use_empty())
        src->erase();
      if (view->use_empty())
        view->erase();
    }
    module->removeAttr(specsAttr);
  }
};
} // namespace

namespace mlir::triton::xpu {
std::unique_ptr<Pass> createCaptureRank1WrapPass() {
  return std::make_unique<CaptureRank1Wrap>();
}
std::unique_ptr<Pass> createRepairRank1WrapPass() {
  return std::make_unique<RepairRank1Wrap>();
}
} // namespace mlir::triton::xpu
