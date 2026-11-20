#include "LayoutConversionUtils.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Conversion/CommonIRToTTGIR/CommonIRLowerTensorTiles.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <memory>
#include <tuple>

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIROPTIMIZETENSORTILELAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

Attribute getTileStoreSourcePreference(gpu::LocalStoreOp store) {
  Value source = store.getSrc();
  SmallVector<Operation *> views;
  bool hasTile = false;
  // Follow only logical views. Loads stay effect boundaries: use their
  // existing coalesced encoding without cloning or moving the memory read.
  while (true) {
    if (Value input = getConversionSource(source)) {
      if (tile::getExplicitValueEncoding(source))
        return {};
      source = input;
      continue;
    }
    Operation *op = source.getDefiningOp();
    if (isMctleExtractTile(op) && isStaticMctleTile(op))
      hasTile = true;
    else if (!isa_and_nonnull<triton::TransOp>(op))
      break;
    views.push_back(op);
    source = op->getOperand(0);
  }
  if (!hasTile)
    return {};

  Attribute encoding;
  // Only follow definitions that can supply this store. A loop-carried
  // value can come from either the initial value or the previous iteration.
  llvm::SmallSetVector<Value, 4> sources;
  sources.insert(source);
  for (unsigned i = 0; i < sources.size(); ++i) {
    Value value = sources[i];
    if (auto load = value.getDefiningOp<triton::LoadOp>()) {
      auto type = dyn_cast<RankedTensorType>(load.getType());
      if (!type || !isa<gpu::BlockedEncodingAttr>(type.getEncoding()))
        return {};
      if (encoding && encoding != type.getEncoding())
        return {};
      encoding = type.getEncoding();
      continue;
    }
    if (!appendTensorSources(value, sources))
      return {};
  }
  if (!encoding)
    return {};
  for (Operation *view : llvm::reverse(views)) {
    if (isa<triton::TransOp>(view))
      encoding = mlir::inferDstEncoding(view, encoding);
    else if (!canLowerTensorTileInEncoding(view, encoding))
      return {};
    if (!encoding)
      return {};
    if (auto fixed = tile::getExplicitValueEncoding(view->getResult(0)))
      if (fixed != encoding)
        return {};
  }
  return encoding;
}

namespace {

using TensorLayoutRequirements = llvm::MapVector<OpOperand *, Attribute>;

struct Recipe {
  Value source;
  RankedTensorType type;
  bool clone = false;
  SmallVector<Recipe *> operands;
  Value materialized;
};

bool canChangeEncoding(Value value, Attribute encoding) {
  auto fixed = tile::getExplicitValueEncoding(value);
  return !fixed || fixed == encoding;
}

Attribute inferOperandEncoding(Operation *op, Attribute encoding) {
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return isa<DenseElementsAttr>(constant.getValue()) ? encoding : Attribute{};
  if (isa<triton::MakeRangeOp, triton::SplatOp, triton::BroadcastOp>(op))
    return encoding;
  if (isStaticMctleTile(op))
    return canLowerTensorTileInEncoding(op, encoding) ? encoding : Attribute{};
  // Rematerialize only address/mask arithmetic and simple tensor views.
  // Loads and other operations with effects remain conversion boundaries.
  bool arithmetic = op->getName().getDialectNamespace() == "arith" &&
                    op->hasTrait<OpTrait::Elementwise>();
  if (!isMemoryEffectFree(op) ||
      !(arithmetic ||
        isa<triton::AddPtrOp, triton::ExpandDimsOp, triton::TransOp>(op)))
    return {};
  return mlir::inferSrcEncoding(op, encoding);
}

class PlannedTensorLayouts {
public:
  explicit PlannedTensorLayouts(const TensorLayoutRequirements &requirements) {
    for (auto [use, encoding] : requirements) {
      Recipe *recipe = get(use->get(), encoding);
      uses.emplace_back(use, recipe);
    }
  }

  void apply() {
    // Retype producers and loop carriers together. Capture existing uses before
    // creating any boundary conversions; planned uses will bypass them below.
    struct OldUses {
      Value value;
      RankedTensorType type;
      SmallVector<OpOperand *> uses;
    };
    SmallVector<OldUses> oldUses;
    for (auto [value, encoding] : assignments) {
      auto type = cast<RankedTensorType>(value.getType());
      if (type.getEncoding() == encoding)
        continue;
      OldUses old{value, type, {}};
      for (OpOperand &use : value.getUses())
        old.uses.push_back(&use);
      oldUses.push_back(std::move(old));
      value.setType(type.cloneWithEncoding(encoding));
    }
    for (const auto &old : oldUses) {
      if (old.uses.empty())
        continue;
      for (OpOperand *use : old.uses)
        use->set(convert(old.value, old.type, use->getOwner()));
    }
    for (auto [use, recipe] : uses)
      use->set(materialize(recipe, use->getOwner()));
  }

private:
  Recipe *make(Value value, RankedTensorType type) {
    recipes.push_back(std::make_unique<Recipe>(Recipe{value, type}));
    return recipes.back().get();
  }

  Recipe *get(Value value, Attribute encoding) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return make(value, {});
    auto target = type.cloneWithEncoding(encoding);
    if (!canChangeEncoding(value, encoding))
      return make(value, target);
    if (Value source = getConversionSource(value))
      return get(source, encoding);
    auto key = std::make_pair(value, encoding);
    if (Recipe *cached = rewritten.lookup(key))
      return cached;
    Recipe *recipe = rewritten[key] = make(value, target);
    if (assignments.count(value) || type.getEncoding() == encoding)
      return recipe;

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
        if (arg.getArgNumber() > 0) {
          planLoopSlot(loop, arg.getArgNumber() - 1, encoding);
          return recipe;
        }
    } else if (auto loop = value.getDefiningOp<scf::ForOp>()) {
      planLoopSlot(loop, cast<OpResult>(value).getResultNumber(), encoding);
      return recipe;
    }

    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumRegions() != 0)
      return recipe;
    Attribute input = inferOperandEncoding(op, encoding);
    if (!input)
      return recipe;
    recipe->clone = true;
    for (Value operand : op->getOperands())
      recipe->operands.push_back(get(operand, input));
    return recipe;
  }

  void planLoopSlot(scf::ForOp loop, unsigned index, Attribute encoding) {
    BlockArgument iterArg = loop.getRegionIterArgs()[index];
    Value result = loop.getResult(index);
    if (!canChangeEncoding(iterArg, encoding) ||
        !canChangeEncoding(result, encoding))
      return;
    // The first requirement chooses the carrier; incompatible users receive
    // conversions. Publish the carrier before visiting the loop backedge.
    if (assignments.count(iterArg))
      return;
    assignments[iterArg] = encoding;
    assignments[result] = encoding;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    OpOperand *initial = loop.getTiedLoopInit(iterArg);
    OpOperand &yielded = yield->getOpOperand(index);
    Recipe *initialRecipe = get(initial->get(), encoding);
    Recipe *nextRecipe = get(yielded.get(), encoding);
    uses.emplace_back(initial, initialRecipe);
    uses.emplace_back(&yielded, nextRecipe);
  }

  Value convert(Value value, RankedTensorType type, Operation *before) {
    if (value.getType() == type)
      return value;
    auto key = std::make_tuple(value, type.getEncoding(), before->getBlock());
    if (Value cached = conversions.lookup(key)) {
      // Share within the block, at the earliest actual consumer. Placing a
      // conversion immediately after the producer can cross an explicit wait
      // and reuse scratch shared memory while async copies are still pending.
      Operation *conversion = cached.getDefiningOp();
      if (conversion != before && !conversion->isBeforeInBlock(before))
        conversion->moveBefore(before);
      return cached;
    }
    return conversions[key] = convertLayout(value, type, before);
  }

  Value materialize(Recipe *recipe, Operation *before) {
    if (recipe->materialized)
      return recipe->materialized;
    if (!recipe->type)
      return recipe->source;
    if (!recipe->clone)
      return convert(recipe->source, recipe->type, before);
    Operation *op = recipe->source.getDefiningOp();
    IRMapping mapping;
    for (auto [operand, input] : llvm::zip(op->getOperands(), recipe->operands))
      mapping.map(operand, materialize(input, op));
    Operation *copy = OpBuilder(op).clone(*op, mapping);
    copy->getResult(0).setType(recipe->type);
    if (auto constant = dyn_cast<arith::ConstantOp>(copy)) {
      auto value = cast<DenseElementsAttr>(constant.getValue());
      constant.setValueAttr(value.reshape(recipe->type));
    }
    return recipe->materialized = copy->getResult(0);
  }

  llvm::MapVector<Value, Attribute> assignments;
  SmallVector<std::unique_ptr<Recipe>> recipes;
  DenseMap<std::pair<Value, Attribute>, Recipe *> rewritten;
  SmallVector<std::pair<OpOperand *, Recipe *>> uses;
  DenseMap<std::tuple<Value, Attribute, Block *>, Value> conversions;
};

void eraseDeadProducers(ModuleOp module) {
  llvm::SetVector<Operation *> worklist;
  module.walk([&](Operation *op) {
    if (op->getNumRegions() == 0)
      worklist.insert(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value operand : op->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        if (producer->getNumRegions() == 0)
          worklist.insert(producer);
    op->erase();
  }
}

struct CommonIROptimizeTensorTileLayoutsPass
    : impl::CommonIROptimizeTensorTileLayoutsBase<
          CommonIROptimizeTensorTileLayoutsPass> {
  void runOnOperation() override {
    TensorLayoutRequirements requirements;
    getOperation().walk([&](gpu::ConvertLayoutOp convert) {
      if (tile::getExplicitResultEncoding(convert, 0))
        return;
      // Leave ordinary conversions to RemoveLayoutConversions.
      llvm::SetVector<Operation *> slice;
      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      options.filter = [](Operation *op) {
        return op->getNumRegions() == 0 && isMemoryEffectFree(op);
      };
      if (failed(getBackwardSlice(convert.getOperation(), &slice, options)) ||
          !llvm::any_of(slice,
                        [](Operation *op) { return isStaticMctleTile(op); }))
        return;
      Attribute encoding = convert.getType().getEncoding();
      for (OpOperand &use : convert.getResult().getUses()) {
        // Intermediate targets can unnecessarily fix loop layouts.
        // Propagate only the final consumer's requirement.
        if (isa<gpu::ConvertLayoutOp>(use.getOwner()))
          continue;
        requirements[&use] = encoding;
      }
    });
    getOperation().walk([&](gpu::LocalStoreOp store) {
      if (auto preferred = getTileStoreSourcePreference(store))
        if (preferred != store.getSrc().getType().getEncoding())
          requirements[&store->getOpOperand(0)] = preferred;
    });
    if (requirements.empty())
      return;
    PlannedTensorLayouts(requirements).apply();
    eraseDeadProducers(getOperation());
  }
};

} // namespace
} // namespace mlir::triton::metax
