#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <memory>
#include <tuple>

namespace mlir::triton::metax {
#define GEN_PASS_DEF_COMMONIROPTIMIZETENSORTILELAYOUTS
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

struct TensorLayoutRequirement {
  OpOperand *use;
  Attribute encoding;
};

struct Recipe {
  Value source;
  RankedTensorType type;
  bool clone = false;
  SmallVector<Recipe *> operands;
  Value materialized;
};

class PlannedTensorLayouts {
public:
  PlannedTensorLayouts(ArrayRef<TensorLayoutRequirement> requirements,
                       const TTGIRLayoutAdapter &adapter)
      : adapter(adapter) {
    for (const auto &requirement : requirements) {
      Recipe *recipe = get(requirement.use->get(), requirement.encoding);
      uses.emplace_back(requirement.use, recipe);
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
    if (!adapter.canChangeEncoding(value, encoding))
      return make(value, target);
    if (Value source = adapter.getConversionSource(value))
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
    Attribute input = adapter.inferOperandEncoding(op, encoding);
    if (!input)
      return recipe;
    recipe->clone = true;
    for (Value operand : op->getOperands())
      recipe->operands.push_back(get(operand, input));
    return recipe;
  }

  void planLoopSlot(scf::ForOp loop, unsigned index, Attribute encoding) {
    auto values = getLoopSlotValues(loop, index);
    if (!adapter.canChangeEncoding(values[1], encoding) ||
        !adapter.canChangeEncoding(values[2], encoding))
      return;
    // The first requirement chooses the carrier; incompatible users receive
    // conversions. Publish the carrier before visiting the loop backedge.
    if (assignments.count(values[1]))
      return;
    assignments[values[1]] = encoding;
    assignments[values[2]] = encoding;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    Recipe *initial = get(values[0], encoding);
    Recipe *next = get(values[3], encoding);
    uses.emplace_back(&loop->getOpOperand(loop.getNumControlOperands() + index),
                      initial);
    uses.emplace_back(&yield->getOpOperand(index), next);
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
    return conversions[key] = adapter.convertLayout(value, type, before);
  }

  Value materialize(Recipe *recipe, Operation *before) {
    if (recipe->materialized)
      return recipe->materialized;
    if (!recipe->type)
      return recipe->source;
    if (!recipe->clone)
      return convert(recipe->source, recipe->type, before);
    SmallVector<Value> operands;
    Operation *op = recipe->source.getDefiningOp();
    for (Recipe *operand : recipe->operands)
      operands.push_back(materialize(operand, op));
    return recipe->materialized =
               adapter.cloneWithLayout(op, recipe->type, operands);
  }

  const TTGIRLayoutAdapter &adapter;
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
    SmallVector<TensorLayoutRequirement> requirements;
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
          !llvm::any_of(slice, [](Operation *op) {
            return isa<tile::ExtractTileOp, tile::InsertTileOp>(op);
          }))
        return;
      Attribute encoding = convert.getType().getEncoding();
      for (OpOperand &use : convert.getResult().getUses()) {
        // Intermediate targets can unnecessarily fix loop layouts.
        // Propagate only the final consumer's requirement.
        if (isa<gpu::ConvertLayoutOp>(use.getOwner()))
          continue;
        requirements.push_back({&use, encoding});
      }
    });
    if (requirements.empty())
      return;
    TTGIRLayoutAdapter adapter;
    PlannedTensorLayouts(requirements, adapter).apply();
    eraseDeadProducers(getOperation());
  }
};

} // namespace
} // namespace mlir::triton::metax
