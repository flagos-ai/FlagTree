#include "triton/Conversion/CommonIRToTTGIR/TensorLayoutPropagation.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <tuple>

namespace mlir::triton::metax {
namespace {

SmallVector<Value> loopSlot(scf::ForOp loop, unsigned index) {
  return {
      loop.getInitArgs()[index], loop.getRegionIterArgs()[index],
      loop.getResult(index),
      cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(index)};
}

struct Recipe {
  Value source;
  RankedTensorType type;
  bool clone = false;
  SmallVector<Recipe *> operands;
  Value materialized;
};

class PlannedTensorLayouts final : public TensorLayoutPlan {
public:
  PlannedTensorLayouts(const TensorLayoutRequirements &requirements,
                       const TensorLayoutAdapter &adapter)
      : adapter(adapter), assignments(requirements.assignments) {
    for (const auto &requirement : requirements.uses) {
      Recipe *recipe = get(requirement.use->get(), requirement.encoding,
                           requirement.propagate);
      uses.emplace_back(requirement.use, recipe);
    }
  }

  void apply() override {
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

  Recipe *get(Value value, Attribute encoding, bool propagate = true) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return make(value, {});
    auto target = type.cloneWithEncoding(encoding);
    if (Value source = adapter.getConversionSource(value))
      return get(source, encoding, propagate);
    if (!propagate)
      return make(value, target);
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
    auto values = loopSlot(loop, index);
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

  const TensorLayoutAdapter &adapter;
  llvm::MapVector<Value, Attribute> assignments;
  SmallVector<std::unique_ptr<Recipe>> recipes;
  DenseMap<std::pair<Value, Attribute>, Recipe *> rewritten;
  SmallVector<std::pair<OpOperand *, Recipe *>> uses;
  DenseMap<std::tuple<Value, Attribute, Block *>, Value> conversions;
};

} // namespace

SmallVector<Value>
collectTensorLayoutComponent(Value seed, const TensorLayoutAdapter &adapter) {
  llvm::SetVector<Value> values;
  values.insert(seed);
  auto addSlot = [&](scf::ForOp loop, unsigned index) {
    for (Value value : loopSlot(loop, index))
      values.insert(value);
  };
  for (unsigned i = 0; i < values.size(); ++i) {
    Value value = values[i];
    if (Value source = adapter.getConversionSource(value))
      values.insert(source);
    else if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
        if (arg.getArgNumber() > 0)
          addSlot(loop, arg.getArgNumber() - 1);
    } else if (auto loop = value.getDefiningOp<scf::ForOp>()) {
      addSlot(loop, cast<OpResult>(value).getResultNumber());
    }
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        if (use.getOperandNumber() >= loop.getNumControlOperands())
          addSlot(loop, use.getOperandNumber() - loop.getNumControlOperands());
      } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(yield->getParentOp()))
          addSlot(loop, use.getOperandNumber());
      } else if (user->getNumResults() == 1 &&
                 adapter.getConversionSource(user->getResult(0)) == value) {
        values.insert(user->getResult(0));
      }
    }
  }
  return values.takeVector();
}

std::unique_ptr<TensorLayoutPlan>
planTensorLayouts(const TensorLayoutRequirements &requirements,
                  const TensorLayoutAdapter &adapter) {
  return std::make_unique<PlannedTensorLayouts>(requirements, adapter);
}

} // namespace mlir::triton::metax
