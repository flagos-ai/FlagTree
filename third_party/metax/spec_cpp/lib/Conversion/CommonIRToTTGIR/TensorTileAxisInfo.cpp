#include "triton/Conversion/CommonIRToTTGIR/TensorTileAxisInfo.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/CommonIRToTTGIR/CommonIRLowerTensorTiles.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <numeric>

namespace mlir::triton::metax {
namespace {

using Operands = ArrayRef<const dataflow::Lattice<AxisInfo> *>;

// Facts that hold for any tile starting at a multiple of its shape.
AxisInfo getAlignedTileAxisInfo(const AxisInfo &source, RankedTensorType tile) {
  if (source.getRank() != tile.getRank())
    return {};
  auto pointer = dyn_cast<PointerType>(tile.getElementType());
  int64_t step = 1;
  if (pointer && pointer.getPointeeType().isIntOrFloat())
    step = std::max<int64_t>(1, getPointeeBitWidth(pointer) / 8);
  AxisInfo::DimVectorT contiguity, divisibility, constancy;
  for (auto [d, size] : llvm::enumerate(tile.getShape())) {
    int64_t run = source.getContiguity(d);
    int64_t divisor = source.getDivisibility(d);
    int64_t same = source.getConstancy(d);
    if (size <= 0 || run <= 0 || divisor <= 0 || same <= 0)
      return {};
    int64_t tileRun = std::gcd(run, size);
    if (tileRun < run) {
      // Restrict alignment to the distance between possible group starts.
      int64_t common = std::gcd(divisor, step);
      divisor = common * std::gcd(divisor / common, tileRun);
    }
    contiguity.push_back(tileRun);
    divisibility.push_back(divisor);
    constancy.push_back(std::gcd(same, size));
  }
  return AxisInfo(contiguity, divisibility, constancy,
                  source.getConstantValue());
}

class ExtractTileAxisInfoVisitor final : public AxisInfoVisitor {
public:
  bool match(Operation *op) override { return isMctleExtractTile(op); }

  AxisInfo getAxisInfo(Operation *op, Operands operands) override {
    auto tileType = cast<RankedTensorType>(op->getResult(0).getType());
    return getAlignedTileAxisInfo(operands[0]->getValue(), tileType);
  }
};

class InsertTileAxisInfoVisitor final : public AxisInfoVisitor {
public:
  bool match(Operation *op) override { return isMctleInsertTile(op); }

  AxisInfo getAxisInfo(Operation *op, Operands operands) override {
    Value full = op->getOperand(0), tile = op->getOperand(1);
    auto fullType = cast<RankedTensorType>(full.getType());
    auto tileType = cast<RankedTensorType>(tile.getType());
    const AxisInfo &base = operands[0]->getValue();
    const AxisInfo &update = operands[1]->getValue();
    if (tileType.getShape() == fullType.getShape())
      return update;
    if (auto *extract = tile.getDefiningOp(); isMctleExtractTile(extract))
      if (extract->getOperand(0) == full &&
          extract->getOperand(1) == op->getOperand(2))
        return base;
    if (base.getConstantValue() &&
        base.getConstantValue() == update.getConstantValue())
      return base;

    // Each tile in the result is either an original tile or the replacement.
    AxisInfo original = getAlignedTileAxisInfo(base, tileType);
    if (!original.getRank() || update.getRank() != original.getRank())
      return {};
    return AxisInfo::join(original, update);
  }
};

} // namespace

void addTensorTileAxisInfoVisitors(AxisInfoVisitorList &visitors) {
  visitors.append<ExtractTileAxisInfoVisitor, InsertTileAxisInfoVisitor>();
}

} // namespace mlir::triton::metax
