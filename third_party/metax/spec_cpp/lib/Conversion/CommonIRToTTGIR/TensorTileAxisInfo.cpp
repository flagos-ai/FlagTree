#include "triton/Conversion/CommonIRToTTGIR/TensorTileAxisInfo.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/CommonIRToTTGIR/CommonIRLowerTensorTiles.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

namespace mlir::triton::metax {
namespace {

using Operands = ArrayRef<const dataflow::Lattice<AxisInfo> *>;

// A runtime tile number still starts at a multiple of the tile extent.
// Use -1 for coordinates not known more precisely than that.
std::optional<SmallVector<int64_t>> getOffsets(RankedTensorType full,
                                               RankedTensorType part,
                                               std::optional<int64_t> index) {
  if (full.getRank() != part.getRank() ||
      full.getElementType() != part.getElementType() || (index && *index < 0))
    return std::nullopt;
  SmallVector<int64_t> offsets(full.getRank());
  int64_t remaining = index.value_or(0);
  for (int d = full.getRank() - 1; d >= 0; --d) {
    int64_t size = full.getDimSize(d), tileSize = part.getDimSize(d);
    if (size <= 0 || tileSize <= 0 || size % tileSize)
      return std::nullopt;
    int64_t tiles = size / tileSize;
    offsets[d] = index ? (remaining % tiles) * tileSize : (tiles == 1 ? 0 : -1);
    remaining /= tiles;
  }
  if (remaining)
    return std::nullopt;
  return offsets;
}

// Pointer divisibility is measured in bytes; contiguity counts elements.
int64_t getContiguousStep(Type element) {
  if (auto pointer = dyn_cast<PointerType>(element)) {
    Type pointee = pointer.getPointeeType();
    if (pointee.isIntOrFloat())
      return std::max<int64_t>(1, pointee.getIntOrFloatBitWidth() / 8);
  }
  return 1;
}

// gcd(divisibility, step * offset), without overflowing the multiplication.
int64_t shiftedDivisibility(int64_t divisibility, int64_t step,
                            int64_t offset) {
  int64_t common = std::gcd(divisibility, step);
  return common * std::gcd(divisibility / common, offset);
}

bool hasFacts(const AxisInfo &info, unsigned rank) {
  if (info.getRank() != rank)
    return false;
  for (unsigned d = 0; d < rank; ++d)
    if (info.getContiguity(d) <= 0 || info.getDivisibility(d) <= 0 ||
        info.getConstancy(d) <= 0)
      return false;
  return true;
}

// Splitting a contiguous run introduces additional group starts. Their
// alignment is bounded by the new run length, even when the old start was
// much more aligned.
int64_t splitDivisibility(const AxisInfo &info, unsigned dim, int64_t run,
                          int64_t extent, int64_t step) {
  int64_t divisor = info.getDivisibility(dim);
  if (run < info.getContiguity(dim) && run < extent)
    divisor = shiftedDivisibility(divisor, step, run);
  return divisor;
}

class ExtractTileAxisInfoVisitor final : public AxisInfoVisitor {
public:
  bool match(Operation *op) override { return isMctleExtractTile(op); }

  AxisInfo getAxisInfo(Operation *op, Operands operands) override {

    auto full = cast<RankedTensorType>(op->getOperand(0).getType());
    auto part = cast<RankedTensorType>(op->getResult(0).getType());
    const AxisInfo &source = operands[0]->getValue();
    auto offsets =
        getOffsets(full, part, operands[1]->getValue().getConstantValue());
    if (!offsets || !hasFacts(source, full.getRank()))
      return AxisInfo::getPessimisticValueState(op->getResult(0));

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    int64_t step = getContiguousStep(full.getElementType());
    for (unsigned d = 0; d < full.getRank(); ++d) {
      int64_t size = part.getDimSize(d), offset = (*offsets)[d];
      int64_t run = std::gcd(source.getContiguity(d), size);
      int64_t same = std::gcd(source.getConstancy(d), size);
      int64_t divisor = splitDivisibility(source, d, run, size, step);
      if (offset < 0) {
        if (size < source.getContiguity(d))
          divisor = shiftedDivisibility(divisor, step, size);
      } else {
        divisor = shiftedDivisibility(divisor, step,
                                      offset % source.getContiguity(d));
      }
      contiguity.push_back(run);
      divisibility.push_back(divisor);
      constancy.push_back(same);
    }
    return AxisInfo(contiguity, divisibility, constancy,
                    source.getConstantValue());
  }
};

class InsertTileAxisInfoVisitor final : public AxisInfoVisitor {
public:
  bool match(Operation *op) override { return isMctleInsertTile(op); }

  AxisInfo getAxisInfo(Operation *op, Operands operands) override {

    auto full = cast<RankedTensorType>(op->getOperand(0).getType());
    auto part = cast<RankedTensorType>(op->getOperand(1).getType());
    const AxisInfo &base = operands[0]->getValue();
    const AxisInfo &update = operands[1]->getValue();
    if (!getOffsets(full, part, operands[2]->getValue().getConstantValue()) ||
        !hasFacts(base, full.getRank()) || !hasFacts(update, full.getRank()))
      return AxisInfo::getPessimisticValueState(op->getResult(0));

    if (full.getShape() == part.getShape())
      return update;
    if (auto *extract = op->getOperand(1).getDefiningOp();
        isMctleExtractTile(extract)) {
      if (extract->getOperand(0) == op->getOperand(0) &&
          extract->getOperand(1) == op->getOperand(2))
        return base;
    }
    if (base.getConstantValue() &&
        base.getConstantValue() == update.getConstantValue())
      return base;

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    int64_t step = getContiguousStep(full.getElementType());
    for (unsigned d = 0; d < full.getRank(); ++d) {
      int64_t size = part.getDimSize(d);
      // Matching strides or constant runs inside both operands do not prove
      // continuity or equality across the boundary of the replacement tile.
      int64_t run = std::gcd(
          std::gcd(base.getContiguity(d), update.getContiguity(d)), size);
      contiguity.push_back(run);
      constancy.push_back(std::gcd(
          std::gcd(base.getConstancy(d), update.getConstancy(d)), size));
      divisibility.push_back(
          std::gcd(splitDivisibility(base, d, run, full.getDimSize(d), step),
                   splitDivisibility(update, d, run, size, step)));
    }
    return AxisInfo(contiguity, divisibility, constancy);
  }
};

} // namespace

void addTensorTileAxisInfoVisitors(AxisInfoVisitorList &visitors) {
  visitors.append<ExtractTileAxisInfoVisitor, InsertTileAxisInfoVisitor>();
}

} // namespace mlir::triton::metax
