// Copyright 2026 FlagOS Contributors

#ifndef TRITON_TLE_ANALYSIS_GATHER_AXIS_INFO_H
#define TRITON_TLE_ANALYSIS_GATHER_AXIS_INFO_H

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>

namespace mlir::triton::tle {

class GatherOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto gather = cast<triton::GatherOp>(op);
    const AxisInfo &source = operands[0]->getValue();
    auto sourceShape = gather.getSrc().getType().getShape();
    auto resultShape = gather.getType().getShape();
    if (source.getRank() != sourceShape.size())
      return AxisInfo();

    // Divisibility describes the first element of each contiguous group.
    // Only groups of length one prove a divisor of every source element.
    // Such a divisor holds after any valid gather, regardless of indices.
    // All divisors are powers of two, so their maximum is their LCM.
    int64_t divisor = 1;
    bool uniform = true;
    for (int d = 0; d < source.getRank(); ++d) {
      if (source.getContiguity(d) == 1)
        divisor = std::max(divisor, source.getDivisibility(d));
      uniform &= source.getConstancy(d) >= sourceShape[d];
    }

    // Arbitrary indices can break both consecutive values and repetitions.
    // A globally uniform source is the exception: every selected value is
    // the same, including when the gather changes the length of its axis.
    AxisInfo::DimVectorT contiguity(resultShape.size(), 1);
    AxisInfo::DimVectorT divisibility(resultShape.size(), divisor);
    AxisInfo::DimVectorT constancy(resultShape.size(), 1);
    if (uniform)
      constancy.assign(resultShape.begin(), resultShape.end());
    return AxisInfo(contiguity, divisibility, constancy,
                    uniform ? source.getConstantValue() : std::nullopt);
  }

  bool match(Operation *op) override { return isa<triton::GatherOp>(op); }
};

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_GATHER_AXIS_INFO_H
