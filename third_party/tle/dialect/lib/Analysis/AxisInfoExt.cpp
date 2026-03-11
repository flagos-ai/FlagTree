#include "tle/dialect/include/Analysis/AxisInfoExt.h"

#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <cstdlib>
#include <numeric>

namespace mlir::triton::tle {

namespace {

template <typename... Args> int64_t gcd(int64_t a, int64_t b, Args... args) {
  if constexpr (sizeof...(args) == 0) {
    return std::gcd(a, b);
  } else {
    return gcd(std::gcd(a, b), args...);
  }
}

int64_t multiplyDivisor(int64_t lhs, int64_t rhs) {
  // Safe as lhs and rhs are powers of 2.
  return std::abs(lhs * rhs);
}

class TleLocalPointersOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto local = dyn_cast<triton::tle::LocalPointersOp>(op);
    if (!local || operands.size() < 2)
      return AxisInfo();

    auto resultTy = dyn_cast<RankedTensorType>(local.getResult().getType());
    if (!resultTy)
      return AxisInfo();

    auto memDescTy =
        dyn_cast<triton::gpu::MemDescType>(local.getSrc().getType());
    if (!memDescTy)
      return AxisInfo();

    const int rank = resultTy.getRank();
    if (rank == 0)
      return AxisInfo();

    auto scaleAxisInfoByConstant = [&](const AxisInfo &src, int64_t scale) {
      AxisInfo::DimVectorT contiguity, divisibility, constancy;
      contiguity.reserve(rank);
      divisibility.reserve(rank);
      constancy.reserve(rank);
      for (int d = 0; d < rank; ++d) {
        const int64_t srcContig = src.getContiguity(d);
        const int64_t srcDiv = src.getDivisibility(d);
        const int64_t srcConstancy = src.getConstancy(d);
        // Mirror MulIOp behavior for x * C where C is compile-time constant.
        contiguity.push_back(scale == 1 ? srcContig : 1);
        constancy.push_back(srcConstancy);
        int64_t srcDivAdjusted = srcDiv;
        if (srcContig > 1 && scale != 1) {
          // Treat [2^n,2^n+1,...]'s divisibility as 1 if contiguity > 1.
          srcDivAdjusted = 1;
        }
        const int64_t scaleDiv = highestPowOf2Divisor<int64_t>(scale);
        divisibility.push_back(multiplyDivisor(srcDivAdjusted, scaleDiv));
      }
      std::optional<int64_t> constantValue = std::nullopt;
      if (src.getConstantValue().has_value())
        constantValue = src.getConstantValue().value() * scale;
      return AxisInfo(contiguity, divisibility, constancy, constantValue);
    };

    auto addAxisInfo = [&](const AxisInfo &lhs, const AxisInfo &rhs) {
      AxisInfo::DimVectorT contiguity, divisibility, constancy;
      contiguity.reserve(rank);
      divisibility.reserve(rank);
      constancy.reserve(rank);
      for (int d = 0; d < rank; ++d) {
        // Mirror AddSubOpAxisInfoVisitor<arith::AddIOp>.
        contiguity.push_back(
            std::max(gcd(lhs.getConstancy(d), rhs.getContiguity(d)),
                     gcd(lhs.getContiguity(d), rhs.getConstancy(d))));
        divisibility.push_back(
            gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
        constancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
      }
      std::optional<int64_t> constantValue = std::nullopt;
      if (lhs.getConstantValue().has_value() &&
          rhs.getConstantValue().has_value())
        constantValue =
            lhs.getConstantValue().value() + rhs.getConstantValue().value();
      return AxisInfo(contiguity, divisibility, constancy, constantValue);
    };

    // Build flattened offset axis info from row-major linearization:
    // offset = sum_i index_i * stride_i, where stride_i = prod(shape[i+1:]).
    AxisInfo offsetInfo;
    bool initialized = false;
    const auto memShape = memDescTy.getShape();
    const size_t maxTerms = std::min(memShape.size(), operands.size() - 1);
    for (size_t i = 0; i < maxTerms; ++i) {
      const AxisInfo &idxInfo = operands[i + 1]->getValue();
      if (idxInfo.getRank() != rank)
        return AxisInfo();

      int64_t stride = 1;
      for (size_t j = i + 1; j < memShape.size(); ++j)
        stride = multiplyDivisor(stride, memShape[j]);

      AxisInfo termInfo = scaleAxisInfoByConstant(idxInfo, stride);
      if (!initialized) {
        offsetInfo = termInfo;
        initialized = true;
      } else {
        offsetInfo = addAxisInfo(offsetInfo, termInfo);
      }
    }

    if (!initialized)
      return AxisInfo();

    // Pointer divisibility is tracked in bytes for alignment queries.
    auto ptrTy = dyn_cast<PointerType>(resultTy.getElementType());
    int64_t elemBytes = 1;
    if (ptrTy)
      elemBytes = std::max<int64_t>(1, getPointeeBitWidth(ptrTy) / 8);
    AxisInfo::DimVectorT byteDivisibility = offsetInfo.getDivisibility();
    for (int d = 0; d < rank; ++d)
      byteDivisibility[d] = multiplyDivisor(byteDivisibility[d], elemBytes);

    std::optional<int64_t> constantValue = std::nullopt;
    if (offsetInfo.getConstantValue().has_value())
      constantValue = offsetInfo.getConstantValue().value() * elemBytes;

    return AxisInfo(offsetInfo.getContiguity(), byteDivisibility,
                    offsetInfo.getConstancy(), constantValue);
  }

  bool match(Operation *op) override {
    return isa<triton::tle::LocalPointersOp>(op);
  }
};

class TleRemotePointersOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto remote = dyn_cast<triton::tle::RemotePointersOp>(op);
    if (!remote || operands.empty())
      return AxisInfo();

    const AxisInfo &baseInfo = operands[0]->getValue();
    if (baseInfo.getRank() == 0)
      return AxisInfo();

    // shard_id is expected to be a scalar/splat across the pointer tensor.
    // In this common case, keep the source pointer axis info unchanged.
    if (operands.size() < 2)
      return baseInfo;
    const AxisInfo &shardInfo = operands[1]->getValue();
    if (shardInfo.getRank() == 0)
      return baseInfo;

    const int rank = baseInfo.getRank();
    if (shardInfo.getRank() != rank)
      return baseInfo;
    bool shardIsUniform = true;
    for (int d = 0; d < rank; ++d) {
      if (shardInfo.getConstancy(d) <= 1) {
        shardIsUniform = false;
        break;
      }
    }
    if (shardIsUniform)
      return baseInfo;

    // Fallback to Add-style composition when shard tensor is non-uniform.
    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    contiguity.reserve(rank);
    divisibility.reserve(rank);
    constancy.reserve(rank);
    for (int d = 0; d < rank; ++d) {
      contiguity.push_back(
          std::max(gcd(baseInfo.getConstancy(d), shardInfo.getContiguity(d)),
                   gcd(baseInfo.getContiguity(d), shardInfo.getConstancy(d))));
      divisibility.push_back(
          gcd(baseInfo.getDivisibility(d), shardInfo.getDivisibility(d)));
      constancy.push_back(
          gcd(baseInfo.getConstancy(d), shardInfo.getConstancy(d)));
    }
    std::optional<int64_t> constantValue = std::nullopt;
    if (baseInfo.getConstantValue().has_value() &&
        shardInfo.getConstantValue().has_value()) {
      constantValue = baseInfo.getConstantValue().value() +
                      shardInfo.getConstantValue().value();
    }
    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }

  bool match(Operation *op) override {
    return isa<triton::tle::RemotePointersOp>(op);
  }
};

} // namespace

void AxisInfoExt::addVisitors(mlir::triton::AxisInfoVisitorList &visitors) {
  visitors.append<TleLocalPointersOpAxisInfoVisitor>();
  visitors.append<TleRemotePointersOpAxisInfoVisitor>();
}

} // namespace mlir::triton::tle
