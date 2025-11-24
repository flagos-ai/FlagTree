#ifndef ILUVATAR_TRITON_ANALYSIS_AXISINFO_H
#define ILUVATAR_TRITON_ANALYSIS_AXISINFO_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#define FLAGTREE_SPEC_AxisInfo
#define FLAGTREE_SPEC_AxisInfo_classes
#define FLAGTREE_SPEC_AxisInfo_AxisInfo_functions
#define FLAGTREE_SPEC_AxisInfo_ModuleAxisInfoAnalysis_update

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
class AxisInfo {
public:
  typedef SmallVector<int64_t> DimVectorT;

public:
  AxisInfo() : AxisInfo({}, {}, {}, {}) {}

  AxisInfo(DimVectorT contiguity, DimVectorT divisibility, DimVectorT constancy,
           DimVectorT corexFlag)
      : AxisInfo(contiguity, divisibility, constancy, std::nullopt, corexFlag) {
  }

  AxisInfo(DimVectorT contiguity, DimVectorT divisibility, DimVectorT constancy,
           std::optional<int64_t> constantValue, DimVectorT corexFlag)
      : contiguity(contiguity), divisibility(divisibility),
        constancy(constancy), corexFlag(corexFlag),
        constantValue(constantValue) {
    assert(divisibility.size() == contiguity.size());
    assert(constancy.size() == contiguity.size());
  }

  // contiguity[d] is the length of the shortest sequence of contiguous integers
  // along dimension d.
  //
  // If we have an array of N elements with a contiguity value C, then the array
  // can be divided into a list of N/C sequences of C contiguous elements.
  // Since we have N = 2^k, C must be a power of two.
  //
  // For example, the 2D array
  //
  //   [[10, 11, 12, 13, 18, 19, 20, 21],
  //    [20, 21, 22, 23, 28, 29, 30, 31]]
  //
  // has contiguity [1, 4], and
  //
  //   [[12, 16, 20, 24],
  //    [13, 17, 21, 25],
  //    [14, 18, 22, 26],
  //    [15, 19, 23, 27],
  //    [18, 22, 26, 30],
  //    [19, 23, 27, 31]]
  //
  // has contiguity [2, 1].
  int64_t getContiguity(size_t dim) const { return contiguity[dim]; }
  const DimVectorT &getContiguity() const { return contiguity; }

  // divisibility[d] is the largest power of two that divides the first element
  // of all groups of length contiguity[d] along dimension d.
  //
  // For example,
  //
  //   [[10, 11, 12, 13, 18, 19, 20, 21],
  //    [20, 21, 22, 23, 28, 29, 30, 31]]
  //
  //  has divisibility [1, 2], and
  //
  //    [[12, 16, 20, 24],
  //     [13, 17, 21, 25],
  //     [14, 18, 22, 26],
  //     [15, 19, 23, 27]]
  //
  // has divisibility [4, 1].
  //
  // On the other hand,
  //
  //   [0, 1, 2, 0, 4, 5, 6, 7]
  //
  // has divisibility 1 because its contiguity is 1.
  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  // constancy[d] is the length of the shortest sequence of repeating integers
  // along dimension d.
  //
  // This is particularly useful to infer the contiguity of operations (e.g.
  // add) involving a constant.
  //
  // If we have an array of N elements, with a constancy value C, then the array
  // can be divided into a list of N/C sequences of C elements with the same
  // value.  Since we have N = 2^k, C must be a power of two.
  //
  // For example
  //
  //   [[8, 8, 8, 8, 12, 12, 12, 12],
  //    [16, 16, 16, 16, 20, 20, 20, 20]]
  //
  // has constancy [1, 4].
  int64_t getConstancy(size_t dim) const { return constancy[dim]; }
  const DimVectorT &getConstancy() const { return constancy; }

  // corexFlag is used to determine whether special instructions can be used to
  // accelerate data loading.
  int64_t getCorexFlag(size_t dim) const { return corexFlag[dim]; }
  const DimVectorT &getCorexFlag() const { return corexFlag; }

  int getRank() const { return contiguity.size(); }

  std::optional<int64_t> getConstantValue() const { return constantValue; }

  template <class T>
  static void
  initPessimisticStateFromFunc(int argNumber, T funcOp, DimVectorT *contiguity,
                               DimVectorT *divisibility, DimVectorT *constancy,
                               DimVectorT *corex_stride);

  bool operator==(const AxisInfo &other) const {
    return contiguity == other.contiguity &&
           divisibility == other.divisibility && constancy == other.constancy &&
           corexFlag == other.corexFlag && constantValue == other.constantValue;
  }

  static AxisInfo getPessimisticValueState(Value value);

  // The gcd of both arguments for each dimension
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs);

  void print(raw_ostream &os) const {
    auto print = [&](StringRef name, DimVectorT vec) {
      os << name << " = [";
      llvm::interleaveComma(vec, os);
      os << "]";
    };
    print("contiguity", contiguity);
    print(", divisibility", divisibility);
    print(", constancy", constancy);
    print(", corexflag", corexFlag);
    os << ", constant_value = ";
    if (constantValue)
      os << *constantValue;
    else
      os << "<none>";
  }

private:
  DimVectorT contiguity;
  DimVectorT divisibility;
  DimVectorT constancy;
  // The constant value of the lattice if we can infer it.

  std::optional<int64_t> constantValue;
  DimVectorT corexFlag;
};

} // namespace mlir::triton

#endif // ILUVATAR_TRITON_ANALYSIS_AXISINFO_H
