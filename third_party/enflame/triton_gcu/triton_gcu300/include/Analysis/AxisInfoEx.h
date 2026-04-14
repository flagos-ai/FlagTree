/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRITON_ANALYSIS_AXISINFO_EX_H
#define TRITON_ANALYSIS_AXISINFO_EX_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#include <optional>
#include <type_traits>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
//===--------------------------------------------------------------------===//
// The main logical is modified from
// include/triton/Analysis/AxisInfo.h in the triton repo.
//===--------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gcu {
//===----------------------------------------------------------------------===//
// AxisInfoEx
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
class AxisInfoEx {
public:
  typedef SmallVector<int64_t> DimVectorT;
  typedef ArrayRef<int64_t> DimRefT;
  constexpr static int64_t kInitDivisibility = 1;
  constexpr static int64_t kDefaultContinueSize = 1;
  constexpr static int64_t kDefaultContinualInterval = -1;

public:
  AxisInfoEx() : AxisInfoEx({}, {}, {}) {}

  AxisInfoEx(DimRefT pDivisibility, DimRefT pContinualSize,
             DimRefT pContinualInterval)
      : AxisInfoEx(pDivisibility, pContinualSize, pContinualInterval,
                   std::nullopt) {}

  AxisInfoEx(DimRefT pDivisibility, DimRefT pContinualSize,
             DimRefT pContinualInterval,
             std::optional<int64_t> pConstantValue) {
    divisibility.append(pDivisibility.begin(), pDivisibility.end());
    continualSize.append(pContinualSize.begin(), pContinualSize.end());
    continualInterval.append(pContinualInterval.begin(),
                             pContinualInterval.end());
    constantValue = pConstantValue;
    rank = continualSize.size();
    assert(divisibility.size() == static_cast<size_t>(rank));
    assert(continualSize.size() == static_cast<size_t>(rank));
    assert(continualInterval.size() == static_cast<size_t>(rank));
  }

  int64_t getContiguity(size_t dim) const {
    int64_t dimContinualSize = 1;
    if (continualInterval[dim] == 1)
      dimContinualSize = continualSize[dim];
    return dimContinualSize;
  }

  int64_t getConstancy(size_t dim) const {
    int64_t dimConstancySize = 1;
    if (continualInterval[dim] == 0)
      dimConstancySize = continualSize[dim];
    return dimConstancySize;
  }

  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  int64_t getContinualSize(size_t dim) const { return continualSize[dim]; }
  const DimVectorT &getContinualSize() const { return continualSize; }

  int64_t getContinualInterval(size_t dim) const {
    return continualInterval[dim];
  }
  const DimVectorT &getContinualInterval() const { return continualInterval; }

  int getRank() const { return rank; }

  std::optional<int64_t> getConstantValue() const { return constantValue; }

  bool isContinualLowDim(ArrayRef<int64_t> shape, int dim) const {
    return getContiguity(dim) == shape[dim];
  }

  bool isConstantDim(ArrayRef<int64_t> shape, int dim) const {
    return getConstancy(dim) == shape[dim];
  }

  bool isContinualDim(ArrayRef<int64_t> shape, int dim) const {
    return getContinualSize(dim) == shape[dim];
  }

  bool isStridedContinualDim(ArrayRef<int64_t> shape, int dim) const {
    if (continualInterval.size() < 1 || continualSize.size() < 1)
      return false;
    if (shape[dim] == 1)
      return true;
    return getContinualInterval(dim) == 1 && getContinualSize(dim) > 1 &&
           shape[dim] % getContiguity(dim) == 0;
  }

  bool isStridedConstantDim(ArrayRef<int64_t> shape, int dim) const {
    if (continualInterval.size() < 1 || continualSize.size() < 1)
      return false;
    if (shape[dim] == 1)
      return true;
    return getContinualInterval(dim) == 0 && getContinualSize(dim) > 1 &&
           shape[dim] % getConstancy(dim) == 0;
  }

  bool operator==(const AxisInfoEx &other) const {
    return divisibility == other.divisibility &&
           continualSize == other.continualSize &&
           continualInterval == other.continualInterval &&
           constantValue == other.constantValue && rank == other.rank;
  }

  template <class T>
  static void initPessimisticStateFromFunc(int argNumber, T funcOp, int rank,
                                           DimVectorT *contiguity,
                                           DimVectorT *divisibility,
                                           DimVectorT *constancy);

  static AxisInfoEx getPessimisticValueState(Value value);

  // The gcd of both arguments for each dimension
  static AxisInfoEx join(const AxisInfoEx &lhs, const AxisInfoEx &rhs);

  void print(raw_ostream &os) const {
    auto print = [&](StringRef name, DimVectorT vec) {
      os << name << " = [";
      llvm::interleaveComma(vec, os);
      os << "]";
    };
    print("divisibility", divisibility);
    print(", continualSize", continualSize);
    print(", continualInterval", continualInterval);
    os << ", constant_value = ";
    if (constantValue)
      os << *constantValue;
    else
      os << "<none>";
  }

private:
  // The continual information maps the `d`-th
  // dimension to the length of the shortest
  // sequence of integers of the same continualInterval along it.
  // Suppose we have an array of N elements,
  // with a continualSize value C,
  // the array can be divided into a list of
  // N/C sequences of C subsequence of integers of the same continualInterval.
  // For example:
  // [10, 11, 12, 13, 18, 19, 20, 21]
  // [20, 21, 22, 23, 28, 29, 30, 31]
  // Would have continualSize [2, 4] and have continualInterval [10, 1].
  // and
  // [12, 16, 20, 24]
  // [13, 17, 21, 25]
  // [14, 18, 22, 26]
  // [15, 19, 23, 27]
  // [18, 22, 26, 30]
  // [19, 23, 27, 31]
  // Would have continualSize [2, 4] and have continualInterval [1, 4].
  DimVectorT continualSize;
  DimVectorT continualInterval;

  // the divisibility information maps the `d`-th dimension to
  // the largest power of two that divides the first element
  // of all the values along it.
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
  DimVectorT divisibility;

  // The constant value of the lattice if we can infer it.
  std::optional<int64_t> constantValue;

  // Number of dimensions of the lattice.
  int rank{};
};

// Module level axis info analysis based on the call graph, assuming that we do
// not have recursive functions.
//
// Since each function will be called multiple times, we need to calculate the
// axis info based on the axis info of all the callers.  In the future, we can
// perform optimization using function cloning so that each call site will have
// unique axis info.
using AxisInfoExMapT = DenseMap<Value, AxisInfoEx>;
class ModuleAxisInfoExAnalysis : public CallGraph<AxisInfoExMapT> {
public:
  explicit ModuleAxisInfoExAnalysis(ModuleOp moduleOp)
      : CallGraph<AxisInfoExMapT>(moduleOp) {
    SmallVector<FunctionOpInterface> funcs;
    for (auto root : getRoots()) {
      walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
          // Pre-order edge walk callback
          [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
          // Post-order node walk callback
          [&](FunctionOpInterface funcOp) {
            funcs.push_back(funcOp);
            funcMap.try_emplace(funcOp, AxisInfoExMapT{});
          });
      (void)root;
    }
    SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
    for (auto funcOp : llvm::reverse(sortedFuncs)) {
      initialize(funcOp);
      funcOp.walk([&](CallOpInterface callOp) {
        auto callee = dyn_cast<FunctionOpInterface>(callOp.resolveCallable());
        update(callOp, callee);
      });
    }
  }

  AxisInfoEx *getAxisInfoEx(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *axisInfoExMap = getFuncData(funcOp);
    if (!axisInfoExMap) {
      return nullptr;
    }
    auto it = axisInfoExMap->find(value);
    if (it == axisInfoExMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }

  unsigned getPtrContiguity(Value ptr);
  unsigned getPtrAlignment(Value ptr);
  unsigned getMaskAlignment(Value mask);

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};

} // namespace gcu
} // namespace triton
} // namespace mlir

#endif
