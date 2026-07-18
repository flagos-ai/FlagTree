// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_WMMAGROUP_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_WMMAGROUP_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

struct WmmaIntrinsic {
  // Chooses a suitable wmma instrinsic for the given input case.
  static FailureOr<WmmaIntrinsic> selectFor(int version, unsigned mDim,
                                            unsigned nDim, unsigned inputKDim,
                                            Type aElemType, Type bElemType,
                                            Type dElemType);
  // Gets the wmma intrinsic based on exact match of all parameters.
  static FailureOr<WmmaIntrinsic> get(int version, unsigned mDim, unsigned nDim,
                                      unsigned kDim, Type aElemType,
                                      Type bElemType, Type dElemType);

  WmmaIntrinsic(StringRef symbol, unsigned m, unsigned n, unsigned k,
                unsigned kB, Type aET, Type bET, Type dET)
      : name(symbol), mDim(m), nDim(n), kDim(k), kBase(kB), aElementType(aET),
        bElementType(bET), dElementType(dET) {}
  WmmaIntrinsic(const WmmaIntrinsic &other) = default;
  WmmaIntrinsic(WmmaIntrinsic &&other) = default;
  WmmaIntrinsic() = default;
  WmmaIntrinsic &operator=(WmmaIntrinsic &&other) = default;

  llvm::StringRef name;

  // m, n, and k refer to the shapes of the two operands of an wmma intrinsic:
  // Operand A has shape [m]x[k]; operand B has shape [k]x[n].

  unsigned mDim;
  unsigned nDim;
  unsigned kDim;

  // kBase is the number of elements each thread holds.
  unsigned kBase;

  Type aElementType;
  Type bElementType;
  Type dElementType;
};
} // namespace mlir

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_WMMAGROUP_H_
