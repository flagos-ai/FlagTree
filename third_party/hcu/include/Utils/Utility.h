// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_UTILS_UTILITY_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_UTILS_UTILITY_H_

#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <vector>
namespace mlir::LLVM::HCU {

template <typename T, typename U, typename BinaryOp>
std::vector<unsigned> multiDimElementwise(const ArrayRef<T> &lhs,
                                          const ArrayRef<U> &rhs, BinaryOp op) {
  assert(lhs.size() == rhs.size() && "Input dimensions must match");
  std::vector<unsigned> result;
  result.reserve(lhs.size());
  for (size_t i = 0, n = lhs.size(); i < n; ++i) {
    unsigned a = static_cast<unsigned>(lhs[i]);
    unsigned b = static_cast<unsigned>(rhs[i]);
    result.push_back(op(a, b));
  }
  return result;
}
} // namespace mlir::LLVM::HCU
#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_UTILS_UTILITY_H_
