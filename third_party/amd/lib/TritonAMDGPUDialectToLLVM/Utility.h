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

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_

#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;

namespace mlir::LLVM::AMD {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

ElemLocationKey getElemCoordinatesFromRegisters(tt::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx);

std::optional<int> getRegFromCoordinates(tt::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx);

} // namespace mlir::LLVM::AMD
#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUDIALECTTOLLVM_UTILITY_H_
