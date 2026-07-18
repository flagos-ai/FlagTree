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

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_PROPAGATION_UTILITY_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_PROPAGATION_UTILITY_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Tools/LinearLayout.h"
#include <optional>

namespace mlir::triton::gpu {

// Given the result |dstLayout|, infer the source layout that we should use for
// global load if we propagate through op def chain of |defOp|. Returns
// std::nullopt if fails to infer or cannot reach a global load.
std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(const LinearLayout &dstLayout, Operation *defOp);
std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(LinearEncodingAttr dstLayout, Operation *defOp);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_PROPAGATION_UTILITY_H_
