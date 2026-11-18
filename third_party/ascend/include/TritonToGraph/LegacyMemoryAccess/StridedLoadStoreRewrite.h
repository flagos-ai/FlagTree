/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_GRAPH_LEGACY_MEMORY_ACCESS_STRIDED_LOAD_STORE_REWRITE_H
#define TRITON_TO_GRAPH_LEGACY_MEMORY_ACCESS_STRIDED_LOAD_STORE_REWRITE_H

#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace StridedLoadStoreRewrite {

using namespace mlir;
using namespace triton;

inline constexpr const char *RewrittenByStridedLoadStoreRewriteTAG =
    mlir::triton::memory_access::RewrittenByStridedLoadStoreRewriteTAG;
inline constexpr const char *InspectedByStridedLoadStoreRewriteTAG =
    mlir::triton::memory_access::InspectedByStridedLoadStoreRewriteTAG;

class LoadConverter : public OpRewritePattern<triton::LoadOp> {
public:
  explicit LoadConverter(MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context) {}

  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

class StoreConverter : public OpRewritePattern<triton::StoreOp> {
public:
  explicit StoreConverter(MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context) {}

  using OpRewritePattern<triton::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace StridedLoadStoreRewrite

#endif // TRITON_TO_GRAPH_LEGACY_MEMORY_ACCESS_STRIDED_LOAD_STORE_REWRITE_H
