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

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

static bool preservesSqmmaOperandBoundary(arith::TruncFOp trunc) {
  auto contract = triton::musa::recoverUniqueSqmmaConsumerContractFromTensor(
      trunc.getResult());
  if (failed(contract))
    return true;
  return contract->has_value();
}

static bool sinkTruncAfterMmaConvert(ttg::ConvertLayoutOp cvt,
                                     RewriterBase &rewriter) {
  auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
  if (!srcTy || !dstTy)
    return false;
  if (!isa_and_nonnull<ttg::MUSASqmmaEncodingAttr>(srcTy.getEncoding()))
    return false;
  if (!isa_and_nonnull<ttg::BlockedEncodingAttr>(dstTy.getEncoding()))
    return false;
  if (!isa<FloatType>(srcTy.getElementType()) ||
      !isa<FloatType>(dstTy.getElementType()))
    return false;

  SmallVector<arith::TruncFOp> truncUsers;
  for (Operation *user : cvt->getUsers()) {
    auto trunc = dyn_cast<arith::TruncFOp>(user);
    if (!trunc)
      return false;
    truncUsers.push_back(trunc);
  }
  if (truncUsers.empty())
    return false;

  bool changed = false;
  for (arith::TruncFOp trunc : truncUsers) {
    if (preservesSqmmaOperandBoundary(trunc))
      continue;
    auto truncDstTy = dyn_cast<RankedTensorType>(trunc.getType());
    if (!truncDstTy)
      continue;
    if (!isa<FloatType>(truncDstTy.getElementType()))
      continue;
    auto mmaTruncTy = RankedTensorType::get(
        srcTy.getShape(), truncDstTy.getElementType(), srcTy.getEncoding());
    rewriter.setInsertionPoint(trunc);
    Value mmaTrunc = arith::TruncFOp::create(rewriter, trunc.getLoc(),
                                             mmaTruncTy, cvt.getSrc());
    Value cvtAfterTrunc = ttg::ConvertLayoutOp::create(rewriter, trunc.getLoc(),
                                                       truncDstTy, mmaTrunc);
    rewriter.replaceOp(trunc, cvtAfterTrunc);
    changed = true;
  }

  if (changed && cvt->use_empty())
    rewriter.eraseOp(cvt);
  return changed;
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUCANONICALIZESQMMARESULTCONVERSIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUCanonicalizeSqmmaResultConversionsPass
    : impl::TritonMUSAGPUCanonicalizeSqmmaResultConversionsBase<
          TritonMUSAGPUCanonicalizeSqmmaResultConversionsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    for (tt::FuncOp func : mod.getOps<tt::FuncOp>()) {
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<ttg::ConvertLayoutOp> cvtOps;
        func.walk([&](ttg::ConvertLayoutOp op) { cvtOps.push_back(op); });
        for (ttg::ConvertLayoutOp cvt : cvtOps) {
          if (!cvt->getBlock())
            continue;
          if (sinkTruncAfterMmaConvert(cvt, rewriter))
            changed = true;
        }
      }
    }
  }
};

} // namespace mlir
