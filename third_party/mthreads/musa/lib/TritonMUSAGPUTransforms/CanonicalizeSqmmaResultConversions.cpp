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
