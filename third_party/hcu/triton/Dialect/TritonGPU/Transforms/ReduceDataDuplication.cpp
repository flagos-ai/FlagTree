#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#ifdef __TLE__
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

namespace mlir {
namespace triton {
namespace gpu {

#ifdef __TLE__
static bool isLikelyRemotePtr(Value ptr) {
  constexpr StringLiteral kRemoteShardCarrierAttr =
      "tle.remote_shard_id_carrier";
  SmallVector<Value> worklist{ptr};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    if (def->getName().getStringRef() == "tle.remote_pointers")
      return true;
    if (auto addPtr = dyn_cast<triton::AddPtrOp>(def)) {
      if (addPtr->hasAttr(kRemoteShardCarrierAttr))
        return true;
      worklist.push_back(addPtr.getPtr());
      worklist.push_back(addPtr.getOffset());
      continue;
    }
    if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(def)) {
      worklist.push_back(cvt.getSrc());
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      worklist.push_back(bcast.getSrc());
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
      worklist.push_back(expand.getSrc());
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(def)) {
      worklist.push_back(reshape.getSrc());
      continue;
    }
  }
  return false;
}

static bool comesFromRemoteLoad(Value value, DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;

  if (auto load = dyn_cast<triton::LoadOp>(def))
    return isLikelyRemotePtr(load.getPtr());

  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    unsigned idx = result.getResultNumber();
    return comesFromRemoteLoad(ifOp.thenYield().getOperand(idx), visited) ||
           comesFromRemoteLoad(ifOp.elseYield().getOperand(idx), visited);
  }

  for (Value operand : def->getOperands()) {
    if (comesFromRemoteLoad(operand, visited))
      return true;
  }
  return false;
}
#endif

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUReduceDataDuplicationPass
    : public impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
#ifdef __TLE__
      DenseSet<Value> visited;
      if (comesFromRemoteLoad(cvtOp.getSrc(), visited))
        return;
#endif
      if (!cvtNeedsSharedMemory(srcType, dstType))
        return;
      auto order = getOrderForMemory(srcType);
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SwizzledSharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), order,
              triton::gpu::getCTALayout(srcEncoding), srcType.getElementType()),
          sharedMemorySpace);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
