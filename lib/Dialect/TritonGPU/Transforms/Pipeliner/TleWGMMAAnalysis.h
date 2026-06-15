#ifdef __TLE__
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINER_TLEWGMMAANALYSIS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINER_TLEWGMMAANALYSIS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::gpu::detail {

struct MemDescResource {
  Value root;
  SmallVector<Value, 2> indices;
  bool unknown = false;
};

struct PendingSharedWgmmaGroup {
  SmallVector<triton::nvidia_gpu::WarpGroupDotOp, 2> dots;
  SmallVector<MemDescResource, 4> reads;
};

class TlePipeResourceAnalysis {
public:
  bool isLifetimeBoundary(Operation *op) const;
  bool dotReadsSharedMemory(triton::nvidia_gpu::WarpGroupDotOp dotOp) const;
  SmallVector<MemDescResource, 2>
  getDotReadResources(triton::nvidia_gpu::WarpGroupDotOp dotOp) const;
  SmallVector<MemDescResource, 2>
  getBoundaryReleasedResources(Operation *op) const;
  bool
  releasedResourcesMayAliasReads(ArrayRef<MemDescResource> releasedResources,
                                 ArrayRef<MemDescResource> readResources) const;
  bool boundaryMayAliasReads(Operation *op,
                             ArrayRef<MemDescResource> readResources) const;
  bool hasAliasingLifetimeBoundaryBetween(
      Operation *from, Operation *to,
      ArrayRef<MemDescResource> readResources) const;
};

class TleWgmmaScheduleAnalysis {
public:
  TleWgmmaScheduleAnalysis(scf::ForOp forOp,
                           const TlePipeResourceAnalysis &resources)
      : forOp(forOp), resources(resources) {}

  bool canDeferWaitToLaterDotC(triton::nvidia_gpu::WarpGroupDotOp dotOp) const;
  bool
  canDeferCommitToLaterDotC(triton::nvidia_gpu::WarpGroupDotOp dotOp) const;
  bool canAppendToCurrentWgmmaCommitGroup(
      triton::nvidia_gpu::WarpGroupDotOp dotOp) const;
  bool
  canReuseAccumulatorChainC(triton::nvidia_gpu::WarpGroupDotOp dotOp) const;

private:
  scf::ForOp forOp;
  const TlePipeResourceAnalysis &resources;
};

void scheduleTleWgmmaAsyncLaunch(scf::ForOp forOp);

} // namespace mlir::triton::gpu::detail

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINER_TLEWGMMAANALYSIS_H_
#endif // __TLE__
