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

// TLE (Triton Language Extension) distributed ops → GCU lowering.
//
// Uses SharedGenericConversionPattern (string-based op matching) for TLE
// dialect ops (no TLE headers needed), and SharedConversionPattern for
// typed triton_gcu ops.
//
// Handles:
//   tle.distributed_barrier      → gcu.cluster_barrier
//   triton_gcu.remote_memdesc    → gcu.remote_memref

#include <map>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "tle-op-to-gcu"

using namespace mlir;

namespace {

// ===----------------------------------------------------------------------===
// tle.distributed_barrier → dispatch on group_kind
//
// On NVIDIA Hopper this lowers to cluster arrive/wait PTX instructions.
// On GCU 400/410, the thread hierarchy differs from NVIDIA GPUs:
//   GCU thread   = GPU CTA (block)
//   GCU subthread = GPU warp/thread
//
// group_kind variants from Python frontend:
//   - (none) / "cluster" : cluster barrier  → gcu.cluster_barrier
//                          (inter-CTA sync via __syncthreads)
//   - "submesh"          : submesh barrier  → TODO: implement later
//   - "grid"             : grid barrier     → not supported on GCU
// ===----------------------------------------------------------------------===
struct TleDistributedBarrierOpLowering : SharedGenericConversionPattern {
  TleDistributedBarrierOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::PrivateTagPool &pTagPool)
      : SharedGenericConversionPattern("tle.distributed_barrier", converter,
                                       ctx, userAnalysis, replaced2Origin,
                                       pTagPool) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op))
      pTagPool.releaseMap(op);

    auto loc = op->getLoc();

    auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
    StringRef kind = kindAttr ? kindAttr.getValue() : "";

    LLVM_DEBUG({
      llvm::dbgs() << "[TleOpToGCU] distributed_barrier: group_kind=";
      if (kindAttr)
        llvm::dbgs() << "\"" << kind << "\"";
      else
        llvm::dbgs() << "(none/cluster)";
      llvm::dbgs() << "\n";
    });

    if (kind == "grid") {
      rewriter.create<gcu::GridBarrierOp>(loc);
      leaveTritionOp(rewriter, op);
      rewriter.eraseOp(op);
      return success();
    }

    if (kind == "submesh") {
      // TODO(xingxing.li): implement submesh barrier (leader CTA SMEM atomic
      // barrier) For now, fall through to cluster barrier as a conservative
      // over-synchronization.
      LLVM_DEBUG(llvm::dbgs() << "[TleOpToGCU] submesh barrier not yet "
                                 "implemented, falling back to cluster "
                                 "barrier\n");
    }

    // Cluster barrier: synchronize all CTAs in the cluster.
    // gcu.cluster_barrier → tops_syncthreads → __syncthreads()
    // (On GCU, __syncthreads synchronizes threads = CTAs within a cluster)
    rewriter.create<gcu::ClusterBarrierOp>(loc);

    leaveTritionOp(rewriter, op);
    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// triton_gcu.remote_memdesc → gcu.remote_memref
//
// At this stage the type converter has already turned the !ttg.memdesc
// operands into memrefs. We simply create a gcu.remote_memref with the
// converted source memref and the shard_id.
// ===----------------------------------------------------------------------===
struct RemoteMemDescOpLowering
    : SharedConversionPattern<triton::gcu::RemoteMemDescOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::RemoteMemDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());

    auto loc = op.getLoc();
    auto srcMemRef = dyn_cast<MemRefType>(adaptor.getSrc().getType());
    if (!srcMemRef)
      return failure();

    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!resultType)
      return failure();

    Value shardId = adaptor.getShardId();
    if (!shardId.getType().isInteger(32))
      shardId =
          rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), shardId);

    auto remote = rewriter.create<mlir::gcu::RemoteMemRefOp>(
        loc, resultType, adaptor.getSrc(), shardId);

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, remote.getResult());
    return success();
  }
};

} // namespace

void mlir::triton::populateTleOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    ConversionTarget &target, triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  auto *ctx = patterns.getContext();

  patterns.add<TleDistributedBarrierOpLowering>(converter, ctx, userAnalysis,
                                                replaced2Origin, pTagPool);
  patterns.add<RemoteMemDescOpLowering>(converter, ctx, userAnalysis,
                                        replaced2Origin, pTagPool);

  target.addDynamicallyLegalOp(OperationName("tle.distributed_barrier", ctx),
                               [](Operation *) { return false; });
  target.addIllegalOp<triton::gcu::RemoteMemDescOp>();
}
