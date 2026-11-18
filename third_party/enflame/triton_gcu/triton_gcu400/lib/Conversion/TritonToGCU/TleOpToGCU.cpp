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
#ifdef ENABLE_TLE
#include "tle/dialect/include/IR/Dialect.h"
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
      triton::gcu::DTETagPool &pTagPool)
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

// ===----------------------------------------------------------------------===
// tle.dsl_region (deferred raw TLE) -> a plain llvm.call.
//
// The deferred body is only a stub (`tle.yield %arg[idx]`), i.e. every result
// is an alias of one of the operands. The shell is therefore dropped entirely:
// the call is emitted in place and the results are replaced by the operands.
// Keeping the shell would force TleDSLRegionInline to split the surrounding
// block later on, which produces multi-block scf regions and fails to verify
// (e.g. "'scf.for' op expects region #0 to have 0 or 1 blocks").
//
// Non-deferred (already materialized) regions are not supported on GCU.
// ===----------------------------------------------------------------------===
#ifdef ENABLE_TLE
struct TleDSLRegionOpLowering
    : SharedConversionPattern<triton::tle::DSLRegionOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tle::DSLRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only deferred (raw TLE) regions are supported on GCU.
    auto dslFileAttr = op->getAttrOfType<StringAttr>("tle_raw.dsl_file_name");
    if (!dslFileAttr)
      return op.emitError(
          "non-deferred tle.dsl_region is not supported on GCU");
    auto externAttr = op->getAttrOfType<StringAttr>("tle_raw.extern_func_name");
    if (!externAttr)
      return op.emitError(
          "deferred tle.dsl_region is missing tle_raw.extern_func_name");

    Region &region = op.getBody();
    if (!region.hasOneBlock())
      return op.emitError(
          "deferred tle.dsl_region must have exactly one block");
    Block &body = region.front();
    if (body.empty())
      return op.emitError("deferred tle.dsl_region has an empty body");
    for (Operation &innerOp : body)
      if (isa<LLVM::CallOp>(innerOp))
        return op.emitError("deferred tle.dsl_region is already materialized");

    auto loc = op.getLoc();
    SmallVector<Value> newOperands;
    ValueRange convertedInputs = adaptor.getInputs();
    newOperands.reserve(op.getNumOperands());
    auto llvmPtrTy =
        LLVM::LLVMPointerType::get(rewriter.getContext(), /*addressSpace=*/1);
    for (Value operand : convertedInputs) {
      if (isa<gcu::PtrType>(operand.getType())) {
        Value asInt = rewriter.create<gcu::PtrToIntOp>(loc, operand);
        newOperands.push_back(
            rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrTy, asInt));
      } else if (auto memrefTy = dyn_cast<MemRefType>(operand.getType())) {
        auto gcuPtr = rewriter.create<mlir::gcu::MemRefToPtrOp>(
            loc,
            mlir::gcu::PtrType::get(rewriter.getContext(),
                                    memrefTy.getElementType()),
            operand);
        Value asInt = rewriter.create<gcu::PtrToIntOp>(loc, gcuPtr);
        newOperands.push_back(
            rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrTy, asInt));
      } else {
        newOperands.push_back(operand);
      }
    }

    auto outputIndicesAttr =
        op->getAttrOfType<DenseI32ArrayAttr>("output_operand_indices");
    if (!outputIndicesAttr)
      return op.emitError(
          "tle.dsl_region requires an output_operand_indices attribute");
    ArrayRef<int32_t> outputIndices = outputIndicesAttr.asArrayRef();
    if (op->getResultTypes().size() != outputIndices.size())
      return op.emitError("tle.dsl_region output_operand_indices size (")
             << outputIndices.size() << ") does not match result count ("
             << op->getResultTypes().size() << ")";
    for (int32_t idx : outputIndices)
      if (idx < 0 || static_cast<size_t>(idx) >= newOperands.size())
        return op.emitError("tle.dsl_region output_operand_indices contains "
                            "out-of-range index ")
               << idx;

    // The shell is dropped below, so the stub body must be exactly
    // `tle.yield %arg[idx]...`; anything else would be silently lost.
    auto yieldOp = dyn_cast<triton::tle::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return op.emitError("deferred tle.dsl_region must end with tle.yield");
    if (yieldOp.getNumOperands() != outputIndices.size())
      return op.emitError("deferred tle.dsl_region yields ")
             << yieldOp.getNumOperands() << " values but declares "
             << outputIndices.size() << " results";
    for (auto [i, idx] : llvm::enumerate(outputIndices))
      if (yieldOp.getOperand(i) != body.getArgument(idx))
        return op.emitError("deferred tle.dsl_region result #")
               << i << " must alias operand " << idx;

    // Signature of the extern "C" device function: memref operands are passed
    // as a raw pointer followed by rank dims and rank strides, e.g.
    //   void VectorAdd(float *A, int A_dim0, int A_stride0, ...)
    StringRef externFuncName = externAttr.getValue();
    SmallVector<Type> paramTys;
    paramTys.reserve(newOperands.size());
    auto i32Ty = rewriter.getI32Type();
    for (size_t idx = 0; idx < newOperands.size(); ++idx) {
      paramTys.push_back(newOperands[idx].getType());
      if (auto memrefTy =
              dyn_cast<MemRefType>(convertedInputs[idx].getType())) {
        unsigned rank = memrefTy.getRank();
        for (unsigned d = 0; d < rank; ++d)
          paramTys.push_back(i32Ty); // dim
        for (unsigned d = 0; d < rank; ++d)
          paramTys.push_back(i32Ty); // stride
      }
    }
    auto funcTy = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), paramTys);
    auto gpuModule = op->getParentOfType<gpu::GPUModuleOp>();
    LLVM::LLVMFuncOp funcOp =
        gpuModule.lookupSymbol<LLVM::LLVMFuncOp>(externFuncName);
    if (!funcOp) {
      OpBuilder declBuilder(gpuModule.getBody(), gpuModule.getBody()->begin());
      funcOp =
          declBuilder.create<LLVM::LLVMFuncOp>(loc, externFuncName, funcTy);
      funcOp.setLinkage(LLVM::Linkage::External);
      funcOp.setPassthroughAttr(declBuilder.getArrayAttr({
          declBuilder.getStringAttr("tle_raw.source_file"),
          dslFileAttr,
      }));
    } else if (funcOp.getFunctionType() != funcTy) {
      // The same extern symbol must be called with one consistent signature.
      return op.emitError("extern function '")
             << externFuncName << "' was already declared with a different "
             << "signature";
    }

    // Call arguments plus static dims/strides for memref-derived operands.
    SmallVector<Value> callOperands;
    callOperands.reserve(paramTys.size());
    for (size_t idx = 0; idx < newOperands.size(); ++idx) {
      callOperands.push_back(newOperands[idx]);
      if (auto memrefTy =
              dyn_cast<MemRefType>(convertedInputs[idx].getType())) {
        unsigned rank = memrefTy.getRank();
        for (unsigned d = 0; d < rank; ++d) {
          int64_t dim = memrefTy.getDimSize(d);
          callOperands.push_back(rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(dim))));
        }
        auto [strides, offset] = memrefTy.getStridesAndOffset();
        for (unsigned d = 0; d < rank; ++d) {
          callOperands.push_back(rewriter.create<arith::ConstantOp>(
              loc,
              rewriter.getI32IntegerAttr(static_cast<int32_t>(strides[d]))));
        }
      }
    }

    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());
    LLVM::CallOp callOp =
        rewriter.create<LLVM::CallOp>(loc, funcOp, callOperands);
    callOp.setAlwaysInline(true);
    leaveTritionOp(rewriter, op.getOperation());

    // The stub body yields the aliased operands, so the results are just the
    // operands themselves. Reuse the pre-LLVM (converted) values so users keep
    // their !gcu.ptr / memref types.
    SmallVector<Value> replacements;
    replacements.reserve(outputIndices.size());
    for (int32_t idx : outputIndices)
      replacements.push_back(convertedInputs[idx]);
    rewriter.replaceOp(op, replacements);
    return success();
  }
};
#endif

} // namespace

void mlir::triton::populateTleOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    ConversionTarget &target, triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::DTETagPool &pTagPool) {
  auto *ctx = patterns.getContext();

  patterns.add<TleDistributedBarrierOpLowering>(converter, ctx, userAnalysis,
                                                replaced2Origin, pTagPool);
  patterns.add<RemoteMemDescOpLowering>(converter, ctx, userAnalysis,
                                        replaced2Origin, pTagPool);
#ifdef ENABLE_TLE
  patterns.add<TleDSLRegionOpLowering>(converter, ctx, userAnalysis,
                                       replaced2Origin, pTagPool);
#endif
  target.addDynamicallyLegalOp(OperationName("tle.distributed_barrier", ctx),
                               [](Operation *) { return false; });
  target.addIllegalOp<triton::gcu::RemoteMemDescOp>();
  // tle.dsl_region is legal once it has been converted (tle_raw.converted
  // attribute present); otherwise it must be rewritten.
#ifdef ENABLE_TLE
  target.addDynamicallyLegalOp<triton::tle::DSLRegionOp>(
      [](triton::tle::DSLRegionOp op) {
        return op->hasAttr("tle_raw.converted");
      });
  // LLVM ops created during tle.dsl_region lowering (ptr2int → inttoptr,
  // deferred llvm.call, llvm.func declaration, addrspacecast).
  target.addLegalOp<LLVM::IntToPtrOp, LLVM::PtrToIntOp, LLVM::AddrSpaceCastOp,
                    LLVM::CallOp, LLVM::LLVMFuncOp>();
#endif
}
