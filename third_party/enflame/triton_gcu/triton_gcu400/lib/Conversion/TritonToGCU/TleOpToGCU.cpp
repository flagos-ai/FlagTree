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
#include "PatternTritonGPUOpToGCU.h"
#include "Utility.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
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

// ===----------------------------------------------------------------------===
// tle.dsl_region → tle.dsl_region (with !gcu.ptr/memreftype operands lowered to
// !llvm.ptr and dim/stride)
// ===----------------------------------------------------------------------===
#ifdef ENABLE_TLE
struct TleDSLRegionOpLowering
    : SharedConversionPattern<triton::tle::DSLRegionOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tle::DSLRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Already converted in a previous run — nothing to do.
    if (op->hasAttr("tle_raw.converted"))
      return failure();
    bool hasCall = false;
    if (op->getNumRegions() > 0) {
      Region &body = op->getRegion(0);
      if (!body.empty()) {
        Block *entryBlock = &body.front();
        for (Operation &innerOp : *entryBlock) {
          if (isa<LLVM::CallOp>(innerOp)) {
            hasCall = true;
            break;
          }
        }
      }
    }

    auto loc = op.getLoc();
    bool changed = false;
    SmallVector<Value> newOperands;
    ValueRange convertedInputs = adaptor.getInputs();
    newOperands.reserve(op.getNumOperands());
    auto llvmPtrTy =
        LLVM::LLVMPointerType::get(rewriter.getContext(), /*addressSpace=*/1);
    for (Value operand : convertedInputs) {
      if (auto gcuPtrTy = dyn_cast<gcu::PtrType>(operand.getType())) {
        (void)gcuPtrTy; // elementType is implicit in the gcu.ptr2int result
        Value asInt = rewriter.create<gcu::PtrToIntOp>(loc, operand);
        Value asPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrTy, asInt);
        newOperands.push_back(asPtr);
        changed = true;
      } else if (auto memrefTy = dyn_cast<MemRefType>(operand.getType())) {
        auto gcuPtr = rewriter.create<mlir::gcu::MemRefToPtrOp>(
            loc,
            mlir::gcu::PtrType::get(rewriter.getContext(),
                                    memrefTy.getElementType()),
            operand);
        Value asInt = rewriter.create<gcu::PtrToIntOp>(loc, gcuPtr);
        Value asPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrTy, asInt);
        newOperands.push_back(asPtr);
      } else {
        newOperands.push_back(operand);
      }
    }

    // Check whether this is a deferred DSL region.
    auto externAttr = op->getAttrOfType<StringAttr>("tle_raw.extern_func_name");
    bool isDeferred = (externAttr != nullptr);

    if (!changed && !isDeferred)
      return failure();
    if (hasCall && !changed)
      return failure();

    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());
    SmallVector<Type> newResultType;
    auto outputIndicesAttr =
        op->getAttrOfType<DenseI32ArrayAttr>("output_operand_indices");
    if (outputIndicesAttr) {
      auto outPutIdx = outputIndicesAttr.asArrayRef();
      for (int32_t idx : outPutIdx) {
        if (idx >= 0 && static_cast<size_t>(idx) < newOperands.size())
          newResultType.push_back(newOperands[idx].getType());
      }
      if (op->getResultTypes().size() != outPutIdx.size() ||
          newResultType.size() != outPutIdx.size()) {
        llvm::report_fatal_error(
            "tle::DSLRegionOp output_operand_indices contains an out-of-range "
            "index or does not match result count");
        op->dump();
      }
    } else {
      llvm::report_fatal_error(
          "tle::DSLRegionOp should be with output_operand_indices attritute");
      op.dump();
    }
    // Recreate tle.dsl_region with the converted operands.
    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    if (!newResultType.empty())
      state.addTypes(newResultType);
    else
      state.addTypes(op->getResultTypes());
    for (NamedAttribute attr : op->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    // Mark as converted so we don't process it again.
    state.addAttribute("tle_raw.converted", rewriter.getUnitAttr());
    for (unsigned i = 0; i < op->getNumRegions(); ++i)
      state.addRegion();
    Operation *newOp = rewriter.create(state);

    // Move the region bodies and update block argument types to match the
    // converted operands.
    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
      Region &oldRegion = op->getRegion(i);
      Region &newRegion = newOp->getRegion(i);
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
      for (Block &block : newRegion) {
        for (auto [arg, newOperand] :
             llvm::zip(block.getArguments(), newOperands)) {
          if (arg.getType() != newOperand.getType())
            arg.setType(newOperand.getType());
        }
      }
    }

    // For deferred DSL regions, declare the external function (if missing)
    // and insert an llvm.call inside the region body.
    if (isDeferred && !hasCall) {
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
        OpBuilder declBuilder(gpuModule.getBody(),
                              gpuModule.getBody()->begin());
        funcOp =
            declBuilder.create<LLVM::LLVMFuncOp>(loc, externFuncName, funcTy);
        funcOp.setLinkage(LLVM::Linkage::External);
        auto dslFileAttr =
            op->getAttrOfType<StringAttr>("tle_raw.dsl_file_name");
        if (dslFileAttr) {
          funcOp.setPassthroughAttr(declBuilder.getArrayAttr({
              declBuilder.getStringAttr("tle_raw.source_file"),
              dslFileAttr,
          }));
        }
      }

      // Insert llvm.call before the region terminator.
      Region &body = newOp->getRegion(0);
      if (!body.empty()) {
        Block *entryBlock = &body.front();
        Operation *terminator = entryBlock->getTerminator();
        OpBuilder bodyBuilder(rewriter.getContext());
        bodyBuilder.setInsertionPoint(terminator);

        SmallVector<Value> callOperands;
        TypeRange funcArgTys = funcOp.getArgumentTypes();
        unsigned numArgs =
            std::min(static_cast<unsigned>(entryBlock->getNumArguments()),
                     static_cast<unsigned>(funcArgTys.size()));
        if (numArgs != convertedInputs.size()) {
          llvm::report_fatal_error(
              "tle::DSLRegionOp arg number should same with input number!");
        }
        for (unsigned i = 0; i < numArgs; ++i) {
          Value arg = entryBlock->getArgument(i);
          Type paramTy = funcArgTys[i];
          if (arg.getType() == paramTy) {
            callOperands.push_back(arg);
          } else if (isa<LLVM::LLVMPointerType>(arg.getType()) &&
                     isa<LLVM::LLVMPointerType>(paramTy)) {
            // Address-space mismatch between block arg and func param.
            callOperands.push_back(bodyBuilder.create<LLVM::AddrSpaceCastOp>(
                loc, cast<LLVM::LLVMPointerType>(paramTy), arg));
          } else {
            callOperands.push_back(arg);
          }
          // For memref-derived operands, append rank dim values and rank
          // stride values to match the extern "C" device function signature:
          //   void VectorAdd(float *A, int A_dim0, int A_stride0, ...)
          if (auto memrefTy =
                  dyn_cast<MemRefType>(convertedInputs[i].getType())) {
            unsigned rank = memrefTy.getRank();
            // dims
            for (unsigned d = 0; d < rank; ++d) {
              int64_t dim = memrefTy.getDimSize(d);
              Value dimVal = bodyBuilder.create<arith::ConstantOp>(
                  loc,
                  bodyBuilder.getI32IntegerAttr(static_cast<int32_t>(dim)));
              callOperands.push_back(dimVal);
            }
            // strides
            auto [strides, offset] = memrefTy.getStridesAndOffset();
            for (unsigned d = 0; d < rank; ++d) {
              Value strideVal = bodyBuilder.create<arith::ConstantOp>(
                  loc, bodyBuilder.getI32IntegerAttr(
                           static_cast<int32_t>(strides[d])));
              callOperands.push_back(strideVal);
            }
          }
        }
        LLVM::CallOp callOp =
            bodyBuilder.create<LLVM::CallOp>(loc, funcOp, callOperands);
        callOp.setAlwaysInline(true);
      }
    }
    leaveTritionOp(rewriter, op.getOperation());
    SmallVector<Value> replacements;
    for (unsigned i = 0; i < newOp->getResults().size(); ++i) {
      int32_t outputIdx = outputIndicesAttr.asArrayRef()[i];
      if (auto memrefTy =
              dyn_cast<MemRefType>(convertedInputs[outputIdx].getType())) {
        // The newOp result is !llvm.ptr<1>; convert it back to a memref:
        //   !llvm.ptr → i64 → !gcu.ptr<T> → memref<?xT> → memref<final>
        Value llvmPtr = newOp->getResult(i);
        // !llvm.ptr → i64
        Value asInt = rewriter.create<LLVM::PtrToIntOp>(
            loc, rewriter.getI64Type(), llvmPtr);
        // i64 → !gcu.ptr<T>
        auto gcuPtrTy =
            gcu::PtrType::get(rewriter.getContext(), memrefTy.getElementType());
        auto gcuPtr = rewriter.create<gcu::IntToPtrOp>(loc, gcuPtrTy, asInt);
        // !gcu.ptr<T> → memref<?xT> (dynamic 1-D, no memory space)
        auto dynMemrefTy = MemRefType::get(
            ArrayRef<int64_t>{ShapedType::kDynamic}, memrefTy.getElementType());
        auto buffer =
            rewriter.create<gcu::PtrToMemRefOp>(loc, dynMemrefTy, gcuPtr);
        // memref<?xT> → memref<final shape> via ReinterpretCast
        auto [strides, offset] = memrefTy.getStridesAndOffset();
        auto output = rewriter.create<memref::ReinterpretCastOp>(
            loc, memrefTy, buffer, offset, memrefTy.getShape(), strides);
        replacements.push_back(output);
      } else {
        replacements.push_back(newOp->getResult(i));
      }
    }
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
    triton::gcu::PrivateTagPool &pTagPool) {
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
