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
#include "Utility.h"

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Conversion/TritonToGCU/ReduceScanCommon.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "Dialect/MathExt/IR/MathExtTypes.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-ir-to-gcu-ir-util"
#define DBG_TRITON_IR 1

namespace mlir {

int32_t getMasterThreadId(Region *region, int32_t defaultWarpId) {
  Operation *parentOp = region->getParentOp();
  auto wsOp = dyn_cast<mlir::gcu::WarpSpecializeOp>(parentOp);
  if (!wsOp)
    wsOp = parentOp->getParentOfType<mlir::gcu::WarpSpecializeOp>();
  if (wsOp)
    return wsOp.getMasterThreadId(region);
  return defaultWarpId;
}

int32_t getMasterThreadId(Operation *op, int32_t defaultWarpId) {
  return getMasterThreadId(op->getParentRegion(), defaultWarpId);
}

template <typename WarpSpecializeOpTy>
void captureValuesToWarpSpecializeOp(WarpSpecializeOpTy wsOp, Value capture) {
  wsOp->insertOperands(wsOp.getNumOperands(), capture);
  for (Region *region : wsOp.getPartitionRegions()) {
    BlockArgument arg =
        region->addArgument(capture.getType(), capture.getLoc());
    replaceAllUsesInRegionWith(capture, arg, *region);
  }
}

#if TRITON_VERSION >= 37
// In Triton 3.7, triton::gpu::WarpSpecializeOp no longer has explicitCaptures
// operands; they moved to WarpSpecializePartitionsOp.
template <>
void captureValuesToWarpSpecializeOp<triton::gpu::WarpSpecializeOp>(
    triton::gpu::WarpSpecializeOp wsOp, Value capture) {
  auto partOp = wsOp.getPartitionOp();
  partOp->insertOperands(partOp.getNumOperands(), capture);
  for (Region *region : wsOp.getPartitionRegions()) {
    BlockArgument arg =
        region->addArgument(capture.getType(), capture.getLoc());
    replaceAllUsesInRegionWith(capture, arg, *region);
  }
}
#endif

template <typename WarpSpecializeOpTy>
void captureValuesToWarpSpecializeOp(Operation *entryFunc, Value capture) {
  entryFunc->walk([&](Operation *op) {
    if (auto wsOp = dyn_cast<WarpSpecializeOpTy>(op))
      captureValuesToWarpSpecializeOp(wsOp, capture);
  });
}

Value getShareDTETag(OpBuilder &builder, Operation *op) {
  OpBuilder::InsertionGuard guard(builder);
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto secondIter = func.getFunctionBody().getBlocks().front().begin();
  secondIter++;
  auto secondOp = &(*secondIter);
  auto tagType = MemRefType::get(ArrayRef<int64_t>{1}, builder.getI32Type());
  if (isa<memref::AllocOp>(secondOp) && secondOp->getAttr("gcu.share_tag"))
    return secondOp->getResult(0);
  builder.setInsertionPoint(secondOp);
  auto tag = builder.create<memref::AllocOp>(op->getLoc(), tagType);
  tag->setAttr("gcu.share_tag", builder.getUnitAttr());
  return tag;
}

DenseSet<unsigned> getSlicedAxies(Type type) {
  DenseSet<unsigned> axies;
  if (auto tType = dyn_cast<TensorType>(type)) {
    auto numElems = triton::gcu::getElemsPerThread(type);
    for (unsigned i = 0; i < tType.getRank(); ++i) {
      if (numElems[i] != tType.getDimSize(i)) {
        axies.insert(i);
      }
    }
  }
  return axies;
}

SmallVector<Value, 4> getWarpIds(OpBuilder &builder, Location loc, Type type) {
  SmallVector<Value, 4> warpIds;
  if (auto tType = dyn_cast<RankedTensorType>(type)) {
    if (auto dotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
            tType.getEncoding())) {
      auto blockedLayout =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(dotEnc.getParent());
      auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
      auto rank = warpsPerCTA.size();
      bool isM = dotEnc.getOpIdx() == 0;
      bool rhsColMajor = !isM && dotEnc.getColumnMajor();
      unsigned nIdx = rhsColMajor ? (rank - 2) : (rank - 1);
      for (unsigned i = 0; i < tType.getRank(); ++i) {
        if (isM && i == rank - 2) {
          auto id = builder.create<arith::DivSIOp>(
              loc,
              builder.create<arith::RemSIOp>(
                  loc, builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
                  builder.create<arith::ConstantIndexOp>(
                      loc, warpsPerCTA[rank - 2] * warpsPerCTA[rank - 1])),
              builder.create<arith::ConstantIndexOp>(loc,
                                                     warpsPerCTA[rank - 1]));
          warpIds.push_back(id);
        } else if (!isM && i == nIdx) {
          auto id = builder.create<arith::RemSIOp>(
              loc,
              builder.create<arith::RemSIOp>(
                  loc, builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
                  builder.create<arith::ConstantIndexOp>(
                      loc, warpsPerCTA[rank - 2] * warpsPerCTA[rank - 1])),
              builder.create<arith::ConstantIndexOp>(loc,
                                                     warpsPerCTA[rank - 1]));
          warpIds.push_back(id);
        } else {
          warpIds.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
        }
      }
    } else if (auto blockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(
                   tType.getEncoding())) {
      auto slicedAxies = getSlicedAxies(type);
      auto warps = blockEnc.getWarpsPerCTA();
      auto shapePerCTA =
          triton::gpu::getShapePerCTA(blockEnc, tType.getShape());
      SmallVector<unsigned> warpMods(warps.size());
      SmallVector<unsigned> warpStrides(warps.size());
      unsigned warpMod = 1;
      unsigned warpStride = 1;
      for (int i = warps.size() - 1; i >= 0; --i) {
        warpMod *= warps[i];
        warpMods[i] = warpMod;
        warpStrides[i] = warpStride;
        warpStride *= warps[i];
      }
      unsigned i = 0;
      for (auto num : triton::gcu::getElemsPerThread(type)) {
        (void)num;
        if (slicedAxies.count(i)) {
          auto repeatNum =
              shapePerCTA[i] > warps[i] ? 1 : warps[i] / shapePerCTA[i];
          auto id = builder.create<arith::DivSIOp>(
              loc,
              builder.create<arith::RemSIOp>(
                  loc, builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
                  builder.create<arith::ConstantIndexOp>(loc, warpMods[i] /
                                                                  repeatNum)),
              builder.create<arith::ConstantIndexOp>(loc, warpStrides[i]));
          warpIds.push_back(id);
        } else {
          warpIds.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
        }
        ++i;
      }
    } else if (auto blockEnc = dyn_cast<triton::gpu::LinearEncodingAttr>(
                   tType.getEncoding())) {
      auto slicedAxies = getSlicedAxies(type);
      auto warps = blockEnc.getWarpsPerCTA();
      auto shapePerCTA =
          triton::gpu::getShapePerCTA(blockEnc, tType.getShape());
      SmallVector<unsigned> warpMods(warps.size());
      SmallVector<unsigned> warpStrides(warps.size());
      unsigned warpMod = 1;
      unsigned warpStride = 1;
      for (int i = warps.size() - 1; i >= 0; --i) {
        warpMod *= warps[i];
        warpMods[i] = warpMod;
        warpStrides[i] = warpStride;
        warpStride *= warps[i];
      }
      unsigned i = 0;
      for (auto num : triton::gcu::getElemsPerThread(type)) {
        (void)num;
        if (slicedAxies.count(i)) {
          auto repeatNum =
              shapePerCTA[i] > warps[i] ? 1 : warps[i] / shapePerCTA[i];
          auto id = builder.create<arith::DivSIOp>(
              loc,
              builder.create<arith::RemSIOp>(
                  loc, builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
                  builder.create<arith::ConstantIndexOp>(loc, warpMods[i] /
                                                                  repeatNum)),
              builder.create<arith::ConstantIndexOp>(loc, warpStrides[i]));
          warpIds.push_back(id);
        } else {
          warpIds.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
        }
        ++i;
      }
    } else if (auto sliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(
                   tType.getEncoding())) {
      auto parent = sliceEnc.getParent();
      auto outShape = sliceEnc.paddedShape(tType.getShape());
      SmallVector<unsigned> sliceDims;
      sliceDims.push_back(sliceEnc.getDim());
      while (auto innerSliceEnc =
                 dyn_cast<triton::gpu::SliceEncodingAttr>(parent)) {
        auto curSliceDim = innerSliceEnc.getDim();
        for (size_t idx = 0; idx < sliceDims.size(); idx++) {
          if (sliceDims[idx] >= curSliceDim) {
            sliceDims[idx] = sliceDims[idx] + 1;
          }
        }
        llvm::ArrayRef<int64_t> inputShpe = outShape;
        outShape = innerSliceEnc.paddedShape(inputShpe);
        sliceDims.push_back(curSliceDim);
        parent = innerSliceEnc.getParent();
      }
      size_t rank = outShape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      SmallVector<unsigned> warpsPerCTA;
      SmallVector<unsigned> threadsPerWarp;
      SmallVector<int64_t> shapePerCTA;
      if (auto blockEncParent =
              dyn_cast<triton::gpu::BlockedEncodingAttr>(parent)) {
        warpsPerCTA = SmallVector<unsigned>(blockEncParent.getWarpsPerCTA());
        threadsPerWarp =
            SmallVector<unsigned>(blockEncParent.getThreadsPerWarp());
        shapePerCTA = triton::gpu::getShapePerCTA(blockEncParent, outShape);
      } else if (auto linearEncParent =
                     dyn_cast<triton::gpu::LinearEncodingAttr>(parent)) {
        warpsPerCTA = SmallVector<unsigned>(linearEncParent.getWarpsPerCTA());
        threadsPerWarp =
            SmallVector<unsigned>(linearEncParent.getThreadsPerWarp());
        shapePerCTA = triton::gpu::getShapePerCTA(linearEncParent, outShape);
      } else {
        llvm::report_fatal_error("[Error] bad slice layout parent");
        assert(false && "bad slice layout parent");
        return warpIds;
      }
      assert(rank == sizePerThread.size() &&
             "unexpected rank in SliceEncodingAttr::getWarpIds");
      SmallVector<unsigned> parentElemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        parentElemsPerThread[i] =
            ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
      }
      DenseSet<unsigned> slicedAxies;
      for (unsigned i = 0; i < rank; ++i) {
        if (parentElemsPerThread[i] != outShape[i]) {
          slicedAxies.insert(i);
        }
      }
      SmallVector<unsigned> warpMods(warpsPerCTA.size());
      SmallVector<unsigned> warpStrides(warpsPerCTA.size());
      unsigned warpMod = 1;
      unsigned warpStride = 1;
      for (int i = warpsPerCTA.size() - 1; i >= 0; --i) {
        warpMod *= warpsPerCTA[i];
        warpMods[i] = warpMod;
        warpStrides[i] = warpStride;
        warpStride *= warpsPerCTA[i];
      }
      SmallVector<Value, 4> parentWarpIds;
      for (unsigned i = 0; i < rank; ++i) {
        if (slicedAxies.count(i)) {
          if (llvm::is_contained(sliceDims, i)) {
            llvm::report_fatal_error("[Error] bad slice layout shape");
            assert(false && "bad slice layout shape");
          }
          auto repeatNum = shapePerCTA[i] > warpsPerCTA[i]
                               ? 1
                               : warpsPerCTA[i] / shapePerCTA[i];
          auto id = builder.create<arith::DivSIOp>(
              loc,
              builder.create<arith::RemSIOp>(
                  loc, builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
                  builder.create<arith::ConstantIndexOp>(loc, warpMods[i] /
                                                                  repeatNum)),
              builder.create<arith::ConstantIndexOp>(loc, warpStrides[i]));
          warpIds.push_back(id);
        } else {
          if (!llvm::is_contained(sliceDims, i)) {
            warpIds.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
          }
        }
      }
    } else {
      for (unsigned i = 0; i < tType.getRank(); ++i) {
        warpIds.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
      }
    }
  } else {
    warpIds.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
  }
  return warpIds;
}

SmallVector<Value, 4> getElemsPerThread(OpBuilder &builder, Location loc,
                                        Type type) {
  SmallVector<Value, 4> numElems;
  if (auto ty = dyn_cast<RankedTensorType>(type)) {
    auto warpIds = getWarpIds(builder, loc, type);
    unsigned i = 0;
    for (auto num : triton::gcu::getElemsPerThread(type)) {
      auto dim = builder.create<arith::ConstantIndexOp>(loc, ty.getDimSize(i));
      auto slice = builder.create<arith::ConstantIndexOp>(loc, num);
      auto minNum = builder.create<arith::MinSIOp>(
          loc, slice,
          builder.create<arith::SubIOp>(
              loc, dim, builder.create<arith::MulIOp>(loc, warpIds[i], slice)));
      numElems.push_back(minNum);
      ++i;
    }
  } else {
    numElems.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));
  }
  return numElems;
}

void doSlicePadOrMemsetSlice(OpBuilder &rewriter, Location loc, Operation *op,
                             Value output, Value src,
                             SmallVector<Value, 4> &offsets,
                             SmallVector<Value, 4> &sliceShape,
                             SmallVector<Value, 4> &padSizes,
                             Value defaultValue, triton::gcu::TagInfo tag) {
  auto maxPadSize = rewriter.create<arith::ConstantIndexOp>(loc, 2047);
  auto outputType = dyn_cast<MemRefType>(output.getType());
  auto legalPad = rewriter.create<arith::ConstantIntOp>(loc, 1, 1).getResult();
  unsigned totalNumElems = 1;
  for (int i = 0; i < outputType.getRank(); i++) {
    totalNumElems *= outputType.getShape()[i];
    auto padSize = padSizes[i];
    auto legalPadRank = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, padSize, maxPadSize);
    legalPad = rewriter.create<arith::AndIOp>(loc, legalPad, legalPadRank);
  }
  rewriter.create<scf::IfOp>(
      loc, legalPad,
      [&](OpBuilder &builder, Location loc) {
        builder.create<memref_ext::SlicePadStartOp>(
            loc, output, src, offsets, sliceShape, defaultValue, tag.getTag(),
            ValueRange{tag.getIdx()});
        builder.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &childBuilder, Location loc) {
        doMemset(childBuilder, tag, op, output, defaultValue, totalNumElems);
        auto rank = outputType.getRank();
        SmallVector<Value, 4> dstOffsets;
        auto zeroI32 = childBuilder.create<arith::ConstantIntOp>(loc, 0, 32);
        for (int i = 0; i < rank; i++) {
          dstOffsets.push_back(zeroI32);
        }
        childBuilder.create<memref_ext::SliceDesliceStartOp>(
            loc, output, src, offsets, sliceShape, dstOffsets, tag.getTag(),
            ValueRange{tag.getIdx()});
        childBuilder.create<scf::YieldOp>(loc);
      });
}

void doMemFence(OpBuilder &rewriter, Operation *op) { /*NOLINT*/
  rewriter.create<gcu::MFenceOp>(op->getLoc());
}

void doMemsetConfig(OpBuilder &rewriter, Location loc, Value output, Value v,
                    triton::gcu::TagInfo tag) {
  rewriter.create<memref_ext::MemsetStartOp>(loc, output, v, tag.getTag(),
                                             ValueRange{tag.getIdx()});
}

void doMemset(OpBuilder &rewriter, triton::gcu::TagInfo tag, Operation *op,
              Value output, Value v, int totalNumElems) {
  auto loc = op->getLoc();
  if (totalNumElems > 128 || totalNumElems <= 0) {
    auto outType = dyn_cast<MemRefType>(output.getType());
    bool isNeedMerge = outType.getRank() > 5;
    Value outbuffer = output;
    if (isNeedMerge && totalNumElems > 128) {
      outbuffer = rewriter.create<memref::ReinterpretCastOp>(
          loc,
          MemRefType::get(ArrayRef<int64_t>{totalNumElems},
                          outType.getElementType()),
          output, 0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1});
    }
    rewriter.create<memref_ext::MemsetStartOp>(loc, outbuffer, v, tag.getTag(),
                                               ValueRange{tag.getIdx()});
    if (v.getType() != rewriter.getI64Type()) {
      rewriter.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
    }
  } else {
    auto type = dyn_cast<MemRefType>(output.getType());
    affine::buildAffineLoopNest(
        rewriter, loc, SmallVector<int64_t, 4>(type.getRank(), 0),
        type.getShape(), SmallVector<int64_t, 4>(type.getRank(), 1),
        [&](OpBuilder &builder, Location loc, ValueRange iters) {
          builder.create<memref::StoreOp>(loc, v, output, iters);
        });
    doMemFence(rewriter, op);
  }
}

bool canReuseAccumulatorBuffer(triton::DotOp op, int32_t number) {
  if (!op->hasAttr("acc_reuse_candidate"))
    return false;

  if (op.getType().getRank() != 2)
    return false;

  if (number != 2)
    return false;

  Value origAcc = op.getC();
  auto accType = dyn_cast<RankedTensorType>(origAcc.getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getType());
  if (!accType || !resultType)
    return false;

  auto blockArg = dyn_cast<BlockArgument>(origAcc);
  if (!blockArg)
    return false;

  auto *parentOp = blockArg.getOwner()->getParentOp();
  auto forOp = dyn_cast<scf::ForOp>(parentOp);
  if (!forOp)
    return false;

  unsigned argIdx = blockArg.getArgNumber();
  if (argIdx == 0)
    return false;
  unsigned iterArgIdx = argIdx - 1;

  if (!origAcc.hasOneUse())
    return false;

  auto *terminator = forOp.getBody()->getTerminator();
  auto yieldOp = dyn_cast<scf::YieldOp>(terminator);
  if (!yieldOp || iterArgIdx >= yieldOp.getNumOperands())
    return false;
  if (yieldOp.getOperand(iterArgIdx) != op.getResult())
    return false;

  Value initArg = forOp.getInitArgs()[iterArgIdx];
  if (auto outerFor = forOp->getParentOfType<scf::ForOp>()) {
    // Reject if nested three or more levels deep -- only two-level
    // nesting is supported for accumulator reuse.
    if (outerFor->getParentOfType<scf::ForOp>())
      return false;
    // Reject if init comes from outer for's block arg (carried through
    // the outer loop).
    if (auto outerBlockArg = dyn_cast<BlockArgument>(initArg))
      if (outerBlockArg.getOwner()->getParentOp() == outerFor.getOperation())
        return false;
    // Reject if outer for yields the inner for's accumulator result.
    if (auto outerYield =
            dyn_cast<scf::YieldOp>(outerFor.getBody()->getTerminator())) {
      Value innerResult = forOp->getResult(iterArgIdx);
      for (auto yieldedVal : outerYield.getOperands())
        if (yieldedVal == innerResult)
          return false;
    }
  }
  return true;
}

bool isMustAliasOp(OpOperand &use) {
  Operation *op = use.getOwner();
  if (llvm::isa<triton::PtrToIntOp, triton::IntToPtrOp, triton::BitcastOp,
                gcu::PtrToMemRefOp, gcu::MemRefToPtrOp, gcu::IntToPtrOp,
                gcu::PtrToIntOp, triton::gcu::MemDescToPtrOp>(op)) {
    // memdesc_to_ptr only extracts the base address of a shared-memory
    // memdesc, so the result still aliases the source storage.
    return true;
  } else if (llvm::isa<triton::AddPtrOp>(op)) {
    // tt.addptr only aliases the source buffer through its pointer operand
    // (operand 0); the offset operand (operand 1) is consumed to compute the
    // new address and must not be treated as aliasing.
    return use.getOperandNumber() == 0;
  } else if (llvm::isa<triton::gpu::MemDescTransOp,
                       triton::gpu::MemDescSubsliceOp,
                       triton::gpu::MemDescReshapeOp,
                       triton::gpu::MemDescReinterpretOp,
                       triton::gpu::MemDescIndexOp>(op)) {
    // ttg.memdesc_* view ops only create a new descriptor over the same
    // shared-memory buffer, so the result still aliases the source storage.
    return true;
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
    // tt.splat of a pointer only replicates the base address (view), while
    // tt.splat of a scalar broadcasts data (copy). Only the former aliases.
    return isa<triton::PointerType>(splatOp.getSrc().getType());
  } else if (llvm::isa<triton::gpu::ConvertLayoutOp>(op)) {
    auto convertLayout = cast<triton::gpu::ConvertLayoutOp>(op);
    auto src = convertLayout.getSrc();
    auto srcNumElems = triton::gcu::getElemsPerThread(src.getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(convertLayout.getType());

    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    auto dstTy = dyn_cast<RankedTensorType>(convertLayout.getType());
    if ((!srcTy) || (!dstTy)) {
      assert(false && "srcTy or dstTy not a RankedTensorType");
    }

    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (srcLayout == dstLayout) {
      return true;
    }
    if (srcNumElems == dstNumElems &&
        src.getType().getShape() == convertLayout.getType().getShape()) {
      if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
          isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (cast<triton::gpu::SliceEncodingAttr>(srcLayout).getDim() ==
            cast<triton::gpu::SliceEncodingAttr>(dstLayout).getDim()) {
          return true;
        } else {
          return false;
        }
      } else {
        return true;
      }
    }
    // gcu400 dot use subview no data copy
    if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      return true;
    }
    // return canUseSubView(srcTy, dstTy);
    return false;
  } else if (isa<triton::ExpandDimsOp>(op)) {
    auto expandDimOp = cast<triton::ExpandDimsOp>(op);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(expandDimOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(expandDimOp.getType());
    srcNumElems.insert(srcNumElems.begin() + expandDimOp.getAxis(), 1);
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
    // Triton allows tt.reshape with allow_reorder to permute per-thread
    // elements. GCU does not implement that semantics yet, so must-alias here
    // only checks layout via isExpensiveView and ignores allow_reorder.
    return !triton::gcu::isExpensiveView(reshapeOp.getSrc().getType(),
                                         reshapeOp.getType());
  } else if (isa<triton::BroadcastOp>(op)) {
    auto broastOp = cast<triton::BroadcastOp>(op);
    // Broadcasting a pointer tensor only replicates addresses, so the result
    // still aliases the source storage (view). For data tensors fall through
    // to the layout-based check below.
    if (auto srcTy = dyn_cast<RankedTensorType>(broastOp.getSrc().getType())) {
      if (isa<triton::PointerType>(srcTy.getElementType()))
        return true;
    }
    auto srcNumElems =
        triton::gcu::getElemsPerThread(broastOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(broastOp.getType());
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else if (isa<triton::gpu::LocalLoadOp>(op)) {
    auto localLoadOp = cast<triton::gpu::LocalLoadOp>(op);
    auto srcType = localLoadOp.getSrc().getType();
    auto dstType = localLoadOp.getType();
    auto srcLayout = cast<triton::gpu::TensorOrMemDesc>(srcType).getEncoding();
    auto dstLayout = dyn_cast<RankedTensorType>(dstType).getEncoding();
    // share to Distributed
    if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      // copy to local
      ModuleOp builtinModule = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gcu::getNumWarps(builtinModule);
      if (numWarps > 1) {
        return canUseSubView(srcType, dstType);
      } else {
        return true;
      }
    } else if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      // subview for gcu 400
      return true;
    }
    return false;
  } else if (isa<triton::DotOp>(op)) {
    int32_t number = use.getOperandNumber();
    return canReuseAccumulatorBuffer(cast<triton::DotOp>(op), number);
  } else {
    return false;
  }
}
// Find last user which is located at parent region of the op
mlir::Operation *
promoteLastUser(std::pair<mlir::Operation *, int> &lastUser,
                triton::gcu::FirstLastUserAnalysis &userAnalysis,
                std::map<Operation *, Operation *> &replaced2Origin) {
  if (!isa_and_nonnull<scf::YieldOp>(lastUser.first)) {
    return nullptr;
  }

  mlir::Operation *newAllocOpPos = nullptr;
  std::pair<mlir::Operation *, int> curLastUser = lastUser;
  mlir::Operation *parent = curLastUser.first->getParentOp();
  mlir::Operation *originParent = nullptr;

  while (
      isa_and_nonnull<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
          parent) &&
      isa_and_nonnull<scf::YieldOp>(curLastUser.first)) {
    if (replaced2Origin.count(parent) == 0) {
      if (llvm::none_of(parent->getOperandTypes(),
                        [](auto t) { return isa<MemRefType>(t); }) &&
          llvm::none_of(parent->getResultTypes(),
                        [](auto t) { return isa<MemRefType>(t); })) {
        originParent = parent;
      } else {
        llvm_unreachable("can't find the origin op");
      }
    } else {
      originParent = replaced2Origin[parent];
    }
    // Need to be the replaced op
    newAllocOpPos = parent;
    curLastUser = userAnalysis.getLastUser(
        originParent->getResults()[curLastUser.second]);
    parent = curLastUser.first ? curLastUser.first->getParentOp() : nullptr;
  }

  lastUser = curLastUser;
  return newAllocOpPos;
}

void addDeallocAfterLastUser(OpBuilder &builder,
                             std::pair<Operation *, int> lastUser,
                             Value alloc) {
  if (lastUser.first == nullptr) {
    return;
  }
  if (isa<scf::YieldOp>(lastUser.first) ||
      isa<triton::ReduceReturnOp>(lastUser.first) ||
      isa<triton::ScanReturnOp>(lastUser.first) ||
      isa<func::ReturnOp>(lastUser.first) ||
      isa<mlir::triton::gcu::YieldOp>(lastUser.first)) {
    return;
  }
  if (isa<MemRefType>(alloc.getType())) {
    if (lastUser.first->mightHaveTrait<OpTrait::IsTerminator>()) {
      return;
    }
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(lastUser.first);
    builder.create<memref::DeallocOp>(lastUser.first->getLoc(), alloc);
    builder.restoreInsertionPoint(ip);
  }
  return;
}

MemRefType getMemRefTypeFromSharedMem(RankedTensorType tType, Type elemType) {
  auto ctx = tType.getContext();
  auto numElems = triton::gcu::getElemsPerThread(tType);
  auto smemSpace = IntegerAttr::get(IntegerType::get(ctx, 64), 2);
  auto tileShape = SmallVector<int64_t, 4>(numElems.begin(), numElems.end());
  auto rank = tileShape.size();
  auto srcShape = tType.getShape();
  if (rank == 0)
    return MemRefType::get(tileShape, elemType, AffineMap{}, smemSpace);

  SmallVector<int64_t> srcStrides(rank);
  srcStrides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i)
    srcStrides[i] = srcStrides[i + 1] * srcShape[i + 1];

  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, srcStrides);
  return MemRefType::get(tileShape, elemType, layout, smemSpace);
}

// lowering
Value syncAllocOp(OpBuilder &builder, Location &loc,
                  std::pair<Operation *, int> lastUser,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  MemRefType type, int64_t memoryAlignment) {
  auto newAllocOpPos = promoteLastUser(lastUser, userAnalysis, replaced2Origin);
  Value output;
  if (newAllocOpPos == nullptr) {
    auto allocOp = builder.create<memref::AllocOp>(loc, type);
    if (memoryAlignment != INVALID_ALIGNMENT) {
      allocOp->setAttr(kAlignment, builder.getI64IntegerAttr(memoryAlignment));
    }
    output = allocOp.getResult();
  } else {
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(newAllocOpPos);
    auto allocOp = builder.create<memref::AllocOp>(loc, type);
    if (memoryAlignment != INVALID_ALIGNMENT) {
      allocOp->setAttr(kAlignment, builder.getI64IntegerAttr(memoryAlignment));
    }
    output = allocOp.getResult();
    builder.restoreInsertionPoint(ip);
  }
  addDeallocAfterLastUser(builder, lastUser, output);
  return output;
}

void createPrintfOp(OpBuilder &rewriter, Location loc,
                    ::llvm::StringRef printOpPrefix, bool hex, Value value) {
  auto printSingleElement = [&](Value operand, size_t i, size_t n,
                                ValueRange iters) {
    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << printOpPrefix << ": ";
    if (n > 1)
      os << "(operand " << i << ") ";

    // format
    auto msg = TypeSwitch<Type, StringRef>(operand.getType())
                   .Case<mlir::triton::gcu::PtrType, IntegerType, IndexType>(
                       [&](auto ty) {
                         if (hex) {
                           os << "0x%x ";
                           return "0x%x ";
                         } else {
                           os << "%d ";
                           return "%d ";
                         }
                       })
                   .Default([&](auto ty) {
                     os << "%f ";
                     return "%f ";
                   });

    // value
    SmallVector<Value, 4> values;
    auto value = TypeSwitch<Type, Value>(operand.getType())
                     .Case<gcu::PtrType>([&](auto ty) {
                       return rewriter.create<gcu::PtrToIntOp>(loc, operand);
                     })
                     .Default([&](auto ty) { return operand; });
    values.push_back(value);

    if (!iters.empty()) {
      // idx format
      os << "(idx ";
      for (auto iter = iters.begin(); iter != iters.end(); ++iter) {
        if (iter != iters.begin())
          os << ", ";
        os << "%d";
      }
      os << ")";
      // idx value
      values.append(iters.begin(), iters.end());
    }
    os << "\n";

    if (!msg.empty())
      rewriter.create<gpu::PrintfOp>(loc, formatStr, ValueRange{values});
  };

  auto printOperand = [&](Value operand, size_t i, size_t n) {
    TypeSwitch<Type>(operand.getType())
        .Case<MemRefType>([&](auto ty) {
          affine::buildAffineLoopNest(
              rewriter, loc, SmallVector<int64_t, 4>(ty.getRank(), 0),
              ty.getShape(), SmallVector<int64_t, 4>(ty.getRank(), 1),
              [&](OpBuilder &builder, Location loc, ValueRange iters) {
                auto v = builder.create<memref::LoadOp>(loc, operand, iters);
                printSingleElement(v, i, n, iters);
              });
        })
        .Default([&](auto ty) { printSingleElement(operand, i, n, {}); });
  };

  printOperand(value, 0, 1);
}

void enterTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent) {
  if (DBG_TRITON_IR) {
    auto border =
        rewriter.create<gcu::TritonBorder>(ttParent->getLoc()).getOperation();
    border->setAttr("enter", ttParent->getName().getIdentifier());
  }
  return;
}

void leaveTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent) {
  if (DBG_TRITON_IR) {
    auto border =
        rewriter.create<gcu::TritonBorder>(ttParent->getLoc()).getOperation();
    border->setAttr("leave", ttParent->getName().getIdentifier());
  }
  return;
}

bool canUseSubView(Type srcType, Type dstType) {
  int64_t srcRank = 0;
  ArrayRef<int64_t> srcShape;
  if (!TypeSwitch<Type, bool>(srcType)
           .Case<MemRefType, triton::gpu::TensorOrMemDesc>([&](auto ty) {
             srcRank = ty.getRank();
             srcShape = ty.getShape();
             return true;
           })
           .Default([](auto) { return false; }))
    return false;

  auto dstTensorType = dyn_cast<RankedTensorType>(dstType);
  if (!dstTensorType || srcRank != dstTensorType.getRank()) {
    return false;
  }

  auto elemsPerThread = triton::gcu::getElemsPerThread(dstType);
  for (unsigned i = 1; i < srcRank; ++i) {
    if (srcShape[i] != elemsPerThread[i] ||
        dstTensorType.getShape()[i] != elemsPerThread[i]) {
      return false;
    }
  }
  return true;
}

void mergeContinuousDims(OpBuilder &subBuilder, Location loc,
                         Value &sharedMemref, Value &warpMemref,
                         SmallVector<Value, 4> &offsets,
                         SmallVector<Value, 4> &mergedOffsets,
                         MemRefType &sharedMemType, MemRefType &warpMemType,
                         Value &sharedBuffer, Value &warpOutput) {
  SmallVector<int64_t> mergedSharedMemShapes;
  SmallVector<int64_t> mergedWarpMemShapes;
  auto zeroI32 = subBuilder.create<arith::ConstantOp>(
      loc, subBuilder.getIntegerAttr(subBuilder.getIntegerType(32), 0));
  int64_t mergeShape = 1;
  for (int i = 0; i < sharedMemType.getRank(); i++) {
    if (sharedMemType.getShape()[i] != warpMemType.getShape()[i]) {
      if (i > 0 &&
          sharedMemType.getShape()[i - 1] == warpMemType.getShape()[i - 1]) {
        mergedSharedMemShapes.push_back(mergeShape);
        mergedWarpMemShapes.push_back(mergeShape);
        mergedOffsets.push_back(zeroI32);
      }
      mergedSharedMemShapes.push_back(sharedMemType.getShape()[i]);
      mergedWarpMemShapes.push_back(warpMemType.getShape()[i]);
      mergedOffsets.push_back(offsets[i]);
      mergeShape = 1;
    } else {
      if (i == sharedMemType.getRank() - 1) {
        mergedSharedMemShapes.push_back(sharedMemType.getShape()[i] *
                                        mergeShape);
        mergedWarpMemShapes.push_back(warpMemType.getShape()[i] * mergeShape);
        mergedOffsets.push_back(zeroI32);
      } else {
        mergeShape *= sharedMemType.getShape()[i];
      }
    }
  }
  auto mergedSharedMemType = MemRefType::get(
      mergedSharedMemShapes, sharedMemType.getElementType(), AffineMap{},
      subBuilder.getI64IntegerAttr(2) /*shared memory*/);
  auto mergedWarpMemType =
      MemRefType::get(mergedWarpMemShapes, warpMemType.getElementType());
  auto [sharedMemStrides, sharedMemOffset] =
      mergedSharedMemType.getStridesAndOffset();
  sharedMemref = subBuilder.create<memref::ReinterpretCastOp>(
      loc, mergedSharedMemType, sharedBuffer, sharedMemOffset,
      mergedSharedMemShapes, sharedMemStrides);
  auto [warpMemStrides, warpMemOffset] =
      mergedWarpMemType.getStridesAndOffset();
  warpMemref = subBuilder.create<memref::ReinterpretCastOp>(
      loc, mergedWarpMemType, warpOutput, warpMemOffset, mergedWarpMemShapes,
      warpMemStrides);
  return;
}

Value stripUnrealizedConversionCastOp(Value v) {
  auto defOp = v.getDefiningOp();
  if (auto ucc = dyn_cast_or_null<UnrealizedConversionCastOp>(defOp)) {
    if (ucc.getNumOperands() == 1 && ucc.getNumResults() == 1)
      return ucc.getOperand(0);
  }
  return v;
}

bool isContiguousRowMajor(MemRefType ty) {
  // Only the strides matter here: the offset is always foldable into the
  // base pointer (getMemRefBasePtr / DTE bufferPtr), so a non-zero offset
  // does not break contiguity.
  auto [strides, offset] = ty.getStridesAndOffset();
  (void)offset;
  int64_t expected = 1;
  for (int i = static_cast<int>(ty.getRank()) - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(strides[i]) || strides[i] != expected)
      return false;
    expected *= ty.getShape()[i];
  }
  return true;
}

bool isContiguousMemDescView(Value memdesc) {
  auto ty = dyn_cast<triton::gpu::MemDescType>(memdesc.getType());
  if (!ty)
    return false;
  Operation *def = memdesc.getDefiningOp();
  if (!def)
    return true;

  // Root allocation: a plain, contiguous SMEM buffer.
  if (isa<triton::gpu::LocalAllocOp>(def))
    return true;

  // A transpose is never linear.
  if (isa<triton::gpu::MemDescTransOp>(def))
    return false;

  // A reshape of a contiguous buffer only reinterprets the same linear range
  // with row-major strides, so it stays contiguous.
  if (auto reshape = dyn_cast<triton::gpu::MemDescReshapeOp>(def))
    return isContiguousMemDescView(reshape.getSrc());

  // Same for a pure type reinterpretation.
  if (auto reinterpret = dyn_cast<triton::gpu::MemDescReinterpretOp>(def))
    return isContiguousMemDescView(reinterpret.getSrc());

  // Peeling dim 0 keeps the trailing dims (and their strides) intact.
  if (auto index = dyn_cast<triton::gpu::MemDescIndexOp>(def))
    return isContiguousMemDescView(index.getSrc());

  // A remote (cluster-partner) memdesc describes the very same buffer layout in
  // another CTA's shared memory; only the address space differs.
  if (auto remote = dyn_cast<triton::gcu::RemoteMemDescOp>(def))
    return isContiguousMemDescView(remote.getSrc());

  if (auto sub = dyn_cast<triton::gpu::MemDescSubsliceOp>(def)) {
    auto srcTy = dyn_cast<triton::gpu::MemDescType>(sub.getSrc().getType());
    if (!srcTy || !isContiguousMemDescView(sub.getSrc()))
      return false;
    ArrayRef<int64_t> srcShape = srcTy.getShape();
    ArrayRef<int64_t> subShape = ty.getShape();
    if (subShape.size() != srcShape.size())
      return false;
    // A subslice keeps the source strides and only changes the base offset
    // (which is foldable into the destination pointer), so contiguity is a
    // property of the resulting SHAPE alone: with row-major source strides
    // `st[i] = prod(srcShape[i+1:])`, the view is row-major iff
    // `st[i] == prod(subShape[i+1:])` for every i, and since each extent only
    // shrinks that holds exactly when no dimension after dim 0 is reduced -
    // i.e. the slice may only peel the outermost dimension. Compare
    // isContiguousRowMajor() above, which makes the same check on the lowered
    // strided memref.
    for (int64_t i = 1, e = srcShape.size(); i < e; ++i)
      if (subShape[i] != srcShape[i])
        return false;
    return true;
  }

  return false;
}

Value getMemRefBasePtr(OpBuilder &rewriter, Value v) {
  auto loc = v.getLoc();
  auto memRefTy = cast<MemRefType>(v.getType());
  auto elemTy = memRefTy.getElementType();
  auto ptrTy = gcu::PtrType::get(rewriter.getContext(), elemTy);

  // Identity layout: the existing memref2ptr lowering returns element 0.
  if (memRefTy.getLayout().isIdentity())
    return rewriter.create<gcu::MemRefToPtrOp>(loc, ptrTy, v);

  Value alignedIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), v);
  Value elemOffset =
      rewriter.create<gcu::GetMemRefOffsetOp>(loc, v, ValueRange{});
  unsigned bpe = elemTy.getIntOrFloatBitWidth() / 8;
  Value byteOffset = rewriter.create<arith::MulIOp>(
      loc, elemOffset, rewriter.create<arith::ConstantIndexOp>(loc, bpe));
  Value byteAddr = rewriter.create<arith::AddIOp>(loc, alignedIdx, byteOffset);
  Value addrI64 =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), byteAddr);
  return rewriter.create<gcu::IntToPtrOp>(loc, ptrTy, addrI64);
}

Value loadFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<Operation *, int> lastTTUser,
                        std::pair<Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin,
                        bool useSubView) {
  auto loc = buffer.getLoc();
  auto srcType = dyn_cast<MemRefType>(buffer.getType());
  auto elemType = srcType.getElementType();
  auto numElems = triton::gcu::getElemsPerThread(type);
  auto totalNumElems = builder.create<arith::ConstantIndexOp>(
      loc, triton::gcu::getTotalElemsPerThread(type));
  auto outputType =
      MemRefType::get(SmallVector<int64_t>(numElems.begin(), numElems.end()),
                      srcType.getElementType());

  auto warpIds = getWarpIds(builder, loc, type);

  // Load data from shared memory directly without any data transfer
  if (useSubView && canUseSubView(srcType, type)) {
    auto srcStrides = srcType.getStridesAndOffset().first;
    Value offsetElems = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (unsigned i = 0; i < srcType.getRank(); ++i) {
      Value dimOffset = builder.create<arith::MulIOp>(
          loc,
          builder.create<arith::MulIOp>(
              loc, builder.create<arith::ConstantIndexOp>(loc, numElems[i]),
              warpIds[i]),
          builder.create<arith::ConstantIndexOp>(loc, srcStrides[i]));
      offsetElems = builder.create<arith::AddIOp>(loc, offsetElems, dimOffset);
    }

    // Convert element offset to byte offset
    auto bpe = triton::gcu::getBpe(elemType);
    Value offsetBytes = builder.create<arith::MulIOp>(
        loc, offsetElems, builder.create<arith::ConstantIndexOp>(loc, bpe));

    // Get SMEM base pointer and fold the view offset into the base address.
    // memref2ptr requires an identity layout, so a strided subview source
    // (e.g. memdesc_subslice with a non-zero offset) must go through
    // getMemRefBasePtr, whose strided branch folds the view offset.
    auto ptrType = gcu::PtrType::get(buffer.getContext(), elemType);
    Value basePtr = getMemRefBasePtr(builder, buffer);
    Value baseAddr =
        builder.create<gcu::PtrToIntOp>(loc, builder.getI64Type(), basePtr);
    Value offsetBytesI64 = builder.create<arith::IndexCastOp>(
        loc, builder.getI64Type(), offsetBytes);
    Value adjustedAddr =
        builder.create<arith::AddIOp>(loc, baseAddr, offsetBytesI64);
    Value adjustedPtr =
        builder.create<gcu::IntToPtrOp>(loc, ptrType, adjustedAddr);

    // Wrap pointer into a 1D i8 memref
    MemRefType memType1D =
        MemRefType::get({ShapedType::kDynamic}, builder.getI8Type());
    auto buffer1D =
        builder.create<gcu::PtrToMemRefOp>(loc, memType1D, adjustedPtr);

    // Change to the result memory space (metadata only, address preserved)
    auto bufferWithSpace = builder.create<memref::MemorySpaceCastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                        AffineMap{}, outputType.getMemorySpace()),
        buffer1D);

    // Final view with offset 0 (offset already folded into base address)
    return builder.create<memref::ViewOp>(
        loc, outputType, bufferWithSpace,
        builder.create<arith::ConstantIndexOp>(loc, 0), ValueRange{});
  }

  SmallVector<Value, 4> offsets;
  for (unsigned i = 0; i < srcType.getRank(); ++i) {
    offsets.push_back(builder.create<arith::MulIOp>(
        loc, builder.create<arith::ConstantIntOp>(loc, numElems[i], 32),
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                           warpIds[i])));
  }

  auto output = syncAllocOp(builder, loc, lastTTUser, userAnalysis,
                            replaced2Origin, outputType);

  auto masterWarpId = getMasterThreadId(buffer.getParentRegion());
  auto isMasterThread = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      builder.create<arith::ConstantIndexOp>(loc, masterWarpId));
  auto defaultValue =
      triton::gcu::createConstantZero(builder, loc, srcType.getElementType());
  bool isNeedMerge = srcType.getRank() > 5;
  SmallVector<Value, 4> mergedOffsets;
  Value src;
  Value dst;
  if (!firstTTUser.first || !tag.isAsync()) {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            if (isNeedMerge) {
              mergeContinuousDims(builder, loc, src, dst, offsets,
                                  mergedOffsets, srcType, outputType, buffer,
                                  output);
              builder.create<memref_ext::SliceStartOp>(
                  loc, dst, src, mergedOffsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(
                  loc, output, buffer, offsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
            }
            builder.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      if (isNeedMerge) {
        mergeContinuousDims(builder, loc, src, dst, offsets, mergedOffsets,
                            srcType, outputType, buffer, output);
        builder.create<memref_ext::SliceStartOp>(loc, dst, src, mergedOffsets,
                                                 defaultValue, tag.getTag(),
                                                 ValueRange{tag.getIdx()});
        auto [oriOutputStrides, oriOutputOffset] =
            outputType.getStridesAndOffset();
        builder.create<memref::ReinterpretCastOp>(
            loc, outputType, dst, oriOutputOffset,
            SmallVector<int64_t>(numElems.begin(), numElems.end()),
            oriOutputStrides);
      } else {
        builder.create<memref_ext::SliceStartOp>(loc, output, buffer, offsets,
                                                 defaultValue, tag.getTag(),
                                                 ValueRange{tag.getIdx()});
      }
      builder.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
    }
  } else {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            if (isNeedMerge) {
              mergeContinuousDims(builder, loc, src, dst, offsets,
                                  mergedOffsets, srcType, outputType, buffer,
                                  output);
              builder.create<memref_ext::SliceStartOp>(
                  loc, dst, src, mergedOffsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(
                  loc, output, buffer, offsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
            }
            builder.create<scf::YieldOp>(loc);
          });
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser.first);
      builder.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
      builder.restoreInsertionPoint(ip);
    } else {
      if (isNeedMerge) {
        mergeContinuousDims(builder, loc, src, dst, offsets, mergedOffsets,
                            srcType, outputType, buffer, output);
        builder.create<memref_ext::SliceStartOp>(loc, dst, src, mergedOffsets,
                                                 defaultValue, tag.getTag(),
                                                 ValueRange{tag.getIdx()});
        auto [oriOutputStrides, oriOutputOffset] =
            outputType.getStridesAndOffset();
        builder.create<memref::ReinterpretCastOp>(
            loc, outputType, dst, oriOutputOffset,
            SmallVector<int64_t>(numElems.begin(), numElems.end()),
            oriOutputStrides);
      } else {
        builder.create<memref_ext::SliceStartOp>(loc, output, buffer, offsets,
                                                 defaultValue, tag.getTag(),
                                                 ValueRange{tag.getIdx()});
      }
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser.first);
      builder.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
      builder.restoreInsertionPoint(ip);
    }
  }
  return output;
}

Value CopyFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<Operation *, int> lastTTUser,
                        std::pair<Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin) {
  auto loc = buffer.getLoc();
  auto srcType = dyn_cast<MemRefType>(buffer.getType());
  auto shape = srcType.getShape();
  auto numElems = triton::gcu::getElemsPerThread(type);
  auto totalNumElems = builder.create<arith::ConstantIndexOp>(
      loc, triton::gcu::getTotalElemsPerThread(type));
  auto outputType =
      MemRefType::get(SmallVector<int64_t>(numElems.begin(), numElems.end()),
                      srcType.getElementType());

  auto output = syncAllocOp(builder, loc, lastTTUser, userAnalysis,
                            replaced2Origin, outputType);
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto masterWarpId = getMasterThreadId(buffer.getParentRegion());
  auto isMasterThread = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      builder.create<arith::ConstantIndexOp>(loc, masterWarpId));
  if (!firstTTUser.first || !tag.isAsync()) {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaStartOp>(
                loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
                SmallVector<Value, 4>(shape.size(), zero), totalNumElems,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      builder.create<memref::DmaStartOp>(
          loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
          SmallVector<Value, 4>(shape.size(), zero), totalNumElems,
          tag.getTag(), ValueRange{tag.getIdx()});
      builder.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
    }
  } else {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaStartOp>(
                loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
                SmallVector<Value, 4>(shape.size(), zero), totalNumElems,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          });
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser.first);
      builder.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
      builder.restoreInsertionPoint(ip);
    } else {
      builder.create<memref::DmaStartOp>(
          loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
          SmallVector<Value, 4>(shape.size(), zero), totalNumElems,
          tag.getTag(), ValueRange{tag.getIdx()});
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser.first);
      builder.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
      builder.restoreInsertionPoint(ip);
    }
  }
  return output;
}

void storeToSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag,
                      TensorType type, Value sharedBuffer, Value buffer,
                      bool onlyThread0) {
  auto loc = buffer.getLoc();
  auto srcType = dyn_cast<MemRefType>(buffer.getType());
  auto outputType = dyn_cast<MemRefType>(sharedBuffer.getType());
  auto totalNumElems = builder.create<arith::ConstantIndexOp>(
      loc, triton::gcu::getTotalElemsPerThread(type));

  SmallVector<Value, 4> offsets;
  SmallVector<int64_t> outputSize;
  auto warpIds = getWarpIds(builder, loc, type);
  for (unsigned i = 0; i < srcType.getRank(); ++i) {
    offsets.push_back(builder.create<arith::MulIOp>(
        loc,
        builder.create<arith::ConstantIntOp>(loc, srcType.getDimSize(i), 32),
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                           warpIds[i])));
    outputSize.push_back(outputType.getShape()[i]);
  }
  auto masterWarpId = getMasterThreadId(buffer.getParentRegion());
  auto isMasterThread = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      builder.create<arith::ConstantIndexOp>(loc, masterWarpId));
  bool isNeedMerge = srcType.getRank() > 5;
  SmallVector<Value, 4> mergedOffsets;
  Value src;
  Value dst;
  if (onlyThread0) {
    builder.create<scf::IfOp>(
        loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
          if (isNeedMerge) {
            mergeContinuousDims(builder, loc, dst, src, offsets, mergedOffsets,
                                outputType, srcType, sharedBuffer, buffer);
            auto [oriOutputStrides, oriOutputOffset] =
                outputType.getStridesAndOffset();
            builder.create<memref_ext::DesliceStartOp>(
                loc, dst, src, mergedOffsets, tag.getTag(),
                ValueRange{tag.getIdx()});
            builder.create<memref::ReinterpretCastOp>(
                loc, outputType, dst, oriOutputOffset, outputSize,
                oriOutputStrides);
          } else {
            builder.create<memref_ext::DesliceStartOp>(
                loc, sharedBuffer, buffer, offsets, tag.getTag(),
                ValueRange{tag.getIdx()});
          }
          builder.create<memref::DmaWaitOp>(
              loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
          builder.create<scf::YieldOp>(loc);
        });
  } else {
    if (isNeedMerge) {
      mergeContinuousDims(builder, loc, dst, src, offsets, mergedOffsets,
                          outputType, srcType, sharedBuffer, buffer);
      auto [oriOutputStrides, oriOutputOffset] =
          outputType.getStridesAndOffset();
      builder.create<memref_ext::DesliceStartOp>(
          loc, dst, src, mergedOffsets, tag.getTag(), ValueRange{tag.getIdx()});
      builder.create<memref::ReinterpretCastOp>(
          loc, outputType, dst, oriOutputOffset, outputSize, oriOutputStrides);
    } else {
      builder.create<memref_ext::DesliceStartOp>(loc, sharedBuffer, buffer,
                                                 offsets, tag.getTag(),
                                                 ValueRange{tag.getIdx()});
    }
    builder.create<memref::DmaWaitOp>(loc, tag.getTag(),
                                      ValueRange{tag.getIdx()}, totalNumElems);
  }
  builder.create<gpu::BarrierOp>(loc);
}

Value storeToSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag,
                       TensorType type, Value buffer, bool onlyThread0,
                       std::pair<Operation *, int> lastTTUser,
                       triton::gcu::FirstLastUserAnalysis &userAnalysis,
                       std::map<Operation *, Operation *> &replaced2Origin) {
  auto loc = buffer.getLoc();
  auto mergedType = MemRefType::get(
      type.getShape(), dyn_cast<MemRefType>(buffer.getType()).getElementType(),
      AffineMap{},
      builder.getI64IntegerAttr(2)); // shared memory

  auto merged = syncAllocOp(builder, loc, lastTTUser, userAnalysis,
                            replaced2Origin, mergedType);
  storeToSharedMem(builder, tag, type, merged, buffer, onlyThread0);
  return merged;
}

// Emit a vector load/store for a flat per-thread run (the last dim of a
// per-thread block). The GCU400 store granularity is kOaccSizeInBytes (512B):
// a run at least that large uses a plain vector load/store; a smaller run is
// masked on a kOaccSizeInBytes-wide vector with a partial mask.
static Value emitFlatVectorLoad(OpBuilder &builder, Location loc, Type elemType,
                                Value memref, Value offset,
                                unsigned flatElems) {
  unsigned bpe = triton::gcu::getBpe(elemType);
  unsigned padN = kOaccSizeInBytes / bpe;
  if (flatElems * bpe >= kOaccSizeInBytes) {
    auto vecTy = VectorType::get({flatElems}, elemType);
    return builder
        .create<vector::LoadOp>(loc, vecTy, memref, ValueRange{offset})
        .getResult();
  }
  auto vecTy = VectorType::get({padN}, elemType);
  auto mask = builder.create<vector::ConstantMaskOp>(
      loc, VectorType::get({padN}, builder.getI1Type()),
      DenseI64ArrayAttr::get(builder.getContext(),
                             ArrayRef<int64_t>{flatElems}));
  auto padding = builder.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(vecTy, builder.getZeroAttr(elemType)));
  return builder
      .create<vector::MaskedLoadOp>(loc, vecTy, memref, ValueRange{offset},
                                    mask, padding)
      .getResult();
}

static void emitFlatVectorStore(OpBuilder &builder, Location loc, Type elemType,
                                Value memref, Value offset, Value val,
                                unsigned flatElems) {
  unsigned bpe = triton::gcu::getBpe(elemType);
  if (flatElems * bpe >= kOaccSizeInBytes) {
    builder.create<vector::StoreOp>(loc, val, memref, ValueRange{offset});
    return;
  }
  unsigned padN = kOaccSizeInBytes / bpe;
  auto mask = builder.create<vector::ConstantMaskOp>(
      loc, VectorType::get({padN}, builder.getI1Type()),
      DenseI64ArrayAttr::get(builder.getContext(),
                             ArrayRef<int64_t>{flatElems}));
  builder.create<vector::MaskedStoreOp>(loc, memref, ValueRange{offset}, mask,
                                        val);
}

template <typename T> T static getProduct(ArrayRef<T> ins) {
  return std::accumulate(ins.begin(), ins.end(), T{1}, std::multiplies<T>());
}

template <typename T>
static SmallVector<T> getStrides(ArrayRef<T> shape, T start = 1) {
  int rank = shape.size();
  SmallVector<T> strides(rank);
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = start;
    start *= shape[i];
  }
  return strides;
}

template <typename T>
static SmallVector<T> getStrides(SmallVector<T> &shape, T start = 1) {
  return getStrides(ArrayRef<T>(shape), start);
}

// Emit one vector load/store per leading-dim grid point of the per-thread
// block. The block is [n0..n_{N-1}]; the last dim (n_{N-1}) is the vector
// width, the leading dims are nested loops (unrolled at build time). `privBuf`
// is the flattened private buffer [prod(n_i)] in row-major; `sharedBuf` is the
// flat shared tensor [prod(shape_i)]; `sharedBaseOffset` is the thread's base
// flat offset in the shared buffer. `isLoad=false` moves privBuf -> shared
// (store), `isLoad=true` moves shared -> privBuf.
static void emitChunkVectorOps(OpBuilder &builder, Location loc, Type elemType,
                               Value privBuf, Value sharedBuf,
                               Value sharedBaseOffset,
                               ArrayRef<unsigned> leadingElems,
                               ArrayRef<int64_t> sharedStrides,
                               unsigned lastWidth, bool isLoad) {
  unsigned leadingTotal = getProduct(leadingElems);
  SmallVector<unsigned> privStrides = getStrides(leadingElems, lastWidth);
  for (unsigned idx = 0; idx < leadingTotal; ++idx) {
    unsigned tmp = idx;
    unsigned privOff = 0;
    unsigned sharedOff = 0;
    for (unsigned d = 0; d < leadingElems.size(); ++d) {
      unsigned c = tmp % leadingElems[d];
      tmp /= leadingElems[d];
      privOff += c * privStrides[d];
      sharedOff += c * sharedStrides[d];
    }
    auto sharedIndex = builder.create<arith::AddIOp>(
        loc, sharedBaseOffset,
        builder.create<arith::ConstantIndexOp>(loc, sharedOff));
    auto privIndex = builder.create<arith::ConstantIndexOp>(loc, privOff);
    if (isLoad) {
      auto vec = emitFlatVectorLoad(builder, loc, elemType, sharedBuf,
                                    sharedIndex, lastWidth);
      emitFlatVectorStore(builder, loc, elemType, privBuf, privIndex, vec,
                          lastWidth);
    } else {
      auto vec = emitFlatVectorLoad(builder, loc, elemType, privBuf, privIndex,
                                    lastWidth);
      emitFlatVectorStore(builder, loc, elemType, sharedBuf, sharedIndex, vec,
                          lastWidth);
    }
  }
}

// Compute the per-thread block dims, the full tensor shape, and the
// thread's base flat offset in the shared buffer, for any rank.
std::tuple<Value, SmallVector<unsigned>,
           SmallVector<int64_t>> static getChunkInfo(OpBuilder &builder,
                                                     Location loc,
                                                     RankedTensorType type) {
  auto warpIds = triton::gcu::getWarpIds(builder, loc, type);
  auto numElems = triton::gcu::getElemsPerThread(type);
  auto shape = type.getShape();
  unsigned rank = type.getRank();
  SmallVector<int64_t> flatStrides = getStrides(shape);

  Value base = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i < rank; ++i) {
    auto term = builder.create<arith::MulIOp>(
        loc,
        builder.create<arith::MulIOp>(
            loc, builder.create<arith::ConstantIndexOp>(loc, numElems[i]),
            warpIds[i]),
        builder.create<arith::ConstantIndexOp>(loc, flatStrides[i]));
    base = builder.create<arith::AddIOp>(loc, base, term);
  }
  return {base, numElems, SmallVector<int64_t>(shape.begin(), shape.end())};
}

// Vector-store a private per-thread chunk (block dims `elemsPerThread` of the
// full tensor `tensorShape`) into the shared staging buffer, then barrier.
// Handles arbitrary rank: the last dim is the vector width, the leading dims
// are moved per-point (strided when the warp split is not only on dim 0).
void storeToSharedMemVector(OpBuilder &builder, Location loc, Value srcBuf,
                            RankedTensorType srcType, Value sharedBuf) {
  auto dstType = dyn_cast<MemRefType>(sharedBuf.getType());
  if (!dstType) {
    llvm_unreachable("dst type cannot be cast to MemRefType");
    return;
  }

  // Source memref
  auto [sharedBaseOffset, elemsPerThread, srcShape] =
      getChunkInfo(builder, loc, srcType);
  auto elemType = srcType.getElementType();

  unsigned total = getProduct(ArrayRef<unsigned>(elemsPerThread));
  auto flatSrcType = MemRefType::get({total}, elemType);
  auto src1D = builder.create<memref::ReinterpretCastOp>(
      loc, flatSrcType, srcBuf, /*offset=*/0, ArrayRef<int64_t>{total},
      ArrayRef<int64_t>{1});

  // Dst memref
  Value sharedBuf1D = sharedBuf;
  if (dstType.getRank() != 1) {
    auto dstTotal = dstType.getNumElements();
    auto sharedType = MemRefType::get({dstTotal}, dstType.getElementType(),
                                      AffineMap{}, dstType.getMemorySpace());
    sharedBuf1D = builder.create<memref::ReinterpretCastOp>(
        loc, sharedType, sharedBuf, /*offset*/ 0, dstTotal, /*stride*/ 1);
  }

  unsigned rank = elemsPerThread.size();
  unsigned lastWidth = elemsPerThread[rank - 1];
  SmallVector<int64_t> flatStrides = dstType.getRank() == srcType.getRank()
                                         ? dstType.getStridesAndOffset().first
                                         : getStrides(srcShape);
  emitChunkVectorOps(builder, loc, elemType, src1D, sharedBuf1D,
                     sharedBaseOffset,
                     ArrayRef<unsigned>(elemsPerThread).drop_back(),
                     ArrayRef<int64_t>(flatStrides).drop_back(), lastWidth,
                     /*isLoad=*/false);
  builder.create<gpu::BarrierOp>(loc);
}

Value storeToSharedMemVector(
    OpBuilder &builder, Value srcBuf, RankedTensorType srcType,
    std::pair<Operation *, int> lastUser,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin) {
  // Shared staging buffer.
  auto loc = srcBuf.getLoc();
  auto sharedType =
      MemRefType::get(srcType.getShape(), srcType.getElementType(), AffineMap{},
                      builder.getI64IntegerAttr(2));
  auto sharedBuffer = syncAllocOp(builder, loc, lastUser, userAnalysis,
                                  replaced2Origin, sharedType);

  // Each warp writes its private chunk into shared.
  storeToSharedMemVector(builder, loc, srcBuf, srcType, sharedBuffer);
  return sharedBuffer;
}

// Vector-load a private per-thread chunk (block dims `elemsPerThread` of the
// full tensor `tensorShape`) from the shared staging buffer and return it as a
// private [prod(elemsPerThread)] buffer. When `useSubView` is set and
// the chunk is a single contiguous alias of the shared buffer (only dim 0
// split across warps: elemsPerThread[i] == tensorShape[i] for i>=1), skip the
// data movement and return a shared-space view with the per-thread offset
// folded into the base address (mirrors TTLocalLoadOpLowering's
// onlyHighestDimSplit alias path).
Value loadFromSharedMemVector(
    OpBuilder &builder, Value sharedBuf, RankedTensorType dstType,
    MemRefType dstResTy, std::pair<Operation *, int> lastUser,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin, bool useSubView) {
  auto loc = sharedBuf.getLoc();
  Type elemType = dstType.getElementType();

  auto srcType = dyn_cast<MemRefType>(sharedBuf.getType());
  Value sharedBuf1D = sharedBuf;
  if (srcType && srcType.getRank() != 1) {
    auto srcTotal = srcType.getNumElements();
    auto sharedType = MemRefType::get({srcTotal}, srcType.getElementType(),
                                      AffineMap{}, srcType.getMemorySpace());
    sharedBuf1D = builder.create<memref::ReinterpretCastOp>(
        loc, sharedType, sharedBuf, /*offset*/ 0, srcTotal, /*stride*/ 1);
  }

  auto [sharedBaseOffset, elemsPerThread, dstShape] =
      getChunkInfo(builder, loc, dstType);
  if (useSubView && canUseSubView(srcType, dstType)) {
    unsigned bpe = triton::gcu::getBpe(elemType);
    auto ptrType = gcu::PtrType::get(builder.getContext(), elemType);
    Value basePtr =
        builder.create<gcu::MemRefToPtrOp>(loc, ptrType, sharedBuf1D);
    Value baseAddr =
        builder.create<gcu::PtrToIntOp>(loc, builder.getI64Type(), basePtr);
    Value offsetBytes = builder.create<arith::MulIOp>(
        loc, sharedBaseOffset,
        builder.create<arith::ConstantIndexOp>(loc, bpe));
    Value offsetBytesI64 = builder.create<arith::IndexCastOp>(
        loc, builder.getI64Type(), offsetBytes);
    Value adjustedAddr =
        builder.create<arith::AddIOp>(loc, baseAddr, offsetBytesI64);
    Value adjustedPtr =
        builder.create<gcu::IntToPtrOp>(loc, ptrType, adjustedAddr);
    auto buffer1D = builder.create<gcu::PtrToMemRefOp>(
        loc, MemRefType::get({ShapedType::kDynamic}, builder.getI8Type()),
        adjustedPtr);
    auto bufferWithSpace = builder.create<memref::MemorySpaceCastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                        AffineMap{}, srcType.getMemorySpace()),
        buffer1D);
    return builder.create<memref::ViewOp>(
        loc, dstResTy, bufferWithSpace,
        builder.create<arith::ConstantIndexOp>(loc, 0), ValueRange{});
  }

  auto dstBuf = syncAllocOp(builder, loc, lastUser, userAnalysis,
                            replaced2Origin, dstResTy);
  auto dstBuf1D = dstBuf;
  if (dstType.getRank() != 1) {
    auto dstTotal = dstType.getNumElements();
    auto flatDstType = MemRefType::get({dstTotal}, dstType.getElementType());
    dstBuf1D = builder.create<memref::ReinterpretCastOp>(
        loc, flatDstType, dstBuf, /*offset*/ 0, dstTotal, /*stride*/ 1);
  }

  unsigned rank = elemsPerThread.size();
  unsigned lastWidth = elemsPerThread[rank - 1];
  SmallVector<int64_t> flatStrides = getStrides(dstShape);
  emitChunkVectorOps(builder, loc, elemType, dstBuf1D, sharedBuf1D,
                     sharedBaseOffset,
                     ArrayRef<unsigned>(elemsPerThread).drop_back(),
                     ArrayRef<int64_t>(flatStrides).drop_back(), lastWidth,
                     /*isLoad=*/true);
  return dstBuf;
}

// refine yiled memref operand
void AnalysisYieldOperendUseStage(
    Operation *module, triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, std::map<uint64_t, bool>>
        &TTYeiledOPerandHasMultiUseStage) {
  module->walk<WalkOrder::PreOrder>([&](scf::YieldOp op) {
    if (isa<scf::ForOp, scf::WhileOp>(op.getOperation()->getParentOp())) {
      // check arg user
      for (uint64_t i = 0; i < op.getOperands().size(); ++i) {
        TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = false;
        auto operand = op.getOperands()[i];
        auto definingOp = operand.getDefiningOp();
        if (!definingOp) {
          TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
          continue;
        }
        if (isa<TensorType>(operand.getType()) &&
            isa<scf::ForOp>(op.getOperation()->getParentOp())) {
          auto forOp = llvm::cast<scf::ForOp>(op.getOperation()->getParentOp());
          auto reginArg = forOp.getRegionIterArgs()[i];
          auto lastUser =
              userAnalysis.getLastUser(reginArg, definingOp->getParentRegion());
          if (lastUser.first == nullptr) {
            TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
          } else {
            bool isMustAlias = false;
            for (auto &use : reginArg.getUses()) {
              if (use.getOwner() == definingOp) {
                if (isMustAliasOp(use))
                  isMustAlias = true;
                break;
              }
            }
            if (!isMustAlias && definingOp->isBeforeInBlock(lastUser.first)) {
              TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
            }
          }
        } else if (isa<TensorType>(operand.getType()) &&
                   isa<scf::WhileOp>(op.getOperation()->getParentOp())) {
          auto whileOp =
              llvm::cast<scf::WhileOp>(op.getOperation()->getParentOp());
          auto reginArg = whileOp.getAfterArguments()[i];
          auto lastUser =
              userAnalysis.getLastUser(reginArg, definingOp->getParentRegion());
          if (lastUser.first == nullptr) {
            TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
          } else {
            if (definingOp->isBeforeInBlock(lastUser.first)) {
              TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
            }
          }
        }
      }
    }
  });
}

void GetOrderValueByStride(
    OpBuilder &rewriter, Location loc, SmallVector<unsigned> nInitStrideDims,
    SmallVector<Value, 4> &initStride, SmallVector<Value, 4> &initShape,
    SmallVector<Value, 4> &initOffset, SmallVector<Value, 4> &orderStride,
    SmallVector<Value, 4> &orderShape, SmallVector<Value, 4> &orderOffset,
    SmallVector<Value, 4> &vOrder) {
  int64_t rank = static_cast<int64_t>(nInitStrideDims.size());

  SmallVector<Value, 4> tmpStrideBuffer = initStride;
  SmallVector<Value, 4> tmpShapeBuffer = initShape;
  SmallVector<Value, 4> tmpOffsetBuffer = initOffset;
  SmallVector<Value, 4> tmpOrderBuffer;
  for (int64_t i = 0; i < rank; ++i) {
    Value stride =
        rewriter.create<arith::ConstantIntOp>(loc, nInitStrideDims[i], 32);
    tmpOrderBuffer.push_back(stride);
  }

  for (int64_t i = 0; i < rank - 1; ++i) {
    for (int64_t j = 0; j < rank - 1 - i; j++) {
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 tmpStrideBuffer[j],
                                                 tmpStrideBuffer[j + 1]);

      Value tmpStride = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpStrideBuffer[j], tmpStrideBuffer[j + 1]);
      tmpStrideBuffer[j] = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpStrideBuffer[j + 1], tmpStrideBuffer[j]);
      tmpStrideBuffer[j + 1] = tmpStride;

      Value tmpShape = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpShapeBuffer[j], tmpShapeBuffer[j + 1]);
      tmpShapeBuffer[j] = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpShapeBuffer[j + 1], tmpShapeBuffer[j]);
      tmpShapeBuffer[j + 1] = tmpShape;

      Value tmpOffset = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpOffsetBuffer[j], tmpOffsetBuffer[j + 1]);
      tmpOffsetBuffer[j] = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpOffsetBuffer[j + 1], tmpOffsetBuffer[j]);
      tmpOffsetBuffer[j + 1] = tmpOffset;

      Value tmpOrder = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpOrderBuffer[j], tmpOrderBuffer[j + 1]);
      tmpOrderBuffer[j] = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpOrderBuffer[j + 1], tmpOrderBuffer[j]);
      tmpOrderBuffer[j + 1] = tmpOrder;
    }
  }

  vOrder = tmpOrderBuffer;
  orderStride = tmpStrideBuffer;
  for (int64_t i = 0; i < rank; ++i) {
    orderOffset.push_back(rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), tmpOffsetBuffer[i]));
  }

  orderShape.push_back(tmpShapeBuffer[0]);
  for (int64_t i = 0; i < rank - 1; ++i) {
    orderShape.push_back(rewriter.create<arith::DivSIOp>(loc, orderStride[i],
                                                         orderStride[i + 1]));
  }
}

void GetTransByOrder(OpBuilder &rewriter, Location loc,
                     SmallVector<Value, 4> &order,
                     SmallVector<Value, 4> &transOrder) {
  unsigned rank = order.size();

  SmallVector<Value, 4> tmpOrderBuffer = order;
  SmallVector<Value, 4> tmpTransOrderBuffer;
  for (unsigned i = 0; i < rank; ++i) {
    Value idx = rewriter.create<arith::ConstantIntOp>(loc, i, 32);
    tmpTransOrderBuffer.push_back(idx);
  }

  for (unsigned i = 0; i < rank - 1; ++i) {
    for (unsigned j = 0; j < rank - 1 - i; j++) {
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 tmpOrderBuffer[j],
                                                 tmpOrderBuffer[j + 1]);

      Value tmpOrder = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpOrderBuffer[j], tmpOrderBuffer[j + 1]);
      tmpOrderBuffer[j] = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpOrderBuffer[j + 1], tmpOrderBuffer[j]);
      tmpOrderBuffer[j + 1] = tmpOrder;

      Value tmpTransOrder = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpTransOrderBuffer[j], tmpTransOrderBuffer[j + 1]);
      tmpTransOrderBuffer[j] = rewriter.create<arith::SelectOp>(
          loc, cmp, tmpTransOrderBuffer[j + 1], tmpTransOrderBuffer[j]);
      tmpTransOrderBuffer[j + 1] = tmpTransOrder;
    }
  }
  transOrder = tmpTransOrderBuffer;
}

/// True when `configOffsets` can be non-zero at runtime: either a constant
/// non-zero entry, or an entry the analysis could not fold to a constant.
static bool hasNonZeroOffset(mlir::ValueRange configOffsets, unsigned rank) {
  if (configOffsets.size() != rank)
    return false;
  return llvm::any_of(configOffsets, [](Value offset) {
    auto cst = getConstantIntValue(offset);
    return !cst || *cst != 0;
  });
}

bool hasOffsetsAreDst(Operation *op) {
  if (auto load = dyn_cast<triton::gcu::LoadOp>(op))
    return load.getOffsetsAreDst();
  if (auto copy = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(op))
    return copy.getOffsetsAreDst();
  return false;
}

/// True when the offsets of `op` are destination tile offsets: PtrAnalysis
/// folded the mask start into the base pointer (foldMaskStartIntoBase) and
/// exposed it through the offsets (`offsets_are_dst`), so the valid block has
/// to be written at the matching tile columns instead of at tile offset 0.
static bool isDstTileOffsetOp(Operation *op, mlir::ValueRange configOffsets,
                              unsigned rank) {
  return hasOffsetsAreDst(op) && hasNonZeroOffset(configOffsets, rank);
}

/// Destination (tile) offsets of the valid block: the load offsets are in the
/// original (source) dims, while the destination memref is indexed in memory
/// order, so map them through the static order permutation. Dims added by
/// ConfigGcuLoad itself (reshape / synthetic stride dim) have no offset and
/// stay at zero.
static SmallVector<Value, 4> getDstTileOffsets(OpBuilder &rewriter,
                                               Location loc,
                                               mlir::ValueRange configOffsets,
                                               ArrayRef<int32_t> order,
                                               unsigned rank, Value zero) {
  SmallVector<Value, 4> dstOffsets(rank);
  for (unsigned i = 0; i < rank; ++i) {
    Value offset = i < configOffsets.size() ? configOffsets[i] : zero;
    dstOffsets[order[i]] =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), offset);
  }
  return dstOffsets;
}

Value ConfigGcuLoad(OpBuilder &rewriter, Location loc, Value srcOut,
                    mlir::Operation *op, MemRefType resultType, Value loadPtr,
                    mlir::ValueRange configStrides,
                    mlir::ValueRange configShapes, Value defaultValue,
                    triton::gcu::TagInfo tag, bool IsShareOutput,
                    mlir::ValueRange configOffsets) {
  if ((!llvm::isa_and_nonnull<triton::gcu::LoadOp>(op)) &&
      (!llvm::isa_and_nonnull<triton::gcu::CopyGlobalToLocalOp>(op))) {
    assert(false && "please check IR ConfigGcuLoad got a bad input op");
  }
  auto getOrderHint = [](mlir::Operation *op) {
    if (llvm::isa_and_nonnull<triton::gcu::LoadOp>(op)) {
      auto load = llvm::cast<triton::gcu::LoadOp>(op);
      return load.getOrderHint();
    } else if (llvm::isa_and_nonnull<triton::gcu::CopyGlobalToLocalOp>(op)) {
      auto loadToShare = llvm::cast<triton::gcu::CopyGlobalToLocalOp>(op);
      return loadToShare.getOrderHint();
    } else {
      assert(false && "please check IR ConfigGcuLoad got a bad input op");
    }
    return llvm::ArrayRef<int32_t>();
  };
  auto getDefaultValue = [](mlir::Operation *op) {
    if (llvm::isa_and_nonnull<triton::gcu::LoadOp>(op)) {
      auto load = llvm::cast<triton::gcu::LoadOp>(op);
      return load.getDefaultValue();
    } else if (llvm::isa_and_nonnull<triton::gcu::CopyGlobalToLocalOp>(op)) {
      auto loadToShare = llvm::cast<triton::gcu::CopyGlobalToLocalOp>(op);
      return loadToShare.getDefaultValue();
    } else {
      assert(false && "please check IR ConfigGcuLoad got a bad input op");
    }
    return Value();
  };

  auto getBroadcastDims = [](mlir::Operation *op) -> llvm::ArrayRef<int32_t> {
    if (auto load = llvm::dyn_cast_if_present<triton::gcu::LoadOp>(op))
      return load.getBroadcastDims();
    if (auto copy =
            llvm::dyn_cast_if_present<triton::gcu::CopyGlobalToLocalOp>(op))
      return copy.getBroadcastDims();
    return {};
  };

  auto elemType = resultType.getElementType();
  int64_t rank = resultType.getRank();

  auto buffer = rewriter.create<gcu::PtrToMemRefOp>(
      loc, MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType),
      loadPtr);

  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  bool bDynamicStride = false;
  bool bStaticTranspose = false;
  bool bStaticBroadcast = false;
  bool bReshape = true;
  SmallVector<unsigned> updateStrideDims;
  SmallVector<unsigned> nInitStrideDims;
  auto hint = getOrderHint(op);
  int64_t hint_size = static_cast<int64_t>(hint.size());
  assert(hint_size == rank /* || hint_size == 0*/);
  SmallVector<int32_t> order_hint;
  for (unsigned i = 0; i < rank; ++i)
    if (hint_size == 0)
      order_hint.push_back(Dynamic_stride_symbol);
    else
      order_hint.push_back(hint[i]);

  auto broadcastDims = getBroadcastDims(op);
  bStaticBroadcast =
      llvm::any_of(broadcastDims, [](int32_t v) { return v != 0; });

  for (unsigned i = 0; i < rank; ++i) {
    if (order_hint[i] == Dynamic_stride_symbol) {
      bDynamicStride = true;
      if ((triton::gcu::get_bool_env("TRITON_GCU_DEBUG") ||
           triton::gcu::get_bool_env("TRITON_ENABLE_ASAN"))) {
        auto trueCondition = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, configStrides[i], zero);
        rewriter.create<triton::gcu::AssertOp>(
            loc, trueCondition,
            "Not Support dynamic stride is 0,"
            "please add tl.constexpr to stride arg in kernel args list",
            "", "", 0);
      }
    }
  }
  if (bDynamicStride && (triton::gcu::get_bool_env("TRITON_GCU_DEBUG") ||
                         triton::gcu::get_bool_env("TRITON_ENABLE_ASAN"))) {
    for (int i = 0; i < rank; ++i) {
      for (int j = i + 1; j < rank; ++j) {
        if ((order_hint[i] == 0) && (order_hint[j] == 0))
          continue;
        auto remOp_1 = rewriter.create<arith::RemSIOp>(loc, configStrides[i],
                                                       configStrides[j]);
        auto remOp_2 = rewriter.create<arith::RemSIOp>(loc, configStrides[j],
                                                       configStrides[i]);
        auto trueCondition = rewriter.create<arith::OrIOp>(
            loc,
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           remOp_1, zero),
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           remOp_2, zero));
        rewriter.create<triton::gcu::AssertOp>(
            loc, trueCondition,
            "Not Support stride_large rem stride_lower is not zero,"
            "please add ENABLE_STRIDE_GATHER=True in kernel args list",
            "", "", 0);
      }
    }
  }

  for (int i = 0; i < rank; ++i) {
    if ((bDynamicStride && order_hint[i] == 1) ||
        (!bDynamicStride && order_hint[i] == 0)) {
      bReshape = false;
      break;
    }
  }

  // When the static innermost physical dim has constant stride K > 1 and no
  // dim already has unit stride, the runtime row-major mdspan would collapse
  // the innermost stride to 1 and read the wrong elements. Append a synthetic
  // innermost dim of shape K / stride 1 to the source memref while keeping
  // the slice size 1 in that dim, so the physical layout becomes contiguous.
  //
  // Only safe for identity (non-transpose) order: the downstream
  // vOrderShapes is derived by stride division (vOrderStrides[i] /
  // vOrderStrides[i+1]), which assumes monotonically decreasing strides.
  // A transpose reorders strides non-monotonically (e.g. [2,4,1]) and the
  // division yields garbage shapes, so we must not append in that case.
  bool bAppendStrideDim = false;
  std::optional<int64_t> innerStrideConst;
  if (!bDynamicStride && !bReshape && rank > 1 && rank < 4) {
    bool isIdentityOrder = true;
    for (int i = 0; i < rank; ++i) {
      if (order_hint[i] != i) {
        isIdentityOrder = false;
        break;
      }
    }
    bool hasUnitStride = false;
    for (int i = 0; i < rank; ++i) {
      if (auto strideAttr = getConstantIntValue(configStrides[i])) {
        if (*strideAttr == 1)
          hasUnitStride = true;
      }
    }
    if (isIdentityOrder && !hasUnitStride) {
      if (auto innerAttr = getConstantIntValue(configStrides[rank - 1])) {
        int64_t K = *innerAttr;
        if (K > 1) {
          innerStrideConst = K;
          bAppendStrideDim = true;
        }
      }
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (bDynamicStride && order_hint[i] == 0)
      updateStrideDims.push_back(i);
    else
      nInitStrideDims.push_back(i);
  }

  SmallVector<Value, 4> vSrcOffsets;
  if (IsShareOutput) {
    if (hasOffsetsAreDst(op)) {
      // The offsets are DESTINATION tile offsets: PtrAnalysis folded the mask
      // start into the base (foldMaskStartIntoBase), so the valid block
      // already starts at element 0 of the source memref. Reading them as
      // source offsets would apply the start twice and would collapse
      // vSlicehape to 0; the destination offsets are consumed by the
      // isDstTileOffsetOp branch below.
      for (unsigned i = 0; i < nInitStrideDims.size(); ++i)
        vSrcOffsets.push_back(zero);
    } else {
      // Plain source offsets, e.g. the sub-region offset folded in by
      // FuseMemDescSubsliceLoadLocalAllocPattern.
      for (unsigned i = 0; i < nInitStrideDims.size(); ++i) {
        if (configOffsets.empty())
          vSrcOffsets.push_back(zero);
        else
          vSrcOffsets.push_back(configOffsets[nInitStrideDims[i]]);
      }
    }
  } else {
    auto load = llvm::cast<triton::gcu::LoadOp>(op);
    auto loadType = load.getType();
    auto numElems = triton::gcu::getElemsPerThread(loadType);
    const auto &warpIds = getWarpIds(rewriter, loc, loadType);
    for (auto dim : nInitStrideDims) {
      Value offset = rewriter.create<arith::MulIOp>(
          loc, warpIds[dim],
          rewriter.create<arith::ConstantIndexOp>(loc, numElems[dim]));
      vSrcOffsets.push_back(offset);
    }
  }

  SmallVector<Value, 4> vSrcStrides;
  SmallVector<Value, 4> vSrcShapes;
  for (auto dim : nInitStrideDims) {
    vSrcStrides.push_back(configStrides[dim]);
    vSrcShapes.push_back(configShapes[dim]);
  }

  // rewriter.create<gpu::PrintfOp>(loc, "init stride %d, %d, %d\n",
  // ValueRange{vSrcStrides[0], vSrcStrides[1], vSrcStrides[2]});
  // rewriter.create<gpu::PrintfOp>(loc, "init shape %d, %d, %d\n",
  // ValueRange{vSrcShapes[0], vSrcShapes[1], vSrcShapes[2]});

  SmallVector<Value, 4> vResultShapes;
  SmallVector<int64_t, 4> resultShapes;
  for (unsigned i = 0; i < rank; ++i) {
    resultShapes.push_back(resultType.getShape()[i]);
    vResultShapes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, resultShapes[i]));
  }

  Value reshapeOut = srcOut;
  if (bReshape && rank < 4) {
    vSrcOffsets.push_back(zero);
    vSrcShapes.push_back(one);
    vSrcStrides.push_back(one);
    resultShapes.push_back(1);
    vResultShapes.push_back(one);
    if (bDynamicStride) {
      order_hint.push_back(1);
      nInitStrideDims.push_back(rank);
    } else {
      for (int i = 0; i < rank; ++i)
        order_hint[i]--;
      order_hint.push_back(rank);
    }

    rank += 1;
    auto srcMemRefType = dyn_cast<MemRefType>(srcOut.getType());
    auto reshapeMemrefType = MemRefType::get(
        resultShapes, elemType, AffineMap{},
        srcMemRefType ? srcMemRefType.getMemorySpace() : Attribute{});
    auto [reshapeStrides, reshapeOffset] =
        reshapeMemrefType.getStridesAndOffset();
    reshapeOut = rewriter.create<memref::ReinterpretCastOp>(
        loc, reshapeMemrefType, srcOut, reshapeOffset, resultShapes,
        reshapeStrides);
  } else if (bAppendStrideDim && rank < 4) {
    int64_t K = *innerStrideConst;
    vSrcOffsets.push_back(zero);
    vSrcShapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, K));
    vSrcStrides.push_back(one);
    resultShapes.push_back(1);
    vResultShapes.push_back(one);
    order_hint.push_back(rank);
    nInitStrideDims.push_back(rank);

    rank += 1;
    // The tensor-side buffer (srcOut, the load's result) only holds
    // resultShape elements; the appended innermost dim must keep size 1 here
    // so the reinterpreted memref stays in-bounds. The ptr-side source
    // receives shape K via vSrcShapes/vOrderShapes below.
    auto srcMemRefType = dyn_cast<MemRefType>(srcOut.getType());
    auto reshapeMemrefType = MemRefType::get(
        resultShapes, elemType, AffineMap{},
        srcMemRefType ? srcMemRefType.getMemorySpace() : Attribute{});
    auto [reshapeStrides, reshapeOffset] =
        reshapeMemrefType.getStridesAndOffset();
    reshapeOut = rewriter.create<memref::ReinterpretCastOp>(
        loc, reshapeMemrefType, srcOut, reshapeOffset, resultShapes,
        reshapeStrides);
  }

  if (rank == 2 && bDynamicStride) {
    if (order_hint[1] == 1) {
      order_hint[0] = 0;
      order_hint[1] = 1;
      bDynamicStride = false;
    } else if (order_hint[0] == 1) {
      order_hint[0] = 1;
      order_hint[1] = 0;
      bDynamicStride = false;
    }
  }

  SmallVector<Value, 4> vOrderStrides;
  SmallVector<Value, 4> vOrderShapes;
  SmallVector<Value, 4> vOrderOffsets;
  SmallVector<Value, 4> vTempOrder;
  SmallVector<Value, 4> vTransOrder;
  if (bDynamicStride) {
    GetOrderValueByStride(rewriter, loc, nInitStrideDims, vSrcStrides,
                          vSrcShapes, vSrcOffsets, vOrderStrides, vOrderShapes,
                          vOrderOffsets, vTempOrder);
    for (auto updateDim : updateStrideDims) {
      auto updateStride = rewriter.create<arith::MulIOp>(
          loc, vOrderStrides[updateDim], vOrderShapes[updateDim]);
      vOrderStrides.insert(vOrderStrides.begin() + updateDim, updateStride);
      vSrcStrides.insert(vSrcStrides.begin() + updateDim, updateStride);
      vOrderShapes.insert(vOrderShapes.begin() + updateDim, one);
      vSrcShapes.insert(vSrcShapes.begin() + updateDim, one);
      vOrderOffsets.insert(vOrderOffsets.begin() + updateDim,
                           rewriter.create<arith::IndexCastOp>(
                               loc, rewriter.getI32Type(), zero));
      vSrcOffsets.insert(vSrcOffsets.begin() + updateDim, zero);
      vTempOrder.insert(
          vTempOrder.begin() + updateDim,
          rewriter.create<arith::ConstantIntOp>(loc, updateDim, 32));
    }
    GetTransByOrder(rewriter, loc, vTempOrder, vTransOrder);
  } else {
    SmallVector<int32_t, 4> static_order(order_hint.begin(), order_hint.end());
    SmallVector<Value> staticOrderStrides(rank);
    SmallVector<Value> staticOrderShapes(rank);
    SmallVector<Value> staticOrderOffsets(rank);
    for (int i = 0; i < rank; ++i) {
      staticOrderStrides[static_order[i]] = vSrcStrides[i];
      staticOrderShapes[static_order[i]] = vSrcShapes[i];
      staticOrderOffsets[static_order[i]] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), vSrcOffsets[i]);
    }

    for (int i = 0; i < rank; ++i) {
      vOrderStrides.push_back(staticOrderStrides[i]);
      vOrderOffsets.push_back(staticOrderOffsets[i]);
      vTransOrder.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, static_order[i], 32));
    }
    if (static_order.size() > 0)
      vOrderShapes.push_back(staticOrderShapes[0]);
    for (int i = 0; i < rank - 1; ++i) {
      vOrderShapes.push_back(rewriter.create<arith::DivSIOp>(
          loc, vOrderStrides[i], vOrderStrides[i + 1]));
    }

    for (int i = 0; i < rank; ++i) {
      if (static_order[i] != i) {
        bStaticTranspose = true;
        break;
      }
    }
  }

  SmallVector<Value, 4> vSlicehape;
  SmallVector<Value, 4> vIntSlicehape;
  Value totalSize = one;
  for (unsigned i = 0; i < rank; ++i) {
    auto shape = rewriter.create<arith::MinSIOp>(
        loc, vResultShapes[i],
        rewriter.create<arith::MaxSIOp>(
            loc, zero,
            rewriter.create<arith::SubIOp>(loc, vSrcShapes[i],
                                           vSrcOffsets[i])));
    vSlicehape.push_back(shape);
    vIntSlicehape.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), shape));
    totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, shape);
  }

  SmallVector<Value, 4> padOffsets;
  SmallVector<Value, 4> padSizes;
  Value padSize = zero;
  for (unsigned i = 0; i < rank; ++i) {
    auto dim_diff =
        rewriter.create<arith::SubIOp>(loc, vResultShapes[i], vSlicehape[i]);
    padSize = rewriter.create<arith::AddIOp>(loc, padSize, dim_diff);
    padSizes.push_back(dim_diff);
    padOffsets.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), zero));
  }

  auto isNeedPad = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, padSize, zero);

  Value isDynamicTrans = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, vTransOrder[0],
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), zero));
  for (unsigned i = 1; i < rank; ++i) {
    auto isDimTrans = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, vTransOrder[i],
        rewriter.create<arith::ConstantIntOp>(loc, i, 32));
    isDynamicTrans =
        rewriter.create<arith::OrIOp>(loc, isDynamicTrans, isDimTrans);
  }

  auto sourceType = MemRefType::get(
      SmallVector<int64_t>(rank, ShapedType::kDynamic), elemType);
  auto src = rewriter.create<memref::ReinterpretCastOp>(
      loc, sourceType, buffer, zero, vOrderShapes, vOrderStrides);

  // rewriter.create<gpu::PrintfOp>(loc, "order stride %d, %d, %d\n",
  // ValueRange{vOrderStrides[0], vOrderStrides[1], vOrderStrides[2]});
  // rewriter.create<gpu::PrintfOp>(loc, "order shape %d, %d, %d\n",
  // ValueRange{vOrderShapes[0], vOrderShapes[1], vOrderShapes[2]});
  // rewriter.create<gpu::PrintfOp>(loc, "slice shape %d, %d, %d\n",
  // ValueRange{vIntSlicehape[0], vIntSlicehape[1], vIntSlicehape[2]});

  if (IsShareOutput) {
    auto masterWarpId = getMasterThreadId(op);
    auto isMasterThread = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
        rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
    auto isPad = rewriter.create<arith::AndIOp>(loc, isNeedPad, isMasterThread);
    if (bDynamicStride) {
      auto isTrans =
          rewriter.create<arith::AndIOp>(loc, isDynamicTrans, isMasterThread);
      rewriter.create<scf::IfOp>(
          loc, isTrans,
          [&](OpBuilder &builder, Location loc) {
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vTransOrder, defaultValue,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isPad,
                [&](OpBuilder &childBuilder, Location loc) {
                  doSlicePadOrMemsetSlice(childBuilder, loc, op, reshapeOut,
                                          src, vOrderOffsets, vIntSlicehape,
                                          padSizes, defaultValue, tag);
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<scf::IfOp>(
                      loc, isMasterThread,
                      [&](OpBuilder &child2Builder, Location loc) {
                        child2Builder.create<memref_ext::SliceStartOp>(
                            loc, reshapeOut, src, vOrderOffsets, defaultValue,
                            tag.getTag(), ValueRange{tag.getIdx()});
                        child2Builder.create<scf::YieldOp>(loc);
                      });
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    } else if (bStaticTranspose) {
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vTransOrder, defaultValue,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          });
    } else if (bStaticBroadcast) {
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<memref_ext::SliceBroadcastStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vIntSlicehape,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          });
    } else if (isDstTileOffsetOp(op, configOffsets, rank)) {
      // The valid block starts inside the tile (a `>=`-style mask): the start
      // is folded into the source base, so slice_start / slice_pad - which can
      // only write at tile offset 0 - would put the data in the masked-out
      // lanes. Fill the whole tile with the default value first, then DTE the
      // valid block into its destination columns. doMemset waits for its own
      // DTE, so the deslice starts from a completed fill.
      auto dstOffsets = getDstTileOffsets(rewriter, loc, configOffsets,
                                          order_hint, rank, zero);
      auto isNotZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, totalSize, zero);
      int64_t totalNumElems = 1;
      for (int64_t s : resultShapes)
        totalNumElems *= s;
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isNotZero,
                [&](OpBuilder &childBuilder, Location loc) {
                  if (totalNumElems > 0)
                    doMemset(childBuilder, tag, op, reshapeOut, defaultValue,
                             totalNumElems);
                  childBuilder.create<memref_ext::SliceDesliceStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, vIntSlicehape,
                      dstOffsets, tag.getTag(), ValueRange{tag.getIdx()});
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  if (getDefaultValue(op))
                    doMemsetConfig(childBuilder, loc, reshapeOut, defaultValue,
                                   tag);
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      rewriter.create<scf::IfOp>(
          loc, isPad,
          [&](OpBuilder &builder, Location loc) {
            doSlicePadOrMemsetSlice(builder, loc, op, reshapeOut, src,
                                    vOrderOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tag);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isMasterThread,
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, defaultValue,
                      tag.getTag(), ValueRange{tag.getIdx()});
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    }
  } else {
    auto isNotZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, totalSize, zero);
    auto isPad = rewriter.create<arith::AndIOp>(loc, isNeedPad, isNotZero);
    // Private (non-shared) masked loads may carry the within-tile mask start
    // through configOffsets (see PtrAnalysis::getPtrInfo /
    // exposeMaskStartOffset): the valid block then has to be written at tile
    // column >= mask start, not at column 0. Detect that here so the static
    // (non-transposed) path below emits a memset + slice_deslice with dst
    // offsets.
    bool bDstTileOffsetLoad = hasNonZeroOffset(configOffsets, rank);
    if (bDynamicStride) {
      auto isTrans =
          rewriter.create<arith::AndIOp>(loc, isDynamicTrans, isNotZero);
      rewriter.create<scf::IfOp>(
          loc, isTrans,
          [&](OpBuilder &builder, Location loc) {
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vTransOrder, defaultValue,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isPad,
                [&](OpBuilder &childBuilder, Location loc) {
                  doSlicePadOrMemsetSlice(childBuilder, loc, op, reshapeOut,
                                          src, vOrderOffsets, vIntSlicehape,
                                          padSizes, defaultValue, tag);
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<scf::IfOp>(
                      loc, isNotZero,
                      [&](OpBuilder &child2Builder, Location loc) {
                        child2Builder.create<memref_ext::SliceStartOp>(
                            loc, reshapeOut, src, vOrderOffsets, defaultValue,
                            tag.getTag(), ValueRange{tag.getIdx()});
                        child2Builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &child2Builder, Location loc) {
                        if (getDefaultValue(op)) {
                          doMemsetConfig(child2Builder, loc, reshapeOut,
                                         defaultValue, tag);
                        }
                        child2Builder.create<scf::YieldOp>(loc);
                      });
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    } else if (bStaticTranspose) {
      rewriter.create<scf::IfOp>(
          loc, isNotZero,
          [&](OpBuilder &builder, Location loc) {
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vTransOrder, defaultValue,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            if (getDefaultValue(op)) {
              doMemsetConfig(builder, loc, reshapeOut, defaultValue, tag);
            }
            builder.create<scf::YieldOp>(loc);
          });
    } else if (bDstTileOffsetLoad) {
      // Masked load whose valid block starts at a non-zero tile column
      // (configOffsets carries the per-dim mask start; the base is already
      // folded so the DTE reads the valid memory region). SlicePad/SliceStart
      // can only place the loaded block at tile offset 0, which puts the data
      // in the masked-out lanes. Instead, fill the whole result tile with the
      // default value first, then DTE the valid block into its tile-column
      // range (memset + slice_deslice with dst offsets).
      // The dst offsets are expressed in the result-tile (memory) dims: map
      // configOffsets (original dims) through static_order -> memory position.
      auto dstOffsets = getDstTileOffsets(rewriter, loc, configOffsets,
                                          order_hint, rank, zero);
      int64_t totalNumElems = 1;
      for (int64_t s : resultShapes)
        totalNumElems *= s;
      rewriter.create<scf::IfOp>(
          loc, isNotZero,
          [&](OpBuilder &builder, Location loc) {
            if (totalNumElems > 0)
              doMemset(builder, tag, op, reshapeOut, defaultValue,
                       totalNumElems);
            builder.create<memref_ext::SliceDesliceStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vIntSlicehape, dstOffsets,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            if (getDefaultValue(op)) {
              doMemsetConfig(builder, loc, reshapeOut, defaultValue, tag);
            }
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      rewriter.create<scf::IfOp>(
          loc, isPad,
          [&](OpBuilder &builder, Location loc) {
            doSlicePadOrMemsetSlice(builder, loc, op, reshapeOut, src,
                                    vOrderOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tag);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isNotZero,
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, defaultValue,
                      tag.getTag(), ValueRange{tag.getIdx()});
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  if (getDefaultValue(op)) {
                    doMemsetConfig(childBuilder, loc, reshapeOut, defaultValue,
                                   tag);
                  }
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    }
  }
  return totalSize;
}

Value ConfigGcuStore(OpBuilder &rewriter, Location loc, Value storeValue,
                     /*Value transOut,*/ mlir::Operation *op,
                     MemRefType storeValueType, Value storePtr,
                     mlir::ValueRange configStrides,
                     mlir::ValueRange configShapes,
                     mlir::ValueRange configOffsets, triton::gcu::TagInfo tag) {
  auto storeOp = dyn_cast<triton::gcu::StoreOp>(op);
  assert(storeOp);

  auto storeType = storeOp.getValue().getType(); // Tensor Type
  auto elemType = storeOp.getPtr().getType().getElementType();

  auto buffer = rewriter.create<gcu::PtrToMemRefOp>(
      loc, MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType),
      storePtr);

  int64_t rank = storeValueType.getRank(); // Memref Type
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  bool bDynamicStride = false;
  bool bStaticTranspose = false;
  bool bReshape = true;
  SmallVector<unsigned> updateStrideDims;
  SmallVector<unsigned> nInitStrideDims;

  auto hint = storeOp.getOrderHint();
  int64_t hint_size = static_cast<int64_t>(hint.size());
  assert(hint_size == rank || hint_size == 0);
  SmallVector<int32_t> order_hint;
  for (unsigned i = 0; i < rank; ++i)
    if (hint_size == 0)
      order_hint.push_back(Dynamic_stride_symbol);
    else
      order_hint.push_back(hint[i]);

  for (unsigned i = 0; i < rank; ++i) {
    if (order_hint[i] == Dynamic_stride_symbol) {
      bDynamicStride = true;
      if (triton::gcu::get_bool_env("TRITON_GCU_DEBUG") ||
          triton::gcu::get_bool_env("TRITON_ENABLE_ASAN")) {
        auto trueCondition = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, configStrides[i], zero);
        rewriter.create<triton::gcu::AssertOp>(
            loc, trueCondition, "Not Support dynamic stride is 0", "", "", 0);
      }
    }
  }

  for (int i = 0; i < rank; ++i) {
    if ((order_hint[i] == 0 && !bDynamicStride) ||
        (order_hint[i] == 1 && bDynamicStride)) {
      bReshape = false;
      break;
    }
  }

  // When the static innermost physical dim has constant stride K > 1 and no
  // dim already has unit stride, the runtime row-major mdspan would collapse
  // the innermost stride to 1 and write the wrong elements. Append a synthetic
  // innermost dim of shape K / stride 1 to the destination memref while
  // keeping the slice size 1 in that dim.
  //
  // Only safe for identity (non-transpose) order; see ConfigGcuLoad for the
  // rationale (vOrderShapes derives shapes by stride division, which breaks
  // on a non-monotonic transposed stride layout).
  bool bAppendStrideDim = false;
  std::optional<int64_t> innerStrideConst;
  if (!bDynamicStride && !bReshape && rank > 1 && rank < 4) {
    bool isIdentityOrder = true;
    for (int i = 0; i < rank; ++i) {
      if (order_hint[i] != i) {
        isIdentityOrder = false;
        break;
      }
    }
    bool hasUnitStride = false;
    for (int i = 0; i < rank; ++i) {
      if (auto strideAttr = getConstantIntValue(configStrides[i])) {
        if (*strideAttr == 1)
          hasUnitStride = true;
      }
    }
    if (isIdentityOrder && !hasUnitStride) {
      if (auto innerAttr = getConstantIntValue(configStrides[rank - 1])) {
        int64_t K = *innerAttr;
        if (K > 1) {
          innerStrideConst = K;
          bAppendStrideDim = true;
        }
      }
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (bDynamicStride && order_hint[i] == 0)
      updateStrideDims.push_back(i);
    else
      nInitStrideDims.push_back(i);
  }

  SmallVector<Value, 4> vSrcOffsets;
  auto numElems = triton::gcu::getElemsPerThread(storeType);
  SmallVector<Value, 4> vNumElems;
  for (unsigned i = 0; i < rank; ++i)
    vNumElems.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, numElems[i]));
  auto warpIds = getWarpIds(rewriter, loc, storeType);
  for (auto dim : nInitStrideDims) {
    Value offset =
        rewriter.create<arith::MulIOp>(loc, warpIds[dim], vNumElems[dim]);
    vSrcOffsets.push_back(offset);
  }

  SmallVector<Value, 4> vSrcStrides;
  SmallVector<Value, 4> vSrcShapes;
  for (auto dim : nInitStrideDims) {
    vSrcStrides.push_back(configStrides[dim]);
    vSrcShapes.push_back(configShapes[dim]);
  }

  SmallVector<Value, 4> vStoreShapes;
  SmallVector<int64_t, 4> storeShapes;
  for (unsigned i = 0; i < rank; ++i) {
    storeShapes.push_back(storeValueType.getShape()[i]);
    vStoreShapes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, storeShapes[i]));
  }

  Value reshapeStoreValue = storeValue;
  if (bReshape) {
    assert(rank < 4 && "not support stride is no 1 for rank >= 4");
    vSrcOffsets.push_back(zero);
    vSrcShapes.push_back(one);
    vSrcStrides.push_back(one);
    storeShapes.push_back(1);
    vStoreShapes.push_back(one);
    vNumElems.push_back(one);
    if (bDynamicStride) {
      order_hint.push_back(1);
      nInitStrideDims.push_back(rank);
    } else {
      for (int i = 0; i < rank; ++i)
        order_hint[i]--;
      order_hint.push_back(rank);
    }
    rank += 1;
    auto reshapeStoreType = MemRefType::get(storeShapes, elemType);
    auto [reshapeStrides, reshapeOffset] =
        reshapeStoreType.getStridesAndOffset();
    reshapeStoreValue = rewriter.create<memref::ReinterpretCastOp>(
        loc, reshapeStoreType, storeValue, reshapeOffset, storeShapes,
        reshapeStrides);
  } else if (bAppendStrideDim && rank < 4) {
    int64_t K = *innerStrideConst;
    vSrcOffsets.push_back(zero);
    vSrcShapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, K));
    vSrcStrides.push_back(one);
    storeShapes.push_back(1);
    vStoreShapes.push_back(one);
    vNumElems.push_back(one);
    order_hint.push_back(rank);
    nInitStrideDims.push_back(rank);

    rank += 1;
    // The tensor-side storeValue only holds storeShapes elements; the
    // appended innermost dim keeps size 1 here. The ptr-side destination
    // receives shape K via vSrcShapes/vOrderShapes below.
    auto reshapeStoreType = MemRefType::get(storeShapes, elemType);
    auto [reshapeStrides, reshapeOffset] =
        reshapeStoreType.getStridesAndOffset();
    reshapeStoreValue = rewriter.create<memref::ReinterpretCastOp>(
        loc, reshapeStoreType, storeValue, reshapeOffset, storeShapes,
        reshapeStrides);
  }

  if (rank == 2 && bDynamicStride) {
    if (order_hint[1] == 1) {
      order_hint[0] = 0;
      order_hint[1] = 1;
      bDynamicStride = false;
    } else if (order_hint[0] == 1) {
      order_hint[0] = 1;
      order_hint[1] = 0;
      bDynamicStride = false;
    }
  }

  SmallVector<Value, 4> vOrderStrides;
  SmallVector<Value, 4> vOrderShapes;
  SmallVector<Value, 4> vOrderOffsets;
  SmallVector<Value, 4> vTransOrder;
  SmallVector<Value, 4> vTempOrder;
  if (bDynamicStride) {
    GetOrderValueByStride(rewriter, loc, nInitStrideDims, vSrcStrides,
                          vSrcShapes, vSrcOffsets, vOrderStrides, vOrderShapes,
                          vOrderOffsets, vTempOrder);
    for (auto updateDim : updateStrideDims) {
      auto updateStride = rewriter.create<arith::MulIOp>(
          loc, vOrderStrides[updateDim], vOrderShapes[updateDim]);
      vOrderStrides.insert(vOrderStrides.begin() + updateDim, updateStride);
      vSrcStrides.insert(vSrcStrides.begin() + updateDim, updateStride);
      vOrderShapes.insert(vOrderShapes.begin() + updateDim, one);
      vSrcShapes.insert(vSrcShapes.begin() + updateDim, one);
      vOrderOffsets.insert(vOrderOffsets.begin() + updateDim,
                           rewriter.create<arith::IndexCastOp>(
                               loc, rewriter.getI32Type(), zero));
      vSrcOffsets.insert(vSrcOffsets.begin() + updateDim, zero);
      vTempOrder.insert(
          vTempOrder.begin() + updateDim,
          rewriter.create<arith::ConstantIntOp>(loc, updateDim, 32));
    }
    GetTransByOrder(rewriter, loc, vTempOrder, vTransOrder);
  } else {
    SmallVector<int32_t, 4> static_order(order_hint.begin(), order_hint.end());
    SmallVector<Value> staticOrderStrides(rank);
    SmallVector<Value> staticOrderShapes(rank);
    SmallVector<Value> staticOrderOffsets(rank);
    for (int i = 0; i < rank; ++i) {
      staticOrderStrides[static_order[i]] = vSrcStrides[i];
      staticOrderShapes[static_order[i]] = vSrcShapes[i];
      staticOrderOffsets[static_order[i]] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), vSrcOffsets[i]);
    }

    for (int i = 0; i < rank; ++i) {
      vOrderStrides.push_back(staticOrderStrides[i]);
      vOrderOffsets.push_back(staticOrderOffsets[i]);
      vTransOrder.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, static_order[i], 32));
    }
    if (static_order.size() > 0)
      vOrderShapes.push_back(staticOrderShapes[0]);
    for (int i = 0; i < rank - 1; ++i) {
      vOrderShapes.push_back(rewriter.create<arith::DivSIOp>(
          loc, vOrderStrides[i], vOrderStrides[i + 1]));
    }

    for (int i = 0; i < rank; ++i) {
      if (static_order[i] != i) {
        bStaticTranspose = true;
        break;
      }
    }
  }

  SmallVector<unsigned, 4>
      memOrderToSrc; // memOrderToSrc[k] = src dim at memory pos k
  if (bDynamicStride) {
    // For dynamic-stride stores the memory order vTempOrder[k] is a runtime
    // arith.select chain (the strides are runtime values), so it cannot be
    // extracted into a compile-time memOrderToSrc here. The per-position warp
    // offset/shape are taken from vOrderOffsets/vOrderShapes below, and the
    // per-source-dim vectors (vNumElems, configOffsets, vStoreShapes) are
    // gathered at runtime by vTempOrder[k]. memOrderToSrc is only used in the
    // static path (all strides compile-time constant).
    for (unsigned k = 0; k < rank; ++k) {
      auto ordConst = getConstantIntValue(vTempOrder[k]);
      memOrderToSrc.push_back(ordConst ? static_cast<unsigned>(*ordConst) : k);
    }
  } else {
    // static_order[i] = memory position of original dim i; invert it.
    SmallVector<int32_t, 4> static_order(order_hint.begin(), order_hint.end());
    memOrderToSrc.resize(rank);
    for (unsigned i = 0; i < rank; ++i)
      memOrderToSrc[static_order[i]] = i;
  }

  // Pad configOffsets to `rank` so the runtime gather below is always safe.
  SmallVector<Value> configOffsetsPadded(configOffsets.begin(),
                                         configOffsets.end());
  while (configOffsetsPadded.size() < static_cast<size_t>(rank))
    configOffsetsPadded.push_back(zero);

  // Runtime gather: return srcDimVals[d] where d = memOrder (i32, the source
  // dim placed at this memory position). Needed because the memory order is
  // runtime for dynamic-stride stores.
  auto gatherByMemOrder = [&](Value memOrder, ArrayRef<Value> srcDimVals) {
    Value res = srcDimVals.front();
    for (unsigned d = 1; d < srcDimVals.size(); ++d) {
      auto idx = rewriter.create<arith::ConstantIntOp>(loc, d, 32);
      auto eq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               memOrder, idx);
      res = rewriter.create<arith::SelectOp>(loc, eq, srcDimVals[d], res);
    }
    return res;
  };
  auto memOrderAt = [&](unsigned k) -> Value {
    if (bDynamicStride)
      return vTempOrder[k];
    return rewriter.create<arith::ConstantIntOp>(loc, memOrderToSrc[k], 32)
        .getResult();
  };

  SmallVector<Value, 4> vSlicehape;
  SmallVector<Value, 4> vIntSlicehape;
  SmallVector<Value, 4>
      sliceOffsets; // src (value buffer) slice offset, memory order
  SmallVector<Value, 4>
      dstOffsets; // dst (global tile) write offset, memory order
  Value totalSize = one;
  for (unsigned k = 0; k < rank; ++k) {
    Value memOrder = memOrderAt(k);
    // mask start (tile-internal valid region start) of the src dim at pos k.
    Value maskStart = gatherByMemOrder(memOrder, configOffsetsPadded);
    if (!maskStart.getType().isIndex())
      maskStart = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), maskStart);
    // warp offset of the src dim at memory pos k: vOrderOffsets is already in
    // memory order (unlike vSrcOffsets), so it carries the runtime-correct
    // per-warp offset here.
    Value warpBase = vOrderOffsets[k];
    if (!warpBase.getType().isIndex())
      warpBase = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), warpBase);
    Value warpEnd = rewriter.create<arith::AddIOp>(
        loc, warpBase, gatherByMemOrder(memOrder, vNumElems));
    // src slice offset (warp-local): how far into this warp's buffer the
    // valid region starts. num_warps=1 (warpBase=0) -> maskStart; num_warps>1
    // warp2 (warpBase=4, maskStart=4) -> 0.
    Value srcOff = rewriter.create<arith::MaxSIOp>(
        loc, zero, rewriter.create<arith::SubIOp>(loc, maskStart, warpBase));
    // dst offset (tile-global): where in the tile this warp writes.
    Value dstOff = rewriter.create<arith::MaxSIOp>(loc, warpBase, maskStart);
    // valid region high end in tile space. NOTE: use the mask-valid extent of
    // the src dim at this memory position (vSrcShapes / configShapes), NOT
    // vOrderShapes: vOrderShapes is rebuilt by GetOrderValueByStride from
    // stride ratios and always yields the FULL extent (e.g. block_shape_k=9),
    // losing the mask-derived valid count (min(BLOCK_K, block_shape_k - base))
    // — with the base offset folded into the dst pointer that would over-write
    // past the masked-out lanes. Add maskStart to recover the full-tile coord.
    Value validRegionEnd = rewriter.create<arith::AddIOp>(
        loc, maskStart, gatherByMemOrder(memOrder, vSrcShapes));
    Value validEnd =
        rewriter.create<arith::MinSIOp>(loc, warpEnd, validRegionEnd);
    auto shape = rewriter.create<arith::MaxSIOp>(
        loc, zero, rewriter.create<arith::SubIOp>(loc, validEnd, dstOff));
    vSlicehape.push_back(shape);
    vIntSlicehape.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), shape));
    sliceOffsets.push_back(rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), srcOff));
    dstOffsets.push_back(rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), dstOff));
    totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, shape);
  }

  // diff uses vStoreShapes in *memory* order to match vSlicehape (also built
  // in memory order above). vStoreShapes was filled in original order, so
  // gather it to memory order before differencing.
  SmallVector<Value, 4> vStoreShapesMem;
  for (unsigned k = 0; k < rank; ++k)
    vStoreShapesMem.push_back(gatherByMemOrder(memOrderAt(k), vStoreShapes));

  Value diff = zero;
  for (unsigned k = 0; k < rank; ++k) {
    auto dim_diff =
        rewriter.create<arith::SubIOp>(loc, vStoreShapesMem[k], vSlicehape[k]);
    diff = rewriter.create<arith::AddIOp>(loc, diff, dim_diff);
  }

  auto resultType = MemRefType::get(
      SmallVector<int64_t>(rank, ShapedType::kDynamic), elemType);
  auto dst = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, buffer, zero, vOrderShapes, vOrderStrides);
  // Shrink the contiguous src to the mask-valid counts so the DMA copy
  // amount is exactly the valid region; scatter memory-order counts back
  // to source order.
  SmallVector<Value, 4> vSliceShapeSrc(rank, zero);
  SmallVector<Value, 4> vSliceOffSrc(rank, zero);
  if (!bDynamicStride) {
    for (unsigned k = 0; k < rank; ++k) {
      unsigned srcDim = memOrderToSrc[k];
      vSliceShapeSrc[srcDim] = vSlicehape[k];
      vSliceOffSrc[srcDim] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), sliceOffsets[k]);
    }
  } else {
    for (unsigned k = 0; k < rank; ++k) {
      Value memOrder = memOrderAt(k);
      Value srcOffIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), sliceOffsets[k]);
      for (unsigned d = 0; d < rank; ++d) {
        auto idx = rewriter.create<arith::ConstantIntOp>(loc, d, 32);
        auto eq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 memOrder, idx);
        vSliceShapeSrc[d] = rewriter.create<arith::SelectOp>(
            loc, eq, vSlicehape[k], vSliceShapeSrc[d]);
        vSliceOffSrc[d] = rewriter.create<arith::SelectOp>(loc, eq, srcOffIdx,
                                                           vSliceOffSrc[d]);
      }
    }
  }
  // Full-tile row-major strides (buffer layout unchanged); fold the per-dim
  // offsets into the single linear base offset.
  auto fullType = MemRefType::get(storeShapes, elemType);
  auto [fullStrides, fullOffset] = fullType.getStridesAndOffset();
  (void)fullOffset;
  Value srcBaseOff = zero;
  SmallVector<Value, 4> fullStridesVals;
  for (unsigned d = 0; d < rank; ++d) {
    auto strideVal =
        rewriter.create<arith::ConstantIndexOp>(loc, fullStrides[d]);
    fullStridesVals.push_back(strideVal);
    srcBaseOff = rewriter.create<arith::AddIOp>(
        loc, srcBaseOff,
        rewriter.create<arith::MulIOp>(loc, vSliceOffSrc[d], strideVal));
  }
  auto srcSliced = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, reshapeStoreValue, srcBaseOff, vSliceShapeSrc,
      fullStridesVals);

  auto isNeedSlice = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sgt, diff, zero);

  auto isNotZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  totalSize, zero);

  Value isDynamicTrans = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, vTransOrder[0],
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), zero));
  for (unsigned i = 1; i < rank; ++i) {
    auto isDimTrans = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, vTransOrder[i],
        rewriter.create<arith::ConstantIntOp>(loc, i, 32));
    isDynamicTrans =
        rewriter.create<arith::OrIOp>(loc, isDynamicTrans, isDimTrans);
  }

  if (bDynamicStride) {
    auto isTrans =
        rewriter.create<arith::AndIOp>(loc, isDynamicTrans, isNotZero);
    auto isSlice = rewriter.create<arith::AndIOp>(loc, isNeedSlice, isNotZero);
    rewriter.create<scf::IfOp>(
        loc, isTrans,
        [&](OpBuilder &builder, Location loc) {
          builder.create<memref_ext::TransposeDesliceStartOp>(
              loc, dst, srcSliced, vTransOrder, dstOffsets, tag.getTag(),
              ValueRange{tag.getIdx()});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isSlice,
              [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::SliceDesliceStartOp>(
                    loc, dst, reshapeStoreValue, sliceOffsets, vIntSlicehape,
                    dstOffsets, tag.getTag(), ValueRange{tag.getIdx()});
                childBuilder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<scf::IfOp>(
                    loc, isNotZero,
                    [&](OpBuilder &child2Builder, Location loc) {
                      child2Builder.create<memref_ext::DesliceStartOp>(
                          loc, dst, reshapeStoreValue, dstOffsets, tag.getTag(),
                          ValueRange{tag.getIdx()});
                      child2Builder.create<scf::YieldOp>(loc);
                    });
                childBuilder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  } else if (bStaticTranspose) {
    // Single DMA: srcSliced limits the copy to the mask-valid region.
    rewriter.create<scf::IfOp>(
        loc, isNotZero, [&](OpBuilder &build, Location loc) {
          build.create<memref_ext::TransposeDesliceStartOp>(
              loc, dst, srcSliced, vTransOrder, dstOffsets, tag.getTag(),
              ValueRange{tag.getIdx()});
          build.create<scf::YieldOp>(loc);
        });
  } else {
    auto isSlice = rewriter.create<arith::AndIOp>(loc, isNeedSlice, isNotZero);
    rewriter.create<scf::IfOp>(
        loc, isSlice,
        [&](OpBuilder &builder, Location loc) {
          builder.create<memref_ext::SliceDesliceStartOp>(
              loc, dst, reshapeStoreValue, sliceOffsets, vIntSlicehape,
              dstOffsets, tag.getTag(), ValueRange{tag.getIdx()});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isNotZero, [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::DesliceStartOp>(
                    loc, dst, reshapeStoreValue, dstOffsets, tag.getTag(),
                    ValueRange{tag.getIdx()});
                childBuilder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  }
  return totalSize;
}

void WaitGcuLoadStore(OpBuilder &rewriter, Location loc,
                      triton::gcu::TagInfo tag, Value totalSize) {
  rewriter.create<memref::DmaWaitOp>(loc, tag.getTag(),
                                     ValueRange{tag.getIdx()}, totalSize);
}

void forEachAccDotOrMatmul(Value loadResult,
                           llvm::function_ref<void(Operation *)> callback) {
  for (auto *user : loadResult.getUsers()) {
    if (isa<triton::DotOp, gcu::MatMulOp>(user))
      callback(user);
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg != loadResult)
          continue;
        auto blockArg = forOp.getRegionIterArg(idx);
        for (auto *bodyUser : blockArg.getUsers()) {
          if (isa<triton::DotOp, gcu::MatMulOp>(bodyUser))
            callback(bodyUser);
        }
      }
    }
  }
}

StringRef getMatrixLoadMode(triton::gcu::LoadOp loadOp) {
  if (loadOp.getType().getRank() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "getMatrixLoadMode: loadOp shape rank != 2\n");
    return "";
  }
  StringRef result = "";
  forEachAccDotOrMatmul(loadOp.getResult(), [&](Operation *op) {
    if (result.empty()) {
      if (auto accLoad = op->getAttr(kAccLoad))
        result = cast<StringAttr>(accLoad).getValue();
    }
  });
  return result;
}

void ConfigMatrixLoad(OpBuilder &rewriter, Location loc,
                      triton::gcu::LoadOp loadOp, Value value, Value ptr,
                      ValueRange srcShapes, ValueRange srcStrides,
                      ValueRange srcOffsets, bool loadFromLocalMem) {
  auto loadType = loadOp.getType();
  int64_t rank = loadType.getRank();
  assert(rank == 2 && "matrix_load value must be 2D tensor");
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  auto numElems = triton::gcu::getElemsPerThread(loadType);
  SmallVector<Value, 4> vNumElems;
  for (unsigned i = 0; i < rank; ++i)
    vNumElems.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, numElems[i]));

  auto warpIds = getWarpIds(rewriter, loc, loadType);
  SmallVector<Value, 2> warpOffsets;
  for (unsigned i = 0; i < rank; ++i)
    warpOffsets.push_back(
        rewriter.create<arith::MulIOp>(loc, warpIds[i], vNumElems[i]));

  // Dst ptr with block-level offset applied
  Value srcPtr = ptr;
  if (!loadFromLocalMem) {
    auto srcElemType = loadOp.getPtr().getType().getElementType();
    int64_t elemBytes = (srcElemType.getIntOrFloatBitWidth() + 7) / 8;
    auto i64Type = rewriter.getI64Type();
    Value ptrInt = rewriter.create<gcu::PtrToIntOp>(loc, ptr);
    Value linearOffset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    for (unsigned i = 0; i < rank; ++i) {
      Value offset = rewriter.create<arith::IndexCastOp>(
          loc, i64Type,
          rewriter.create<arith::AddIOp>(loc, warpOffsets[i], srcOffsets[i]));
      Value stride =
          rewriter.create<arith::IndexCastOp>(loc, i64Type, srcStrides[i]);
      Value product = rewriter.create<arith::MulIOp>(loc, offset, stride);
      linearOffset = rewriter.create<arith::AddIOp>(loc, linearOffset, product);
    }
    Value elemSizeVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(elemBytes));
    Value byteOffset =
        rewriter.create<arith::MulIOp>(loc, linearOffset, elemSizeVal);
    Value offsetPtrInt =
        rewriter.create<arith::AddIOp>(loc, ptrInt, byteOffset);
    srcPtr = rewriter.create<gcu::IntToPtrOp>(loc, ptr.getType(), offsetPtrInt);
  }

  SmallVector<Value, 2> memDims;
  if (loadFromLocalMem) {
    auto valueShape = dyn_cast<MemRefType>(value.getType()).getShape();
    memDims.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, valueShape[0]));
    memDims.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, valueShape[1]));
  } else {
    memDims.push_back(srcShapes[0]);
    memDims.push_back(srcStrides[0]);
  }

  SmallVector<Value, 2> realDims;
  for (unsigned i = 0; i < rank; ++i) {
    Value remaining =
        rewriter.create<arith::SubIOp>(loc, srcShapes[i], warpOffsets[i]);
    Value clamped = rewriter.create<arith::MaxSIOp>(loc, zero, remaining);
    Value sliceShape =
        rewriter.create<arith::MinSIOp>(loc, vNumElems[i], clamped);
    realDims.push_back(sliceShape);
  }

  rewriter.create<gcu::MatrixLoadOp>(loc, value, srcPtr, memDims, realDims);
}

// Walks the def-use chain of `val` and returns the `memref.alloca` it is
// backed by, looking through reinterpret casts, loop-carried buffers and
// in-place fusion regions. `remap` resolves values whose replacement has only
// been registered by an in-flight dialect conversion (rollback mode).
memref::AllocaOp getAllocaOp(Value val, triton::gcu::ValueRemapFn remap) {
  SmallPtrSet<Value, 16> visited;
  while (val && visited.insert(val).second) {
    if (Value remapped = remap(val))
      val = remapped;
    if (auto castOp = val.getDefiningOp<memref::ReinterpretCastOp>()) {
      val = castOp.getSource();
      continue;
    }
    if (auto allocaOp = val.getDefiningOp<memref::AllocaOp>()) {
      return allocaOp;
    }
    // Loop-carried buffer: trace it back to the value the loop starts from.
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp)
        return nullptr;
      val = forOp.getInitArgs()[blockArg.getArgNumber() -
                                forOp.getNumInductionVars()];
      continue;
    }
    if (auto forOp = val.getDefiningOp<scf::ForOp>()) {
      unsigned resultIdx = cast<OpResult>(val).getResultNumber();
      val = forOp.getRegionIterArg(resultIdx);
      continue;
    }
    // In-place fusion region: the result is written to its in-place operand.
    if (auto fusionOp =
            val.getDefiningOp<triton::gcu::ElementwiseFusionRegionOp>()) {
      if (auto inplaceAttr =
              fusionOp->getAttrOfType<IntegerAttr>(kAccReuseInplaceOperand)) {
        val = fusionOp->getOperand(inplaceAttr.getInt());
        continue;
      }
      return nullptr;
    }
    return nullptr;
  }
  return nullptr;
}

bool useMatrixStore(triton::gcu::StoreOp storeOp, Value adaptedValue,
                    triton::gcu::ValueRemapFn remap) {
  if (storeOp.getValue().getType().getRank() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "useMatrixStore: storeOp shape rank != 2\n");
    return false;
  }
  // return storeOp->hasAttr(kMaxtrixStore);
  return getAllocaOp(adaptedValue, remap) != nullptr;
}

bool useMatrixStore(triton::gcu::StoreOp storeOp, Value adaptedValue) {
  return useMatrixStore(storeOp, adaptedValue,
                        [](Value value) { return value; });
}

void ConfigMatrixStore(OpBuilder &rewriter, Location loc,
                       triton::gcu::StoreOp storeOp, Value value, Value ptr,
                       ValueRange dstShapes, ValueRange dstStrides,
                       ValueRange dstOffsets, bool storeToLocalMem) {
  auto storeType = storeOp.getValue().getType();
  int64_t rank = storeType.getRank();
  assert(rank == 2 && "matrix_store value must be 2D memref");
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Single warp info
  auto numElems = triton::gcu::getElemsPerThread(storeType);
  SmallVector<Value, 4> vNumElems;
  for (unsigned i = 0; i < rank; ++i)
    vNumElems.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, numElems[i]));

  auto warpIds = getWarpIds(rewriter, loc, storeType);
  SmallVector<Value, 2> warpOffsets;
  for (unsigned i = 0; i < rank; ++i)
    warpOffsets.push_back(
        rewriter.create<arith::MulIOp>(loc, warpIds[i], vNumElems[i]));

  // Dst ptr with block-level offset applied
  Value dstPtr = ptr;
  if (!storeToLocalMem) {
    auto dstElemType = storeOp.getPtr().getType().getElementType();
    int64_t elemBytes = (dstElemType.getIntOrFloatBitWidth() + 7) / 8;
    auto i64Type = rewriter.getI64Type();
    Value ptrInt = rewriter.create<gcu::PtrToIntOp>(loc, ptr);
    Value linearOffset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    for (unsigned i = 0; i < rank; ++i) {
      Value offset = rewriter.create<arith::IndexCastOp>(
          loc, i64Type,
          rewriter.create<arith::AddIOp>(loc, warpOffsets[i], dstOffsets[i]));
      Value stride =
          rewriter.create<arith::IndexCastOp>(loc, i64Type, dstStrides[i]);
      Value product = rewriter.create<arith::MulIOp>(loc, offset, stride);
      linearOffset = rewriter.create<arith::AddIOp>(loc, linearOffset, product);
    }
    Value elemSizeVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(elemBytes));
    Value byteOffset =
        rewriter.create<arith::MulIOp>(loc, linearOffset, elemSizeVal);
    Value offsetPtrInt =
        rewriter.create<arith::AddIOp>(loc, ptrInt, byteOffset);
    dstPtr = rewriter.create<gcu::IntToPtrOp>(loc, ptr.getType(), offsetPtrInt);
  }

  // Dst mem dims
  SmallVector<Value, 2> memDims;
  if (storeToLocalMem) {
    auto valueShape = dyn_cast<MemRefType>(value.getType()).getShape();
    memDims.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, valueShape[0]));
    memDims.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, valueShape[1]));
  } else {
    memDims.push_back(dstShapes[0]);
    memDims.push_back(dstStrides[0]);
  }

  // Dst real dims
  SmallVector<Value, 2> realDims;
  for (unsigned i = 0; i < rank; ++i) {
    Value remaining =
        rewriter.create<arith::SubIOp>(loc, dstShapes[i], warpOffsets[i]);
    Value clamped = rewriter.create<arith::MaxSIOp>(loc, zero, remaining);
    Value sliceShape =
        rewriter.create<arith::MinSIOp>(loc, vNumElems[i], clamped);
    realDims.push_back(sliceShape);
  }

  rewriter.create<gcu::MatrixStoreOp>(loc, value, dstPtr, memDims, realDims);
}

Operation *ConfigMatrixStoreLocal(OpBuilder &rewriter, Location loc,
                                  MemRefType resultType, Value out,
                                  Value adaptedResult) {
  auto outPtrType =
      gcu::PtrType::get(rewriter.getContext(), resultType.getElementType());
  auto outPtr = rewriter.create<gcu::MemRefToPtrOp>(loc, outPtrType, out);
  auto outShape = resultType.getShape();
  SmallVector<Value, 2> memDims, realDims;
  memDims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outShape[0]));
  memDims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outShape[1]));
  realDims.push_back(rewriter.create<memref::DimOp>(loc, out, 0));
  realDims.push_back(rewriter.create<memref::DimOp>(loc, out, 1));

  return rewriter.create<gcu::MatrixStoreOp>(loc, adaptedResult, outPtr,
                                             memDims, realDims);
}

void removeRedundantZeroFill(ConversionPatternRewriter &rewriter,
                             memref::AllocOp allocOp) {
  if (!allocOp)
    return;
  for (auto *user :
       llvm::make_early_inc_range(allocOp.getResult().getUsers())) {
    auto rcOp = dyn_cast<memref::ReinterpretCastOp>(user);
    if (!rcOp)
      continue;

    SmallVector<Operation *> chain;
    SmallPtrSet<Operation *, 16> visited;
    scf::ForOp zeroFillForOp;
    std::function<void(Operation *)> collect = [&](Operation *o) {
      if (!visited.insert(o).second)
        return;
      for (auto r : o->getResults())
        for (auto *u : r.getUsers())
          collect(u);
      if (auto f = dyn_cast<scf::ForOp>(o))
        zeroFillForOp = f;
      chain.push_back(o);
    };
    collect(rcOp);

    if (!zeroFillForOp)
      continue;

    bool isZeroFill = true;
    bool hasBodyOps = false;
    for (auto &bodyOp : zeroFillForOp.getBody()->without_terminator()) {
      hasBodyOps = true;
      auto tarStore = dyn_cast<gcu::TarStoreOp>(&bodyOp);
      if (!tarStore) {
        isZeroFill = false;
        break;
      }
      auto bc = tarStore.getV().getDefiningOp<vector::BroadcastOp>();
      if (!bc) {
        isZeroFill = false;
        break;
      }
      auto cst = bc.getSource().getDefiningOp<arith::ConstantOp>();
      if (!cst) {
        isZeroFill = false;
        break;
      }
      if (auto fAttr = dyn_cast<FloatAttr>(cst.getValue())) {
        if (!fAttr.getValue().isZero()) {
          isZeroFill = false;
          break;
        }
      } else if (auto iAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
        if (!iAttr.getValue().isZero()) {
          isZeroFill = false;
          break;
        }
      } else {
        isZeroFill = false;
        break;
      }
    }

    if (!isZeroFill || !hasBodyOps)
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "removeRedundantZeroFill: removing zero-fill chain for "
                  "constant-zero init arg\n");
    for (auto *o : chain)
      rewriter.eraseOp(o);
  }
}

void moveDeallocOp(ConversionPatternRewriter &rewriter, Value v, Operation *pos,
                   size_t depth) {
  if (depth > 1)
    return;

  Operation *allocOp = v.getDefiningOp();
  if (llvm::isa_and_nonnull<mlir::UnrealizedConversionCastOp>(allocOp)) {
    // not define in current block;
    return;
  }
  unsigned operandIdx = cast<OpResult>(v).getResultNumber();
  while (allocOp && !mlir::isa<memref::AllocOp>(allocOp)) {
    mlir::TypeSwitch<mlir::Operation *>(allocOp)
        .Case<memref::ReinterpretCastOp, memref::MemorySpaceCastOp>(
            [&](auto castOp) {
              allocOp = castOp.getSource().getDefiningOp();
              operandIdx = cast<OpResult>(castOp.getSource()).getResultNumber();
            })
        .Case<scf::ForOp>([&](auto forOp) {
          auto yieldOp =
              llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          Value operand = yieldOp.getOperands()[operandIdx];
          if (rewriter.getRemappedValue(operand)) {
            operand = rewriter.getRemappedValue(operand);
          }
          allocOp = operand.getDefiningOp();
          operandIdx = cast<OpResult>(operand).getResultNumber();

          Value initValue = forOp.getInitArgs()[operandIdx];
          if (rewriter.getRemappedValue(initValue)) {
            initValue = rewriter.getRemappedValue(initValue);
          }
          moveDeallocOp(rewriter, initValue, pos, ++depth);
        })
        .Case<scf::IfOp>([&](auto ifOp) {
          auto thenYieldOp = ifOp.thenYield();
          Value operand = thenYieldOp.getOperands()[operandIdx];
          if (rewriter.getRemappedValue(operand)) {
            operand = rewriter.getRemappedValue(operand);
          }
          allocOp = operand.getDefiningOp();
          operandIdx = cast<OpResult>(operand).getResultNumber();

          if (ifOp.getNumRegions() > 1) {
            auto elseYieldOp = ifOp.elseYield();
            operand = elseYieldOp.getOperands()[operandIdx];
            if (rewriter.getRemappedValue(operand)) {
              operand = rewriter.getRemappedValue(operand);
            }
            moveDeallocOp(rewriter, operand, pos, ++depth);
          }
        })
        .Default([&](auto op) { allocOp = nullptr; });
  }
  if (!allocOp)
    llvm_unreachable("can't find allocation position");

  Operation *deallocOp = nullptr;
  for (const auto &user : allocOp->getUsers()) {
    if (llvm::isa<memref::DeallocOp>(user)) {
      deallocOp = user;
      break;
    }
  }
  if (deallocOp && deallocOp->getBlock() == pos->getBlock() &&
      deallocOp->isBeforeInBlock(pos)) {
    deallocOp->moveAfter(pos);
  }
}

namespace triton {
namespace gcu {

namespace {

/// Map the tag SSA value (same as the corresponding warp_specialize operand) to
/// the partition entry block argument. Operand index matches block arg index
/// for explicit captures;
template <typename WarpSpecializeOpTy>
Value lookupPartitionTagArg(WarpSpecializeOpTy wsOp, Operation *op,
                            Value tagMemref) {
  Region *opRegion = op->getParentRegion();
  if (!opRegion)
    return {};
  for (Region *part : wsOp.getPartitionRegions()) {
    if (!part->isAncestor(opRegion))
      continue;
    if (part->empty())
      return {};
    for (unsigned i = 0; i < wsOp.getNumOperands(); ++i) {
      if (wsOp.getOperand(i) == tagMemref)
        return part->getArgument(i);
    }
    captureValuesToWarpSpecializeOp(wsOp, tagMemref);
    return part->getArgument(wsOp.getNumOperands() - 1);
  }
  return {};
}

#if TRITON_VERSION >= 37
// In Triton 3.7, triton::gpu::WarpSpecializeOp operands moved to PartitionsOp.
template <>
Value lookupPartitionTagArg<triton::gpu::WarpSpecializeOp>(
    triton::gpu::WarpSpecializeOp wsOp, Operation *op, Value tagMemref) {
  Region *opRegion = op->getParentRegion();
  if (!opRegion)
    return {};
  auto partOp = wsOp.getPartitionOp();
  for (Region *part : wsOp.getPartitionRegions()) {
    if (!part->isAncestor(opRegion))
      continue;
    if (part->empty())
      return {};
    for (unsigned i = 0; i < partOp.getNumOperands(); ++i) {
      if (partOp.getOperand(i) == tagMemref)
        return part->getArgument(i);
    }
    captureValuesToWarpSpecializeOp(wsOp, tagMemref);
    return part->getArgument(partOp.getNumOperands() - 1);
  }
  return {};
}
#endif

} // namespace

DTETagPool::DTETagPool(mlir::Operation *entryFunc, int32_t numWarps,
                       bool useAsyncSharedTag, bool useAllTags)
    : useAsyncSharedTag(useAsyncSharedTag) {
  OpBuilder builder(entryFunc);
  auto func = llvm::dyn_cast<FunctionOpInterface>(entryFunc);
  auto firstOp = &func.getFunctionBody().getBlocks().front().front();
  int32_t totalTagsSize =
      ((numWarps <= maxWarpPerSlot && useAllTags) || numWarps > maxWarpPerSlot)
          ? dteTagSizePerSip
          : dteTagSizePerSip / kernelSlotPerSip;
  if (this->useAsyncSharedTag) {
    pTagsSize = 1;
    sTagsSize = totalTagsSize - numWarps;
  } else {
    sTagsSize = 1;
    pTagsSize = (totalTagsSize - 1) / numWarps;
  }

  pTagsPeakSize = 1;
  pTagsBitset = std::vector<bool>(pTagsSize, false);
  pTagsBitset[0] = true;

  sTagsPeakSize = 1;
  sTagsBitset = std::vector<bool>(sTagsSize, false);
  sTagsBitset[0] = true;

  builder.setInsertionPoint(firstOp);
  auto pTagsType =
      MemRefType::get(ArrayRef<int64_t>{pTagsSize}, builder.getI32Type());
  pTagsAllocOp =
      builder.create<memref::AllocOp>(entryFunc->getLoc(), pTagsType);
  pTagsAllocOp->setAttr("gcu_private_tag", builder.getUnitAttr());

  auto sTagsType =
      MemRefType::get(ArrayRef<int64_t>{sTagsSize}, builder.getI32Type());
  sTagsAllocOp =
      builder.create<memref::AllocOp>(entryFunc->getLoc(), sTagsType);
  sTagsAllocOp->setAttr("gcu_shared_tag", builder.getUnitAttr());

  this->usedSyncSharedTag = false;
}

TagInfo DTETagPool::getPrivateSyncTagInfo(mlir::Operation *op) {
  auto loc = op->getLoc();
  auto builder = OpBuilder(op);
  auto tags = getPrivateTagsValue(op);
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  return TagInfo(tags, zero, false);
}

TagInfo DTETagPool::tryGetPrivateAsyncTagInfo(mlir::Operation *op) {
  int32_t idx = -1;
  for (int32_t i = 1; i < pTagsSize; i++) {
    if (!pTagsBitset[i]) {
      pTagsPeakSize = pTagsPeakSize < (i + 1) ? (i + 1) : pTagsPeakSize;
      idx = i;
      break;
    }
  }

  auto loc = op->getLoc();
  auto builder = OpBuilder(op);
  auto tags = getPrivateTagsValue(op);
  if (idx == -1) {
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    return TagInfo(tags, zero, false);
  } else {
    pTagsBitset[idx] = true;
    auto idxValue = builder.create<arith::ConstantIndexOp>(loc, idx);
    return TagInfo(tags, idxValue, false);
  }
}

TagInfo DTETagPool::getSharedSyncTagInfo(mlir::Operation *op) {
  auto loc = op->getLoc();
  auto builder = OpBuilder(op);
  auto tags = getSharedTagsValue(op);
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  usedSyncSharedTag = true;
  return TagInfo(tags, zero, true);
}

TagInfo DTETagPool::tryGetSharedAsyncTagInfo(mlir::Operation *op) {
  int32_t idx = -1;
  for (int32_t i = 1; i < sTagsSize; i++) {
    if (!sTagsBitset[i]) {
      sTagsPeakSize = sTagsPeakSize < (i + 1) ? (i + 1) : sTagsPeakSize;
      idx = i;
      break;
    }
  }

  auto loc = op->getLoc();
  auto builder = OpBuilder(op);
  auto tags = getSharedTagsValue(op);
  if (idx == -1) {
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    usedSyncSharedTag = true;
    return TagInfo(tags, zero, true);
  } else {
    sTagsBitset[idx] = true;
    auto idxValue = builder.create<arith::ConstantIndexOp>(loc, idx);
    return TagInfo(tags, idxValue, true);
  }
}

void DTETagPool::setMap(Operation *op, TagInfo tagInfo) {
  op2TagInfoMap[op].push_back(tagInfo);
}

bool DTETagPool::isExistInMap(Operation *op) const {
  return op2TagInfoMap.count(op) != 0;
}

void DTETagPool::releaseMap(Operation *op) {
  if (op2TagInfoMap.count(op) == 0)
    return;

  auto tags = op2TagInfoMap[op];
  for (size_t i = 0; i < tags.size(); ++i) {
    if (tags[i].isSharedTag()) {
      sTagsBitset[tags[i].getIdxInt()] = false;
    } else {
      pTagsBitset[tags[i].getIdxInt()] = false;
    }
  }
  op2TagInfoMap.erase(op);
}

Type DTETagPool::getPrivateTagsType() {
  return pTagsAllocOp->getResult(0).getType();
}

Type DTETagPool::getSharedTagsType() {
  return sTagsAllocOp->getResult(0).getType();
}

Value DTETagPool::getPrivateTagsValue(Operation *op) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  assert(func && "can't find func op");

  Value pTagsMemref = pTagsAllocOp->getResult(0);
  if (auto wsOp = op->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
    if (!wsOp->isProperAncestor(pTagsAllocOp))
      if (Value v = lookupPartitionTagArg(wsOp, op, pTagsMemref))
        return v;
  }
  if (auto wsOp = op->getParentOfType<mlir::gcu::WarpSpecializeOp>()) {
    if (!wsOp->isProperAncestor(pTagsAllocOp))
      if (Value v = lookupPartitionTagArg(wsOp, op, pTagsMemref))
        return v;
  }

  if (pTagsArgPosMap.count(func.getName()) != 0)
    return func.getArgument(pTagsArgPosMap[func.getName()]);
  return pTagsMemref;
}

Value DTETagPool::getSharedTagsValue(Operation *op) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  assert(func && "can't find func op");

  Value sTagsMemref = sTagsAllocOp->getResult(0);
  if (auto wsOp = op->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
    if (!wsOp->isProperAncestor(sTagsAllocOp))
      if (Value v = lookupPartitionTagArg(wsOp, op, sTagsMemref))
        return v;
  }
  if (auto wsOp = op->getParentOfType<mlir::gcu::WarpSpecializeOp>()) {
    if (!wsOp->isProperAncestor(sTagsAllocOp))
      if (Value v = lookupPartitionTagArg(wsOp, op, sTagsMemref))
        return v;
  }

  if (sTagsArgPosMap.count(func.getName()) != 0)
    return func.getArgument(sTagsArgPosMap[func.getName()]);

  return sTagsMemref;
}

void DTETagPool::updateUsedSize() {
  OpBuilder builder(pTagsAllocOp);
  auto loc = pTagsAllocOp->getLoc();

  (void)useAsyncSharedTag;
  auto pTagsType =
      MemRefType::get(ArrayRef<int64_t>{pTagsPeakSize}, builder.getI32Type());
  auto newPTagsAllocOp = builder.create<memref::AllocOp>(loc, pTagsType);
  newPTagsAllocOp->setAttr("gcu_private_tag", builder.getUnitAttr());

  pTagsAllocOp->getResult(0).replaceAllUsesWith(newPTagsAllocOp->getResult(0));
  pTagsAllocOp->dropAllUses();
  pTagsAllocOp->erase();

  builder.setInsertionPoint(sTagsAllocOp);
  loc = sTagsAllocOp->getLoc();
  auto sTagsType =
      MemRefType::get(ArrayRef<int64_t>{sTagsPeakSize}, builder.getI32Type());
  auto newSTagsAllocOp = builder.create<memref::AllocOp>(loc, sTagsType);
  newSTagsAllocOp->setAttr("gcu_shared_tag", builder.getUnitAttr());

  if (!usedSyncSharedTag) {
    newSTagsAllocOp->setAttr("unused_gcu_sync_shared_tag",
                             builder.getUnitAttr());
  }

  sTagsAllocOp->getResult(0).replaceAllUsesWith(newSTagsAllocOp->getResult(0));
  sTagsAllocOp->dropAllUses();
  sTagsAllocOp->erase();

  mlir::gpu::GPUModuleOp moduleOp =
      newPTagsAllocOp->getParentOfType<mlir::gpu::GPUModuleOp>();
  if (!moduleOp) {
    llvm::report_fatal_error("can't find GPUModuleOp for tags");
  }

  moduleOp.walk([&](mlir::gcu::WarpSpecializeOp wsOp) {
    for (Region *region : wsOp.getPartitionRegions()) {
      for (Block &block : *region) {
        for (auto [idx, arg] : llvm::enumerate(block.getArguments())) {
          if (idx >= wsOp->getNumOperands())
            continue;
          Type captureType = wsOp->getOperand(idx).getType();
          if (arg.getType() == captureType)
            continue;
          if (isa<MemRefType>(arg.getType()) && isa<MemRefType>(captureType))
            arg.setType(captureType);
        }
      }
    }
  });

  auto updateFuncType = [](func::FuncOp funcOp, int32_t argPos, Type newType) {
    if (static_cast<unsigned>(argPos) >= funcOp.getNumArguments()) {
      llvm::report_fatal_error("arg index out of range for tags");
      return;
    }

    FunctionType currentType = funcOp.getFunctionType();

    SmallVector<Type> newInputTypes = llvm::to_vector(currentType.getInputs());
    newInputTypes[argPos] = newType;

    FunctionType newFuncType = FunctionType::get(
        funcOp.getContext(), newInputTypes, currentType.getResults());
    funcOp.setType(newFuncType);

    Block &entryBlock = funcOp.getBody().front();
    entryBlock.getArgument(argPos).setType(newType);
  };

  for (auto &[funcName, argPos] : pTagsArgPosMap) {
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      llvm::report_fatal_error(
          "can't find func op in GPUModuleOp for private tags");
      continue;
    }
    updateFuncType(funcOp, argPos, pTagsType);
  }

  for (auto &[funcName, argPos] : sTagsArgPosMap) {
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
    if (!funcOp) {
      llvm::report_fatal_error(
          "can't find func op in GPUModuleOp for shared tags");
      continue;
    }
    updateFuncType(funcOp, argPos, sTagsType);
  }
}

void DTETagPool::setPrivateFuncNameMap(Operation *op, int argNum) {
  assert(llvm::isa<FunctionOpInterface>(op) && "not a triton func op");
  auto func = llvm::cast<FunctionOpInterface>(op);
  assert(argNum >= 0 && "arg number is negative");
  assert(static_cast<unsigned>(argNum) < func.getNumArguments() &&
         "arg number is too big");
  pTagsArgPosMap[func.getName()] = argNum;
}

void DTETagPool::setSharedFuncNameMap(Operation *op, int argNum) {
  assert(llvm::isa<FunctionOpInterface>(op) && "not a triton func op");
  auto func = llvm::cast<FunctionOpInterface>(op);
  assert(argNum >= 0 && "arg number is negative");
  assert(static_cast<unsigned>(argNum) < func.getNumArguments() &&
         "arg number is too big");
  sTagsArgPosMap[func.getName()] = argNum;
}

bool get_bool_env(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string str_value(value);
  std::transform(str_value.begin(), str_value.end(), str_value.begin(),
                 ::tolower);
  return (str_value == "true" || str_value == "1" || str_value == "on" ||
          str_value == "yes");
}

Value createConstantZero(OpBuilder &builder, Location loc, Type elemType) {
  if (elemType.isIntOrIndex()) {
    return builder.create<arith::ConstantIntOp>(
        loc, 0, elemType.getIntOrFloatBitWidth());
  } else if (elemType.isF32()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::IEEEsingle(), "0"));
  } else if (elemType.isF16()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::IEEEhalf(), "0"));
  } else if (elemType.isBF16()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::BFloat(), "0"));
  } else if (elemType.isF64()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::IEEEdouble(), "0"));
  } else if (llvm::isa<Float8E4M3B11FNUZType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::Float8E4M3B11FNUZ(), "0"));
  } else if (llvm::isa<Float8E4M3FNUZType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::Float8E4M3FNUZ(), "0"));
  } else if (llvm::isa<Float8E5M2FNUZType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::Float8E5M2FNUZ(), "0"));
  } else if (llvm::isa<Float8E4M3FNType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::Float8E4M3FN(), "0"));
  } else if (llvm::isa<Float8E5M2Type>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, dyn_cast<FloatType>(elemType),
        APFloat(llvm::APFloatBase::Float8E5M2(), "0"));
  } else {
    std::string o;
    llvm::raw_string_ostream os(o);
    elemType.print(os);
    llvm_unreachable((o + " is unsupported").c_str());
  }
  return Value();
}

SmallVector<unsigned> getWarpsPerCTA(Attribute layout) {
  if (auto blockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(layout)) {
    auto warpsPerCTA = blockEnc.getWarpsPerCTA();
    return SmallVector<unsigned>(warpsPerCTA.begin(), warpsPerCTA.end());
  } else if (auto sliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parent = sliceEnc.getParent();
    SmallVector<unsigned> sliceDims;
    sliceDims.push_back(sliceEnc.getDim());
    while (auto innerSliceEnc =
               dyn_cast<triton::gpu::SliceEncodingAttr>(parent)) {
      auto curSliceDim = innerSliceEnc.getDim();
      for (size_t idx = 0; idx < sliceDims.size(); idx++) {
        if (sliceDims[idx] >= curSliceDim) {
          sliceDims[idx] = sliceDims[idx] + 1;
        }
      }
      sliceDims.push_back(curSliceDim);
      parent = innerSliceEnc.getParent();
    }
    SmallVector<unsigned> parentWarpsPerCTA;
    if (auto blockEncParent =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(parent)) {
      parentWarpsPerCTA =
          SmallVector<unsigned>(blockEncParent.getWarpsPerCTA());
    } else if (auto linearEncParent =
                   dyn_cast<triton::gpu::LinearEncodingAttr>(parent)) {
      parentWarpsPerCTA =
          SmallVector<unsigned>(linearEncParent.getWarpsPerCTA());
    } else {
      llvm::report_fatal_error("[Error] bad slice layout parent");
      assert(false && "bad slice layout parent");
    }
    SmallVector<unsigned> warpsPerCTA;
    for (unsigned i = 0; i < parentWarpsPerCTA.size(); ++i) {
      if (!llvm::is_contained(sliceDims, i)) {
        warpsPerCTA.push_back(parentWarpsPerCTA[i]);
      }
    }
    return warpsPerCTA;

  } else if (auto linearEnc =
                 dyn_cast<triton::gpu::LinearEncodingAttr>(layout)) {
    auto warpsPerCTA = linearEnc.getWarpsPerCTA();
    return SmallVector<unsigned>(warpsPerCTA.begin(), warpsPerCTA.end());
  } else {
    assert(false && "not supported layout");
  }
  return SmallVector<unsigned>();
}

bool needsSmemRelay(RankedTensorType srcTy, ArrayRef<int64_t> tileShape) {
  auto encoding = srcTy.getEncoding();
  if (!encoding)
    return true;

  SmallVector<unsigned> warpsPerCTA;
  if (auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding))
    warpsPerCTA = SmallVector<unsigned>(blocked.getWarpsPerCTA());
  else if (auto linear = dyn_cast<triton::gpu::LinearEncodingAttr>(encoding))
    warpsPerCTA = SmallVector<unsigned>(linear.getWarpsPerCTA());
  else
    return true;

  auto srcShape = srcTy.getShape();
  unsigned rank = srcShape.size();
  if (rank != warpsPerCTA.size())
    return true;

  for (unsigned i = 0; i < rank; ++i) {
    if (tileShape[i] != srcShape[i] && warpsPerCTA[i] != 1)
      return true;
  }
  return false;
}

bool isExpensiveView(triton::ExpandDimsOp op) {
  auto srcNumElems = triton::gcu::getElemsPerThread(op.getSrc().getType());
  auto dstNumElems = triton::gcu::getElemsPerThread(op.getType());
  srcNumElems.insert(srcNumElems.begin() + op.getAxis(), 1);
  return srcNumElems != dstNumElems;
}

bool isExpensiveView(Type srcType, Type dstType) {
  auto mergeContig = [](RankedTensorType type) {
    auto elemsPerThread = getElemsPerThread(type);
    auto warpsPerCTA = getWarpsPerCTA(type.getEncoding());
    SmallVector<unsigned> mergedElemsPerThread;
    unsigned acc = 1;
    for (int i = elemsPerThread.size() - 1; i >= 0; --i) {
      acc *= elemsPerThread[i];
      if (warpsPerCTA[i] != 1) {
        mergedElemsPerThread.push_back(acc);
        acc = 1;
      }
    }
    if (acc != 1) {
      mergedElemsPerThread.push_back(acc);
    }
    std::reverse(mergedElemsPerThread.begin(), mergedElemsPerThread.end());
    return mergedElemsPerThread;
  };
  if (triton::gpu::isExpensiveView(srcType, dstType)) {
    return true;
  }
  return mergeContig(cast<RankedTensorType>(srcType)) !=
         mergeContig(cast<RankedTensorType>(dstType));
}

SmallVector<unsigned> getElemsPerThread(Type type) {
  if (auto tType = dyn_cast<RankedTensorType>(type)) {
    if (auto dotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
            tType.getEncoding())) {
      // dot lhs and rhs should have different slicing by op id but
      // DotOperandEncodingAttr no supported and currently support 2D dot first
      auto shape = tType.getShape();
      if (auto blockedLayout =
              dyn_cast<triton::gpu::BlockedEncodingAttr>(dotEnc.getParent())) {
        auto rank = shape.size();
        SmallVector<unsigned> elemsPerthread(rank, 1);
        // low 2 rank do dot
        auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
        for (unsigned idx = 0; idx < rank - 2; idx++) {
          elemsPerthread[idx] = shape[idx];
          if (warpsPerCTA[idx] > 1) {
            LLVM_DEBUG({
              llvm::dbgs() << "hi slice should in lower 2 dims for dot \n";
              dotEnc.dump();
            });
          }
          assert((warpsPerCTA[idx] == 1) &&
                 "hi slice should in lower 2 dims for dot\n");
        }
        bool isM = dotEnc.getOpIdx() == 0;
        // only debug check
        if (isM) {
          int64_t k = shape[rank - 1];
          elemsPerthread[rank - 1] = k;
          elemsPerthread[rank - 2] = shape[rank - 2] / warpsPerCTA[rank - 2];
        } else {
          if (dotEnc.getColumnMajor()) {
            // Column-major rhs: shared memory is [N, K].
            // K (reduction) at rank-1: each warp needs all elements.
            // N (non-reduction) at rank-2: split by N's warpsPerCTA.
            int64_t k = shape[rank - 1];
            elemsPerthread[rank - 1] = k;
            elemsPerthread[rank - 2] = shape[rank - 2] / warpsPerCTA[rank - 1];
          } else {
            // Row-major rhs: shared memory is [K, N].
            int64_t k = shape[rank - 2];
            elemsPerthread[rank - 2] = k;
            elemsPerthread[rank - 1] = shape[rank - 1] / warpsPerCTA[rank - 1];
          }
        }
        return elemsPerthread;
      }
    } else if (mlir::isa<triton::gpu::SharedEncodingTrait>(
                   tType.getEncoding())) {
      return SmallVector<unsigned>(tType.getShape().begin(),
                                   tType.getShape().end());
    } else if (auto blockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(
                   tType.getEncoding())) {
      auto shape = tType.getShape();
      size_t rank = shape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      auto warpsPerCTA = blockEnc.getWarpsPerCTA();
      auto threadsPerWarp = blockEnc.getThreadsPerWarp();
      auto shapePerCTA = triton::gpu::getShapePerCTA(blockEnc, shape);
      assert(rank == sizePerThread.size() &&
             "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
      SmallVector<unsigned> elemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        elemsPerThread[i] =
            ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
      }
      return elemsPerThread;
    } else if (auto linearEnc = dyn_cast<triton::gpu::LinearEncodingAttr>(
                   tType.getEncoding())) {
      auto shape = tType.getShape();
      size_t rank = shape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      auto warpsPerCTA = linearEnc.getWarpsPerCTA();
      auto threadsPerWarp = linearEnc.getThreadsPerWarp();
      assert(rank == sizePerThread.size() &&
             "unexpected rank in LinearEncodingAttr::getElemsPerThread");
      SmallVector<unsigned> elemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        elemsPerThread[i] = ceil<unsigned>(shape[i], t) * sizePerThread[i];
      }
      return elemsPerThread;
    } else if (auto sliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(
                   tType.getEncoding())) {
      auto parent = sliceEnc.getParent();
      auto outShape = sliceEnc.paddedShape(tType.getShape());
      SmallVector<unsigned> sliceDims;
      sliceDims.push_back(sliceEnc.getDim());
      while (auto innerSliceEnc =
                 dyn_cast<triton::gpu::SliceEncodingAttr>(parent)) {
        llvm::ArrayRef<int64_t> inputShpe = outShape;
        outShape = innerSliceEnc.paddedShape(inputShpe);
        auto curSliceDim = innerSliceEnc.getDim();
        for (size_t idx = 0; idx < sliceDims.size(); idx++) {
          if (sliceDims[idx] >= curSliceDim) {
            sliceDims[idx] = sliceDims[idx] + 1;
          }
        }
        sliceDims.push_back(curSliceDim);
        parent = innerSliceEnc.getParent();
      }
      if (!isa<triton::gpu::BlockedEncodingAttr>(parent)) {
        return triton::gpu::getElemsPerThread(type);
      }
      auto blockEncParent = dyn_cast<triton::gpu::BlockedEncodingAttr>(parent);
      size_t rank = outShape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      auto warpsPerCTA = blockEncParent.getWarpsPerCTA();
      auto threadsPerWarp = blockEncParent.getThreadsPerWarp();
      auto shapePerCTA = triton::gpu::getShapePerCTA(blockEncParent, outShape);
      assert(rank == sizePerThread.size() &&
             "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
      SmallVector<unsigned> parentElemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        parentElemsPerThread[i] =
            ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
      }
      SmallVector<unsigned> elemsPerThread;
      for (unsigned i = 0; i < rank; ++i) {
        if (!llvm::is_contained(sliceDims, i)) {
          elemsPerThread.push_back(parentElemsPerThread[i]);
        }
      }
      return elemsPerThread;
    } else {
      return triton::gpu::getElemsPerThread(type);
    }
  }
  return triton::gpu::getElemsPerThread(type);
}

unsigned getTotalElemsPerThread(Type type) {
  if (auto tType = dyn_cast<RankedTensorType>(type)) {
    if (auto enc = tType.getEncoding()) {
      if (llvm::isa_and_nonnull<triton::gpu::DotOperandEncodingAttr>(enc)) {
        auto elemsPerthread = gcu::getElemsPerThread(type);
        return std::accumulate(elemsPerthread.begin(), elemsPerthread.end(), 1,
                               std::multiplies<unsigned>());
      } else if (mlir::isa<triton::gpu::SharedEncodingTrait>(enc)) {
        return std::accumulate(tType.getShape().begin(), tType.getShape().end(),
                               1, std::multiplies<unsigned>());
      } else if (llvm::isa_and_nonnull<triton::gpu::BlockedEncodingAttr>(
                     tType.getEncoding()) ||
                 llvm::isa_and_nonnull<triton::gpu::LinearEncodingAttr>(
                     tType.getEncoding())) {
        auto elemsPerthread = gcu::getElemsPerThread(type);
        return std::accumulate(elemsPerthread.begin(), elemsPerthread.end(), 1,
                               std::multiplies<unsigned>());

      } else if (llvm::isa_and_nonnull<triton::gpu::SliceEncodingAttr>(
                     tType.getEncoding())) {
        auto elemsPerthread = gcu::getElemsPerThread(type);
        return std::accumulate(elemsPerthread.begin(), elemsPerthread.end(), 1,
                               std::multiplies<unsigned>());
      } else {
        return triton::gpu::getTotalElemsPerThread(type);
      }
    }
  }
  return triton::gpu::getTotalElemsPerThread(type);
}

int getNumWarps(ModuleOp mod) {
  if (!mod->hasAttr(kNumWarps))
    llvm::report_fatal_error(
        "TritonGPU module should contain a ttg.num-warps attribute");
  return cast<IntegerAttr>(mod->getAttr(kNumWarps)).getInt();
}

int getNumWarps(Operation *op) {
  if (auto mod = op->getParentOfType<ModuleOp>())
    return getNumWarps(mod);
  return 1;
}

int64_t getOaccMaxNum(Operation *op) {
  int numWarps = getNumWarps(op);
  return numWarps == 1 ? OACC_MAX_NUM * 2 : OACC_MAX_NUM;
}

// Some ops keep every input live while producing their result, and the result
// cannot share/overwrite an input register:
//   - multi-operand extern elementwise (mac/mas/imas/wadd/add/wmul/mul)
//   - select (condition + both arms)
//   - conversions (the result has a different element type/width)
static bool opCannotReuseInputRegister(Operation *o) {
  if (auto ext = dyn_cast<triton::ExternElementwiseOp>(o)) {
    StringRef sym = ext.getSymbol();
    for (StringRef prefix :
         {"__gcu_mac", "__gcu_mas", "__gcu_imas", "__gcu_wadd", "__gcu_add",
          "__gcu_wmul", "__gcu_mul"})
      if (sym.starts_with(prefix))
        return true;
    return false;
  }
  return isa<arith::SelectOp, arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp,
             arith::ExtUIOp, arith::TruncIOp, arith::FPToSIOp, arith::SIToFPOp,
             arith::UIToFPOp, arith::FPToUIOp, arith::IndexCastOp, math::SqrtOp,
             math::RsqrtOp, math::ExpOp, math::Exp2Op, math::LogOp,
             math::Log2Op, math::SinOp, math::CosOp, math::TanOp, math::PowFOp,
             math::FloorOp, math::CeilOp, math::RoundOp, math::RoundEvenOp>(o);
}

unsigned
computeFusionLivePerStep(triton::gcu::ElementwiseFusionRegionOp op,
                         const llvm::SmallPtrSetImpl<Value> &excluded) {
  Block *body = op.getBody();
  SmallVector<Operation *> ops;
  for (Operation &o : body->without_terminator())
    ops.push_back(&o);
  unsigned numOps = ops.size();

  // birth: index at which a tensor value starts occupying a register.
  // Region arguments and values captured from above are materialized at region
  // entry (birth 0), not at their first use: the lowering has to keep them
  // available for the whole chain.
  DenseMap<Value, unsigned> birth, death;
  for (unsigned i = 0; i < numOps; ++i)
    for (Value result : ops[i]->getResults())
      if (isa<TensorType>(result.getType()) && !excluded.contains(result))
        birth.try_emplace(result, i);

  auto noteUses = [&](Operation *o, unsigned idx) {
    for (Value operand : o->getOperands()) {
      if (!isa<TensorType>(operand.getType()) || excluded.contains(operand))
        continue;
      if (!birth.count(operand))
        birth[operand] = 0;
      death[operand] = idx;
    }
  };
  for (unsigned i = 0; i < numOps; ++i)
    noteUses(ops[i], i);
  noteUses(body->getTerminator(), numOps);

  auto isLiveAt = [&](Value v, unsigned def, unsigned i) {
    auto it = death.find(v);
    unsigned end = it == death.end() ? numOps : it->second;
    return def < i && end >= i;
  };

  // Values present at region entry.
  unsigned liveMax = 0;
  for (auto &entry : birth)
    if (entry.second == 0)
      ++liveMax;

  for (unsigned i = 0; i < numOps; ++i) {
    unsigned liveBefore = 0;
    for (auto &[value, def] : birth)
      if (isLiveAt(value, def, i))
        ++liveBefore;

    unsigned outs = 0;
    for (Value result : ops[i]->getResults())
      if (isa<TensorType>(result.getType()) && !excluded.contains(result))
        ++outs;
    if (outs == 0) {
      liveMax = std::max(liveMax, liveBefore);
      continue;
    }

    unsigned pressure;
    if (ops[i]->getNumOperands() < 3 && opCannotReuseInputRegister(ops[i])) {
      // Every input stays live while the result is written.
      pressure = liveBefore + outs;
    } else {
      // Elementwise/math ops may overwrite an operand whose last use is here.
      SmallPtrSet<Value, 4> dying;
      for (Value operand : ops[i]->getOperands()) {
        if (!isa<TensorType>(operand.getType()) || excluded.contains(operand))
          continue;
        auto it = death.find(operand);
        if (it != death.end() && it->second == i)
          dying.insert(operand);
      }
      unsigned liveAfter =
          liveBefore - std::min(liveBefore, (unsigned)dying.size()) + outs;
      pressure = std::max(liveBefore, liveAfter);
    }
    liveMax = std::max(liveMax, pressure);
  }
  return liveMax;
}

FusionUnrollPlan
computeFusionUnrollPlan(triton::gcu::ElementwiseFusionRegionOp op,
                        unsigned vectorLength, unsigned maxBpe,
                        unsigned tripCount, Value forcedPublishedValue) {
  FusionUnrollPlan plan;
  plan.oaccPerValue = std::max(1u, (vectorLength * maxBpe) / kOaccSizeInBytes);

  // A dot operand with acc_store=none stays in OACC until this region consumes
  // it (a 128x128 f32 accumulator = 128 of the per-slot 192 OACC), so charge
  // it against the budget and pick the largest power-of-two unroll whose
  // live-per-copy set still fits.
  unsigned reservedOacc = 0;
  SmallPtrSet<Value, 4> countedAccs;
  SmallPtrSet<Value, 4> publishedAccSet;
  for (Value operand : op.getOperands()) {
    if (!countedAccs.insert(operand).second)
      continue;
    // A loop-carried accumulator reaches the region as the enclosing
    // scf.for's result (dot result -> scf.yield -> for result); resolve it
    // back to the yielded value so the reservation sees the dot.
    Value accValue = operand;
    if (auto res = dyn_cast<OpResult>(operand)) {
      if (auto forOp = dyn_cast<scf::ForOp>(res.getOwner());
          forOp && !forOp.getRegion().empty()) {
        // The plan also runs mid-conversion, where a loop body may not have
        // its terminator yet -- read the block tail instead of calling
        // getTerminator() (which asserts in that state).
        Block &body = forOp.getRegion().front();
        unsigned resultNo = res.getResultNumber();
        if (!body.empty())
          if (auto yield = dyn_cast<scf::YieldOp>(&body.back()))
            if (resultNo < yield.getNumOperands())
              accValue = yield.getOperand(resultNo);
      }
    }
    auto dotOp = dyn_cast_or_null<triton::DotOp>(accValue.getDefiningOp());
    if (!dotOp)
      continue;
    // The annotator asks about a dot it is about to publish; at that point
    // acc_store is still `local`, so it names the value to model it as if it
    // were already published. The lowering does not pass this: by then the
    // dot carries acc_store=none and is counted by the check below.
    bool forcedPublished =
        forcedPublishedValue && operand == forcedPublishedValue;
    auto accStore = dotOp->getAttrOfType<StringAttr>(kAccStore);
    if (!forcedPublished && (!accStore || accStore.getValue() != kAccStoreNone))
      continue;
    auto accTy = dyn_cast<RankedTensorType>(dotOp.getResult().getType());
    if (!accTy)
      continue;
    unsigned elems = getTotalElemsPerThread(accTy);
    unsigned bpe = getBpe(accTy.getElementType());
    reservedOacc +=
        llvm::divideCeil(elems * bpe, static_cast<unsigned>(kOaccSizeInBytes));
    ++plan.publishedAccs;
    publishedAccSet.insert(operand);
  }
  plan.reservedOacc = reservedOacc;
  plan.livePerStep = computeFusionLivePerStep(op, publishedAccSet);

  // A single-warp program can address both banks (getOaccMaxNum doubles the
  // count); two such programs then occupy the two kernel slots, so one
  // program's share is a single bank's worth.
  int64_t oaccMax = getOaccMaxNum(op.getOperation());
  unsigned oaccSlots = getNumWarps(op.getOperation()) == 1 ? 2u : 1u;
  unsigned oaccBudget = static_cast<unsigned>(oaccMax / oaccSlots);
  plan.freeOacc = reservedOacc < oaccBudget ? oaccBudget - reservedOacc : 1u;

  // A published accumulator reserves OACC in front of the unrolled live set.
  // Fit the largest power-of-two unroll into what is left, capping the base at
  // kLoopUnrollTimes/oaccPerValue.
  unsigned unrollLimit = kLoopUnrollTimes;
  if (plan.oaccPerValue > 1 && reservedOacc > 0) {
    unsigned base = std::max(1u, kLoopUnrollTimes / plan.oaccPerValue);
    unsigned liveCost = std::max(1u, plan.livePerStep) * plan.oaccPerValue;
    unsigned fit = std::max(1u, plan.freeOacc / liveCost);
    unrollLimit = std::max(std::min(base, pow2Floor(fit)),
                           kLoopUnrollTimes / 2 / plan.oaccPerValue);
  }

  plan.loopCnt = std::min(tripCount, unrollLimit);

  // Silicon sweep knob: force the per-region unroll (0/absent = keep the
  // plan).  Used to chart loopCnt against OACC spills and wall time.
  // if (std::getenv("TRITON_GCU_FUSION_PLAN_DEBUG"))
  //   llvm::errs() << "[FUSION-PLAN] loopCnt=" << plan.loopCnt
  //                << " (trip=" << tripCount << ", vl=" << vectorLength
  //                << ", maxBpe=" << maxBpe << ", opv=" << plan.oaccPerValue
  //                << ", live=" << plan.livePerStep
  //                << ", pub=" << plan.publishedAccs
  //                << ", reserved=" << plan.reservedOacc
  //                << ", free=" << plan.freeOacc << ")\n";
  // if (const char *forced = std::getenv("TRITON_GCU_FUSION_LOOPCNT")) {
  //   unsigned forcedCnt = static_cast<unsigned>(std::atoi(forced));
  //   if (forcedCnt > 0) {
  //     plan.loopCnt = std::min(tripCount, forcedCnt);
  //     llvm::errs() << "[FUSION-PLAN] forced loopCnt=" << plan.loopCnt
  //                  << " (trip=" << tripCount << ", vl=" << vectorLength
  //                  << ", maxBpe=" << maxBpe << ", opv=" << plan.oaccPerValue
  //                  << ", live=" << plan.livePerStep
  //                  << ", pub=" << plan.publishedAccs
  //                  << ", reserved=" << plan.reservedOacc << ")\n";
  //   }
  // }
  return plan;
}

int getTotalNumWarps(mlir::gpu::GPUModuleOp mod) {
  if (!mod->hasAttr(kTotalNumWarps))
    return triton::gpu::lookupNumWarps(mod);
  return cast<IntegerAttr>(mod->getAttr(kTotalNumWarps)).getInt();
}

SmallVector<bool> getFreeWarpMask(Type type) {
  auto tType = dyn_cast<RankedTensorType>(type);
  if (!tType)
    return {true};

  SmallVector<unsigned> warps;
  SmallVector<int64_t> shapePerCTA;
  auto encoding = tType.getEncoding();

  if (auto blockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding)) {
    warps = SmallVector<unsigned>(blockEnc.getWarpsPerCTA());
    shapePerCTA = triton::gpu::getShapePerCTA(blockEnc, tType.getShape());
  } else if (auto linearEnc =
                 dyn_cast<triton::gpu::LinearEncodingAttr>(encoding)) {
    warps = SmallVector<unsigned>(linearEnc.getWarpsPerCTA());
    shapePerCTA = triton::gpu::getShapePerCTA(linearEnc, tType.getShape());
  } else if (auto sliceEnc =
                 dyn_cast<triton::gpu::SliceEncodingAttr>(encoding)) {
    auto parent = sliceEnc.getParent();
    auto outShape = sliceEnc.paddedShape(tType.getShape());
    SmallVector<unsigned> sliceDims;
    sliceDims.push_back(sliceEnc.getDim());
    while (auto inner = dyn_cast<triton::gpu::SliceEncodingAttr>(parent)) {
      auto curDim = inner.getDim();
      for (auto &d : sliceDims) {
        if (d >= curDim)
          d++;
      }
      llvm::ArrayRef<int64_t> inShape = outShape;
      outShape = inner.paddedShape(inShape);
      sliceDims.push_back(curDim);
      parent = inner.getParent();
    }
    if (auto blockP = dyn_cast<triton::gpu::BlockedEncodingAttr>(parent)) {
      warps = SmallVector<unsigned>(blockP.getWarpsPerCTA());
      shapePerCTA = triton::gpu::getShapePerCTA(blockP, outShape);
    } else {
      return {true};
    }
  } else {
    return {true};
  }

  unsigned rank = warps.size();
  unsigned totalWarps = 1;
  for (auto w : warps)
    totalWarps *= w;

  auto slicedAxies = getSlicedAxies(type);

  SmallVector<unsigned> effectiveWarps(rank);
  for (unsigned i = 0; i < rank; ++i) {
    if (slicedAxies.count(i)) {
      unsigned s = static_cast<unsigned>(shapePerCTA[i]);
      unsigned repeatNum = s >= warps[i] ? 1 : warps[i] / s;
      effectiveWarps[i] = warps[i] / repeatNum;
    } else {
      effectiveWarps[i] = 1;
    }
  }

  SmallVector<unsigned> warpMods(rank);
  SmallVector<unsigned> warpStrides(rank);
  unsigned warpMod = 1;
  unsigned warpStride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    warpMod *= warps[i];
    warpMods[i] = warpMod;
    warpStrides[i] = warpStride;
    warpStride *= warps[i];
  }

  SmallVector<bool> mask(totalWarps, false);
  for (unsigned w = 0; w < totalWarps; ++w) {
    bool nonRedundant = true;
    for (unsigned i = 0; i < rank; ++i) {
      unsigned warpIdx = (w % warpMods[i]) / warpStrides[i];
      if (warpIdx >= effectiveWarps[i]) {
        nonRedundant = false;
        break;
      }
    }
    mask[w] = nonRedundant;
  }
  return mask;
}

unsigned getBpe(Type type) {
  if (llvm::isa<triton::PointerType>(type)) {
    // use i64 for pointer type
    return 8;
  }
  assert(type.isIntOrFloat());
  return ((type.getIntOrFloatBitWidth() + 7) / 8);
}

Type TritonGCUBuilder::getTarType() {
  return VectorType::get(ArrayRef<int64_t>{1}, builder->getI64Type());
}

Value TritonGCUBuilder::tarAddr(Value memref) {
  // Workaround. The bitwidth of IndexType is 32, so if we use
  // ExtractAlignedPointerAsIndexOp and ExtractStridedMetadataOp, the
  // address will be truncated.
  assert(isa<MemRefType>(memref.getType()));
  auto ptr = builder->create<mlir::gcu::MemRefToPtrOp>(
      loc,
      mlir::gcu::PtrType::get(
          builder->getContext(),
          dyn_cast<MemRefType>(memref.getType()).getElementType()),
      memref);
  auto addr = builder->create<mlir::gcu::PtrToIntOp>(loc, ptr);
  return builder->create<mlir::gcu::TarInitOp>(loc, getTarType(), addr)
      .getOut();
}

Value TritonGCUBuilder::tarValue(int64_t v) {
  return builder
      ->create<mlir::gcu::TarInitOp>(
          loc, getTarType(),
          builder->create<arith::ConstantIntOp>(loc, v, 64).getResult())
      .getOut();
}

Value TritonGCUBuilder::tarValue(Value v) {
  assert(v.getType().isInteger());
  if (!v.getType().isInteger(64)) {
    v = builder->create<arith::ExtSIOp>(loc, builder->getI64Type(), v);
  }
  return builder->create<mlir::gcu::TarInitOp>(loc, getTarType(), v).getOut();
}

Value TritonGCUBuilder::tarStride(VectorType type, int64_t stride) {
  auto getBpe = [](Type type) {
    assert(type.isIntOrFloat());
    return ((type.getIntOrFloatBitWidth() + 7) / 8);
  };
  auto numOacc =
      type.getDimSize(0) * getBpe(type.getElementType()) / oaccSizeInBytes;
  assert(numOacc == 1 || numOacc == 2 || numOacc == 4);
  stride -= (numOacc - 1) * oaccSizeInBytes;
  return builder
      ->create<mlir::gcu::TarInitOp>(
          loc, getTarType(),
          builder->create<arith::ConstantIntOp>(loc, stride, 64).getResult())
      .getOut();
}

Value TritonGCUBuilder::tarLoad(VectorType type, Value &tarAddr,
                                const Value &tarStride) {
  auto tarLoadOp = builder->create<mlir::gcu::TarLoadOp>(
      loc, TypeRange{type, getTarType()}, tarAddr, tarStride);
  tarAddr = tarLoadOp.getDstAddr();
  return tarLoadOp.getV();
}

void TritonGCUBuilder::tarStore(Value v, Value &tarAddr,
                                const Value &tarStride) {
  tarAddr = builder
                ->create<mlir::gcu::TarStoreOp>(loc, getTarType(), v, tarAddr,
                                                tarStride)
                .getDstAddr();
}

Value TritonGCUBuilder::tarGather(VectorType type, Value &tarAddr, Value num,
                                  Value other, Value mask) {
  auto tarGatherOp = builder->create<mlir::gcu::TarGatherOp>(
      loc, TypeRange{type, getTarType()}, tarAddr, num, other, mask);
  tarAddr = tarGatherOp.getDstAddr();
  return tarGatherOp.getV();
}

void TritonGCUBuilder::tarScatter(Value &tarAddr, Value v, Value num,
                                  Value mask) {
  tarAddr = builder
                ->create<mlir::gcu::TarScatterOp>(loc, getTarType(), tarAddr, v,
                                                  num, mask)
                .getDstAddr();
}

void TritonGCUBuilder::tarJump(Value &tarAddr, const Value &tarValue) {
  tarAddr = builder->create<arith::AddIOp>(loc, tarAddr, tarValue);
}

bool isMixedPrecisionSymbol(StringRef symbol) {
  static const std::string mixedPrecisionSymbolPrefixList[] = {
      "__gcu_wadd", "__gcu_add",  "__gcu_wmul",    "__gcu_mul",     "__gcu_mac",
      "__gcu_mas",  "__gcu_imas", "__gcu_sigmoid", "__gcu_softplus"};
  return llvm::any_of(mixedPrecisionSymbolPrefixList,
                      [&symbol](StringRef symbolPrefix) {
                        return symbol.starts_with(symbolPrefix);
                      });
}

static bool isAllocaInputValueImpl(Value v, ValueRemapFn remap) {
  if (!v) {
    return false;
  }
  // In rollback mode the conversion registers the replacement of a value before
  // it reaches the IR (TTDotOpLowering swaps the accumulator buffer for an OACC
  // alloca this way). Resolve it here so the query observes what the IR will
  // hold once the conversion commits.
  if (Value remapped = remap(v))
    v = remapped;
  auto defOp = v.getDefiningOp();
  if (!defOp) {
    auto blockArg = dyn_cast<mlir::BlockArgument>(v);
    if (!blockArg) {
      return false;
    }
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp) {
      return false;
    }
    return isAllocaInputValueImpl(
        forOp.getInitArgs()[blockArg.getArgNumber() -
                            forOp.getNumInductionVars()],
        remap);
  }
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(v).getResultNumber();
    auto innerArg = forOp.getRegionIterArg(resultIdx);
    if (llvm::none_of(innerArg.getUsers(), [](Operation *user) {
          return isa<mlir::gcu::MatMulOp>(user);
        }))
      return false;
    return isAllocaInputValueImpl(forOp.getInitArgs()[resultIdx], remap);
  }
  if (isa_and_nonnull<memref::AllocaOp>(defOp)) {
    return true;
  }
  if (isa_and_nonnull<memref::ReinterpretCastOp>(defOp)) {
    return isAllocaInputValueImpl(defOp->getOperand(0), remap);
  }
  return false;
}

bool isAllocaInputValue(Value v, ValueRemapFn remap) {
  return isAllocaInputValueImpl(v, remap);
}

LogicalResult resolveAccumulatorOperand(Value originalValue, Value &operand,
                                        const AccPublishContext &ctx) {
  // A consumer may sit behind a passthrough layout conversion: an identity
  // convert_layout lowers to a plain replacement of its source, so the
  // published buffer must be found through it as well. This keeps the
  // resolution in one place instead of every consumer's lowering.
  Value key = originalValue;
  while (auto cvt = key.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
    auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
    if (!srcTy || !dstTy || srcTy.getEncoding() != dstTy.getEncoding())
      break;
    key = cvt.getSrc();
  }
  auto it = ctx.buffers.find(key);
  if (it != ctx.buffers.end()) {
    operand = it->second;
    return success();
  }
  if (ctx.expected.contains(key))
    return failure();
  return success();
}

bool isAllocaInputValue(Value v) {
  return isAllocaInputValue(v, [](Value value) { return value; });
}

ReduceGenerator::ReduceGenerator(triton::ReduceOp op,
                                 PrologueArgListType prologueArgList,
                                 PrologueOpIteratorRange prologueOps)
    : op(op), prologueOps(prologueOps), prologueArgList(prologueArgList) {
  auto inputType = op.getInputTypes()[0];
  auto numElems = triton::gcu::getElemsPerThread(inputType);
  int axis = op.getAxis();
  for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
    if (i == axis) {
      if (reduceInputDims[j] == 1) {
        reduceInputDims[j] = numElems[i];
      } else {
        reduceInputDims[--j] = numElems[i];
      }
      reduceAxis = j;
      reduceOutputDims[reduceAxis] = 1;
      --j;
    } else {
      reduceInputDims[j] *= numElems[i];
      reduceOutputDims[j] = reduceInputDims[j];
    }
  }
  assert(reduceAxis == 1 || reduceAxis == 2);
  auto slicedAxies = getSlicedAxies(inputType);
  if (slicedAxies.count(axis) != 0) {
    partialReduce = true;
  }
  collectVectorizeInfo();
}

void ReduceGenerator::collectVectorizeInfo() {
  unsigned maxBpe = 1;
  unsigned minBpe = 8;
  for (auto elementTy : op.getElementTypes()) {
    auto bpe = getBpe(elementTy);
    maxBpe = std::max(maxBpe, bpe);
    minBpe = std::min(minBpe, bpe);
  }
  auto numOacc = maxBpe / minBpe;
  if (numOacc > 4) {
    return;
  }
  vectorLength = oaccSizeInBytes / minBpe;
  for (int i = 2; i >= 0; --i) {
    if (reduceInputDims[i] >= vectorLength) {
      vectorizeAxis = i;
      break;
    }
  }
  if (vectorizeAxis == 3 && reduceAxis == 1 && prologueArgList.empty()) {
    vectorizeAxis = 2;
  }
  if (vectorizeAxis == 3 || prologueOps.empty()) {
    return;
  }
  if (llvm::any_of(prologueOps, [](auto &op) {
        return !op.template hasTrait<OpTrait::Elementwise>();
      })) {
    vectorizeAxis = 3;
    return;
  }
  for (auto arg : prologueArgList) {
    auto elementTy = getElementTypeOrSelf(arg.getType());
    auto bpe = getBpe(elementTy);
    maxBpe = std::max(maxBpe, bpe);
    minBpe = std::min(minBpe, bpe);
  }

  for (auto &elementwiseOp : prologueOps) {
    for (auto resultType : elementwiseOp.getResultTypes()) {
      auto elementTy = getElementTypeOrSelf(resultType);
      if (elementTy.isInteger(1) &&
          isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::CmpFOp,
              arith::CmpIOp, arith::SelectOp>(elementwiseOp)) {
        continue;
      }
      auto bpe = getBpe(elementTy);
      maxBpe = std::max(maxBpe, bpe);
      minBpe = std::min(minBpe, bpe);
    }
  }

  numOacc = maxBpe / minBpe;
  if (numOacc > 4) {
    vectorizeAxis = 3;
    return;
  }
  vectorLength = oaccSizeInBytes / minBpe;
  if (vectorLength > reduceInputDims[vectorizeAxis]) {
    if (reduceAxis == 2 && vectorizeAxis == 2 &&
        reduceInputDims[2] * reduceInputDims[1] >= vectorLength) {
      reduceInputDims[1] =
          reduceInputDims[1] * reduceInputDims[2] / vectorLength;
      reduceOutputDims[1] = reduceInputDims[1];
      reduceOutputDims[2] = vectorLength / reduceInputDims[2];
      reduceInputDims[2] = vectorLength;
    } else {
      vectorizeAxis = 3;
    }
  }
  return;
}

void ReduceGenerator::processPrologue(OpBuilder &builder, Location loc,
                                      IRMapping &mapper) {
  for (auto &op : prologueOps) {
    if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op)) {
      auto symbol = externElementwiseOp.getSymbol();
      auto operands =
          llvm::to_vector(llvm::map_range(op.getOperands(), [&](auto operand) {
            return mapper.lookup(operand);
          }));
      auto resultTy = VectorType::get(
          ArrayRef<int64_t>{vectorLength},
          getElementTypeOrSelf(externElementwiseOp.getResult()));
      Value result;
      if (symbol.starts_with("__ef_")) {
        std::string efSymbol = "__ef_v";
        efSymbol += symbol.drop_front(strlen("__ef_"));
        result = builder.create<mlir::gcu::ExternElementwiseOp>(
            loc, resultTy, operands, efSymbol);
      } else if (isMixedPrecisionSymbol(symbol)) {
        result = builder.create<mlir::gcu::ExternElementwiseOp>(
            loc, resultTy, operands, symbol);
      } else {
        llvm_unreachable(
            ("unsupported extern elementwise: " + symbol).str().c_str());
      }
      mapper.map(externElementwiseOp.getResult(), result);
      continue;
    }
    auto cloneOp = builder.clone(op, mapper);
    SmallVector<Type> resultTypes;
    auto typeInterface = dyn_cast<InferTypeOpInterface>(cloneOp);
    if (!typeInterface ||
        failed(typeInterface.inferReturnTypes(
            cloneOp->getContext(), cloneOp->getLoc(), cloneOp->getOperands(),
            cloneOp->getAttrDictionary(), cloneOp->getPropertiesStorage(),
            cloneOp->getRegions(), resultTypes))) {
      resultTypes.clear();
      llvm::transform(op.getResultTypes(), std::back_inserter(resultTypes),
                      [&](auto resultType) {
                        return VectorType::get(
                            ArrayRef<int64_t>{vectorLength},
                            getElementTypeOrSelf(resultType));
                      });
    }
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      cloneOp->getResult(i).setType(resultTypes[i]);
      mapper.map(op.getResult(i), cloneOp->getResult(i));
    }
  }
}

bool ReduceGenerator::hasVectorizeImpl() const {
  if (vectorizeAxis == 3) {
    return false;
  }
  // Vectorized lowering is only implemented for reduceAxis==2 and
  // vectorizeAxis==2 (see applyVectorizeImpl<2, 2> below). Other axis
  // combinations are not supported yet.
  auto isReduce1D = reduceInputDims[0] == 1 && reduceInputDims[1] == 1;
  if (vectorizeAxis == 2 && reduceAxis == 2 && !isReduce1D && !partialReduce) {
    return true;
  }
  return false;
}

template <unsigned reduceAxis, unsigned vectorizeAxis>
void ReduceGenerator::applyVectorizeImpl(OpBuilder &builder, Location loc,
                                         ArrayRef<Value> outputs,
                                         ArrayRef<Value> inputs) {
  llvm_unreachable("Not implemented");
}

template <>
void ReduceGenerator::applyVectorizeImpl<2, 2>(OpBuilder &builder, Location loc,
                                               ArrayRef<Value> outputs,
                                               ArrayRef<Value> inputs) {
  auto isReduce1D = reduceInputDims[0] == 1 && reduceInputDims[1] == 1;
  if (isReduce1D) {
    llvm_unreachable("Not implemented");
  }
  unsigned numInputs = inputs.size();
  unsigned numOutputs = outputs.size();
  SmallVector<VectorType> loadTypes;
  SmallVector<Value> tarAddrs;
  SmallVector<SmallVector<Value>> inputTarStrides(numInputs);
  auto loopLimit = reduceInputDims[1];
  auto loopCnt = loopLimit > loopUnrollTime ? loopUnrollTime : loopLimit;
  triton::gcu::TritonGCUBuilder b(loc, builder);
  for (unsigned i = 0; i < numInputs; ++i) {
    auto elementTy = cast<MemRefType>(inputs[i].getType()).getElementType();
    loadTypes.emplace_back(
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy));
    auto bpe = getBpe(elementTy);
    tarAddrs.emplace_back(b.tarAddr(inputs[i]));
    // input stride for looping over dim2
    inputTarStrides[i].emplace_back(
        b.tarStride(loadTypes.back(), reduceInputDims[2] * bpe));
    // back stride for looping over dim2
    inputTarStrides[i].emplace_back(
        b.tarStride(loadTypes.back(),
                    (vectorLength - (loopCnt - 1) * reduceInputDims[2]) * bpe));
    // input stride for looping over dim1
    inputTarStrides[i].emplace_back(
        b.tarValue((loopCnt - 1) * reduceInputDims[2] * bpe));
  }
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto fetchInputValues = [&](SmallVector<Value> &tarAddrs) {
    SmallVector<Value> values;
    if (prologueArgList.empty()) {
      for (unsigned i = 0; i < loopCnt - 1; ++i) {
        for (unsigned j = 0; j < numInputs; ++j) {
          values.emplace_back(
              b.tarLoad(loadTypes[j], tarAddrs[j], inputTarStrides[j][0]));
        }
      }
      for (unsigned i = 0; i < numInputs; ++i) {
        values.emplace_back(
            b.tarLoad(loadTypes[i], tarAddrs[i], inputTarStrides[i][1]));
      }
    } else {
      SmallVector<IRMapping> operandMaps(loopCnt);
      for (unsigned i = 0; i < loopCnt - 1; ++i) {
        for (unsigned j = 0; j < numInputs; ++j) {
          operandMaps[i].map(
              prologueArgList[j],
              b.tarLoad(loadTypes[j], tarAddrs[j], inputTarStrides[j][0]));
        }
      }
      for (unsigned i = 0; i < numInputs; ++i) {
        operandMaps[loopCnt - 1].map(
            prologueArgList[i],
            b.tarLoad(loadTypes[i], tarAddrs[i], inputTarStrides[i][1]));
      }
      if (reduceOutputDims[2] > 1) {
        auto reduceNumInputs = op.getNumOperands();
        values.resize(loopCnt * reduceNumInputs * reduceOutputDims[2]);
        for (unsigned i = 0; i < loopCnt; ++i) {
          processPrologue(builder, loc, operandMaps[i]);
        }
        for (unsigned i = 0; i < reduceNumInputs; ++i) {
          auto splitVectorType = VectorType::get(
              ArrayRef<int64_t>{vectorLength / reduceOutputDims[2]},
              getElementTypeOrSelf(op.getOperand(i).getType()));
          for (unsigned j = 0; j < loopCnt; ++j) {
            auto splitValues =
                builder
                    .create<mlir::gcu::VectorConvertOp>(
                        loc, TypeRange{splitVectorType, splitVectorType},
                        operandMaps[j].lookup(op.getOperand(i)))
                    .getResults();
            for (unsigned k = 0; k < splitValues.size(); ++k) {
              values[j * reduceNumInputs * reduceOutputDims[2] +
                     reduceNumInputs * k + i] = splitValues[k];
            }
          }
        }
      } else {
        for (unsigned i = 0; i < loopCnt; ++i) {
          processPrologue(builder, loc, operandMaps[i]);
          for (auto operand : op.getOperands()) {
            auto mappingValue = operandMaps[i].lookupOrNull(operand);
            assert(mappingValue);
            values.emplace_back(mappingValue);
          }
        }
      }
    }
    return values;
  };
  triton::gcu::CombineOpDesc combineOpDesc(op);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, reduceInputDims[0]), one,
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter2, ValueRange iterArgs2) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, reduceInputDims[1]),
            builder.create<arith::ConstantIndexOp>(loc, loopCnt), tarAddrs,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              SmallVector<Value> initArgs(iterArgs1);
              initArgs.append(fetchInputValues(initArgs));
              auto loop0 = builder.create<scf::ForOp>(
                  loc,
                  builder.create<arith::ConstantIndexOp>(loc, vectorLength),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         reduceInputDims[2]),
                  builder.create<arith::ConstantIndexOp>(loc, vectorLength),
                  initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter0,
                      ValueRange iterArgs0) {
                    SmallVector<Value> inputTarAddrs(
                        iterArgs0.take_front(numInputs));
                    auto curValues = fetchInputValues(inputTarAddrs);
                    SmallVector<Value> terminatorOperands(inputTarAddrs);
                    SmallVector<Value, 4> args(numOutputs * 2);
                    for (unsigned i = 0; i < loopCnt * reduceOutputDims[2];
                         ++i) {
                      for (unsigned j = 0; j < numOutputs; ++j) {
                        args[j] = iterArgs0[numInputs + i * numOutputs + j];
                        args[numOutputs + j] = curValues[i * numOutputs + j];
                      }
                      terminatorOperands.append(
                          combineOpDesc.applyVectorizedCombine(
                              builder, loc, args,
                              vectorLength / reduceOutputDims[2]));
                    }
                    builder.create<scf::YieldOp>(loc, terminatorOperands);
                  });
              for (unsigned i = 0; i < numInputs; ++i) {
                initArgs[i] = loop0.getResult(i);
                b.tarJump(initArgs[i], inputTarStrides[i][2]);
              }
              for (unsigned i = 0; i < loopCnt; ++i) {
                for (unsigned k = 0; k < reduceOutputDims[2]; ++k) {
                  auto results = triton::gcu::reduceVectorLanes(
                      builder, loc, combineOpDesc,
                      ValueRange(loop0.getResults().slice(
                          numInputs +
                              (i * reduceOutputDims[2] + k) * numOutputs,
                          numOutputs)));
                  for (unsigned j = 0; j < numOutputs; ++j) {
                    builder.create<memref::StoreOp>(
                        loc, results[j], outputs[j],
                        ValueRange{
                            iter2,
                            builder.create<arith::AddIOp>(
                                loc,
                                builder.create<arith::ConstantIndexOp>(loc, i),
                                iter1),
                            builder.create<arith::ConstantIndexOp>(loc, k)});
                  }
                }
              }
              builder.create<scf::YieldOp>(
                  loc, ValueRange(initArgs.begin(), numInputs));
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduce(OpBuilder &builder, Location loc,
                                  ArrayRef<Value> outputs,
                                  ArrayRef<Value> inputs) {
  assert(prologueArgList.empty() || prologueArgList.size() == inputs.size());
  if (reduceAxis == 2 && vectorizeAxis == 2) {
    applyVectorizeImpl<2, 2>(builder, loc, outputs, inputs);
  } else {
    llvm_unreachable("Not implemented");
  }
}

SmallVector<Value> ReduceGenerator::normalizeInputs(OpBuilder &builder,
                                                    Location loc,
                                                    ValueRange inputs) {
  return llvm::to_vector(llvm::map_range(inputs, [&](auto input) {
    auto elementTy = cast<MemRefType>(input.getType()).getElementType();
    if (elementTy.isInteger(1)) {
      auto inputPtr = builder.create<mlir::gcu::MemRefToPtrOp>(
          loc, mlir::gcu::PtrType::get(builder.getContext(), elementTy), input);
      elementTy = builder.getI8Type();
      input = builder.create<mlir::gcu::PtrToMemRefOp>(
          loc, mlir::gcu::PtrType::get(builder.getContext(), elementTy),
          inputPtr);
    }
    elementTy = elementTy.isInteger(1) ? builder.getI8Type() : elementTy;
    input = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get(reduceInputDims, elementTy), input, 0,
        ArrayRef<int64_t>{reduceInputDims},
        ArrayRef<int64_t>{reduceInputDims[1] * reduceInputDims[2],
                          reduceInputDims[2], 1});
    return input;
  }));
}

SmallVector<Value> ReduceGenerator::normalizeOutputs(OpBuilder &builder,
                                                     Location loc,
                                                     ValueRange outputs) {
  return llvm::to_vector(llvm::map_range(outputs, [&](auto output) {
    auto elementTy = cast<MemRefType>(output.getType()).getElementType();
    if (elementTy.isInteger(1)) {
      auto outputPtr = builder.create<mlir::gcu::MemRefToPtrOp>(
          loc, mlir::gcu::PtrType::get(builder.getContext(), elementTy),
          output);
      elementTy = builder.getI8Type();
      output = builder.create<mlir::gcu::PtrToMemRefOp>(
          loc, mlir::gcu::PtrType::get(builder.getContext(), elementTy),
          outputPtr);
    }
    output = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get(reduceOutputDims, elementTy), output, 0,
        ArrayRef<int64_t>{reduceOutputDims},
        ArrayRef<int64_t>{reduceOutputDims[1] * reduceOutputDims[2],
                          reduceOutputDims[2], 1});
    return output;
  }));
}

} // namespace gcu
} // namespace triton
} // namespace mlir
