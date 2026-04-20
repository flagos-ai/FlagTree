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
#include <utility>
#include <vector>

#include "Analysis/FirstLastUserAnalysis.h"
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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-ir-to-gcu-ir-util"
#define DBG_TRITON_IR 1

namespace mlir {

template <typename WarpSpecializeOpTy>
void captureValuesToWarpSpecializeOp(WarpSpecializeOpTy wsOp, Value capture) {
  wsOp->insertOperands(wsOp.getNumOperands(), capture);
  for (Region *region : wsOp.getPartitionRegions()) {
    BlockArgument arg =
        region->addArgument(capture.getType(), capture.getLoc());
    replaceAllUsesInRegionWith(capture, arg, *region);
  }
}

template <typename WarpSpecializeOpTy>
void captureValuesToWarpSpecializeOp(Operation *entryFunc, Value capture) {
  entryFunc->walk([&](Operation *op) {
    if (auto wsOp = dyn_cast<WarpSpecializeOpTy>(op))
      captureValuesToWarpSpecializeOp(wsOp, capture);
  });
}

Value getPrivateDTETag(OpBuilder &builder, Operation *op) {
  OpBuilder::InsertionGuard guard(builder);
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto firstOp = &func.getFunctionBody().getBlocks().front().front();
  auto tagType = MemRefType::get(ArrayRef<int64_t>{1}, builder.getI32Type());
  if (isa<memref::AllocOp>(firstOp) && firstOp->getAttr("gcu.private_tag"))
    return firstOp->getResult(0);
  builder.setInsertionPoint(firstOp);
  auto tag = builder.create<memref::AllocOp>(op->getLoc(), tagType);
  tag->setAttr("gcu.private_tag", builder.getUnitAttr());
  return tag;
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

Value createPrivateDTETag(OpBuilder &builder, Operation *op) {
  auto tagType = MemRefType::get(ArrayRef<int64_t>{1}, builder.getI32Type());
  auto tag = builder.create<memref::AllocOp>(op->getLoc(), tagType);
  tag->setAttr("gcu.private_tag", builder.getUnitAttr());
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
        } else if ((!isM) && i == rank - 1) {
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
      if (!isa<triton::gpu::BlockedEncodingAttr>(parent)) {
        llvm::report_fatal_error("[Error] bad slice layout parent");
        assert(false && "bad slice layout parent");
        return warpIds;
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

func::FuncOp getOrDefineFunction(gpu::GPUModuleOp moduleOp, Location loc,
                                 OpBuilder &rewriter, StringRef name,
                                 FunctionType type) {
  func::FuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<func::FuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<func::FuncOp>(loc, name, type);
    ret.setPrivate();
  }
  return ret;
}

void doSlicePadOrMemsetSlice(OpBuilder &rewriter, Location loc,
                             triton::gcu::PrivateTagPool &pTagPool,
                             Operation *op, Value output, Value src,
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
        doMemset(childBuilder, pTagPool, op, output, defaultValue,
                 totalNumElems);
        childBuilder.create<memref_ext::SliceStartOp>(
            loc, output, src, offsets, defaultValue, tag.getTag(),
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

void doMemset(OpBuilder &rewriter, triton::gcu::PrivateTagPool &pTagPool,
              Operation *op, Value output, Value v, int totalNumElems) {
  auto loc = op->getLoc();
  if (totalNumElems > 128 || totalNumElems <= 0) {
    auto tag = pTagPool.getSyncTagInfo(op);
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

Value castToMemref1D(OpBuilder &rewriter, Location loc, Value v,
                     Value totalNumElems) {
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  return rewriter.create<memref::ReinterpretCastOp>(
      loc,
      MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                      dyn_cast<MemRefType>(v.getType()).getElementType()),
      v, zero, ArrayRef<Value>{totalNumElems}, ArrayRef<Value>{one});
}

bool isMustAliasOp(mlir::Operation *op) {
  if (llvm::isa<triton::PtrToIntOp, triton::IntToPtrOp, triton::BitcastOp,
                gcu::PtrToMemRefOp, gcu::MemRefToPtrOp, gcu::IntToPtrOp,
                gcu::PtrToIntOp>(op)) {
    return true;
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
      if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout)) {
        return true;
      } else if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (cast<triton::gpu::SliceEncodingAttr>(srcLayout).getDim() ==
            cast<triton::gpu::SliceEncodingAttr>(dstLayout).getDim()) {
          return true;
        }
      } else if (isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
        return true;
      } else if (isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
        return true;
      }
    }
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
  } else if (isa<triton::ReshapeOp>(op)) {
    auto reshapeOp = cast<triton::ReshapeOp>(op);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(reshapeOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(reshapeOp.getType());
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else if (isa<triton::BroadcastOp>(op)) {
    auto broastOp = cast<triton::BroadcastOp>(op);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(broastOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(broastOp.getType());
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else if (isa<triton::gpu::LocalLoadOp>(op)) {
    ModuleOp builtinModule = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gcu::getNumWarps(builtinModule);
    if (numWarps == 1) {
      return true;
    }
    return false;
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

Value asyncAllocOp(OpBuilder &builder, Operation *ttParent, MemRefType type,
                   int64_t memoryAlignment) {
  auto allocOp = builder.create<memref::AllocOp>(ttParent->getLoc(), type);
  if (memoryAlignment != INVALID_ALIGNMENT) {
    allocOp->setAttr(kAlignment, builder.getI64IntegerAttr(memoryAlignment));
  }
  return allocOp.getResult();
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

Value loadFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<Operation *, int> lastTTUser,
                        std::pair<Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin) {
  auto loc = buffer.getLoc();
  auto srcType = dyn_cast<MemRefType>(buffer.getType());
  auto numElems = triton::gcu::getElemsPerThread(type);
  auto totalNumElems = builder.create<arith::ConstantIndexOp>(
      loc, triton::gcu::getTotalElemsPerThread(type));
  auto outputType =
      MemRefType::get(SmallVector<int64_t>(numElems.begin(), numElems.end()),
                      srcType.getElementType());

  auto warpIds = getWarpIds(builder, loc, type);
  SmallVector<Value, 4> offsets;
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i < srcType.getRank(); ++i) {
    offsets.push_back(builder.create<arith::MulIOp>(
        loc, builder.create<arith::ConstantIntOp>(loc, numElems[i], 32),
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                           warpIds[i])));
  }

  auto output = syncAllocOp(builder, loc, lastTTUser, userAnalysis,
                            replaced2Origin, outputType);

  auto isThread0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
  auto defaultValue =
      triton::gcu::createConstantZero(builder, loc, srcType.getElementType());
  bool isNeedMerge = srcType.getRank() > 5;
  SmallVector<Value, 4> mergedOffsets;
  Value src;
  Value dst;
  if (!firstTTUser.first || !tag.isAsync()) {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
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
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
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
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
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
  auto isThread0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
  if (!firstTTUser.first || !tag.isAsync()) {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
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
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaStartOp>(
                loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
                SmallVector<Value, 4>(shape.size(), zero), totalNumElems,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          });
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser.first);
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
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
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i < srcType.getRank(); ++i) {
    offsets.push_back(builder.create<arith::MulIOp>(
        loc,
        builder.create<arith::ConstantIntOp>(loc, srcType.getDimSize(i), 32),
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                           warpIds[i])));
    outputSize.push_back(outputType.getShape()[i]);
  }
  auto isThread0 = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
  bool isNeedMerge = srcType.getRank() > 5;
  SmallVector<Value, 4> mergedOffsets;
  Value src;
  Value dst;
  if (onlyThread0) {
    builder.create<scf::IfOp>(
        loc, isThread0, [&](OpBuilder &builder, Location loc) {
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
            if (definingOp->isBeforeInBlock(lastUser.first)) {
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

Value ConfigGcuLoad(OpBuilder &rewriter, Location loc,
                    triton::gcu::PrivateTagPool &pTagPool, Value srcOut,
                    mlir::Operation *op, MemRefType resultType, Value loadPtr,
                    mlir::ValueRange configStrides,
                    mlir::ValueRange configShapes, Value defaultValue,
                    triton::gcu::TagInfo tag, bool IsShareOutput) {
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

  auto elemType = resultType.getElementType();
  int64_t rank = resultType.getRank();

  auto buffer = rewriter.create<gcu::PtrToMemRefOp>(
      loc, MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType),
      loadPtr);

  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  bool bDynamicStride = false;
  bool bStaticTranspose = false;
  bool bReshape = true;
  SmallVector<unsigned> updateStrideDims;
  SmallVector<unsigned> nInitStrideDims;
  auto hint = getOrderHint(op);
  int64_t hint_size = static_cast<int64_t>(hint.size());
  assert(hint_size == rank /* || hint_size == 0*/);
  SmallVector<int32_t> order_hint;
  for (unsigned i = 0; i < rank; ++i)
    if (hint_size == 0)
      order_hint.push_back(-1);
    else
      order_hint.push_back(hint[i]);

  for (unsigned i = 0; i < rank; ++i) {
    if (order_hint[i] == -1) {
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

  for (int i = 0; i < rank; ++i) {
    if (bDynamicStride && order_hint[i] == 0)
      updateStrideDims.push_back(i);
    else
      nInitStrideDims.push_back(i);
  }

  SmallVector<Value, 4> vSrcOffsets;
  if (IsShareOutput) {
    for (unsigned i = 0; i < nInitStrideDims.size(); ++i)
      vSrcOffsets.push_back(zero);
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
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    auto isPad = rewriter.create<arith::AndIOp>(loc, isNeedPad, isThread0);
    if (bDynamicStride) {
      auto isTrans =
          rewriter.create<arith::AndIOp>(loc, isDynamicTrans, isThread0);
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
                  doSlicePadOrMemsetSlice(childBuilder, loc, pTagPool, op,
                                          reshapeOut, src, vOrderOffsets,
                                          vIntSlicehape, padSizes, defaultValue,
                                          tag);
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<scf::IfOp>(
                      loc, isThread0,
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
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, reshapeOut, src, vOrderOffsets, vTransOrder, defaultValue,
                tag.getTag(), ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      rewriter.create<scf::IfOp>(
          loc, isPad,
          [&](OpBuilder &builder, Location loc) {
            doSlicePadOrMemsetSlice(builder, loc, pTagPool, op, reshapeOut, src,
                                    vOrderOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tag);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isThread0, [&](OpBuilder &childBuilder, Location loc) {
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
                  doSlicePadOrMemsetSlice(childBuilder, loc, pTagPool, op,
                                          reshapeOut, src, vOrderOffsets,
                                          vIntSlicehape, padSizes, defaultValue,
                                          tag);
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
    } else {
      rewriter.create<scf::IfOp>(
          loc, isPad,
          [&](OpBuilder &builder, Location loc) {
            doSlicePadOrMemsetSlice(builder, loc, pTagPool, op, reshapeOut, src,
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
                     mlir::ValueRange configShapes, triton::gcu::TagInfo tag) {
  auto storeOp = dyn_cast<triton::gcu::StoreOp>(op);
  assert(storeOp);

  auto storeType = storeOp.getValue().getType();
  auto elemType = storeOp.getPtr().getType().getElementType();
  auto buffer = rewriter.create<gcu::PtrToMemRefOp>(
      loc, MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType),
      storePtr);

  int64_t rank = storeValueType.getRank();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto zero32 =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), zero);

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
      order_hint.push_back(-1);
    else
      order_hint.push_back(hint[i]);

  for (unsigned i = 0; i < rank; ++i) {
    if (order_hint[i] == -1) {
      bDynamicStride = true;
      auto trueCondition = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, configStrides[i], zero);
      rewriter.create<triton::gcu::AssertOp>(
          loc, trueCondition, "Not Support dynamic stride is 0", "", "", 0);
    }
  }

  for (int i = 0; i < rank; ++i) {
    if ((order_hint[i] == 0 && !bDynamicStride) ||
        (order_hint[i] == 1 && bDynamicStride)) {
      bReshape = false;
      break;
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

  SmallVector<Value, 4> vSlicehape;
  SmallVector<Value, 4> vIntSlicehape;
  Value totalSize = one;
  for (unsigned i = 0; i < rank; ++i) {
    auto shape = rewriter.create<arith::MinSIOp>(
        loc, vNumElems[i],
        rewriter.create<arith::MaxSIOp>(
            loc, zero,
            rewriter.create<arith::SubIOp>(loc, vSrcShapes[i],
                                           vSrcOffsets[i])));
    vSlicehape.push_back(shape);
    vIntSlicehape.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), shape));
    totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, shape);
  }

  SmallVector<Value, 4> sliceOffsets(rank, zero32);
  Value diff = zero;
  for (unsigned i = 0; i < rank; ++i) {
    auto dim_diff =
        rewriter.create<arith::SubIOp>(loc, vStoreShapes[i], vSlicehape[i]);
    diff = rewriter.create<arith::AddIOp>(loc, diff, dim_diff);
  }

  auto resultType = MemRefType::get(
      SmallVector<int64_t>(rank, ShapedType::kDynamic), elemType);
  auto dst = rewriter.create<memref::ReinterpretCastOp>(
      loc, resultType, buffer, zero, vOrderShapes, vOrderStrides);

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
              loc, dst, reshapeStoreValue, vTransOrder, vOrderOffsets,
              tag.getTag(), ValueRange{tag.getIdx()});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isSlice,
              [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::SliceDesliceStartOp>(
                    loc, dst, reshapeStoreValue, sliceOffsets, vIntSlicehape,
                    vOrderOffsets, tag.getTag(), ValueRange{tag.getIdx()});
                childBuilder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<scf::IfOp>(
                    loc, isNotZero,
                    [&](OpBuilder &child2Builder, Location loc) {
                      child2Builder.create<memref_ext::DesliceStartOp>(
                          loc, dst, reshapeStoreValue, vOrderOffsets,
                          tag.getTag(), ValueRange{tag.getIdx()});
                      child2Builder.create<scf::YieldOp>(loc);
                    });
                childBuilder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  } else if (bStaticTranspose) {
    rewriter.create<scf::IfOp>(
        loc, isNotZero, [&](OpBuilder &build, Location loc) {
          build.create<memref_ext::TransposeDesliceStartOp>(
              loc, dst, reshapeStoreValue, vTransOrder, vOrderOffsets,
              tag.getTag(), ValueRange{tag.getIdx()});
          build.create<scf::YieldOp>(loc);
        });
  } else {
    auto isSlice = rewriter.create<arith::AndIOp>(loc, isNeedSlice, isNotZero);
    rewriter.create<scf::IfOp>(
        loc, isSlice,
        [&](OpBuilder &builder, Location loc) {
          builder.create<memref_ext::SliceDesliceStartOp>(
              loc, dst, reshapeStoreValue, sliceOffsets, vIntSlicehape,
              vOrderOffsets, tag.getTag(), ValueRange{tag.getIdx()});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isNotZero, [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::DesliceStartOp>(
                    loc, dst, reshapeStoreValue, vOrderOffsets, tag.getTag(),
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
  if (deallocOp && deallocOp->getBlock() == pos->getBlock()) {
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
    // Ops inside nested regions (e.g. scf.for body) have parentRegion != part.
    if (!part->isAncestor(opRegion))
      continue;
    if (part->empty())
      return {};
    for (unsigned i = 0; i < wsOp.getNumOperands(); ++i) {
      if (wsOp.getOperand(i) == tagMemref)
        return part->getArgument(i);
    }
    captureValuesToWarpSpecializeOp(wsOp, tagMemref);
    return part->getArgument(wsOp.getNumOperands());
  }
  return {};
}

} // namespace

Value PrivateTagPool::getTagsValue(Operation *op) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  assert(func && "can't find func op");

  Value tagMemref = tagsAllocOp->getResult(0);
  if (auto wsOp = op->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
    if (!wsOp->isProperAncestor(tagsAllocOp))
      if (Value v = lookupPartitionTagArg(wsOp, op, tagMemref))
        return v;
  }

  if (auto wsOp = op->getParentOfType<mlir::gcu::WarpSpecializeOp>()) {
    if (!wsOp->isProperAncestor(tagsAllocOp))
      if (Value v = lookupPartitionTagArg(wsOp, op, tagMemref))
        return v;
  }

  if (funcName2TagArgPos.count(func.getName()) != 0)
    return func.getArgument(funcName2TagArgPos[func.getName()]);
  return tagMemref;
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

Value createConstantNaN(OpBuilder &builder, Location loc, Type elemType) {
  const llvm::fltSemantics *sem = nullptr;
  if (elemType.isF32()) {
    sem = &llvm::APFloatBase::IEEEsingle();
  } else if (elemType.isBF16()) {
    sem = &llvm::APFloatBase::BFloat();
  } else if (elemType.isF16()) {
    sem = &llvm::APFloatBase::IEEEhalf();
  } else if (elemType.isF64()) {
    sem = &llvm::APFloatBase::IEEEdouble();
  } else if (llvm::isa<Float8E4M3B11FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3B11FNUZ();
  } else if (llvm::isa<Float8E4M3FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FNUZ();
  } else if (llvm::isa<Float8E5M2FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2FNUZ();
  } else if (llvm::isa<Float8E4M3FNType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FN();
  } else if (llvm::isa<Float8E5M2Type>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2();
  } else {
    std::string o;
    llvm::raw_string_ostream os(o);
    elemType.print(os);
    llvm_unreachable((o + " is unsupported").c_str());
  }
  return builder.create<arith::ConstantFloatOp>(
      loc, dyn_cast<FloatType>(elemType), APFloat::getNaN(*sem));
}

Value createConstantInf(OpBuilder &builder, Location loc, Type elemType,
                        bool isNegative) {
  const llvm::fltSemantics *sem = nullptr;
  if (elemType.isF32()) {
    sem = &llvm::APFloatBase::IEEEsingle();
  } else if (elemType.isBF16()) {
    sem = &llvm::APFloatBase::BFloat();
  } else if (elemType.isF16()) {
    sem = &llvm::APFloatBase::IEEEhalf();
  } else if (elemType.isF64()) {
    sem = &llvm::APFloatBase::IEEEdouble();
  } else if (llvm::isa<Float8E4M3B11FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3B11FNUZ();
  } else if (llvm::isa<Float8E4M3FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FNUZ();
  } else if (llvm::isa<Float8E5M2FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2FNUZ();
  } else if (llvm::isa<Float8E4M3FNType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FN();
  } else if (llvm::isa<Float8E5M2Type>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2();
  } else {
    std::string o;
    llvm::raw_string_ostream os(o);
    elemType.print(os);
    llvm_unreachable((o + " is unsupported").c_str());
  }
  return builder.create<arith::ConstantFloatOp>(
      loc, dyn_cast<FloatType>(elemType), APFloat::getInf(*sem, isNegative));
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
    if (!isa<triton::gpu::BlockedEncodingAttr>(parent)) {
      llvm::report_fatal_error("[Error] bad slice layout parent");
      assert(false && "bad slice layout parent");
    }
    auto blockEncParent = dyn_cast<triton::gpu::BlockedEncodingAttr>(parent);
    auto parentWarpsPerCTA = blockEncParent.getWarpsPerCTA();
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
          int64_t k = shape[rank - 2];
          elemsPerthread[rank - 2] = k;
          elemsPerthread[rank - 1] = shape[rank - 1] / warpsPerCTA[rank - 1];
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
  if (!mod->hasAttr("ttg.num-warps"))
    llvm::report_fatal_error(
        "TritonGPU module should contain a ttg.num-warps attribute");
  return cast<IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
}

unsigned getBpe(Type type) {
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

bool isNvLibDeviceSymbol(StringRef symbol) {
  static const llvm::StringSet<> symbolSet = {
      ///  unary op
      "__nv_abs",
      "__nv_llabs",
      "__nv_fabsf",
      "__nv_brev",
      "__nv_brevll",
      "__nv_clz",
      "__nv_clzll",
      "__nv_ffs",
      "__nv_ffsll",
      "__nv_popc",
      "__nv_popcll",
      "__nv_acosf",
      "__nv_acoshf",
      "__nv_asinf",
      "__nv_asinhf",
      "__nv_atanf",
      "__nv_atanhf",
      "__nv_cbrtf",
      "__nv_ceilf",
      "__nv_cosf",
      "__nv_coshf",
      "__nv_cospif",
      "__nv_erfcf",
      "__nv_erfcinvf",
      "__nv_erfcxf",
      "__nv_erff",
      "__nv_erfinvf",
      "__nv_exp10f",
      "__nv_exp2f",
      "__nv_expf",
      "__nv_expm1f",
      "__nv_floorf",
      "__nv_frcp_rd",
      "__nv_frcp_rn",
      "__nv_frcp_ru",
      "__nv_frcp_rz",
      "__nv_frsqrt_rn",
      "__nv_fsqrt_rd",
      "__nv_fsqrt_rn",
      "__nv_fsqrt_ru",
      "__nv_fsqrt_rz",
      "__nv_j0f",
      "__nv_j1f",
      "__nv_lgammaf",
      "__nv_log10f",
      "__nv_log1pf",
      "__nv_log2f",
      "__nv_logbf",
      "__nv_logf",
      "__nv_nearbyintf",
      "__nv_normcdff",
      "__nv_normcdfinvf",
      "__nv_rcbrtf",
      "__nv_rintf",
      "__nv_roundf",
      "__nv_rsqrtf",
      "__nv_saturatef",
      "__nv_sinf",
      "__nv_sinhf",
      "__nv_sinpif",
      "__nv_sqrtf",
      "__nv_tanf",
      "__nv_tanhf",
      "__nv_tgammaf",
      "__nv_truncf",
      "__nv_y0f",
      "__nv_y1f",
      ///  unary int32 float
      "__nv_finitef",
      "__nv_ilogbf",
      "__nv_isinff",
      "__nv_isnanf",
      "__nv_signbitf",
      ///  unary int64 float
      "__nv_llrintf",
      "__nv_llroundf",
      ///  unary type convert
      "__nv_float_as_int",
      "__nv_int_as_float",
      "__nv_float2int_rd",
      "__nv_float2int_rn",
      "__nv_float2int_ru",
      "__nv_float2int_rz",
      "__nv_float2uint_rd",
      "__nv_float2uint_rn",
      "__nv_float2uint_ru",
      "__nv_float2uint_rz",
      "__nv_int2float_rd",
      "__nv_int2float_rn",
      "__nv_int2float_ru",
      "__nv_int2float_rz",
      "__nv_uint2float_rd",
      "__nv_uint2float_rn",
      "__nv_uint2float_ru",
      "__nv_uint2float_rz",
      "__nv_float2ll_rd",
      "__nv_float2ll_rn",
      "__nv_float2ll_ru",
      "__nv_float2ll_rz",
      "__nv_float2ull_rd",
      "__nv_float2ull_rn",
      "__nv_float2ull_ru",
      "__nv_float2ull_rz",
      "__nv_ll2float_rd",
      "__nv_ll2float_rn",
      "__nv_ll2float_ru",
      "__nv_ll2float_rz",
      "__nv_ull2float_rd",
      "__nv_ull2float_rn",
      "__nv_ull2float_ru",
      "__nv_ull2float_rz",
      ///  binary op
      "__nv_hadd",
      "__nv_uhadd",
      "__nv_llmax",
      "__nv_ullmax",
      "__nv_max",
      "__nv_umax",
      "__nv_fmaxf",
      "__nv_llmin",
      "__nv_ullmin",
      "__nv_min",
      "__nv_umin",
      "__nv_fminf",
      "__nv_mul24",
      "__nv_umul24",
      "__nv_mulhi",
      "__nv_umulhi",
      "__nv_mul64hi",
      "__nv_umul64hi",
      "__nv_rhadd",
      "__nv_urhadd",
      "__nv_atan2f",
      "__nv_copysignf",
      "__nv_fadd_rd",
      "__nv_fadd_rn",
      "__nv_fadd_ru",
      "__nv_fadd_rz",
      "__nv_fdimf",
      "__nv_fdiv_rd",
      "__nv_fdiv_rn",
      "__nv_fdiv_ru",
      "__nv_fdiv_rz",
      "__nv_fmodf",
      "__nv_fmul_rd",
      "__nv_fmul_rn",
      "__nv_fmul_ru",
      "__nv_fmul_rz",
      "__nv_fsub_rd",
      "__nv_fsub_rn",
      "__nv_fsub_ru",
      "__nv_fsub_rz",
      "__nv_hypotf",
      "__nv_nextafterf",
      "__nv_powif",
      "__nv_powf",
      "__nv_remainderf",
      ///  binary float float int32
      "__nv_ldexpf",
      "__nv_scalbnf",
      ///  ternary op
      "__nv_fmaf",
      "__nv_fmaf_rd",
      "__nv_fmaf_rn",
      "__nv_fmaf_ru",
      "__nv_fmaf_rz",
  };
  return symbolSet.contains(symbol);
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
      if (isNvLibDeviceSymbol(symbol)) {
        std::string efSymbol = "__ef_v";
        efSymbol += symbol.drop_front(strlen("__nv_"));
        result = builder.create<mlir::gcu::ExternElementwiseOp>(
            loc, resultTy, operands, efSymbol);
      } else if (symbol == "__nv_ffs") {
        result = builder.create<math::CtPopOp>(loc, operands.front().getType(),
                                               operands);
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
        values.resize(loopCnt * numInputs * reduceOutputDims[2]);
        for (unsigned i = 0; i < loopCnt; ++i) {
          processPrologue(builder, loc, operandMaps[i]);
        }
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
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
              values[j * numInputs * reduceOutputDims[2] + k * numInputs + i] =
                  splitValues[k];
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
  auto &combineOp = op.getCombineOp();
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
                  builder.create<arith::ConstantIndexOp>(
                      loc, reduceInputDims[2] / reduceOutputDims[2]),
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
                          vectorizeCombineOpWithoutTerminator(
                              builder, loc, combineOp, args, vectorLength));
                    }
                    vectorizeCombineOpTerminator(builder, loc,
                                                 terminatorOperands);
                  });
              for (unsigned i = 0; i < numInputs; ++i) {
                initArgs[i] = loop0.getResult(i);
                b.tarJump(initArgs[i], inputTarStrides[i][2]);
              }
              for (unsigned i = 0; i < loopCnt * reduceOutputDims[2]; ++i) {
                for (unsigned j = 0; j < numOutputs; ++j) {
                  auto results =
                      vReduce(builder, loc, combineOp,
                              ValueRange(loop0.getResults().slice(
                                  numInputs + i * numOutputs, numOutputs)),
                              vectorLength);
                  for (unsigned k = 0; k < reduceOutputDims[2]; ++k) {
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

SmallVector<Value> ReduceGenerator::vectorizeCombineOpWithoutTerminator(
    OpBuilder &builder, Location loc, Region &combineOp, ValueRange operands,
    unsigned vectorLength) {
  IRMapping mapper;
  for (auto [arg, operand] : llvm::zip(combineOp.getArguments(), operands)) {
    mapper.map(arg, operand);
  }
  for (auto &o : combineOp.back().without_terminator()) {
    for (auto operand : o.getOperands()) {
      if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
        if (!mapper.lookupOrNull(operand)) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(constantOp);
          if (operand.getType().isInteger(1)) {
            auto boolAttr = dyn_cast<BoolAttr>(constantOp.getValue());
            auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
            if ((boolAttr && !boolAttr.getValue()) ||
                (integerAttr && integerAttr.getValue().isZero())) {
              mapper.map(operand,
                         builder.create<vector::ConstantMaskOp>(
                             loc,
                             VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             operand.getType()),
                             DenseI64ArrayAttr::get(builder.getContext(),
                                                    ArrayRef<int64_t>{0})));
            } else {
              mapper.map(
                  operand,
                  builder.create<vector::ConstantMaskOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      operand.getType()),
                      DenseI64ArrayAttr::get(builder.getContext(),
                                             ArrayRef<int64_t>{vectorLength})));
            }
          } else {
            mapper.map(operand,
                       builder.create<vector::BroadcastOp>(
                           loc,
                           VectorType::get(ArrayRef<int64_t>{vectorLength},
                                           operand.getType()),
                           operand));
          }
        }
      }
    }
    Operation *cloneOp;
    if (auto selectOp = dyn_cast<arith::SelectOp>(o)) {
      auto condition = selectOp.getCondition();
      auto mappingValue = mapper.lookup(condition);
      if (cast<VectorType>(mappingValue.getType())
              .getElementType()
              .isInteger(8)) {
        mapper.map(condition,
                   builder
                       .create<mlir::gcu::VectorConvertOp>(
                           loc,
                           VectorType::get(ArrayRef<int64_t>{vectorLength},
                                           builder.getIntegerType(1)),
                           mappingValue)
                       .getResult(0));
        cloneOp = builder.clone(o, mapper);
        mapper.map(condition, mappingValue);
      } else {
        cloneOp = builder.clone(o, mapper);
      }
    } else {
      cloneOp = builder.clone(o, mapper);
    }
    SmallVector<Type> resultTypes;
    auto typeInterface = dyn_cast<InferTypeOpInterface>(cloneOp);
    if (typeInterface &&
        succeeded(typeInterface.inferReturnTypes(
            cloneOp->getContext(), cloneOp->getLoc(), cloneOp->getOperands(),
            cloneOp->getAttrDictionary(), cloneOp->getPropertiesStorage(),
            cloneOp->getRegions(), resultTypes))) {
      for (unsigned i = 0; i < resultTypes.size(); ++i) {
        cloneOp->getResult(i).setType(resultTypes[i]);
        mapper.map(o.getResult(i), cloneOp->getResult(i));
      }
    } else {
      for (unsigned i = 0; i < o.getNumResults(); ++i) {
        auto elementTy = getElementTypeOrSelf(o.getResult(i).getType());
        cloneOp->getResult(i).setType(
            VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy));
        mapper.map(o.getResult(i), cloneOp->getResult(i));
      }
    }
  }
  auto terminatorOprands = llvm::to_vector(llvm::map_range(
      llvm::cast<triton::ReduceReturnOp>(combineOp.back().getTerminator())
          .getResult(),
      [&](auto v) {
        auto mappingValue = mapper.lookupOrNull(v);
        assert(mappingValue != nullptr);
        return mappingValue;
      }));
  return terminatorOprands;
}

void ReduceGenerator::vectorizeCombineOpTerminator(OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange operands) {
  builder.create<triton::ReduceReturnOp>(loc, operands);
}

std::optional<vector::CombiningKind>
ReduceGenerator::matchReduceCombiningKind(Region &combineOp) {
  auto &opList = combineOp.front().getOperations();
  if (opList.size() != 2) {
    return std::nullopt;
  }
  auto elementWiseOp = &opList.front();
  return TypeSwitch<Operation *, std::optional<vector::CombiningKind>>(
             elementWiseOp)
      .Case<triton::ExternElementwiseOp>(
          [&](auto externElementwiseOp)
              -> std::optional<vector::CombiningKind> {
            auto symbol = externElementwiseOp.getSymbol();
            if (symbol == "__nv_fmaxf") {
              return vector::CombiningKind::MAXIMUMF;
            } else if (symbol == "__nv_max") {
              return vector::CombiningKind::MAXSI;
            } else if (symbol == "__nv_umax") {
              return vector::CombiningKind::MAXUI;
            } else if (symbol == "__nv_fminf") {
              return vector::CombiningKind::MINIMUMF;
            } else if (symbol == "__nv_min") {
              return vector::CombiningKind::MINSI;
            } else if (symbol == "__nv_umin") {
              return vector::CombiningKind::MINUI;
            } else {
              return std::nullopt;
            }
          })
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return vector::CombiningKind::ADD; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return vector::CombiningKind::MUL; })
      .Case<arith::MaxSIOp>(
          [&](auto op) { return vector::CombiningKind::MAXSI; })
      .Case<arith::MaxUIOp>(
          [&](auto op) { return vector::CombiningKind::MAXUI; })
      .Case<arith::MaxNumFOp>(
          [&](auto op) { return vector::CombiningKind::MAXNUMF; })
      .Case<arith::MaximumFOp>(
          [&](auto op) { return vector::CombiningKind::MAXIMUMF; })
      .Case<arith::MinSIOp>(
          [&](auto op) { return vector::CombiningKind::MINSI; })
      .Case<arith::MinUIOp>(
          [&](auto op) { return vector::CombiningKind::MINUI; })
      .Case<arith::MinNumFOp>(
          [&](auto op) { return vector::CombiningKind::MINNUMF; })
      .Case<arith::MinimumFOp>(
          [&](auto op) { return vector::CombiningKind::MINIMUMF; })
      .Case<arith::AndIOp>([&](auto op) { return vector::CombiningKind::AND; })
      .Case<arith::OrIOp>([&](auto op) { return vector::CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return vector::CombiningKind::XOR; })
      .Default([&](auto op) { return std::nullopt; });
}

SmallVector<Value> ReduceGenerator::vReduce(OpBuilder &builder, Location loc,
                                            Region &combineOp,
                                            ValueRange vecValues,
                                            unsigned vectorLength) {
  assert(llvm::all_of(vecValues.getTypes(), [&](auto ty) {
    auto vecTy = dyn_cast<VectorType>(ty);
    return vecTy && vecTy.getRank() == 1 && vecTy.getDimSize(0) == vectorLength;
  }));

  if (auto kind = matchReduceCombiningKind(combineOp)) {
    assert(vecValues.size() == 1);
    return {builder.create<vector::ReductionOp>(loc, *kind, vecValues[0])};
  }

  SmallVector<Value> calValues;
  SmallVector<Type, 4> resultElemTypes;
  for (auto v : vecValues) {
    calValues.emplace_back(builder.create<vector::ExtractOp>(loc, v, 0));
    resultElemTypes.push_back(calValues.back().getType());
  }
  auto loop = builder.create<scf::ForOp>(
      loc, builder.create<arith::ConstantIndexOp>(loc, 1),
      builder.create<arith::ConstantIndexOp>(loc, vectorLength),
      builder.create<arith::ConstantIndexOp>(loc, 1), calValues,
      [&](OpBuilder &builder, Location loc, Value iter, ValueRange iterArgs) {
        SmallVector<Value, 4> operands(iterArgs.begin(), iterArgs.end());
        for (auto v : vecValues) {
          operands.emplace_back(
              builder.create<vector::ExtractOp>(loc, v, iter));
        }
        auto executeRegionOp =
            builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
        executeRegionOp.getRegion().emplaceBlock();
        IRMapping map;
        for (auto [arg, operand] :
             llvm::zip(combineOp.getArguments(), operands)) {
          map.map(arg, operand);
        }
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(
              &executeRegionOp.getRegion().getBlocks().back());
          for (auto &o : combineOp.getBlocks().back()) {
            auto newO = builder.clone(o, map);
            for (auto [result, newResult] :
                 llvm::zip(o.getResults(), newO->getResults())) {
              map.map(result, newResult);
            }
          }
        }
        builder.create<scf::YieldOp>(loc,
                                     ValueRange{executeRegionOp.getResults()});
      });
  return loop.getResults();
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
