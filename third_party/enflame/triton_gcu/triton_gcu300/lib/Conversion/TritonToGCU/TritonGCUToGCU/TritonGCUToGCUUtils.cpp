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

#include "TritonGCUToGCUUtils.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "Analysis/FirstLastUserAnalysis.h"
#include "ConstantUtil.h"
#include "Conversion/TritonToGCU/Utils.h"

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

#define DBG_TRITON_IR 1
using namespace mlir;

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

void doSlicePadOrMemsetSlice(OpBuilder &rewriter, Location loc, Operation *op,
                             Value output, Value src,
                             SmallVector<Value, 4> &offsets,
                             SmallVector<Value, 4> &sliceShape,
                             SmallVector<Value, 4> &padSizes,
                             Value defaultValue, Value tag, Value tagIdx) {
  auto maxPadSize = rewriter.create<arith::ConstantIndexOp>(loc, 2047);
  auto outputType = dyn_cast<MemRefType>(output.getType());
  auto legalPad =
      rewriter.create<arith::ConstantIntOp>(loc, 1, rewriter.getI1Type())
          .getResult();
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
        builder.create<memref_ext::SlicePadStartOp>(loc, output, src, offsets,
                                                    sliceShape, defaultValue,
                                                    tag, ValueRange{tagIdx});
        builder.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &childBuilder, Location loc) {
        doMemset(childBuilder, op, output, defaultValue, totalNumElems);
        childBuilder.create<memref_ext::SliceStartOp>(
            loc, output, src, offsets, defaultValue, tag, ValueRange{tagIdx});
        childBuilder.create<scf::YieldOp>(loc);
      });
}

void doMemFence(OpBuilder &rewriter, Operation *op) { /*NOLINT*/
  rewriter.create<gcu::MFenceOp>(op->getLoc());
}

void doMemsetConfig(OpBuilder &rewriter, Location loc, Value output, Value v,
                    Value tagDte, Value tagIdx) {
  rewriter.create<memref_ext::MemsetStartOp>(loc, output, v, tagDte,
                                             ValueRange{tagIdx});
}

void doMemset(OpBuilder &rewriter, Operation *op, Value output, Value v,
              unsigned totalNumElems) {
  auto loc = op->getLoc();
  if (totalNumElems > 128) {
    auto tag = getPrivateDTETag(rewriter, op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<memref_ext::MemsetStartOp>(loc, output, v, tag,
                                               ValueRange{zero});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag, ValueRange{zero},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
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
  if (llvm::isa<triton::AddPtrOp, triton::PtrToIntOp, triton::IntToPtrOp,
                triton::BitcastOp, gcu::PtrToMemRefOp, gcu::MemRefToPtrOp,
                gcu::IntToPtrOp, gcu::PtrToIntOp>(op)) {
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
      if (!mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout)) {
        return true;
      } else if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (cast<triton::gpu::SliceEncodingAttr>(srcLayout).getDim() ==
            cast<triton::gpu::SliceEncodingAttr>(dstLayout).getDim()) {
          return true;
        }
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
  } else {
    return false;
  }
}

// Find last user which is located at parent region of the op
mlir::Operation *
promoteLastUser(mlir::Operation *&lastUser,
                triton::gcu::FirstLastUserAnalysis &userAnalysis,
                std::map<Operation *, Operation *> &replaced2Origin) {
  if (!isa_and_nonnull<scf::YieldOp>(lastUser)) {
    return nullptr;
  }
  mlir::Operation *newAllocOpPos = nullptr;
  mlir::Operation *curLastUser = lastUser;
  mlir::Operation *parent = curLastUser->getParentOp();
  mlir::Operation *originParent = nullptr;

  while (isa<scf::IfOp, scf::IndexSwitchOp>(parent) &&
         isa<scf::YieldOp>(curLastUser)) {
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
    curLastUser = userAnalysis.getLastUserOp(originParent);
    parent = curLastUser->getParentOp();
  }

  lastUser = curLastUser;
  return newAllocOpPos;
}

void addDeallocAfterLastUser(OpBuilder &builder, mlir::Operation *lastUser,
                             Value alloc) {
  if (lastUser == nullptr) {
    return;
  }
  if (isa<scf::YieldOp>(lastUser) || isa<triton::ReduceReturnOp>(lastUser) ||
      isa<triton::ScanReturnOp>(lastUser) || isa<func::ReturnOp>(lastUser) ||
      isa<mlir::triton::gcu::YieldOp>(lastUser)) {
    return;
  }
  if (isa<MemRefType>(alloc.getType())) {
    if (lastUser->mightHaveTrait<OpTrait::IsTerminator>()) {
      return;
    }
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(lastUser);
    builder.create<memref::DeallocOp>(lastUser->getLoc(), alloc);
    builder.restoreInsertionPoint(ip);
  }
  return;
}

// lowering
Value syncAllocOp(OpBuilder &builder, Location &loc, Operation *lastUser,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  MemRefType type) {
  auto newAllocOpPos = promoteLastUser(lastUser, userAnalysis, replaced2Origin);

  Value output;
  if (newAllocOpPos == nullptr) {
    output = builder.create<memref::AllocOp>(loc, type);
  } else {
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(newAllocOpPos);
    output = builder.create<memref::AllocOp>(loc, type);
    builder.restoreInsertionPoint(ip);
  }
  addDeallocAfterLastUser(builder, lastUser, output);
  return output;
}

Value asyncAllocOp(OpBuilder &builder, Operation *ttParent, MemRefType type) {
  return builder.create<memref::AllocOp>(ttParent->getLoc(), type);
}

void createPrintfOp(ConversionPatternRewriter &rewriter, Location loc,
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

Value loadFromSharedMem(OpBuilder &builder, Value tag, Type type, Value buffer,
                        bool onlyThread0, Operation *lastTTUser,
                        Operation *firstTTUser,
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
        loc,
        builder.create<arith::ConstantIntOp>(loc, numElems[i],
                                             builder.getI32Type()),
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
  if (!firstTTUser) {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            if (isNeedMerge) {
              mergeContinuousDims(builder, loc, src, dst, offsets,
                                  mergedOffsets, srcType, outputType, buffer,
                                  output);
              builder.create<memref_ext::SliceStartOp>(
                  loc, dst, src, mergedOffsets, defaultValue, tag,
                  ValueRange{zero});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(loc, output, buffer,
                                                       offsets, defaultValue,
                                                       tag, ValueRange{zero});
            }
            builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                              totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      if (isNeedMerge) {
        mergeContinuousDims(builder, loc, src, dst, offsets, mergedOffsets,
                            srcType, outputType, buffer, output);
        builder.create<memref_ext::SliceStartOp>(
            loc, dst, src, mergedOffsets, defaultValue, tag, ValueRange{zero});
        auto [oriOutputStrides, oriOutputOffset] =
            outputType.getStridesAndOffset();
        builder.create<memref::ReinterpretCastOp>(
            loc, outputType, dst, oriOutputOffset,
            SmallVector<int64_t>(numElems.begin(), numElems.end()),
            oriOutputStrides);
      } else {
        builder.create<memref_ext::SliceStartOp>(
            loc, output, buffer, offsets, defaultValue, tag, ValueRange{zero});
      }
      builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                        totalNumElems);
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
                  loc, dst, src, mergedOffsets, defaultValue, tag,
                  ValueRange{zero});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(loc, output, buffer,
                                                       offsets, defaultValue,
                                                       tag, ValueRange{zero});
            }
            builder.create<scf::YieldOp>(loc);
          });
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser);
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                              totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
      builder.restoreInsertionPoint(ip);
    } else {
      if (isNeedMerge) {
        mergeContinuousDims(builder, loc, src, dst, offsets, mergedOffsets,
                            srcType, outputType, buffer, output);
        builder.create<memref_ext::SliceStartOp>(
            loc, dst, src, mergedOffsets, defaultValue, tag, ValueRange{zero});
        auto [oriOutputStrides, oriOutputOffset] =
            outputType.getStridesAndOffset();
        builder.create<memref::ReinterpretCastOp>(
            loc, outputType, dst, oriOutputOffset,
            SmallVector<int64_t>(numElems.begin(), numElems.end()),
            oriOutputStrides);
      } else {
        builder.create<memref_ext::SliceStartOp>(
            loc, output, buffer, offsets, defaultValue, tag, ValueRange{zero});
      }
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser);
      builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                        totalNumElems);
      builder.restoreInsertionPoint(ip);
    }
  }
  return output;
}

Value CopyFromSharedMem(OpBuilder &builder, Value tag, Type type, Value buffer,
                        bool onlyThread0, Operation *lastTTUser,
                        Operation *firstTTUser,
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
  if (!firstTTUser) {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaStartOp>(
                loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
                SmallVector<Value, 4>(shape.size(), zero), totalNumElems, tag,
                ValueRange{zero});
            builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                              totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      builder.create<memref::DmaStartOp>(
          loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
          SmallVector<Value, 4>(shape.size(), zero), totalNumElems, tag,
          ValueRange{zero});
      builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                        totalNumElems);
    }
  } else {
    if (onlyThread0) {
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaStartOp>(
                loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
                SmallVector<Value, 4>(shape.size(), zero), totalNumElems, tag,
                ValueRange{zero});
            builder.create<scf::YieldOp>(loc);
          });
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser);
      builder.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                              totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
      builder.restoreInsertionPoint(ip);
    } else {
      builder.create<memref::DmaStartOp>(
          loc, buffer, SmallVector<Value, 4>(shape.size(), zero), output,
          SmallVector<Value, 4>(shape.size(), zero), totalNumElems, tag,
          ValueRange{zero});
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(firstTTUser);
      builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                        totalNumElems);
      builder.restoreInsertionPoint(ip);
    }
  }
  return output;
}

Value loadFromSharedMemForDotOperand(
    OpBuilder builder, Value tag, Type type, ArrayRef<int64_t> mnShape,
    Value sharedBuffer, Operation *lastTTUser, Operation *firstTTUser,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin) {
  auto loc = sharedBuffer.getLoc();
  auto srcType = dyn_cast<MemRefType>(sharedBuffer.getType());
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
        loc,
        builder.create<arith::ConstantIntOp>(loc, numElems[i],
                                             builder.getI32Type()),
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                           warpIds[i])));
  }

  auto output = syncAllocOp(builder, loc, lastTTUser, userAnalysis,
                            replaced2Origin, outputType);
  auto defaultValue =
      triton::gcu::createConstantZero(builder, loc, srcType.getElementType());
  builder.create<memref_ext::SliceStartOp>(loc, output, sharedBuffer, offsets,
                                           defaultValue, tag, ValueRange{zero});
  if (firstTTUser == nullptr) {
    builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                      totalNumElems);
  } else {
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(firstTTUser);
    builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                      totalNumElems);
    builder.restoreInsertionPoint(ip);
  }
  return output;
}

void storeToSharedMem(OpBuilder &builder, Value tag, TensorType type,
                      Value sharedBuffer, Value buffer, bool onlyThread0) {
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
        builder.create<arith::ConstantIntOp>(loc, srcType.getDimSize(i),
                                             builder.getI32Type()),
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
                loc, dst, src, mergedOffsets, tag, ValueRange{zero});
            builder.create<memref::ReinterpretCastOp>(
                loc, outputType, dst, oriOutputOffset, outputSize,
                oriOutputStrides);
          } else {
            builder.create<memref_ext::DesliceStartOp>(
                loc, sharedBuffer, buffer, offsets, tag, ValueRange{zero});
          }
          builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                            totalNumElems);
          builder.create<scf::YieldOp>(loc);
        });
  } else {
    if (isNeedMerge) {
      mergeContinuousDims(builder, loc, dst, src, offsets, mergedOffsets,
                          outputType, srcType, sharedBuffer, buffer);
      auto [oriOutputStrides, oriOutputOffset] =
          outputType.getStridesAndOffset();
      builder.create<memref_ext::DesliceStartOp>(loc, dst, src, mergedOffsets,
                                                 tag, ValueRange{zero});
      builder.create<memref::ReinterpretCastOp>(
          loc, outputType, dst, oriOutputOffset, outputSize, oriOutputStrides);
    } else {
      builder.create<memref_ext::DesliceStartOp>(
          loc, sharedBuffer, buffer, offsets, tag, ValueRange{zero});
    }
    builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                      totalNumElems);
  }
  builder.create<gpu::BarrierOp>(loc);
}

Value storeToSharedMem(OpBuilder &builder, Value tag, TensorType type,
                       Value buffer, bool onlyThread0, Operation *lastTTUser,
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
          auto lastOp = userAnalysis.getLastUserOp(
              reginArg, definingOp->getParentRegion());
          if (lastOp == nullptr) {
            TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
          } else {
            if (definingOp->isBeforeInBlock(lastOp)) {
              TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
            }
          }
        } else if (isa<TensorType>(operand.getType()) &&
                   isa<scf::WhileOp>(op.getOperation()->getParentOp())) {
          auto whileOp =
              llvm::cast<scf::WhileOp>(op.getOperation()->getParentOp());
          mlir::Value reginArg = whileOp.getAfterArguments()[i];
          auto lastOp = userAnalysis.getLastUserOp(
              reginArg, definingOp->getParentRegion());
          if (lastOp == nullptr) {
            TTYeiledOPerandHasMultiUseStage[op.getOperation()][i] = true;
          } else {
            if (definingOp->isBeforeInBlock(lastOp)) {
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
  auto elementTyIdx = rewriter.getIndexType();
  auto elementTyI32 = rewriter.getI32Type();
  auto tmpStrideBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyIdx));
  auto tmpShapeBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyIdx));
  auto tmpOffsetBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyIdx));
  auto tmpOrderBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyI32));
  for (unsigned i = 0; i < rank; ++i) {
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    rewriter.create<memref::StoreOp>(loc, initStride[i], tmpStrideBuffer,
                                     ValueRange{idx});
    rewriter.create<memref::StoreOp>(loc, initShape[i], tmpShapeBuffer,
                                     ValueRange{idx});
    rewriter.create<memref::StoreOp>(loc, initOffset[i], tmpOffsetBuffer,
                                     ValueRange{idx});
    rewriter.create<memref::StoreOp>(
        loc,
        rewriter.create<arith::ConstantIntOp>(loc, nInitStrideDims[i],
                                              rewriter.getI32Type()),
        tmpOrderBuffer, ValueRange{idx});
  }

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value cEnd = rewriter.create<arith::ConstantIndexOp>(loc, rank - 1);
  Value cStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::buildLoopNest(
      rewriter, loc, zero, cEnd, cStep,
      [&](OpBuilder &outerBuilder, Location outerLoc, ValueRange ivsOuter) {
        Value i = ivsOuter[0];
        Value cInnerEnd = outerBuilder.create<arith::SubIOp>(outerLoc, cEnd, i);
        scf::buildLoopNest(
            outerBuilder, outerLoc, zero, cInnerEnd, cStep,
            [&](OpBuilder &innerBuilder, Location innerLoc,
                ValueRange ivsInner) {
              Value j = ivsInner[0];
              Value jNext =
                  innerBuilder.create<arith::AddIOp>(innerLoc, j, cStep);

              Value vStrideJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpStrideBuffer, j);
              Value vStrideJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpStrideBuffer, jNext);
              Value vShapeJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpShapeBuffer, j);
              Value vShapeJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpShapeBuffer, jNext);
              Value vOffsetJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpOffsetBuffer, j);
              Value vOffsetJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpOffsetBuffer, jNext);
              Value vOrderJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpOrderBuffer, j);
              Value vOrderJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpOrderBuffer, jNext);

              Value cmp = innerBuilder.create<arith::CmpIOp>(
                  innerLoc, arith::CmpIPredicate::slt, vStrideJ, vStrideJNext);
              innerBuilder.create<scf::IfOp>(
                  innerLoc, cmp,
                  [&](OpBuilder &thenBuilder, Location thenLoc) {
                    thenBuilder.create<memref::StoreOp>(thenLoc, vStrideJNext,
                                                        tmpStrideBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vStrideJ,
                                                        tmpStrideBuffer, jNext);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vShapeJNext,
                                                        tmpShapeBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vShapeJ,
                                                        tmpShapeBuffer, jNext);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vOffsetJNext,
                                                        tmpOffsetBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vOffsetJ,
                                                        tmpOffsetBuffer, jNext);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vOrderJNext,
                                                        tmpOrderBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vOrderJ,
                                                        tmpOrderBuffer, jNext);
                    thenBuilder.create<scf::YieldOp>(thenLoc);
                  },
                  [&](OpBuilder &elseBuilder, Location elseLoc) {
                    elseBuilder.create<scf::YieldOp>(elseLoc);
                  });
            });
      });

  for (unsigned i = 0; i < rank; ++i) {
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    vOrder.push_back(
        rewriter.create<memref::LoadOp>(loc, tmpOrderBuffer, ValueRange{idx}));
    orderStride.push_back(
        rewriter.create<memref::LoadOp>(loc, tmpStrideBuffer, ValueRange{idx}));
    orderOffset.push_back(rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(),
        rewriter.create<memref::LoadOp>(loc, tmpOffsetBuffer,
                                        ValueRange{idx})));
  }
  orderShape.push_back(
      rewriter.create<memref::LoadOp>(loc, tmpShapeBuffer, ValueRange{zero}));
  for (unsigned i = 0; i < rank - 1; ++i) {
    orderShape.push_back(rewriter.create<arith::DivSIOp>(loc, orderStride[i],
                                                         orderStride[i + 1]));
  }

  rewriter.create<memref::DeallocOp>(loc, tmpStrideBuffer);
  rewriter.create<memref::DeallocOp>(loc, tmpShapeBuffer);
  rewriter.create<memref::DeallocOp>(loc, tmpOffsetBuffer);
  rewriter.create<memref::DeallocOp>(loc, tmpOrderBuffer);
}

void GetOrderSlicefor30(OpBuilder &rewriter, Location loc, int64_t rank,
                        SmallVector<Value, 4> &initStride,
                        SmallVector<Value, 4> &initSliceShape,
                        SmallVector<Value, 4> &orderSliceShape) {
  auto elementTyIdx = rewriter.getIndexType();
  auto tmpStrideBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyIdx));
  auto tmpShapeBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyIdx));
  for (unsigned i = 0; i < rank; ++i) {
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    rewriter.create<memref::StoreOp>(loc, initStride[i], tmpStrideBuffer,
                                     ValueRange{idx});
    rewriter.create<memref::StoreOp>(loc, initSliceShape[i], tmpShapeBuffer,
                                     ValueRange{idx});
  }

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value cEnd = rewriter.create<arith::ConstantIndexOp>(loc, rank - 1);
  Value cStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::buildLoopNest(
      rewriter, loc, zero, cEnd, cStep,
      [&](OpBuilder &outerBuilder, Location outerLoc, ValueRange ivsOuter) {
        Value i = ivsOuter[0];
        Value cInnerEnd = outerBuilder.create<arith::SubIOp>(outerLoc, cEnd, i);
        scf::buildLoopNest(
            outerBuilder, outerLoc, zero, cInnerEnd, cStep,
            [&](OpBuilder &innerBuilder, Location innerLoc,
                ValueRange ivsInner) {
              Value j = ivsInner[0];
              Value jNext =
                  innerBuilder.create<arith::AddIOp>(innerLoc, j, cStep);

              Value vStrideJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpStrideBuffer, j);
              Value vStrideJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpStrideBuffer, jNext);
              Value vShapeJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpShapeBuffer, j);
              Value vShapeJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpShapeBuffer, jNext);

              Value cmp = innerBuilder.create<arith::CmpIOp>(
                  innerLoc, arith::CmpIPredicate::slt, vStrideJ, vStrideJNext);
              innerBuilder.create<scf::IfOp>(
                  innerLoc, cmp,
                  [&](OpBuilder &thenBuilder, Location thenLoc) {
                    thenBuilder.create<memref::StoreOp>(thenLoc, vStrideJNext,
                                                        tmpStrideBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vStrideJ,
                                                        tmpStrideBuffer, jNext);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vShapeJNext,
                                                        tmpShapeBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vShapeJ,
                                                        tmpShapeBuffer, jNext);
                    thenBuilder.create<scf::YieldOp>(thenLoc);
                  },
                  [&](OpBuilder &elseBuilder, Location elseLoc) {
                    elseBuilder.create<scf::YieldOp>(elseLoc);
                  });
            });
      });

  for (unsigned i = 0; i < rank; ++i) {
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    orderSliceShape.push_back(
        rewriter.create<memref::LoadOp>(loc, tmpShapeBuffer, ValueRange{idx}));
  }

  rewriter.create<memref::DeallocOp>(loc, tmpStrideBuffer);
  rewriter.create<memref::DeallocOp>(loc, tmpShapeBuffer);
}

void GetTransByOrder(OpBuilder &rewriter, Location loc,
                     SmallVector<Value, 4> &order,
                     SmallVector<Value, 4> &transOrder) {
  unsigned rank = order.size();

  auto elementTyI32 = rewriter.getI32Type();
  auto tmpOrderBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyI32));
  auto tmpTransOrderBuffer = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(rank, elementTyI32));
  for (unsigned i = 0; i < rank; ++i) {
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    rewriter.create<memref::StoreOp>(
        loc,
        rewriter.create<arith::ConstantIntOp>(loc, i, rewriter.getI32Type()),
        tmpTransOrderBuffer, ValueRange{idx});
    rewriter.create<memref::StoreOp>(loc, order[i], tmpOrderBuffer,
                                     ValueRange{idx});
  }

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value cEnd = rewriter.create<arith::ConstantIndexOp>(loc, rank - 1);
  Value cStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::buildLoopNest(
      rewriter, loc, zero, cEnd, cStep,
      [&](OpBuilder &outerBuilder, Location outerLoc, ValueRange ivsOuter) {
        Value i = ivsOuter[0];
        Value cInnerEnd = outerBuilder.create<arith::SubIOp>(outerLoc, cEnd, i);
        scf::buildLoopNest(
            outerBuilder, outerLoc, zero, cInnerEnd, cStep,
            [&](OpBuilder &innerBuilder, Location innerLoc,
                ValueRange ivsInner) {
              Value j = ivsInner[0];
              Value jNext =
                  innerBuilder.create<arith::AddIOp>(innerLoc, j, cStep);

              Value vOrderJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpOrderBuffer, j);
              Value vOrderJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpOrderBuffer, jNext);
              Value vTransJ = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpTransOrderBuffer, j);
              Value vTransJNext = innerBuilder.create<memref::LoadOp>(
                  innerLoc, tmpTransOrderBuffer, jNext);

              Value cmp = innerBuilder.create<arith::CmpIOp>(
                  innerLoc, arith::CmpIPredicate::sgt, vOrderJ, vOrderJNext);
              innerBuilder.create<scf::IfOp>(
                  innerLoc, cmp,
                  [&](OpBuilder &thenBuilder, Location thenLoc) {
                    thenBuilder.create<memref::StoreOp>(thenLoc, vOrderJNext,
                                                        tmpOrderBuffer, j);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vOrderJ,
                                                        tmpOrderBuffer, jNext);
                    thenBuilder.create<memref::StoreOp>(thenLoc, vTransJNext,
                                                        tmpTransOrderBuffer, j);
                    thenBuilder.create<memref::StoreOp>(
                        thenLoc, vTransJ, tmpTransOrderBuffer, jNext);
                    thenBuilder.create<scf::YieldOp>(thenLoc);
                  },
                  [&](OpBuilder &elseBuilder, Location elseLoc) {
                    elseBuilder.create<scf::YieldOp>(elseLoc);
                  });
            });
      });

  for (unsigned i = 0; i < rank; ++i) {
    auto idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    transOrder.push_back(rewriter.create<memref::LoadOp>(
        loc, tmpTransOrderBuffer, ValueRange{idx}));
  }

  rewriter.create<memref::DeallocOp>(loc, tmpOrderBuffer);
  rewriter.create<memref::DeallocOp>(loc, tmpTransOrderBuffer);
}

Value ConfigGcuLoad(OpBuilder &rewriter, Location loc, Value srcOut,
                    Value transOut, mlir::Operation *op, MemRefType resultType,
                    Value loadPtr, mlir::ValueRange configStrides,
                    mlir::ValueRange configShapes, Value defaultValue,
                    Value tagDte, Value tagIdx, bool IsShareOutput) {
  if ((!llvm::isa_and_nonnull<triton::gcu::LoadOp>(op)) &&
      (!llvm::isa_and_nonnull<triton::gcu::AsyncLoadGlobalToShareOp>(op))) {
    assert(false && "please check IR ConfigGcuLoad got a bad input op");
  }
  auto getOrderHint = [](mlir::Operation *op) {
    if (llvm::isa_and_nonnull<triton::gcu::LoadOp>(op)) {
      auto load = llvm::cast<triton::gcu::LoadOp>(op);
      return load.getOrderHint();
    } else if (llvm::isa_and_nonnull<triton::gcu::AsyncLoadGlobalToShareOp>(
                   op)) {
      auto dynamicLoad = llvm::cast<triton::gcu::AsyncLoadGlobalToShareOp>(op);
      return dynamicLoad.getOrderHint();
    } else {
      assert(false && "please check IR ConfigGcuLoad got a bad input op");
    }
    return llvm::ArrayRef<int32_t>();
  };
  auto getDefaultValue = [](mlir::Operation *op) {
    if (llvm::isa_and_nonnull<triton::gcu::LoadOp>(op)) {
      auto load = llvm::cast<triton::gcu::LoadOp>(op);
      return load.getDefaultValue();
    } else if (llvm::isa_and_nonnull<triton::gcu::AsyncLoadGlobalToShareOp>(
                   op)) {
      auto dynamicLoad = llvm::cast<triton::gcu::AsyncLoadGlobalToShareOp>(op);
      return dynamicLoad.getDefaultValue();
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
      auto trueCondition = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, configStrides[i], zero);
      rewriter.create<triton::gcu::AssertOp>(
          loc, trueCondition,
          "Not Support dynamic stride is 0, please add tl.constexpr to stride "
          "arg in kernel args list",
          "", "", 0);
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
    // currently pinpong only support share layout
    assert((!llvm::isa_and_nonnull<triton::gcu::AsyncLoadGlobalToShareOp>(op)));
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

  SmallVector<Value, 4> vResultShapes;
  SmallVector<int64_t, 4> resultShapes;
  for (unsigned i = 0; i < rank; ++i) {
    resultShapes.push_back(resultType.getShape()[i]);
    vResultShapes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, resultShapes[i]));
  }

  Value reshapeOut = srcOut;
  if (bReshape) {
    assert(rank < 4 && "not support stride is no 1 for rank >=4");
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
    auto reshapeMemrefType = MemRefType::get(resultShapes, elemType);
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
      vTempOrder.insert(vTempOrder.begin() + updateDim,
                        rewriter.create<arith::ConstantIntOp>(
                            loc, updateDim, rewriter.getI32Type()));
    }
    GetTransByOrder(rewriter, loc, vTempOrder, vTransOrder);
  } else {
    SmallVector<int32_t, 4> static_order(order_hint.begin(), order_hint.end());
    for (int i = 0; i < rank; ++i) {
      vOrderStrides.push_back(vSrcStrides[static_order[i]]);
      vOrderOffsets.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), vSrcOffsets[static_order[i]]));
      vTransOrder.push_back(rewriter.create<arith::ConstantIntOp>(
          loc, static_order[i], rewriter.getI32Type()));
    }
    if (static_order.size() > 0)
      vOrderShapes.push_back(vSrcShapes[static_order[0]]);
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
        rewriter.create<arith::ConstantIntOp>(loc, i, rewriter.getI32Type()));
    isDynamicTrans =
        rewriter.create<arith::OrIOp>(loc, isDynamicTrans, isDimTrans);
  }

  auto sourceType = MemRefType::get(
      SmallVector<int64_t>(rank, ShapedType::kDynamic), elemType);
  auto src = rewriter.create<memref::ReinterpretCastOp>(
      loc, sourceType, buffer, zero, vOrderShapes, vOrderStrides);

  if (IsShareOutput) {
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    auto isPad = rewriter.create<arith::AndIOp>(loc, isNeedPad, isThread0);
    if (bDynamicStride) {
      auto isTrans =
          rewriter.create<arith::AndIOp>(loc, isDynamicTrans, isThread0);
      auto isAll = rewriter.create<arith::AndIOp>(loc, isPad, isTrans);
      rewriter.create<scf::IfOp>(
          loc, isAll,
          [&](OpBuilder &builder, Location loc) {
            auto trans_buffer = rewriter.create<memref::ReinterpretCastOp>(
                loc, sourceType, transOut, zero, vSlicehape, vSrcStrides);
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, trans_buffer, src, vOrderOffsets, vTransOrder,
                defaultValue, tagDte, ValueRange{tagIdx});
            builder.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                              totalSize);
            doSlicePadOrMemsetSlice(builder, loc, op, reshapeOut, trans_buffer,
                                    padOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tagDte, tagIdx);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isTrans,
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceTransposeStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, vTransOrder,
                      defaultValue, tagDte, ValueRange{tagIdx});
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<scf::IfOp>(
                      loc, isPad,
                      [&](OpBuilder &child2Builder, Location loc) {
                        doSlicePadOrMemsetSlice(child2Builder, loc, op,
                                                reshapeOut, src, vOrderOffsets,
                                                vIntSlicehape, padSizes,
                                                defaultValue, tagDte, tagIdx);
                        child2Builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &child2Builder, Location loc) {
                        child2Builder.create<scf::IfOp>(
                            loc, isThread0,
                            [&](OpBuilder &child3Builder, Location loc) {
                              child3Builder.create<memref_ext::SliceStartOp>(
                                  loc, reshapeOut, src, vOrderOffsets,
                                  defaultValue, tagDte, ValueRange{tagIdx});
                              child3Builder.create<scf::YieldOp>(loc);
                            });
                        child2Builder.create<scf::YieldOp>(loc);
                      });
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    } else if (bStaticTranspose) {
      rewriter.create<scf::IfOp>(
          loc, isPad,
          [&](OpBuilder &builder, Location loc) {
            auto trans_buffer = rewriter.create<memref::ReinterpretCastOp>(
                loc, sourceType, transOut, zero, vSlicehape, vSrcStrides);
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, trans_buffer, src, vOrderOffsets, vTransOrder,
                defaultValue, tagDte, ValueRange{tagIdx});
            builder.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                              totalSize);
            doSlicePadOrMemsetSlice(builder, loc, op, reshapeOut, trans_buffer,
                                    padOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tagDte, tagIdx);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isThread0, [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceTransposeStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, vTransOrder,
                      defaultValue, tagDte, ValueRange{tagIdx});
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
                                    defaultValue, tagDte, tagIdx);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isThread0, [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, defaultValue, tagDte,
                      ValueRange{tagIdx});
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
      auto isAll = rewriter.create<arith::AndIOp>(loc, isPad, isTrans);
      rewriter.create<scf::IfOp>(
          loc, isAll,
          [&](OpBuilder &builder, Location loc) {
            auto trans_buffer = rewriter.create<memref::ReinterpretCastOp>(
                loc, sourceType, transOut, zero, vSlicehape, vSrcStrides);
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, trans_buffer, src, vOrderOffsets, vTransOrder,
                defaultValue, tagDte, ValueRange{tagIdx});
            builder.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                              totalSize);
            doSlicePadOrMemsetSlice(builder, loc, op, reshapeOut, trans_buffer,
                                    padOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tagDte, tagIdx);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isTrans,
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceTransposeStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, vTransOrder,
                      defaultValue, tagDte, ValueRange{tagIdx});
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<scf::IfOp>(
                      loc, isPad,
                      [&](OpBuilder &child2Builder, Location loc) {
                        doSlicePadOrMemsetSlice(child2Builder, loc, op,
                                                reshapeOut, src, vOrderOffsets,
                                                vIntSlicehape, padSizes,
                                                defaultValue, tagDte, tagIdx);
                        child2Builder.create<scf::YieldOp>(loc);
                      },
                      [&](OpBuilder &child2Builder, Location loc) {
                        child2Builder.create<scf::IfOp>(
                            loc, isNotZero,
                            [&](OpBuilder &child3Builder, Location loc) {
                              child3Builder.create<memref_ext::SliceStartOp>(
                                  loc, reshapeOut, src, vOrderOffsets,
                                  defaultValue, tagDte, ValueRange{tagIdx});
                              child3Builder.create<scf::YieldOp>(loc);
                            },
                            [&](OpBuilder &child3Builder, Location loc) {
                              if (getDefaultValue(op)) {
                                doMemsetConfig(child3Builder, loc, reshapeOut,
                                               defaultValue, tagDte, tagIdx);
                              }
                              child3Builder.create<scf::YieldOp>(loc);
                            });
                        child2Builder.create<scf::YieldOp>(loc);
                      });
                  childBuilder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    } else if (bStaticTranspose) {
      rewriter.create<scf::IfOp>(
          loc, isPad,
          [&](OpBuilder &builder, Location loc) {
            auto trans_buffer = rewriter.create<memref::ReinterpretCastOp>(
                loc, sourceType, transOut, zero, vSlicehape, vSrcStrides);
            builder.create<memref_ext::SliceTransposeStartOp>(
                loc, trans_buffer, src, vOrderOffsets, vTransOrder,
                defaultValue, tagDte, ValueRange{tagIdx});
            builder.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                              totalSize);
            doSlicePadOrMemsetSlice(builder, loc, op, reshapeOut, trans_buffer,
                                    padOffsets, vIntSlicehape, padSizes,
                                    defaultValue, tagDte, tagIdx);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isNotZero,
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceTransposeStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, vTransOrder,
                      defaultValue, tagDte, ValueRange{tagIdx});
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  if (getDefaultValue(op)) {
                    doMemsetConfig(childBuilder, loc, reshapeOut, defaultValue,
                                   tagDte, tagIdx);
                  }
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
                                    defaultValue, tagDte, tagIdx);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(
                loc, isNotZero,
                [&](OpBuilder &childBuilder, Location loc) {
                  childBuilder.create<memref_ext::SliceStartOp>(
                      loc, reshapeOut, src, vOrderOffsets, defaultValue, tagDte,
                      ValueRange{tagIdx});
                  childBuilder.create<scf::YieldOp>(loc);
                },
                [&](OpBuilder &childBuilder, Location loc) {
                  if (getDefaultValue(op)) {
                    doMemsetConfig(childBuilder, loc, reshapeOut, defaultValue,
                                   tagDte, tagIdx);
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
                     Value transOut, mlir::Operation *op,
                     MemRefType storeValueType, Value storePtr,
                     mlir::ValueRange configStrides,
                     mlir::ValueRange configShapes, Value tagDte,
                     Value tagIdx) {
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
    assert(rank < 4 && "not support stride is no 1 for rank >=4");
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
      vTempOrder.insert(vTempOrder.begin() + updateDim,
                        rewriter.create<arith::ConstantIntOp>(
                            loc, updateDim, rewriter.getI32Type()));
    }
    GetTransByOrder(rewriter, loc, vTempOrder, vTransOrder);
  } else {
    SmallVector<int32_t, 4> static_order(order_hint.begin(), order_hint.end());
    for (int i = 0; i < rank; ++i) {
      vOrderStrides.push_back(vSrcStrides[static_order[i]]);
      vOrderOffsets.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), vSrcOffsets[static_order[i]]));
      vTransOrder.push_back(rewriter.create<arith::ConstantIntOp>(
          loc, static_order[i], rewriter.getI32Type()));
    }
    if (static_order.size() > 0)
      vOrderShapes.push_back(vSrcShapes[static_order[0]]);
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

  SmallVector<Value, 4> vOrderSlicehape;
  if (bDynamicStride) {
    GetOrderSlicefor30(rewriter, loc, rank, vSrcStrides, vSlicehape,
                       vOrderSlicehape);
  } else {
    for (int i = 0; i < rank; ++i) {
      SmallVector<int32_t, 4> static_order(order_hint.begin(),
                                           order_hint.end());
      vOrderSlicehape.push_back(vSlicehape[static_order[i]]);
    }
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
        rewriter.create<arith::ConstantIntOp>(loc, i, rewriter.getI32Type()));
    isDynamicTrans =
        rewriter.create<arith::OrIOp>(loc, isDynamicTrans, isDimTrans);
  }

  if (bDynamicStride) {
    auto isTrans =
        rewriter.create<arith::AndIOp>(loc, isDynamicTrans, isNotZero);
    auto isSlice = rewriter.create<arith::AndIOp>(loc, isNeedSlice, isNotZero);
    auto isAll = rewriter.create<arith::AndIOp>(loc, isSlice, isTrans);
    rewriter.create<scf::IfOp>(
        loc, isAll,
        [&](OpBuilder &builder, Location loc) {
          auto trans_buffer = rewriter.create<memref::ReinterpretCastOp>(
              loc, resultType, transOut, zero, vOrderSlicehape, vOrderStrides);
          builder.create<memref_ext::SliceTransposeStartOp>(
              loc, trans_buffer, reshapeStoreValue, sliceOffsets, vTransOrder,
              triton::gcu::createConstantZero(rewriter, loc, elemType), tagDte,
              ValueRange{tagIdx});
          builder.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                            totalSize);
          builder.create<memref_ext::DesliceStartOp>(loc, dst, trans_buffer,
                                                     vOrderOffsets, tagDte,
                                                     ValueRange{tagIdx});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isTrans,
              [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::TransposeDesliceStartOp>(
                    loc, dst, reshapeStoreValue, vTransOrder, vOrderOffsets,
                    tagDte, ValueRange{tagIdx});
                childBuilder.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<scf::IfOp>(
                    loc, isSlice,
                    [&](OpBuilder &child2Builder, Location loc) {
                      child2Builder.create<memref_ext::SliceDesliceStartOp>(
                          loc, dst, reshapeStoreValue, sliceOffsets,
                          vIntSlicehape, vOrderOffsets, tagDte,
                          ValueRange{tagIdx});
                      child2Builder.create<scf::YieldOp>(loc);
                    },
                    [&](OpBuilder &child2Builder, Location loc) {
                      child2Builder.create<scf::IfOp>(
                          loc, isNotZero,
                          [&](OpBuilder &child3Builder, Location loc) {
                            child3Builder.create<memref_ext::DesliceStartOp>(
                                loc, dst, reshapeStoreValue, vOrderOffsets,
                                tagDte, ValueRange{tagIdx});
                            child3Builder.create<scf::YieldOp>(loc);
                          });
                      child2Builder.create<scf::YieldOp>(loc);
                    });
                childBuilder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  } else if (bStaticTranspose) {
    auto isSlice = rewriter.create<arith::AndIOp>(loc, isNeedSlice, isNotZero);
    rewriter.create<scf::IfOp>(
        loc, isSlice,
        [&](OpBuilder &builder, Location loc) {
          auto trans_buffer = rewriter.create<memref::ReinterpretCastOp>(
              loc, resultType, transOut, zero, vOrderSlicehape, vOrderStrides);
          builder.create<memref_ext::SliceTransposeStartOp>(
              loc, trans_buffer, reshapeStoreValue, sliceOffsets, vTransOrder,
              triton::gcu::createConstantZero(rewriter, loc, elemType), tagDte,
              ValueRange{tagIdx});
          builder.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                            totalSize);
          builder.create<memref_ext::DesliceStartOp>(loc, dst, trans_buffer,
                                                     vOrderOffsets, tagDte,
                                                     ValueRange{tagIdx});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isNotZero, [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::TransposeDesliceStartOp>(
                    loc, dst, reshapeStoreValue, vTransOrder, vOrderOffsets,
                    tagDte, ValueRange{tagIdx});
                childBuilder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  } else {
    auto isSlice = rewriter.create<arith::AndIOp>(loc, isNeedSlice, isNotZero);
    rewriter.create<scf::IfOp>(
        loc, isSlice,
        [&](OpBuilder &builder, Location loc) {
          builder.create<memref_ext::SliceDesliceStartOp>(
              loc, dst, reshapeStoreValue, sliceOffsets, vIntSlicehape,
              vOrderOffsets, tagDte, ValueRange{tagIdx});
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::IfOp>(
              loc, isNotZero, [&](OpBuilder &childBuilder, Location loc) {
                childBuilder.create<memref_ext::DesliceStartOp>(
                    loc, dst, reshapeStoreValue, vOrderOffsets, tagDte,
                    ValueRange{tagIdx});
                childBuilder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  }
  return totalSize;
}

void WaitGcuLoadStore(OpBuilder &rewriter, Location loc, Value tagDte,
                      Value tagIdx, Value totalSize) {
  rewriter.create<memref::DmaWaitOp>(loc, tagDte, ValueRange{tagIdx},
                                     totalSize);
}

void moveDeallocOp(ConversionPatternRewriter &rewriter, Value v, Operation *pos,
                   size_t depth) {
  if (depth > 1)
    return;

  Operation *allocOp = v.getDefiningOp();
  if (llvm::isa<mlir::UnrealizedConversionCastOp>(allocOp)) {
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
