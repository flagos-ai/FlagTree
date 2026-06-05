/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "TritonMETAXGPUTransforms/MACACommon.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace ttg = triton::gpu;

namespace maca {
namespace debug {

DEFINE_MACA_ENV_API_NE(get_enable_debug_internal,
                       "TRITON_ENABLE_DEBUG_INTERNAL");

/*
[getDebugOptionString]
Get the string for DebugOption enum value
*/
std::string getDebugOptionString(DebugOption option) {
  switch (option) {
  case DebugOption::kCheckCollectValidOp:
    return "checkCollectValidOp";
  default:
    llvm_unreachable("unsupported getDebugOption");
  }
}

void setParentFunctionOpAttrDebug(Operation *op, DebugOption option,
                                  DebugResult result) {
  if (!maca::debug::get_enable_debug_internal()) {
    return;
  }
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  if (funcOp) {
    int value = static_cast<int>(result);
    auto attr_name = getDebugOptionString(option);
    MLIRContext *ctx = funcOp.getContext();
    funcOp->setAttr(attr_name,
                    IntegerAttr::get(IntegerType::get(ctx, 32), value));
  }
}

} // namespace debug
} // namespace maca

bool updateLayout(llvm::SmallVector<unsigned, 3> &elemsPerThread,
                  llvm::SmallVector<unsigned, 2> &warpsPerTile,
                  llvm::SmallVector<int, 4> tile, int &version, int numWarps,
                  bool enableTf32, mlir::Type dtype, ArrayRef<unsigned> aorder,
                  ArrayRef<unsigned> border, bool disablePrefetch,
                  bool chainDot, bool storeCoalesce, int computeCapability) {
  llvm::SmallVector<unsigned, 2> orderA = {0, 0};
  llvm::SmallVector<unsigned, 2> orderB = {0, 0};
  llvm::copy(aorder, orderA.begin());
  llvm::copy(border, orderB.begin());
  auto orders = std::make_pair(orderA, orderB);
  auto layout = layouttable.at(orders);
  bool isOpt = false;
  // TODO: remove check chainDot if we support prefetch multi dot
  if (std::getenv("TRITON_DISABLE_MACA_OPT_MMA_PREFETCH") == nullptr &&
      disablePrefetch == false && !chainDot) {
    // k dim
    tile[2] /= 2;
  }

  auto pattern = std::make_pair(tile, layout);
  if (dtype.isF32() && !enableTf32) {
    isOpt = matchTable(pattern, fp32table, numWarps, enableTf32, dtype, aorder,
                       border, elemsPerThread, warpsPerTile, storeCoalesce);
  } else if (dtype.isF32() && enableTf32) {
    isOpt = matchTable(pattern, tf32table, numWarps, enableTf32, dtype, aorder,
                       border, elemsPerThread, warpsPerTile, storeCoalesce);
    auto tk = elemsPerThread[2];
    assert((tk >= 2 && tk % 2 == 0) &&
           "tk is not meet conditon when tf32 type");
  } else if (dtype.isF16() || dtype.isBF16()) {
    if (chainDot) {
      isOpt = false;
    } else {
      if (computeCapability >= 86) {
        isOpt = matchTable(pattern, fp16table_86, numWarps, enableTf32, dtype,
                           aorder, border, elemsPerThread, warpsPerTile,
                           storeCoalesce);
      } else {
        isOpt =
            matchTable(pattern, fp16table, numWarps, enableTf32, dtype, aorder,
                       border, elemsPerThread, warpsPerTile, storeCoalesce);
      }
    }
    auto tk = elemsPerThread[2];
    assert((tk >= 4 && tk % 4 == 0) &&
           "tk is not meet conditon when half type");
  } else if (dtype.isInteger(8)) {
    if (computeCapability >= 86) {
      isOpt =
          matchTable(pattern, i8table_86, numWarps, enableTf32, dtype, aorder,
                     border, elemsPerThread, warpsPerTile, storeCoalesce);
      auto tk = elemsPerThread[2];
      assert((tk >= 8 && tk % 8 == 0) &&
             "tk is not meet conditon when int8 type");
    } else {
      isOpt = matchTable(pattern, i8table, numWarps, enableTf32, dtype, aorder,
                         border, elemsPerThread, warpsPerTile, storeCoalesce);
      auto tk = elemsPerThread[2];
      assert((tk >= 4 && tk % 4 == 0) &&
             "tk is not meet conditon when int8 type");
    }
  } else if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dtype)) {
    assert(computeCapability >= 86);
    isOpt = matchTable(pattern, i8table_86, numWarps, enableTf32, dtype, aorder,
                       border, elemsPerThread, warpsPerTile, storeCoalesce);
    auto tk = elemsPerThread[2];
    assert((tk >= 8 && tk % 8 == 0) && "tk is not meet conditon when f8 type");
  }
  return isOpt;
}

bool matchTable(const std::pair<llvm::SmallVector<int, 4>, Layout> &pattern,
                const TileTable &table, const int &numWarps,
                const bool enableTf32, const mlir::Type dtype,
                const ArrayRef<unsigned> aorder,
                const ArrayRef<unsigned> border,
                llvm::SmallVector<unsigned, 3> &elemsPerThread,
                llvm::SmallVector<unsigned, 2> &warpsPerTile,
                bool storeCoalesce) {
  if (table.count(pattern)) { // matche table
    auto newLayout = table.at(pattern)[0];
    if (storeCoalesce && table.at(pattern).size() > 1)
      newLayout = table.at(pattern)[1];
    auto newWarps = std::get<1>(newLayout);
    if (numWarps == newWarps[0] * newWarps[1]) {
      elemsPerThread = std::get<0>(newLayout);
      warpsPerTile = newWarps;
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

llvm::SmallVector<unsigned, 2>
matchMNStage(const std::pair<llvm::SmallVector<int, 4>, Layout> &pattern,
             mlir::Type dtype, bool enableTf32) {
  if (dtype.isF32() && !enableTf32) {
    if (fp32StageMNTable.count(pattern)) {
      return fp32StageMNTable.at(pattern);
    }
  } else if (dtype.isF32() && enableTf32) {
    if (tf32StageMNTable.count(pattern)) {
      return tf32StageMNTable.at(pattern);
    }
  } else if (dtype.isF16() || dtype.isBF16()) {
    if (fp16StageMNTable.count(pattern)) {
      return fp16StageMNTable.at(pattern);
    }
  } else if (dtype.isInteger(8)) {
    if (i8StageMNTable.count(pattern)) {
      return i8StageMNTable.at(pattern);
    }
  }
  // If the match fails, please add stage-m and stage-n to the StageMNTable.
  return {0, 0};
}

llvm::SmallVector<unsigned, 3>
getDefaultElemsPerThread(Type type, bool enableTf32, int computeCapability) {
  if (type.isF32() && enableTf32) {
    return {1, 1, 2};
  } else if (type.isF32() || type.isF64()) {
    return {1, 1, 1};
  } else if (type.isF16() || type.isBF16()) {
    return {1, 1, 4};
  } else if (type.isSignlessInteger(8)) {
    if (computeCapability >= 86) {
      return {1, 1, 8};
    }
    return {1, 1, 4};
  } else if (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(type)) {
    assert(computeCapability >= 86);
    return {1, 1, 8};
  } else {
    assert(false && "Invalid dot operand type!");
    return {1, 1, 1};
  }
}

int getGVMNumberPerOp(BlockedEncodingAttr &enc, RankedTensorType &tensor) {
  auto shape = tensor.getShape();
  auto dims = shape.size();
  auto tileShape = ttg::getShapePerCTATile(tensor);
  int bytes = tensor.getElementType().getIntOrFloatBitWidth();
  auto instrs = ceil(bytes, 128);
  int gvmNumber = 1;
  for (int i = 0; i < dims; i++) {
    gvmNumber *= (shape[i] / tileShape[i]);
  }
  gvmNumber *= instrs;

  return gvmNumber;
}

unsigned getGVMNumberPerOp(triton::LoadOp &load) {
  // copy from LoadStoreOpToLLVM getVectorSize() and
  // getThreadConstRepeatTimes()
  auto tensorTy = dyn_cast<RankedTensorType>(load.getPtr().getType());
  if (!tensorTy)
    return 1;
  ModuleOp moduleOp = load->getParentOfType<ModuleOp>();
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  unsigned inVec = axisInfoAnalysis.getContiguity(load.getPtr());
  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  inVec = std::min<unsigned>(128 / pointeeBitWidth, inVec);
  if (load.getMask()) {
    unsigned maskVec = axisInfoAnalysis.getMaskAlignment(load.getMask());
    inVec = std::min<unsigned>(maskVec, inVec);
  }

  if (!getenv("TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT")) {
    auto order = triton::gpu::getOrder(tensorTy);
    long int sizePerThread = 0;
    if (auto blockEncoding = dyn_cast<triton::gpu::BlockedEncodingAttr>(
            tensorTy.getEncoding())) {
      sizePerThread = blockEncoding.getSizePerThread()[order[0]];
    } else {
      auto srcEncodingLinear = triton::gpu::toLinearEncoding(tensorTy);
      sizePerThread = srcEncodingLinear.getSizePerThread()[order[0]];
    }
    auto *axisInfo = axisInfoAnalysis.getAxisInfo(load.getPtr());
    auto constancy = axisInfo->getConstancy(order[0]);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(tensorTy)[order[0]];

    unsigned constRepeatPerThread = constancy;
    if (sizePerThread == 1 && (constancy > shapePerCTATile) &&
        (constancy % shapePerCTATile == 0)) {
      constRepeatPerThread = constancy / shapePerCTATile;
    } else if ((constancy > sizePerThread) &&
               (constancy % sizePerThread == 0)) {
      constRepeatPerThread = sizePerThread;
    } else if (sizePerThread % constancy) {
      constRepeatPerThread = 1;
    }

    if (load.getMask()) {
      unsigned maskVec = axisInfoAnalysis.getMaskAlignment(load.getMask());
      constRepeatPerThread = std::min<unsigned>(constRepeatPerThread, maskVec);
    }
    inVec = inVec * constRepeatPerThread;
  }

  unsigned numElems = triton::gpu::getTotalElemsPerThread(tensorTy);
  return std::max<unsigned>(1, numElems / inVec);
}

SmallVector<unsigned, 2> getOrder(triton::gpu::ConvertLayoutOp &op) {
  Value src = op.getSrc();
  Operation *trans = src.getDefiningOp();
  auto transOp = dyn_cast_or_null<triton::TransOp>(trans);
  if (transOp) {
    Value transSrc = transOp.getSrc();
    auto transSrcTy = cast<RankedTensorType>(transSrc.getType());

    if (auto transEnc =
            dyn_cast<SwizzledSharedEncodingAttr>(transSrcTy.getEncoding())) {
      auto transOrder = transEnc.getOrder();
      Operation *cvtPre = transSrc.getDefiningOp();
      auto cvtPreOp = dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre);
      if (cvtPreOp) {
        auto cvtPreSrc = cvtPreOp.getSrc();
        auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
        auto encPreSrcTy =
            dyn_cast<BlockedEncodingAttr>(cvtPreSrcTy.getEncoding());
        if (encPreSrcTy) {
          auto encPreSrcOrder = encPreSrcTy.getOrder();
          SmallVector<unsigned, 2> order = {encPreSrcOrder[1],
                                            encPreSrcOrder[0]};
          return order;
        } else {
          return {};
        }
      } else {
        return {};
      }
    } else if (auto transEnc =
                   dyn_cast<BlockedEncodingAttr>(transSrcTy.getEncoding())) {
      auto transOrder = transEnc.getOrder();
      SmallVector<unsigned, 2> order = {transOrder[1], transOrder[0]};
      return order;
    } else {
      return {};
    }

  } else {
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto enc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    if (enc) {
      auto enc_order = enc.getOrder();
      SmallVector<unsigned, 2> order = {enc_order[0], enc_order[1]};
      return order;
    } else {
      return {};
    }
  }
}

SmallVector<unsigned, 2> getOrder(Value &src) {
  Operation *trans = src.getDefiningOp();
  auto transOp = dyn_cast<triton::TransOp>(trans);
  if (transOp) {
    Value transSrc = transOp.getSrc();
    if (auto transSrcTy = dyn_cast<RankedTensorType>(transSrc.getType())) {
      if (auto transEnc =
              dyn_cast<SwizzledSharedEncodingAttr>(transSrcTy.getEncoding())) {
        Operation *cvtPre = transSrc.getDefiningOp();
        if (auto cvtPreOp = dyn_cast<triton::gpu::LocalAllocOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          auto encPreSrcTy =
              dyn_cast<BlockedEncodingAttr>(cvtPreSrcTy.getEncoding());
          if (encPreSrcTy) {
            auto encPreSrcOrder = encPreSrcTy.getOrder();
            SmallVector<unsigned, 2> order = {encPreSrcOrder[1],
                                              encPreSrcOrder[0]};
            return order;
          } else {
            return {};
          }
        } else if (auto cvtPreOp =
                       dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          auto encPreSrcTy =
              dyn_cast<BlockedEncodingAttr>(cvtPreSrcTy.getEncoding());
          if (encPreSrcTy) {
            auto encPreSrcOrder = encPreSrcTy.getOrder();
            SmallVector<unsigned, 2> order = {encPreSrcOrder[1],
                                              encPreSrcOrder[0]};
            return order;
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else if (auto transEnc = dyn_cast<ttg::LinearEncodingAttr>(
                     transSrcTy.getEncoding())) {
        Operation *cvtPre = transSrc.getDefiningOp();
        if (auto cvtPreOp = dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          auto encPreSrcTy =
              dyn_cast<BlockedEncodingAttr>(cvtPreSrcTy.getEncoding());
          if (encPreSrcTy) {
            auto encPreSrcOrder = encPreSrcTy.getOrder();
            SmallVector<unsigned, 2> order = {encPreSrcOrder[1],
                                              encPreSrcOrder[0]};
            return order;
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else if (auto transEnc =
                     dyn_cast<BlockedEncodingAttr>(transSrcTy.getEncoding())) {
        auto transOrder = transEnc.getOrder();
        SmallVector<unsigned, 2> order = {transOrder[1], transOrder[0]};
        return order;
      } else {
        return {};
      }
    } else if (auto transSrcTy = dyn_cast<mlir::triton::gpu::MemDescType>(
                   transSrc.getType())) {
      if (auto transEnc =
              dyn_cast<SwizzledSharedEncodingAttr>(transSrcTy.getEncoding())) {
        Operation *cvtPre = transSrc.getDefiningOp();
        if (auto cvtPreOp = dyn_cast<triton::gpu::LocalAllocOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          auto encPreSrcTy =
              dyn_cast<BlockedEncodingAttr>(cvtPreSrcTy.getEncoding());
          if (encPreSrcTy) {
            auto encPreSrcOrder = encPreSrcTy.getOrder();
            SmallVector<unsigned, 2> order = {encPreSrcOrder[1],
                                              encPreSrcOrder[0]};
            return order;
          } else {
            return {};
          }
        } else if (auto cvtPreOp =
                       dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          auto encPreSrcTy =
              dyn_cast<BlockedEncodingAttr>(cvtPreSrcTy.getEncoding());
          if (encPreSrcTy) {
            auto encPreSrcOrder = encPreSrcTy.getOrder();
            SmallVector<unsigned, 2> order = {encPreSrcOrder[1],
                                              encPreSrcOrder[0]};
            return order;
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else if (auto transEnc =
                     dyn_cast<BlockedEncodingAttr>(transSrcTy.getEncoding())) {
        auto transOrder = transEnc.getOrder();
        SmallVector<unsigned, 2> order = {transOrder[1], transOrder[0]};
        return order;
      } else {
        return {};
      }
    }
  } else {
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto enc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    if (enc) {
      auto enc_order = enc.getOrder();
      SmallVector<unsigned, 2> order = {enc_order[0], enc_order[1]};
      return order;
    } else {
      return {};
    }
  }
  return {};
}

SmallVector<int64_t, 2> getShape(Value &src) {
  Operation *trans = src.getDefiningOp();
  auto transOp = dyn_cast<triton::TransOp>(trans);
  if (transOp) {
    Value transSrc = transOp.getSrc();
    if (auto transSrcTy = dyn_cast<RankedTensorType>(transSrc.getType())) {
      if (auto transEnc = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
              transSrcTy.getEncoding())) {
        Operation *cvtPre = transSrc.getDefiningOp();
        if (auto cvtPreOp = dyn_cast<triton::gpu::LocalAllocOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          if (cvtPreSrcTy) {
            auto encPreSrcShape = cvtPreSrcTy.getShape();
            SmallVector<int64_t, 2> shape = {encPreSrcShape[1],
                                             encPreSrcShape[0]};
            return shape;
          } else {
            return {};
          }
        } else if (auto cvtPreOp =
                       dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          if (cvtPreSrcTy) {
            auto encPreSrcShape = cvtPreSrcTy.getShape();
            SmallVector<int64_t, 2> shape = {encPreSrcShape[1],
                                             encPreSrcShape[0]};
            return shape;
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else if (auto transEnc = dyn_cast<ttg::LinearEncodingAttr>(
                     transSrcTy.getEncoding())) {
        Operation *cvtPre = transSrc.getDefiningOp();
        if (auto cvtPreOp = dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          if (cvtPreSrcTy) {
            auto encPreSrcShape = cvtPreSrcTy.getShape();
            SmallVector<int64_t, 2> shape = {encPreSrcShape[1],
                                             encPreSrcShape[0]};
            return shape;
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else if (auto transEnc =
                     dyn_cast<BlockedEncodingAttr>(transSrcTy.getEncoding())) {
        auto transSrcTyShape = transSrcTy.getShape();
        SmallVector<int64_t, 2> shape = {transSrcTyShape[1],
                                         transSrcTyShape[0]};
        return shape;
      } else {
        return {};
      }
    } else if (auto transSrcTy =
                   dyn_cast<ttg::MemDescType>(transSrc.getType())) {
      if (auto transEnc =
              dyn_cast<SwizzledSharedEncodingAttr>(transSrcTy.getEncoding())) {
        Operation *cvtPre = transSrc.getDefiningOp();
        if (auto cvtPreOp = dyn_cast<triton::gpu::LocalAllocOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          if (cvtPreSrcTy) {
            auto encPreSrcShape = cvtPreSrcTy.getShape();
            SmallVector<int64_t, 2> shape = {encPreSrcShape[1],
                                             encPreSrcShape[0]};
            return shape;
          } else {
            return {};
          }
        } else if (auto cvtPreOp =
                       dyn_cast<triton::gpu::ConvertLayoutOp>(cvtPre)) {
          auto cvtPreSrc = cvtPreOp.getSrc();
          auto cvtPreSrcTy = cast<RankedTensorType>(cvtPreSrc.getType());
          if (cvtPreSrcTy) {
            auto encPreSrcShape = cvtPreSrcTy.getShape();
            SmallVector<int64_t, 2> shape = {encPreSrcShape[1],
                                             encPreSrcShape[0]};
            return shape;
          } else {
            return {};
          }
        } else {
          return {};
        }
      } else if (auto transEnc =
                     dyn_cast<BlockedEncodingAttr>(transSrcTy.getEncoding())) {
        auto transSrcTyShape = transSrcTy.getShape();
        SmallVector<int64_t, 2> shape = {transSrcTyShape[1],
                                         transSrcTyShape[0]};
        return shape;
      } else {
        return {};
      }
    }
  } else {
    auto srcTy = cast<RankedTensorType>(src.getType());
    if (srcTy) {
      auto enc_shape = srcTy.getShape();
      SmallVector<int64_t, 2> shape = {enc_shape[0], enc_shape[1]};
      return shape;
    } else {
      return {};
    }
  }
  return {};
}

mlir::Type getType(const Value &src) {
  Operation *trans = src.getDefiningOp();
  auto transOp = dyn_cast<triton::TransOp>(trans);
  if (transOp) {
    Value transSrc = transOp.getSrc();
    if (auto transSrcTy = dyn_cast<RankedTensorType>(transSrc.getType())) {
      auto type = transSrcTy.getElementType();
      return type;
    } else if (auto transSrcTy =
                   dyn_cast<ttg::MemDescType>(transSrc.getType())) {
      auto type = transSrcTy.getElementType();
      return type;
    }
  } else {
    auto srcTy = cast<RankedTensorType>(src.getType());
    if (srcTy) {
      auto type = srcTy.getElementType();
      return type;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

bool isNeedMerge(SwizzledSharedEncodingAttr srcSharedEnc,
                 SwizzledSharedEncodingAttr dstSharedEnc,
                 ArrayRef<int64_t> shape) {
  auto srcOrder = srcSharedEnc.getOrder();
  int srcVec = srcSharedEnc.getVec();
  int srcPerPhase = srcSharedEnc.getPerPhase();
  int srcMaxPhase = srcSharedEnc.getMaxPhase();

  auto dstOrder = dstSharedEnc.getOrder();
  int dstVec = dstSharedEnc.getVec();
  int dstPerPhase = dstSharedEnc.getPerPhase();
  int dstMaxPhase = dstSharedEnc.getMaxPhase();

  if (srcVec == dstVec && srcPerPhase == dstPerPhase &&
      srcMaxPhase == dstMaxPhase && srcOrder == dstOrder)
    return false;

  // not totaly same but equal, then can be merged
  // The swizzled range should be same, which means the number of rows and cols
  // are same srcPerPhase*srcMaxPhase == dstPerPhase*dstMaxPhase &&
  // srcMaxPhase*srcVec == dstMaxPhase*dstVec
  // ==> srcPerPhase/dstPerPhase == srcVec/dstVec
  return (srcOrder == dstOrder) &&
         (srcPerPhase * srcMaxPhase == dstPerPhase * dstMaxPhase) &&
         (srcPerPhase >= dstPerPhase) && (srcPerPhase % dstPerPhase == 0) &&
         (srcVec >= dstVec) && (srcVec % dstVec == 0) &&
         (srcMaxPhase * srcVec == dstMaxPhase * dstVec) &&
         (shape[dstOrder[0]] % (dstMaxPhase * dstVec) == 0) &&
         (shape[srcOrder[0]] % (srcMaxPhase * srcVec) == 0);
}

std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
extractSubCtaAndAllElem(unsigned subTensorSize, unsigned shapePerCTA,
                        unsigned totalSizePerThread, unsigned subTensorIdx) {
  SmallVector<unsigned> ctaIdx, elemIdx;
  int subNumCtas = subTensorSize / shapePerCTA;
  for (int i = 0; i < subNumCtas; i++) {
    ctaIdx.push_back(subTensorIdx * subNumCtas + i);
  }
  for (int j = 0; j < totalSizePerThread; j++) {
    elemIdx.push_back(j);
  }
  return std::make_pair(ctaIdx, elemIdx);
}

std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
extractAllCtaAndSubElem(unsigned numCtas, unsigned subSizePerThread,
                        unsigned subTensorIdx) {
  SmallVector<unsigned> ctaIdx, elemIdx;
  for (int i = 0; i < numCtas; i++) {
    ctaIdx.push_back(i);
  }
  for (int j = 0; j < subSizePerThread; j++) {
    elemIdx.push_back(subTensorIdx * subSizePerThread + j);
  }
  return std::make_pair(ctaIdx, elemIdx);
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
calExtractTensorIdx(RankedTensorType tensorType, RankedTensorType subTensorType,
                    ArrayRef<int64_t> subTensorIdx) {
  // subTensorIdx as ctaTile or elemTile idx
  auto totalEnc = tensorType.getEncoding();
  auto subEnc = subTensorType.getEncoding();
  auto totalTensorSize = tensorType.getShape();
  auto subTensorSize = subTensorType.getShape();
  SmallVector<int64_t> linearCtaIdx, linearElemIdx;
  SmallVector<unsigned> subSizePerThread, totalSizePerThread, shapePerCTA,
      numCTAs, order;
  // ArrayRef<unsigned> subSizePerThread, totalSizePerThread;
  int rank;
  if (isa<triton::gpu::MACAMmaEncodingAttr>(totalEnc)) {
    auto totalMMAEnc = dyn_cast<triton::gpu::MACAMmaEncodingAttr>(totalEnc);
    auto subMMAEnc = dyn_cast<triton::gpu::MACAMmaEncodingAttr>(subEnc);
    subSizePerThread = {subMMAEnc.getElementsMNK()[0],
                        subMMAEnc.getElementsMNK()[1]};
    totalSizePerThread = {totalMMAEnc.getElementsMNK()[0],
                          totalMMAEnc.getElementsMNK()[1]};
    shapePerCTA = ttg::getShapePerCTATile(tensorType);
    for (int i = 0; i < 2; i++)
      numCTAs.push_back(totalTensorSize[i] / shapePerCTA[i]);
    if (totalMMAEnc.getColMajor()) {
      order = {0, 1};
    } else {
      order = {1, 0};
    }
    rank = 2;
  } else if (isa<triton::gpu::BlockedEncodingAttr>(totalEnc)) {
    auto totalBlockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(totalEnc);
    auto subBlockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(subEnc);
    subSizePerThread = llvm::to_vector(subBlockEnc.getSizePerThread());
    totalSizePerThread = llvm::to_vector(totalBlockEnc.getSizePerThread());
    shapePerCTA = ttg::getShapePerCTATile(tensorType);
    for (int i = 0; i < 2; i++)
      numCTAs.push_back(totalTensorSize[i] / shapePerCTA[i]);
    order = mlir::triton::gpu::getOrder(tensorType);
    rank = order.size();
  } else if (isa<triton::gpu::DotOperandEncodingAttr>(totalEnc)) {
    // TODO: CHECK DOTOP logic here
    auto totalDotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(totalEnc);
    auto subDotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(subEnc);
    auto totalDotLinearEnc = dyn_cast<ttg::LinearEncodingAttr>(totalEnc);
    auto subDotLinearEnc = dyn_cast<ttg::LinearEncodingAttr>(subEnc);

    if (totalDotLinearEnc && subDotLinearEnc) {
      subSizePerThread = subDotLinearEnc.getSizePerThread();
      totalSizePerThread = totalDotLinearEnc.getSizePerThread();
    } else if (totalDotEnc && subDotEnc) {
      auto subSrcEncodingLinear = triton::gpu::toLinearEncoding(subTensorType);
      auto srcEncodingLinear = triton::gpu::toLinearEncoding(tensorType);
      subSizePerThread = subSrcEncodingLinear.getSizePerThread();
      totalSizePerThread = srcEncodingLinear.getSizePerThread();
    } else {
      assert(false && "Tensor Enc only support LinearLayout or DotEncoding");
    }
    shapePerCTA = ttg::getShapePerCTATile(tensorType);
    for (int i = 0; i < 2; i++)
      numCTAs.push_back(totalTensorSize[i] / shapePerCTA[i]);
    auto opIdx = totalDotEnc.getOpIdx();
    if (opIdx) { // B
      order = {0, 1};
    } else { // A
      order = {1, 0};
    }
    rank = 2;
  } else {
    assert(false &&
           "Extract/Insert Tensor Only Support Blocked/MMA/Dot Layout");
  }
  assert(rank == 2 && "Extract/Insert Tensor Only Support rank=2 for now");

  SmallVector<SmallVector<unsigned>, 2> oneDimCtaIdx, oneDimElemIdx;
  for (int i = 0; i < rank; i++) {
    std::pair<SmallVector<unsigned>, SmallVector<unsigned>> retIdx;
    if (totalSizePerThread[i] == subSizePerThread[i]) {
      // extract sub CTAs (which are contiguous) and all elems in one cta
      retIdx = extractSubCtaAndAllElem(subTensorSize[i], shapePerCTA[i],
                                       totalSizePerThread[i], subTensorIdx[i]);
    } else if (totalSizePerThread[i] != subSizePerThread[i]) {
      // extract all CTAs and sub elems (which are contiguous) in one cta
      retIdx = extractAllCtaAndSubElem(numCTAs[i], subSizePerThread[i],
                                       subTensorIdx[i]);
    }
    oneDimCtaIdx.insert(oneDimCtaIdx.begin() + i, retIdx.first);
    oneDimElemIdx.insert(oneDimElemIdx.begin() + i, retIdx.second);
  }

  for (int i = 0; i < oneDimCtaIdx[0].size(); i++) {
    for (int j = 0; j < oneDimCtaIdx[1].size(); j++) {
      linearCtaIdx.push_back(maca::getSubLinearIndex<unsigned>(
          SmallVector<unsigned>{oneDimCtaIdx[0][i], oneDimCtaIdx[1][j]},
          numCTAs, order));
    }
  }

  for (int i = 0; i < oneDimElemIdx[0].size(); i++) {
    for (int j = 0; j < oneDimElemIdx[1].size(); j++) {
      linearElemIdx.push_back(maca::getSubLinearIndex<unsigned>(
          SmallVector<unsigned>{oneDimElemIdx[0][i], oneDimElemIdx[1][j]},
          totalSizePerThread, order));
    }
  }

  std::sort(linearCtaIdx.begin(), linearCtaIdx.end());
  std::sort(linearElemIdx.begin(), linearElemIdx.end());

  return std::make_pair(linearCtaIdx, linearElemIdx);
}

void genSwiMask(Value dep, unsigned inVec,
                triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                RankedTensorType srcTy, triton::gpu::MemDescType resTy,
                OpBuilder &builder, ArrayRef<int64_t> shape,
                ArrayRef<int64_t> tileIdx, int numStages,
                ArrayRef<int64_t> elemTileNum, IRMapping &mapping, Location loc,
                Value cstValue, int allStage) {
  auto depOp = dep.getDefiningOp();
  auto inOrder = triton::gpu::getOrder(srcTy);
  auto resEncoding = cast<ttg::SwizzledSharedEncodingAttr>(resTy.getEncoding());
  auto outVec = resEncoding.getVec();
  auto perPhase = resEncoding.getPerPhase();
  auto maxPhase = resEncoding.getMaxPhase();
  if (auto cmpIOp = dyn_cast<arith::CmpIOp>(depOp)) {
    auto lhs = cmpIOp.getLhs();
    auto rhs = cmpIOp.getRhs();
    auto lhsAxisInfo = axisInfoAnalysis.getAxisInfo(lhs);
    auto rhsAxisInfo = axisInfoAnalysis.getAxisInfo(rhs);
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    if (!lhsTy)
      assert(false);
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!rhsTy)
      assert(false);
    auto srcShape = lhsTy.getShape();

    if (lhs.getDefiningOp<triton::gpu::SwizzleTensorOp>() ||
        rhs.getDefiningOp<triton::gpu::SwizzleTensorOp>())
      return;

    if (lhsTy.getRank() == 1) {
      if ((rhsAxisInfo->getContiguity(0) < srcShape[0]) &&
          (rhsAxisInfo->getConstancy(0) < srcShape[0])) {
        assert(false);
      }
      // won't make swizzle if swizzle mask already.
      if (isa<BlockArgument>(lhs)) {
        if (rhsAxisInfo->getDivisibility(0) < inVec) {
          assert(false);
        }
      } else {
        if ((lhsAxisInfo->getContiguity(0) < srcShape[0]) &&
            (lhsAxisInfo->getConstancy(0) < srcShape[0])) {
          assert(false);
        }
        // rhs/lhs's divisibility must be larger than inVec;
        if (lhsAxisInfo->getDivisibility(0) < inVec ||
            rhsAxisInfo->getDivisibility(0) < inVec) {
          assert(false);
        }
      }

      // makerange -> cmpi -> expanddims -> broadcast -> andi
      // makerange -> expandims -> brodacast -> (swi) -> cmpi -> andi
      Operation *op = *cmpIOp.getResult().getUsers().begin();
      auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op);
      if (!expandDimsOp)
        assert(false);
      auto expandTy =
          dyn_cast<RankedTensorType>(expandDimsOp.getResult().getType());
      if (!expandTy)
        assert(false);
      auto expandType = RankedTensorType::get(
          expandTy.getShape(), lhsTy.getElementType(), expandTy.getEncoding());
      auto lhsExpand = builder.create<triton::ExpandDimsOp>(
          loc, expandType, lhs, expandDimsOp.getAxis());
      auto rhsExpand = builder.create<triton::ExpandDimsOp>(
          loc, expandType, rhs, expandDimsOp.getAxis());
      auto subType = RankedTensorType::get(shape, lhsTy.getElementType(),
                                           expandTy.getEncoding());
      auto newSrc = dyn_cast<RankedTensorType>(lhsExpand.getType());
      if (!newSrc)
        assert(false);
      auto newSrcShape = newSrc.getShape();
      Value newLhs, newRhs, newCmpI;
      if (newSrcShape[inOrder[1]] == 1) {
        newLhs = builder.create<triton::BroadcastOp>(loc, subType, lhsExpand);
        newRhs = builder.create<triton::BroadcastOp>(loc, subType, rhsExpand);
      } else if (newSrcShape[inOrder[1]] == numStages * shape[inOrder[1]]) {
        newLhs = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, lhsExpand, tileIdx, shape);
        newRhs = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, rhsExpand, tileIdx, shape);
      } else if (!newSrcShape.equals(shape)) {
        assert(false);
      }

      if (lhsAxisInfo->getContiguity(0) >= newSrcShape[inOrder[0]]) {
        newLhs = builder.create<triton::gpu::SwizzleTensorOp>(
            loc, subType, newLhs, inVec, outVec, perPhase, maxPhase);
      }
      if (rhsAxisInfo->getContiguity(0) >= newSrcShape[inOrder[0]]) {
        newRhs = builder.create<triton::gpu::SwizzleTensorOp>(
            loc, subType, newRhs, inVec, outVec, perPhase, maxPhase);
      }

      newCmpI = builder.create<arith::CmpIOp>(loc, cmpIOp.getPredicate(),
                                              newLhs, newRhs);
      mapping.map(cmpIOp.getResult(), newCmpI);
    } else {
      if (srcShape[inOrder[0]] != shape[inOrder[0]])
        assert(false);
      // won't make swizzle if swizzle mask already.
      if ((lhsAxisInfo->getContiguity(inOrder[0]) < srcShape[inOrder[0]]) &&
          (lhsAxisInfo->getConstancy(inOrder[0]) < srcShape[inOrder[0]])) {
        assert(false);
      }
      if ((rhsAxisInfo->getContiguity(inOrder[0]) < srcShape[inOrder[0]]) &&
          (rhsAxisInfo->getConstancy(inOrder[0]) < srcShape[inOrder[0]])) {
        assert(false);
      }
      // rhs/lhs's divisibility must be larger than inVec;
      if (lhsAxisInfo->getDivisibility(inOrder[0]) < inVec ||
          rhsAxisInfo->getDivisibility(inOrder[0]) < inVec) {
        assert(false);
      }

      auto subType = RankedTensorType::get(shape, lhsTy.getElementType(),
                                           lhsTy.getEncoding());
      if (shape.size() == 2 && product<int64_t>(elemTileNum) > 1 &&
          isa<triton::gpu::BlockedEncodingAttr>(lhsTy.getEncoding())) {
        auto newEnc =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(lhsTy.getEncoding());
        SmallVector<unsigned> sizePerThread(newEnc.getSizePerThread());
        for (int i = 0; i < sizePerThread.size(); i++) {
          // if sizePerThread[i] == 0, it will cause the
          // 'BlockedEncodingAttr::get()' check failed on triton3.6, but on
          // Triton3.0, this check does not exist.
          sizePerThread[i] =
              (sizePerThread[i] + elemTileNum[i] - 1) / elemTileNum[i];
        }
        newEnc = triton::gpu::BlockedEncodingAttr::get(
            lhsTy.getContext(), sizePerThread, newEnc.getThreadsPerWarp(),
            newEnc.getWarpsPerCTA(), newEnc.getOrder(), newEnc.getCTALayout());
        subType = RankedTensorType::get(shape, lhsTy.getElementType(), newEnc);
      }
      Value newLhs, newRhs, newCmpI;
      if (srcShape[inOrder[1]] == 1) {
        // Because the srcLayout of BroadcastOp must be the same as the
        // resLayout, So Add a ConvertLayoutOp to convert LhsType to subType
        // ConvertLayoutOp can be removed by following pass
        if (subType.getEncoding() != lhsTy.getEncoding()) {
          auto newLhsType = RankedTensorType::get(
              srcShape, lhsTy.getElementType(), subType.getEncoding());
          lhs = builder.create<triton::gpu::ConvertLayoutOp>(loc, newLhsType,
                                                             lhs);
        }
        if (subType.getEncoding() != rhsTy.getEncoding()) {
          auto newRhsType = RankedTensorType::get(
              rhsTy.getShape(), rhsTy.getElementType(), subType.getEncoding());
          rhs = builder.create<triton::gpu::ConvertLayoutOp>(loc, newRhsType,
                                                             rhs);
        }
        newLhs = builder.create<triton::BroadcastOp>(loc, subType, lhs);
        newRhs = builder.create<triton::BroadcastOp>(loc, subType, rhs);
      } else if (srcShape[inOrder[1]] == numStages * shape[inOrder[1]]) {
        auto ret = calExtractTensorIdx(lhsTy, subType, tileIdx);
        newLhs = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, lhs, ret.first, ret.second);
        newRhs = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, rhs, ret.first, ret.second);
      } else if (!srcShape.equals(shape)) {
        assert(false);
      }
      if (lhsAxisInfo->getContiguity(inOrder[0]) >= srcShape[inOrder[0]]) {
        newLhs = builder.create<triton::gpu::SwizzleTensorOp>(
            loc, subType, newLhs, inVec, outVec, perPhase, maxPhase);
      }
      if (rhsAxisInfo->getContiguity(inOrder[0]) >= srcShape[inOrder[0]]) {
        newRhs = builder.create<triton::gpu::SwizzleTensorOp>(
            loc, subType, newRhs, inVec, outVec, perPhase, maxPhase);
      }

      if (allStage >= numStages) {
        auto ret =
            calExtractTensorIdx(lhsTy, subType, SmallVector<int64_t>{0, 0});
        auto newLhsCst = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, cstValue, ret.first, ret.second);
        newLhs = builder.create<arith::AddIOp>(loc, newLhs, newLhsCst);
      }

      newCmpI = builder.create<arith::CmpIOp>(loc, cmpIOp.getPredicate(),
                                              newLhs, newRhs);
      mapping.map(cmpIOp.getResult(), newCmpI);
    }
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(depOp)) {
    // makerange -> cmpi -> expanddims -> broadcast -> andi
    // makerange -> expandims -> brodacast -> (swi) -> cmpi -> andi
    auto newRes = mapping.lookup(expandDimsOp.getSrc());
    if (!newRes)
      assert(false);
    mapping.map(expandDimsOp.getResult(), newRes);
  } else if (auto bdcast = dyn_cast<triton::BroadcastOp>(depOp)) {
    auto newSrc = mapping.lookup(bdcast.getSrc());
    if (!newSrc)
      assert(false);
    mapping.map(dep, newSrc);
  } else {
    auto depOperands = depOp->getOperands();
    for (int i = 0; i < depOperands.size(); ++i) {
      Value v = depOperands[i];
      Value newOperand = mapping.lookupOrDefault(v);
      auto operandTy = dyn_cast<RankedTensorType>(newOperand.getType());
      auto operandShape = operandTy.getShape();
      auto subType = RankedTensorType::get(shape, operandTy.getElementType(),
                                           operandTy.getEncoding());
      if (shape.size() == 2 && product<int64_t>(elemTileNum) > 1 &&
          isa<triton::gpu::BlockedEncodingAttr>(operandTy.getEncoding())) {
        auto newEnc =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(operandTy.getEncoding());
        SmallVector<unsigned> sizePerThread(newEnc.getSizePerThread());
        for (int i = 0; i < sizePerThread.size(); i++) {
          // if sizePerThread[i] == 0, it will cause the
          // 'BlockedEncodingAttr::get()' check failed on triton3.6, but on
          // Triton3.0, this check does not exist.
          sizePerThread[i] =
              (sizePerThread[i] + elemTileNum[i] - 1) / elemTileNum[i];
        }
        newEnc = triton::gpu::BlockedEncodingAttr::get(
            operandTy.getContext(), sizePerThread, newEnc.getThreadsPerWarp(),
            newEnc.getWarpsPerCTA(), newEnc.getOrder(), newEnc.getCTALayout());
        subType =
            RankedTensorType::get(shape, operandTy.getElementType(), newEnc);
      }
      if (operandShape[inOrder[0]] != shape[inOrder[0]])
        assert(false);
      if (operandShape[inOrder[1]] == 1) {
        if (subType.getEncoding() != operandTy.getEncoding()) {
          auto newOperandTy = RankedTensorType::get(
              operandShape, operandTy.getElementType(), subType.getEncoding());
          newOperand = builder.create<triton::gpu::ConvertLayoutOp>(
              loc, newOperandTy, newOperand);
        }
        newOperand =
            builder.create<triton::BroadcastOp>(loc, subType, newOperand);
        mapping.map(v, newOperand);
      } else if (operandShape[inOrder[1]] == numStages * shape[inOrder[1]]) {
        auto ret = calExtractTensorIdx(operandTy, subType, tileIdx);
        newOperand = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, newOperand, ret.first, ret.second);
        mapping.map(v, newOperand);
      } else if (operandShape[inOrder[1]] == 64 && numStages == 4 &&
                 (operandShape[inOrder[1]] ==
                  (numStages / 2) * shape[inOrder[1]])) {
        auto ret = calExtractTensorIdx(operandTy, subType, tileIdx);
        newOperand = builder.create<triton::gpu::ExtractTensorOp>(
            loc, subType, newOperand, ret.first, ret.second);
        mapping.map(v, newOperand);
      } else if (!operandShape.equals(shape)) {
        assert(false);
      }
    }
    Operation *cloneOp = cloneWithInferType(builder, depOp, mapping);
    mapping.map(dep, cloneOp->getResult(0));
  }
}

float checkMaskDepsRecursion(Value src, unsigned inVec,
                             triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                             SmallVector<Operation *> &deps,
                             triton::gpu::MemDescType resTy) {
  auto axisInfo = axisInfoAnalysis.getAxisInfo(src);
  auto op = src.getDefiningOp();
  if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
    auto srcShape = srcTy.getShape();
    auto srcTyRank = srcTy.getRank();

    SmallVector<unsigned, 2> inOrder = {0, 0};
    if (isa<triton::gpu::BlockedEncodingAttr>(srcTy.getEncoding())) {
      auto srcBlocked =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(srcTy.getEncoding());
      inOrder = getOrder(src);
    } else if (isa<triton::gpu::SliceEncodingAttr>(srcTy.getEncoding())) {
      auto slice =
          dyn_cast<triton::gpu::SliceEncodingAttr>(srcTy.getEncoding());
      auto sliceParent = slice.getParent();
      if (!isa<triton::gpu::BlockedEncodingAttr>(sliceParent)) {
        return 0;
      }
      auto srcBlocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(sliceParent);
      inOrder = getOrder(src);
    } else {
      return 0;
    }
    // check constancy
    if (srcTyRank == 1 && axisInfo->getConstancy(0) >= srcShape[0]) {
      return 1;
    } else if (srcTyRank != 1 &&
               axisInfo->getConstancy(inOrder[0]) >= srcShape[inOrder[0]]) {
      return 1;
    }

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
      auto lhs = cmpOp.getLhs();
      auto rhs = cmpOp.getRhs();
      auto lhsAxisInfo = axisInfoAnalysis.getAxisInfo(lhs);
      auto rhsAxisInfo = axisInfoAnalysis.getAxisInfo(rhs);
      // check if lhs or rhs is swizzled
      auto sharedEnc =
          cast<ttg::SwizzledSharedEncodingAttr>(resTy.getEncoding());
      if (auto lhsOp = lhs.getDefiningOp<triton::gpu::SwizzleTensorOp>()) {
        if (!(lhsOp.getOutVec() == sharedEnc.getVec() &&
              lhsOp.getPerPhase() == sharedEnc.getPerPhase() &&
              lhsOp.getMaxPhase() == sharedEnc.getMaxPhase() &&
              lhsOp.getInVec() == inVec)) {
          return 0;
        }
      } else if (auto rhsOp =
                     rhs.getDefiningOp<triton::gpu::SwizzleTensorOp>()) {
        if (!(rhsOp.getOutVec() == sharedEnc.getVec() &&
              rhsOp.getPerPhase() == sharedEnc.getPerPhase() &&
              rhsOp.getMaxPhase() == sharedEnc.getMaxPhase() &&
              rhsOp.getInVec() == inVec)) {
          return 0;
        }
      } else {
        if (srcTyRank == 1) {
          if ((rhsAxisInfo->getContiguity(0) < srcShape[0]) &&
              (rhsAxisInfo->getConstancy(0) < srcShape[0])) {
            return 0;
          }
          if (isa<BlockArgument>(lhs)) {
            if (rhsAxisInfo->getDivisibility(0) < inVec) {
              return 0;
            }
          } else {
            // lhs is not continguous and not constant
            if ((lhsAxisInfo->getContiguity(0) < srcShape[0]) &&
                (lhsAxisInfo->getConstancy(0) < srcShape[0])) {
              return 0;
            }
            // rhs/lhs's divisibility must be larger than inVec;
            if (lhsAxisInfo->getDivisibility(0) < inVec ||
                rhsAxisInfo->getDivisibility(0) < inVec) {
              return 0;
            }
          }
        } else {
          // lhs is not continguous and not constant
          if ((lhsAxisInfo->getContiguity(inOrder[0]) < srcShape[inOrder[0]]) &&
              (lhsAxisInfo->getConstancy(inOrder[0]) < srcShape[inOrder[0]])) {
            return 0;
          }
          // rhs is not continguous and not constant
          if ((rhsAxisInfo->getContiguity(inOrder[0]) < srcShape[inOrder[0]]) &&
              (rhsAxisInfo->getConstancy(inOrder[0]) < srcShape[inOrder[0]])) {
            return 0;
          }
          // rhs/lhs's divisibility must be larger than inVec;
          if (lhsAxisInfo->getDivisibility(inOrder[0]) < inVec ||
              rhsAxisInfo->getDivisibility(inOrder[0]) < inVec) {
            return 0;
          }
        }
      }
      deps.push_back(op);
      return 1.1;
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      // can not make swizzle if mask is originally loaded.
      return 0;
    }
  } else {
    return 0;
  }
  float return_val = 1;
  // LRD collect all deps
  for (Value v : op->getOperands()) {
    return_val = return_val * checkMaskDepsRecursion(v, inVec, axisInfoAnalysis,
                                                     deps, resTy);
  }
  if (return_val > 1)
    deps.push_back(op);
  return return_val;
}

Value genSwiSubMask(Value mask, triton::LoadOp &op, RankedTensorType srcTy,
                    triton::gpu::MemDescType resTy, OpBuilder &builder,
                    ArrayRef<int64_t> shape, ArrayRef<int64_t> tileIdx,
                    int numStages, ArrayRef<int64_t> elemTileNum,
                    Value cstValue, int allStage) {
  ModuleOp moduleOp = mask.getDefiningOp()->getParentOfType<ModuleOp>();
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  unsigned inVec = axisInfoAnalysis.getContiguity(op.getPtr());
  IRMapping maskDepsMapping;
  SmallVector<Operation *> orderedMaskDeps;
  bool validMaskSwizzle = checkMaskDepsRecursion(mask, inVec, axisInfoAnalysis,
                                                 orderedMaskDeps, resTy) > 1;
  assert(validMaskSwizzle);
  for (auto dep : orderedMaskDeps) {
    genSwiMask(dep->getResult(0), inVec, axisInfoAnalysis, srcTy, resTy,
               builder, shape, tileIdx, numStages, elemTileNum, maskDepsMapping,
               op.getLoc(), cstValue, allStage);
  }
  Value submask = maskDepsMapping.lookup(mask);
  if (!submask)
    assert(false);
  return submask;
}

bool checkUseCpAsync(triton::LoadOp &op, mlir::Type elemTy,
                     triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                     triton::gpu::SwizzledSharedEncodingAttr &resSharedLayout) {
  auto src = op.getPtr();
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto srcBlocked =
      dyn_cast<triton::gpu::BlockedEncodingAttr>(srcTy.getEncoding());
  auto inOrder = triton::gpu::getOrder(srcTy);
  unsigned inVec = axisInfoAnalysis.getContiguity(src);
  unsigned outVec = resSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  auto srcShape = srcTy.getShape();
  // assert here?
  assert(srcShape.size() == 2 && "insert_slice_async: Unexpected rank of %src");

  bool isContiguity = false;
  // Just need to judge the continuity of the swizzle vec.
  for (NamedAttribute attr : op->getAttrDictionary().getValue()) {
    if (attr.getName() == "tt.contiguity") {
      Attribute val = attr.getValue();
      auto denseValue = dyn_cast<DenseElementsAttr>(val);
      auto valsInt = denseValue.getValues<int>();

      if (inOrder.size() != 2) {
        isContiguity = false;
        break;
      }
      isContiguity = srcShape[inOrder[0]] <= valsInt[inOrder[0]];
      break;
    }
  }
  auto *axisInfo = axisInfoAnalysis.getAxisInfo(src);
  isContiguity = isContiguity ||
                 (srcShape[inOrder[0]] <= axisInfo->getContiguity(inOrder[0]));
  auto sizePerThread = srcBlocked.getSizePerThread();
  int numElemsPerThreadContigue = sizePerThread[inOrder[0]];

  auto resByteWidth = elemTy.getIntOrFloatBitWidth() / 8;
  int copyByteWidth = numElemsPerThreadContigue * resByteWidth;

  if (isContiguity && copyByteWidth <= 16 &&
      numElemsPerThreadContigue <= minVec) {
    return true;
  }
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  if (maxPhase == 1 && copyByteWidth <= 16) {
    return true;
  }
  return false;
}

LogicalResult
checkMaskDeps(scf::ForOp &forOp, SetVector<Operation *> &ops,
              DenseMap<Value, bool> &loadGenMask,
              DenseMap<Value, ttg::MemDescType> &loadsBufferType) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  for (Operation *op : ops) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto resSharedLayout = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
          loadsBufferType[loadOp].getEncoding());
      bool use_cpasync =
          checkUseCpAsync(loadOp, loadsBufferType[loadOp].getElementType(),
                          axisInfoAnalysis, resSharedLayout);
      bool gen_mask = false;
      if (auto mask = loadOp.getMask()) {
        SmallVector<Operation *> deps;
        unsigned inVec = axisInfoAnalysis.getContiguity(loadOp.getPtr());
        float mask_signal = checkMaskDepsRecursion(
            mask, inVec, axisInfoAnalysis, deps, loadsBufferType[loadOp]);
        use_cpasync = mask_signal >= 1 ? use_cpasync : false;
        gen_mask = mask_signal > 1 ? true : false;
        if (gen_mask) {
          for (auto dep : deps) {
            int uses_in_deps = 0;
            for (auto user : dep->getResult(0).getUsers()) {
              auto it = std::find(deps.begin(), deps.end(), user);
              if (it != deps.end())
                uses_in_deps += 1;
            }
            if (uses_in_deps > 1) {
              use_cpasync = false;
              break;
            }
          }
        }
      }
      if (use_cpasync) {
        loadGenMask[loadOp] = gen_mask;
      } else {
        return failure();
      }
    }
  }
  return success();
}

void checkMatchCpAsync(
    scf::ForOp &forOp, SetVector<Operation *> &ops,
    DenseMap<Value, bool> &loadGenMask,
    DenseMap<Value, triton::gpu::MemDescType> &loadsBufferType,
    DenseMap<Operation *, bool> &loadCanCpAsync,
    DenseMap<Operation *, bool> &loadUseRegPipeline) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  for (Operation *op : ops) {
    if (loadUseRegPipeline[op]) {
      loadCanCpAsync[op] = false;
      continue;
    }
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto resSharedLayout = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
          loadsBufferType[loadOp].getEncoding());
      bool use_cpasync =
          checkUseCpAsync(loadOp, loadsBufferType[loadOp].getElementType(),
                          axisInfoAnalysis, resSharedLayout);
      bool gen_mask = false;
      if (auto mask = loadOp.getMask()) {
        SmallVector<Operation *> deps;
        unsigned inVec = axisInfoAnalysis.getContiguity(loadOp.getPtr());
        float mask_signal = checkMaskDepsRecursion(
            mask, inVec, axisInfoAnalysis, deps, loadsBufferType[loadOp]);
        use_cpasync = mask_signal >= 1 ? use_cpasync : false;
        gen_mask = mask_signal > 1 ? true : false;
        if (gen_mask) {
          for (auto dep : deps) {
            int uses_in_deps = 0;
            for (auto user : dep->getResult(0).getUsers()) {
              auto it = std::find(deps.begin(), deps.end(), user);
              if (it != deps.end())
                uses_in_deps += 1;
            }
            if (uses_in_deps > 1) {
              use_cpasync = false;
              break;
            }
          }
        }
      }
      loadCanCpAsync[op] = use_cpasync;
    }
  }
}

void collectAddPtrDepsRecursion(Operation *op, scf::ForOp &forOp,
                                SetVector<Value> &validAddPtrs,
                                SmallVector<triton::AddPtrOp> &addptrs) {
  if (op->getNumOperands() == 0)
    return;
  auto v = op->getOperand(0);
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (v.getParentRegion() != forOp.getRegion() || arg.getArgNumber() == 0)
      return;
    auto operand = forOp.getInitArgs()[arg.getArgNumber() - 1];
    Operation *next = operand.getDefiningOp();
    collectAddPtrDepsRecursion(next, forOp, validAddPtrs, addptrs);
  } else if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
    if (validAddPtrs.contains(addPtrOp))
      return;
    addptrs.push_back(addPtrOp);
    Operation *next = addPtrOp.getPtr().getDefiningOp();
    collectAddPtrDepsRecursion(next, forOp, validAddPtrs, addptrs);
  } else {
    return;
  }
}

void collectAddPtrDeps(scf::ForOp &forOp, bool enableSaddrOpt,
                       SmallVector<Operation *> &orderedDeps,
                       SetVector<Value> &validAddPtrs,
                       DenseMap<triton::AddPtrOp, SmallVector<triton::AddPtrOp>>
                           &validPtrDepMapping) {
  if (!enableSaddrOpt)
    return;
  for (Operation *op : orderedDeps) {
    SmallVector<triton::AddPtrOp> addptrs;
    if (auto addptr = llvm::dyn_cast<triton::AddPtrOp>(op)) {
      collectAddPtrDepsRecursion(op, forOp, validAddPtrs, addptrs);
      if (addptrs.size() > 0)
        validPtrDepMapping[addptr] = addptrs;
    }
  }
}

Value addInt32OrInt64(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                      Operation *anchor) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto lhsBits = lhsType.getElementType().getIntOrFloatBitWidth();
  assert(lhsType);
  assert(lhsBits >= 32);
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  auto rhsBits = rhsType.getElementType().getIntOrFloatBitWidth();
  assert(rhsType);
  assert(rhsBits >= 32);
  if (lhsBits < rhsBits) {
    lhs = builder.create<arith::ExtUIOp>(loc, rhsType, lhs);
    if (anchor)
      lhs.getDefiningOp()->moveBefore(anchor);
  } else if (lhsBits > rhsBits) {
    rhs = builder.create<arith::ExtUIOp>(loc, lhsType, rhs);
    if (anchor)
      rhs.getDefiningOp()->moveBefore(anchor);
  }
  Value result = builder.create<arith::AddIOp>(loc, lhs, rhs);
  if (anchor)
    result.getDefiningOp()->moveBefore(anchor);
  return result;
}

int calWaitArriveCounts(int stage_m, int stage_n, int i, int numStagesInner) {
  if (std::max(stage_m, stage_n) != numStagesInner) {
    assert("At least stage_m and stage_n must be equal to numStagesInner");
  }

  int waitstage1 = (i - 1 + numStagesInner) % numStagesInner;
  int waitstage2 = (i + numStagesInner) % numStagesInner;
  int Awaitcount1 = stage_m > waitstage1 ? 1 : 0;
  int Awaitcount2 = stage_m > waitstage2 ? 1 : 0;
  int Bwaitcount1 = stage_n > waitstage1 ? 1 : 0;
  int Bwaitcount2 = stage_n > waitstage2 ? 1 : 0;
  int sumWaitCount = Awaitcount1 + Awaitcount2 + Bwaitcount1 + Bwaitcount2;
  return sumWaitCount;
}

bool getIfLdsTrans(const SmallVector<unsigned, 3> &elemsMNK, unsigned major,
                   unsigned minor, ArrayRef<unsigned> order, bool isA,
                   Type elemTy) {
  if (std::getenv("TRITON_ENABLE_LDS_TRANS") == nullptr)
    return false;
  if (order.empty() || elemsMNK.empty())
    return false;
  // elemsMNK[2] >= 4 is required from bfloat/float16 mma instruction.
  // elemsMNK[2] >= 8 is required from int8 mma instruction.
  if (((isA && order[0] == 0) || (!isA && order[0] == 1)) && major == 2 &&
      minor == 6 &&
      (((elemsMNK[2] >= 4) && (elemTy.isF16() || elemTy.isBF16())) ||
       ((elemsMNK[2] >= 8) && (elemTy.getIntOrFloatBitWidth() == 8)))) {
    return true;
  } else {
    return false;
  }
}

void collectValueDep(scf::ForOp &forOp, scf::YieldOp &yieldOp, Value v,
                     int stage, SetVector<Value> &deps) {
  // Loop-invariant value, skip
  // if (v.getParentRegion() != &forOp.getLoopBody())
  if (v.getParentRegion() != forOp.getRegion())
    return;

  // Since we only need to peel the loop numStages-1 times, don't worry
  // about depends that are too far away
  if (stage < 0)
    return;

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getArgNumber() > 0) {
      deps.insert(v);
      collectValueDep(forOp, yieldOp,
                      yieldOp->getOperand(arg.getArgNumber() - 1), stage - 1,
                      deps);
    }
  } else { // value
    Operation *defOp = v.getDefiningOp();
    deps.insert(v);
    // recursively collect all nested ops deps.
    defOp->walk<WalkOrder::PreOrder>([&](Operation *nested) {
      for (OpOperand &operand : nested->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if ((def && !defOp->isAncestor(def)) ||
            isa<BlockArgument>(operand.get()))
          collectValueDep(forOp, yieldOp, operand.get(), stage, deps);
      }
    });
  }
}

std::pair<SetVector<Operation *>, SetVector<Operation *>>
collectValidLoadRecursion(scf::ForOp &forOp, scf::YieldOp &yieldOp,
                          DenseMap<Value, Value> &loadsMapping, int numStages,
                          Value v, SetVector<Operation *> &ops) {
  Operation *op = v.getDefiningOp();
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    if (!op->getResult(0).hasOneUse())
      return std::make_pair(SetVector<Operation *>(), SetVector<Operation *>());
    if (v.getParentRegion() != forOp.getRegion())
      return std::make_pair(SetVector<Operation *>(), SetVector<Operation *>());
    SetVector<Operation *> loads;
    SetVector<Value> deps;
    // collect valid loadOp deps and check.
    bool isValid = true;
    for (Value v : loadOp.getOperands()) {
      collectValueDep(forOp, yieldOp, v, numStages - 1,
                      deps); // zmw? no need this
    }
    assert(!deps.empty());
    // for mla the pattern is: dot -> cvt -> load(data) -> ... load(index)
    // so check the loadop in ops instead of deps
    for (Operation *other : ops) {
      if (isa<triton::LoadOp>(other)) {
        if (deps.contains(other->getResult(0))) {
          isValid = false;
          break;
        }
      }
    }
    // for (Value v : deps) {
    //   // Don't pipeline valid loads that depend on other valid loads
    //   Operation *op_ = v.getDefiningOp();
    //   if (op_ && isa<triton::LoadOp>(op_))
    //     isValid = false;
    // }
    if (isValid)
      loads.insert(op);
    return std::make_pair(loads, SetVector<Operation *>());
    // constant && make range && splat
  } else if (isa<arith::ConstantOp>(op) || isa<triton::MakeRangeOp>(op) ||
             isa<triton::SplatOp>(op)) {
    // TODO(): maybe need to check splat's operand is arg?
    auto constant = dyn_cast<arith::ConstantOp>(op);
    // TODO(MACA): only dense constant
    if (constant) {
      if (!isa<DenseElementsAttr>(constant.getValue()))
        return std::make_pair(SetVector<Operation *>(),
                              SetVector<Operation *>());
    }
    SetVector<Operation *> loads;
    SetVector<Operation *> deps;
    loads.insert(op);
    deps.insert(op);
    return std::make_pair(loads, deps); // zmw? first is not all loads
    // block argument
  } else if (auto arg = dyn_cast<BlockArgument>(v)) {
    return std::make_pair(SetVector<Operation *>(), SetVector<Operation *>());
    // TODO(MACA): only support trans <- cvt(B2S) <- load.
  } else if (auto trans = dyn_cast<triton::TransOp>(op)) {
    auto transSrc = trans.getSrc();
    auto srcTy = isa<mlir::triton::gpu::MemDescType>(transSrc.getType());
    auto cvt = isa<triton::gpu::LocalAllocOp>(transSrc.getDefiningOp());
    if (cvt && srcTy) {
      auto cvtSrc = transSrc.getDefiningOp()->getOperand(0);
      auto load = dyn_cast<triton::LoadOp>(cvtSrc.getDefiningOp());
      if (load) {
        auto collector = collectValidLoadRecursion(forOp, yieldOp, loadsMapping,
                                                   numStages, transSrc, ops);
        if (!collector.first.empty()) {
          auto deps = collector.second;
          deps.insert(op);
          return std::make_pair(collector.first, deps);
        }
      }
    }
    return std::make_pair(SetVector<Operation *>(), SetVector<Operation *>());
  } else {

    // TODO(MACA): only support RankedTensorType out of loop.
    if (v.getParentRegion() != forOp.getRegion())
      if (auto ty = dyn_cast<mlir::triton::gpu::MemDescType>(v.getType()))
        return std::make_pair(SetVector<Operation *>(),
                              SetVector<Operation *>());
    SetVector<Operation *> loads;
    SetVector<Operation *> deps;
    Attribute commonAttr = nullptr;
    for (Value operand : op->getOperands()) {
      auto operandTy =
          llvm::dyn_cast<triton::gpu::TensorOrMemDesc>(operand.getType());
      if (!operandTy)
        return std::make_pair(SetVector<Operation *>(),
                              SetVector<Operation *>());
      Attribute operandAttr = operandTy.getEncoding();
      // require all operands have same encoding attribute.
      if (!commonAttr)
        commonAttr = operandAttr;
      if (commonAttr != operandAttr)
        return std::make_pair(SetVector<Operation *>(),
                              SetVector<Operation *>());
      auto collector = collectValidLoadRecursion(forOp, yieldOp, loadsMapping,
                                                 numStages, operand, ops);
      if (collector.first.empty())
        return std::make_pair(SetVector<Operation *>(),
                              SetVector<Operation *>());
      for (Operation *load : collector.first)
        loads.insert(load);
      for (Operation *dep : collector.second)
        deps.insert(dep);
    }
    if (!loads.empty())
      deps.insert(op);
    if (auto convertLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      if (auto tensorType =
              dyn_cast<RankedTensorType>(convertLayout.getResult().getType())) {
        if (auto dotOpEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
                tensorType.getEncoding())) {
          for (Operation *load : loads)
            loadsMapping[load->getResult(0)] = convertLayout;
          for (Operation *dep : deps)
            loadsMapping[dep->getResult(0)] = convertLayout;
        }
      }
    }

    if (auto convertLayout = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      if (auto tensorType =
              dyn_cast<RankedTensorType>(convertLayout.getResult().getType())) {
        if (auto dotOpEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
                tensorType.getEncoding())) {
          for (Operation *load : loads)
            loadsMapping[load->getResult(0)] = convertLayout;
          for (Operation *dep : deps)
            loadsMapping[dep->getResult(0)] = convertLayout;
        }
      }
    }
    return std::make_pair(loads, deps);
  }
}

void removeReplicateOp(SetVector<Operation *> &ops) {
  for (Operation *op : ops) {
    int num = 0;
    for (Operation *op_ : ops) {
      if (op == op_) {
        num += 1;
        if (num > 1)
          ops.remove(op_);
      }
    }
  }
}

void reorderedDeps(scf::ForOp &forOp, SetVector<Operation *> &orderedOps,
                   SetVector<Operation *> &opsOutOfLoop,
                   SetVector<Operation *> &ops) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (ops.contains(&op))
      orderedOps.insert(&op);
  }
  // TODO(MACA): reorder ops out of loop
  for (Operation *op : ops) {
    if (!orderedOps.contains(op))
      opsOutOfLoop.insert(op);
  }
}

LogicalResult collectValidOp(scf::ForOp &forOp, scf::YieldOp &yieldOp,
                             int numStages,
                             DenseMap<Value, Value> &loadsMapping,
                             DenseMap<Value, Value> &elemsMapping,
                             SetVector<Value> &validLoads,
                             SetVector<Operation *> &dotsDeps,
                             SetVector<Operation *> &dotsDepsOutOfLoop,
                             SetVector<Operation *> &ops, bool enableDotStage) {
  // collect dots & loads.
  SetVector<triton::DotOp> dotsInLoop;
  for (Operation &op : forOp) {
    if (auto dotOp = dyn_cast<triton::DotOp>(&op)) {
      dotsInLoop.insert(dotOp);
    }
  }
  if (enableDotStage && dotsInLoop.size() > 1)
    return failure();
  // collect valid loadOp.
  SetVector<Operation *> deps;
  DenseMap<Value, Value> cvtMapping;
  for (triton::DotOp dot : dotsInLoop) {
    // collect operands of a
    auto aOps = collectValidLoadRecursion(forOp, yieldOp, cvtMapping, numStages,
                                          dot.getA(), ops);
    // collect operands of b
    auto bOps = collectValidLoadRecursion(forOp, yieldOp, cvtMapping, numStages,
                                          dot.getB(), ops);
    for (Operation *op : aOps.first) {
      auto res = op->getResult(0);
      if (isa<triton::LoadOp>(op)) {
        if (validLoads.contains(res))
          return failure();
        loadsMapping[res] = cvtMapping[res];
        validLoads.insert(res);
        ops.insert(op);
      } else {
        elemsMapping[res] = cvtMapping[res];
      }
    }
    for (Operation *op : bOps.first) {
      auto res = op->getResult(0);
      if (isa<triton::LoadOp>(op)) {
        if (validLoads.contains(res))
          return failure();
        loadsMapping[res] = cvtMapping[res];
        validLoads.insert(res);
        ops.insert(op);
      } else {
        elemsMapping[res] = cvtMapping[res];
      }
    }
    for (Operation *op : aOps.second) {
      auto res = op->getResult(0);
      deps.insert(op);
      elemsMapping[res] = cvtMapping[res];
    }
    for (Operation *op : bOps.second) {
      auto res = op->getResult(0);
      deps.insert(op);
      elemsMapping[res] = cvtMapping[res];
    }
    deps.insert(dot);
  }
  removeReplicateOp(deps);
  reorderedDeps(forOp, dotsDeps, dotsDepsOutOfLoop, deps);
  if (validLoads.empty())
    return failure();
  return success();
}

std::pair<bool, SmallVector<Operation *>>
removeReplicateOps(SmallVector<Operation *> genDeps) {
  bool needIterCheck = false;
  SmallVector<Operation *> remainDeps;
  for (Operation *op : genDeps) {
    // remove redundent cvts
    if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      Value src = cvt.getSrc();
      Value res = cvt.getResult();
      auto srcTy = dyn_cast<triton::gpu::TensorOrMemDesc>(src.getType());
      auto resTy = dyn_cast<triton::gpu::TensorOrMemDesc>(res.getType());
      Attribute srcAttr = srcTy.getEncoding();
      Attribute resAttr = resTy.getEncoding();
      if (srcAttr == resAttr) {
        cvt.getResult().replaceAllUsesWith(src);
        cvt.erase();
        needIterCheck = true;
        continue;
      }
    } else if (isa<triton::gpu::LocalLoadOp>(op) ||
               isa<triton::gpu::LocalAllocOp>(op)) {
      Value src = op->getOperand(0);
      Value res = op->getResult(0);
      auto srcTy = dyn_cast<triton::gpu::TensorOrMemDesc>(src.getType());
      auto resTy = dyn_cast<triton::gpu::TensorOrMemDesc>(res.getType());
      Attribute srcAttr = srcTy.getEncoding();
      Attribute resAttr = resTy.getEncoding();
      if (srcAttr == resAttr) {
        res.replaceAllUsesWith(src);
        op->erase();
        needIterCheck = true;
        continue;
      }
      // check valid and remove redundent trans op
    } else if (auto trans = dyn_cast<triton::TransOp>(op)) {
      Value src = trans.getSrc();
      Value res = trans.getResult();
      auto transTy = dyn_cast<mlir::triton::gpu::MemDescType>(src.getType());
      Attribute srcAttr = transTy.getEncoding();
      auto transResTy = dyn_cast<mlir::triton::gpu::MemDescType>(res.getType());
      Attribute resAttr = transResTy.getEncoding();
      assert(transTy);
      assert(transResTy);
      if (srcAttr == resAttr) {
        res.replaceAllUsesWith(src);
        trans.erase();
        needIterCheck = true;
        continue;
      } else {
        auto srcOrder = triton::gpu::getOrder(transTy);
        auto dstOrder = triton::gpu::getOrder(transResTy);
        assert(srcOrder[0] == dstOrder[1]);
      }
      // check valid dot op
    } else if (auto dot = dyn_cast<triton::DotOp>(op)) {
      Value a = dot.getA();
      Value b = dot.getB();
      auto aTy = dyn_cast<RankedTensorType>(a.getType());
      auto bTy = dyn_cast<RankedTensorType>(b.getType());
      assert(aTy);
      assert(bTy);
      Attribute aAttr = aTy.getEncoding();
      Attribute bAttr = bTy.getEncoding();
      auto aDotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(aAttr);
      auto bDotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(bAttr);
      assert(aDotEnc);
      assert(bDotEnc);
      // // check valid tensor type of other elementwise op
      // } else {
      //   auto res = op->getResult(0);
      //   auto resTy =
      //       dyn_cast<RankedTensorType>(res.getType());
      //   assert(resTy);
    }
    remainDeps.push_back(op);
  }
  return std::make_pair(needIterCheck, remainDeps);
}

SmallVector<Operation *> genDotDeps(
    OpBuilder &builder, scf::ForOp &forOp, SetVector<Value> &validLoads,
    IRMapping &nextMapping, IRMapping &mapping,
    DenseMap<Value, Value> &loadsMapping, DenseMap<Value, Value> &elemsMapping,
    SetVector<Operation *> &dotsDeps, DenseMap<Value, Value> &nextDotOperands,
    DenseMap<Value, Value> &cvtStageBuffer, DenseMap<Value, Value> &cvtsExtract,
    Value extractSliceIdx, bool enableDotStage, int loadIdx, bool multiDot) {
  // generate dot's deps in loop
  SmallVector<Operation *> genDeps;
  if (dotsDeps.empty()) {
    return genDeps;
  }
  for (Operation *op : dotsDeps) {
    // skip TransOp
    if (auto trans = dyn_cast<triton::TransOp>(op)) {
      Value src = trans.getSrc();
      Value mapSrc = nextMapping.lookupOrDefault(src);
      Value dst = trans.getResult();
      nextMapping.map(dst, mapSrc);
      mapping.map(dst, mapSrc);

      auto convertLayout =
          llvm::dyn_cast<ttg::ConvertLayoutOp>(src.getDefiningOp());
      assert(convertLayout);
      auto v = convertLayout.getSrc();
      auto loadOp = llvm::dyn_cast<triton::LoadOp>(
          convertLayout.getSrc().getDefiningOp());
      if (loadOp) {
        auto operandVal = nextMapping.lookupOrDefault(loadOp);
        assert(operandVal);
        auto operandTy =
            cast<triton::gpu::TensorOrMemDesc>(operandVal.getType());
        Attribute operandAttr = operandTy.getEncoding();
        auto operandBlocked =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(operandAttr);
        if (operandBlocked) {
          auto res = op->getResult(0);
          auto resTy = dyn_cast<triton::gpu::TensorOrMemDesc>(res.getType());
          auto resShape = resTy.getShape();
          auto resAttr = resTy.getEncoding();
          auto resShared =
              dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(resAttr);
          auto resDot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(resAttr);

          if (resShared || resDot) {
            assert(extractSliceIdx);
            auto cvt = elemsMapping[res];
            auto alloc = cvtStageBuffer[cvt];
            mlir::triton::gpu::MemDescType allocTy =
                cast<mlir::triton::gpu::MemDescType>(alloc.getType());
            Value zero =
                builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
            genDeps.push_back(zero.getDefiningOp());
            MLIRContext *ctx = alloc.getContext();
            auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
            mlir::triton::gpu::MemDescType subviewTy = ttg::MemDescType::get(
                allocTy.getShape().drop_front(), allocTy.getElementType(),
                allocTy.getEncoding(), sharedMemorySpace,
                /*mutableMemory=*/true);
            auto loc = op->getLoc();

            auto view = builder.create<triton::gpu::MemDescIndexOp>(
                loc, subviewTy, alloc, extractSliceIdx);
            genDeps.push_back(view);
            nextMapping.map(v, view->getResult(0));

            auto lstore = builder.create<triton::gpu::LocalStoreOp>(
                loc, operandVal, view, true);
            genDeps.push_back(lstore);
            cvtsExtract[cvt] = v;
          }
        }
      }
    } else if (auto dot = dyn_cast<triton::DotOp>(op)) {
      // a of dotop_1 come from mapping, so we need to check
      if (multiDot) {
        auto a = mapping.lookupOrDefault(dot.getA());
        if (a != dot.getA()) {
          nextMapping.map(dot.getA(), a);
        }
        auto b = mapping.lookupOrDefault(dot.getB());
        if (b != dot.getB()) {
          nextMapping.map(dot.getB(), b);
        }
        auto c = mapping.lookupOrDefault(dot.getC());
        if (c != dot.getC()) {
          nextMapping.map(dot.getC(), c);
        }
      }
      auto newOp = builder.clone(*op, nextMapping);
      nextMapping.map(op->getResult(0), newOp->getResult(0));
      mapping.map(op->getResult(0), newOp->getResult(0));
      genDeps.push_back(newOp);
    } else if (isa<triton::MakeRangeOp>(op) || isa<arith::ConstantOp>(op) ||
               isa<triton::SplatOp>(op)) {
      Operation *newOp = nullptr;
      auto res = op->getResult(0);
      auto resTy = dyn_cast<RankedTensorType>(res.getType());
      assert(resTy);
      Attribute resAttr = resTy.getEncoding();
      auto cvt = elemsMapping[res];
      auto cvtDstTy = dyn_cast<RankedTensorType>(cvt.getType());
      auto cvtDstAttr = cvtDstTy.getEncoding();
      Attribute newAttr;
      if (auto slice = dyn_cast<triton::gpu::SliceEncodingAttr>(resAttr)) {
        newAttr = triton::gpu::SliceEncodingAttr::get(
            resTy.getContext(), slice.getDim(),
            cast<ttg::DistributedEncodingTrait>(cvtDstAttr));
      } else {
        newAttr = cvtDstAttr;
      }
      auto newTy = RankedTensorType::get(resTy.getShape(),
                                         resTy.getElementType(), newAttr);
      newOp = builder.clone(*op, nextMapping);
      auto constant = dyn_cast<arith::ConstantOp>(newOp);
      if (constant) {
        if (auto attr = dyn_cast<DenseElementsAttr>(constant.getValue())) {
          auto elemAttr =
              DenseElementsAttr::get(newTy, attr.getSplatValue<Attribute>());
          constant.setValueAttr(elemAttr);
        } else {
          assert(false);
        }
      }
      newOp->getResult(0).setType(newTy);
      nextMapping.map(op->getResult(0), newOp->getResult(0));
      mapping.map(op->getResult(0), newOp->getResult(0));
      genDeps.push_back(newOp);
    } else {
      auto res = op->getResult(0);
      auto resTy = dyn_cast<triton::gpu::TensorOrMemDesc>(res.getType());
      auto resShape = resTy.getShape();
      auto resAttr = resTy.getEncoding();
      auto resShared =
          dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(resAttr);
      auto resDot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(resAttr);
      Attribute nextAttr = nullptr;
      bool skipBuild = false;
      auto cvt = elemsMapping[res];
      for (auto v : op->getOperands()) {
        auto operandVal = nextMapping.lookupOrDefault(v);
        assert(operandVal);
        auto operandTy =
            cast<triton::gpu::TensorOrMemDesc>(operandVal.getType());
        Attribute operandAttr = operandTy.getEncoding();
        auto operandShared =
            dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(operandAttr);
        auto operandDot =
            dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandAttr);
        auto operandSlice =
            dyn_cast<triton::gpu::SliceEncodingAttr>(operandAttr);
        auto operandBlocked =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(operandAttr);
        if (operandShared) {
          // generate shared->dot or shared->trans->dot
          auto load = dyn_cast<triton::LoadOp>(v.getDefiningOp());
          assert(load);
          Value cvt = loadsMapping[load];
          Value cvtSrc = cvt.getDefiningOp()->getOperand(0);
          auto transOp = dyn_cast<triton::TransOp>(cvtSrc.getDefiningOp());
          Value newOperand = operandVal;
          if (transOp) {
            auto operandOrder = triton::gpu::getOrder(operandTy);
            auto transSrc = transOp.getSrc();
            auto transTy =
                dyn_cast<mlir::triton::gpu::MemDescType>(transSrc.getType());
            assert(transTy);
            auto transSrcOrder = triton::gpu::getOrder(transTy);
            assert(operandOrder[0] == transSrcOrder[0]);
            newOperand = builder.create<triton::TransOp>(
                op->getLoc(), newOperand, transOp.getOrder());
            genDeps.push_back(newOperand.getDefiningOp());
          }
          auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
          assert(dstTy);
          nextAttr = dstTy.getEncoding();
          auto dstDot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(nextAttr);
          assert(dstDot);
          auto newDstTy = RankedTensorType::get(
              dstTy.getShape(), operandTy.getElementType(), nextAttr);
          Value lload = builder.create<triton::gpu::LocalLoadOp>(
              op->getLoc(), newDstTy, newOperand);
          if (enableDotStage) {
            auto opIdx = dstDot.getOpIdx();
            auto it = std::find(validLoads.begin(), validLoads.end(),
                                load.getResult());
            if (it != validLoads.end()) {
              auto loadArgIdx = std::distance(validLoads.begin(), it);
              nextDotOperands[load.getResult()] = lload;
              nextMapping.map(v,
                              forOp.getRegionIterArgs()[loadIdx + loadArgIdx]);
            } else {
              assert(false);
            }
          } else {
            nextMapping.map(v, lload);
          }
          genDeps.push_back(lload.getDefiningOp());
        } else if (operandDot) {
          nextAttr = operandAttr;
          continue;
        } else if (operandBlocked) {
          if (resShared || resDot) {
            assert(extractSliceIdx);
            auto alloc = cvtStageBuffer[cvt];
            mlir::triton::gpu::MemDescType allocTy =
                cast<mlir::triton::gpu::MemDescType>(alloc.getType());
            Value zero =
                builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
            genDeps.push_back(zero.getDefiningOp());

            MLIRContext *ctx = alloc.getContext();
            auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
            mlir::triton::gpu::MemDescType subviewTy = ttg::MemDescType::get(
                allocTy.getShape().drop_front(), allocTy.getElementType(),
                allocTy.getEncoding(), sharedMemorySpace,
                /*mutableMemory=*/true);
            auto loc = op->getLoc();

            auto view = builder.create<triton::gpu::MemDescIndexOp>(
                loc, subviewTy, alloc, extractSliceIdx);
            genDeps.push_back(view);
            nextMapping.map(v, view->getResult(0));

            auto lstore = builder.create<triton::gpu::LocalStoreOp>(
                loc, operandVal, view, true);
            genDeps.push_back(lstore);
            cvtsExtract[cvt] = v;
          }
          nextAttr = resAttr;
          if (resDot)
            // skip build share -> dot here becasue
            // it will be built when operand is shared.
            skipBuild = true;
        } else if (operandSlice) {
          if (isa<triton::ExpandDimsOp>(op)) {
            nextAttr = operandSlice.getParent();
          } else {
            nextAttr = operandAttr;
          }
        } else {
          assert(false);
        }
      }
      if (skipBuild)
        continue;
      auto nextBlocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(nextAttr);
      if (nextBlocked)
        cvtsExtract[cvt] = res;
      auto newOp = builder.clone(*op, nextMapping);
      auto newType =
          RankedTensorType::get(resShape, resTy.getElementType(), nextAttr);
      newOp->getResult(0).setType(newType);
      nextMapping.map(res, newOp->getResult(0));
      mapping.map(res, newOp->getResult(0));
      genDeps.push_back(newOp);
    }
  }
  if (!multiDot) {
    bool needIterCheck = true;
    while (needIterCheck) {
      auto iters = removeReplicateOps(genDeps);
      needIterCheck = iters.first;
      genDeps = iters.second;
    }
    assert(genDeps.size() != 0);
  }
  return genDeps;
}

void OptimizeCStore(OpBuilder &builder, triton::StoreOp &storeOp) {
  if (!storeOp.getMask()) {
    return;
  }
  auto storeArg0 = storeOp.getOperand(0);
  auto storeArg1 = storeOp.getOperand(1);
  auto storeArg2 = storeOp.getOperand(2);
  auto storeArg0Type = cast<RankedTensorType>(storeArg0.getType());
  auto storeArg1Type = cast<RankedTensorType>(storeArg1.getType());
  auto storeArg2Type = cast<RankedTensorType>(storeArg2.getType());
  auto argEncoding = storeArg1Type.getEncoding();
  // check if cvtOp mma -> blocked
  if (!isa<triton::gpu::BlockedEncodingAttr>(argEncoding)) {
    return;
  }

  auto cvtOp =
      dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(storeArg1.getDefiningOp());
  if (!cvtOp) {
    auto fpToFpOp =
        dyn_cast_or_null<triton::FpToFpOp>(storeArg1.getDefiningOp());
    auto truncfOp =
        dyn_cast_or_null<arith::TruncFOp>(storeArg1.getDefiningOp());
    if (!fpToFpOp && !truncfOp) {
      return;
    }

    Value castTypeOpArg;
    if (fpToFpOp) {
      castTypeOpArg = fpToFpOp.getOperand();
    } else {
      castTypeOpArg = truncfOp.getOperand();
    }
    auto mulfOp =
        dyn_cast_or_null<arith::MulFOp>(castTypeOpArg.getDefiningOp());
    if (!mulfOp) {
      return;
    }
    auto mulfLhsOp = mulfOp.getLhs();
    auto mulfRhsOp = mulfOp.getRhs();

    auto lhsCvtOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        mulfLhsOp.getDefiningOp());
    if (!lhsCvtOp) {
      return;
    }
    auto rhsBroadCastOp =
        dyn_cast_or_null<triton::BroadcastOp>(mulfRhsOp.getDefiningOp());
    if (!rhsBroadCastOp) {
      return;
    }
    auto broadCastArg = rhsBroadCastOp.getOperand();
    auto rhsCvtOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        broadCastArg.getDefiningOp());
    if (!rhsCvtOp) {
      return;
    }
    auto rhsCvtArg = rhsCvtOp.getOperand();

    auto lhsCvtOpArg = lhsCvtOp.getOperand();
    auto lhsCvtOpArgType = cast<RankedTensorType>(lhsCvtOpArg.getType());
    if (!lhsCvtOpArgType) {
      return;
    }
    auto lhsMMaEncoding = lhsCvtOpArgType.getEncoding();
    auto lhsMmaLayout = cast<triton::gpu::MACAMmaEncodingAttr>(lhsMMaEncoding);
    if (!lhsMmaLayout) {
      return;
    }

    auto elemsMnk = lhsMmaLayout.getElementsMNK();
    auto elementType = storeArg1Type.getElementType();
    auto bitwidth = elementType.getIntOrFloatBitWidth() * elemsMnk[1];

    // must no partial write
    bool can_bypass_sharemem = (bitwidth >= 32);
    if (!can_bypass_sharemem) {
      return;
    }

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(storeOp);

    // convert block3 to mma
    auto rhsCvtArgType = cast<RankedTensorType>(rhsCvtArg.getType());
    if (!rhsCvtArgType) {
      return;
    }
    auto rhsCvtNewType = RankedTensorType::get(
        rhsCvtArgType.getShape(), rhsCvtArgType.getElementType(), lhsMmaLayout);
    auto rhsCvtNew = builder.create<triton::gpu::ConvertLayoutOp>(
        storeOp.getLoc(), rhsCvtNewType, rhsCvtArg);

    auto rhsBroadCastOpType = cast<RankedTensorType>(rhsBroadCastOp.getType());
    if (!rhsBroadCastOpType) {
      return;
    }
    auto rhsCvtArgTypeNew = RankedTensorType::get(
        rhsBroadCastOpType.getShape(), rhsBroadCastOpType.getElementType(),
        lhsMmaLayout);
    auto newMulfRhs = builder.create<triton::BroadcastOp>(
        storeOp.getLoc(), rhsCvtArgTypeNew, rhsCvtNew.getResult());
    auto newMulf = builder.create<mlir::arith::MulFOp>(
        storeOp.getLoc(), lhsCvtOpArg, newMulfRhs.getResult());

    RankedTensorType castOpType;
    if (fpToFpOp) {
      castOpType = cast<RankedTensorType>(fpToFpOp.getType());
    } else {
      castOpType = cast<RankedTensorType>(truncfOp.getType());
    }
    if (!castOpType) {
      return;
    }
    auto castOpTypeNew = RankedTensorType::get(
        castOpType.getShape(), castOpType.getElementType(), lhsMmaLayout);

    auto arg0TypeNew = RankedTensorType::get(
        storeArg0Type.getShape(), storeArg0Type.getElementType(), lhsMmaLayout);
    auto arg2TypeNew = RankedTensorType::get(
        storeArg2Type.getShape(), storeArg2Type.getElementType(), lhsMmaLayout);
    auto arg0New = builder.create<triton::gpu::ConvertLayoutOp>(
        storeOp.getLoc(), arg0TypeNew, storeArg0);
    auto arg2New = builder.create<triton::gpu::ConvertLayoutOp>(
        storeOp.getLoc(), arg2TypeNew, storeArg2);

    if (fpToFpOp) {
      auto roundingMode = fpToFpOp.getRounding().value();
      auto roundingAttr =
          triton::RoundingModeAttr::get(builder.getContext(), roundingMode);
      auto newFpToFp = builder.create<triton::FpToFpOp>(
          storeOp.getLoc(), castOpTypeNew, newMulf.getResult(), roundingAttr);
      auto new_store = builder.create<triton::StoreOp>(
          storeOp.getLoc(), arg0New.getResult(), newFpToFp.getResult(),
          arg2New.getResult(), storeOp.getCache(), storeOp.getEvict());
    } else {
      auto newtruncfOp = builder.create<mlir::arith::TruncFOp>(
          storeOp.getLoc(), castOpTypeNew, newMulf.getResult());
      auto new_store = builder.create<triton::StoreOp>(
          storeOp.getLoc(), arg0New.getResult(), newtruncfOp.getResult(),
          arg2New.getResult(), storeOp.getCache(), storeOp.getEvict());
    }

    storeOp.erase();
    return;
  }

  auto cvtArg = cvtOp.getOperand();
  auto cvtArgType = cast<RankedTensorType>(cvtArg.getType());
  auto mmaEncoding = cvtArgType.getEncoding();
  if (!isa<triton::gpu::MACAMmaEncodingAttr>(mmaEncoding)) {
    return;
  }

  // mma layout info
  auto mmaLayout = cast<triton::gpu::MACAMmaEncodingAttr>(mmaEncoding);
  auto elemsMnk = mmaLayout.getElementsMNK();
  auto colMajor = mmaLayout.getColMajor();
  if (colMajor == 1) {
    return;
  }
  auto elementType = cvtArgType.getElementType();
  auto bitwidth = elementType.getIntOrFloatBitWidth() * elemsMnk[1];

  // must no partial write
  bool can_bypass_sharemem = (bitwidth >= 32);
  if (!can_bypass_sharemem) {
    return;
  }

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfter(storeOp);

  auto arg0TypeNew = RankedTensorType::get(
      storeArg0Type.getShape(), storeArg0Type.getElementType(), mmaLayout);
  auto arg2TypeNew = RankedTensorType::get(
      storeArg2Type.getShape(), storeArg2Type.getElementType(), mmaLayout);
  auto arg0New = builder.create<triton::gpu::ConvertLayoutOp>(
      storeOp.getLoc(), arg0TypeNew, storeArg0);
  auto arg2New = builder.create<triton::gpu::ConvertLayoutOp>(
      storeOp.getLoc(), arg2TypeNew, storeArg2);

  // TODO: check if layout convert is expensive
  // simulateBackwardMaterialization
  auto new_store = builder.create<triton::StoreOp>(
      storeOp.getLoc(), arg0New.getResult(), cvtArg, arg2New.getResult(),
      storeOp.getCache(), storeOp.getEvict());

  storeOp.erase();

  return;
}

int getMmaInstNum(Type type) {
  if (type.isF32()) {
    return 8;
  } else if (type.isF16() || type.isBF16()) {
    return 4;
  } else if (type.isSignlessInteger(8)) {
    return 2;
  } else {
    assert(false && "invalid dtype for mma");
  }
}

int getNumWarps(ModuleOp mod) {
  if (!mod->hasAttr("ttg.num-warps"))
    llvm::report_fatal_error(
        "TritonGPU module should contain a ttg.num-warps attribute");
  return cast<IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
}

mlir::Attribute getDotOperandsAttr(Value v) {
  // walk back to conversion
  Operation *op = v.getDefiningOp();
  mlir::Attribute layout;
  while (op) {
    if (op->getNumOperands() != 1)
      break;
    if (!op->getResult(0).hasOneUse())
      break;
    if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      auto cvtTy = op->getOperand(0).getType();
      if (auto memTy = dyn_cast<ttg::MemDescType>(cvtTy)) {
        if (mlir::isa<SwizzledSharedEncodingAttr>(memTy.getEncoding())) {
          layout = memTy.getEncoding();
          break;
        }
      }
    } else if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto cvtTy = op->getOperand(0).getType();
      if (auto memTy = dyn_cast<ttg::MemDescType>(cvtTy)) {
        if (mlir::isa<SwizzledSharedEncodingAttr>(memTy.getEncoding())) {
          layout = memTy.getEncoding();
          break;
        }
      } else if (auto tensorTy = dyn_cast<mlir::RankedTensorType>(cvtTy)) {
        if (mlir::isa<triton::gpu::BlockedEncodingAttr>(
                tensorTy.getEncoding())) {
          layout = tensorTy.getEncoding();
          break;
        }
      }
    }
    op = op->getOperand(0).getDefiningOp();
  }
  return layout;
}
