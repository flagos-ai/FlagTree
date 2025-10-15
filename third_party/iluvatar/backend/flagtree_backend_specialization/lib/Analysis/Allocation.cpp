#include "triton/Analysis/Allocation.h"

#include "triton/Dialect/Triton/IR/Utility.h"

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getUniqueContigPerThread;
using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

namespace mlir {
namespace triton {

SmallVector<unsigned> getRepShapeForCvtLayout(triton::gpu::ConvertLayoutOp op) {
  auto srcTy = op.getSrc().getType();
  auto dstTy = op.getType();
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  if (shouldUseDistSmem(srcLayout, dstLayout)) {
    // TODO: padding to avoid bank conflicts
    return convertType<unsigned, int64_t>(getShapePerCTA(srcTy));
  }

  if (isMfmaToDotShortcut(srcTy, dstTy))
    return {};

  // MmaToDotShortcut and MmaToMmaShortcut doesn't use shared mem
  if (auto srcMmaLayout = mlir::dyn_cast<NvidiaMmaEncodingAttr>(srcLayout)) {
    if (mlir::isa<DotOperandEncodingAttr>(dstLayout)) {
      if (isMmaToDotShortcut(srcTy, dstTy)) {
        return {};
      }
    } else if (auto dstMmaLayout =
                   mlir::dyn_cast<NvidiaMmaEncodingAttr>(dstLayout)) {
      if (isMmaToMmaShortcut(srcTy, dstTy)) {
        return {};
      }
    }
  }
  if (auto srcMmaLayout = mlir::dyn_cast<IluvatarMmaEncodingAttr>(srcLayout)) {
    if (mlir::isa<DotOperandEncodingAttr>(dstLayout)) {
      if (isMmaToDotShortcut(srcTy, dstTy)) {
        return {};
      } else if (isMmaToDotSlowShortcut(srcTy, dstTy)) {
        return getShapePerCTATile(srcMmaLayout);
      }
    } else if (auto dstMmaLayout =
                   mlir::dyn_cast<IluvatarMmaEncodingAttr>(dstLayout)) {
      if (isMmaToMmaShortcut(srcTy, dstTy)) {
        return {};
      }
    }
  }

  if (auto srcSliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
    if (auto dstSliceLayout = dstLayout.dyn_cast<SliceEncodingAttr>()) {
      if (srcSliceLayout.getParent().isa<IluvatarMmaEncodingAttr>() &&
          dstSliceLayout.getParent().isa<IluvatarMmaEncodingAttr>()) {
        return {};
      }
    }
  }

  assert(srcLayout && dstLayout && "Unexpected layout in getRepShape()");

  auto srcShapePerCTA = getShapePerCTA(srcTy);
  auto dstShapePerCTA = getShapePerCTA(dstTy);
  auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
  auto dstShapePerCTATile = getShapePerCTATile(dstLayout, dstTy.getShape());

  unsigned rank = dstTy.getRank();
  SmallVector<unsigned> repShape(rank);
  for (unsigned d = 0; d < rank; ++d) {
    repShape[d] =
        std::max(std::min<unsigned>(srcShapePerCTA[d], srcShapePerCTATile[d]),
                 std::min<unsigned>(dstShapePerCTA[d], dstShapePerCTATile[d]));
  }
  return repShape;
}

SmallVector<unsigned>
getScratchConfigForCvtLayout(triton::gpu::ConvertLayoutOp op, unsigned &inVec,
                             unsigned &outVec) {
  auto repShape = getRepShapeForCvtLayout(op);
  if (repShape.empty())
    return repShape;
  auto rank = repShape.size();
  auto srcTy = op.getSrc().getType();
  auto dstTy = op.getType();
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  assert(!isMfmaToDotShortcut(srcTy, dstTy));
  if (isMmaToDotSlowShortcut(srcTy, dstTy))
    return repShape;

  auto [inOrd, outOrd] = getCvtOrder(srcLayout, dstLayout);
  unsigned srcContigPerThread =
      getUniqueContigPerThread(srcLayout, srcTy.getShape())[inOrd[0]];
  unsigned dstContigPerThread =
      getUniqueContigPerThread(dstLayout, dstTy.getShape())[outOrd[0]];
  // TODO: Fix the legacy issue that ourOrd[0] == 0 always means
  //       that we cannot do vectorization.
  unsigned innerDim = rank - 1;
  inVec = outOrd[0] != innerDim  ? 1
          : inOrd[0] != innerDim ? 1
                                 : srcContigPerThread;
  outVec = outOrd[0] != innerDim ? 1 : dstContigPerThread;

  // For conversions to MmaV1 (Nvidia V100), this inVec is hardcoded in the
  // codegen.
  if (auto mma = mlir::dyn_cast<NvidiaMmaEncodingAttr>(srcLayout)) {
    if (mma.getVersionMajor() == 1) {
      inVec = srcContigPerThread;
    } else if (mlir::isa<BlockedEncodingAttr>(dstLayout)) {
      // when storing from mma layout and loading in blocked layout vectorizing
      // the load back gives better performance even if there is a
      // transposition.
      outVec = dstContigPerThread;
    }
  }

  if (rank <= 1)
    return repShape;
  // pad the last dimension
  unsigned paddedDim = rank - 1;
  if (auto dstBlockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(dstLayout)) {
    paddedDim = dstBlockedLayout.getOrder()[0];
  }
  unsigned pad = std::max(inVec, outVec);
  if (mlir::dyn_cast<IluvatarMmaEncodingAttr>(srcLayout) &&
      mlir::isa<BlockedEncodingAttr>(dstLayout)) {
    pad = 16;
  }
  repShape[paddedDim] += pad;
  return repShape;
}

unsigned getScratchValueSizeElems(const SmallVector<unsigned> &smemShape) {
  if (smemShape.empty())
    return 0;
  return std::accumulate(smemShape.begin(), smemShape.end(), 1u, std::multiplies<>());
}

void AllocationAnalysis_dump(llvm::MapVector<Allocation::BufferT *, Interval<size_t>> bufferRange) {
  llvm::outs() << "DUMP: "
                << "\n";
  for (auto bufferIter : bufferRange) {
    llvm::outs() << "ID= " << bufferIter.first->id << "\n";
    if (bufferIter.first->kind == Allocation::BufferT::BufferKind::Explicit)
      llvm::outs() << "     Kind= Explict\n";
    else if (bufferIter.first->kind == Allocation::BufferT::BufferKind::Scratch)
      llvm::outs() << "     Kind= Scratch\n";
    else if (bufferIter.first->kind == Allocation::BufferT::BufferKind::Virtual)
      llvm::outs() << "     Kind= Virtual\n";
    llvm::outs() << "     Size= " << bufferIter.first->size << "\n";
    llvm::outs() << "     Offs= " << bufferIter.first->offset << "\n";
    llvm::outs() << "     Interval= [" << bufferIter.second.start() << ", "
                  << bufferIter.second.end() << ")\n";
  }
}

}
}
