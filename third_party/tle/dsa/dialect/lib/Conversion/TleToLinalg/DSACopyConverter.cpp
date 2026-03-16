// Copyright 2026- Xcoresigma Technology Co., Ltd

#include "tle/dsa/dialect/include/Conversion/TleToLinalg/DSACopyConverter.h"
#if __has_include("bishengir/Dialect/HIVM/IR/HIVM.h")
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"

namespace TleCopyConverter {

using namespace mlir;

memref::SubViewOp makeSubViewOp(Value src,
                                const llvm::SmallVector<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter) {
  auto srcType = cast<MemRefType>(src.getType());
  SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(srcType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<memref::SubViewOp>(loc, dyn_cast<MemRefType>(dstType),
                                            src, offsets, sizes, strides);
}

CopyConverter::CopyConverter(MLIRContext *context)
    : OpConversionPattern<triton::tle::DSACopyOp>(context) {}

LogicalResult
CopyConverter::matchAndRewrite(triton::tle::DSACopyOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto src = adaptor.getSrc();
  auto dst = adaptor.getDst();
  auto loc = op.getLoc();

  if (!dyn_cast<MemRefType>(src.getType()) ||
      !dyn_cast<MemRefType>(dst.getType())) {
    op.emitError("Unexpected copy type!");
    return failure();
  }

  llvm::SmallVector<mlir::OpFoldResult> shapeValues;
  for (auto shape : adaptor.getShape()) {
    Value indexShape = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), shape);
    shapeValues.push_back(indexShape);
  }

  // create copyOp
  auto srcSubView = makeSubViewOp(src, shapeValues, loc, rewriter);
  auto dstSubView = makeSubViewOp(dst, shapeValues, loc, rewriter);

  MemRefType srcMemRefTy = cast<MemRefType>(srcSubView.getType());
  MemRefType dstMemRefTy = cast<MemRefType>(dstSubView.getType());

  // Extract AddressSpace from MemRefType.
  auto getAddressSpace = [](MemRefType ty) -> hivm::AddressSpace {
    auto attr = ty.getMemorySpace();
    if (!attr) {
      // The default memory attribute is GM.
      return hivm::AddressSpace::GM;
    }
    auto addrSpaceAttr = dyn_cast<hivm::AddressSpaceAttr>(attr);
    if (!addrSpaceAttr) {
      return hivm::AddressSpace::GM;
    }
    return addrSpaceAttr.getAddressSpace();
  };

  hivm::AddressSpace srcAddrSpace = getAddressSpace(srcMemRefTy);
  hivm::AddressSpace dstAddrSpace = getAddressSpace(dstMemRefTy);

  Operation *copyOp = nullptr;
  if (srcAddrSpace == hivm::AddressSpace::GM &&
          dstAddrSpace == hivm::AddressSpace::UB ||
      srcAddrSpace == hivm::AddressSpace::UB &&
          dstAddrSpace == hivm::AddressSpace::GM) {
    copyOp = rewriter.create<memref::CopyOp>(loc, srcSubView, dstSubView);
  } else if (srcAddrSpace == hivm::AddressSpace::GM &&
             dstAddrSpace == hivm::AddressSpace::L1) {
    copyOp = rewriter.create<hivm::ND2NZOp>(
        loc, /*result_tensor=*/TypeRange{},
        /*src=*/srcSubView, /*dst=*/dstSubView,
        /*dst_continuous=*/UnitAttr::get(rewriter.getContext()));
  }
  /// else if (srcAddrSpace == hivm::AddressSpace::L0C &&
  ///            dstAddrSpace == hivm::AddressSpace::GM) {
  ///   copyOp = rewriter.create<hivm::FixpipeOp>(loc,
  ///       /*result_tensor=*/TypeRange{}, /*src=*/srcSubView,
  ///       /*dst=*/dstSubView,
  ///       /*enable_nz2nd=*/UnitAttr::get(rewriter.getContext())

  ///       // #ifdef BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
  ///       /*nullptr,
  ///       hivm::FixpipeDMAModeAttr::get(rewriter.getContext(),
  ///       hivm::FixpipeDMAMode::NZ2ND), nullptr, nullptr, nullptr, nullptr,
  ///       nullptr*/
  ///     );
  /// }
  else {
    op.emitError("Not implemented!");
    return failure();
  }

  copyOp->setAttrs(op->getAttrs());
  rewriter.replaceOp(op, copyOp);
  return success();
}

} // namespace TleCopyConverter

namespace mlir::triton::tle {
void populateTleCopyOpConversionPatterns(mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
  patterns.add<TleCopyConverter::CopyConverter>(patterns.getContext());
}
} // namespace mlir::triton::tle
