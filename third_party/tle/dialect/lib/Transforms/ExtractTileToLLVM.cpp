
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::tle;

// ============================================================================
// Utility functions
// ============================================================================

template <typename T1, typename T2, typename BinaryOp>
static SmallVector<T2> multiDimElementwise(ArrayRef<T1> lhs, ArrayRef<T2> rhs,
                                           BinaryOp op) {
  assert(lhs.size() == rhs.size());
  SmallVector<T2> result;
  result.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(static_cast<T2>(op(lhs[i], rhs[i])));
  return result;
}

static SmallVector<unsigned> getCTATileOrder(RankedTensorType type) {
  if (auto bl = dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding()))
    return SmallVector<unsigned>(bl.getOrder().begin(), bl.getOrder().end());
  unsigned rank = type.getRank();
  SmallVector<unsigned> order(rank);
  for (unsigned i = 0; i < rank; ++i) order[i] = rank - 1 - i;
  return order;
}

static SmallVector<unsigned> delinearize(unsigned idx,
                                         ArrayRef<unsigned> shape,
                                         ArrayRef<unsigned> order) {
  SmallVector<unsigned> result(shape.size(), 0);
  for (size_t i = 0; i < order.size(); ++i) {
    result[order[i]] = idx % shape[order[i]];
    idx /= shape[order[i]];
  }
  return result;
}

static unsigned linearize(ArrayRef<unsigned> coords, ArrayRef<unsigned> shape,
                           ArrayRef<unsigned> order) {
  unsigned result = 0, stride = 1;
  for (size_t i = 0; i < order.size(); ++i) {
    result += coords[order[i]] * stride;
    stride *= shape[order[i]];
  }
  return result;
}

// shapePerCTATile = sizePerThread * threadsPerWarp * warpsPerCTA (elementwise)
static SmallVector<unsigned> getShapePerCTATile(RankedTensorType type) {
  auto enc = type.getEncoding();
  if (!enc) llvm_unreachable("extract_tile requires tensor with encoding");
  if (auto bl = dyn_cast<ttg::BlockedEncodingAttr>(enc)) {
    auto s = bl.getSizePerThread(), t = bl.getThreadsPerWarp(), w = bl.getWarpsPerCTA();
    SmallVector<unsigned> r;
    for (size_t i = 0; i < type.getShape().size(); ++i)
      r.push_back((unsigned)s[i] * (unsigned)t[i] * (unsigned)w[i]);
    return r;
  }
  llvm_unreachable("extract_tile only supports BlockedEncoding");
}

static SmallVector<int64_t> getTileShape(ExtractTileOp op) {
  SmallVector<int64_t> ts;
  if (auto a = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(op->getAttr("tile_shape")))
    for (auto v : a.asArrayRef()) ts.push_back(v);
  return ts;
}

static std::optional<int64_t> getStaticIndex(ExtractTileOp op) {
  if (auto c = op->getOperand(1).getDefiningOp<mlir::arith::ConstantOp>())
    return mlir::cast<mlir::IntegerAttr>(c.getValue()).getInt();
  return std::nullopt;
}

static bool isCTATileAligned(ExtractTileOp op, int64_t linearIndex) {
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto srcShape = srcTy.getShape();
  auto tileShape = getTileShape(op);
  auto ctaTile = getShapePerCTATile(srcTy);
  int rank = srcShape.size();
  SmallVector<int64_t> logicalGrid(rank), tileCoords(rank);
  for (int i = 0; i < rank; ++i) logicalGrid[i] = srcShape[i] / tileShape[i];
  int64_t remain = linearIndex;
  for (int i = rank - 1; i >= 0; --i) {
    tileCoords[i] = remain % logicalGrid[i];
    remain /= logicalGrid[i];
  }
  for (int i = 0; i < rank; ++i) {
    int64_t off = tileCoords[i] * tileShape[i];
    if (tileShape[i] % (int64_t)ctaTile[i] != 0) return false;
    if (off % (int64_t)ctaTile[i] != 0) return false;
  }
  return true;
}

// ============================================================================
// Compute runtime threadOffset[d] as LLVM IR Values
//
// For BlockedEncoding, threadIdx.x encodes thread coordinates in the CTA
// ordered by `order` (fastest to slowest):
//   - threadsPerWarp dimensions (in order)
//   - warpsPerCTA dimensions (in order)
//
// threadOffset[d] = (warpInDim[d] * threadsPerWarp[d] + laneInDim[d])
//                   * sizePerThread[d]
//
// This is the per-dimension global coordinate contribution from the thread
// identity, which must be added to the compile-time srcOffsets[i][d] to get
// the true global coordinate of element i for the current thread.
// ============================================================================
static SmallVector<Value> computeThreadOffsets(
    Location loc,
    ConversionPatternRewriter &rewriter,
    RankedTensorType tensorType) {

  auto bl = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  auto sizePerThread  = bl.getSizePerThread();   // e.g. [1, 1]
  auto threadsPerWarp = bl.getThreadsPerWarp();  // e.g. [1, 32]
  auto warpsPerCTA    = bl.getWarpsPerCTA();     // e.g. [1, 4]
  auto order          = bl.getOrder();           // e.g. [1, 0]
  int rank = tensorType.getRank();

  auto i32Ty = rewriter.getIntegerType(32);

  // threadIdx.x (flat thread index within CTA)
  Value threadId = rewriter.create<NVVM::ThreadIdXOp>(loc, i32Ty);

  // warpSize = product of threadsPerWarp
  unsigned warpSizeVal = 1;
  for (auto t : threadsPerWarp) warpSizeVal *= t;
  Value warpSizeV = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)warpSizeVal));

  Value laneId = rewriter.create<LLVM::URemOp>(loc, i32Ty, threadId, warpSizeV);
  Value warpId = rewriter.create<LLVM::UDivOp>(loc, i32Ty, threadId, warpSizeV);

  // Decompose laneId into per-dimension lane coordinates, following `order`
  // (order[0] is the fastest-changing dimension)
  SmallVector<Value> laneInDim(rank);
  {
    Value rem = laneId;
    for (int i = 0; i < rank; ++i) {
      unsigned dim   = order[i];
      unsigned count = threadsPerWarp[dim];
      Value cv = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)count));
      laneInDim[dim] = rewriter.create<LLVM::URemOp>(loc, i32Ty, rem, cv);
      rem = rewriter.create<LLVM::UDivOp>(loc, i32Ty, rem, cv);
    }
  }

  // Decompose warpId into per-dimension warp coordinates, following `order`
  SmallVector<Value> warpInDim(rank);
  {
    Value rem = warpId;
    for (int i = 0; i < rank; ++i) {
      unsigned dim   = order[i];
      unsigned count = warpsPerCTA[dim];
      Value cv = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)count));
      warpInDim[dim] = rewriter.create<LLVM::URemOp>(loc, i32Ty, rem, cv);
      rem = rewriter.create<LLVM::UDivOp>(loc, i32Ty, rem, cv);
    }
  }

  // threadOffset[d] = (warpInDim[d] * threadsPerWarp[d] + laneInDim[d])
  //                   * sizePerThread[d]
  SmallVector<Value> threadOffsets(rank);
  for (int d = 0; d < rank; ++d) {
    Value tpw = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)threadsPerWarp[d]));
    Value spt = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)sizePerThread[d]));
    Value warpContrib = rewriter.create<LLVM::MulOp>(loc, i32Ty, warpInDim[d], tpw);
    Value threadCoord = rewriter.create<LLVM::AddOp>(loc, i32Ty, warpContrib, laneInDim[d]);
    threadOffsets[d] = rewriter.create<LLVM::MulOp>(loc, i32Ty, threadCoord, spt);
  }

  return threadOffsets;
}

// ============================================================================
// Path 1: Static register permutation (unchanged)
// ============================================================================
static LogicalResult lowerExtractTileStatic(
    ExtractTileOp op, ExtractTileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    const LLVMTypeConverter *typeConverter, int64_t linearIndex) {
  Location loc = op->getLoc();
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape(), dstShape = dstTy.getShape();
  auto tileShape = getTileShape(op);
  int rank = srcShape.size();
  auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  auto shapePerCTATile = getShapePerCTATile(srcTy);
  auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
      srcShape, shapePerCTATile, std::divides<unsigned>());
  auto dstCTAShape = multiDimElementwise<int64_t, unsigned>(
      dstShape, shapePerCTATile, std::divides<unsigned>());
  SmallVector<int64_t> logicalGrid(rank), logicalCoords(rank), elementCoords(rank);
  for (int i = 0; i < rank; ++i) logicalGrid[i] = srcShape[i] / tileShape[i];
  int64_t remain = linearIndex;
  for (int i = rank - 1; i >= 0; --i) {
    logicalCoords[i] = remain % logicalGrid[i];
    remain /= logicalGrid[i];
  }
  for (int i = 0; i < rank; ++i) elementCoords[i] = logicalCoords[i] * tileShape[i];
  auto firstTileCoord = multiDimElementwise<int64_t, unsigned>(
      elementCoords, shapePerCTATile, std::divides<unsigned>());
  auto srcCTAOrder = getCTATileOrder(srcTy), dstCTAOrder = getCTATileOrder(dstTy);
  unsigned totalSrcCTAs = std::accumulate(srcCTAShape.begin(), srcCTAShape.end(),
                                           1, std::multiplies<>());
  unsigned elemsPerCTA = ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
  unsigned numDstCTAs = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                         1, std::multiplies<>());
  SmallVector<Value> resultVals;
  resultVals.reserve(ttg::getTotalElemsPerThread(dstTy));
  for (unsigned i = 0; i < numDstCTAs; ++i) {
    auto coordInDst = delinearize(i, dstCTAShape, dstCTAOrder);
    auto coordInSrc = multiDimElementwise<unsigned, unsigned>(
        coordInDst, firstTileCoord, std::plus<unsigned>());
    unsigned linearInSrc = linearize(coordInSrc, srcCTAShape, srcCTAOrder);
    size_t startIdx = linearInSrc * elemsPerCTA;
    if (startIdx + elemsPerCTA > vals.size())
      return op.emitError("Static path: register index out of bounds");
    llvm::append_range(resultVals, llvm::ArrayRef(vals).slice(startIdx, elemsPerCTA));
  }
  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

// ============================================================================
// Path 2: Dynamic SMEM relay 
// ============================================================================
static LogicalResult lowerExtractTileViaSMEM(
    ExtractTileOp op, ExtractTileOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter,
    const LLVMTypeConverter *typeConverter) {

  Location loc = op->getLoc();
  auto srcTy  = cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy  = cast<RankedTensorType>(op.getType());
  auto srcShape = srcTy.getShape(), dstShape = dstTy.getShape();
  auto tileShape = getTileShape(op);
  int rank = srcShape.size();

  MLIRContext *ctx = rewriter.getContext();
  auto i1Ty  = rewriter.getIntegerType(1);
  auto i8Ty  = rewriter.getIntegerType(8);
  auto i32Ty = rewriter.getIntegerType(32);
  auto elemTy = srcTy.getElementType();
  Type llvmElemTy = typeConverter->convertType(elemTy);
  if (!llvmElemTy)
    return op.emitError("SMEM path: failed to convert element type");
  int64_t elemBytes = llvmElemTy.getIntOrFloatBitWidth() / 8;

  int64_t totalDstElems = 1;
  for (auto dim : dstShape) totalDstElems *= dim;

  // Offsets for thread 0 (compile-time constants, wrapping semantics)
  auto srcOffsets = mlir::emitOffsetForLayout(srcTy.getEncoding(), srcTy);
  auto dstOffsets = mlir::emitOffsetForLayout(dstTy.getEncoding(), dstTy);
  unsigned totalElemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  unsigned dstElemsPerThread   = ttg::getTotalElemsPerThread(dstTy);

  if (srcOffsets.size() != totalElemsPerThread)
    return op.emitError("SMEM path: src offsets size mismatch");
  if (dstOffsets.size() != dstElemsPerThread)
    return op.emitError("SMEM path: dst offsets size mismatch");

  auto dstOrder = getCTATileOrder(dstTy);

  // SMEM layout: order-based strides (matching dst blocked layout access order)
  SmallVector<int64_t> smemStrides(rank, 0);
  {
    int64_t s = 1;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = dstOrder[i];
      smemStrides[dim] = s;
      s *= dstShape[dim];
    }
  }

  // Compute runtime per-thread offsets for src and dst layouts
  auto srcThreadOffsets = computeThreadOffsets(loc, rewriter, srcTy);
  auto dstThreadOffsets = computeThreadOffsets(loc, rewriter, dstTy);

  // ------------------------------------------------------------------
  // Step 1: Allocate SMEM buffer
  // ------------------------------------------------------------------
  auto smemPtrTy = LLVM::LLVMPointerType::get(ctx, 3);
  auto smemArrTy = LLVM::LLVMArrayType::get(
      i8Ty, static_cast<uint64_t>(totalDstElems * elemBytes));
  std::string smemName =
      "__smem_extract_tile_" +
      std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    rewriter.create<LLVM::GlobalOp>(loc, smemArrTy, false,
        LLVM::Linkage::Internal, smemName, Attribute{}, 16, 3);
  }
  Value smemBufArr = rewriter.create<LLVM::AddressOfOp>(loc, smemPtrTy, smemName);
  Value zero32 = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(0));
  Value smemBase = rewriter.create<LLVM::GEPOp>(
      loc, smemPtrTy, smemArrTy, smemBufArr,
      ValueRange{zero32, zero32}, LLVM::GEPNoWrapFlags::inbounds);

  // ------------------------------------------------------------------
  // Step 2: Runtime compute tileStart[d] and tileEnd[d] (global coords)
  //
  // tileStart[d] = ((dynIndex / suffix[d]) % logicalGrid[d]) * tileShape[d]
  // tileEnd[d]   = tileStart[d] + tileShape[d]
  // ------------------------------------------------------------------
  SmallVector<int64_t> logicalGrid(rank), suffix(rank, 1);
  for (int d = 0; d < rank; ++d) logicalGrid[d] = srcShape[d] / tileShape[d];
  for (int d = rank - 2; d >= 0; --d) suffix[d] = suffix[d+1] * logicalGrid[d+1];

  Value dynIndex = adaptor.getIndex();
  SmallVector<Value> tileStartVals(rank), tileEndVals(rank);
  for (int d = 0; d < rank; ++d) {
    Value sv = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)suffix[d]));
    Value gv = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)logicalGrid[d]));
    Value tv = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
    Value coord = rewriter.create<LLVM::UDivOp>(loc, i32Ty, dynIndex, sv);
    coord = rewriter.create<LLVM::URemOp>(loc, i32Ty, coord, gv);
    tileStartVals[d] = rewriter.create<LLVM::MulOp>(loc, i32Ty, coord, tv);
    tileEndVals[d]   = rewriter.create<LLVM::AddOp>(loc, i32Ty, tileStartVals[d], tv);
  }

  // ------------------------------------------------------------------
  // Step 3: Conditionally write src elements to SMEM
  //
  // For each src register element i:
  //   globalCoord[d] = srcOffsets[i][d]   <- thread-0-relative, compile-time
  //                  + srcThreadOffsets[d] <- per-thread delta, runtime
  //
  //   in-range: globalCoord[d] in [tileStart[d], tileEnd[d])  for all d
  //   SMEM byte offset = sum_d( (globalCoord[d] - tileStart[d])
  //                             * smemStrides[d] * elemBytes )
  // ------------------------------------------------------------------
  auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

  for (unsigned i = 0; i < totalElemsPerThread; ++i) {
    Value inRange = rewriter.create<LLVM::ConstantOp>(
        loc, i1Ty, rewriter.getIntegerAttr(i1Ty, 1));
    Value smemByteOffset = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      // Compile-time base offset for element i (thread-0 relative)
      Value baseOff = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)srcOffsets[i][d]));
      // True global coordinate for current thread
      Value globalCoordV = rewriter.create<LLVM::AddOp>(
          loc, i32Ty, baseOff, srcThreadOffsets[d]);

      // in-range check: globalCoord in [tileStart, tileEnd)
      Value ge = rewriter.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::uge, globalCoordV, tileStartVals[d]);
      Value lt = rewriter.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ult, globalCoordV, tileEndVals[d]);
      inRange = rewriter.create<LLVM::AndOp>(loc,
          rewriter.create<LLVM::AndOp>(loc, ge, lt), inRange);

      // Local coordinate within tile: globalCoord - tileStart
      Value localInTile = rewriter.create<LLVM::SubOp>(
          loc, i32Ty, globalCoordV, tileStartVals[d]);
      Value sb = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffset = rewriter.create<LLVM::AddOp>(loc, i32Ty, smemByteOffset,
          rewriter.create<LLVM::MulOp>(loc, i32Ty, localInTile, sb));
    }

    // Conditional write via basic block splitting
    Block *cur   = rewriter.getInsertionBlock();
    Block *then_ = rewriter.splitBlock(cur, rewriter.getInsertionPoint());
    Block *merge = rewriter.splitBlock(then_, then_->begin());

    rewriter.setInsertionPointToEnd(cur);
    rewriter.create<LLVM::CondBrOp>(loc, inRange, then_, merge);

    rewriter.setInsertionPointToStart(then_);
    Value sp = rewriter.create<LLVM::GEPOp>(
        loc, smemPtrTy, i8Ty, smemBase, ValueRange{smemByteOffset},
        LLVM::GEPNoWrapFlags::inbounds);
    rewriter.create<LLVM::StoreOp>(loc, srcVals[i], sp, elemBytes);
    rewriter.create<LLVM::BrOp>(loc, merge);

    rewriter.setInsertionPointToStart(merge);
  }

  // ------------------------------------------------------------------
  // Step 4: __syncthreads() -- ensure all writes are visible
  // ------------------------------------------------------------------
  rewriter.create<NVVM::Barrier0Op>(loc);

  // ------------------------------------------------------------------
  // Step 5: Read dst registers from SMEM (FIXED in v4)
  //
  // For each dst register element i:
  //   globalCoord[d] = dstOffsets[i][d]   <- thread-0-relative, compile-time
  //                  + dstThreadOffsets[d] <- per-thread delta, runtime
  //
  // KEY FIX: When shapePerCTATile[d] > dstShape[d]=tileShape[d], the blocked
  // layout wraps around (multiple threads map to the same tile column).
  // emitOffsetForLayout uses wrapping semantics, but computeThreadOffsets
  // returns absolute coords (no modulo). Mixing them yields:
  //   thread 64: gc[1]=0+64=64  (should be 64%64=0)
  //   → reads smem[256] instead of smem[0]  → wrong value
  //   → overwrites the correct value written by thread 0 in the output buffer
  //
  // Fix: apply modulo tileShape[d] to get the true tile-local coordinate.
  //   tileLocalCoord[d] = (dstOffsets[i][d] + dstThreadOffsets[d]) % tileShape[d]
  //   smemByteOffset    = sum_d( tileLocalCoord[d] * smemStrides[d] * elemBytes )
  // ------------------------------------------------------------------
  SmallVector<Value> dstVals;
  dstVals.reserve(dstElemsPerThread);

  for (unsigned i = 0; i < dstElemsPerThread; ++i) {
    Value smemByteOffsetV = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(0));

    for (int d = 0; d < rank; ++d) {
      // Compile-time base offset for dst element i (thread-0 relative, wrapping)
      Value baseOff = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)dstOffsets[i][d]));
      // Absolute coordinate contribution from current thread (no wrapping)
      Value globalCoordV = rewriter.create<LLVM::AddOp>(
          loc, i32Ty, baseOff, dstThreadOffsets[d]);

      // ★ FIX: convert absolute coord to tile-local coord via modulo.
      //   Needed when shapePerCTATile[d] > tileShape[d]=dstShape[d]:
      //   multiple threads share the same tile-local slot and must all
      //   read from the same SMEM address.
      Value tileShapeV = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)tileShape[d]));
      Value tileLocalCoordV = rewriter.create<LLVM::URemOp>(
          loc, i32Ty, globalCoordV, tileShapeV);

      Value sb = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty,
          rewriter.getI32IntegerAttr((int32_t)(smemStrides[d] * elemBytes)));
      smemByteOffsetV = rewriter.create<LLVM::AddOp>(loc, i32Ty, smemByteOffsetV,
          rewriter.create<LLVM::MulOp>(loc, i32Ty, tileLocalCoordV, sb));
    }

    Value lp = rewriter.create<LLVM::GEPOp>(
        loc, smemPtrTy, i8Ty, smemBase, ValueRange{smemByteOffsetV},
        LLVM::GEPNoWrapFlags::inbounds);
    dstVals.push_back(
        rewriter.create<LLVM::LoadOp>(loc, llvmElemTy, lp, elemBytes));
  }

  // ------------------------------------------------------------------
  // Step 6: __syncthreads() -- allow SMEM reuse after reads complete
  // ------------------------------------------------------------------
  rewriter.create<NVVM::Barrier0Op>(loc);

  // ------------------------------------------------------------------
  // Step 7: Pack result registers
  // ------------------------------------------------------------------
  Value ret = packLLElements(loc, typeConverter, dstVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

// ============================================================================
// Dispatcher (unchanged)
// ============================================================================
struct ExtractTileOpConversion
    : public ConvertOpToLLVMPattern<ExtractTileOp> {
  using ConvertOpToLLVMPattern<ExtractTileOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExtractTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy)
      return op.emitError("extract_tile operands must be ranked tensors");
    if (!srcTy.getEncoding() || !dstTy.getEncoding())
      return op.emitError("extract_tile requires tensors with encoding");
    if (!isa<ttg::BlockedEncodingAttr>(srcTy.getEncoding()))
      return op.emitError("extract_tile only supports BlockedEncodingAttr");

    auto staticIndex = getStaticIndex(op);
    if (staticIndex.has_value() && isCTATileAligned(op, staticIndex.value()))
      return lowerExtractTileStatic(op, adaptor, rewriter,
                                    this->getTypeConverter(), staticIndex.value());
    return lowerExtractTileViaSMEM(op, adaptor, rewriter, this->getTypeConverter());
  }
};

} // anonymous namespace

namespace mlir::triton::tle {
void populateExtractTileOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         unsigned benefit) {
  patterns.add<ExtractTileOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::tle
