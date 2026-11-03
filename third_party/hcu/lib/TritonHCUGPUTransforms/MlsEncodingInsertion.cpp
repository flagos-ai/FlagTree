#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUGPUTransforms/MfmaGroup.h"
#include "TritonHCUGPUTransforms/MlsGroup.h"
#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

SmallVector<unsigned> getMatrixLoadTensorOrder(triton::MatrixLoadOp matrixOp,
                                               int opIdx) {
  auto strides = matrixOp.getStrides();
  assert(strides.size() == 2);

  auto strideKIdx = opIdx == 0 ? 1 : 0;
  auto strideK = strides[strideKIdx];

  bool kMajor = false;
  if (auto constantOp = strideK.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      kMajor = attr.getValue().isOne();
    }
  }

  if (opIdx == 0) {
    return {kMajor, !kMajor};
  } else {
    return {!kMajor, kMajor};
  }
}

// Chooses a proper MLS instruction
FailureOr<MlsInsn> chooseMlsInstruction(tt::DotOp dot, int opIdx,
                                        triton::MatrixLoadOp matrixOp,
                                        bool kMajor, int version,
                                        int numWarps) {
  // get mfma encoding info
  auto aType = dot.getA().getType();
  auto bType = dot.getB().getType();
  auto dType = dot.getResult().getType();
  auto mfmaEncoding = dyn_cast<ttg::AMDMfmaEncodingAttr>(dType.getEncoding());

  // get the mls shape info
  auto rank = matrixOp.getType().getRank();
  auto elemType = matrixOp.getType().getElementType();
  auto bitwidth = elemType.getIntOrFloatBitWidth();
  auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
  auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];

  // Determine the kDim based on the matrix load and numWarps Info.
  unsigned kTile = 0;
  unsigned nonKTile = opIdx == 0 ? mfmaEncoding.getMfmaTile()[0]
                                 : mfmaEncoding.getMfmaTile()[1];
  if (bitwidth == 16) {
    kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
    if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
      kTile = 32;
  } else if (bitwidth == 8) {
    kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
    if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
      kTile = 64;
  }
  auto altKind = MlsInterleaveKind::InterleaveNone;

  // select the mls insn
  auto maybeMlsInsn = MlsInsn::selectOrGetMlsInsn(
      nonKTile, kTile, bitwidth, opIdx, kMajor, altKind, version);
  if (failed(maybeMlsInsn))
    llvm::report_fatal_error("No match found in MLS database\n");

  return maybeMlsInsn;
}

SmallVector<unsigned> warpsPerCTAMatrixLoad(ArrayRef<int64_t> shape,
                                            ArrayRef<unsigned> shapePerWarp,
                                            ArrayRef<unsigned> order,
                                            int numWarps) {
  unsigned rank = shape.size();
  SmallVector<unsigned> warpsPerCTA(rank);

  unsigned remainingWarps = numWarps;
  unsigned prevWarps = 1;

  // starting from the contiguous dimension
  for (unsigned d = 0; d < rank - 1; ++d) {
    unsigned i = order[d];
    unsigned maxWarpsInDim = std::max<unsigned>(
        1, static_cast<unsigned>(shape[i]) / shapePerWarp[i]);
    warpsPerCTA[i] = std::clamp<unsigned>(remainingWarps, 1, maxWarpsInDim);
    remainingWarps /= warpsPerCTA[i];
    prevWarps *= warpsPerCTA[i];
  }

  // Expand the last dimension to fill the remaining lanes and warps
  warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

  return warpsPerCTA;
}

Value convertAndCastTensor(OpBuilder &builder, Value value,
                           Attribute newEncoding) {
  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      builder.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  return convertedTensor;
}


// Resolve only our early one-shot candidates. The generic v1 database must
// never see a rejected candidate (in particular unsupported gfx936 layouts).
struct LegacyV0BlockPtrSelection : OpRewritePattern<triton::MatrixLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::MatrixLoadOp op,
                                PatternRewriter &b) const override {
    if (!op->hasAttr("ttg.hcu.v0_blockptr_candidate"))
      return failure();
    auto mod = op->getParentOfType<ModuleOp>();
    auto attr = [&](StringRef key) -> int64_t {
      auto a = mod->getAttrOfType<IntegerAttr>(key);
      return a ? a.getInt() : 0;
    };
    auto target = mod->getAttrOfType<StringAttr>("ttg.target");
    auto old = op.getType();
    auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(old.getEncoding());
    auto pair = getDotOpIdxFromMatrixLoad(op);
    auto is = [&](StringRef key, int32_t value) {
      auto a = mod->getAttrOfType<IntegerAttr>(key);
      return a && a.getType().isInteger(32) && a.getInt() == value;
    };
    bool indexedCapture = is("ttg.hcu.v0_indexed_a_packet", 1) &&
        is("ttg.hcu.v0_mls_local_prefetch_capture", 1) &&
        is("ttg.hcu.v0_blockptr_loop", 1);
    ttg::DotOperandEncodingAttr dotEnc;
    ttg::AMDMfmaEncodingAttr mma;
    if (succeeded(pair) && (pair->second == 1 ||
                            (indexedCapture && pair->second == 0))) {
      auto operand = pair->second == 0 ? pair->first.getA() : pair->first.getB();
      dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(operand.getType().getEncoding());
      if (dotEnc)
        mma = dyn_cast<ttg::AMDMfmaEncodingAttr>(dotEnc.getParent());
    }
    int64_t w = attr("ttg.num-warps");
    auto policy = op->getAttrOfType<BoolAttr>("tt.load.async");
    bool s25 = w == 4 && old.getShape() == ArrayRef<int64_t>({32, 128});
    bool s26 = w == 8 && old.getShape() == ArrayRef<int64_t>({16, 256});
    // An independent exact alternative. Preserve the entire old KW4/TPW11
    // selection branch; no v1 matrix-load ownership or LP1 layout is widened.
    auto constant = [](Value v, int64_t expected) {
      auto op = v.getDefiningOp<arith::ConstantOp>();
      auto a = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
      return a && a.getValue().isSignedIntN(64) && a.getInt() == expected;
    };
    bool s25Kw8 = s25 && mma && dotEnc && dotEnc.getKWidth() == 8 &&
        attr("ttg.hcu.v0_blockptr_loop") == 1 &&
        attr("ttg.hcu.mmac_tpw_n") == 2 && attr("ttg.hcu.mmac_transposed") == 1 &&
        attr("ttg.hcu.canonical_matrix") == 0 &&
        attr("ttg.hcu.v0_mls_local_prefetch_capture") == 0 &&
        mma.getVersion() == 3 && mma.getElementBitWidth() == 32 &&
        mma.getTilesPerWarp() == ArrayRef<unsigned>({2, 2}) &&
        op.getShape().size() == 2 && op.getStrides().size() == 2 &&
        constant(op.getShape()[0], 1024) && constant(op.getShape()[1], 1024) &&
        constant(op.getStrides()[0], 1024) && constant(op.getStrides()[1], 1) &&
        pair->first.getA().getType().getShape() == ArrayRef<int64_t>({128, 32}) &&
        pair->first.getType().getShape() == ArrayRef<int64_t>({128, 128});
    // Only the indexed capture accepts variable fixed-K source extents.
    // The old KW4 and stage2 KW8 B-only predicates remain unchanged.
    auto indexedSource = [&]() {
      if (!indexedCapture || failed(pair) || pair->second > 1 ||
          op.getShape().size() != 2 || op.getStrides().size() != 2) return false;
      unsigned i = pair->second;
      auto c = op.getShape()[i].getDefiningOp<arith::ConstantOp>();
      auto extent = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
      return extent && extent.getValue().isSignedIntN(64) &&
          extent.getInt() >= 128 && extent.getInt() % 128 == 0 &&
          extent.getInt() < (int64_t(1) << 20) &&
          constant(op.getShape()[1 - i], 1024) &&
          constant(op.getStrides()[0], i == 0 ? 1024 : extent.getInt()) &&
          constant(op.getStrides()[1], 1);
    };
    bool indexedReader = indexedCapture && w == 4 && succeeded(pair) &&
        (pair->second == 0 || pair->second == 1) && mma && dotEnc &&
        dotEnc.getOpIdx() == pair->second && dotEnc.getKWidth() == 8 &&
        mma.getVersion() == 3 && mma.getElementBitWidth() == 32 &&
        mma.getTilesPerWarp() == ArrayRef<unsigned>({2, 2}) &&
        mma.getIsTransposed() && is("ttg.hcu.mmac_tpw_n", 2) &&
        is("ttg.hcu.mmac_transposed", 1) && is("ttg.hcu.canonical_matrix", 0) &&
        indexedSource() &&
        (old.getElementType().isF16() || old.getElementType().isBF16()) &&
        pair->first.getA().getType().getElementType() == old.getElementType() &&
        pair->first.getB().getType().getElementType() == old.getElementType() &&
        pair->first.getA().getType().getShape() == ArrayRef<int64_t>({128, 32}) &&
        pair->first.getB().getType().getShape() == ArrayRef<int64_t>({32, 128}) &&
        pair->first.getType().getShape() == ArrayRef<int64_t>({128, 128}) &&
        pair->first.getType().getElementType().isF32() &&
        old.getShape() == (pair->second == 0 ? ArrayRef<int64_t>({128, 32})
                                           : ArrayRef<int64_t>({32, 128})) &&
        (!op->hasAttr("tt.load.async") || bool(policy)) &&
        op.getBoundaryCheck().empty() && !op.getIsVolatile() &&
        op.getCache() == triton::CacheModifier::NONE &&
        op.getEvict() == triton::EvictionPolicy::NORMAL;
    if (indexedReader) {
      auto base = dyn_cast<BlockArgument>(op.getBase());
      auto ptr = dyn_cast<triton::PointerType>(op.getBase().getType());
      auto f = op->getParentOfType<triton::FuncOp>();
      auto loop = op->getParentOfType<scf::ForOp>();
      indexedReader = base && ptr && f && loop &&
          base.getOwner() == &f.getBody().front() && base.getArgNumber() == pair->second &&
          ptr.getAddressSpace() == 1 && ptr.getPointeeType() == old.getElementType() &&
          constant(loop.getLowerBound(), 0) && constant(loop.getUpperBound(), 32) &&
          constant(loop.getStep(), 1);
      for (StringRef key : {"ttg.hcu.async_slot_war", "ttg.hcu.local_prefetch",
                            "ttg.hcu.v0_swizzled_a_vec8"})
        indexedReader &= !mod->hasAttr(key) || is(key, 0);
    }
    bool oldReader = dotEnc && dotEnc.getKWidth() == 4 && mma &&
        mma.getTilesPerWarp() == ArrayRef<unsigned>({1, 1});
    bool selected = attr("ttg.hcu.v0_blockptr") == 1 && target &&
        target.getValue() == "hip:gfx936" && attr("ttg.num-ctas") == 1 &&
        attr("ttg.threads-per-warp") == 64 && blocked && (s25 || s26 || indexedReader) &&
        !(policy && !policy.getValue()) && mma &&
        ((dotEnc.getOpIdx() == 1 && (oldReader || s25Kw8)) || indexedReader) &&
        mma.getIsTransposed() == (attr("ttg.hcu.mmac_transposed") == 1) &&
        mma.getMmacLayout() == ttg::MmacLayout::LEGACY &&
        mma.getInstrShape() == ArrayRef<unsigned>({16, 16, 16}) &&
        mma.getWarpsPerCTA() == ArrayRef<unsigned>({unsigned(w / 2), 2}) &&
        llvm::all_of(ttg::getCTAsPerCGA(mma), [](unsigned n) { return n == 1; }) &&
        attr("ttg.hcu.alt_matrix") == 0 &&
        (attr("ttg.hcu.canonical_matrix") == 0 ||
         attr("ttg.hcu.canonical_matrix") == 2) &&
        (attr("ttg.hcu.mmac_transposed") == 0 ||
         attr("ttg.hcu.mmac_transposed") == 1);
    if (selected) {
      auto ty = RankedTensorType::get(old.getShape(), old.getElementType(), dotEnc);
      auto result = b.create<triton::MatrixLoadOp>(
          op.getLoc(), ty, op.getBase(), op.getShape(), op.getStrides(),
          op.getTensorShape(), op.getIndices(), op.getBoundaryCheck(),
          op.getCache(), op.getEvict(), op.getIsVolatile());
      result->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
          triton::hcugpu::MlsEncodingAttr::get(op.getContext(), dotEnc.getOpIdx(),
              {unsigned(old.getShape()[0]), unsigned(old.getShape()[1])},
              16, 0, 0, {1, 0}, {unsigned(w / 2), 2}));
      if (policy)
        result->setAttr("tt.load.async", policy);
      b.replaceOp(op, convertAndCastTensor(b, result.getResult(), old.getEncoding()));
      return success();
    }

    // Reconstitute the original full unmasked block-pointer load. No special
    // machine opcode, exotic shared layout, or version1 fatal-error fallback.
    // The compiler-created marker carries an unchanged rank2 tensor. Use its
    // actual distributed encoding for fallback, not the V0-only Blocked guard.
    auto distribution = cast<ttg::DistributedEncodingTrait>(old.getEncoding());
    auto loc = op.getLoc();
    auto ptrTy = RankedTensorType::get(old.getShape(), op.getBase().getType(), distribution);
    auto i64Ty = RankedTensorType::get(old.getShape(), b.getI64Type(), distribution);
    Value ptr = b.create<triton::SplatOp>(loc, ptrTy, op.getBase());
    for (unsigned d = 0; d < 2; ++d) {
      auto slice = ttg::SliceEncodingAttr::get(op.getContext(), 1 - d, distribution);
      auto rangeTy = RankedTensorType::get({old.getShape()[d]}, b.getI32Type(), slice);
      Value range = b.create<triton::MakeRangeOp>(loc, rangeTy, 0, old.getShape()[d]);
      Value origin = b.create<triton::SplatOp>(loc, rangeTy, op.getIndices()[d]);
      Value idx = b.create<arith::AddIOp>(loc, range, origin);
      idx = b.create<triton::ExpandDimsOp>(loc, idx, 1 - d);
      auto idx2Ty = RankedTensorType::get(old.getShape(), b.getI32Type(), distribution);
      idx = b.create<triton::BroadcastOp>(loc, idx2Ty, idx);
      idx = b.create<arith::ExtSIOp>(loc, i64Ty, idx);
      Value stride = b.create<triton::SplatOp>(loc, i64Ty, op.getStrides()[d]);
      Value off = b.create<arith::MulIOp>(loc, idx, stride);
      ptr = b.create<triton::AddPtrOp>(loc, ptrTy, ptr, off);
    }
    auto load = b.create<triton::LoadOp>(loc, ptr, Value{}, Value{}, op.getCache(),
                                        op.getEvict(), op.getIsVolatile());
    if (policy)
      load->setAttr("tt.load.async", policy);
    b.replaceOp(op, load.getResult());
    return success();
  }
};

struct MlsEncodingInsertion : public OpRewritePattern<triton::MatrixLoadOp> {
public:
  MlsEncodingInsertion(MLIRContext *context) : OpRewritePattern(context, 1) {}

  LogicalResult matchAndRewrite(triton::MatrixLoadOp matrixOp,
                                PatternRewriter &rewriter) const override {
    if (matrixOp->hasAttr("ttg.hcu.v0_blockptr_candidate"))
      return failure();
    RankedTensorType oldRetType = matrixOp.getType();
    if (!oldRetType.getEncoding() ||
        !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto maybeDotOpIdxPair = getDotOpIdxFromMatrixLoad(matrixOp);
    if (succeeded(maybeDotOpIdxPair)) {
      auto dotOp = maybeDotOpIdxPair.value().first;
      auto opIdx = maybeDotOpIdxPair.value().second;

      // 1. choose the mls instruction for matrixOp
      assert(isa<ttg::AMDMfmaEncodingAttr>(dotOp.getType().getEncoding()));
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);
      auto mlsInsn =
          chooseMlsInstruction(dotOp, opIdx, matrixOp, kMajor, 1, numWarps);

      // 2. warps per tile calculate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto altKind = mlsInsn->getAlt2Kind();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. create new matrix load op with new encoding and mls attr
      auto newEncoding = opIdx == 0 ? dotOp.getA().getType().getEncoding()
                                    : dotOp.getB().getType().getEncoding();
      if (dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getKWidth() !=
          mlsInsn->getDotLayoutKWidth()) {
        newEncoding = ttg::DotOperandEncodingAttr::get(
            matrixOp.getContext(), opIdx,
            dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getParent(),
            mlsInsn->getDotLayoutKWidth());
      }

      auto oldResultType =
          cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType =
          RankedTensorType::get(oldResultType.getShape(),
                                oldResultType.getElementType(), newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
          matrixOp.getLoc(), newResultType, matrixOp.getBase(),
          matrixOp.getShape(), matrixOp.getStrides(), matrixOp.getTensorShape(),
          matrixOp.getIndices(), matrixOp.getBoundaryCheck(),
          matrixOp.getCache(), matrixOp.getEvict(), matrixOp.getIsVolatile());
      auto mlsEncoding = triton::hcugpu::MlsEncodingAttr::get(
          matrixOp.getContext(), opIdx, mlsTile, elemBitWidth, altKind, version,
          order, warpsPerCTA);
      newMatrixOp->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
                           mlsEncoding);

      // 4. replace the original op
      auto convertedTensor = convertAndCastTensor(
          rewriter, newMatrixOp.getResult(), matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    } else { // non-dot case, try to construct a mfma encoding for the matrix
             // load
      // 1. choose the mls instruction for matrixOp
      unsigned opIdx = 0;
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;

      auto rank = matrixOp.getType().getRank();
      auto elemType = matrixOp.getType().getElementType();
      auto bitwidth = elemType.getIntOrFloatBitWidth();
      auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
      auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);

      unsigned kTile = 0;
      unsigned nonKTile = 0;
      if (bitwidth == 16) {
        nonKTile = kMajor ? 16 : (blockNonK >= 64 ? 64 : 32);
        kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
        if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
          kTile = 32;
      } else if (bitwidth == 8) {
        nonKTile = kMajor ? 16 : (blockNonK >= 128 ? 128 : 64);
        kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
        if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
          kTile = 64;
      }

      auto altKind = MlsInterleaveKind::InterleaveNone;
      auto mlsInsn = MlsInsn::selectOrGetMlsInsn(nonKTile, kTile, bitwidth,
                                                 opIdx, kMajor, altKind, 1);
      if (failed(mlsInsn))
        llvm::report_fatal_error("No match found in MLS database\n");

      assert(blockK % kTile == 0 && blockNonK % nonKTile == 0 &&
             "M or N should be divisible by mDim or nDim");

      // 3. warps per tile calucate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. try to construct a mfma encoding for the matrix load
      constexpr unsigned mfmaVersion = 3;
      auto maybeMfmaInsn = MfmaIntrinsic::selectFor(
          matrixOp.getLoc(), mfmaVersion, 16, 16, blockK, elemType, elemType,
          false, false, HCUISAFeature::MAMC_FP8);
      if (failed(maybeMfmaInsn))
        llvm::report_fatal_error("No match found in MFMA database\n");

      SmallVector<unsigned> tilesPerWarp = {1, 1};
      SmallVector<unsigned> mfmaTiles = {16, 16};
      SmallVector<unsigned> mfmaOrder = {1, 0};
      if (rank == 3) {
        tilesPerWarp.insert(tilesPerWarp.begin(), 1);
        mfmaOrder.insert(mfmaOrder.begin(), 2);
        mfmaTiles.insert(mfmaTiles.begin(), 1);
      }

      auto tileM = nonKTile;
      auto tileN = kTile;

      bool hasBatchDim = rank == 3;
      int mIndex = 0 + hasBatchDim;
      int nIndex = 1 + hasBatchDim;
      tilesPerWarp[mIndex] = tileM / maybeMfmaInsn->mDim;
      tilesPerWarp[nIndex] = tileN / maybeMfmaInsn->nDim;

      SmallVector<unsigned> warpsPerCTAMfma =
          warpsPerCTAMatrixLoad(shape, mfmaTiles, mfmaOrder, numWarps);
      unsigned mfmaElementBitWidth = elemType.isF64() ? 64 : 32;
      auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
          matrixOp.getContext(),
          /*versionMajor*/ mfmaVersion, warpsPerCTAMfma,
          /*instrShape=*/
          SmallVector<unsigned>{maybeMfmaInsn->mDim, maybeMfmaInsn->nDim,
                                maybeMfmaInsn->kDim},
          /*isTransposed=*/false,
          ttg::getCTALayout(matrixOp.getType().getEncoding()), tilesPerWarp,
          mfmaElementBitWidth, ttg::MmacLayout::INTERLEAVE_TRANSPOSE);
      auto dotAEncoding = ttg::DotOperandEncodingAttr::get(
          matrixOp.getContext(), 0, mfmaEnc, maybeMfmaInsn->kBase);

      // 4. create new matrix load op with new encoding and mls attr
      auto newEncoding = dotAEncoding;

      auto oldResultType =
          cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType =
          RankedTensorType::get(oldResultType.getShape(),
                                oldResultType.getElementType(), newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
          matrixOp.getLoc(), newResultType, matrixOp.getBase(),
          matrixOp.getShape(), matrixOp.getStrides(), matrixOp.getTensorShape(),
          matrixOp.getIndices(), matrixOp.getBoundaryCheck(),
          matrixOp.getCache(), matrixOp.getEvict(), matrixOp.getIsVolatile());
      auto mlsEncoding = triton::hcugpu::MlsEncodingAttr::get(
          matrixOp.getContext(), opIdx, mlsTile, elemBitWidth,
          static_cast<unsigned>(altKind), version, order, warpsPerCTA);
      newMatrixOp->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
                           mlsEncoding);

      // 5. replace the original op
      auto convertedTensor = convertAndCastTensor(
          rewriter, newMatrixOp.getResult(), matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUMLSENCODINGINSERTION
#include "TritonHCUGPUTransforms/Passes.h.inc"

class TritonHCUGPUMlsEncodingInsertionPass
    : public impl::TritonHCUGPUMlsEncodingInsertionBase<
          TritonHCUGPUMlsEncodingInsertionPass> {
public:
  using impl::TritonHCUGPUMlsEncodingInsertionBase<
      TritonHCUGPUMlsEncodingInsertionPass>::
      TritonHCUGPUMlsEncodingInsertionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::RewritePatternSet patterns(context);
    // Resolve provisional V0 before the generic version1-only selector.
    patterns.add<LegacyV0BlockPtrSelection>(context, PatternBenefit(2));
    patterns.add<MlsEncodingInsertion>(context);
    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
