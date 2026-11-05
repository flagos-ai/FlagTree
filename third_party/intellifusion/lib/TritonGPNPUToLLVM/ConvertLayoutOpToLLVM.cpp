#include "LoweringUtility.h"
#include "PatternPopulation.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <string>
#include <vector>

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

constexpr int kPtrBitWidth = 64;

// Emit stmatrix as inline PTX because IFWA cannot select the corresponding
// NVVM intrinsic. Keep upstream layout and address calculations.
LogicalResult lowerLdStMatrix(
    Location loc, LinearLayout cvt, bool transpose,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
    Type llvmElemTy, ConversionPatternRewriter &rewriter) {
  bool isStore = !vals.empty();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kOffset = S("offset");
  auto kAddr = S("addr");
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  if ((!transpose && bitwidth > 32) ||
      (transpose && !(bitwidth == 16 || bitwidth == 8)))
    return failure();

  LinearLayout fullTile;
  LinearLayout tile;
  ColumnAction permLanes, permReg;
  ColumnAction accPermReg =
      ColumnAction::identity(kReg, cvt.getInDimSizeLog2(kReg));
  if (!transpose) {
    tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
           LinearLayout::identity1D(4, kLane, kOffset);
    fullTile = tile * LinearLayout::identity1D(8, kLane, kAddr);
  } else {
    auto contigRegs = (isStore && bitwidth == 8 ? 16 : 32) / bitwidth;
    fullTile = LinearLayout::identity1D(contigRegs, kReg, kAddr) *
               LinearLayout::identity1D(4, kLane, kAddr) *
               LinearLayout::identity1D(8, kLane, kOffset) *
               LinearLayout::identity1D(16 / bitwidth, kReg, kOffset);
    if (cvt.getInDimSize(kReg) < fullTile.getInDimSize(kReg))
      return failure();

    std::vector<size_t> regBases, laneBases;
    auto bases = fullTile.invert().getBases().lookup(kOffset);
    for (const auto &basis : bases) {
      assert(basis.size() == 2);
      if (basis[0] != 0) {
        regBases.push_back(llvm::Log2_32(basis[0]));
      } else {
        laneBases.push_back(llvm::Log2_32(basis[1]));
      }
    }
    for (int i = 0; i < cvt.getInDimSizeLog2(kReg); i++) {
      if (!llvm::is_contained(regBases, i))
        regBases.push_back(i);
    }
    for (int i = 0; i < cvt.getInDimSizeLog2(kLane); i++) {
      if (!llvm::is_contained(laneBases, i))
        laneBases.push_back(i);
    }
    assert(laneBases == std::vector<size_t>({2, 3, 4, 0, 1}));
    permReg = ColumnAction(regBases, kReg, cvt.getInDimSizeLog2(kReg));
    permLanes = ColumnAction(laneBases, kLane, cvt.getInDimSizeLog2(kLane));
    cvt = permReg.apply(cvt);
    cvt = permLanes.apply(cvt);
    if (isStore) {
      vals = permReg.apply(vals);
    } else {
      accPermReg = accPermReg.leftCompose(permReg);
    }

    tile = (LinearLayout::identity1D(8, kLane, kOffset) *
            LinearLayout::identity1D(16 / bitwidth, kReg, kOffset))
               .transposeIns({kReg, kLane});
  }

  ColumnAction permDivide;
  if (auto maybePermutation = regPermForDivide(cvt, tile, /*left=*/true)) {
    permDivide = maybePermutation.value();
  } else {
    return failure();
  }

  cvt = permDivide.apply(cvt);
  if (isStore) {
    vals = permDivide.apply(vals);
  } else {
    accPermReg = accPermReg.leftCompose(permDivide);
  }
  auto maybeQuot = divideLeft(cvt, tile);
  if (!maybeQuot.has_value())
    return failure();

  auto reps = zerosLike(tile) * maybeQuot.value();

  if (transpose) {
    reps = permLanes.inverse().apply(reps);
    reps = permReg.inverse().apply(reps);
    if (isStore) {
      vals = permReg.inverse().apply(vals);
    } else {
      accPermReg = accPermReg.leftCompose(permReg.inverse());
    }
  }
  auto regsPerCoreTile = fullTile.getInDimSize(kReg);
  assert(regsPerCoreTile * bitwidth ==
         ((!isStore && bitwidth == 8 && transpose) ? 64 : 32));

  if (maskSpanAffineOffset & (tile.getOutDimSizeLog2(kOffset) - 1))
    return failure();

  auto vec = std::min<int32_t>(128 / bitwidth, reps.getInDimSize(kReg)) /
             regsPerCoreTile;
  assert(vec == 1 || vec == 2 || vec == 4);
  auto fullTileVec = fullTile * LinearLayout::identity1D(vec, kReg, kAddr);
  fullTileVec *= LinearLayout::identity1D(1, kWarp, kAddr);
  auto addrToOffset = fullTileVec.invert().compose(reps);
  assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
         addrToOffset.getInDimSizeLog2(kAddr) <= 5);

  LinearLayout addrLayout =
      LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);
  auto [nAdditive, permStrides] = actionAdditiveStrides(
      reps, addrLayout, maskSpanAffineOffset, fullTileVec.getInDimSize(kReg));
  reps = permStrides.apply(reps);
  if (isStore) {
    vals = permStrides.apply(vals);
  } else {
    accPermReg = accPermReg.leftCompose(permStrides);
  }

  assert(bitwidth >= 8);
  auto i8Tile =
      zerosLike(LinearLayout::identity1D(bitwidth / 8, kReg, kOffset));
  auto i8AddrLayout = i8Tile * addrLayout;

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  auto regBase =
      applyLinearLayout(
          loc, rewriter, i8AddrLayout,
          {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
          .second;

  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitwidth / 8));
  regBase = b.xor_(regBase, affineOffsetI8);

  auto layout = transpose ? NVVM::MMALayout::col : NVVM::MMALayout::row;
  auto eltType = transpose && bitwidth == 8 ? NVVM::LdStMatrixEltType::B8
                                            : NVVM::LdStMatrixEltType::B16;
  int m = fullTile.getOutDimSize(kAddr);
  int n = fullTile.getOutDimSize(kOffset) * bitwidth /
          (eltType == NVVM::LdStMatrixEltType::B8 ? 8 : 16);
  if (transpose)
    std::swap(m, n);
  auto shape = NVVM::LdStMatrixShapeAttr::get(ctx, m, n);

  auto elemsPerInstr = fullTileVec.getInDimSize(kReg);
  auto elemsPerVec = 32 / bitwidth;
  auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
  for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    auto regIdxI8 = regIdx * (bitwidth / 8);
    Value offset = b.xor_(regBase, b.i32_val(regIdxI8));
    for (int i2 = 0; i2 < nAdditive; i2 += elemsPerInstr) {
      auto regIdxAdd =
          reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxAddI8 = regIdxAdd * (bitwidth / 8);
      Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
      auto vecAddr = b.gep(smemPtrTy, i8_ty, smemBase, innerOffset,
                           LLVM::GEPNoWrapFlags::inbounds);
      if (isStore) {
        SmallVector<Value> inputs;
        for (int j = 0; j < elemsPerInstr; j += elemsPerVec) {
          Value input = b.undef(vecTy);
          for (int k = 0; k < elemsPerVec; k++) {
            input = b.insert_element(vecTy, input, vals[i + i2 + j + k],
                                     b.i32_val(k));
          }
          inputs.push_back(b.bitcast(input, i32_ty));
        }
        std::string asmString = "stmatrix.sync.aligned.m" + std::to_string(m) +
                                "n" + std::to_string(n) + ".x" +
                                std::to_string(vec);
        if (transpose)
          asmString += ".trans";
        asmString += eltType == NVVM::LdStMatrixEltType::B8 ? ".shared.b8"
                                                            : ".shared.b16";
        asmString += " [$0], {";
        std::string constraints = "r";
        for (unsigned j = 0; j < inputs.size(); ++j) {
          if (j)
            asmString += ", ";
          asmString += "$" + std::to_string(j + 1);
          constraints += ",r";
        }
        asmString += "};";
        constraints += ",~{memory}";
        SmallVector<Value> asmOperands{vecAddr};
        llvm::append_range(asmOperands, inputs);
        LLVM::InlineAsmOp::create(
            rewriter, loc, void_ty(ctx), asmOperands, asmString, constraints,
            /*has_side_effects=*/true, /*is_align_stack=*/false,
            LLVM::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
      } else {
        unsigned numLdMatrix = elemsPerInstr / elemsPerVec;
        assert(numLdMatrix > 0 &&
               "ldmatrix must load at least one 8x8 tile per instruction");
        Type ldResultTy =
            elemsPerInstr == elemsPerVec
                ? i32_ty
                : static_cast<Type>(LLVM::LLVMStructType::getLiteral(
                      ctx, SmallVector<Type>(numLdMatrix, i32_ty)));
        auto res = NVVM::LdMatrixOp::create(rewriter, loc, ldResultTy, vecAddr,
                                            vec, layout, shape, eltType)
                       .getResult();
        for (int j = 0; j < elemsPerInstr / elemsPerVec; j++) {
          Value output = elemsPerInstr == elemsPerVec
                             ? res
                             : b.extract_val(i32_ty, res, j);
          output = b.bitcast(output, vecTy);
          for (int k = 0; k < elemsPerVec; k++)
            vals.push_back(b.extract_element(llvmElemTy, output, b.i32_val(k)));
        }
      }
    }
  }
  if (!isStore) {
    assert(vals.size() == cvt.getInDimSize(kReg));
    vals = accPermReg.inverse().apply(vals);
  }
  return success();
}

// Use IFWA shared-memory access and synchronization hooks for swizzling.
struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  explicit ConvertLayoutOpSwizzlingConversion(LLVMTypeConverter &typeConverter,
                                              const TargetInfoBase &targetInfo,
                                              PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    if (!cvtAlwaysUseWarpShuffle(op) && cvtNeedsSharedMemory(srcTy, dstTy)) {
      auto loc = op.getLoc();

      auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto outVals = transferWithinBlockSwizzling(loc, rewriter, srcLayout,
                                                  dstLayout, inVals, llvmElemTy,
                                                  smemBase, op.getOperation());

      Value result =
          packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase,
      Operation *sourceOp) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers.
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, llvmElemTyPtr,
          smemBase, sourceOp);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1.
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals =
          transferWithinBlockSwizzling(loc, rewriter, srcLayout, dstLayout,
                                       newInVals, i8ElemTy, smemBase, sourceOp);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
      }
      return outVals;
    }

    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzling(loc, rewriter, prmtSrc, dstLayout,
                                          newInVals, llvmElemTy, smemBase,
                                          sourceOp);
    }

    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals =
          transferWithinBlockSwizzling(loc, rewriter, srcLayout, prmtDst,
                                       inVals, llvmElemTy, smemBase, sourceOp);
      return broadcastAs(outVals, dstLayout);
    }

    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto kBlock = str_attr("block");
    auto [smem, instr] = Intellifusion::getLayoutConversionSwizzling(
        srcLayout, dstLayout, bitwidth, targetInfo);
    auto [idxSrc, idxDst] = instr;

    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = invertAndComposeBlockLocal(smem, dstLayout);

    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    auto nBlock = storeCvt.getInDimSize(kBlock);
    storeCvt = storeCvt.reshapeOuts(
        {{kOffset, storeCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});
    loadCvt = loadCvt.reshapeOuts(
        {{kOffset, loadCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});

    assert(storeCvt.isTrivialOver({kBlock}));
    assert(loadCvt.isTrivialOver({kBlock}) || idxDst == 0);

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    bool isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
    bool isBlockSync = mlir::isCvtDimSync(srcLayout, dstLayout, kBlock);
    auto emitBarrier = [&]() {
      if (isWarpSync) {
        targetInfo.warpSync(loc, rewriter);
      } else if (isBlockSync) {
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      } else {
        targetInfo.clusterBarrier(loc, rewriter, sourceOp);
      }
    };
    auto dropBlock = [&](const LinearLayout &cvt) {
      SmallVector<StringAttr> inDims = {kReg, kLane, kWarp};
      SmallVector<StringAttr> outDims = {kOffset};
      return cvt.sublayout(inDims, outDims);
    };
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        emitBarrier();
      auto tileInVals =
          to_vector(ArrayRef(permutedInVals).slice(i * tileSize, tileSize));
      // idxSrc 0: st.shared, idxSrc 1: stmatrix, idxSrc 2: stmatrix.trans.
      if (idxSrc == 0) {
        lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                        /*paddingShifts=*/{}, affineOffset,
                        maskSpanAffineOffset, rewriter, targetInfo);
      } else {
        assert(idxSrc == 1 || idxSrc == 2);
        bool transpose = idxSrc == 2;
        auto storeCvtNoBlock = dropBlock(storeCvt);
        auto result = lowerLdStMatrix(
            loc, storeCvtNoBlock, transpose, tileInVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter);
        assert(succeeded(result));
      }
      emitBarrier();

      SmallVector<Value> tileOutVals;
      // idxDst 0: ld.shared, idxDst 1: ldmatrix, idxDst 2: ldmatrix.trans.
      if (idxDst == 0) {
        tileOutVals = lowerLdStShared(
            loc, ctx, loadCvt, {}, llvmElemTy, smemBase, /*paddingShifts=*/{},
            affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      } else {
        assert(idxDst == 1 || idxDst == 2);
        bool transpose = idxDst == 2;
        auto loadCvtNoBlock = dropBlock(loadCvt);
        auto result = lowerLdStMatrix(
            loc, loadCvtNoBlock, transpose, tileOutVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter);
        assert(succeeded(result));
      }
      llvm::append_range(outVals, tileOutVals);
    }

    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }
};

} // namespace

void mlir::triton::Intellifusion::populateGPNPUConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpSwizzlingConversion>(typeConverter, targetInfo,
                                                   benefit.getBenefit() + 1);
}
