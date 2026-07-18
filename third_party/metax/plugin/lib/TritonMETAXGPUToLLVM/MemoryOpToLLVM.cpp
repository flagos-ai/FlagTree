#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

#define TT_MODE 2
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

/**
 * Lds and perm for bf16/fp16 tensor
 * For example, consider dotOprand B with shape = [K, N] (rowmajor)
 * the smem of B is of shape [K, N] and order [1, 0]
 * load and perm a little tile with shape [2, elemsN] at a time
 * so the LinearLayout of permTile has bases
 * {"register":{{N},{1},{2}...{maxPowerOf2SmallerThan(elemsN)}}} if B is of
 * shape [64, 128], and elemsN = 4, then bases = {"register":{{128}, {1}, {2}}}
 */
SmallVector<Value>
lowerLocalLdPerm(Location loc, MLIRContext *ctx,
                 LinearLayout cvt, // Map from registers to offset
                 Type llvmElemTy, triton::gpu::MemDescType srcTy,
                 SharedMemoryObject smemObj, RewriterBase &rewriter,
                 const TargetInfoBase &targetInfo, Operation *localLoadOp,
                 int elemsPerVec) {

  assert(cvt.getNumOutDims() == 1);
  assert(*cvt.getOutDimNames().begin() == str_attr("offset"));
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
  auto smemPtrTy = ptr_ty(ctx, 3);
  SmallVector<Value> outVals(cvt.getInDimSize(kReg));

  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);

  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    outVals = lowerLocalLdPerm(loc, ctx, prmtCvt, llvmElemTy, srcTy, smemObj,
                               rewriter, targetInfo, localLoadOp, elemsPerVec);
    outVals = broadcastAs(outVals, cvt);
    return outVals;
  }

  while (elemsPerVec * bitwidth > 128) {
    elemsPerVec /= 2;
  }

  int ord = getOrder(srcTy)[0];
  int smemNumCols = srcTy.getShape()[ord];
  // create LL permTile = [2, elemVec]
  // because permTile is not surjective for, so create by define and concat the
  // base
  auto elemsVec1D = LinearLayout::identity1D(elemsPerVec, kReg, kOffset);
  std::vector<std::vector<int32_t>> permBases = {{smemNumCols}};
  auto outSize = llvm::NextPowerOf2(smemNumCols);
  for (auto &basis : elemsVec1D.getBases().lookup(kReg)) {
    permBases.push_back(basis);
    outSize = std::max<int32_t>(outSize, llvm::NextPowerOf2(basis[0]));
  }
  auto permTile =
      LinearLayout({{kReg, std::move(permBases)}}, {{kOffset, outSize}},
                   /*requireSurjective*/ false);

  auto maybePermutation = regPermForDivide(cvt, permTile, /*left=*/true);
  // fall back to default localLoad
  if (!maybePermutation.has_value()) {
    return lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, srcTy, smemObj,
                          rewriter, targetInfo, localLoadOp);
  }
  LinearLayout permCvt = maybePermutation.value().apply(cvt);
  // divideLeft can't handle this situation
  // so replace permTile with zerosLikeTile by change basis in permCvt directly
  // auto maybeQuot = divideLeft(permCvt, permTile);
  // if (!maybeQuot.has_value()) {
  //   return lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, srcTy,
  //                         smemObj, rewriter, targetInfo, localLoadOp);
  // }
  // LinearLayout reps = zerosLike(permTile) * maybeQuot.value();

  bool maybeQuot = true;
  auto bases = permCvt.getBases();
  auto &regBases = bases[kReg];
  auto tileBases = permTile.getBases().lookup(kReg);
  for (int r = 0; r < permTile.getInDimSizeLog2(kReg); r++) {
    if (regBases[r] != tileBases[r]) {
      maybeQuot = false;
      break;
    }
    regBases[r] = {0};
  }
  // check other basis,
  // don't have same outdim bit with permTile basis (don't need xor)
  for (auto dim : cvt.getInDimNames()) {
    for (auto basis : bases[dim]) {
      for (auto tileBasis : tileBases) {
        if (basis[0] & tileBasis[0]) {
          maybeQuot = false;
          break;
        }
      }
    }
  }
  // fall back to default localLoad
  if (!maybeQuot) {
    return lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, srcTy, smemObj,
                          rewriter, targetInfo, localLoadOp);
  }
  LinearLayout reps = LinearLayout(bases, permCvt.getOutDims(), false);

  // additive tile
  LinearLayout addrLayout =
      LinearLayout({{kLane, reps.getBases().lookup(kLane)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   reps.getOutDims(), false);
  auto [nAdditive, permStrides] =
      actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset);
  reps = permStrides.apply(reps);

  auto i8Tile =
      zerosLike(LinearLayout::identity1D(bitwidth / 8, kReg, kOffset));
  auto i8AddrLayout = i8Tile * addrLayout;

  // lane warp i8 base
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  auto regBaseI8 =
      applyLinearLayout(
          loc, rewriter, i8AddrLayout,
          {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
          .second;

  // It's fine that we don't compute the offset in bytes as affineOffset
  // will be folded into a constant
  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitwidth / 8));
  // xor for add base offset
  regBaseI8 = b.xor_(regBaseI8, affineOffsetI8);

  auto smemBase = smemObj.getBase();

  for (int i = 0; i < permCvt.getInDimSize(kReg); i += nAdditive) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    auto regIdxI8 = regIdx * (bitwidth / 8);
    Value offset = b.xor_(regBaseI8, b.i32_val(regIdxI8));

    for (int j = 0; j < nAdditive; j += 2 * elemsPerVec) {
      // firstKVec = load <elemsPerVec/2 * i32> （k==0）
      // secondKVec = load <elemsPerVec/2 * i32> （k==1)
      // for _ from 0 to elemsPerVec/2:
      //   k01_n0 = perm(firstKVec, secondKVec, front)
      //   k01_n1 = perm(firstKVec, secondKVec, back)
      //   insert_to_outVals
      auto firstRegIdxAdd =
          reps.apply({{kReg, j}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto firstRegIdxAddI8 = firstRegIdxAdd * (bitwidth / 8);
      Value firstInnerOffset = b.add(offset, b.i32_val(firstRegIdxAddI8));
      auto firstVecAddr = b.gep(smemPtrTy, i8_ty, smemBase, firstInnerOffset,
                                LLVM::GEPNoWrapFlags::inbounds);

      auto secondRegIdxAdd = permTile.apply({{kReg, 1}})[0].second;
      auto secondRegIdxAddI8 = secondRegIdxAdd * (bitwidth / 8);
      Value secondInnerOffset =
          b.add(firstInnerOffset, b.i32_val(secondRegIdxAddI8));
      auto secondVecAddr = b.gep(smemPtrTy, i8_ty, smemBase, secondInnerOffset,
                                 LLVM::GEPNoWrapFlags::inbounds);

      // emit lds
      Type i32VecTy = vec_ty(i32_ty, elemsPerVec / 2);
      Type elemV2Ty = vec_ty(llvmElemTy, 2);
      // Value firstKVec = undef(smemPtrTy);
      // Value secondKVec = undef(smemPtrTy);
      auto firstKVec =
          b.load(i32VecTy, firstVecAddr, /*align=*/elemsPerVec * bitwidth / 8);
      auto secondKVec =
          b.load(i32VecTy, secondVecAddr, /*align=*/elemsPerVec * bitwidth / 8);

      auto emit_perm_builtin = [&](Value firstVec, Value secondVec, int i32Idx,
                                   bool is_back) -> Value {
        std::string intrinsicPermName = "llvm.mxc.byte.perm";
        StringRef permName(intrinsicPermName);

        Value offsetFront = b.i32_val(0x01000504);
        Value offsetBack = b.i32_val(0x03020706);

        SmallVector<Value> inputs(3);
        if (is_back) {
          inputs[2] = offsetBack;
        } else {
          inputs[2] = offsetFront;
        }
        inputs[0] = b.extract_element(firstVec, b.i32_val(i32Idx));
        inputs[1] = b.extract_element(secondVec, b.i32_val(i32Idx));
        ValueRange permValueRange(inputs);
        Value res = b.undef(i32_ty);
        res = mlir::LLVM::createBuiltinFunc(rewriter, loc, localLoadOp,
                                            permName, i32_ty, permValueRange);
        return res;
      };

      for (int vec2idx = 0; vec2idx < elemsPerVec / 2; ++vec2idx) {
        auto permValueI32Front =
            emit_perm_builtin(firstKVec, secondKVec, vec2idx, /*is_back*/ 0);
        auto permValueI32Back =
            emit_perm_builtin(firstKVec, secondKVec, vec2idx, /*is_back*/ 1);
        auto halfValuex2Front = b.bitcast(permValueI32Front, elemV2Ty);
        auto halfValuex2Back = b.bitcast(permValueI32Back, elemV2Ty);

        outVals[i + j + (vec2idx * 2) * 2] =
            b.extract_element(halfValuex2Front, b.i32_val(0));
        outVals[i + j + (vec2idx * 2) * 2 + 1] =
            b.extract_element(halfValuex2Front, b.i32_val(1));

        outVals[i + j + (vec2idx * 2 + 1) * 2] =
            b.extract_element(halfValuex2Back, b.i32_val(0));
        outVals[i + j + (vec2idx * 2 + 1) * 2 + 1] =
            b.extract_element(halfValuex2Back, b.i32_val(1));
      }
    }
  }

  auto invPermStrides = permStrides.inverse();
  outVals = invPermStrides.apply(outVals);
  if (maybePermutation.has_value()) {
    auto invPerm = maybePermutation.value().inverse();
    outVals = invPerm.apply(outVals);
  }

  return outVals;
}

/**
 * Lds for bf16/fp16 tensor and currently only support pipelineTT.
 * after lds, perm operations will be performed on permOp.
 * TODO():When supporting gluon in the future,
 * it is necessary to check whether the usage point of the operation is permOp,
 * otherwise it will cause the case to throw an exception.
 */
SmallVector<Value>
lowerLocalLdswithoutPerm(Location loc, MLIRContext *ctx,
                         LinearLayout cvt, // Map from registers to offset
                         Type llvmElemTy, triton::gpu::MemDescType srcTy,
                         SharedMemoryObject smemObj, RewriterBase &rewriter,
                         const TargetInfoBase &targetInfo,
                         Operation *localLoadOp, int elemsPerVec) {

  assert(cvt.getNumOutDims() == 1);
  assert(*cvt.getOutDimNames().begin() == str_attr("offset"));
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
  auto smemPtrTy = ptr_ty(ctx, 3);
  SmallVector<Value> outVals(cvt.getInDimSize(kReg));

  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);

  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    outVals = lowerLocalLdswithoutPerm(loc, ctx, prmtCvt, llvmElemTy, srcTy,
                                       smemObj, rewriter, targetInfo,
                                       localLoadOp, elemsPerVec);
    outVals = broadcastAs(outVals, cvt);
    return outVals;
  }

  while (elemsPerVec * bitwidth > 128) {
    elemsPerVec /= 2;
  }

  int ord = getOrder(srcTy)[0];
  int smemNumCols = srcTy.getShape()[ord];
  // create LL permTile = [2, elemVec]
  // because permTile is not surjective for, so create by define and concat the
  // base
  auto elemsVec1D = LinearLayout::identity1D(elemsPerVec, kReg, kOffset);
  std::vector<std::vector<int32_t>> permBases = {{smemNumCols}};
  auto outSize = llvm::NextPowerOf2(smemNumCols);
  for (auto &basis : elemsVec1D.getBases().lookup(kReg)) {
    permBases.push_back(basis);
    outSize = std::max<int32_t>(outSize, llvm::NextPowerOf2(basis[0]));
  }

  auto permTile =
      LinearLayout({{kReg, std::move(permBases)}}, {{kOffset, outSize}},
                   /*requireSurjective*/ false);

  auto maybePermutation = regPermForDivide(cvt, permTile, /*left=*/true);
  // fall back to default localLoad
  if (!maybePermutation.has_value()) {
    return lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, srcTy, smemObj,
                          rewriter, targetInfo, localLoadOp);
  }

  LinearLayout permCvt = maybePermutation.value().apply(cvt);
  // divideLeft can't handle this situation
  // so replace permTile with zerosLikeTile by change basis in permCvt directly
  bool maybeQuot = true;
  auto bases = permCvt.getBases();
  auto &regBases = bases[kReg];
  auto tileBases = permTile.getBases().lookup(kReg);
  for (int r = 0; r < permTile.getInDimSizeLog2(kReg); r++) {
    if (regBases[r] != tileBases[r]) {
      maybeQuot = false;
      break;
    }
    regBases[r] = {0};
  }
  // check other basis,
  // don't have same outdim bit with permTile basis (don't need xor)
  for (auto dim : cvt.getInDimNames()) {
    for (auto basis : bases[dim]) {
      for (auto tileBasis : tileBases) {
        if (basis[0] & tileBasis[0]) {
          maybeQuot = false;
          break;
        }
      }
    }
  }
  // fall back to default localLoad
  if (!maybeQuot) {
    return lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, srcTy, smemObj,
                          rewriter, targetInfo, localLoadOp);
  }
  LinearLayout reps = LinearLayout(bases, permCvt.getOutDims(), false);

  // additive tile
  LinearLayout addrLayout =
      LinearLayout({{kLane, reps.getBases().lookup(kLane)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   reps.getOutDims(), false);
  auto [nAdditive, permStrides] =
      actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset);
  reps = permStrides.apply(reps);

  if (!permStrides.isIdentity()) {
    assert(false &&
           "Currently does not support changing the order of registers");
  }

  auto i8Tile =
      zerosLike(LinearLayout::identity1D(bitwidth / 8, kReg, kOffset));
  auto i8AddrLayout = i8Tile * addrLayout;

  // lane warp i8 base
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  auto regBaseI8 =
      applyLinearLayout(
          loc, rewriter, i8AddrLayout,
          {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
          .second;

  // It's fine that we don't compute the offset in bytes as affineOffset
  // will be folded into a constant
  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitwidth / 8));
  // xor for add base offset
  regBaseI8 = b.xor_(regBaseI8, affineOffsetI8);

  auto smemBase = smemObj.getBase();

  for (int i = 0; i < permCvt.getInDimSize(kReg); i += nAdditive) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    auto regIdxI8 = regIdx * (bitwidth / 8);
    Value offset = b.xor_(regBaseI8, b.i32_val(regIdxI8));

    for (int j = 0; j < nAdditive / 2; j += 2 * elemsPerVec) {
      // firstKVec = load <elemsPerVec/2 * i32> （k==0）
      // secondKVec = load <elemsPerVec/2 * i32> （k==1)
      // firstKVec = load <elemsPerVec/2 * i32 + 512> （k==0）
      // secondKVec = load <elemsPerVec/2 * i32 + 512> （k==1)
      // for _ from 0 to elemsPerVec/2:
      //   k01_n0 = perm(firstKVec, secondKVec, front)
      //   k01_n1 = perm(firstKVec, secondKVec, back)
      //   insert_to_outVals
      auto firstRegIdxAdd =
          reps.apply({{kReg, j}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto firstRegIdxAddI8 = firstRegIdxAdd * (bitwidth / 8);
      Value firstInnerOffset = b.add(offset, b.i32_val(firstRegIdxAddI8));
      auto firstVecAddr = b.gep(smemPtrTy, i8_ty, smemBase, firstInnerOffset,
                                LLVM::GEPNoWrapFlags::inbounds);

      auto secondRegIdxAdd = permTile.apply({{kReg, 1}})[0].second;
      auto secondRegIdxAddI8 = secondRegIdxAdd * (bitwidth / 8);
      Value secondInnerOffset =
          b.add(firstInnerOffset, b.i32_val(secondRegIdxAddI8));
      auto secondVecAddr = b.gep(smemPtrTy, i8_ty, smemBase, secondInnerOffset,
                                 LLVM::GEPNoWrapFlags::inbounds);

      // emit lds
      Type i32VecTy = vec_ty(i32_ty, elemsPerVec / 2);
      Type elemV2Ty = vec_ty(llvmElemTy, 2);

      Type int32Ty = type::i32Ty(ctx);
      Type int32xnTy = vec_ty(type::i32Ty(ctx), 2);
      Type int32xnPtrTy = ptr_ty(rewriter.getContext(), 3);
      Type elemType;
      if (llvmElemTy.isF16()) {
        elemType = type::f16Ty(ctx);
      } else if (llvmElemTy.isBF16() || llvmElemTy.isSignlessInteger(16)) {
        elemType = type::i16Ty(ctx);
      } else if (llvmElemTy.isF32()) { // TF32
        elemType = type::f32Ty(ctx);
      } else {
        assert(false && "Invalid smem load");
      }
      // Type elem_ptr_ty = ptr_ty(rewriter.getContext(), elemType, 3);
      Type elem_ptr_ty = ptr_ty(rewriter.getContext(), 3);
      std::string lds_intrinsic = "llvm.mxc.lds";
      mlir::LLVM::METAX::appendIntrinsicModifer(lds_intrinsic, 2, int32Ty);
      auto localloadOp = dyn_cast<triton::gpu::LocalLoadOp>(localLoadOp);

      // first lds
      auto fisrtVecPtr =
          b.bitcast(b.gep(elem_ptr_ty, elemType, firstVecAddr, b.i32_val(0)),
                    int32xnPtrTy);
      Value firstKVec = b.undef(int32xnPtrTy);
      firstKVec = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
          rewriter, loc, localloadOp, lds_intrinsic, int32xnTy, {fisrtVecPtr});

      auto fisrtVecPtrSt =
          b.bitcast(b.gep(elem_ptr_ty, elemType, firstVecAddr, b.i32_val(512)),
                    int32xnPtrTy);
      Value firstKVecSt = b.undef(int32xnPtrTy);
      firstKVecSt = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
          rewriter, loc, localloadOp, lds_intrinsic, int32xnTy,
          {fisrtVecPtrSt});

      // second lds
      auto secondVecPtr =
          b.bitcast(b.gep(elem_ptr_ty, elemType, secondVecAddr, b.i32_val(0)),
                    int32xnPtrTy);
      Value secondKVec = b.undef(int32xnPtrTy);
      secondKVec = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
          rewriter, loc, localloadOp, lds_intrinsic, int32xnTy, {secondVecPtr});

      auto secondVecPtrSt =
          b.bitcast(b.gep(elem_ptr_ty, elemType, secondVecAddr, b.i32_val(512)),
                    int32xnPtrTy);
      Value secondKVecSt = b.undef(int32xnPtrTy);
      secondKVecSt = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
          rewriter, loc, localloadOp, lds_intrinsic, int32xnTy,
          {secondVecPtrSt});

      int halfOfElemsSize = permCvt.getInDimSize(kReg) / 2;
      for (int vec2idx = 0; vec2idx < elemsPerVec / 2; ++vec2idx) {
        outVals[i + j + (vec2idx * 2) * 2] =
            b.extract_element(firstKVec, b.i32_val(vec2idx));
        outVals[i + j + (vec2idx * 2) * 2 + 1] =
            b.extract_element(secondKVec, b.i32_val(vec2idx));
        outVals[i + j + (vec2idx * 2 + 1) * 2] =
            b.extract_element(firstKVec, b.i32_val(vec2idx));
        outVals[i + j + (vec2idx * 2 + 1) * 2 + 1] =
            b.extract_element(secondKVec, b.i32_val(vec2idx));

        outVals[i + j + (vec2idx * 2) * 2 + halfOfElemsSize] =
            b.extract_element(firstKVecSt, b.i32_val(vec2idx));
        outVals[i + j + (vec2idx * 2) * 2 + 1 + halfOfElemsSize] =
            b.extract_element(secondKVecSt, b.i32_val(vec2idx));

        outVals[i + j + (vec2idx * 2 + 1) * 2 + halfOfElemsSize] =
            b.extract_element(firstKVecSt, b.i32_val(vec2idx));
        outVals[i + j + (vec2idx * 2 + 1) * 2 + 1 + halfOfElemsSize] =
            b.extract_element(secondKVecSt, b.i32_val(vec2idx));
      }
    }
  }

  return outVals;
}

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto sharedEnc =
        cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto regLayout = toLinearLayout(regTy);
    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = regLayout.invertAndCompose(sharedLL);
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      // cvt: reg to offset LL
      cvt = regLayout.invertAndCompose(sharedLayout);
    }
    auto kBlock = str_attr("block");
    // NYI. We would need to emit a map.shared::cluster instruction.
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }
    // remove block dim
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    // if can perm:
    const int elemBytes = llvmElemTy.getIntOrFloatBitWidth() / 8;
    bool canMatrixPerm = false;
    auto dotOpEnc = mlir::dyn_cast<DotOperandEncodingAttr>(regTy.getEncoding());
    int elemsPerVec;
    if (!paddedEnc && dotOpEnc) {
      auto mmaEnc = mlir::dyn_cast<MACAMmaEncodingAttr>(dotOpEnc.getParent());
      if (mmaEnc) {
        // shared to dot
        assert(!mmaEnc.getIsATrans() && !mmaEnc.getIsBTrans() &&
               "lds trans not supported yet");

        auto regOrder = mmaEnc.getRepOrderForOperand(dotOpEnc.getOpIdx());
        auto smemOrder = getOrder(memDescTy);

        auto elemMNK = mmaEnc.getElementsMNK();
        elemsPerVec = elemMNK[dotOpEnc.getOpIdx()];
        int elemsK = elemMNK[2];
        if (regOrder != smemOrder && elemBytes == 2 && elemsPerVec % 2 == 0 &&
            elemsK % 2 == 0) {
          canMatrixPerm = true;
        }
      }
    }

    SmallVector<Value> outVals;
    if (canMatrixPerm) {
      if (std::getenv("TRITON_DISABLE_SPLIT_PERM") == nullptr &&
          op.getMmaMode() == TT_MODE) {
        // only lds and perm as a standalone operation.
        outVals = lowerLocalLdswithoutPerm(loc, ctx, cvt, llvmElemTy, memDescTy,
                                           smemObj, rewriter, targetInfo, op,
                                           elemsPerVec);
      } else {
        // shared to dot && need perm
        outVals =
            lowerLocalLdPerm(loc, ctx, cvt, llvmElemTy, memDescTy, smemObj,
                             rewriter, targetInfo, op, elemsPerVec);
      }
    } else {
      outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                               smemObj, rewriter, targetInfo, op);
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::METAX::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
