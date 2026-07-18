#include "BufferOpsEmitter.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonHCUGPUTransforms/MlsGroup.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "nvidia/backend/include/cuda.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::HCUMlsSharedEncodingAttr;
using ::mlir::triton::hcugpu::MlsEncodingAttr;

namespace {

// ===----------------------------------------------------------------------===//
// Utility functions
// ===----------------------------------------------------------------------===//

SmallVector<unsigned> getShapePerCTA(ArrayRef<unsigned> mlsTile,
                                     ArrayRef<unsigned> warpsPerCTA) {
  assert(mlsTile.size() == warpsPerCTA.size() && mlsTile.size() == 2);

  return {warpsPerCTA[0] * mlsTile[0], warpsPerCTA[1] * mlsTile[1]};
}

SmallVector<unsigned> getNumReps(const MlsEncodingAttr &blockLayout,
                                 ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  assert(rank == 2);

  SmallVector<unsigned> numReps(rank);
  auto shapePerCTA =
      getShapePerCTA(blockLayout.getMlsTile(), blockLayout.getWarpsPerCTA());
  for (unsigned d = 0; d < rank; ++d) {
    numReps[d] = std::max<unsigned>(1, shape[d] / shapePerCTA[d]);
  }
  return numReps;
}

bool isKMajor(llvm::ArrayRef<unsigned> order, int opIdx) {
  auto rank = order.size();
  int kdim = opIdx == 0 ? rank - 1 : rank - 2;
  return order[0] == kdim;
}

} // namespace

//--------------------------------------------------------------------------------------------------

namespace {

struct MLSMatrixLoadToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::MatrixLoadToLocalOp> {
  using ConvertOpToLLVMPattern<
      triton::hcugpu::MatrixLoadToLocalOp>::ConvertOpToLLVMPattern;

  MLSMatrixLoadToLocalOpConversion(LLVMTypeConverter &converter,
                                   const HCU::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::hcugpu::MatrixLoadToLocalOp>(converter,
                                                                    benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::MatrixLoadToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    bool mABIsNeed =
        targetInfo.getGPUKind() == llvm::AMDGPU::GPUKind::GK_GFX938;

    auto ctx = rewriter.getContext();

    auto dstTy = cast<MemDescType>(op.getDest().getType());
    auto sharedLayout = dyn_cast<HCUMlsSharedEncodingAttr>(dstTy.getEncoding());
    auto blockLayout =
        op->getAttrOfType<MlsEncodingAttr>(MlsEncodingAttr::getMnemonic());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDest(),
                                                         llvmElemTy, rewriter);

    auto opIdx = blockLayout.getOpIdx();
    auto kDimIdx = opIdx == 0 ? 1 : 0;
    auto nonKDimIdx = opIdx == 0 ? 0 : 1;
    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
        blockLayout.getMlsTile()[nonKDimIdx], blockLayout.getMlsTile()[kDimIdx],
        blockLayout.getElemBitWidth(), opIdx,
        blockLayout.getOrder()[0] == kDimIdx,
        static_cast<MlsInterleaveKind>(blockLayout.getAlt2Kind()),
        blockLayout.getVersion());
    auto mlInsnAttr = mlsInsn->getMatrixLoadInsnAttr();

    auto allocOp = op.getResult().getDefiningOp<triton::gpu::LocalAllocOp>();
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;

    auto llBasePtr = adaptor.getBase();
    auto llShape = adaptor.getShape();
    auto llStrides = adaptor.getStrides();
    auto llOffsets = adaptor.getIndices();
    auto llBlockPos = adaptor.getIndices();

    auto ldBlkOff = dot64(loc, rewriter, llOffsets, llStrides);

    auto boundaryCheck = adaptor.getBoundaryCheck();
    bool needBoundaryCheck = boundaryCheck.size() > 0;
    SmallVector<bool> boundaryCheckInfo = {
        needBoundaryCheck && boundaryCheck[0] == 0,
        needBoundaryCheck && boundaryCheck[boundaryCheck.size() - 1] == 1};
    SmallVector<unsigned> numReps = getNumReps(blockLayout, dstTy.getShape());
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];

    bool useMatrixLoadStoreOffsetsWithMlOffs = true;
    if (boundaryCheckInfo[0] && boundaryCheckInfo[1]) {
      // Per mls inst need a rsrc desc for boundary padding.
      useMatrixLoadStoreOffsetsWithMlOffs = false;
    }

    if (!useMatrixLoadStoreOffsetsWithMlOffs) {
      SmallVector<std::pair<Value, Value>> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixLoadStoreOffsets(
          loc, rewriter, blockLayout, mlInsnAttr, sharedLayout,
          dstTy.getShape(), llStrides, mlsInsn->isRowMajor(), boundaryCheckInfo,
          ldInBlkOffCoordMappings);

      unsigned iterIdx = 0;
      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldInBlkOff64 = ldstInBlkGroup.first;

        // global mem address calc
        Value ldOffset = b.add(ldInBlkOff64, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1), elemByteWidth == 1 ? i8_ty : i16_ty,
                            llBasePtr, ldOffset);

        // create matrix load lds inst
        Value llStride = llStrides[sharedLayout.getOrder()[1]];
        llStride = b.trunc(i32_ty, llStride);

        Value paddingLenX = b.i32_val(0);
        Value paddingLenY = b.i32_val(0);
        if (boundaryCheckInfo[0]) {
          Value ldInBlockOffCoordX = ldInBlkOffCoordMappings[iterIdx].first;
          Value posEnd = b.add(b.add(llBlockPos[0], ldInBlockOffCoordX),
                               b.i32_val(mlInsnAttr.instrShape[0]));
          Value posGt = b.icmp_sgt(posEnd, llShape[0]);
          paddingLenX =
              b.select(posGt, b.sub(posEnd, llShape[0]), b.i32_val(0));
        }
        if (boundaryCheckInfo[1]) {
          Value ldInBlockOffCoordY = ldInBlkOffCoordMappings[iterIdx].second;
          Value posEnd = b.add(b.add(llBlockPos[1], ldInBlockOffCoordY),
                               b.i32_val(mlInsnAttr.instrShape[1]));
          Value posGt = b.icmp_sgt(posEnd, llShape[1]);
          paddingLenY =
              b.select(posGt, b.sub(posEnd, llShape[1]), b.i32_val(0));
        }
        Value mPadding =
            mlsInsn->getMajorDimIndex() == 0 ? paddingLenX : paddingLenY;
        Value nmPadding =
            mlsInsn->getMajorDimIndex() == 0 ? paddingLenY : paddingLenX;

        Value rsrcDesc =
            createRsrcDesc(loc, rewriter, ldPtr, llStride,
                           blockLayout.getAlt2Kind(), mPadding, nmPadding);

        // share mem address calc
        Value stInBlkOff32 = ldstInBlkGroup.second;
        Value smemOff = stInBlkOff32;
        Value smemPtr =
            b.gep(ptr_ty(ctx, 3), elemByteWidth == 1 ? i8_ty : i16_ty,
                  smemObj.getBase(), smemOff);
        if (mlsInsn->isTranspose() && mABIsNeed) {
          Value stPtrInt = b.ptrtoint(i32_ty, smemPtr);
          stPtrInt = b.or_(stPtrInt, b.i32_val(0x80000000));
          smemPtr = b.inttoptr(ptr_ty(ctx, 3), stPtrInt);
        }
        bool t = mlsInsn->isRowMajor();
        bool r = false;
        generateMatrixLoadLDSOp(loc, rewriter, mlInsnAttr.insn, rsrcDesc,
                                smemPtr, t, 0, r, false, false, false);
        iterIdx++;
      }
    } else {
      // compute matrix load/store offsets with desc group and diff ml offset.
      bool mlOffsInY;
      DenseMap<Value, Value> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixLoadStoreOffsetsWithMlOffs(
          loc, rewriter, blockLayout, mlInsnAttr, sharedLayout,
          dstTy.getShape(), llStrides, mlsInsn->isRowMajor(), boundaryCheckInfo,
          mlOffsInY, ldInBlkOffCoordMappings);
      bool groupPaddingInY = !mlOffsInY;

      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldInBlkOff64 = ldstInBlkGroup.first;

        // global mem address calc
        Value ldOffset = b.add(ldInBlkOff64, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1), elemByteWidth == 1 ? i8_ty : i16_ty,
                            llBasePtr, ldOffset);

        // create matrix load lds inst
        Value llStride = llStrides[sharedLayout.getOrder()[1]];
        llStride = b.trunc(i32_ty, llStride);

        Value rsrcDesc;
        for (auto &ldstInBlkOff : ldstInBlkGroup.second) {
          unsigned ldInBlkOffC = ldstInBlkOff.first;
          Value stInBlkOff32 = ldstInBlkOff.second;

          if (!rsrcDesc) {
            Value paddingLen = b.i32_val(0);
            if (needBoundaryCheck) {
              int dimIdx = groupPaddingInY ? 1 : 0;
              Value ldInBlockOffCoordV = ldInBlkOffCoordMappings[ldInBlkOff64];
              Value posEnd =
                  b.add(b.add(llBlockPos[dimIdx], ldInBlockOffCoordV),
                        b.i32_val(mlInsnAttr.instrShape[dimIdx]));
              Value posGt = b.icmp_sgt(posEnd, llShape[dimIdx]);
              paddingLen =
                  b.select(posGt, b.sub(posEnd, llShape[dimIdx]), b.i32_val(0));
            }

            Value mPadding = ((groupPaddingInY && mlsInsn->isRowMajor()) ||
                              (!groupPaddingInY && !mlsInsn->isRowMajor()))
                                 ? paddingLen
                                 : b.i32_val(0);
            Value nmPadding =
                mPadding == paddingLen ? b.i32_val(0) : paddingLen;
            rsrcDesc =
                createRsrcDesc(loc, rewriter, ldPtr, llStride,
                               blockLayout.getAlt2Kind(), mPadding, nmPadding);
          }

          // share mem address calc
          Value smemOff = stInBlkOff32;
          Value smemPtr =
              b.gep(ptr_ty(ctx, 3), elemByteWidth == 1 ? i8_ty : i16_ty,
                    smemObj.getBase(), smemOff);
          if (mlsInsn->isTranspose() && mABIsNeed) {
            Value stPtrInt = b.ptrtoint(i32_ty, smemPtr);
            stPtrInt = b.or_(stPtrInt, b.i32_val(0x80000000));
            smemPtr = b.inttoptr(ptr_ty(ctx, 3), stPtrInt);
          }

          bool t = mlsInsn->isRowMajor();
          bool r = mlOffsInY;
          assert(ldInBlkOffC < 1024 && "ldInBlkOffC out of range"); // 2^10
          generateMatrixLoadLDSOp(loc, rewriter, mlInsnAttr.insn, rsrcDesc,
                                  smemPtr, t, ldInBlkOffC, r, false, false,
                                  false);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;

  /*
   * Calculate matrix load store offsets in blk: ldOffVal, stOffVal
   **/
  SmallVector<std::pair<Value, Value>> computeMatrixLoadStoreOffsets(
      Location loc, RewriterBase &rewriter, const MlsEncodingAttr &blockLayout,
      const MatrixLoadInsnAttr &mlInsnAttr,
      const HCUMlsSharedEncodingAttr &sharedLayout,
      ArrayRef<int64_t> tensorShape, ValueRange llStrides, bool mlsIsRowMajor,
      SmallVector<bool> boundaryCheckInfo,
      SmallVector<std::pair<Value, Value>> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(blockLayout.getMlsTile(), warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    auto dummyMapping = computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, blockLayout, mlInsnAttr, tensorShape, multiDimWarpIds);

    auto dsByteOffsets = mlInsnAttr.dsByteOffsets;
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    SmallVector<unsigned> flattenedMlsShape{tensorShape};
    flattenedMlsShape[0] = tensorShape[0] / blockLayout.getMlsTile()[0];
    flattenedMlsShape[1] = tensorShape[1] / blockLayout.getMlsTile()[1];

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2],
                           b.i32_val(blockLayout.getMlsTile()[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1],
                           b.i32_val(blockLayout.getMlsTile()[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(blockLayout, tensorShape);
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];
    bool needBoundaryCheck = boundaryCheckInfo[0] || boundaryCheckInfo[1];

    SmallVector<std::pair<Value, Value>> loadStoreOffsets;

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < mlInsnAttr.instrsPerWarp[0];
             ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < mlInsnAttr.instrsPerWarp[1];
               ++instIdxY) {
            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            Value ldRow =
                b.add(offWarpX,
                      b.i32_val(ctaRow + instIdxX * mlInsnAttr.instrShape[0]));
            Value ldCol =
                b.add(offWarpY,
                      b.i32_val(ctaCol + instIdxY * mlInsnAttr.instrShape[1]));

            Value ldOffV = dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            Value stMlsRow =
                b.add(b.i32_val(ctaX * warpsPerCta[0]), multiDimWarpIds[0]);
            Value stMlsCol =
                b.add(b.i32_val(ctaY * warpsPerCta[1]), multiDimWarpIds[1]);

            Value stMlsOff =
                b.mul(linearize(rewriter, loc, {stMlsRow, stMlsCol},
                                flattenedMlsShape, sharedLayout.getOrder()),
                      b.i32_val(product(blockLayout.getMlsTile())));
            unsigned instIdx =
                instIdxX * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            Value stInMlsOff =
                b.i32_val(dsByteOffsets[instIdx] /
                          elemByteWidth); // element offset in mls
            Value stOffV = b.add(stMlsOff, stInMlsOff);

            loadStoreOffsets.push_back(std::make_pair(ldOffV, stOffV));
            ldInBlkOffCoordMappings.push_back(std::make_pair(ldRow, ldCol));
          }
        }
      }
    }

    return loadStoreOffsets;
  }

  /*
   * Calculate matrix load with mls offset field.
   * For each vector, share same rsrc desc with ldOffValue, but has different
   *ldOffConst offset & stOffValue. key: ldOffValue, value:
   *std::pair<ldOffConst, stOffValue>
   **/
  DenseMap<Value, SmallVector<std::pair<unsigned, Value>>>
  computeMatrixLoadStoreOffsetsWithMlOffs(
      Location loc, RewriterBase &rewriter, const MlsEncodingAttr &blockLayout,
      const MatrixLoadInsnAttr &mlInsnAttr,
      const HCUMlsSharedEncodingAttr &sharedLayout,
      ArrayRef<int64_t> tensorShape, ValueRange llStrides, bool mlsIsRowMajor,
      SmallVector<bool> boundaryCheckInfo,
      bool &mlOffsInY, /* key: ldOffValue, value: ldCoordValue */
      DenseMap<Value, Value> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(blockLayout.getMlsTile(), warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    auto dummyMapping = computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, blockLayout, mlInsnAttr, tensorShape, multiDimWarpIds);

    auto dsByteOffsets = mlInsnAttr.dsByteOffsets;
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    SmallVector<unsigned> flattenedMlsShape{tensorShape};
    flattenedMlsShape[0] = tensorShape[0] / blockLayout.getMlsTile()[0];
    flattenedMlsShape[1] = tensorShape[1] / blockLayout.getMlsTile()[1];

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2],
                           b.i32_val(blockLayout.getMlsTile()[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1],
                           b.i32_val(blockLayout.getMlsTile()[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(blockLayout, tensorShape);
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];
    bool membersPerGroupInY =
        (numRepsX == numRepsY)
            ? mlsIsRowMajor
            : numRepsX < numRepsY; // per numRepsY mls insts share same coordX ?
    bool needBoundaryPadding = boundaryCheckInfo[0] || boundaryCheckInfo[1];
    if (needBoundaryPadding) {
      assert(!(boundaryCheckInfo[0] && boundaryCheckInfo[1]) &&
             "both padding condition should not here!");
      if (boundaryCheckInfo[1] && membersPerGroupInY)
        membersPerGroupInY = false;
    }
    mlOffsInY = membersPerGroupInY;
    bool groupPaddingInY = !mlOffsInY;

    SmallVector<SmallVector<std::tuple<Value, unsigned, Value, Value>>>
        loadStoreOffsets;
    unsigned numGroups = membersPerGroupInY
                             ? numRepsX
                             : numRepsY; /* need create numGroups rsrc desc */
    unsigned membersPerGroup =
        membersPerGroupInY
            ? numRepsY
            : numRepsX; /* per group has membersPerGroup mls insts */
    loadStoreOffsets.resize(numGroups);
    for (auto &innerVec : loadStoreOffsets) {
      innerVec.resize(membersPerGroup);
    }

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < mlInsnAttr.instrsPerWarp[0];
             ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < mlInsnAttr.instrsPerWarp[1];
               ++instIdxY) {
            unsigned groupIdx =
                membersPerGroupInY
                    ? ctaX * mlInsnAttr.instrsPerWarp[0] + instIdxX
                    : ctaY * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            unsigned memberIdx =
                membersPerGroupInY
                    ? ctaY * mlInsnAttr.instrsPerWarp[1] + instIdxY
                    : ctaX * mlInsnAttr.instrsPerWarp[0] + instIdxX;

            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            Value ldRow = membersPerGroupInY
                              ? b.add(offWarpX, b.i32_val(ctaRow))
                              : offWarpX;
            Value ldCol = membersPerGroupInY
                              ? offWarpY
                              : b.add(offWarpY, b.i32_val(ctaCol));

            Value ldOffV = dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            unsigned ldOffC =
                membersPerGroupInY
                    ? ctaCol + instIdxY * mlInsnAttr.instrShape[1]
                    : ctaRow + instIdxX * mlInsnAttr.instrShape[0];

            Value stMlsRow =
                b.add(b.i32_val(ctaX * warpsPerCta[0]), multiDimWarpIds[0]);
            Value stMlsCol =
                b.add(b.i32_val(ctaY * warpsPerCta[1]), multiDimWarpIds[1]);

            Value stMlsOff =
                b.mul(linearize(rewriter, loc, {stMlsRow, stMlsCol},
                                flattenedMlsShape, sharedLayout.getOrder()),
                      b.i32_val(product(blockLayout.getMlsTile())));
            unsigned instIdx =
                instIdxX * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            Value stInMlsOff =
                b.i32_val(dsByteOffsets[instIdx] /
                          elemByteWidth); // element offset in mls
            Value stOffset = b.add(stMlsOff, stInMlsOff);

            Value ldInBlockOffVal = groupPaddingInY ? ldCol : ldRow;
            loadStoreOffsets[groupIdx][memberIdx] =
                std::make_tuple(ldOffV, ldOffC, stOffset, ldInBlockOffVal);
          }
        }
      }
    }

    DenseMap<Value, SmallVector<std::pair<unsigned, Value>>>
        loadStoreOffsetsMap;
    for (auto &group : loadStoreOffsets) {
      Value ldOffV = std::get<0>(group[0]);
      auto &curMap = loadStoreOffsetsMap[ldOffV];
      curMap.reserve(group.size());
      for (auto &member : group) {
        unsigned ldOffC = std::get<1>(member);
        Value stOffset = std::get<2>(member);
        curMap.emplace_back(ldOffC, stOffset);
      }

      Value ldInBlockOffVal = std::get<3>(group[0]);
      ldInBlkOffCoordMappings[ldOffV] = ldInBlockOffVal;
    }

    return loadStoreOffsetsMap;
  }

  // reference: emitBaseIndexForMfmaLayout, emitMfmaOffsetForCTA
  SmallVector<SmallVector<Value>> computeMatrixLoadStoreOffsetsMappingInCTA(
      Location loc, RewriterBase &rewriter, const MlsEncodingAttr &blockLayout,
      const MatrixLoadInsnAttr &mlInsnAttr, ArrayRef<int64_t> shape,
      SmallVector<Value> &multiDimWarpIds) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto rank = shape.size();
    assert(rank == 2);

    auto warpOrder = blockLayout.getOrder();
    auto _warpsPerCTA = blockLayout.getWarpsPerCTA();
    SmallVector<Value> warpsPerCTA;
    for (unsigned i = 0; i < rank; ++i)
      warpsPerCTA.push_back(b.i32_val(_warpsPerCTA[i]));

    auto mlsTile = blockLayout.getMlsTile();

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(64);
    Value laneId = b.urem(threadId, warpSize);
    // Note: To make the warpId as a uniform value for Matrix Load Rsrc Desc.
    Value warpId = rewriter.create<ROCDL::ReadfirstlaneOp>(
        loc, i32_ty, b.udiv(threadId, warpSize));
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, _warpsPerCTA, warpOrder);

    if (shape[rank - 2] >= mlsTile[rank - 2]) {
      assert(shape[rank - 2] % mlsTile[rank - 2] == 0);
      multiDimWarpId[rank - 2] =
          b.urem(multiDimWarpId[rank - 2],
                 b.i32_val(ceil<unsigned>(shape[rank - 2], mlsTile[rank - 2])));
    }
    if (shape[rank - 1] >= mlsTile[rank - 1]) {
      assert(shape[rank - 1] % mlsTile[rank - 1] == 0);
      multiDimWarpId[rank - 1] =
          b.urem(multiDimWarpId[rank - 1],
                 b.i32_val(ceil<unsigned>(shape[rank - 1], mlsTile[rank - 1])));
    }

    multiDimWarpIds.push_back(multiDimWarpId[rank - 2]);
    multiDimWarpIds.push_back(multiDimWarpId[rank - 1]);

    Value offWarpX =
        b.mul(multiDimWarpId[rank - 2], b.i32_val(mlsTile[rank - 2]));
    Value offWarpY =
        b.mul(multiDimWarpId[rank - 1], b.i32_val(mlsTile[rank - 1]));

    SmallVector<SmallVector<Value>> multiDimOffsets;
    for (unsigned i = 0; i < mlInsnAttr.instrsPerWarp[0]; i++) {
      for (unsigned j = 0; j < mlInsnAttr.instrsPerWarp[1]; j++) {
        assert((mlInsnAttr.instrsPerWarp[0] == 1 ||
                mlInsnAttr.instrsPerWarp[1] == 1) &&
               "add more code support!");
        Value offElemX = b.add(
            offWarpX, b.mul(b.i32_val(i), b.i32_val(mlInsnAttr.instrShape[0])));
        Value offElemY = b.add(
            offWarpY, b.mul(b.i32_val(j), b.i32_val(mlInsnAttr.instrShape[1])));

        multiDimOffsets.push_back({offElemX, offElemY});
      }
    }

    return multiDimOffsets;
  }

  Value createRsrcDesc(Location loc, RewriterBase &rewriter, Value llBasePtr,
                       Value llStride, unsigned alt2Kind, Value mPadding,
                       Value nmPadding) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // 1. Create the resource descriptor
    // dword 0: base_addr_lo, bit pos: [31:0], Byte base address 32 LSBs for
    // global adressing dwrod 1: base_addr_hi, bit pos: [15:0], Byte base
    // address 16 MSBs for global adressing dword 2: stride,       bit pos:
    // [31:0], The unit is matrix element. dword 3:
    //   mfilter,      bit pos: [7:0]
    //                 The number of row/column needed to be zero-padded in
    //                 matrix major direction.
    //   nfilter,      bit pos: [15:8]
    //                 The number of row/column needed to be zero-padded ...
    //   cache_swizzle_enable, bit pos: [16]
    //                 1: enable cache swizzle operation in L1 cache.
    //   mfmt,         bit pos: [18:17], The matrix layout format in memory.
    //                 0: non-interleaved
    //                 1: 2-interleaved
    //                 2: 4-interleaved
    //                 3: 8-interleaved
    // Note: from Spec, it requires the base ptr to be 16-byte aligned !!!

    // dword 0: base_addr_lo [31:0]
    Value baseLow = b.ptrtoint(i32_ty, llBasePtr);
    // dword 1: base_addr_hi [15:0]
    Value baseHigh = b.and_(
        b.trunc(i32_ty, b.lshr(b.ptrtoint(i64_ty, llBasePtr), b.i64_val(32))),
        b.i32_val(0xFFFF));

    // dword 2: stride [31:0]
    Value stride = llStride;

    // dword 3:
    Value mfilter = b.and_(mPadding, b.i32_val(0xFF));
    Value nfilter = b.and_(nmPadding, b.i32_val(0xFF));
    Value cacheSwizzle = b.i32_val(0);
    Value mfmt = b.i32_val(alt2Kind);

    Value controlField = mfilter;                                     // [7:0]
    controlField = b.or_(controlField, b.shl(nfilter, b.i32_val(8))); // [15:8]
    controlField =
        b.or_(controlField, b.shl(cacheSwizzle, b.i32_val(16)));    // [16]
    controlField = b.or_(controlField, b.shl(mfmt, b.i32_val(17))); // [18:17]

    Value resource = b.undef(vec_ty(i32_ty, 4));
    resource =
        b.insert_element(vec_ty(i32_ty, 4), resource, baseLow, b.i32_val(0));
    resource =
        b.insert_element(vec_ty(i32_ty, 4), resource, baseHigh, b.i32_val(1));
    resource =
        b.insert_element(vec_ty(i32_ty, 4), resource, stride, b.i32_val(2));
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, controlField,
                                b.i32_val(3));

    return resource;
  }

  void generateMatrixLoadLDSOp(
      Location loc, RewriterBase &rewriter, StringRef mlInsnOpName,
      Value rsrcDesc, Value soffset,
      bool t,         // Transposition. 0: column major, 1: row major
      int32_t offset, // Global address offset [9:0]
      bool r,         // Golbal address offset is in row direction ?
      bool glc = false, bool slc = false, bool bps = false) const {
    OperationState loweredOp(loc, mlInsnOpName);
    loweredOp.addOperands(rsrcDesc);
    loweredOp.addOperands(soffset);

    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));
    loweredOp.addAttribute("t", rewriter.getBoolAttr(t));
    loweredOp.addAttribute("r", rewriter.getBoolAttr(r));
    loweredOp.addAttribute("glc", rewriter.getBoolAttr(glc));
    loweredOp.addAttribute("slc", rewriter.getBoolAttr(slc));
    loweredOp.addAttribute("bps", rewriter.getBoolAttr(bps));

    rewriter.create(loweredOp);
  }

  Value dot64(Location loc, RewriterBase &rewriter, ValueRange offsets32,
              ValueRange strides64) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert(offsets32.size() == strides64.size());
    Value ret = b.i64_val(0);
    for (auto [offset, stride] : llvm::zip(offsets32, strides64)) {
      ret = b.add(ret, b.mul(b.zext(i64_ty, offset), stride));
    }
    return ret;
  }
};

//--------------------------------------------------------------------------------------------------

struct MLSLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  MLSLocalAllocOpConversion(LLVMTypeConverter &converter,
                            const HCU::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto resultTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        dyn_cast<HCUMlsSharedEncodingAttr>(resultTy.getEncoding());
    if (!sharedLayout)
      return failure();

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto smemObj = LLVM::SharedMemoryObject(smemBase, llvmElemTy,
                                            resultTy.getRank(), loc, rewriter);
    assert(!op.getSrc());
    auto retVal = LLVM::getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

//--------------------------------------------------------------------------------------------------

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, wpt, order);

  return b.urem(multiDimWarpId[nonKIdx],
                b.i32_val(tensorSizeNonK / elemPerInstrNonK));
}

struct MLSLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  MLSLocalLoadOpConversion(LLVMTypeConverter &converter,
                           const HCU::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<triton::gpu::HCUMlsSharedEncodingAttr>(srcLayout)) {
      if (isa<DotOperandEncodingAttr>(dstLayout) &&
          isa<AMDMfmaEncodingAttr>(
              cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
        return lowerMLSSharedToDotOperand(op, adaptor, getTypeConverter(),
                                          rewriter);
      } else {
        assert(false && "should been canonicalized and not reach here!");
        return failure();
      }
    }

    return failure();
  }

private:
  const HCU::TargetInfo &targetInfo;

  // shared -> matrix_core_dot_operand
  // reference: SharedToDotOperandMFMA::convertLayout in
  // SharedToDotOperandMFMA.cpp
  LogicalResult
  lowerMLSSharedToDotOperand(triton::gpu::LocalLoadOp op,
                             triton::gpu::LocalLoadOpAdaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto srcTensorTy = cast<MemDescType>(src.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());

    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);

    Value res;
    auto opIdx = dotOperandLayout.getOpIdx();
    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(dotOperandLayout.getParent());
    Value threadId = getThreadId(rewriter, loc);
    if ((opIdx == 0 &&
         mfmaLayout.getMfmaTile()[0] == mfmaLayout.getInstrShape()[0]) ||
        (opIdx == 1 &&
         mfmaLayout.getMfmaTile()[1] == mfmaLayout.getInstrShape()[1])) {
      res = convertLayoutUnitTilesPerWarp(dotOperandLayout.getOpIdx(), rewriter,
                                          loc, src, dotOperandLayout, smemObj,
                                          typeConverter, threadId);
    } else {
      res = convertLayoutMultiTilesPerWarp(dotOperandLayout.getOpIdx(),
                                           rewriter, loc, src, dotOperandLayout,
                                           smemObj, typeConverter, threadId);
    }

    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }

  Value convertLayoutUnitTilesPerWarp(int opIdx,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc, Value tensor,
                                      DotOperandEncodingAttr encoding,
                                      const SharedMemoryObject &smemObj,
                                      const LLVMTypeConverter *typeConverter,
                                      Value thread) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
    auto tensorTy = cast<MemDescType>(tensor.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    auto rank = shape.size();
    int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
    int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;
    int kDimIdx2D = opIdx == 0 ? 1 : 0;
    int nonKDimIdx2D = opIdx == 0 ? 0 : 1;

    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
    assert(((opIdx == 0 && mfmaLayout.getInstrsPerWarp()[0] == 1) ||
            (opIdx == 1 && mfmaLayout.getInstrsPerWarp()[1] == 1)) &&
           "only support unit tiles per warp mfma layout!");
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

    auto elemTy = tensorTy.getElementType();
    auto kWidth = encoding.getKWidth();
    auto elemsPerInstr = mfmaLayout.getInstrShapeForOperand(kWidth, opIdx);

    int64_t mfmaInstrNonK;
    int64_t mfmaInstrK;
    // TODO(Lixun): make it simpler
    // getInstrShapeForOperand always returns a 2D vector
    if (rank == 3) {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
      mfmaInstrK = elemsPerInstr[kDimIdx - 1];
    } else {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
      mfmaInstrK = elemsPerInstr[kDimIdx];
    }

    if (mfmaInstrNonK > shape[nonKDimIdx] || mfmaInstrK > shape[kDimIdx]) {
      // This pattern does not support cases tensor shape is smaller than
      // one instruction size, it will be processed by LinearLayout converter
      return Value();
    }

    auto numReps = mfmaLayout.getRepForOperand(shape, kWidth, opIdx);
    auto numRepNonK = numReps[nonKDimIdx];
    auto numRepK = numReps[kDimIdx];
    auto repB = numReps[0];
    // TODO(Lixun): make it simpler
    // getRepForOperand always returns a 3D vector
    if (rank == 2) {
      numRepNonK = numReps[nonKDimIdx + 1];
      numRepK = numReps[kDimIdx + 1];
    }

    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);
    Value warpSize = b.i32_val(iWarpSize);
    Value linearWarpId = b.udiv(thread, warpSize);
    Value lane = b.urem(thread, warpSize);

    auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
    Value spatialWarpId = getWarpIdInBlock(
        rewriter, loc, linearWarpId, warpsPerCTA, mfmaInstrNonK,
        shape[nonKDimIdx], nonKDimIdx, warpOrder);

    int numSubBlocks = 1;
    // numOfElemsPerThreadPerMfmaInstr
    int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize;
    assert(numOfElems >= 1);

    unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
    int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
    int warpsPerBatch =
        rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
    Value warpIdInBatch = b.urem(linearWarpId, b.i32_val(warpsPerBatch));
    elemTy = typeConverter->convertType(elemTy);

    // 1. get ds_read_matrix inst info
    auto sharedLayout = cast<HCUMlsSharedEncodingAttr>(tensorTy.getEncoding());
    auto mlsTile = sharedLayout.getMlsTile();
    bool kMajor = sharedLayout.getOrder()[0] == kDimIdx;
    assert(kMajor && "m16n16 mfma & mls only support kMajor layout!");

    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
        mlsTile[nonKDimIdx2D], mlsTile[kDimIdx2D],
        sharedLayout.getElemBitWidth(), opIdx, kMajor,
        static_cast<MlsInterleaveKind>(sharedLayout.getAlt2Kind()),
        sharedLayout.getVersion());

    auto dsInsnAttr = mlsInsn->getDsReadMatrixInsnAttr();

    assert(mlsTile[kDimIdx2D] % mfmaInstrK == 0 &&
           shape[kDimIdx] % mlsTile[kDimIdx2D] == 0 &&
           "mls tile kdim and shape kdim should be divisible!");

    unsigned mlsNumRepK = shape[kDimIdx] / mlsTile[kDimIdx];
    unsigned mlsNumRepNonK = numRepNonK;
    SmallVector<unsigned> mlsNumReps{mlsNumRepNonK, mlsNumRepK};
    if (opIdx == 1) {
      std::swap(mlsNumReps[0], mlsNumReps[1]);
    }

    // 2. compute ds_read_matrix offsets
    auto offsets = computeDsReadMatrixOffsets(
        rewriter, loc, sharedLayout, dsInsnAttr, spatialWarpId,
        warpsPerBlockNonK, mlsNumReps, smemObj, shape);
    Value smemBase = smemObj.getBase();
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    assert(mlsInsn->getElemBitWidth() == 16 || mlsInsn->getElemBitWidth() == 8);
    auto dsLoadsPerK = product(dsInsnAttr.instrsPerWarp);

    // 3. create ds_read_matrix ops
    SmallVector<Value> loadedValues;
    for (int batchIdx = 0; batchIdx < repB; ++batchIdx) {
      int operandSize = shape[rank - 1] * shape[rank - 2];
      Value batchOffset =
          b.mul(b.i32_val(operandSize),
                b.add(warpIdInBatch, b.i32_val(batchIdx * warpsPerBatch)));
      for (int mlsNonK = 0; mlsNonK < mlsNumRepNonK; ++mlsNonK) {
        for (int mlsKIdx = 0; mlsKIdx < mlsNumRepK; ++mlsKIdx) {
          auto loadOffset = offsets[mlsNonK * mlsNumRepK + mlsKIdx];
          loadOffset = b.add(loadOffset, batchOffset);
          for (int loadIdx = 0; loadIdx < dsLoadsPerK; ++loadIdx) {
            Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, loadOffset);

            auto dsInsnResTy =
                getDsReadMatrixInsnResType(loc, rewriter, tensorTy, dsInsnAttr,
                                           mlsInsn->getElemBitWidth());
            Value loadedValue = generateDsReadMatrixOp(
                loc, rewriter, dsInsnAttr.insn, dsInsnResTy, loadAddress,
                dsInsnAttr.dsByteOffsets[loadIdx], dsInsnAttr.flags);
            auto unpackedValues = unpackDsReadMatrixInsnRes(
                loc, rewriter, tensorTy, dsInsnAttr, mlsInsn->getElemBitWidth(),
                loadedValue);
            loadedValues.append(unpackedValues);
          }
        }
      }
    }

    assert(loadedValues.size() ==
           repB * mlsNumRepNonK * mlsNumRepK * dsLoadsPerK *
               (dsInsnAttr.instrShape[kDimIdx2D] *
                dsInsnAttr.instrShape[nonKDimIdx2D] / iWarpSize));

    MLIRContext *ctx = mfmaLayout.getContext();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
    auto result =
        packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
    return result;
  }

  Value convertLayoutMultiTilesPerWarp(int opIdx,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, Value tensor,
                                       DotOperandEncodingAttr encoding,
                                       const SharedMemoryObject &smemObj,
                                       const LLVMTypeConverter *typeConverter,
                                       Value thread) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
    auto tensorTy = cast<MemDescType>(tensor.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    auto rank = shape.size();
    int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
    int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;
    int kDimIdx2D = opIdx == 0 ? 1 : 0;
    int nonKDimIdx2D = opIdx == 0 ? 0 : 1;

    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

    auto elemTy = tensorTy.getElementType();
    auto kWidth = encoding.getKWidth();
    auto elemsPerInstr = mfmaLayout.getInstrShapeForOperand(kWidth, opIdx);

    int64_t mfmaInstrNonK;
    int64_t mfmaInstrK;
    // TODO(Lixun): make it simpler
    // getInstrShapeForOperand always returns a 2D vector
    if (rank == 3) {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
      mfmaInstrK = elemsPerInstr[kDimIdx - 1];
    } else {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
      mfmaInstrK = elemsPerInstr[kDimIdx];
    }

    if (mfmaInstrNonK > shape[nonKDimIdx] || mfmaInstrK > shape[kDimIdx]) {
      // This pattern does not support cases tensor shape is smaller than
      // one instruction size, it will be processed by LinearLayout converter
      return Value();
    }

    auto numReps = mfmaLayout.getRepForOperand(shape, kWidth, opIdx);
    auto numRepNonK = numReps[nonKDimIdx];
    auto numRepK = numReps[kDimIdx];
    auto repB = numReps[0];
    // TODO(Lixun): make it simpler
    // getRepForOperand always returns a 3D vector
    if (rank == 2) {
      numRepNonK = numReps[nonKDimIdx + 1];
      numRepK = numReps[kDimIdx + 1];
    }

    auto mfmaTileNonK =
        opIdx == 0 ? mfmaLayout.getMfmaTile()[0] : mfmaLayout.getMfmaTile()[1];
    auto mfmaTileK = mfmaInstrK;
    auto mfmaInstrsPerWarpNonK = mfmaLayout.getInstrsPerWarp()[nonKDimIdx2D];
    auto mfmaTileNumReps =
        mfmaLayout.getMfmaTileRepForOperand(shape, kWidth, opIdx);
    auto mfmaTileNumRepNonK = mfmaTileNumReps[nonKDimIdx];
    if (rank == 2) {
      mfmaTileNumRepNonK = mfmaTileNumReps[nonKDimIdx + 1];
    }

    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);
    Value warpSize = b.i32_val(iWarpSize);
    Value linearWarpId = b.udiv(thread, warpSize);
    Value lane = b.urem(thread, warpSize);

    auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
    Value spatialWarpId =
        getWarpIdInBlock(rewriter, loc, linearWarpId, warpsPerCTA, mfmaTileNonK,
                         shape[nonKDimIdx], nonKDimIdx, warpOrder);

    int numSubBlocks = 1;
    // numOfElemsPerThreadPerMfmaInstr
    int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize;
    assert(numOfElems >= 1);

    unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaTileNonK;
    int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
    int warpsPerBatch =
        rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
    Value warpIdInBatch = b.urem(linearWarpId, b.i32_val(warpsPerBatch));
    elemTy = typeConverter->convertType(elemTy);

    // 1. get ds_read_matrix inst info
    auto sharedLayout = cast<HCUMlsSharedEncodingAttr>(tensorTy.getEncoding());
    auto mlsTile = sharedLayout.getMlsTile();
    bool kMajor = sharedLayout.getOrder()[0] == kDimIdx;

    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
        mlsTile[nonKDimIdx2D], mlsTile[kDimIdx2D],
        sharedLayout.getElemBitWidth(), opIdx, kMajor,
        static_cast<MlsInterleaveKind>(sharedLayout.getAlt2Kind()),
        sharedLayout.getVersion());

    auto dsInsnAttr = mlsInsn->getDsReadMatrixInsnAttr();

    assert(mlsTile[kDimIdx2D] % mfmaInstrK == 0 &&
           shape[kDimIdx] % mlsTile[kDimIdx2D] == 0 &&
           "mls tile kdim and shape kdim should be divisible!");

    unsigned mlsNumRepK = shape[kDimIdx] / mlsTile[kDimIdx2D];
    unsigned mlsNumRepNonK = mfmaTileNumRepNonK;
    SmallVector<unsigned> mlsNumReps{mlsNumRepNonK, mlsNumRepK};
    if (opIdx == 1) {
      std::swap(mlsNumReps[0], mlsNumReps[1]);
    }

    // 2. compute ds_read_matrix offsets
    auto offsets = computeDsReadMatrixOffsets(
        rewriter, loc, sharedLayout, dsInsnAttr, spatialWarpId,
        warpsPerBlockNonK, mlsNumReps, smemObj, shape);
    Value smemBase = smemObj.getBase();
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    assert(mlsInsn->getElemBitWidth() == 16 || mlsInsn->getElemBitWidth() == 8);
    auto dsLoadsPerK = product(dsInsnAttr.instrsPerWarp);

    // 3. create ds_read_matrix ops
    assert((dsInsnAttr.instrsPerWarp[0] == 1 ||
            dsInsnAttr.instrsPerWarp[1] == 1) &&
           "add more code support!");
    bool isDsInsnRepInKDirection = dsInsnAttr.instrsPerWarp[kDimIdx2D] != 1;
    unsigned mfmaGroupPerDsInsn =
        dsInsnAttr.instrShape[nonKDimIdx2D] / mfmaInstrNonK;
    assert(mfmaGroupPerDsInsn != 1);

    unsigned numOfElemsPerDsInsn = dsInsnAttr.instrShape[kDimIdx2D] *
                                   dsInsnAttr.instrShape[nonKDimIdx2D] /
                                   iWarpSize;
    unsigned totalElemsMfma = repB *
                              (mfmaTileNumRepNonK * mfmaInstrsPerWarpNonK) *
                              numRepK * numOfElems;
    unsigned totalElemsMls =
        repB * mlsNumRepNonK * mlsNumRepK * dsLoadsPerK * numOfElemsPerDsInsn;
    assert(totalElemsMfma == totalElemsMls);

    SmallVector<Value> loadedValues(totalElemsMfma);
    for (int batchIdx = 0; batchIdx < repB; ++batchIdx) {
      int operandSize = shape[rank - 1] * shape[rank - 2];
      Value batchOffset =
          b.mul(b.i32_val(operandSize),
                b.add(warpIdInBatch, b.i32_val(batchIdx * warpsPerBatch)));
      for (int mlsNonK = 0; mlsNonK < mlsNumRepNonK; ++mlsNonK) {
        int mfmaTileNonKIdx = mlsNonK;
        for (int mlsKIdx = 0; mlsKIdx < mlsNumRepK; ++mlsKIdx) {
          int mfmaTileKIdx = mlsKIdx * (mlsTile[kDimIdx2D] / mfmaTileK);

          auto loadOffset = offsets[mlsNonK * mlsNumRepK + mlsKIdx];
          loadOffset = b.add(loadOffset, batchOffset);
          for (int loadIdx = 0; loadIdx < dsLoadsPerK; ++loadIdx) {
            Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, loadOffset);

            auto dsInsnResTy =
                getDsReadMatrixInsnResType(loc, rewriter, tensorTy, dsInsnAttr,
                                           mlsInsn->getElemBitWidth());
            Value loadedValue = generateDsReadMatrixOp(
                loc, rewriter, dsInsnAttr.insn, dsInsnResTy, loadAddress,
                dsInsnAttr.dsByteOffsets[loadIdx], dsInsnAttr.flags);
            auto unpackedValues = unpackDsReadMatrixInsnRes(
                loc, rewriter, tensorTy, dsInsnAttr, mlsInsn->getElemBitWidth(),
                loadedValue);

            assert(unpackedValues.size() % mfmaGroupPerDsInsn == 0 &&
                   "unpackedValues should be divisible by dsGroupSize");
            unsigned elemsPerGroup = unpackedValues.size() / mfmaGroupPerDsInsn;

            for (int i = 0; i < mfmaGroupPerDsInsn; ++i) {
              int mfmaInsnNonKIdxInTile =
                  isDsInsnRepInKDirection ? i
                                          : loadIdx * mfmaGroupPerDsInsn + i;
              int mfmaInsnKIdxInTile =
                  isDsInsnRepInKDirection
                      ? loadIdx * (dsInsnAttr.instrShape[kDimIdx2D] / mfmaTileK)
                      : 0;
              unsigned groupOffset =
                  batchIdx * (mfmaTileNumRepNonK * mfmaInstrsPerWarpNonK) *
                      numRepK * numOfElems /* batch idx */
                  + mfmaTileNonKIdx * mfmaInstrsPerWarpNonK * numRepK *
                        numOfElems /* block idx*/
                  + mfmaInsnNonKIdxInTile * numRepK *
                        numOfElems + /* inblock idx */
                  (mfmaTileKIdx + mfmaInsnKIdxInTile) * numOfElems;

              for (int j = 0; j < elemsPerGroup; ++j) {
                loadedValues[groupOffset + j] =
                    unpackedValues[i * elemsPerGroup + j];
              }
            }
          }
        }
      }
    }

    assert(loadedValues.size() == totalElemsMfma);

    MLIRContext *ctx = mfmaLayout.getContext();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
    auto result =
        packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
    return result;
  }

  llvm::SmallVector<Value> computeDsReadMatrixOffsets(
      ConversionPatternRewriter &rewriter, Location loc,
      const HCUMlsSharedEncodingAttr &sharedLayout,
      const DsReadMatrixInsnAttr &dsInsnAttr, Value warpNonKId,
      int warpsPerBlockNonK, ArrayRef<unsigned> mlsReps,
      SharedMemoryObject smemObj, ArrayRef<int64_t> shape) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto opIdx = sharedLayout.getOpIdx();
    auto mlsTile = sharedLayout.getMlsTile();
    auto kDimIdx = opIdx == 0 ? 1 : 0;
    auto nonKDimIdx = opIdx == 0 ? 0 : 1;

    assert(mlsReps.size() == 2);
    auto numRepK = mlsReps[kDimIdx];
    auto numRepNonK = mlsReps[nonKDimIdx];

    const auto numBlocks = numRepNonK;
    const auto numMlsTilesPerBlock = numRepK;
    const auto blockSize = numMlsTilesPerBlock;

    SmallVector<unsigned> flattenedMlsShape{shape};
    flattenedMlsShape[0] = shape[0] / sharedLayout.getMlsTile()[0];
    flattenedMlsShape[1] = shape[1] / sharedLayout.getMlsTile()[1];

    SmallVector<unsigned> tensorShape{shape};
    llvm::SmallVector<Value> offsets(numBlocks * numMlsTilesPerBlock);
    for (int block = 0; block < numBlocks; ++block) {
      // assert(isKMajor(sharedLayout.getOrder(), opIdx));
      int blockNonKOffset = block * warpsPerBlockNonK;

      for (int mlsTile = 0; mlsTile < numMlsTilesPerBlock; ++mlsTile) {
        Value mlsNonKOff = b.add(b.i32_val(blockNonKOffset), warpNonKId);
        Value mlsKOff = b.i32_val(mlsTile);

        std::array<Value, 2> mlsCoords =
            opIdx == 0 ? std::array<Value, 2>{mlsNonKOff, mlsKOff}
                       : std::array<Value, 2>{mlsKOff, mlsNonKOff};

        Value mlsOff =
            b.mul(linearize(rewriter, loc, mlsCoords, flattenedMlsShape,
                            sharedLayout.getOrder()),
                  b.i32_val(product(sharedLayout.getMlsTile())));
        offsets[block * blockSize + mlsTile] = mlsOff;
      }
    }

    return offsets;
  }

  Type getDsReadMatrixInsnResType(Location loc, RewriterBase &rewriter,
                                  const MemDescType &tensorTy,
                                  const DsReadMatrixInsnAttr &dsInsnAttr,
                                  unsigned elemBitWidth,
                                  unsigned iWarpSize = 64) const {
    auto dsInsnElemTy = tensorTy.getElementType();
    auto dsInsnNumOfElems =
        dsInsnAttr.instrShape[0] * dsInsnAttr.instrShape[1] / iWarpSize;

    if (elemBitWidth == 8) {
      dsInsnElemTy = i32_ty;
      dsInsnNumOfElems = 4;
    }
    return vec_ty(dsInsnElemTy, dsInsnNumOfElems);
  }

  Value generateDsReadMatrixOp(Location loc, RewriterBase &rewriter,
                               StringRef dsOpName, Type resType, Value soffset,
                               unsigned offset, unsigned flags) const {
    OperationState loweredOp(loc, dsOpName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands(soffset);
    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));

    int8_t elem3 = MLS_DS_FLAGS_GET_ELEM3(flags);
    int8_t row3 = MLS_DS_FLAGS_GET_ROW3(flags);
    int8_t col3 = MLS_DS_FLAGS_GET_COL3(flags);
    int8_t alt2 = MLS_DS_FLAGS_GET_ALT2(flags);

    loweredOp.addAttribute("element", rewriter.getI8IntegerAttr(elem3));
    loweredOp.addAttribute("row", rewriter.getI8IntegerAttr(row3));
    loweredOp.addAttribute("col", rewriter.getI8IntegerAttr(col3));
    loweredOp.addAttribute("alt", rewriter.getI8IntegerAttr(alt2));

    return rewriter.create(loweredOp)->getResult(0);
  }

  SmallVector<Value>
  unpackDsReadMatrixInsnRes(Location loc, RewriterBase &rewriter,
                            const MemDescType &tensorTy,
                            const DsReadMatrixInsnAttr &dsInsnAttr,
                            unsigned elemBitWidth, Value dsLoadedValue) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dsInsnRetTy = cast<VectorType>(dsLoadedValue.getType());
    auto dsInsnElemTy = dsInsnRetTy.getElementType();
    auto dsInsnNumOfElems = dsInsnRetTy.getNumElements();
    bool needCvt = false;
    unsigned numExtracts = 1;
    if (elemBitWidth == 8) {
      numExtracts = dsInsnElemTy.getIntOrFloatBitWidth() / elemBitWidth;
    }

    SmallVector<Value> retValues;
    auto elemTy = typeConverter->convertType(tensorTy.getElementType());
    for (int elemId = 0; elemId < dsInsnNumOfElems; ++elemId) {
      Value elemVal =
          b.extract_element(dsInsnElemTy, dsLoadedValue, b.i32_val(elemId));

      if (numExtracts != 1) {
        auto vecVal = b.bitcast(elemVal, vec_ty(elemTy, numExtracts));
        for (int extractId = 0; extractId < numExtracts; ++extractId) {
          Value subElemVal =
              b.extract_element(elemTy, vecVal, b.i32_val(extractId));
          retValues.push_back(subElemVal);
        }
      } else {
        retValues.push_back(elemVal);
      }
    }
    return retValues;
  }
};

} // namespace

namespace mlir::triton::HCU {
void populateMLSOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 const TargetInfo &targetInfo,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<MLSLocalAllocOpConversion, MLSMatrixLoadToLocalOpConversion,
               MLSLocalLoadOpConversion>(typeConverter, targetInfo,
                                         axisInfoAnalysis, benefit);
}
} // namespace mlir::triton::HCU
