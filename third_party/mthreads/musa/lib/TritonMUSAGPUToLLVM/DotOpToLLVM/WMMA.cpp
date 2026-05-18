#include "Dialect/MUSA/IR/Dialect.h"
#include "DotOpToLLVM.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MMAEncodingUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
static SmallVector<Value>
packElemsToI32(TritonLLVMOpBuilder &b, Location loc, ArrayRef<Value> elems,
               Type elemTy, const LLVMTypeConverter *typeConverter) {
  Type packElemTy = elemTy;
  if (!elems.empty())
    packElemTy = elems.front().getType();
  else if (typeConverter) {
    Type converted = typeConverter->convertType(elemTy);
    if (converted)
      packElemTy = converted;
  }

  const unsigned bitwidth = packElemTy.getIntOrFloatBitWidth();
  Type i32Ty = b.builder->getI32Type();
  SmallVector<Value> packed;
  if (bitwidth == 32) {
    packed.reserve(elems.size());
    for (Value v : elems) {
      Value normalized =
          (v.getType() == packElemTy) ? v : b.bitcast(v, packElemTy);
      Value i32Val = normalized.getType().isInteger(32)
                         ? normalized
                         : b.bitcast(normalized, i32Ty);
      packed.push_back(i32Val);
    }
    return packed;
  }

  if (bitwidth == 16) {
    if (elems.size() % 2 != 0)
      llvm::report_fatal_error("WMMA: expected even number of 16-bit elements");
    Type vecTy = vec_ty(packElemTy, 2);
    for (size_t i = 0; i < elems.size(); i += 2) {
      Value pack = b.undef(vecTy);
      Value e0 = (elems[i].getType() == packElemTy)
                     ? elems[i]
                     : b.bitcast(elems[i], packElemTy);
      Value e1 = (elems[i + 1].getType() == packElemTy)
                     ? elems[i + 1]
                     : b.bitcast(elems[i + 1], packElemTy);
      pack = b.insert_element(vecTy, pack, e0, b.i32_val(0));
      pack = b.insert_element(vecTy, pack, e1, b.i32_val(1));
      packed.push_back(b.bitcast(pack, i32Ty));
    }
    return packed;
  }

  if (bitwidth == 8) {
    if (elems.size() % 4 != 0)
      llvm::report_fatal_error("WMMA: expected 4x8-bit packing");
    Type vecTy = vec_ty(packElemTy, 4);
    for (size_t i = 0; i < elems.size(); i += 4) {
      Value pack = b.undef(vecTy);
      Value e0 = (elems[i].getType() == packElemTy)
                     ? elems[i]
                     : b.bitcast(elems[i], packElemTy);
      Value e1 = (elems[i + 1].getType() == packElemTy)
                     ? elems[i + 1]
                     : b.bitcast(elems[i + 1], packElemTy);
      Value e2 = (elems[i + 2].getType() == packElemTy)
                     ? elems[i + 2]
                     : b.bitcast(elems[i + 2], packElemTy);
      Value e3 = (elems[i + 3].getType() == packElemTy)
                     ? elems[i + 3]
                     : b.bitcast(elems[i + 3], packElemTy);
      pack = b.insert_element(vecTy, pack, e0, b.i32_val(0));
      pack = b.insert_element(vecTy, pack, e1, b.i32_val(1));
      pack = b.insert_element(vecTy, pack, e2, b.i32_val(2));
      pack = b.insert_element(vecTy, pack, e3, b.i32_val(3));
      packed.push_back(b.bitcast(pack, i32Ty));
    }
    return packed;
  }

  llvm::report_fatal_error("WMMA: unsupported element width for packing");
}

static Value packToVector(TritonLLVMOpBuilder &b, Location loc, Type elemTy,
                          ArrayRef<Value> elems) {
  if (elems.size() == 1)
    return elems.front();
  Type vecTy = vec_ty(elemTy, elems.size());
  Value vec = b.undef(vecTy);
  for (auto [i, v] : llvm::enumerate(elems))
    vec = b.insert_element(vecTy, vec, v, b.i32_val(i));
  return vec;
}

static SmallVector<Value> unpackI32ToElems(TritonLLVMOpBuilder &b, Location loc,
                                           ArrayRef<Value> packed,
                                           Type dstElemTy) {
  const unsigned bitwidth = dstElemTy.getIntOrFloatBitWidth();
  SmallVector<Value> elems;
  if (bitwidth == 32) {
    elems.reserve(packed.size());
    for (Value v : packed) {
      elems.push_back(v.getType().isInteger(32) ? b.bitcast(v, dstElemTy)
                                                : b.bitcast(v, dstElemTy));
    }
    return elems;
  }
  if (bitwidth == 16) {
    Type vecTy = vec_ty(dstElemTy, 2);
    elems.reserve(packed.size() * 2);
    for (Value v : packed) {
      Value vec = b.bitcast(v, vecTy);
      elems.push_back(b.extract_element(dstElemTy, vec, b.i32_val(0)));
      elems.push_back(b.extract_element(dstElemTy, vec, b.i32_val(1)));
    }
    return elems;
  }
  if (bitwidth == 8) {
    Type vecTy = vec_ty(dstElemTy, 4);
    elems.reserve(packed.size() * 4);
    for (Value v : packed) {
      Value vec = b.bitcast(v, vecTy);
      elems.push_back(b.extract_element(dstElemTy, vec, b.i32_val(0)));
      elems.push_back(b.extract_element(dstElemTy, vec, b.i32_val(1)));
      elems.push_back(b.extract_element(dstElemTy, vec, b.i32_val(2)));
      elems.push_back(b.extract_element(dstElemTy, vec, b.i32_val(3)));
    }
    return elems;
  }
  llvm::report_fatal_error("WMMA: unsupported element width for unpacking");
}

static LinearLayout buildPH1WMMATileLayout(MLIRContext *ctx, unsigned rank,
                                           unsigned instM, unsigned instN) {
  auto outDimNames = standardOutDimNames(ctx, rank);
  bool hasBatch = rank == 3;
  StringAttr dimM = outDimNames[hasBatch ? 1 : 0];
  StringAttr dimN = outDimNames[hasBatch ? 2 : 1];

  LinearLayout tileLayout(
      {{str_attr("register"), {}},
       {str_attr("lane"), {{0, 1}, {0, 2}, {0, 4}, {1, 0}, {2, 0}}},
       {str_attr("warp"), {}},
       {str_attr("block"), {}}},
      {dimM, dimN});

  tileLayout *= LinearLayout::identity1D(instN / 8, str_attr("register"), dimN);
  tileLayout *= LinearLayout::identity1D(instM / 4, str_attr("register"), dimM);

  if (hasBatch) {
    tileLayout *=
        LinearLayout::identity1D(1, str_attr("register"), outDimNames[0]);
    tileLayout *= LinearLayout::identity1D(1, str_attr("lane"), outDimNames[0]);
  }

  return tileLayout;
}

} // namespace

namespace mlir::triton::MUSA {

static Value getWmmaOperandA(triton::DotOp op) { return op.getA(); }
static Value getWmmaOperandA(triton::musa::WmmaDotOp op) { return op.getA(); }
static Value getWmmaOperandB(triton::DotOp op) { return op.getB(); }
static Value getWmmaOperandB(triton::musa::WmmaDotOp op) { return op.getB(); }

static Value getWmmaAdaptorA(triton::DotOp::Adaptor adaptor) {
  return adaptor.getA();
}
static Value getWmmaAdaptorA(triton::musa::WmmaDotOp::Adaptor adaptor) {
  return adaptor.getA();
}
static Value getWmmaAdaptorB(triton::DotOp::Adaptor adaptor) {
  return adaptor.getB();
}
static Value getWmmaAdaptorB(triton::musa::WmmaDotOp::Adaptor adaptor) {
  return adaptor.getB();
}
static Value getWmmaAdaptorC(triton::DotOp::Adaptor adaptor) {
  return adaptor.getC();
}
static Value getWmmaAdaptorC(triton::musa::WmmaDotOp::Adaptor adaptor) {
  return adaptor.getC();
}

static Value getWmmaUseCValue(triton::DotOp, triton::DotOp::Adaptor) {
  return Value();
}
static Value getWmmaUseCValue(triton::musa::WmmaDotOp,
                              triton::musa::WmmaDotOp::Adaptor adaptor) {
  return adaptor.getUseC();
}

static triton::musa::SQMMALayout getWmmaLayoutA(triton::DotOp op) {
  return triton::musa::inferWmmaFragmentLayout(op.getA(), 0);
}
static triton::musa::SQMMALayout getWmmaLayoutA(triton::musa::WmmaDotOp op) {
  return op.getLayoutA();
}
static triton::musa::SQMMALayout getWmmaLayoutB(triton::DotOp op) {
  return triton::musa::inferWmmaFragmentLayout(op.getB(), 1);
}
static triton::musa::SQMMALayout getWmmaLayoutB(triton::musa::WmmaDotOp op) {
  return op.getLayoutB();
}

static Value materializeUseCFlag(Location loc, Value useC,
                                 ConversionPatternRewriter &rewriter) {
  return triton::musa::materializeUseCFlag(loc, useC, rewriter);
}

static Value selectAccumulatorVector(Location loc, Value useC, Value acc,
                                     Value zero,
                                     ConversionPatternRewriter &rewriter) {
  return triton::musa::selectAccumulatorValue(loc, useC, acc, zero, rewriter);
}

struct WmmaTileState {
  unsigned regBase;
  int batch;
  int mBase;
  int nBase;
};

struct WmmaTilePlan {
  SmallVector<WmmaTileState> tiles;
  unsigned numRepK;
  unsigned numRepN;
};

struct ContiguousRank2WmmaFastPathState {
  WmmaTilePlan tilePlan;
  SmallVector<Value> aElemsAll;
  SmallVector<Value> bElemsAll;
};

static std::optional<ContiguousRank2WmmaFastPathState>
buildContiguousRank2WmmaFastPathState(
    Location loc, Value adaptorA, Value adaptorB, RankedTensorType aTy,
    RankedTensorType dTy, const triton::musa::WmmaDotOperandContract &aContract,
    const triton::musa::WmmaDotOperandContract &bContract,
    triton::musa::SQMMALayout layoutA, triton::musa::SQMMALayout layoutB,
    unsigned instM, unsigned instN, unsigned instK, unsigned warpSize,
    ConversionPatternRewriter &rewriter) {
  if (dTy.getRank() != 2)
    return std::nullopt;
  if (layoutA != triton::musa::SQMMALayout::row ||
      layoutB != triton::musa::SQMMALayout::col)
    return std::nullopt;
  if (aContract.rank != 2 || bContract.rank != 2)
    return std::nullopt;
  if (aContract.dotEncoding.getParent() != dTy.getEncoding() ||
      bContract.dotEncoding.getParent() != dTy.getEncoding())
    return std::nullopt;

  const unsigned aElemsPerThread = (instM * instK) / warpSize;
  const unsigned bElemsPerThread = (instN * instK) / warpSize;
  if (aElemsPerThread == 0 || bElemsPerThread == 0)
    return std::nullopt;

  ContiguousRank2WmmaFastPathState state;
  state.tilePlan.numRepK = std::max(
      1u, ceil<unsigned>(static_cast<unsigned>(aTy.getShape().back()), instK));
  state.aElemsAll = ::mlir::unpackLLElements(loc, adaptorA, rewriter);
  state.bElemsAll = ::mlir::unpackLLElements(loc, adaptorB, rewriter);
  if (state.aElemsAll.size() % aElemsPerThread != 0 ||
      state.bElemsAll.size() % bElemsPerThread != 0)
    return std::nullopt;
  unsigned aChunks = state.aElemsAll.size() / aElemsPerThread;
  unsigned bChunks = state.bElemsAll.size() / bElemsPerThread;
  if (aChunks % state.tilePlan.numRepK != 0 ||
      bChunks % state.tilePlan.numRepK != 0)
    return std::nullopt;
  unsigned numRepM = aChunks / state.tilePlan.numRepK;
  state.tilePlan.numRepN = bChunks / state.tilePlan.numRepK;
  if (numRepM == 0 || state.tilePlan.numRepN == 0)
    return std::nullopt;
  state.tilePlan.tiles.reserve(numRepM * state.tilePlan.numRepN);
  for (unsigned mTile = 0; mTile < numRepM; ++mTile) {
    for (unsigned nTile = 0; nTile < state.tilePlan.numRepN; ++nTile) {
      unsigned cTileIdx = mTile * state.tilePlan.numRepN + nTile;
      state.tilePlan.tiles.push_back({cTileIdx, 0,
                                      static_cast<int>(mTile * instM),
                                      static_cast<int>(nTile * instN)});
    }
  }

  return state;
}

static FailureOr<WmmaTilePlan>
buildGenericWmmaTilePlan(MLIRContext *ctx, const LinearLayout &cLinearLayout,
                         unsigned rank, unsigned instM, unsigned instN,
                         unsigned cElemsPerThread, unsigned fcSize) {
  auto tileLayout = buildPH1WMMATileLayout(ctx, rank, instM, instN);
  auto quot = divideLeft(cLinearLayout, tileLayout);
  if (!quot)
    return failure();
  auto repLayout = zerosLike(tileLayout) * *quot;

  auto kRegister = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  unsigned repRegs = repLayout.getInDimSize(kRegister);
  if (repRegs != fcSize)
    return failure();

  WmmaTilePlan plan;
  plan.numRepK = 0;
  plan.numRepN = 0;
  plan.tiles.reserve(repRegs / cElemsPerThread);
  for (unsigned regBase = 0; regBase < repRegs; regBase += cElemsPerThread) {
    SmallVector<std::pair<StringAttr, int32_t>, 4> repCoords = {
        {kRegister, static_cast<int32_t>(regBase)},
        {kLane, 0},
        {kWarp, 0},
        {kBlock, 0}};
    auto coords = repLayout.apply(repCoords);
    int batch = 0;
    int mBase = 0;
    int nBase = 0;
    if (rank == 3) {
      batch = coords[0].second;
      mBase = coords[1].second;
      nBase = coords[2].second;
    } else {
      mBase = coords[0].second;
      nBase = coords[1].second;
    }
    plan.tiles.push_back({regBase, batch, mBase, nBase});
  }
  return plan;
}

static SmallVector<Value> extractContiguousRank2WmmaOperandChunk(
    Location loc, ArrayRef<Value> elemsAll, unsigned chunkIdx,
    unsigned elemsPerThread, unsigned validElems, Type elemTy,
    const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) {
  SmallVector<Value> elems;
  elems.reserve(elemsPerThread);
  Type llvmElemTy = typeConverter->convertType(elemTy);
  Value zero = LLVM::ZeroOp::create(rewriter, loc, llvmElemTy);
  unsigned base = chunkIdx * elemsPerThread;
  for (unsigned i = 0; i < elemsPerThread; ++i) {
    if (i < validElems && base + i < elemsAll.size())
      elems.push_back(elemsAll[base + i]);
    else
      elems.push_back(zero);
  }
  return elems;
}

static std::optional<Value> buildContiguousRank2PackedWmmaOperandA(
    Location loc, const ContiguousRank2WmmaFastPathState &state, unsigned mBase,
    unsigned kTile, unsigned instM, unsigned validK, unsigned warpSize,
    unsigned elemsPerThread, Type elemTy, TritonLLVMOpBuilder &b,
    const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) {
  if (mBase % instM != 0)
    return std::nullopt;
  unsigned mTile = mBase / instM;
  unsigned numRepM = state.tilePlan.tiles.size() / state.tilePlan.numRepN;
  if (mTile >= numRepM || kTile >= state.tilePlan.numRepK)
    return std::nullopt;
  unsigned chunkIdx = mTile * state.tilePlan.numRepK + kTile;
  unsigned validElems = (instM * validK) / warpSize;
  auto elems = extractContiguousRank2WmmaOperandChunk(
      loc, state.aElemsAll, chunkIdx, elemsPerThread, validElems, elemTy,
      typeConverter, rewriter);
  auto packed = packElemsToI32(b, loc, elems, elemTy, typeConverter);
  return packToVector(b, loc, rewriter.getI32Type(), packed);
}

static std::optional<Value> buildContiguousRank2PackedWmmaOperandB(
    Location loc, const ContiguousRank2WmmaFastPathState &state, unsigned nBase,
    unsigned kTile, unsigned instN, unsigned validK, unsigned warpSize,
    unsigned elemsPerThread, Type elemTy, TritonLLVMOpBuilder &b,
    const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) {
  if (nBase % instN != 0)
    return std::nullopt;
  unsigned nTile = nBase / instN;
  if (nTile >= state.tilePlan.numRepN || kTile >= state.tilePlan.numRepK)
    return std::nullopt;
  unsigned chunkIdx = kTile * state.tilePlan.numRepN + nTile;
  unsigned validElems = (instN * validK) / warpSize;
  auto elems = extractContiguousRank2WmmaOperandChunk(
      loc, state.bElemsAll, chunkIdx, elemsPerThread, validElems, elemTy,
      typeConverter, rewriter);
  auto packed = packElemsToI32(b, loc, elems, elemTy, typeConverter);
  return packToVector(b, loc, rewriter.getI32Type(), packed);
}

static std::optional<Value> buildPackedWmmaOperandAForTile(
    Location loc,
    const std::optional<ContiguousRank2WmmaFastPathState>
        &contiguousRank2FastPath,
    Value adaptorA, const triton::musa::WmmaDotOperandContract &aContract,
    int batch, int mBase, unsigned kTile, unsigned instM, unsigned instK,
    unsigned warpSize, unsigned elemsPerThread, unsigned kPadding,
    unsigned validK, Type elemTy, Type packedElemTy, TritonLLVMOpBuilder &b,
    const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) {
  if (contiguousRank2FastPath)
    return buildContiguousRank2PackedWmmaOperandA(
        loc, *contiguousRank2FastPath, mBase, kTile, instM, validK, warpSize,
        elemsPerThread, elemTy, b, typeConverter, rewriter);

  Value aVec = triton::musa::extractWmmaOperandVectorFromContract(
      loc, adaptorA, aContract, typeConverter, rewriter, batch, mBase, kTile,
      instK, elemsPerThread, kPadding, elemTy);
  auto aElems = unpackLLVector(loc, aVec, rewriter);
  auto aPacked =
      packElemsToI32(b, loc, aElems, aElems.front().getType(), typeConverter);
  return packToVector(b, loc, packedElemTy, aPacked);
}

static std::optional<Value> buildPackedWmmaOperandBForTile(
    Location loc,
    const std::optional<ContiguousRank2WmmaFastPathState>
        &contiguousRank2FastPath,
    Value adaptorB, const triton::musa::WmmaDotOperandContract &bContract,
    int batch, int nBase, unsigned kTile, unsigned instN, unsigned instK,
    unsigned warpSize, unsigned elemsPerThread, unsigned kPadding,
    unsigned validK, Type elemTy, Type packedElemTy, TritonLLVMOpBuilder &b,
    const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) {
  if (contiguousRank2FastPath)
    return buildContiguousRank2PackedWmmaOperandB(
        loc, *contiguousRank2FastPath, nBase, kTile, instN, validK, warpSize,
        elemsPerThread, elemTy, b, typeConverter, rewriter);

  Value bVec = triton::musa::extractWmmaOperandVectorFromContract(
      loc, adaptorB, bContract, typeConverter, rewriter, batch, nBase, kTile,
      instK, elemsPerThread, kPadding, elemTy);
  auto bElems = unpackLLVector(loc, bVec, rewriter);
  auto bPacked =
      packElemsToI32(b, loc, bElems, bElems.front().getType(), typeConverter);
  return packToVector(b, loc, packedElemTy, bPacked);
}

template <typename DotLikeOp, typename DotLikeAdaptor>
LogicalResult convertWMMADotImpl(DotLikeOp op, DotLikeAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();

  auto dTy = cast<RankedTensorType>(op.getResult().getType());
  auto mmaEnc = dyn_cast<MUSAWmmaEncodingAttr>(dTy.getEncoding());
  if (!mmaEnc)
    return op.emitError("MUSA WMMA: expected #ttg.musa_wmma result encoding");
  if (!triton::musa::supportsMusaWmmaEncoding(mmaEnc))
    return op.emitError("MUSA WMMA: unsupported result encoding version");

  Value opA = getWmmaOperandA(op);
  Value opB = getWmmaOperandB(op);
  Value adaptorA = getWmmaAdaptorA(adaptor);
  Value adaptorB = getWmmaAdaptorB(adaptor);
  Value adaptorC = getWmmaAdaptorC(adaptor);
  auto aTy = cast<RankedTensorType>(opA.getType());
  auto bTy = cast<RankedTensorType>(opB.getType());
  auto aElemTy = aTy.getElementType();
  auto bElemTy = bTy.getElementType();
  auto dElemTy = dTy.getElementType();
  auto aContract = triton::musa::resolveWmmaDotOperandContract(aTy, 0);
  auto bContract = triton::musa::resolveWmmaDotOperandContract(bTy, 1);
  if (failed(aContract) || failed(bContract))
    return op.emitError("MUSA WMMA operands must use DotOperandEncodingAttr");
  if (aContract->dotEncoding.getParent() != dTy.getEncoding() ||
      bContract->dotEncoding.getParent() != dTy.getEncoding())
    return op.emitError("MUSA WMMA operands must point to the same "
                        "#ttg.musa_wmma result encoding");
  Value useCFlag =
      materializeUseCFlag(loc, getWmmaUseCValue(op, adaptor), rewriter);
  std::optional<bool> useCConst = getBoolFromConstant(useCFlag);
  if (aElemTy != bElemTy)
    return op.emitError("MUSA WMMA requires A/B element types to match");

  auto instrShape = mmaEnc.getInstrShape();
  if (instrShape.size() != 3)
    return op.emitError("MUSA WMMA expects 3D instrShape");

  auto signature = triton::musa::lookupWmmaIntrinsic(aElemTy, instrShape);
  if (!signature)
    return op.emitError("MUSA WMMA: unsupported instrShape or element type");

  const unsigned instM = instrShape[0];
  const unsigned instN = instrShape[1];
  const unsigned instK = instrShape[2];
  auto layoutA = getWmmaLayoutA(op);
  auto layoutB = getWmmaLayoutB(op);

  const unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);
  auto contiguousRank2FastPath = buildContiguousRank2WmmaFastPathState(
      loc, adaptorA, adaptorB, aTy, dTy, *aContract, *bContract, layoutA,
      layoutB, instM, instN, instK, warpSize, rewriter);

  auto cLinearLayout = mmaEnc.toLinearLayout(dTy.getShape());

  auto fc = ::mlir::unpackLLElements(loc, adaptorC, rewriter);

  const unsigned aElemsPerThread = (instM * instK) / warpSize;
  const unsigned bElemsPerThread = (instN * instK) / warpSize;
  const unsigned cElemsPerThread = (instM * instN) / warpSize;

  if (aElemsPerThread == 0 || bElemsPerThread == 0 || cElemsPerThread == 0)
    return op.emitError("MUSA WMMA: invalid per-thread element size");

  unsigned kDim = static_cast<unsigned>(aTy.getShape().back());
  unsigned numRepK = std::max(1u, ceil<unsigned>(kDim, instK));

  unsigned accPackedLen = (cElemsPerThread * dElemTy.getIntOrFloatBitWidth());
  if (accPackedLen % 32 != 0)
    return op.emitError("MUSA WMMA: accumulator packing misaligned");
  accPackedLen /= 32;
  if (accPackedLen == 0)
    return op.emitError("MUSA WMMA: invalid accumulator packing size");

  Type accPackedElemTy = rewriter.getI32Type();
  Type accVecTy = accPackedLen == 1 ? accPackedElemTy
                                    : vec_ty(accPackedElemTy, accPackedLen);

  WmmaTilePlan tilePlan;
  tilePlan.numRepK = numRepK;
  tilePlan.numRepN = 0;
  if (contiguousRank2FastPath) {
    tilePlan = contiguousRank2FastPath->tilePlan;
    for (WmmaTileState &tile : tilePlan.tiles)
      tile.regBase *= cElemsPerThread;
    if (fc.size() != tilePlan.tiles.size() * cElemsPerThread)
      return op.emitError("MUSA WMMA: accumulator register size mismatch");
  } else {
    auto genericTilePlan =
        buildGenericWmmaTilePlan(ctx, cLinearLayout, dTy.getRank(), instM,
                                 instN, cElemsPerThread, fc.size());
    if (failed(genericTilePlan))
      return op.emitError("MUSA WMMA: failed to derive repetition layout");
    tilePlan = *genericTilePlan;
    tilePlan.numRepK = numRepK;
  }

  unsigned kPaddingA = 0;
  unsigned kPaddingB = 0;
  if (instK > kDim) {
    unsigned paddingFactor = instK / kDim;
    kPaddingA = aElemsPerThread - (aElemsPerThread / paddingFactor);
    kPaddingB = bElemsPerThread - (bElemsPerThread / paddingFactor);
  }

  for (const WmmaTileState &tileState : tilePlan.tiles) {
    SmallVector<Value> accElems(fc.begin() + tileState.regBase,
                                fc.begin() + tileState.regBase +
                                    cElemsPerThread);
    auto packedAccElems = packElemsToI32(
        b, loc, accElems, accElems.front().getType(), typeConverter);
    if (packedAccElems.size() != accPackedLen)
      return op.emitError("MUSA WMMA: accumulator pack size mismatch");
    Value accVec = packToVector(b, loc, accPackedElemTy, packedAccElems);
    if (!useCConst || !*useCConst) {
      Value zeroAcc = LLVM::ZeroOp::create(rewriter, loc, accVecTy);
      accVec =
          selectAccumulatorVector(loc, useCFlag, accVec, zeroAcc, rewriter);
    }

    for (unsigned kTile = 0; kTile < tilePlan.numRepK; ++kTile) {
      unsigned kOffset = kTile * instK;
      unsigned validK = (kOffset < kDim) ? std::min(instK, kDim - kOffset) : 0;
      auto aOp = buildPackedWmmaOperandAForTile(
          loc, contiguousRank2FastPath, adaptorA, *aContract, tileState.batch,
          tileState.mBase, kTile, instM, instK, warpSize, aElemsPerThread,
          kPaddingA, validK, aElemTy, accPackedElemTy, b, typeConverter,
          rewriter);
      if (!aOp)
        return op.emitError("MUSA WMMA: failed to materialize A fragment");
      auto bOp = buildPackedWmmaOperandBForTile(
          loc, contiguousRank2FastPath, adaptorB, *bContract, tileState.batch,
          tileState.nBase, kTile, instN, instK, warpSize, bElemsPerThread,
          kPaddingB, validK, bElemTy, accPackedElemTy, b, typeConverter,
          rewriter);
      if (!bOp)
        return op.emitError("MUSA WMMA: failed to materialize B fragment");
      SmallVector<Value> args = triton::musa::buildWmmaIntrinsicArgs(
          loc, *aOp, *bOp, accVec, layoutA, layoutB, *signature, rewriter);
      auto call = LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, signature->name, TypeRange{accVecTy}, args);
      accVec = call.getResult(0);
    }

    SmallVector<Value> accPackedElems;
    if (auto vecTy = dyn_cast<VectorType>(accVec.getType())) {
      for (unsigned i = 0; i < vecTy.getNumElements(); ++i)
        accPackedElems.push_back(
            b.extract_element(accPackedElemTy, accVec, b.i32_val(i)));
    } else {
      accPackedElems.push_back(accVec);
    }

    auto accUnpacked = unpackI32ToElems(b, loc, accPackedElems,
                                        typeConverter->convertType(dElemTy));
    if (accUnpacked.size() != cElemsPerThread)
      return op.emitError("MUSA WMMA: accumulator unpack size mismatch");
    for (unsigned i = 0; i < accUnpacked.size(); ++i)
      fc[tileState.regBase + i] = accUnpacked[i];
  }

  Value res = ::mlir::packLLElements(loc, typeConverter, fc, rewriter, dTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertWMMADot(triton::musa::WmmaDotOp op,
                             triton::musa::WmmaDotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter) {
  return convertWMMADotImpl(op, adaptor, typeConverter, rewriter);
}

} // namespace mlir::triton::MUSA
