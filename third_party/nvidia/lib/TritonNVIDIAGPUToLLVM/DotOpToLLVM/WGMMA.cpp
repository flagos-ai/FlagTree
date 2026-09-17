/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "MMAHelpers.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NVIDIA;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingTrait;

#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/IR/SMEMPlan.h"

static constexpr llvm::StringLiteral
    kTleExplicitWgmmaCommitAttr("tle.explicit_wgmma_commit");
static constexpr llvm::StringLiteral
    kTleWgmmaAccumulatorChainCAttr("tle.wgmma_accumulator_chain_c");
static constexpr llvm::StringLiteral kTleWgmmaActiveNAttr("tle.wgmma_active_n");
static constexpr llvm::StringLiteral kTleWgmmaActiveKAttr("tle.wgmma_active_k");
static constexpr llvm::StringLiteral
    kTleTiledSMEMOperandBAttr("tle.tiled_smem_operand_b");
static constexpr llvm::StringLiteral
    kTleTiledSMEMLogicalRowsAttr("tle.tiled_smem_logical_rows");
static constexpr llvm::StringLiteral
    kTleTiledSMEMLogicalColsAttr("tle.tiled_smem_logical_cols");
static constexpr llvm::StringLiteral
    kTleTiledSMEMStorageTileShapeAttr("tle.tiled_smem_storage_tile_shape");

struct TleWgmmaOperand {
  Value value;
  std::optional<int64_t> descriptorImm;
};

// The upstream loader still owns descriptor localization and swizzle phase.
// The TLE plan supplies the integer outer strides that do not fit LinearLayout.
static DotOpMmaSmemLoader
buildTlePlannedSMEMLoader(Location loc, RewriterBase &rewriter, Value smemBase,
                          const tle::WGMMAOperandPlan &plan) {
  const auto &storage = plan.storage;
  auto encoding = triton::gpu::NVMMASharedEncodingAttr::get(
      rewriter.getContext(), storage.encoding.getSwizzlingByteWidth(),
      plan.isKMajor(), storage.encoding.getElementBitWidth(), false,
      storage.encoding.getCTALayout());
  SmallVector<int64_t> shape{storage.tileRows, storage.tileCols};
  if (plan.viewTransposed)
    std::swap(shape[0], shape[1]);
  auto layout =
      triton::gpu::nvmmaSharedToLinearLayout(shape, encoding).pseudoinvert();
  SMEMDescriptor smemDescriptor{};
  smemDescriptor.leadDimensionBaseOffset = plan.lboBytes / 16;
  smemDescriptor.strideDimensionBaseOffset = plan.sboBytes / 16;
  int width = encoding.getSwizzlingByteWidth();
  smemDescriptor.swizzlingMode = width == 128  ? 1
                                 : width == 64 ? 2
                                 : width == 32 ? 3
                                               : 0;
  MMASMEMDescriptor descriptor{smemDescriptor, width,
                               static_cast<int>(encoding.getElementBitWidth()),
                               plan.isKMajor(), false};
  TritonLLVMOpBuilder b(loc, rewriter);
  Value address = b.ptrtoint(rewriter.getI32Type(), smemBase);
  Value base = b.zext(rewriter.getI64Type(),
                      b.and_(b.lshr(address, b.i32_val(4)), b.i32_val(0x3FFF)));
  return DotOpMmaSmemLoader(descriptor, base, layout);
}

#endif

triton::nvgpu::WGMMAEltType getMmaRetType(Value d) {
  auto dTy = cast<RankedTensorType>(d.getType()).getElementType();
  if (dTy.isF32()) {
    return triton::nvgpu::WGMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::nvgpu::WGMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

triton::nvgpu::WGMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  if (aTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::nvgpu::WGMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::nvgpu::WGMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::nvgpu::WGMMAEltType::s8;
  } else if (llvm::isa<Float8E5M2Type>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e5m2;
  } else if (llvm::isa<Float8E4M3FNType>(aTy)) {
    return triton::nvgpu::WGMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
//
// `elements` contains all loaded register values for operand A.
// This consists of operand A for possibly multiple wgmma instructions.
// For each wgmma, each warp in a warp group feeds a single "warp matrix"
// Each warp matrix consists of 2x2 "quads".
// Each thread holds several elements in each quad. Right before a wgmma,
// the sum of bitwidth of
// the elements in each quad should add up to 32.
//
// These values are stored unrolled in `elements`.
// The ordering of dimensions is as follows:
// batch (only 1 batch for Hopper currently)
// matM (m-index of the "warp matrix")
// matK (k-index of the "warp matrix")
// quadK (k-index of the "quad" in the core matrix)
// quadM (m-index of the "quad" in the core matrix)
// vecIdx (index of the element in the quad; this is always along the k-dim)
//
// This ordering is decided when a tensor in DotOpEnc is lowered into llvm.
// For WGMMA this happens in both SharedToDotOperand and MMAToDotOperand.
// Thus, both lowerings must obey this above ordering for the below code to be
// correct.
llvm::SmallVector<Value> loadReg(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const SmallVector<Value> &elements,
                                 int startIndex, int numElements,
                                 Operation *insertBefore) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertBefore);

  if (!elements[0].getType().isIntOrFloat() ||
      elements[0].getType().getIntOrFloatBitWidth() >= 32) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  Type elementType = elements[0].getType();
  int numElemsPer32Bits = 32 / elementType.getIntOrFloatBitWidth();

  // For FP16 and BF16 we need to pack accumulator into 32-bit integers.
  int num32BitValues = numElements / numElemsPer32Bits;
  llvm::SmallVector<Value> mmaOut(num32BitValues);
  Type packTy = vec_ty(elementType, numElemsPer32Bits);
  for (int i = 0; i < num32BitValues; ++i) {
    Value pack = LLVM::UndefOp::create(rewriter, loc, packTy);
    for (int j = 0; j < numElemsPer32Bits; ++j) {
      Value element = elements[startIndex + i * numElemsPer32Bits + j];
      pack = b.insert_element(packTy, pack, element, b.i32_val(j));
    }
    pack = b.bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumulator is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = b.bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(0)));
    results.push_back(
        b.extract_element(rewriter.getF16Type(), elem, b.i32_val(1)));
  }
  return results;
}

static Value faddAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                            Value a, Value b) {
  int numEl = cast<LLVM::LLVMStructType>(a.getType()).getBody().size();
  Value newStruct = LLVM::UndefOp::create(rewriter, loc, a.getType());
  for (int i = 0; i < numEl; ++i) {
    Value lhs = LLVM::ExtractValueOp::create(rewriter, loc, a, i);
    Value rhs = LLVM::ExtractValueOp::create(rewriter, loc, b, i);
    Value add = LLVM::FAddOp::create(rewriter, loc, lhs, rhs);
    newStruct = LLVM::InsertValueOp::create(rewriter, loc, newStruct, add, i);
  }
  return newStruct;
}

static SmallVector<Value> emitWait(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> acc,
                                   int pendings) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Type> types(acc.size(), acc[0].getType());
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  Value llvmStruct = LLVM::UndefOp::create(rewriter, loc, structTy);
  int i = 0;
  for (Value v : acc) {
    llvmStruct = b.insert_val(structTy, llvmStruct, v, i++);
  }
  Value res = triton::nvgpu::WGMMAWaitGroupOp::create(rewriter, loc, llvmStruct,
                                                      pendings);
  SmallVector<Value> results;
  for (int i = 0; i < acc.size(); ++i) {
    results.push_back(b.extract_val(types[0], res, i));
  }
  return results;
}

#ifdef __TLE__
// Planned IR carries the storage shape, encoding and logical view together.
// Only older, pre-plan TTGIR needs to infer the view from its carrier encoding.
static FailureOr<tle::WGMMAOperandPlan>
getTleWGMMAOperandBPlan(Operation *op, MemDescType type, unsigned instructionK,
                        unsigned maxN) {
  auto attr = op->getAttrOfType<DictionaryAttr>(tle::kWGMMAOperandBPlanAttr);
  auto plan = tle::WGMMAOperandPlan::fromAttr(attr);
  if (!op->hasAttr(tle::kWGMMAOperandBPlanAttr)) {
    auto rows = op->getAttrOfType<IntegerAttr>(kTleTiledSMEMLogicalRowsAttr);
    auto cols = op->getAttrOfType<IntegerAttr>(kTleTiledSMEMLogicalColsAttr);
    auto tile =
        op->getAttrOfType<DenseI32ArrayAttr>(kTleTiledSMEMStorageTileShapeAttr);
    if (!rows || !cols || !tile || tile.size() != 2)
      return op->emitOpError(
          "tiled SMEM operand B requires logical extents and a two-dimensional "
          "storage tile shape");
    SmallVector<int64_t> shape{rows.getInt(), cols.getInt()};
    SmallVector<int64_t> tileShape(tile.asArrayRef().begin(),
                                   tile.asArrayRef().end());
    if (tileShape[0] <= 0 || tileShape[1] <= 0 || shape[0] % tileShape[0] ||
        shape[1] % tileShape[1])
      return op->emitOpError(
          "tiled SMEM storage tile must divide its logical extents");
    auto carrier =
        cast<triton::gpu::NVMMASharedEncodingAttr>(type.getEncoding());
    auto encoding = triton::gpu::NVMMASharedEncodingAttr::get(
        op->getContext(), tileShape, {1, 0}, carrier.getCTALayout(),
        type.getElementType(), false);
    if (!tle::isLegalSMEMTile(shape, tileShape, encoding))
      return op->emitOpError("invalid WGMMA shared-memory storage tile");
    tle::SMEMLayoutPlan storage(shape, tileShape, encoding);
    plan = tle::planWGMMAOperand(storage, carrier.getTransposed(), instructionK,
                                 maxN, type.getElementType().isInteger(8));
  }
  if (!plan || maxN % plan->instructionN)
    return op->emitOpError("invalid or incompatible WGMMA shared-memory plan");
  auto verified =
      tle::planWGMMAOperand(plan->storage, plan->viewTransposed, instructionK,
                            maxN, type.getElementType().isInteger(8));
  if (!verified || verified->getAttr() != plan->getAttr() ||
      plan->storage.encoding.getElementBitWidth() !=
          type.getElementType().getIntOrFloatBitWidth())
    return op->emitOpError("WGMMA descriptor does not match its shared layout");
  return *plan;
}

static LogicalResult
convertLogicalDot(const LLVMTypeConverter *typeConverter,
                  ConversionPatternRewriter &rewriter, Location loc,
                  Operation *op, Value a, Value b, Value c, Value d,
                  Value useCOperand, Value loadedA, Value loadedB,
                  Value loadedC, bool allowTF32, bool needsPartialAccumulator,
                  uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<triton::gpu::MemDescType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  bool aInShared = isa<SharedEncodingTrait>(aTensorTy.getEncoding());
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());
  bool tiledSMEMOperandB = op->hasAttr(tle::kWGMMAOperandBPlanAttr) ||
                           op->hasAttr(kTleTiledSMEMOperandBAttr);
  std::optional<SharedMemoryObject> smemObjA;
  Value baseA;
  if (aInShared) {
    baseA = getOffsetedBase(loadedA, cast<MemDescType>(aTensorTy),
                            typeConverter, rewriter, loc);
  }
  Value baseB;
  if (tiledSMEMOperandB) {
    auto llvmElemTy = typeConverter->convertType(bTensorTy.getElementType());
    baseB = LLVM::getSharedMemoryObjectFromStruct(loc, loadedB, llvmElemTy,
                                                  rewriter)
                .getBase();
  } else {
    baseB = getOffsetedBase(loadedB, cast<MemDescType>(bTensorTy),
                            typeConverter, rewriter, loc);
  }
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrMNK = mmaEncoding.getInstrShape();
  unsigned physicalN = instrMNK[1];
  unsigned physicalAccSize = 2 * (physicalN / 4);
  unsigned M = 4 * instrMNK[0];
  unsigned N = physicalN;
  unsigned K = instrMNK[2];
  IntegerAttr activeNAttr =
      op->getAttrOfType<IntegerAttr>(kTleWgmmaActiveNAttr);
  IntegerAttr activeKAttr =
      op->getAttrOfType<IntegerAttr>(kTleWgmmaActiveKAttr);
  if (activeNAttr && activeKAttr)
    return op->emitOpError(
        "tle.wgmma_active_n and tle.wgmma_active_k cannot be combined");
  if (activeNAttr) {
    assert(activeNAttr.getInt() > 0 && activeNAttr.getInt() % 8 == 0 &&
           activeNAttr.getInt() <= physicalN &&
           "active_n verifier must restrict codegen to a positive multiple "
           "of 8 within one physical N carrier");
    N = activeNAttr.getInt();
  }
  unsigned wgmmaAccSize = 2 * (N / 4);
  bool zeroAcc = isZeroConst(c);
  if (activeNAttr && op->hasAttr(kTleWgmmaAccumulatorChainCAttr))
    return op->emitOpError(
        "active_n does not support tle.wgmma_accumulator_chain_c");
  bool reuseAccumulatorChainC =
      !zeroAcc && op->hasAttr(kTleWgmmaAccumulatorChainCAttr);
  auto warpSize = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTATile =
      SmallVector<unsigned>{instrMNK[0] * warpSize[0], physicalN * warpSize[1]};
  unsigned mmaSizeM = shapePerCTATile[0];
  unsigned mmaSizeN = shapePerCTATile[1];
  unsigned mmaSizeK = instrMNK[2];
  int numRepM = ceil<unsigned>(dShapePerCTA[0], mmaSizeM);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], mmaSizeN);
  int physicalNumRepK = ceil<unsigned>(aTensorTy.getShape()[1], mmaSizeK);
  int activeNumRepK = physicalNumRepK;
  if (activeKAttr) {
    int64_t physicalK = aTensorTy.getShape()[1];
    int64_t activeK = activeKAttr.getInt();
    if (physicalK % mmaSizeK != 0 || activeK <= 0 || activeK > physicalK ||
        activeK % mmaSizeK != 0)
      return op->emitOpError(
          "tle.wgmma_active_k must select complete WGMMA K instructions "
          "within the physical carrier");
    activeNumRepK = activeK / mmaSizeK;
  }
  if (reuseAccumulatorChainC && (numRepM != 1 || numRepN != 1))
    return op->emitOpError(
        "cannot reuse WGMMA accumulator chain C for multi-tile results");
  DotOpMmaSmemLoader aLoader;
  SmallVector<Value> structA;
  auto warpGroups = {warpSize[0] / 4, warpSize[1]};
  bool transA = false;
  if (aInShared) {
    aLoader =
        DotOpMmaSmemLoader::build(loc, rewriter, cast<MemDescType>(aTensorTy),
                                  baseA, {M, K}, 0, 3, false, dTensorTy);
    transA = aLoader.getDescriptor().transposed;
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaSmemLoader bLoader;
  std::optional<tle::WGMMAOperandPlan> tiledPlan;
  if (tiledSMEMOperandB) {
    auto plan = getTleWGMMAOperandBPlan(op, bTensorTy, K, N);
    if (failed(plan))
      return failure();
    tiledPlan = *plan;
    if (activeKAttr && activeKAttr.getInt() != tiledPlan->logicalK())
      return op->emitOpError(
          "tle.wgmma_active_k must equal the tiled SMEM logical K extent");
    if (activeNAttr && activeNAttr.getInt() != tiledPlan->logicalN())
      return op->emitOpError(
          "tle.wgmma_active_n must equal the tiled SMEM logical N extent");
    bLoader = buildTlePlannedSMEMLoader(loc, rewriter, baseB, *tiledPlan);
  } else {
    bLoader = DotOpMmaSmemLoader::build(loc, rewriter, bTensorTy, baseB,
                                        {K, physicalN}, 1, 3, false, dTensorTy);
  }
  bool bIsKMajor = bLoader.getDescriptor().transposed;

  SmallVector<Value> fc;
  if (!reuseAccumulatorChainC || activeNAttr)
    fc = unpackLLElements(loc, loadedC, rewriter);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  bool supportsTransposeOperands =
      (eltTypeA == triton::nvgpu::WGMMAEltType::f16 &&
       eltTypeB == triton::nvgpu::WGMMAEltType::f16) ||
      (eltTypeA == triton::nvgpu::WGMMAEltType::bf16 &&
       eltTypeB == triton::nvgpu::WGMMAEltType::bf16);
  if (!supportsTransposeOperands && (transA || !bIsKMajor))
    return op->emitOpError(
        "TF32, FP8, and int8 WGMMA require row-major A and column-major B "
        "descriptors because their PTX instructions have no transpose "
        "operands");

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = bIsKMajor
                                           ? triton::nvgpu::WGMMALayout::col
                                           : triton::nvgpu::WGMMALayout::row;

  unsigned wgmmaInstructionN = tiledPlan ? tiledPlan->instructionN : N;
  unsigned activeNFragments = N / wgmmaInstructionN;
  unsigned instructionAccSize = 2 * (wgmmaInstructionN / 4);

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  // TLE kernels often carry extra live values around WGMMA regions. Keep all
  // ordinary register prep that feeds the first WGMMA outside the fence-to-MMA
  // window so ptxas sees a clean WGMMA pipeline stage.
  // Materialize the initial C accumulator before wgmma.fence.  Otherwise LLVM
  // may leave struct packing/copies between wgmma.fence and the first
  // wgmma.mma_async, which ptxas treats as accumulator definitions inside the
  // WGMMA pipeline stage.
  SmallVector<Type> accTypes;
  SmallVector<Value> initialAccumulators;
  SmallVector<Value> initialUseC;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      for (unsigned fragment = 0; fragment < activeNFragments; ++fragment) {
        Value d;
        Value useC;
        LLVM::LLVMStructType accTy;
        if (reuseAccumulatorChainC && !activeNAttr) {
          accTy = cast<LLVM::LLVMStructType>(loadedC.getType());
          d = loadedC;
          useC = tb.i1_val(1);
        } else {
          llvm::SmallVector<Value> mmaOut =
              loadReg(rewriter, loc, fc,
                      (m * numRepN + n) * physicalAccSize +
                          fragment * instructionAccSize,
                      instructionAccSize, op);
          llvm::SmallVector<Type> elemTypes;
          for (Value accEl : mmaOut)
            elemTypes.push_back(accEl.getType());
          accTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                   elemTypes);
          useC = tb.i1_val(0);
          if (!zeroAcc) {
            d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
            useC = tb.i1_val(1);
          }
        }
        if (useCOperand)
          useC = tb.and_(useC, useCOperand);
        accTypes.push_back(accTy);
        initialAccumulators.push_back(d);
        initialUseC.push_back(useC);
      }
    }
  }

  // Keep shared-memory descriptors near their WGMMA use. Treating them as
  // ordinary pure integer SSA lets LLVM hoist them across the full dot region,
  // which can make ptxas spill descriptor registers under high pressure.
  auto buildRegisterA = [&](int m, int k,
                            Operation *insertBefore) -> TleWgmmaOperand {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertBefore);
    auto aDotOpEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    assert(aDotOpEnc.getKWidth() == 32 / aTensorTy.getElementTypeBitWidth());

    unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
    llvm::SmallVector<Value> regA =
        loadReg(rewriter, loc, structA, (m * physicalNumRepK + k) * regASize,
                regASize, insertBefore);
    auto regATy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        SmallVector<Type>(regA.size(), regA[0].getType()));
    return {packLLElements(loc, typeConverter, regA, rewriter, regATy),
            std::nullopt};
  };

  auto buildSharedA = [&](int m, int k) -> TleWgmmaOperand {
    LocalizedSMEMDescriptor desc = aLoader.localizedSmemLoad(m, k);
    return {desc.baseb128, desc.descriptorImm};
  };

  auto buildSharedB = [&](int k, int n) -> TleWgmmaOperand {
    if (!tiledSMEMOperandB) {
      LocalizedSMEMDescriptor desc = bLoader.localizedSmemLoad(k, n);
      return {desc.baseb128, desc.descriptorImm};
    }

    const auto &storage = tiledPlan->storage;
    int64_t row = tiledPlan->viewTransposed ? n : k;
    int64_t col = tiledPlan->viewTransposed ? k : n;
    int64_t tile =
        storage.tileIndex(row / storage.tileRows, col / storage.tileCols);
    int localK = tiledPlan->viewTransposed ? col % storage.tileCols
                                           : row % storage.tileRows;
    int localN = tiledPlan->viewTransposed ? row % storage.tileRows
                                           : col % storage.tileCols;
    LocalizedSMEMDescriptor desc = bLoader.localizedSmemLoad(localK, localN);
    if (tile)
      desc.baseb128 =
          tb.add(desc.baseb128, tb.i64_val(tile * storage.tileBytes() / 16));
    return {desc.baseb128, desc.descriptorImm};
  };

  auto setDescriptorAttrs = [&](triton::nvgpu::WGMMAOp mmaOp,
                                const TleWgmmaOperand &a,
                                const TleWgmmaOperand &b) {
    if (a.descriptorImm)
      mmaOp->setAttr(nvgpu::kTleWgmmaOperandADescImmAttr,
                     rewriter.getI64IntegerAttr(*a.descriptorImm));
    if (b.descriptorImm)
      mmaOp->setAttr(nvgpu::kTleWgmmaOperandBDescImmAttr,
                     rewriter.getI64IntegerAttr(*b.descriptorImm));
  };

  TleWgmmaOperand firstA;
  TleWgmmaOperand firstB;
  if (numRepM > 0 && numRepN > 0 && activeNumRepK > 0) {
    // The first WGMMA reuses operands materialized before the fence; later
    // descriptors are still localized at their individual WGMMA use sites.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (aInShared) {
      firstA = buildSharedA(/*m=*/0, /*k=*/0);
    } else {
      firstA = buildRegisterA(/*m=*/0, /*k=*/0, op);
    }
    firstB = buildSharedB(/*k=*/0, /*n=*/0);
  }

  Operation *startSequence = op;
  if (!reuseAccumulatorChainC)
    startSequence = NVVM::WgmmaFenceAlignedOp::create(rewriter, loc);
  SmallVector<Value> mmaResults;
  unsigned tileIdx = 0;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      for (unsigned fragment = 0; fragment < activeNFragments; ++fragment) {
        auto accTy = accTypes[tileIdx];
        Value d = initialAccumulators[tileIdx];
        Value useC = initialUseC[tileIdx++];
        uint32_t numLowPrecisionAcc = 0;
        Value partialAcc;
        for (int k = 0; k < activeNumRepK; ++k) {
          Value a;
          TleWgmmaOperand aOperand;
          TleWgmmaOperand bOperand;
          bool isFirstWgmma = m == 0 && n == 0 && fragment == 0 && k == 0;
          if (aInShared) {
            aOperand = isFirstWgmma ? firstA
                                    : buildSharedA(m * mmaSizeM, k * mmaSizeK);
            a = aOperand.value;
          } else {
            if (isFirstWgmma) {
              aOperand = firstA;
              a = aOperand.value;
            } else {
              auto aDotOpEnc =
                  cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
              assert(aDotOpEnc.getKWidth() ==
                     32 / aTensorTy.getElementTypeBitWidth());

              unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
              llvm::SmallVector<Value> regA = loadReg(
                  rewriter, loc, structA, (m * physicalNumRepK + k) * regASize,
                  regASize, startSequence);
              auto regATy = LLVM::LLVMStructType::getLiteral(
                  rewriter.getContext(),
                  SmallVector<Type>(regA.size(), regA[0].getType()));
              a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
              aOperand = {a, std::nullopt};
            }
          }
          int nOffset = n * mmaSizeN + fragment * wgmmaInstructionN;
          bOperand =
              isFirstWgmma ? firstB : buildSharedB(k * mmaSizeK, nOffset);
          auto b = bOperand.value;
          numLowPrecisionAcc += K;
          // If using native accumulation would cause use to do more low
          // precision accumulation than allowed, use a separate accumulator.
          bool requireAddAccumulator =
              needsPartialAccumulator &&
              (numLowPrecisionAcc >= maxNumImpreciseAcc ||
               k == activeNumRepK - 1);
          Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
          auto mmaOp = triton::nvgpu::WGMMAOp::create(
              rewriter, loc, accTy, a, b, useC, mmaAcc, M, wgmmaInstructionN, K,
              eltTypeC, eltTypeA, eltTypeB, layoutA, layoutB);
          setDescriptorAttrs(mmaOp, aOperand, bOperand);
          mmaAcc = mmaOp;
          useC = tb.i1_val(1);
          if (needsPartialAccumulator)
            partialAcc = mmaAcc;
          else
            d = mmaAcc;
          if (requireAddAccumulator) {
            d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
            numLowPrecisionAcc = 0;
            partialAcc = Value();
          }
        }
        auto acc = unpackLLElements(loc, d, rewriter);
        for (int i = 0; i < acc.size(); ++i)
          mmaResults.push_back(acc[i]);
      }
    }
  }
  if (!op->hasAttr(kTleExplicitWgmmaCommitAttr))
    NVVM::WgmmaGroupSyncAlignedOp::create(rewriter, loc);

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  if (activeNAttr) {
    // The logical-domain contract makes the carrier suffix invalid. Retain
    // the physical result type by inserting only the valid prefixes into an
    // undef carrier. This neither reads C's dead suffix nor adds it to the
    // asynchronous wait group.
    assert(dTensorTy.getElementTypeBitWidth() == 32 &&
           "active_n verifier must restrict the accumulator to 32 bits");
    unsigned physicalResultSize = numRepM * numRepN * physicalAccSize;
    auto structTy = LLVM::LLVMStructType::getLiteral(
        mmaEncoding.getContext(),
        SmallVector<Type>(physicalResultSize, dTensorTy.getElementType()));
    Value res = LLVM::UndefOp::create(rewriter, loc, structTy);
    unsigned activeOffset = 0;
    unsigned physicalOffset = 0;
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        for (unsigned i = 0; i < wgmmaAccSize; ++i)
          res = tb.insert_val(structTy, res, mmaResults[activeOffset++],
                              physicalOffset + i);
        physicalOffset += physicalAccSize;
      }
    }
    assert(activeOffset == mmaResults.size() &&
           "active WGMMA accumulator size must match emitted tiles");
    rewriter.replaceOp(op, res);
    return success();
  }

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

#endif // __TLE__

LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value useCOperand, Value loadedA, Value loadedB,
                         Value loadedC, bool allowTF32,
                         bool needsPartialAccumulator,
                         uint32_t maxNumImpreciseAcc, bool sync, Value thread) {
#ifdef __TLE__
  if (op->hasAttr(kTleWgmmaActiveNAttr) || op->hasAttr(kTleWgmmaActiveKAttr) ||
      op->hasAttr(tle::kWGMMAOperandBPlanAttr))
    return convertLogicalDot(typeConverter, rewriter, loc, op, a, b, c, d,
                             useCOperand, loadedA, loadedB, loadedC, allowTF32,
                             needsPartialAccumulator, maxNumImpreciseAcc, sync,
                             thread);
#endif
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto bTensorTy = cast<triton::gpu::MemDescType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  bool aInShared = isa<SharedEncodingTrait>(aTensorTy.getEncoding());
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(dTensorTy.getEncoding());
  std::optional<SharedMemoryObject> smemObjA;
  Value baseA;
  if (aInShared) {
    baseA = getOffsetedBase(loadedA, cast<MemDescType>(aTensorTy),
                            typeConverter, rewriter, loc);
  }
  auto baseB = getOffsetedBase(loadedB, cast<MemDescType>(bTensorTy),
                               typeConverter, rewriter, loc);
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrMNK = mmaEncoding.getInstrShape();
  auto accSize = 2 * (instrMNK[1] / 4);
  unsigned M = 4 * instrMNK[0];
  unsigned N = instrMNK[1];
  unsigned K = instrMNK[2];
  bool zeroAcc = isZeroConst(c);
#ifdef __TLE__
  bool reuseAccumulatorChainC =
      !zeroAcc && op->hasAttr(kTleWgmmaAccumulatorChainCAttr);
#endif
  auto warpSize = mmaEncoding.getWarpsPerCTA();
  auto shapePerCTATile = SmallVector<unsigned>{instrMNK[0] * warpSize[0],
                                               instrMNK[1] * warpSize[1]};
  unsigned mmaSizeM = shapePerCTATile[0];
  unsigned mmaSizeN = shapePerCTATile[1];
  unsigned mmaSizeK = instrMNK[2];
  int numRepM = ceil<unsigned>(dShapePerCTA[0], mmaSizeM);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], mmaSizeN);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], mmaSizeK);
#ifdef __TLE__
  if (reuseAccumulatorChainC && (numRepM != 1 || numRepN != 1))
    return op->emitOpError(
        "cannot reuse WGMMA accumulator chain C for multi-tile results");
#endif
  DotOpMmaSmemLoader aLoader;
  SmallVector<Value> structA;
  auto warpGroups = {warpSize[0] / 4, warpSize[1]};
  bool transA = false;
  if (aInShared) {
    aLoader =
        DotOpMmaSmemLoader::build(loc, rewriter, cast<MemDescType>(aTensorTy),
                                  baseA, {M, K}, 0, 3, false, dTensorTy);
    transA = aLoader.getDescriptor().transposed;
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaSmemLoader bLoader = DotOpMmaSmemLoader::build(
      loc, rewriter, bTensorTy, baseB, {K, N}, 1, 3, false, dTensorTy);
  bool transB = !bLoader.getDescriptor().transposed;

#ifdef __TLE__
  SmallVector<Value> fc;
  if (!reuseAccumulatorChainC)
    fc = unpackLLElements(loc, loadedC, rewriter);
#else
  auto fc = unpackLLElements(loc, loadedC, rewriter);
#endif

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
#ifdef __TLE__
  // TLE kernels often carry extra live values around WGMMA regions. Keep all
  // ordinary register prep that feeds the first WGMMA outside the fence-to-MMA
  // window so ptxas sees a clean WGMMA pipeline stage.
  // Materialize the initial C accumulator before wgmma.fence.  Otherwise LLVM
  // may leave struct packing/copies between wgmma.fence and the first
  // wgmma.mma_async, which ptxas treats as accumulator definitions inside the
  // WGMMA pipeline stage.
  SmallVector<Type> accTypes;
  SmallVector<Value> initialAccumulators;
  SmallVector<Value> initialUseC;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      Value d;
      Value useC;
      LLVM::LLVMStructType accTy;
      if (reuseAccumulatorChainC) {
        accTy = cast<LLVM::LLVMStructType>(loadedC.getType());
        d = loadedC;
        useC = tb.i1_val(1);
      } else {
        llvm::SmallVector<Value> mmaOut = loadReg(
            rewriter, loc, fc, (m * numRepN + n) * accSize, accSize, op);
        llvm::SmallVector<Type> elemTypes;
        for (Value accEl : mmaOut)
          elemTypes.push_back(accEl.getType());
        accTy =
            LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
        useC = tb.i1_val(0);
        if (!zeroAcc) {
          d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
          useC = tb.i1_val(1);
        }
      }
      if (useCOperand)
        useC = tb.and_(useC, useCOperand);
      accTypes.push_back(accTy);
      initialAccumulators.push_back(d);
      initialUseC.push_back(useC);
    }
  }

  // Keep shared-memory descriptors near their WGMMA use. Treating them as
  // ordinary pure integer SSA lets LLVM hoist them across the full dot region,
  // which can make ptxas spill descriptor registers under high pressure.
  auto buildRegisterA = [&](int m, int k,
                            Operation *insertBefore) -> TleWgmmaOperand {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertBefore);
    auto aDotOpEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    assert(aDotOpEnc.getKWidth() == 32 / aTensorTy.getElementTypeBitWidth());

    unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
    llvm::SmallVector<Value> regA =
        loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize, regASize,
                insertBefore);
    auto regATy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        SmallVector<Type>(regA.size(), regA[0].getType()));
    return {packLLElements(loc, typeConverter, regA, rewriter, regATy),
            std::nullopt};
  };

  auto buildSharedA = [&](int m, int k) -> TleWgmmaOperand {
    LocalizedSMEMDescriptor desc = aLoader.localizedSmemLoad(m, k);
    return {desc.baseb128, desc.descriptorImm};
  };

  auto buildSharedB = [&](int k, int n) -> TleWgmmaOperand {
    LocalizedSMEMDescriptor desc = bLoader.localizedSmemLoad(k, n);
    return {desc.baseb128, desc.descriptorImm};
  };

  auto setDescriptorAttrs = [&](triton::nvgpu::WGMMAOp mmaOp,
                                const TleWgmmaOperand &a,
                                const TleWgmmaOperand &b) {
    if (a.descriptorImm)
      mmaOp->setAttr(nvgpu::kTleWgmmaOperandADescImmAttr,
                     rewriter.getI64IntegerAttr(*a.descriptorImm));
    if (b.descriptorImm)
      mmaOp->setAttr(nvgpu::kTleWgmmaOperandBDescImmAttr,
                     rewriter.getI64IntegerAttr(*b.descriptorImm));
  };

  TleWgmmaOperand firstA;
  TleWgmmaOperand firstB;
  if (numRepM > 0 && numRepN > 0 && numRepK > 0) {
    // The first WGMMA reuses operands materialized before the fence; later
    // descriptors are still localized at their individual WGMMA use sites.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (aInShared) {
      firstA = buildSharedA(/*m=*/0, /*k=*/0);
    } else {
      firstA = buildRegisterA(/*m=*/0, /*k=*/0, op);
    }
    firstB = buildSharedB(/*k=*/0, /*n=*/0);
  }

  Operation *startSequence = op;
  if (!reuseAccumulatorChainC)
    startSequence = NVVM::WgmmaFenceAlignedOp::create(rewriter, loc);
  SmallVector<Value> mmaResults;
  unsigned tileIdx = 0;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      auto accTy = accTypes[tileIdx];
      Value d = initialAccumulators[tileIdx];
      Value useC = initialUseC[tileIdx++];
#else
  Operation *startSequence = NVVM::WgmmaFenceAlignedOp::create(rewriter, loc);
  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadReg(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize,
                  startSequence);
      llvm::SmallVector<Type> elemTypes;
      for (Value accEl : mmaOut)
        elemTypes.push_back(accEl.getType());
      auto accTy =
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
      Value d;
      Value useC = tb.i1_val(0);
      if (!zeroAcc) {
        d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
        useC = tb.i1_val(1);
      }
      if (useCOperand)
        useC = tb.and_(useC, useCOperand);
#endif
      uint32_t numLowPrecisionAcc = 0;
      Value partialAcc;
      for (int k = 0; k < numRepK; ++k) {
        Value a;
#ifdef __TLE__
        TleWgmmaOperand aOperand;
        TleWgmmaOperand bOperand;
        bool isFirstWgmma = m == 0 && n == 0 && k == 0;
#endif
        if (aInShared) {
#ifdef __TLE__
          aOperand =
              isFirstWgmma ? firstA : buildSharedA(m * mmaSizeM, k * mmaSizeK);
          a = aOperand.value;
#else
          a = aLoader.smemLoad(m * mmaSizeM, k * mmaSizeK, rewriter, loc);
#endif
        } else {
#ifdef __TLE__
          if (isFirstWgmma) {
            aOperand = firstA;
            a = aOperand.value;
          } else {
#endif
            auto aDotOpEnc =
                cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
            assert(aDotOpEnc.getKWidth() ==
                   32 / aTensorTy.getElementTypeBitWidth());

            unsigned regASize = (instrMNK[0] * instrMNK[2]) / 32;
            llvm::SmallVector<Value> regA =
                loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize,
                        regASize, startSequence);
            auto regATy = LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(),
                SmallVector<Type>(regA.size(), regA[0].getType()));
            a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
#ifdef __TLE__
            aOperand = {a, std::nullopt};
          }
#endif
        }
#ifdef __TLE__
        bOperand =
            isFirstWgmma ? firstB : buildSharedB(k * mmaSizeK, n * mmaSizeN);
        auto b = bOperand.value;
#else
        auto b = bLoader.smemLoad(k * mmaSizeK, n * mmaSizeN, rewriter, loc);
#endif
        numLowPrecisionAcc += K;
        // If using native accumulation would cause use to do more low precion
        // accumulation than allowed do a separate allocation.
        bool requireAddAccumulator =
            needsPartialAccumulator &&
            (numLowPrecisionAcc >= maxNumImpreciseAcc || k == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
#ifdef __TLE__
        auto mmaOp = triton::nvgpu::WGMMAOp::create(
            rewriter, loc, accTy, a, b, useC, mmaAcc, M, N, K, eltTypeC,
            eltTypeA, eltTypeB, layoutA, layoutB);
        setDescriptorAttrs(mmaOp, aOperand, bOperand);
        mmaAcc = mmaOp;
#else
        mmaAcc = triton::nvgpu::WGMMAOp::create(
            rewriter, loc, accTy, a, b, useC, mmaAcc, M, N, K, eltTypeC,
            eltTypeA, eltTypeB, layoutA, layoutB);
#endif
        useC = tb.i1_val(1);
        if (needsPartialAccumulator)
          partialAcc = mmaAcc;
        else
          d = mmaAcc;
        // If we need accumulate separately to have higher precision, insert
        // adds.
        if (requireAddAccumulator) {
          d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
          numLowPrecisionAcc = 0;
          partialAcc = Value();
        }
      }
      auto acc = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }
#ifdef __TLE__
  if (!op->hasAttr(kTleExplicitWgmmaCommitAttr))
    NVVM::WgmmaGroupSyncAlignedOp::create(rewriter, loc);
#else
  NVVM::WgmmaGroupSyncAlignedOp::create(rewriter, loc);
#endif

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  auto AEnc = op.getA().getType().getEncoding();
  auto BEnc = op.getB().getType().getEncoding();
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(),  //
                    op.getA(), op.getB(), op.getC(), op.getD(), op.getUseC(), //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),           //
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.needsPartialAccumulator(), op.getMaxNumImpreciseAcc(),
                    !op.getIsAsync(), thread);
}
