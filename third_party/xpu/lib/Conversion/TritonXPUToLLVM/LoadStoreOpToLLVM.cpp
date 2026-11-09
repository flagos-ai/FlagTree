#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <cstdlib>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const xpu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {
    isBf16Fast = mlir::triton::tools::getBoolEnvXPU("TRITONXPU_BF16_FAST");
  }

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getContiguity(ptr);
  }

  // Look up the module-level `global_smem` (addrspace 2) and return a ptr<2>
  // base to it. Shared by StageSM and the TLE SM-cache lowerings (alloc/copy/
  // local_ptr with scope=smem). Symbol-table lookup, not a module walk: this is
  // called once per SM op (alloc + copy + every local_ptr/vload).
  Value getGlobalSmemBase(Location loc, ConversionPatternRewriter &rewriter,
                          Operation *op) const {
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
    assert(globalSmem && "global_smem not found; initSharedMemory must run "
                         "before SM lowering");
    Value addr = rewriter.create<LLVM::AddressOfOp>(loc, globalSmem);
    return rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 2), addr);
  }

  // ---- TLE SM (cluster-shared) cache -------------------------------------
  // The defining `ttg.local_alloc` of `buffer` iff that buffer is
  // cluster-shared
  // (`xpu.mem_scope == "smem"`, stamped by the TLE frontend for
  // `tle.gpu.alloc(..., scope=tle.gpu.smem)`), null otherwise.
  //
  // This is the SINGLE "is this SM?" predicate: always the ALLOC's own
  // attribute, reached through the buffer operand. Every consumer
  // (copy_g2l / local_ptr / vload / local_store / vstore) must classify a
  // buffer the same way, otherwise one of them silently falls back to the LM
  // path and dereferences the ptr<0> placeholder from the alloc SM branch. The
  // tle_local_ptr behind `ptr`, looking through the layout-only ops the TLE
  // core-tiling pass wraps around a still-unencoded pointer. Returns null when
  // `ptr` does not come from a local_ptr at all.
  static triton::xpu::TLELocalPtrOp tleLocalPtrThroughLayout(Value ptr) {
    for (int depth = 0; depth < 8 && ptr; ++depth) {
      if (auto lp = ptr.getDefiningOp<triton::xpu::TLELocalPtrOp>())
        return lp;
      auto cvt = ptr.getDefiningOp<triton::xpu::ConvertLayoutOp>();
      if (!cvt)
        return nullptr;
      ptr = cvt.getOperand();
    }
    return nullptr;
  }

  static Operation *getTLESmemAlloc(Value buffer) {
    Operation *def = buffer.getDefiningOp();
    if (!def)
      return nullptr;
    auto scope = def->getAttrOfType<StringAttr>("xpu.mem_scope");
    return (scope && scope.getValue() == "smem") ? def : nullptr;
  }
  static bool isTLESmemBuffer(Value buffer) {
    return getTLESmemAlloc(buffer) != nullptr;
  }

  // ptr<2> base of the cluster-shared buffer allocated by `allocOp`:
  // `global_smem + xpu.sm_offset`, the static byte offset assigned by
  // tritonxpu-tle-sm-alloc. `allocOp` may be `op` itself (the alloc branch).
  //
  // The offset attribute is REQUIRED, not defaulted: falling back to 0 would
  // put every smem buffer at the bottom of the SM window, i.e. aliasing both
  // each other and the reduce/scan scratch -- silent data corruption instead of
  // a crash. A missing attribute means tritonxpu-tle-sm-alloc did not run
  // (check the TLE pipeline order in backend/compiler.py).
  Value getTLESmemBase(Location loc, ConversionPatternRewriter &rewriter,
                       Operation *op, Operation *allocOp) const {
    auto offAttr = allocOp->getAttrOfType<IntegerAttr>("xpu.sm_offset");
    assert(offAttr && "scope=smem local_alloc has no xpu.sm_offset; "
                      "tritonxpu-tle-sm-alloc must run before this lowering");
    Value smBase = getGlobalSmemBase(loc, rewriter, op);
    return gep(ptr_ty(rewriter.getContext(), 2), i8_ty, smBase,
               i32_val(static_cast<int32_t>(offAttr.getInt())));
  }

  // Descriptor strides (element units, i64) as passed from Python, or a
  // row-major-contiguous fallback derived from the buffer shape when the
  // descriptor carried none (tensor_descriptor_base has no `.strides`). Shared
  // by copy_g2l (both the SM and LM branches) and copy_l2g.
  SmallVector<Value>
  getDescStridesOrRowMajor(Location loc, ConversionPatternRewriter &rewriter,
                           ValueRange strides,
                           ArrayRef<int64_t> bufShape) const {
    SmallVector<Value> descStrides(strides.begin(), strides.end());
    if (!descStrides.empty())
      return descStrides;
    unsigned rank = bufShape.size();
    int64_t s0 = 1;
    for (unsigned j = 1; j < rank; ++j)
      s0 *= bufShape[j];
    descStrides.push_back(i64_val(s0));
    if (rank > 1)
      descStrides.push_back(i64_val(1));
    return descStrides;
  }

  // Physical cores per cluster, derived from the module instead of hardcoded:
  // threads-per-warp (== product(coresPerGroup) == groupSize) times num-warps
  // (== product(groupsPerCluster) == numGroups). Their product is the core
  // count in every TLE tiling (RowTiled 1x64, LargeN gxm with g*m == coreNum,
  // and the non-tiled 64x1), see TLECoreTiling::setModuleLayoutAttrs.
  static unsigned getPhysCoresPerCluster(Operation *op) {
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    unsigned numWarps = 1;
    if (auto nwAttr =
            mod->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumWarpsName))
      numWarps = nwAttr.getInt();
    return triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod) * numWarps;
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 512 bits on XPUs.
    return std::min<unsigned>(512 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  void getVectorInfo(Type tensorType, unsigned &vecSize,
                     unsigned &elemNbits) const {
    vecSize = 1u;
    elemNbits = 32u;
    if (auto vecType = mlir::dyn_cast<mlir::VectorType>(tensorType)) {
      unsigned numElems = vecType.getNumElements();
      Type elemTy = vecType.getElementType();
      elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                      ? 64u
                      : elemTy.getIntOrFloatBitWidth();
      // The maximum vector size is 512 bits on XPU2.
      vecSize = std::min<unsigned>(512 / elemNbits, numElems);
    }
  }

  void getLayoutInfo(Type type, size_t &ngroup, size_t &groupsize) const {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (auto globalEncoding = dyn_cast<triton::xpu::ClusterLayoutAttr>(
              tensorType.getEncoding())) {
        auto shape = tensorType.getShape();
        ngroup = product(globalEncoding.getGroupsPerCluster());
        groupsize = product(globalEncoding.getCoresPerGroup());
      }
    } else {
      ngroup = 1;
      groupsize = 1;
    }
  }

  Value bf16ToFp16(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                   Type type, Value elem) const {
    Value convertElem = elem;
    if (auto vecType = mlir::dyn_cast<mlir::VectorType>(type)) {
      unsigned numElems = vecType.getNumElements();
      Type elemTy = vecType.getElementType();
      if (elemTy.isBF16()) {
        Type convertTy = VectorType::get(numElems, f16_ty);
        convertElem = bitcast(elem, convertTy);
      }
    }
    return convertElem;
  }

  void setHaddr(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                Value ptr) const {
    Value ptrInt = ptrtoint(i64_ty, ptr);
    Value ptrIntH32 = lshr(ptrInt, int_val(64, 32));
    Value ptrIntS32 = trunc(i32_ty, ptrIntH32);
    rewriter.create<mlir::LLVM::XPU::SetHaddrOp>(loc, ptrIntS32);
  }

  void createGM2LMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, src);
      Value srcAs0 = addrspace_cast(ptr_ty(ctx, 0), src);
      rewriter.create<mlir::LLVM::XPU::GM2LMOp>(loc, srcAs0, dst, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::GM2LMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create GM2LMOp with unsupported xpu architecture.");
    }
  }

  void createLM2GMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, dst);
      Value dstAs0 = addrspace_cast(ptr_ty(ctx, 0), dst);
      rewriter.create<mlir::LLVM::XPU::LM2GMOp>(loc, src, dstAs0, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::LM2GMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create LM2GMOp with unsupported xpu architecture.");
    }
  }

  void createSM2GMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, dst);
      Value dstAs0 = addrspace_cast(ptr_ty(ctx, 0), dst);
      rewriter.create<mlir::LLVM::XPU::SM2GMOp>(loc, src, dstAs0, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::SM2GMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create LM2GMOp with unsupported xpu architecture.");
    }
  }

  void createGM2SMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    rewriter.create<mlir::LLVM::XPU::GM2SMOp_v3>(loc, src, dst, offset, size);
  }

  // One GM->SM staging DMA, PARTITIONED across all cores of the cluster: core
  // `c` copies elements [c*chunk, (c+1)*chunk) of an `elemCount`-element array
  // (`elemBytes` bytes per element) from `gmBase` (ptr<1>) to `smBase`
  // (ptr<2>). Shared by triton_xpu.stage_sm and the TLE copy_g2l SM-cache
  // branch.
  //
  // Mirrors the hand-written reference (layer_norm_fwd.xpu:463-483:
  // `mfence(); sync_all();` then `partition() + GM2SM_ASYNC + mfence();
  // sync_all();`). Both fences are load-bearing:
  //   * LEADING: drain every core's prior SM reads before overwriting SM.
  //   Needed
  //     whenever the staging can execute more than once -- tritonxpu-loop-grid
  //     hoists it out of the grid-stride loop when its whole operand cone is
  //     pure, but falls back to leaving it INSIDE the loop otherwise, and then
  //     iteration N+1's writes would race iteration N's reads.
  //   * TRAILING: every core drains its OWN async DMA, then the barrier orders
  //     all writes before any read. A core0-only DMA + barrier is NOT enough
  //     (verified: both a core0-only SM-mask fence and a core0-only full-mask
  //     fence still race and HW-fault with "sm rdwr conflict").
  void emitPartitionedGM2SM(Location loc, ConversionPatternRewriter &rewriter,
                            MLIRContext *ctx, Operation *op, Value gmBase,
                            Value smBase, Value elemCount,
                            Value elemBytes) const {
    unsigned physCores = getPhysCoresPerCluster(op);
    assert(physCores > 0 && "cluster core count must be positive");
    Value chunk = sdiv(add(elemCount, i32_val((int32_t)(physCores - 1))),
                       i32_val((int32_t)physCores));
    Value startElem = mul(::mlir::LLVM::XPU::getThreadId(rewriter, loc), chunk);
    Value cntElem = smax(smin(sub(elemCount, startElem), chunk), i32_val(0));
    Value cntBytes = mul(cntElem, elemBytes);
    Value startBytes = mul(startElem, elemBytes);
    Value startBytes64 = sext(i64_ty, startBytes);
    Value srcPtrC = gep(ptr_ty(ctx, 1), i8_ty, gmBase, startBytes64);
    Value smDstC = gep(ptr_ty(ctx, 2), i8_ty, smBase, startBytes);

    createMfenceOp(rewriter, loc, 7);
    xpu_barrier();

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *dmaBlock = rewriter.createBlock(afterBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, icmp_sgt(cntElem, i32_val(0)),
                                    dmaBlock, afterBlock);
    rewriter.setInsertionPointToStart(dmaBlock);
    createGM2SMOp(rewriter, ctx, loc, srcPtrC, smDstC, i32_val(0), cntBytes);
    rewriter.create<LLVM::BrOp>(loc, afterBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    createMfenceOp(rewriter, loc, 7);
    xpu_barrier();
  }

  // One SM->GM writeback, cut into CACHE-LINE-SIZED chunks: worker core `w`
  // copies bytes [w*chunkBytes, +chunkBytes) of the cluster-shared buffer at
  // `smBase` (ptr<2>) out to `gmBase` (ptr<1>). Unlike the LM path there is no
  // per-core ownership to respect -- SM holds ONE copy of the whole tile -- so
  // the cut is chosen here, and choosing it wrong is expensive: partitioning a
  // 64-byte statistics row across 64 cores makes each of them issue a
  // sub-cache-line DMA into the SAME GM line, and those serialise on
  // read-modify-write (measured on XPU3: a [16] f32 row costs +10us that way
  // and a [16] f16 row +22us, against +2us once the elements stop sharing a
  // line and +0.8us once one core sends the row in one transfer). So at most
  // ceil(totalBytes/64) cores take part and a small result vector goes out as
  // ONE contiguous transfer from one core.
  //
  // The LEADING fence + barrier are load-bearing: the cores wrote their lanes
  // of this buffer with ordinary SM stores, and the DMA engine must see all of
  // them.
  void emitCoalescedSM2GM(Location loc, ConversionPatternRewriter &rewriter,
                          MLIRContext *ctx, Operation *op, Value smBase,
                          Value gmBase, unsigned totalBytes,
                          bool isSync) const {
    constexpr unsigned kLineBytes = 64;
    unsigned physCores = getPhysCoresPerCluster(op);
    assert(physCores > 0 && "cluster core count must be positive");
    unsigned numWorkers = std::max(
        1u, std::min(physCores, (totalBytes + kLineBytes - 1) / kLineBytes));
    unsigned chunkBytes =
        ((totalBytes + numWorkers - 1) / numWorkers + kLineBytes - 1) /
        kLineBytes * kLineBytes;

    createMfenceOp(rewriter, loc, 7);
    xpu_barrier();

    Value startBytes = mul(::mlir::LLVM::XPU::getThreadId(rewriter, loc),
                           i32_val((int32_t)chunkBytes));
    Value cntBytes = smax(smin(sub(i32_val((int32_t)totalBytes), startBytes),
                               i32_val((int32_t)chunkBytes)),
                          i32_val(0));
    Value smSrc = gep(ptr_ty(ctx, 2), i8_ty, smBase, startBytes);
    Value startBytes64 = sext(i64_ty, startBytes);
    Value gmDst = gep(ptr_ty(ctx, 1), i8_ty, gmBase, startBytes64);

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *dmaBlock = rewriter.createBlock(afterBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, icmp_sgt(cntBytes, i32_val(0)),
                                    dmaBlock, afterBlock);
    rewriter.setInsertionPointToStart(dmaBlock);
    createSM2GMOp(rewriter, ctx, loc, smSrc, gmDst, i32_val(0), cntBytes);
    rewriter.create<LLVM::BrOp>(loc, afterBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    // Trailing fence only when the copy is synchronous, matching the LM path:
    // an async writeback leaves the DMA in flight and the caller owes a
    // tle_dma_wait before it overwrites the buffer. The barrier goes with it --
    // without the fence there is nothing for it to order.
    if (isSync) {
      createMfenceOp(rewriter, loc, 7);
      xpu_barrier();
    }
  }

  void createMemOp(ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx,
                   mlir::Location &loc, Value bufPtr, Value gmPtr, Value offset,
                   Value size, MemCpyType memCpyType) const {
    switch (static_cast<MemCpyType>(memCpyType)) {
    case MemCpyType::GM2LM:
      createGM2LMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    case MemCpyType::LM2GM:
      createLM2GMOp(rewriter, ctx, loc, gmPtr, bufPtr, offset, size);
      break;
    case MemCpyType::SM2GM:
      createSM2GMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    case MemCpyType::GM2SM:
      // Same operand order as GM2LM: the GM side is the source.
      createGM2SMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    default:
      llvm_unreachable("Memory Op only includes GM2LM, LM2GM, SM2GM, GM2SM");
    }
  }

  void createMfenceOp(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                      int32_t mfenceType = 5) const {
    // Mfence mask bits: bit0=LM(1), bit1=SM(2), bit2=GM(4). 5 fences LM+GM.
    rewriter.create<mlir::LLVM::XPU::MfenceOp>(loc, i32_val(mfenceType));
  }

  void createMfenceLMOp(ConversionPatternRewriter &rewriter,
                        mlir::Location &loc) const {
    createMfenceOp(rewriter, loc, 1);
  }

  void createMfenceGMOp(ConversionPatternRewriter &rewriter,
                        mlir::Location &loc) const {
    createMfenceOp(rewriter, loc, 4);
  }

  Value getStartPtr(ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx,
                    mlir::Location &loc, Value gmPtr, Value zeroPtr,
                    Value rowLen, Value elemBytes) const {
    Value gmPtrInt = ptrtoint(i64_ty, gmPtr);
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value offset = sdiv(sub(gmPtrInt, zeroPtrInt), elemBytes);
    Value rem = srem(offset, rowLen);
    Value adjustedRem =
        select(icmp_slt(rem, i64_val(0)), add(rem, rowLen), rem);
    Value startOffset = sub(offset, adjustedRem);
    Value startOffsetBytes = mul(startOffset, elemBytes);
    Value startPtr = gep(ptr_ty(ctx, 0), i8_ty, zeroPtr,
                         startOffsetBytes); // convert ptr first, then move
    return startPtr;
  }

  Value getReadBytes(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value readLen,
                     Value llMask, Value llLen, Value mask, Value len,
                     Value elemBytes) const {
    Value _readLen =
        readLen.getType().isInteger(64) ? trunc(i32_ty, readLen) : readLen;
    Value _len = len;
    if (llLen) {
      _len = _len.getType().isInteger(64) ? trunc(i32_ty, _len) : _len;
    }
    Value _elemBytes = elemBytes.getType().isInteger(64)
                           ? trunc(i32_ty, elemBytes)
                           : elemBytes;
    _readLen = llLen ? smin(smax(_len, i32_val(0)), _readLen) : _readLen;
    Value readBytes = mul(_readLen, _elemBytes);
    readBytes = llMask ? select(mask, readBytes, i32_val(0)) : readBytes;
    return readBytes;
  }

  /* ******************************** without mask zero
   * *****************************************/
  void lowerLocallyContinuousUnfixedStride(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      int64_t _rowLen, int64_t _bufLen, int64_t _elemBytes, Value llGMPtr,
      Value llLMPtr, Value llLen, Value offsetBytes, MemCpyType memCpyType,
      Block *oldBlock, Block *newBlock) const {
    // clang-format off
    /* *****************************************************************************
    def getStartPtr(gmPtr, zeroPtr, rowLen, elemBytes):
        offset = (gmPtr - zeroPtr) / elemBytes
        startOffsetBytes = (offset / rowLen) * rowLen * elemBytes
        return zeroPtr + startOffsetBytes

    _rowMaxTail = _bufLen % _rowLen
    _rowNum = _bufLen / _rowLen
    rowBytes = rowLen * elemBytes
    tailLen = min(rowLen - (gmPtr.front() - zeroPtr) / elemBytes % rowLen, bufLen)
    if _rowMaxTail == 0:
      for i in range(_rowNum):
        gmStartPtr = llGMPtrs[i * _rowLen]
        lmOffsetBytes = (i * _rowLen) * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes;
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    else:
      if 0 < tailLen < rowMaxTail:
        gm2lm(gmPtr.front(), lmPtr, tailBytes)
        for i in range(_rowNum):
          gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+i*_rowLen], zeroPtr, rowLen, elemBytes)
          lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
          lmStartPtr = lmPtr + lmOffsetBytes
          gm2lm(gmStartPtr, lmStartPtr, rowBytes)
        gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
        offset = tailLen + rowNum * rowLen
        lmOffsetBytes = offset * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes
        remainBytes = (bufLen - offset) * elemBytes
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
      else:
          gm2lm(gmPtr.front(), lmPtr, tailBytes)
          if _rowNum >= 1:
            for i in range(_rowNum-1):
              gmPtr1 = gmPtr[_rowMaxTail+i*_rowLen]
              gmPtr2 = gmPtr[_rowMaxTail+(i+1)*_rowLen]
              gmPtr = select(tailLen == rowMaxTail, gmPtr1, gmPtr2)
              gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+(i+1)*_rowLen], zeroPtr, rowLen, elemBytes)
              lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
              lmStartPtr = lmPtr + lmOffsetBytes
              gm2lm(gmStartPtr, lmStartPtr, rowBytes)
            gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
            offset = tailLen + (rowNum - 1) * rowLen
            lmOffsetBytes = offset * elemBytes
            lmStartPtr = lmPtr + lmOffsetBytes
            remainBytes = (bufLen - offset) * elemBytes
            gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    ********************************************************************************/
    // clang-format on
    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    Value gmFrontPtr = llGMPtrs.front();
    Value gmBackPtr = llGMPtrs.back();
    Value lmPtr = llLMPtrs.front();

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(gmFrontPtr);
    Value zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value gmFrontPtrInt = ptrtoint(i64_ty, gmFrontPtr);

    int64_t _rowMaxTail = _bufLen % _rowLen;
    int64_t _rowNum = _bufLen / _rowLen;
    Value rowMaxTail = i64_val(_rowMaxTail);
    Value rowNum = i64_val(_rowNum);
    Value rowLen = i64_val(_rowLen);
    Value bufLen = i64_val(_bufLen);
    Value elemBytes = i64_val(_elemBytes);
    Value rowBytes = trunc(i32_ty, mul(rowLen, elemBytes));

    if (_rowMaxTail == 0) {
      // GM2LM/LM2GM Row Data
      for (int64_t i = 0; i < _rowNum; ++i) {
        Value gmStartPtr = llGMPtrs[i * _rowLen];
        Value lmOffsetBytes = mul(i64_val(i * _rowLen), elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    rowBytes, memCpyType);
      }
    } else {
      Value gmFrontOffset = sdiv(sub(gmFrontPtrInt, zeroPtrInt), elemBytes);
      Value rawRem = srem(gmFrontOffset, rowLen);
      Value floorMod =
          select(icmp_slt(rawRem, i64_val(0)), add(rawRem, rowLen), rawRem);
      Value tailLen = smin(sub(rowLen, floorMod), bufLen);

      Block *thenBB = rewriter.createBlock(newBlock);
      Block *elseBB = rewriter.createBlock(newBlock);
      Block *mfenceBB = rewriter.createBlock(newBlock);
      rewriter.setInsertionPointToEnd(oldBlock);

      Value condTailSgt = icmp_sgt(tailLen, i64_val(0));
      Value condTailSlt = icmp_slt(tailLen, rowMaxTail);
      Value condTailDiff = and_(condTailSgt, condTailSlt);
      rewriter.create<LLVM::CondBrOp>(loc, condTailDiff, thenBB, elseBB);
      // 1. ThenBB
      rewriter.setInsertionPointToEnd(thenBB);
      {
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailBytes = trunc(i32_ty, mul(tailLen, elemBytes));
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    tailBytes, memCpyType);
        // 1.2 GM2LM/LM2GM Row Data
        for (int64_t i = 0; i < _rowNum; ++i) {
          Value gmPtr = llGMPtrs[_rowMaxTail + i * _rowLen];
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value lmOffsetBytes =
              mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      rowBytes, memCpyType);
        }
        // 1.3 GM2LM/LM2GM Remain Data
        Value gmPtr = llGMPtrs.back();
        Value gmStartPtr =
            getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr, rowLen, elemBytes);
        Value offset = add(tailLen, i64_val(_rowNum * _rowLen));
        Value lmOffsetBytes = mul(offset, elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value remainBytes = trunc(i32_ty, mul(sub(bufLen, offset), elemBytes));
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    remainBytes, memCpyType);
      }
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB

      // 2. elseBB
      rewriter.setInsertionPointToEnd(elseBB);
      {
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailBytes = trunc(i32_ty, mul(tailLen, elemBytes));
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    tailBytes, memCpyType);
        if (_rowNum >= 1) {
          // 1.2 GM2LM/LM2GM Row Data
          Value gmCond = icmp_eq(tailLen, rowMaxTail);
          for (int64_t i = 0; i < _rowNum - 1; ++i) {
            Value gmPtr1 = llGMPtrs[_rowMaxTail + i * _rowLen];
            Value gmPtr2 = llGMPtrs[_rowMaxTail + (i + 1) * _rowLen];
            Value gmPtr = select(gmCond, gmPtr1, gmPtr2);
            Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                           rowLen, elemBytes);
            Value lmOffsetBytes =
                mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
            Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
            createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                        rowBytes, memCpyType);
          }
          // 1.3 GM2LM/LM2GM Remain Data
          Value gmPtr = llGMPtrs.back();
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value offset = add(tailLen, i64_val((_rowNum - 1) * _rowLen));
          Value lmOffsetBytes = mul(offset, elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          Value remainBytes =
              trunc(i32_ty, mul(sub(bufLen, offset), elemBytes));
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      remainBytes, memCpyType);
        }
      }
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB

      // 3. mefenceBB
      rewriter.setInsertionPointToEnd(mfenceBB);
    }
  }

  void lowerLocallyContinuousLargeRow(Operation *op, Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      size_t rowSize, size_t rowStride,
                                      Value llGMPtr, Value llLMPtr, Value llLen,
                                      Value bufLen, Value elemBytes,
                                      Value offsetBytes, MemCpyType memCpyType,
                                      Block *oldBlock, Block *newBlock) const {

    /* *************************************************
    gapLen = strideLen - rowLen
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen) * rowLen
    realTailLen = rowLen - (bankOffset - (blockOffset + rowOffset))

    if 0 < realTailLen < bufLen:
      gm2lm(bankPtr, lmPtr, realTailLen * elemBytes)
      gm2lm(bankPtr + (realTailLen + gapLen) * elemBytes, lmPtr + realTailLen,
    elemBytes,（bufLen - realTailLen）* elemBytes)

    else :
      gm2lm(bankPtr, lmPtr, bufLen * elemBytes)
    * ************************************************/

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    size_t gapSize = rowStride - rowSize;
    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value gapLen = i32_val(gapSize);
    Value gapBytes = mul(gapLen, elemBytes);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value realTailLen =
        sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(realTailLen, elemBytes);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *thenBB = rewriter.createBlock(newBlock);
    Block *elseBB = rewriter.createBlock(newBlock);
    Block *mfenceBB = rewriter.createBlock(newBlock);
    rewriter.setInsertionPointToEnd(oldBlock);

    Value condRemSgt = icmp_sgt(realTailLen, i32_val(0));
    Value condRemSlt = icmp_slt(realTailLen, bufLen);
    Value condRemDiff = and_(condRemSgt, condRemSlt);
    rewriter.create<LLVM::CondBrOp>(loc, condRemDiff, thenBB, elseBB);
    rewriter.setInsertionPointToEnd(thenBB);
    // 1. ThenBB
    // 1.1 GM2LM Tail Data
    Value tailLen = realTailLen;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        Value limitedLen =
            smin(smax(llLens[0], i64_val(0)), sext(i64_ty, bufLen));
        tailLen = smin(realTailLen, trunc(i32_ty, limitedLen));
      } else if (llLens[0].getType().isInteger(1)) {
        Value limitedLen = bufLen;
        tailLen = smin(realTailLen, limitedLen);
      } else {
        Value limitedLen = smin(smax(llLens[0], i32_val(0)), bufLen);
        tailLen = smin(realTailLen, limitedLen);
      }
    }
    Value tailBytes = mul(tailLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, tailBytes,
                memCpyType);

    // 1.2 GM2LM Remain Data
    Value startCond;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        startCond = icmp_sge(sext(i64_ty, realTailLen), llLens[0]);
      } else if (llLens[0].getType().isInteger(1)) {
        startCond = icmp_sge(realTailLen, bufLen);
      } else {
        startCond = icmp_sge(realTailLen, llLens[0]);
      }
    }
    Value startPtrInt =
        add(bankPtrInt, zext(i64_ty, add(realTailBytes, gapBytes)));
    Value startPtr =
        rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
    startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
    Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                            realTailBytes); // convert ptr first, then move

    Value remainLen = sub(bufLen, realTailLen);
    Value remainBytes = mul(remainLen, elemBytes);
    if (startCond && memCpyType == MemCpyType::LM2GM) {
      Block *remainBB = rewriter.createBlock(mfenceBB);
      rewriter.setInsertionPointToEnd(thenBB);
      rewriter.create<LLVM::CondBrOp>(loc, startCond, mfenceBB, remainBB);
      rewriter.setInsertionPointToEnd(remainBB);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  remainBytes, memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, mfenceBB);
    } else {
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  remainBytes, memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB
    }

    // 2. elseBB
    rewriter.setInsertionPointToEnd(elseBB);
    // GM2LM the whole bufLen
    Value readBytes = mul(bufLen, elemBytes);
    if (llLen && memCpyType == MemCpyType::LM2GM) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      Value skipCond = llLens[0].getType().isInteger(64)
                           ? icmp_sle(llLens[0], i64_val(0))
                           : icmp_sle(llLens[0], i32_val(0));
      Block *elseDmaBB = rewriter.createBlock(mfenceBB);
      rewriter.setInsertionPointToEnd(elseBB);
      rewriter.create<LLVM::CondBrOp>(loc, skipCond, mfenceBB, elseDmaBB);
      rewriter.setInsertionPointToEnd(elseDmaBB);
      Value limitedLen;
      if (llLens[0].getType().isInteger(64))
        limitedLen = trunc(i32_ty, smin(llLens[0], sext(i64_ty, bufLen)));
      else
        limitedLen = smin(llLens[0], bufLen);
      readBytes = mul(limitedLen, elemBytes);
    }
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                mfenceBB); // Jump to mfenceBB

    // 3. mefenceBB
    rewriter.setInsertionPointToEnd(mfenceBB);
  }

  void lowerLocallyContinuousSmallRow(Operation *op, Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      size_t rowSize, size_t rowStride,
                                      Value llGMPtr, Value llLMPtr, Value llLen,
                                      Value bufLen, Value elemBytes,
                                      Value offsetBytes, MemCpyType memCpyType,
                                      Block *oldBlock, Block *newBlock) const {

    /* *************************************************
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen)
    rowHeadLen = bankOffset - (blockOffset + rowOffset)
    realTailLen = rowLen - rowHeadLen
    rowNum = (bufLen - realTailLen - 1) / rowLen

    gm2lm(bankPtr, lmPtr, realTailLen * elemBytes)

    for(i = 0; i < rowNum; i++) {
        gm2lm(bankPtr + ((i + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (realTailLen + i * rowLen) * elemBytes, rowLen * elemBytes)
    }

    remLen = bufLen - realTailLen - rowNum * rowLen
    gm2lm(bankPtr + ((rowNum + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (realTailLen + rowNum * rowLen) * elemBytes, (remLen * elemBytes)
    *************************************************/

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }
    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value realTailLen =
        sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(realTailLen, elemBytes);
    Value rowBytes = mul(rowLen, elemBytes);
    Value rowHeadLen = sub(rowLen, realTailLen);
    Value rowHeadBytes = sub(rowBytes, realTailBytes);
    Value realRemainLen = sub(sub(bufLen, realTailLen), i32_val(1));
    Value rowNum = sdiv(realRemainLen, rowLen);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *judgeBB = rewriter.createBlock(newBlock, TypeRange{i32_ty}, {loc});
    Block *gm2lmRowBB = rewriter.createBlock(newBlock);
    Block *stepBB = rewriter.createBlock(newBlock);
    Block *gm2lmRemBB = rewriter.createBlock(newBlock);

    // 1.  GM2LM Tail Data
    rewriter.setInsertionPointToEnd(oldBlock);
    Value tailLen = realTailLen;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        tailLen = smin(realTailLen, trunc(i32_ty, llLens[0]));
      } else {
        tailLen = smin(realTailLen, llLens[0]);
      }
    }
    Value tailBytes = mul(tailLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, tailBytes,
                memCpyType);

    Value _init = i32_val(0);
    Value _step = i32_val(1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{_init},
                                judgeBB); // Jump to judgeBB
    Value iter = judgeBB->getArgument(0);

    // 2. GM2LM Row Data
    rewriter.setInsertionPointToEnd(judgeBB);
    Value condSlt = icmp_slt(iter, rowNum);
    rewriter.create<LLVM::CondBrOp>(loc, condSlt, gm2lmRowBB, gm2lmRemBB);

    rewriter.setInsertionPointToEnd(gm2lmRowBB);
    Value skipStride = mul(add(iter, i32_val(1)), strideLen);
    Value skipStrideBytes = mul(skipStride, elemBytes);
    Value skipRowLen = mul(iter, rowLen);
    Value startPtrInt =
        add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
    Value startPtr =
        rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
    startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
    Value startCond;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        startCond =
            icmp_sge(sext(i64_ty, add(skipRowLen, realTailLen)), llLens[0]);
      } else {
        startCond = icmp_sge(add(skipRowLen, realTailLen), llLens[0]);
      }
    }
    startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
    Value actualRowBytes = rowBytes;
    if (startCond && memCpyType == MemCpyType::LM2GM)
      actualRowBytes = select(startCond, i32_val(0), rowBytes);
    Value dstOffset = add(realTailLen, skipRowLen);
    Value dstOffsetBytes = mul(dstOffset, elemBytes);
    Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                            dstOffsetBytes); // convert ptr first, then move
    createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                actualRowBytes, memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, stepBB); // Jump to stepBB

    rewriter.setInsertionPointToEnd(stepBB);
    Value _index = add(iter, _step);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{_index},
                                judgeBB); // Jump back to judgeBB

    // 3 GM2LM Remain Data
    rewriter.setInsertionPointToEnd(gm2lmRemBB);
    {
      Value skipStride = mul(add(rowNum, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(rowNum, rowLen);
      Value remainBytes =
          mul(sub(bufLen, add(realTailLen, skipRowLen)), elemBytes);
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value startCond;
      if (llLen) {
        auto llLens = unpackLLElements(loc, llLen, rewriter);
        if (llLens[0].getType().isInteger(64)) {
          startCond =
              icmp_sge(sext(i64_ty, add(skipRowLen, realTailLen)), llLens[0]);
        } else {
          startCond = icmp_sge(add(skipRowLen, realTailLen), llLens[0]);
        }
      }
      startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
      Value actualRemainBytes = remainBytes;
      if (startCond && memCpyType == MemCpyType::LM2GM)
        actualRemainBytes = select(startCond, i32_val(0), remainBytes);
      Value dstOffset = add(realTailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  actualRemainBytes, memCpyType);
    }
  }

  /* ******************************** with mask zero
   * *****************************************/
  void lowerLocallyContinuousUnfixedStrideMask(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      int64_t _rowLen, int64_t _bufLen, int64_t _elemBytes, Value llGMPtr,
      Value llLMPtr, Value llMask, Value llLen, Value offsetBytes,
      MemCpyType memCpyType, Block *oldBlock, Block *newBlock) const {
    // clang-format off
    /* *****************************************************************************
    def getStartPtr(gmPtr, zeroPtr, rowLen, elemBytes):
        offset = (gmPtr - zeroPtr) / elemBytes
        startOffsetBytes = (offset / rowLen) * rowLen * elemBytes
        return zeroPtr + startOffsetBytes

    _rowMaxTail = _bufLen % _rowLen
    _rowNum = _bufLen / _rowLen
    rowBytes = rowLen * elemBytes
    tailLen = min(rowLen - (gmPtr.front() - zeroPtr) / elemBytes % rowLen, bufLen)
    if _rowMaxTail == 0:
      for i in range(_rowNum):
        gmStartPtr = llGMPtrs[i * _rowLen]
        lmOffsetBytes = (i * _rowLen) * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes;
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    else:
      if 0 < tailLen < rowMaxTail:
        gm2lm(gmPtr.front(), lmPtr, tailBytes)
        for i in range(_rowNum):
          gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+i*_rowLen], zeroPtr, rowLen, elemBytes)
          lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
          lmStartPtr = lmPtr + lmOffsetBytes
          gm2lm(gmStartPtr, lmStartPtr, rowBytes)
        gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
        offset = tailLen + rowNum * rowLen
        lmOffsetBytes = offset * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes
        remainBytes = (bufLen - offset) * elemBytes
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
      else:
          gm2lm(gmPtr.front(), lmPtr, tailBytes)
          if _rowNum >= 1:
            for i in range(_rowNum-1):
              gmPtr1 = gmPtr[_rowMaxTail+i*_rowLen]
              gmPtr2 = gmPtr[_rowMaxTail+(i+1)*_rowLen]
              gmPtr = select(tailLen == rowMaxTail, gmPtr1, gmPtr2)
              gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+(i+1)*_rowLen], zeroPtr, rowLen, elemBytes)
              lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
              lmStartPtr = lmPtr + lmOffsetBytes
              gm2lm(gmStartPtr, lmStartPtr, rowBytes)
            gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
            offset = tailLen + (rowNum - 1) * rowLen
            lmOffsetBytes = offset * elemBytes
            lmStartPtr = lmPtr + lmOffsetBytes
            remainBytes = (bufLen - offset) * elemBytes
            gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    ********************************************************************************/
    // clang-format on
    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    SmallVector<Value> llMasks;
    if (llMask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }
    SmallVector<Value> llLens;
    if (llLen) {
      llLens = unpackLLElements(loc, llLen, rewriter);
    }

    Value gmFrontPtr = llGMPtrs.front();
    Value gmBackPtr = llGMPtrs.back();
    Value lmPtr = llLMPtrs.front();

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(gmFrontPtr);
    Value zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value gmFrontPtrInt = ptrtoint(i64_ty, gmFrontPtr);

    int64_t _rowNum = _bufLen / _rowLen;
    int64_t _rowMaxTail = _rowNum > 0 ? _bufLen % _rowLen : _bufLen - 1;
    Value rowMaxTail = i64_val(_rowMaxTail);
    Value rowNum = i64_val(_rowNum);
    Value rowLen = i64_val(_rowLen);
    Value bufLen = i64_val(_bufLen);
    Value elemBytes = i64_val(_elemBytes);

    if (_rowMaxTail == 0) {
      // GM2LM/LM2GM Row Data
      for (int64_t i = 0; i < _rowNum; ++i) {
        Value gmStartPtr = llGMPtrs[i * _rowLen];
        Value lmOffsetBytes = mul(i64_val(i * _rowLen), elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value mask = llMask ? llMasks[i * _rowLen] : Value();
        Value len = llLen ? llLens[i * _rowLen] : Value();
        Value readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask,
                                       llLen, mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    readBytes, memCpyType);
      }
    } else {
      Value gmFrontOffset = sdiv(sub(gmFrontPtrInt, zeroPtrInt), elemBytes);
      Value rawRem = srem(gmFrontOffset, rowLen);
      Value floorMod =
          select(icmp_slt(rawRem, i64_val(0)), add(rawRem, rowLen), rawRem);
      Value tailLen = smin(sub(rowLen, floorMod), bufLen);

      Block *thenBB = rewriter.createBlock(newBlock);
      Block *elseBB = rewriter.createBlock(newBlock);
      Block *mfenceBB = rewriter.createBlock(newBlock);
      rewriter.setInsertionPointToEnd(oldBlock);

      Value condTailSgt = icmp_sgt(tailLen, i64_val(0));
      Value condTailSlt = icmp_slt(tailLen, rowMaxTail);
      Value condTailDiff = and_(condTailSgt, condTailSlt);
      rewriter.create<LLVM::CondBrOp>(loc, condTailDiff, thenBB, elseBB);
      // 1. ThenBB
      {
        rewriter.setInsertionPointToEnd(thenBB);
        // 1.1 GM2LM/LM2GM Tail Data
        Value mask = llMask ? llMasks[0] : Value();
        Value len = llLen ? llLens[0] : Value();
        Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask,
                                       llLen, mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    readBytes, memCpyType);
        // 1.2 GM2LM/LM2GM Row Data
        for (int64_t i = 0; i < _rowNum; ++i) {
          Value gmPtr = llGMPtrs[_rowMaxTail + i * _rowLen];
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value lmOffsetBytes =
              mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          mask = llMask ? llMasks[_rowMaxTail + i * _rowLen] : Value();
          len = llLen ? llLens[_rowMaxTail + i * _rowLen] : Value();
          readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask, llLen,
                                   mask, len, elemBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      readBytes, memCpyType);
        }
        // 1.3 GM2LM/LM2GM Remain Data
        Value gmPtr = llGMPtrs.back();
        Value gmStartPtr =
            getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr, rowLen, elemBytes);
        Value offset = add(tailLen, i64_val(_rowNum * _rowLen));
        Value lmOffsetBytes = mul(offset, elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value remainLen = sub(bufLen, offset);
        mask = llMask ? llMasks[_rowMaxTail + _rowNum * _rowLen] : Value();
        len = llLen ? llLens[_rowMaxTail + _rowNum * _rowLen] : Value();
        readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask, llLen,
                                 mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    readBytes, memCpyType);

        rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                    mfenceBB); // Jump to mfenceBB
      }

      // 2. elseBB
      {
        rewriter.setInsertionPointToEnd(elseBB);
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailLen = mul(tailLen, elemBytes);
        Value mask = llMask ? llMasks[0] : Value();
        Value len = llLen ? llLens[0] : Value();
        Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask,
                                       llLen, mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    readBytes, memCpyType);
        if (_rowNum >= 1) {
          // 1.2 GM2LM/LM2GM Row Data
          Value gmCond = icmp_eq(tailLen, rowMaxTail);
          for (int64_t i = 0; i < _rowNum - 1; ++i) {
            Value gmPtr1 = llGMPtrs[_rowMaxTail + i * _rowLen];
            Value gmPtr2 = llGMPtrs[_rowMaxTail + (i + 1) * _rowLen];
            Value gmPtr = select(gmCond, gmPtr1, gmPtr2);
            Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                           rowLen, elemBytes);
            Value lmOffsetBytes =
                mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
            Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);

            Value mask1 = llMask ? llMasks[_rowMaxTail + i * _rowLen] : Value();
            Value mask2 =
                llMask ? llMasks[_rowMaxTail + (i + 1) * _rowLen] : Value();
            mask = llMask ? select(gmCond, mask1, mask2) : Value();

            Value len1 = llLen ? llLens[_rowMaxTail + i * _rowLen] : Value();
            Value len2 =
                llLen ? llLens[_rowMaxTail + (i + 1) * _rowLen] : Value();
            len = llLen ? select(gmCond, len1, len2) : Value();
            readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask, llLen,
                                     mask, len, elemBytes);
            createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                        readBytes, memCpyType);
          }
          // 1.3 GM2LM/LM2GM Remain Data
          Value gmPtr = llGMPtrs.back();
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value offset = add(tailLen, i64_val((_rowNum - 1) * _rowLen));
          Value lmOffsetBytes = mul(offset, elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          Value remainLen = sub(bufLen, offset);
          mask = llMask ? llMasks[(_rowNum - 1) * _rowLen] : Value();
          len = llLen ? llLens[(_rowNum - 1) * _rowLen] : Value();
          readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask, llLen,
                                   mask, len, elemBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      readBytes, memCpyType);
        }

        rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                    mfenceBB); // Jump to mfenceBB
      }

      // 3. mefenceBB
      rewriter.setInsertionPointToEnd(mfenceBB);
    }
  }

  void lowerLocallyContinuousUnfixedStrideMask(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      size_t rowSize, size_t rowStride, Value llGMPtr, Value llLMPtr,
      Value llMask, Value llLen, Value bufLen, Value elemBytes,
      Value offsetBytes, MemCpyType memCpyType, Block *oldBlock,
      Block *newBlock) const {
    // clang-format off
    /* *************************************************
    gapLen = strideLen - rowLen
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen) * rowLen
    tailLen = rowLen - (bankOffset - (blockOffset + rowOffset))

    if 0 < tailLen < bufLen:
      gm2lm(bankPtr, lmPtr, tailLen * elemBytes)
      gm2lm(bankPtr + (tailLen + gapLen) * elemBytes, lmPtr + tailLen,
    elemBytes,（bufLen - tailLen）* elemBytes)

    else :
      gm2lm(bankPtr, lmPtr, bufLen * elemBytes)
    * ************************************************/
    // clang-format on

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    SmallVector<Value> llMasks;
    if (llMask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }
    SmallVector<Value> llLens;
    if (llLen) {
      llLens = unpackLLElements(loc, llLen, rewriter);
    }

    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    size_t gapSize = rowStride - rowSize;
    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value gapLen = i32_val(gapSize);
    Value gapBytes = mul(gapLen, elemBytes);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value tailLen = sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value tailBytes = mul(tailLen, elemBytes);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *thenBB = rewriter.createBlock(newBlock);
    Block *elseBB = rewriter.createBlock(newBlock);
    Block *mfenceBB = rewriter.createBlock(newBlock);
    rewriter.setInsertionPointToEnd(oldBlock);

    Value condRemSgt = icmp_sgt(tailLen, i32_val(0));
    Value condRemSlt = icmp_slt(tailLen, bufLen);
    Value condRemDiff = and_(condRemSgt, condRemSlt);
    rewriter.create<LLVM::CondBrOp>(loc, condRemDiff, thenBB, elseBB);
    rewriter.setInsertionPointToEnd(thenBB);
    // 1. ThenBB
    {
      // 1.1 GM2LM Tail Data
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                  memCpyType);

      // 1.2 GM2LM Remain Data
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, add(tailBytes, gapBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              tailBytes); // convert ptr first, then move

      Value remainLen = sub(bufLen, tailLen);
      Value remainBytes = mul(remainLen, elemBytes);
      mask = llMask ? llMasks.back() : Value();
      len = llLen ? llLens.back() : Value();
      readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask, llLen,
                               mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  readBytes, memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB
    }

    // 2. elseBB
    {
      rewriter.setInsertionPointToEnd(elseBB);
      // GM2LM the whole bufLen
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, bufLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                  memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB
    }

    // 3. mefenceBB
    rewriter.setInsertionPointToEnd(mfenceBB);
  }

  void lowerLocallyContinuousSmallRowMask(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      size_t rowSize, size_t rowStride, Value llGMPtr, Value llLMPtr,
      Value llMask, Value llLen, Value bufLen, Value elemBytes,
      Value offsetBytes, MemCpyType memCpyType, Block *oldBlock,
      Block *newBlock) const {
    // clang-format off
    /* *************************************************
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen)
    rowHeadLen = bankOffset - (blockOffset + rowOffset)
    tailLen = rowLen - rowHeadLen
    rowNum = (bufLen - tailLen - 1) / rowLen

    gm2lm(bankPtr, lmPtr, tailLen * elemBytes)

    for(i = 0; i < rowNum; i++) {
        gm2lm(bankPtr + ((i + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (tailLen + i * rowLen) * elemBytes, rowLen * elemBytes)
    }

    remLen = bufLen - tailLen - rowNum * rowLen
    gm2lm(bankPtr + ((rowNum + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (tailLen + rowNum * rowLen) * elemBytes, (remLen * elemBytes)
    *************************************************/
    // clang-format on

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    SmallVector<Value> llMasks;
    if (llMask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }
    SmallVector<Value> llLens;
    if (llLen) {
      llLens = unpackLLElements(loc, llLen, rewriter);
    }

    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }
    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value tailLen = sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(tailLen, elemBytes);
    Value rowBytes = mul(rowLen, elemBytes);
    Value rowHeadLen = sub(rowLen, tailLen);
    Value rowHeadBytes = sub(rowBytes, realTailBytes);
    Value realRemainLen = sub(sub(bufLen, tailLen), i32_val(1));
    Value rowNum = sdiv(realRemainLen, rowLen);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *judgeBB = rewriter.createBlock(newBlock, TypeRange{i32_ty}, {loc});
    Block *gm2lmRowBB = rewriter.createBlock(newBlock);
    Block *stepBB = rewriter.createBlock(newBlock);
    Block *gm2lmRemBB = rewriter.createBlock(newBlock);

    // 1.  GM2LM Tail Data
    {
      rewriter.setInsertionPointToEnd(oldBlock);
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                  memCpyType);
    }

    // 2. GM2LM Row Data
    {
      Value _init = i32_val(0);
      Value _step = i32_val(1);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{_init},
                                  judgeBB); // Jump to judgeBB
      Value iter = judgeBB->getArgument(0);

      rewriter.setInsertionPointToEnd(judgeBB);
      Value condSlt = icmp_slt(iter, rowNum);
      rewriter.create<LLVM::CondBrOp>(loc, condSlt, gm2lmRowBB, gm2lmRemBB);

      rewriter.setInsertionPointToEnd(gm2lmRowBB);
      Value skipStride = mul(add(iter, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(iter, rowLen);
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value dstOffset = add(tailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  readBytes, memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, stepBB); // Jump to stepBB

      rewriter.setInsertionPointToEnd(stepBB);
      Value _index = add(iter, _step);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{_index},
                                  judgeBB); // Jump back to judgeBB
    }

    // 3 GM2LM Remain Data
    {
      rewriter.setInsertionPointToEnd(gm2lmRemBB);

      Value skipStride = mul(add(rowNum, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(rowNum, rowLen);
      Value remainLen = sub(bufLen, add(tailLen, skipRowLen));
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value dstOffset = add(tailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      Value mask = llMask ? llMasks.back() : Value();
      Value len = llLen ? llLens.back() : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask,
                                     llLen, mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  readBytes, memCpyType);
    }
  }

  // ---- bf16 -> f32, fused into an LM load ---------------------------------
  // The mirror of the f32->bf16 store below: bf16 sits in the high half of the
  // f32 pattern, so the conversion is pure bit placement (vmerge_l/h_hf against
  // a zero pad), or, when element order does not matter, two masked loads plus
  // a shuffle that skip the merges entirely. Shared by the GM-staging load and
  // the TLE vector load.
  void VecBF16ToFP32Unordered(mlir::MLIRContext *ctx, Location &loc,
                              ConversionPatternRewriter &rewriter,
                              Type &resElemTy, int numElems, int resVecSize,
                              int ptrDataVecSize, Value &lmBasePtr,
                              SmallVector<Value> &loadedVals) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType veci16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType veci32Ty = VectorType::get(resVecSize, i32_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(resVecSize, bf16_ty);
    VectorType VecFp16Ty = VectorType::get(ptrDataVecSize, f16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
    int mask = 0xaaaaaaaa;
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    Value maskNegVal = i32_val(~mask);
    maskNegVal = bitcast(maskNegVal, vec1Ty);
    for (int i = 0; i < numElems / 2; ++i) {
      Value elemPtr = gep(ptr_ty(ctx, 0), vecBf16Ty, lmBasePtr, i32_val(i));
      Value veven = rewriter.create<mlir::LLVM::XPU::VLOAD_MZOp>(
          loc, veci16Ty, elemPtr, maskVal);
      veven = bitcast(veven, resElemTy);
      loadedVals.emplace_back(veven);
      Value vodd = rewriter.create<mlir::LLVM::XPU::VLOAD_MZOp>(
          loc, veci16Ty, elemPtr, maskNegVal);
      vodd = bitcast(vodd, VecFp16Ty);
      Value voddSl =
          rewriter.create<mlir::LLVM::XPU::VSHUFFLE2Op>(loc, VecFp16Ty, vodd);
      voddSl = bitcast(voddSl, resElemTy);
      loadedVals.emplace_back(voddSl);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value loaded = load(halfVecBf16Ty, elemPtr);
      loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemTy, loaded);
      loadedVals.emplace_back(loaded);
    }
    return;
  }

  void VecBF16ToFP32(mlir::MLIRContext *ctx, Location &loc,
                     ConversionPatternRewriter &rewriter, Type &resElemTy,
                     int numElems, int resVecSize, int ptrDataVecSize,
                     SmallVector<Value> &loadedVals) const {
    VectorType vecFp16Ty = VectorType::get(ptrDataVecSize, f16_ty);
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, vecFp16Ty);
    int16_t pad = 0;
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      padVec =
          insert_element(vecFp16Ty, padVec, f16_val(pad), i16_val(elemIdx));
    }
    SmallVector<Value> newLoadedVals;
    for (int i = 0; i < numElems / 2; ++i) {
      Value val = bitcast(loadedVals[i], vecFp16Ty);
      Value vl = rewriter.create<mlir::LLVM::XPU::VMERGE_L_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vl = bitcast(vl, resElemTy);
      newLoadedVals.emplace_back(vl);
      Value vh = rewriter.create<mlir::LLVM::XPU::VMERGE_H_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vh = bitcast(vh, resElemTy);
      newLoadedVals.emplace_back(vh);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value ext = rewriter.create<LLVM::FPExtOp>(loc, resElemTy,
                                                 loadedVals[remainedIdx]);
      newLoadedVals.emplace_back(ext);
    }
    loadedVals = newLoadedVals;
    return;
  }

  // ---- f32 -> bf16, fused into an LM store --------------------------------
  // XPU3 has no f32->bf16 convert instruction (only vfloat2half_l/h for fp16)
  // and no 16-bit pack, so the conversion is bit arithmetic on the f32 pattern
  // (round, then keep the high 16 bits) and the 16 halves of each source
  // register are PLACED BY THE STORE -- which is why this lives here and not in
  // VTruncF. Shared by the GM-staging store and the TLE vector store:
  // `valueElems` are <valueVecSize x f32> registers over an LM buffer at
  // `lmBasePtr`, register i covering elements [i*valueVecSize,
  // (i+1)*valueVecSize).

  // Round an f32 register for bf16 truncation (round-to-nearest-even):
  // x += 0x7fff + (x & 1).
  Value roundF32ForBf16(mlir::MLIRContext *ctx, Location &loc,
                        ConversionPatternRewriter &rewriter, Value vI32,
                        VectorType veci32Ty) const {
    uint32_t one = 0x0001;
    uint32_t magic = 0x7fff;
    SmallVector<Value, 4> andOperands({i32_val(one), vI32});
    auto vAnd = rewriter.create<LLVM::InlineAsmOp>(
        loc, veci32Ty, andOperands, "vand.u.mz $0{mr1}, $1, $2", "=&v,r,v",
        /*has_side_effects=*/false,
        /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT), ArrayAttr());
    SmallVector<Value, 4> addOperands({i32_val(magic), vAnd.getRes()});
    auto vSvAdd = rewriter.create<LLVM::InlineAsmOp>(
        loc, veci32Ty, addOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
        /*has_side_effects=*/false,
        /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
        LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT), ArrayAttr());
    return add(vI32, vSvAdd.getRes());
  }

  // Ordered f32 -> bf16 store: each source register's 16 rounded high halves
  // are scattered to 16 consecutive bf16 slots (byte offsets 0, 2, ... 30 under
  // the odd-lane mask), so element order is preserved.
  void VecFP32ToBF16Slow(mlir::MLIRContext *ctx, Location &loc,
                         ConversionPatternRewriter &rewriter, int numElems,
                         int valueVecSize, int ptrDataVecSize,
                         SmallVector<Value> &valueElems,
                         Value &lmBasePtr) const {
    VectorType vecI16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    constexpr int mask = 0xaaaaaaaa; // 0b10101010101010101010101010101010
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    SmallVector<int16_t> offset_v = {0,  0,  0,  2,  0,  4,  0,  6,  0,  8, 0,
                                     10, 0,  12, 0,  14, 0,  16, 0,  18, 0, 20,
                                     0,  22, 0,  24, 0,  26, 0,  28, 0,  30};
    Value offsetVec = rewriter.create<LLVM::UndefOp>(loc, vecI16Ty);
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      offsetVec = insert_element(vecI16Ty, offsetVec,
                                 i16_val(offset_v[elemIdx]), i16_val(elemIdx));
    }
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // halfVecBf16Ty
    for (int i = 0; i < numElems / 2; ++i) {
      Value dstPtr1 =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i16_val(2 * i));
      Value vl = bitcast(valueElems[2 * i], veci32Ty);
      vl = roundF32ForBf16(ctx, loc, rewriter, vl, veci32Ty);
      vl = bitcast(vl, vecI16Ty);
      rewriter.create<mlir::LLVM::XPU::SCATTER_MHOp>(loc, vl, maskVal, dstPtr1,
                                                     offsetVec);
      Value vh = bitcast(valueElems[2 * i + 1], veci32Ty);
      vh = roundF32ForBf16(ctx, loc, rewriter, vh, veci32Ty);
      vh = bitcast(vh, vecI16Ty);
      Value dstPtr2 =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i16_val(2 * i + 1));
      rewriter.create<mlir::LLVM::XPU::SCATTER_MHOp>(loc, vh, maskVal, dstPtr2,
                                                     offsetVec);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  void VecFP32ToBF16Unordered(mlir::MLIRContext *ctx, Location &loc,
                              ConversionPatternRewriter &rewriter, int numElems,
                              int valueVecSize, int ptrDataVecSize,
                              SmallVector<Value> &valueElems,
                              Value &lmBasePtr) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType veci16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // vecBf16Ty
    int mask = 0xaaaaaaaa;
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    Value maskNegVal = i32_val(~mask);
    maskNegVal = bitcast(maskNegVal, vec1Ty);
    Value poseVal = i32_val(16);
    for (int i = 0; i < numElems / 2; ++i) {
      Value veven = bitcast(valueElems[2 * i], veci32Ty);
      veven = roundF32ForBf16(ctx, loc, rewriter, veven, veci32Ty);
      veven = bitcast(veven, veci16Ty);
      Value elemPtr = gep(ptr_ty(ctx, 0), vecBf16Ty, lmBasePtr, i32_val(i));
      rewriter.create<mlir::LLVM::XPU::VSTORE_MHOp>(loc, veven, elemPtr,
                                                    maskVal);
      Value vodd = bitcast(valueElems[2 * i + 1], veci32Ty);
      vodd = roundF32ForBf16(ctx, loc, rewriter, vodd, veci32Ty);
      Value voddSr = rewriter.create<mlir::LLVM::XPU::SVSRLPOp>(loc, veci32Ty,
                                                                poseVal, vodd);
      voddSr = bitcast(voddSr, veci16Ty);
      rewriter.create<mlir::LLVM::XPU::VSTORE_MHOp>(loc, voddSr, elemPtr,
                                                    maskNegVal);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  // TRITONXPU_BF16_FAST: let the device library convert+store a register pair
  // (32 bf16 = 64B) in one call. An odd trailing register falls back to fptrunc
  // + plain store.
  void VecFP32ToBF16(Operation *op, mlir::MLIRContext *ctx, Location &loc,
                     ConversionPatternRewriter &rewriter, int numElems,
                     int valueVecSize, int ptrDataVecSize,
                     SmallVector<Value> &valueElems, Value &lmBasePtr) const {
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // halfVecBf16Ty
    for (int i = 0; i < numElems / 2; ++i) {
      Value dstPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(2 * i));
      ValueRange args({dstPtr, valueElems[2 * i], valueElems[2 * i + 1]});
      LLVM::XPU::createDeviceCall("_ZN3xpu10vstore2_lmEPNS_8bfloat16EDv16_fS2_",
                                  rewriter, op, args, loc);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

protected:
  const xpu::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  bool isBf16Fast = false;
};

struct XPULoadOpConversion : public ConvertOpToLLVMPattern<triton::xpu::LoadOp>,
                             public LoadStoreConversionBase {
  XPULoadOpConversion(LLVMTypeConverter &converter,
                      const xpu::TargetInfo &targetInfo,
                      ModuleAxisInfoAnalysis &axisAnalysisPass,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // Escape hatch for the fence below; on by default because without it the
  // aliased-buffer read is simply wrong (findings 1.74).
  static bool lmWARFenceEnabled() {
    static const bool enabled =
        mlir::triton::tools::isEnvValueBool(
            mlir::triton::tools::getStrEnvXPU("TRITONXPU_LM_WAR_FENCE"))
            .value_or(true);
    return enabled;
  }

  // How many GM2LM-family ops fill the LM buffer this load reads. More than one
  // means MemoryInplace aliased several loads onto the same buffer, which is
  // how a later op comes to rewrite the words being read here. A single filler
  // inside a loop refills the same buffer on the next iteration, which is the
  // same hazard at a longer distance; measured and not reproducible
  // (findings 1.76), because the overwrite either reads the just-loaded
  // register or sits behind the mfence the loop's own store already emits.
  static unsigned countBufferFillers(triton::xpu::LoadOp op) {
    Operation *allocaOp = nullptr;
    if (Operation *def = op.getPtr().getDefiningOp()) {
      if (isa<triton::xpu::AllocaOp>(def))
        allocaOp = def;
      else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(def))
        allocaOp = gm2lmOp.getBufPtr().getDefiningOp();
      else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(def))
        allocaOp = gm2lmOp.getBufPtr().getDefiningOp();
    }
    if (!allocaOp || !isa<triton::xpu::AllocaOp>(allocaOp))
      return 0;
    unsigned fillers = 0;
    for (Operation *user : allocaOp->getUsers())
      if (isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp>(user))
        ++fillers;
    return fillers;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value res = op.getResult();
    Value ptr = op.getPtr();
    Value index = op.getIndex();

    int32_t stride = op.getStride();
    int32_t colSize = op.getTensorColSize();
    bool coreDealMultiRows = colSize != -1;
    bool isDiscreteSame = (stride == 0);
    bool isUnknown = stride == INT32_MIN ||
                     (stride != 0 && stride != 1 && !op.getIsDiscrete());
    bool bf16Tofp32Unordered = op.getBf16Tofp32Unordered();

    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llIndex = adaptor.getIndex();

    // Determine Type
    Type ptrTy = ptr.getType();
    Type resTy = res.getType();

    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type resElemTy = typeConverter->convertType(getElementTypeOrSelf(resTy));
    Type ptrDataVecTy = resElemTy;

    unsigned ptrNumElems = getTotalElemsPerThread(ptrTy);
    unsigned resNumElems = getTotalElemsPerThread(resTy);

    Type ptrElemScalarTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      ptrElemScalarTy =
          mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      ptrElemScalarTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }

    Type resElemScalarTy = getElementTypeOrSelf(resElemTy);

    ptrElemScalarTy = typeConverter->convertType(ptrElemScalarTy);
    resElemScalarTy = typeConverter->convertType(resElemScalarTy);

    // Get the LLVM values
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    stride = (stride != INT32_MIN &&
              ptrNumElems * std::abs(stride) <= targetInfo.getXPUBufferSize())
                 ? stride
                 : 1;

    assert(llPtrs.size() == ptrNumElems);
    bool isVectorized = false;
    unsigned vecSize = 1u;
    unsigned elemNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(resElemScalarTy)
            ? 64u
            : resElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(resElemTy)) {
      isVectorized = true;
      getVectorInfo(resElemTy, vecSize, elemNbits);
    }

    // fp16Tofp32
    if (resElemScalarTy.isF32() && ptrElemScalarTy.isF16()) {
      Value fp16LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      Value fp32LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      ValueRange singleOperandRange(
          {fp16LM, fp32LM, i32_val(ptrNumElems * stride)});
      mlir::LLVM::XPU::createDeviceCall("_ZN3xpu10fp16tofp32EPKNS_7float16EPfi",
                                        rewriter, op, singleOperandRange, loc);
      ptrElemScalarTy = resElemScalarTy;
    }
    // bf16Tofp32
    bool bf16Tofp32 = false;
    if (resElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
      int ptrVecSize = std::min(ptrNumElems, vecSize * 2);
      ptrDataVecTy = isVectorized ? VectorType::get(ptrVecSize, ptrElemScalarTy)
                                  : ptrElemScalarTy;
      bf16Tofp32 = true;
    }

    unsigned ptrDataVecSize = 1u;
    unsigned ptrDataNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(ptrElemScalarTy)
            ? 64u
            : ptrElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(ptrDataVecTy)) {
      getVectorInfo(ptrDataVecTy, ptrDataVecSize, ptrDataNbits);
    }

    SmallVector<Value> loadedVals;
    Value lmBasePtr = bitcast(llPtrs[0], ptr_ty(ctx, 0));
    if (stride < 0 && !op.getIsDiscrete()) {
      int32_t absStride = -stride;
      Value endOffset = i32_val((ptrNumElems - 1) * absStride);
      lmBasePtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr, endOffset);
    }
    if (index && !op.getIsDiscrete()) {
      ptrNumElems = resNumElems;
      unsigned _stride =
          (bf16Tofp32 && isVectorized)
              ? std::ceil(static_cast<double>(ptrNumElems * stride) / 2)
              : ptrNumElems * stride;
      _stride = coreDealMultiRows ? stride : _stride;
      Value idx = mul(llIndex, i32_val(_stride));
      lmBasePtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, idx);
      stride = coreDealMultiRows
                   ? std::ceil(static_cast<double>(colSize) / vecSize)
                   : stride;
    }

    if (op.getSVOpt()) {
      Value elemPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value loaded = load(ptrElemScalarTy, elemPtr);
      if (bf16Tofp32) {
        loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
      }
      loadedVals.push_back(loaded);
    } else if (isDiscreteSame) {
      Value elemPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value loaded = load(ptrElemScalarTy, elemPtr);
      if (bf16Tofp32) {
        loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
      }
      for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
        if (isVectorized) {
          Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t idx = 0; idx < vecSize; ++idx) {
            newVector =
                insert_element(resElemTy, newVector, loaded, i32_val(idx));
          }
          loadedVals.push_back(newVector);
        } else {
          loadedVals.push_back(loaded);
        }
      }
    } else if (op.getIsDiscrete()) {
      if (index) {
        unsigned elemsPerIter =
            isVectorized ? resNumElems * vecSize : resNumElems;
        unsigned iterNum = ptrNumElems / elemsPerIter;
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *mergeBlock =
            rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        for (size_t i = 0; i < resNumElems; ++i)
          mergeBlock->addArgument(resElemTy, loc);
        SmallVector<Block *> cases;
        for (unsigned i = 0; i < iterNum; ++i)
          cases.push_back(rewriter.createBlock(mergeBlock));
        rewriter.setInsertionPointToEnd(currentBlock);
        Block *check = currentBlock;
        for (unsigned i = 0; i < iterNum; ++i) {
          if (i + 1 == iterNum) {
            rewriter.create<LLVM::BrOp>(loc, ValueRange{}, cases[i]);
          } else {
            Block *next = rewriter.createBlock(cases[i + 1]);
            auto cond = icmp_eq(llIndex, i32_val(i));
            rewriter.create<LLVM::CondBrOp>(loc, cond, cases[i], next);
            check = next;
            rewriter.setInsertionPointToEnd(check);
          }
        }
        for (unsigned iter = 0; iter < iterNum; ++iter) {
          rewriter.setInsertionPointToEnd(cases[iter]);
          SmallVector<Value> vals;
          unsigned base = iter * elemsPerIter;
          for (size_t e = 0; e < resNumElems; ++e) {
            if (isVectorized) {
              Value vec = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
              for (unsigned lane = 0; lane < vecSize; ++lane) {
                Value ptr =
                    bitcast(llPtrs[base + e * vecSize + lane], ptr_ty(ctx, 0));
                Value v = load(ptrElemScalarTy, ptr);
                if (bf16Tofp32)
                  v = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, v);
                vec = insert_element(resElemTy, vec, v, i32_val(lane));
              }
              vals.push_back(vec);
            } else {
              Value ptr = bitcast(llPtrs[base + e], ptr_ty(ctx, 0));
              Value v = load(ptrElemScalarTy, ptr);
              if (bf16Tofp32)
                v = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, v);
              vals.push_back(v);
            }
          }
          rewriter.create<LLVM::BrOp>(loc, vals, mergeBlock);
        }
        rewriter.setInsertionPointToStart(mergeBlock);
        for (size_t i = 0; i < resNumElems; ++i)
          loadedVals.push_back(mergeBlock->getArgument(i));
      } else if (isVectorized) {
        for (size_t vecIdx = 0; vecIdx < resNumElems; ++vecIdx) {
          Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemIdx = 0; elemIdx < vecSize; ++elemIdx) {
            auto idx = vecIdx * vecSize + elemIdx;
            Value elemPtr = bitcast(llPtrs[idx], ptr_ty(ctx, 0));
            Value loaded = load(ptrElemScalarTy, elemPtr);
            if (bf16Tofp32) {
              loaded =
                  rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
            }
            // insert val to newVector
            newVector =
                insert_element(resElemTy, newVector, loaded, i32_val(elemIdx));
          }
          loadedVals.push_back(newVector);
        }
      } else {
        for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
          Value elemPtr = bitcast(llPtrs[elemIdx], ptr_ty(ctx, 0));
          Value loaded = load(ptrElemScalarTy, elemPtr);
          if (bf16Tofp32) {
            loaded =
                rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
          }
          loadedVals.push_back(loaded);
        }
      }
    } else if (!(coreDealMultiRows && index) && stride > 1 && isVectorized &&
               ptrDataVecSize * ptrDataNbits == 512) {
      // Vgather
      VectorType offsetTy =
          VectorType::get(ptrDataVecSize, int_ty(ptrDataNbits));
      Value offsetVec = rewriter.create<LLVM::UndefOp>(loc, offsetTy);
      for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
        Value offsetVal =
            int_val(ptrDataNbits, (ptrDataNbits / 8u) * stride * elemIdx);
        offsetVec = insert_element(offsetTy, offsetVec, offsetVal,
                                   int_val(ptrDataNbits, elemIdx));
      }
      Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      for (size_t vecIdx = 0;
           vecIdx < (index ? ptrNumElems : (ptrNumElems / ptrDataVecSize));
           ++vecIdx) {
        Value vecPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, _lmBasePtr,
                           int_val(ptrDataNbits, vecIdx * stride));
        Value tmpPtr = bitcast(vecPtr, ptr_ty(ctx, 0));
        Value vgather;
        if (ptrElemScalarTy.isF32() || ptrElemScalarTy.isInteger(32)) {
          vgather = rewriter.create<mlir::LLVM::XPU::VGatherFOp>(
              loc, offsetTy, tmpPtr, offsetVec);
        } else if (ptrElemScalarTy.isF16() || ptrElemScalarTy.isBF16() ||
                   ptrElemScalarTy.isInteger(16)) {
          vgather = rewriter.create<mlir::LLVM::XPU::VGatherHFOp>(
              loc, offsetTy, tmpPtr, offsetVec);
        } else {
          llvm_unreachable("Only support I16/FP16/BF16/I32/FP32 in VGather!");
        }
        Value loaded = bitcast(vgather, resElemTy);
        loadedVals.push_back(loaded);
      }
      if (bf16Tofp32) {
        VecBF16ToFP32(ctx, loc, rewriter, resElemTy, resNumElems, vecSize,
                      ptrDataVecSize, loadedVals);
      }
    } else {
      if (isVectorized) {
        if (bf16Tofp32) {
          if (bf16Tofp32Unordered) {
            VecBF16ToFP32Unordered(ctx, loc, rewriter, resElemTy, resNumElems,
                                   vecSize, ptrDataVecSize, lmBasePtr,
                                   loadedVals);
          } else {
            Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
            for (size_t elemIdx = 0; elemIdx < resNumElems / 2; elemIdx++) {
              Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, _lmBasePtr,
                                  i32_val(elemIdx * stride));
              Value loaded = load(ptrDataVecTy, elemPtr);
              loadedVals.push_back(loaded);
            }
            int remainedIdx = 2 * (resNumElems / 2);
            if (resNumElems - remainedIdx) {
              VectorType halfVecBf16Ty = VectorType::get(vecSize, bf16_ty);
              Value elemPtr = gep(ptr_ty(ctx, 0), halfVecBf16Ty, _lmBasePtr,
                                  i32_val(remainedIdx * stride));
              Value loaded = load(halfVecBf16Ty, elemPtr);
              loadedVals.push_back(loaded);
            }
            VecBF16ToFP32(ctx, loc, rewriter, resElemTy, resNumElems, vecSize,
                          ptrDataVecSize, loadedVals);
          }
        } else {
          Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
          for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
            Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, _lmBasePtr,
                                i32_val(elemIdx * stride));
            Value loaded = load(ptrDataVecTy, elemPtr);
            loadedVals.push_back(loaded);
          }
        }
      } else {
        Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        for (size_t elemIdx = 0; elemIdx < ptrNumElems; elemIdx++) {
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, _lmBasePtr,
                              i32_val(elemIdx * stride));
          Value loaded = load(ptrElemScalarTy, elemPtr);
          if (bf16Tofp32) {
            loaded =
                rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
          }
          loadedVals.push_back(loaded);
        }
      }
    }

    // Write-after-read hazard on an *aliased* LM buffer (findings 1.74).
    // MemoryInplace folds allocas whose IR lifetimes are disjoint into one
    // buffer, so the words this load just read get rewritten a few instructions
    // later by the next load's other-fill. On xpu3 that scalar store beats the
    // in-flight `vload_mask16` to word 0 of the buffer, per core and
    // nondeterministically: a vectorized three-operand welford reduce lost the
    // lane-0 contribution of ~55 of 64 cores, while the third operand -- the
    // one whose buffer is never rewritten afterwards -- was always intact.
    // Fence the reads so the buffer is free to be rewritten. Only a buffer with
    // more than one filler can be in that situation, and fencing just those
    // leaves every other kernel's code byte-identical.
    if (lmWARFenceEnabled() && countBufferFillers(op) > 1)
      createMfenceLMOp(rewriter, loc);

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPULoadScalarIndexedOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LoadScalarIndexedOp>,
      public LoadStoreConversionBase {
  XPULoadScalarIndexedOpConversion(LLVMTypeConverter &converter,
                                   const xpu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LoadScalarIndexedOp>(converter,
                                                                 benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LoadScalarIndexedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value res = op.getResult();
    Type resTy = res.getType();
    Type resElemTy = typeConverter->convertType(getElementTypeOrSelf(resTy));
    Type resElemScalarTy = getElementTypeOrSelf(resElemTy);
    resElemScalarTy = typeConverter->convertType(resElemScalarTy);

    unsigned addrSpace = 0;
    Type ptrTy = op.getPtr().getType();
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      if (auto pt =
              mlir::dyn_cast<triton::PointerType>(ptrTensorTy.getElementType()))
        addrSpace = pt.getAddressSpace();
    } else if (auto pt = mlir::dyn_cast<triton::PointerType>(ptrTy)) {
      addrSpace = pt.getAddressSpace();
    }

    auto llPtrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    Value basePtr = bitcast(llPtrs[0], ptr_ty(ctx, addrSpace));
    Value llIndex = adaptor.getIndex();
    Value elemPtr =
        gep(ptr_ty(ctx, addrSpace), resElemScalarTy, basePtr, llIndex);
    Value loaded = load(resElemScalarTy, elemPtr);

    unsigned resNumElems = getTotalElemsPerThread(resTy);
    bool isVectorized = mlir::isa<mlir::VectorType>(resElemTy);
    unsigned vecSize = 1u;
    if (isVectorized) {
      unsigned elemNbits = 0u;
      getVectorInfo(resElemTy, vecSize, elemNbits);
    }

    SmallVector<Value> loadedVals;
    for (size_t elemIdx = 0; elemIdx < resNumElems; ++elemIdx) {
      if (isVectorized) {
        Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
        for (size_t i = 0; i < vecSize; ++i) {
          newVector = insert_element(resElemTy, newVector, loaded, i32_val(i));
        }
        loadedVals.push_back(newVector);
      } else {
        loadedVals.push_back(loaded);
      }
    }

    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::StoreOp>,
      public LoadStoreConversionBase {
  XPUStoreOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value index = op.getIndex();

    int32_t colSize = op.getTensorColSize();
    bool coreDealMultiRows = colSize != -1;
    bool bf16Tofp32Unordered = op.getBf16Tofp32Unordered();
    auto dtype = op.getDtype();

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getValue();
    Value llIndex = adaptor.getIndex();

    // Determine Type
    Type ptrTy = ptr.getType();
    auto valueTy = value.getType();

    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type valueElemScalarTy = getElementTypeOrSelf(valueElemTy);
    Type ptrElemScalarTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      ptrElemScalarTy =
          mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      ptrElemScalarTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    ptrElemScalarTy = typeConverter->convertType(ptrElemScalarTy);
    valueElemScalarTy = typeConverter->convertType(valueElemScalarTy);

    Type ptrDataVecTy = valueElemTy;

    unsigned valueNumElems = getTotalElemsPerThread(valueTy);
    unsigned ptrNumElems = getTotalElemsPerThread(ptrTy);

    // Get the LLVM values
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    auto llVals = unpackLLElements(loc, llValue, rewriter);
    // Determine the vectorization size
    bool isVectorized = mlir::isa<mlir::VectorType>(valueElemTy);
    unsigned valueVecSize = 1u;
    unsigned valueScalarNbits = 32u;
    if (mlir::isa<mlir::VectorType>(valueElemTy)) {
      isVectorized = true;
      getVectorInfo(valueElemTy, valueVecSize, valueScalarNbits);
    }
    if (valueElemScalarTy.isInteger(32) && ptrElemScalarTy.isInteger(8))
      valueVecSize = dtype == Dtype::FP32 ? 16 : 32;

    // fp32 to bf16
    bool fp32Tobf16 = false;
    if (valueElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
      int ptrVecSize = std::min(ptrNumElems, valueVecSize * 2);
      ptrDataVecTy = isVectorized ? VectorType::get(ptrVecSize, ptrElemScalarTy)
                                  : ptrElemScalarTy;
      fp32Tobf16 = true;
    }

    bool fp32Tofp16 = false;
    if (valueElemScalarTy.isF32() && ptrElemScalarTy.isF16()) {
      ptrElemScalarTy = valueElemScalarTy;
      fp32Tofp16 = true;
    }

    unsigned ptrDataVecSize = 1u;
    unsigned ptrDataNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(ptrElemScalarTy)
            ? 64u
            : ptrElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(ptrDataVecTy)) {
      getVectorInfo(ptrDataVecTy, ptrDataVecSize, ptrDataNbits);
    }

    Value lmBasePtr = bitcast(llPtrs[0], ptr_ty(ctx, 0));
    unsigned stride = 1;
    if (index) {
      ptrNumElems = valueNumElems;
      unsigned _stride = (fp32Tobf16 && isVectorized)
                             ? std::ceil(static_cast<double>(ptrNumElems) / 2)
                             : ptrNumElems;
      _stride = coreDealMultiRows ? 1 : _stride;
      if (valueElemScalarTy.isInteger(32) && ptrElemScalarTy.isInteger(8)) {
        Value idx = mul(llIndex, i32_val(valueNumElems * valueVecSize));
        lmBasePtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr, idx);
      } else {
        Value idx = mul(llIndex, i32_val(_stride));
        lmBasePtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, idx);
      }
      stride = coreDealMultiRows
                   ? std::ceil(static_cast<double>(colSize) / ptrDataVecSize)
                   : stride;
    }

    if (valueElemScalarTy.isInteger(32) && ptrElemScalarTy.isInteger(8)) {
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      if (dtype == Dtype::FP32) {
        for (int i = 0; i < valueNumElems; i += 4) {
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(i * valueVecSize * stride));
          ValueRange args({llVals[i], llVals[i + 1], llVals[i + 2],
                           llVals[i + 3], elemPtr});
          if (valueElemScalarTy.isUnsignedInteger()) {
            LLVM::XPU::createDeviceCall("_ZN3xpu8vstorei8EjjjjPa", rewriter, op,
                                        args, loc);
          } else if (valueElemScalarTy.isInteger(32)) {
            LLVM::XPU::createDeviceCall("_ZN3xpu12vstorei8_i32EiiiiPa",
                                        rewriter, op, args, loc);
          } else if (valueElemScalarTy.isInteger(64)) {
            LLVM::XPU::createDeviceCall("_ZN3xpu12vstorei8_i64EllllPa",
                                        rewriter, op, args, loc);
          }
        }
      } else if (dtype == Dtype::FP16) {
        for (int i = 0; i < valueNumElems; i += 2) {
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(i * valueVecSize * stride));
          ValueRange args({llVals[i], llVals[i + 1], elemPtr});
          LLVM::XPU::createDeviceCall("_ZN3xpu16vstorei8_unroll2EjjPa",
                                      rewriter, op, args, loc);
        }
      } else {
        llvm_unreachable("vstorei8 only supports FP32 or FP16");
      }
    } else {
      if (isVectorized) {
        if (valueElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
          if (bf16Tofp32Unordered) {
            VecFP32ToBF16Unordered(ctx, loc, rewriter, valueNumElems,
                                   valueVecSize, ptrDataVecSize, llVals,
                                   lmBasePtr);
          } else {
            if (isBf16Fast) {
              VecFP32ToBF16(op, ctx, loc, rewriter, valueNumElems, valueVecSize,
                            ptrDataVecSize, llVals, lmBasePtr);
            } else {
              VecFP32ToBF16Slow(ctx, loc, rewriter, valueNumElems, valueVecSize,
                                ptrDataVecSize, llVals, lmBasePtr);
            }
          }
        } else {
          lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
          for (size_t elemIdx = 0; elemIdx < valueNumElems; elemIdx++) {
            Value elem = llVals[elemIdx];
            Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr,
                                i32_val(elemIdx * stride));
            elem = bf16ToFp16(rewriter, loc, valueElemTy, elem);
            store(elem, elemPtr);
          }
        }
      } else {
        lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        for (size_t elemIdx = 0; elemIdx < ptrNumElems; elemIdx++) {
          Value elem = llVals[elemIdx];
          if (fp32Tobf16) {
            elem = rewriter.create<LLVM::FPTruncOp>(loc, ptrElemScalarTy, elem);
          }
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(elemIdx * stride));
          store(elem, elemPtr);
        }
      }
    }

    createMfenceLMOp(rewriter, loc);

    // fp32 to fp16
    if (fp32Tofp16) {
      Value fp16LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      Value fp32LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      ValueRange singleOperandRange({fp32LM, fp16LM, i32_val(ptrNumElems)});
      mlir::LLVM::XPU::createDeviceCall("_ZN3xpu10fp32tofp16Ef", rewriter, op,
                                        singleOperandRange, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct XPUAllocaOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::AllocaOp>,
      public LoadStoreConversionBase {
  XPUAllocaOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::AllocaOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    auto resTy = op.getType();
    Type valueElemTy;
    if (auto resTensorTy = mlir::dyn_cast<RankedTensorType>(resTy)) {
      // Tensor
      valueElemTy =
          mlir::cast<triton::PointerType>(resTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      valueElemTy = mlir::cast<triton::PointerType>(resTy).getPointeeType();
    }
    valueElemTy = typeConverter->convertType(valueElemTy);

    unsigned numElems = getTotalElemsPerThread(resTy);

    auto allocNumElems = numElems;
    if (static_cast<XPUArch>(targetInfo.getXPUArch()) == XPUArch::XPU2 &&
        valueElemTy.isF16()) {
      // algin to 32, cause fp16tofp32 use vector<32*fp16> instruction
      allocNumElems = (allocNumElems + 31) / 32 * 32;
      // double space to accommodate 32*fp32
      allocNumElems *= 2;
    }
    for (auto user : op->getUsers()) {
      if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
        auto fixedStride = gm2lmOp.getFixedStride();
        if (fixedStride != INT32_MIN && fixedStride != 0 &&
            std::abs(fixedStride) * numElems <= targetInfo.getXPUBufferSize()) {
          allocNumElems *= std::abs(fixedStride);
        }
      } else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(user)) {
        auto fixedStride = gm2lmOp.getFixedStride();
        if (fixedStride != INT32_MIN && fixedStride != 0 &&
            std::abs(fixedStride) * numElems <= targetInfo.getXPUBufferSize()) {
          allocNumElems *= std::abs(fixedStride);
        }
      }
    }

    allocNumElems =
        align(allocNumElems, valueElemTy, 64); // 64 bytes aligned for LM
    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    bool boundaryBuf = llvm::any_of(op->getUsers(), [](Operation *user) {
      return isa<triton::xpu::PackOp, triton::xpu::UnpackOp>(user);
    });
    auto lmBuf = boundaryBuf
                     ? allocate(lmPtrTy, valueElemTy, i32_val(allocNumElems),
                                /*alignment=*/64)
                     : allocate(lmPtrTy, valueElemTy, i32_val(allocNumElems));

    SmallVector<Value> lmPtrs;
    for (int i = 0; i < numElems; i++) {
      lmPtrs.push_back(gep(lmPtrTy, valueElemTy, lmBuf, i32_val(i)));
    }
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Value resultStruct = packLLElements(loc, typeConverter, lmPtrs, rewriter,
                                        llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUGM2LMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GM2LMOp>,
      public LoadStoreConversionBase {
  XPUGM2LMOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GM2LMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GM2LMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value len = op.getLen();
    Value res = op.getResult();
    int32_t tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llLen = adaptor.getLen();
    Value llGMPtr = adaptor.getPtr();
    Value llLMPtr = adaptor.getBufPtr();
    Value resultStruct = llLMPtr;

    Type ptrTy = ptr.getType();
    Type resTy = res.getType();
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    assert(llLMPtr && "llBufPtr should not be null.");
    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);

    Value elemBytes = i32_val(elemNbits / 8u);
    Value offsetBytes = i32_val(0);

    bool mask = false;
    unsigned lenElemBit = 32;
    llvm::SmallVector<Value> llLens;
    if (op.getLen()) {
      auto lenElemTy = getElementTypeOrSelf(op.getLen().getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      mask = llGMPtrs.size() > 1 ? mlir::isa<mlir::IntegerType>(lenElemTy) &&
                                       (lenElemBit == 32 || lenElemBit == 64)
                                 : false;
    }

    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (mask) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);

    OffsetState offsetState = static_cast<OffsetState>(op.getOffsetState());
    int32_t fixedStride = op.getFixedStride();
    if (offsetState == OffsetState::Unknown) {
      /*  Small Col Size Opt Mask(14 < 16)

          Before Opt:
              T T T T T T T T
              T T T T T T F F

          After Opt:
              T T T T T T T F
              T T T T T T T F
      */
      SmallVector<bool> maskLists;
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        auto tensorRowSize =
            std::ceil(static_cast<double>(shape[0]) / 64);   // 128 / 64 = 2
        auto memColSize = shape[1];                          // 16
        unsigned rowRemainElem = memColSize - tensorColSize; // 16 - 15 = 1

        for (size_t row_idx = 0; row_idx < tensorRowSize; ++row_idx) {
          for (size_t col_idx = 0; col_idx < tensorColSize; ++col_idx) {
            maskLists.push_back(true);
          }

          for (size_t remainElem = rowRemainElem; remainElem > 0;
               --remainElem) {
            maskLists.push_back(false);
          }
        }
      }

      if (fixedStride > 0 &&
          numElems * fixedStride <= targetInfo.getXPUBufferSize()) {
        // Unknown FixedStride Vgather
        readBytes = mul(i32_val(fixedStride), readBytes);
      } else if (fixedStride < 0 && fixedStride != INT32_MIN &&
                 numElems * (-fixedStride) <= targetInfo.getXPUBufferSize()) {
        int32_t absStride = -fixedStride;
        readBytes = mul(i32_val(absStride), readBytes);
        int64_t elemSizeBytes = static_cast<int64_t>(elemNbits / 8u);
        int64_t byteOffset =
            static_cast<int64_t>(numElems - 1) * fixedStride * elemSizeBytes;
        offsetBytes = add(offsetBytes, i32_val(byteOffset));
      } else {
        // Unknown
        readBytes = elemBytes;
        for (size_t i = 0; i < llGMPtrs.size(); ++i) {
          //  Protect Ptr Boundary Condition
          Value base;
          if (coreDealMultiRows) {
            base = mask ? select(int_val(1, maskLists[i]), llGMPtrs[i],
                                 llGMPtrs[0])
                        : llGMPtrs[i];
          } else {
            if (mask) { // Has Mask
              if (llLens[0].getType().isInteger(32)) {
                base = select(icmp_slt(i32_val(i), llLens[0]), llGMPtrs[i],
                              llGMPtrs[0]);
              } else if (llLens[0].getType().isInteger(64)) {
                base = select(icmp_slt(i64_val(i), llLens[0]), llGMPtrs[i],
                              llGMPtrs[0]);
              } else {
                llvm_unreachable("Unsupported Mask Int Type");
              }
            } else {
              base = llGMPtrs[i];
            }
          }
          Value dstPtr = bitcast(llLMPtrs[i], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(base, ptr_ty(ctx, 1));
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        }
        // Fence once after issuing all per-element DMAs. A per-iteration fence
        // would serialize every element's DMA; a single trailing fence still
        // guarantees the whole LM buffer is populated before the load consumes
        // it, while letting the individual DMAs overlap.
        if (!async)
          createMfenceLMOp(rewriter, loc);

        resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                      llvmResultStructTy);
        rewriter.replaceOp(op, {resultStruct});
        return success();
      }
    } else if (offsetState == OffsetState::Discrete) {
      // Reorder the local buffer ptrs.
      SmallVector<Value> newLmBufPtrs(llGMPtrs.size());
      Value basePtrInt = ptrtoint(i64_ty, llGMPtrs[0]);
      for (size_t idx = 0; idx < llGMPtrs.size(); ++idx) {
        Value elemPtrInt = ptrtoint(i64_ty, llGMPtrs[idx]); // convert to int
        Value offsetBytes =
            sub(elemPtrInt, basePtrInt); // get the offset(Bytes)
        Value elemPtr = gep(ptr_ty(ctx, 0), i8_ty, llLMPtrs[0], offsetBytes);
        newLmBufPtrs[idx] = elemPtr;
      }
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::DiscreteSame) {
      readBytes = elemBytes;
      SmallVector<Value> newLmBufPtrs(llLMPtrs.size(), llLMPtrs[0]);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llLMPtr.getType());
    } else if (offsetState == OffsetState::LocallyScalar) {
      int64_t rowLen = op.getRowLen();
      if (rowLen <= 0)
        rowLen = static_cast<int64_t>(llLMPtrs.size());
      readBytes = elemBytes;
      SmallVector<Value> newLmBufPtrs(llLMPtrs.size());
      for (size_t start = 0; start < llLMPtrs.size();
           start += static_cast<size_t>(rowLen)) {
        Value dstPtr = bitcast(llLMPtrs[start], ptr_ty(ctx, 0));
        Value srcPtr = bitcast(llGMPtrs[start], ptr_ty(ctx, 1));
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
        size_t end = start + static_cast<size_t>(rowLen);
        if (end > llLMPtrs.size())
          end = llLMPtrs.size();
        for (size_t j = start; j < end; ++j)
          newLmBufPtrs[j] = llLMPtrs[start];
      }
      if (!async)
        createMfenceLMOp(rewriter, loc);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
      rewriter.replaceOp(op, {resultStruct});
      return success();
    } else if (offsetState == OffsetState::LocallyContinuous) {
      int64_t _rowLen = op.getRowLen();
      int64_t _rowStride = op.getRowStride();
      if (_rowLen % numElems == 0) {
        offsetState = OffsetState::Continuous;
        LLVM_DEBUG(llvm::dbgs() << "[OffsetState]: GM2LM Update "
                                   "LocallyContinuous to Continuous\n");
      } else {
        auto oldBlock = op->getBlock();
        auto newBlock = oldBlock->splitBlock(op->getNextNode());
        int64_t _elemBytes = elemNbits / 8u;
        int64_t _bufLen = static_cast<int64_t>(numElems);
        LLVM_DEBUG(llvm::dbgs() << "[GM2LM LocallyContinuous]: rowLen is "
                                << _rowLen << ", rowStride is " << _rowStride
                                << ", bufLen is " << _bufLen << "\n");
        if (_rowStride == -1) {
          lowerLocallyContinuousUnfixedStride(
              op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llGMPtr, llLMPtr,
              llLen, offsetBytes, MemCpyType::GM2LM, oldBlock, newBlock);
        } else {
          if (_rowLen > _bufLen) {
            lowerLocallyContinuousLargeRow(
                op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM, oldBlock,
                newBlock);
          } else {
            lowerLocallyContinuousSmallRow(
                op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM, oldBlock,
                newBlock);
          }
        }

        if (!async)
          createMfenceLMOp(rewriter, loc);

        resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                      llvmResultStructTy);
        rewriter.replaceOp(op, {resultStruct});
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
    }

    if (coreDealMultiRows) {
      auto shape = cast<RankedTensorType>(ptrTy).getShape();
      int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
      if (tensorColSize % shape[1] == 0) {
        readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
        Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
        Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      } else {
        readBytes = mul(i32_val(tensorColSize), elemBytes);
        for (int i = 0; i < tensorRowSize; ++i) {
          Value dstPtr = bitcast(llLMPtrs[i * shape[1]], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(llGMPtrs[i * tensorColSize], ptr_ty(ctx, 1));
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        }
      }

      if (!async)
        createMfenceLMOp(rewriter, loc);

      rewriter.replaceOp(op, {resultStruct});
      return success();
    }

    Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
    Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
    createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);
    if (!async)
      createMfenceLMOp(rewriter, loc);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUStageSMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::StageSMOp>,
      public LoadStoreConversionBase {
  XPUStageSMOpConversion(LLVMTypeConverter &converter,
                         const xpu::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::StageSMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::StageSMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Type ptrTy = ptr.getType();
    Type elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    elemTy = typeConverter->convertType(elemTy);
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();

    Value srcPtr = bitcast(adaptor.getPtr(), ptr_ty(ctx, 1));
    Value smBase = getGlobalSmemBase(loc, rewriter, op);
    Value smDst = gep(ptr_ty(ctx, 2), i8_ty, smBase, adaptor.getSmOffset());

    Value elemBytes = i32_val(elemNbits / 8u);
    Value bufElems = adaptor.getBufElems();
    Value readLen = bufElems;
    if (op.getLen()) {
      Value reqLen = smax(adaptor.getLen(), i32_val(0));
      readLen = smin(reqLen, bufElems);
    }

    emitPartitionedGM2SM(loc, rewriter, ctx, op, srcPtr, smDst, readLen,
                         elemBytes);

    // The result is the scalar SM base pointer (opaque ptr<2>).
    rewriter.replaceOp(op, {smDst});
    return success();
  }
};

struct XPULM2GMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LM2GMOp>,
      public LoadStoreConversionBase {

  XPULM2GMOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LM2GMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LM2GMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value len = op.getLen();
    int32_t offsetStateInt = op.getOffsetState();
    OffsetState offsetState = static_cast<OffsetState>(offsetStateInt);
    auto tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    offsetState = (tensorColSize == 1) ? OffsetState::Continuous : offsetState;

    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llLen = adaptor.getLen();
    Value llBufPtr = adaptor.getBufPtr();
    assert(llBufPtr && "llBufPtr should not be null.");

    // Get elemTy and numElems
    Type ptrTy = ptr.getType();
    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    Value elemBytes = i32_val(elemNbits / 8u);
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    // Get base, readBytes and offsetBytes
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);

    Value base = llPtrs[0];
    Value offsetBytes = i32_val(0);

    llvm::SmallVector<Value> llLens;
    bool mask = false;
    unsigned lenElemBit = 32;
    if (op.getLen()) {
      auto lenElemTy = getElementTypeOrSelf(op.getLen().getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      mask = llPtrs.size() > 1 ? mlir::isa<mlir::IntegerType>(lenElemTy) &&
                                     (lenElemBit == 32 || lenElemBit == 64)
                               : false;
    }
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (mask) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);
    auto lmBufPtrs = unpackLLElements(loc, llBufPtr, rewriter);
    Value lmBuf = lmBufPtrs[0];

    // Create LM2GM and mfence
    switch (offsetState) {
    case OffsetState::Continuous: {
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
        if (tensorColSize % shape[1] == 0) {
          readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
          Value srcPtr = bitcast(lmBufPtrs[0], ptr_ty(ctx, 0));
          Value dstPtr = bitcast(llPtrs[0], ptr_ty(ctx, 1));
          createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        } else {
          for (int i = 0; i < tensorRowSize; ++i) {
            readBytes = mul(i32_val(tensorColSize), elemBytes);
            Value srcPtr = bitcast(lmBufPtrs[i * shape[1]], ptr_ty(ctx, 0));
            Value dstPtr = bitcast(llPtrs[i * tensorColSize], ptr_ty(ctx, 1));
            createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                          readBytes);
          }
        }

      } else {
        Value srcPtr = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value basePtr = bitcast(base, ptr_ty(ctx, 1));
        createLM2GMOp(rewriter, ctx, loc, srcPtr, basePtr, offsetBytes,
                      readBytes);
      }
      break;
    }
    case OffsetState::LocallyContinuous: {
      int64_t _rowLen = op.getRowLen();
      int64_t _rowStride = op.getRowStride();
      if (_rowLen % numElems == 0) {
        offsetState = OffsetState::Continuous;
        LLVM_DEBUG(llvm::dbgs() << "[OffsetState]: LM2GM Update "
                                   "LocallyContinuous to Continuous\n");
      } else {
        auto oldBlock = op->getBlock();
        auto newBlock = oldBlock->splitBlock(op->getNextNode());
        int64_t _elemBytes = elemNbits / 8u;
        int64_t _bufLen = static_cast<int64_t>(numElems);
        LLVM_DEBUG(llvm::dbgs() << "[LM2GM LocallyContinuous]: rowLen is "
                                << _rowLen << ", rowStride is " << _rowStride
                                << ", bufLen is " << _bufLen << "\n");

        if (_rowStride == -1) {
          lowerLocallyContinuousUnfixedStride(
              op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llPtr, llBufPtr,
              llLen, offsetBytes, MemCpyType::LM2GM, oldBlock, newBlock);
        } else {
          if (_rowLen > _bufLen) {
            lowerLocallyContinuousLargeRow(
                op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM, oldBlock,
                newBlock);
          } else {
            lowerLocallyContinuousSmallRow(
                op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM, oldBlock,
                newBlock);
          }
        }
        if (!async)
          createMfenceLMOp(rewriter, loc);
        rewriter.eraseOp(op);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
      break;
    }
    case OffsetState::Unknown: {
      for (size_t llPtrIdx = 0; llPtrIdx < llPtrs.size(); ++llPtrIdx) {
        Value maskedIdx;
        if (mask) {
          auto llLenTy = llLens[0].getType();
          if (llLenTy.isInteger(32)) {
            maskedIdx = select(icmp_slt(i32_val(llPtrIdx), llLens[0]),
                               i32_val(llPtrIdx), i32_val(0));
          } else if (llLenTy.isInteger(64)) {
            maskedIdx = select(icmp_slt(i64_val(llPtrIdx), llLens[0]),
                               i64_val(llPtrIdx), i64_val(0));
          } else {
            llvm_unreachable("Unsupported Mask Int Type");
          }
        } else {
          maskedIdx = i32_val(llPtrIdx);
        }

        lmBuf = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value elemPtr = gep(ptr_ty(ctx, 0), elemTy, lmBuf, maskedIdx);
        Value srcPtr = bitcast(elemPtr, ptr_ty(ctx, 0));
        // Protect Ptr Boundary Condition
        Value dstPtr;
        if (mask) {
          auto llLenTy = llLens[0].getType();
          if (llLenTy.isInteger(32)) {
            dstPtr = select(icmp_slt(i32_val(llPtrIdx), llLens[0]),
                            llPtrs[llPtrIdx], llPtrs[0]);
          } else if (llLenTy.isInteger(64)) {
            dstPtr = select(icmp_slt(i64_val(llPtrIdx), llLens[0]),
                            llPtrs[llPtrIdx], llPtrs[0]);
          } else {
            llvm_unreachable("Unsupported Mask Int Type");
          }
        } else {
          dstPtr = llPtrs[llPtrIdx];
        }
        createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      elemBytes);
      }
      break;
    }
    default:
      llvm_unreachable("Unknown offset state");
      break;
    }
    if (!async)
      createMfenceLMOp(rewriter, loc);
    rewriter.eraseOp(op);

    return success();
  }
};

struct XPUGM2LMMaskOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GM2LMMaskOp>,
      public LoadStoreConversionBase {
  XPUGM2LMMaskOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GM2LMMaskOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GM2LMMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value len = op.getLen();
    Value res = op.getResult();
    int32_t tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llMask = adaptor.getMask();
    Value llLen = adaptor.getLen();
    Value llGMPtr = adaptor.getPtr();
    Value llLMPtr = adaptor.getBufPtr();
    Value resultStruct = llLMPtr;

    Type ptrTy = ptr.getType();
    Type resTy = res.getType();
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    assert(llLMPtr && "llBufPtr should not be null.");
    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    llvm::SmallVector<Value> llLens;

    Value elemBytes = i32_val(elemNbits / 8u);
    Value offsetBytes = i32_val(0);

    llvm::SmallVector<Value> llMasks;
    if (mask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }

    unsigned lenElemBit = 32;
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (len) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      auto lenElemTy = getElementTypeOrSelf(len.getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);

    Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
    Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));

    int32_t fixedStride = op.getFixedStride();
    int64_t _rowLen = op.getRowLen();
    int64_t _rowStride = op.getRowStride();
    OffsetState offsetState = static_cast<OffsetState>(op.getOffsetState());
    if (offsetState == OffsetState::LocallyContinuous &&
        _rowLen % numElems == 0) {
      offsetState = OffsetState::Continuous;
      LLVM_DEBUG(
          llvm::dbgs()
          << "[OffsetState]: GM2LM Update LocallyContinuous to Continuous\n");
    }
    if (offsetState == OffsetState::Unknown) {
      /*  Small Col Size Opt Mask(14 < 16)

          Before Opt:
              T T T T T T T T
              T T T T T T F F

          After Opt:
              T T T T T T T F
              T T T T T T T F
      */
      SmallVector<bool> maskLists;
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        auto tensorRowSize =
            std::ceil(static_cast<double>(shape[0]) / 64);   // 128 / 64 = 2
        auto memColSize = shape[1];                          // 16
        unsigned rowRemainElem = memColSize - tensorColSize; // 16 - 15 = 1

        for (size_t row_idx = 0; row_idx < tensorRowSize; ++row_idx) {
          for (size_t col_idx = 0; col_idx < tensorColSize; ++col_idx) {
            maskLists.push_back(true);
          }

          for (size_t remainElem = rowRemainElem; remainElem > 0;
               --remainElem) {
            maskLists.push_back(false);
          }
        }
      }

      if (fixedStride > 0 &&
          numElems * fixedStride <= targetInfo.getXPUBufferSize()) {
        // Unknown FixedStride Vgather
        readBytes = mul(i32_val(fixedStride), readBytes);
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      } else if (fixedStride < 0 && fixedStride != INT32_MIN &&
                 numElems * (-fixedStride) <= targetInfo.getXPUBufferSize()) {
        int32_t absStride = -fixedStride;
        readBytes = mul(i32_val(absStride), readBytes);
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        int64_t elemSizeBytes = static_cast<int64_t>(elemNbits / 8u);
        int64_t byteOffset =
            static_cast<int64_t>(numElems - 1) * fixedStride * elemSizeBytes;
        Value srcPtrInt = ptrtoint(i64_ty, llGMPtrs[0]);
        Value adjustedSrcInt = add(srcPtrInt, i64_val(byteOffset));
        srcPtr =
            bitcast(inttoptr(ptr_ty(ctx, 1), adjustedSrcInt), ptr_ty(ctx, 1));
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      } else {
        // Unknown
        for (size_t i = 0; i < llGMPtrs.size(); ++i) {
          //  Protect Ptr Boundary Condition
          Value base = llGMPtrs[i];
          if (coreDealMultiRows) {
            base =
                len ? select(int_val(1, maskLists[i]), llGMPtrs[i], llGMPtrs[0])
                    : llGMPtrs[i];
          }
          Value dstPtr = bitcast(llLMPtrs[i], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(llGMPtrs[i], ptr_ty(ctx, 1));
          Value _readBytes =
              mask ? select(llMasks[i], elemBytes, i32_val(0)) : elemBytes;
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        _readBytes);
        }
      }
      resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::Discrete) {
      // Reorder the local buffer ptrs.
      SmallVector<Value> newLmBufPtrs(llGMPtrs.size());
      Value basePtrInt = ptrtoint(i64_ty, llGMPtrs[0]);
      for (size_t idx = 0; idx < llGMPtrs.size(); ++idx) {
        Value elemPtrInt = ptrtoint(i64_ty, llGMPtrs[idx]); // convert to int
        Value offsetBytes =
            sub(elemPtrInt, basePtrInt); // get the offset(Bytes)
        Value elemPtr = gep(ptr_ty(ctx, 0), i8_ty, llLMPtrs[0], offsetBytes);
        newLmBufPtrs[idx] = elemPtr;
      }
      readBytes = mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
      createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);

      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::DiscreteSame) {
      readBytes = elemBytes;
      readBytes = mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
      createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);

      SmallVector<Value> newLmBufPtrs(llLMPtrs.size(), llLMPtrs[0]);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::LocallyContinuous) {
      auto oldBlock = op->getBlock();
      auto newBlock = oldBlock->splitBlock(op->getNextNode());
      int64_t _elemBytes = elemNbits / 8u;
      int64_t _bufLen = static_cast<int64_t>(numElems);
      LLVM_DEBUG(llvm::dbgs() << "[GM2LM LocallyContinuous]: rowLen is "
                              << _rowLen << ", rowStride is " << _rowStride
                              << ", bufLen is " << _bufLen << "\n");
      if (_rowStride == -1) {
        lowerLocallyContinuousUnfixedStrideMask(
            op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llGMPtr, llLMPtr,
            llMask, llLen, offsetBytes, MemCpyType::GM2LM, oldBlock, newBlock);
      } else {
        if (_rowLen > _bufLen) {
          lowerLocallyContinuousUnfixedStrideMask(
              op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM,
              oldBlock, newBlock);
        } else {
          lowerLocallyContinuousSmallRowMask(
              op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM,
              oldBlock, newBlock);
        }
      }
      resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                    llvmResultStructTy);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
    } else if (offsetState == OffsetState::Continuous) {
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
        if (tensorColSize > 0 && tensorColSize % shape[1] == 0) {
          Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
          readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
          readBytes =
              mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        } else {
          // readBytes = mul(i32_val(tensorColSize), elemBytes);
          for (int i = 0; i < tensorRowSize; ++i) {
            Value srcPtr = bitcast(llGMPtrs[i * shape[1]], ptr_ty(ctx, 1));
            Value dstPtr = bitcast(llLMPtrs[i * shape[1]], ptr_ty(ctx, 0));
            auto _readBytes =
                mask ? select(llMasks[i * shape[1]], readBytes, i32_val(0))
                     : readBytes;
            createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                          _readBytes);
          }
        }
      } else {
        Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
        Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[GM2LM]: offsetState is " << offsetState
                              << ", is not supported\n");
    }

    if (!async)
      createMfenceLMOp(rewriter, loc);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPULM2GMMaskOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LM2GMMaskOp>,
      public LoadStoreConversionBase {

  XPULM2GMMaskOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LM2GMMaskOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LM2GMMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();
    Value len = op.getLen();
    int32_t offsetStateInt = op.getOffsetState();
    OffsetState offsetState = static_cast<OffsetState>(offsetStateInt);
    auto tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    offsetState = (tensorColSize == 1) ? OffsetState::Continuous : offsetState;

    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llLen = adaptor.getLen();
    Value llBufPtr = adaptor.getBufPtr();
    assert(llBufPtr && "llBufPtr should not be null.");

    // Get elemTy and numElems
    Type ptrTy = ptr.getType();
    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    Value elemBytes = i32_val(elemNbits / 8u);
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    // Get base, readBytes and offsetBytes
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    llvm::SmallVector<Value> llMasks;
    llvm::SmallVector<Value> llLens;
    Value base = llPtrs[0];
    Value offsetBytes = i32_val(0);
    if (mask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }

    unsigned lenElemBit = 32;
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (len) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      auto lenElemTy = getElementTypeOrSelf(len.getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }

    Value readBytes = mul(readLen, elemBytes);
    auto lmBufPtrs = unpackLLElements(loc, llBufPtr, rewriter);
    Value lmBuf = lmBufPtrs[0];

    // Create LM2GM and mfence
    int64_t _rowLen = op.getRowLen();
    int64_t _rowStride = op.getRowStride();
    if (offsetState == OffsetState::LocallyContinuous &&
        _rowLen % numElems == 0) {
      offsetState = OffsetState::Continuous;
      LLVM_DEBUG(
          llvm::dbgs()
          << "[OffsetState]: LM2GM Update LocallyContinuous to Continuous\n");
    }
    switch (offsetState) {
    case OffsetState::Continuous: {
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
        if (tensorColSize > 0 && tensorColSize % shape[1] == 0) {
          readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
          Value srcPtr = bitcast(lmBufPtrs[0], ptr_ty(ctx, 0));
          Value dstPtr = bitcast(llPtrs[0], ptr_ty(ctx, 1));
          readBytes =
              mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
          createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        } else {
          // readBytes = mul(i32_val(tensorColSize), elemBytes);
          for (int i = 0; i < tensorRowSize; ++i) {
            Value srcPtr = bitcast(lmBufPtrs[i * shape[1]], ptr_ty(ctx, 0));
            Value dstPtr = bitcast(llPtrs[i * shape[1]], ptr_ty(ctx, 1));
            auto _readBytes =
                mask ? select(llMasks[i * shape[1]], readBytes, i32_val(0))
                     : readBytes;
            createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                          _readBytes);
          }
        }
      } else {
        Value srcPtr = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value basePtr = bitcast(base, ptr_ty(ctx, 1));
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        createLM2GMOp(rewriter, ctx, loc, srcPtr, basePtr, offsetBytes,
                      readBytes);
      }
      break;
    }
    case OffsetState::LocallyContinuous: {

      auto oldBlock = op->getBlock();
      auto newBlock = oldBlock->splitBlock(op->getNextNode());
      int64_t _elemBytes = elemNbits / 8u;
      int64_t _bufLen = static_cast<int64_t>(numElems);
      LLVM_DEBUG(llvm::dbgs() << "[LM2GM LocallyContinuous]: rowLen is "
                              << _rowLen << ", rowStride is " << _rowStride
                              << ", bufLen is " << _bufLen << "\n");

      if (_rowStride == -1) {
        lowerLocallyContinuousUnfixedStrideMask(
            op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llPtr, llBufPtr,
            llMask, llLen, offsetBytes, MemCpyType::LM2GM, oldBlock, newBlock);
      } else {
        if (_rowLen > _bufLen) {
          lowerLocallyContinuousUnfixedStrideMask(
              op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM,
              oldBlock, newBlock);
        } else {
          lowerLocallyContinuousSmallRowMask(
              op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM,
              oldBlock, newBlock);
        }
        if (!async)
          createMfenceLMOp(rewriter, loc);
        rewriter.eraseOp(op);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
      createMfenceLMOp(rewriter, loc);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
      break;
    }
    case OffsetState::Unknown: {
      size_t ngroup = 1;
      size_t groupsize = 1;
      Value isGroupZero = icmp_eq(i32_val(0), i32_val(0));
      getLayoutInfo(value.getType(), ngroup, groupsize);
      if (ngroup * groupsize <= 64 && ngroup == 1) {
        Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
        Value groupId = sdiv(coreId, i32_val(groupsize));
        isGroupZero = icmp_eq(groupId, i32_val(0));
      }
      for (size_t llPtrIdx = 0; llPtrIdx < llPtrs.size(); ++llPtrIdx) {
        Value maskedIdx = i32_val(llPtrIdx);
        Value _lmBuf = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value elemPtr = gep(ptr_ty(ctx, 0), elemTy, _lmBuf, maskedIdx);
        Value srcPtr = bitcast(elemPtr, ptr_ty(ctx, 0));
        Value dstPtr = llPtrs[llPtrIdx];
        Value _mask;
        if (mask) {
          _mask = llMasks[llPtrIdx];
          if (ngroup * groupsize < 64 && ngroup == 1) {
            _mask = and_(_mask, isGroupZero);
          }
        }
        readBytes = mask ? select(_mask, elemBytes, i32_val(0)) : elemBytes;
        createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      }
      break;
    }
    default: {
      llvm_unreachable("Unknown offset state");
      break;
    }
    }
    if (!async)
      createMfenceLMOp(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

struct XPUAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {

  XPUAtomicRMWOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Value val = op.getVal();
    Value mask = op.getMask();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    auto llValues = unpackLLElements(loc, llValue, rewriter);

    auto resTy = op.getType();
    Type valueElemTy = getElementTypeOrSelf(getElementTypeOrSelf(resTy));
    unsigned numElems = getTotalElemsPerThread(resTy);

    std::string funcName;
    if (valueElemTy.isF16()) {
      switch (atomicRmwAttr) {
      case RMWOp::ADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1DF16_DF16_";
        break;
      case RMWOp::FADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1DF16_DF16_";
        break;
      default:
        return failure();
      }
    } else {
      switch (atomicRmwAttr) {
      case RMWOp::ADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1ff";
        break;
      case RMWOp::FADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1ff";
        break;
      default:
        return failure();
      }
    }

    SmallVector<Value> resultVals(numElems);
    for (unsigned i = 0; i < numElems; ++i) {
      ValueRange operandRange({llPtrs[i], llValues[i]});
      Value devCall = mlir::LLVM::XPU::createDeviceCall(
          funcName, rewriter, op, valueElemTy, operandRange, loc);
      resultVals[i] = devCall;
    }

    Type structTy = this->getTypeConverter()->convertType(resTy);
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, resultStruct);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// TLE Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower ttg::LocalAllocOp to XPU local memory allocation.
/// This is the TLE equivalent of XPUAllocaOpConversion.
///
/// SPMD model: the memdesc shape describes the WHOLE tile, but on XPU
/// each core owns only a contiguous slice of the tile. The tile is partitioned
/// across `groupSize` (= coresPerGroup = threads-per-warp) cores, so each core
/// allocates only `ceil(tileElems / groupSize)` elements of private LM. This
/// matches make_range/local_ptr which index the buffer with per-core-local
/// indices [0, elemsPerCore).
struct XPUTLELocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>,
      public LoadStoreConversionBase {
  XPUTLELocalAllocOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy = op.getType();
    auto shape = memDescTy.getShape();
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = getTypeConverter()->convertType(elemTy);

    // SM (cluster-shared) buffer: no per-core LM allocation. All cores share
    // one copy at global_smem + xpu.sm_offset (assigned by
    // tritonxpu-tle-sm-alloc).
    //
    // The value produced here is a DEAD PLACEHOLDER: every consumer of an smem
    // buffer (copy_g2l / local_ptr / vload) recomputes the ptr<2> base itself
    // via getTLESmemBase, because the MemDescType -> ptr<0> type-converter
    // contract forces this result into addrspace 0 and an addrspace_cast'ed SM
    // pointer must never actually be dereferenced. It exists only to satisfy
    // that contract (and to keep the IR verifiable); writes through it are
    // rejected by the asserts in the local_store / vstore lowerings.
    if (Operation *allocOp = getTLESmemAlloc(op.getResult())) {
      Value smDst = getTLESmemBase(loc, rewriter, op, allocOp);
      Value placeholder = addrspace_cast(ptr_ty(ctx, 0), smDst);
      rewriter.replaceOp(op, placeholder);
      return success();
    }

    // Total elements in the whole tile.
    unsigned tileElems = 1;
    for (auto dim : shape)
      tileElems *= dim;

    // Per-core element count: partition the tile across cores in a group.
    // When tritonxpu-tle-core-tiling stamped a ClusterLayout, size the private
    // LM from the layout: numElems = Prod_k ceil(shape[k],
    // coresPerGroup[k]*groupsPerCluster[k]) (== getTotalElemsPerThread, and ==
    // the legacy flat cut for single-level layouts). Otherwise fall back to the
    // flat cut ceil(tileElems, groupSize).
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned elemsPerCore;
    if (auto tileLayout = op->getAttrOfType<triton::xpu::ClusterLayoutAttr>(
            "xpu.tile_layout")) {
      auto coresPerGroup = tileLayout.getCoresPerGroup();
      auto groupsPerCluster = tileLayout.getGroupsPerCluster();
      auto sizePerCore = tileLayout.getSizePerCore();
      unsigned rank = shape.size();
      if (rank == 1 && groupsPerCluster[0] > 1) {
        // LargeN result buffer: distributed over `numGroups` groups only (the
        // groupSize col-cores replicate the row), so size = ceil(M, numGroups)
        // = sizePerCore[0].
        elemsPerCore = sizePerCore[0];
      } else {
        elemsPerCore = 1;
        for (unsigned d = 0; d < rank; ++d) {
          unsigned coresAlongDim = coresPerGroup[d] * groupsPerCluster[d];
          assert(coresAlongDim > 0 &&
                 "cluster layout dim has zero cores (invalid ClusterLayout)");
          elemsPerCore *= (shape[d] + coresAlongDim - 1) / coresAlongDim;
        }
      }
    } else {
      unsigned groupSize =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      if (groupSize == 0)
        groupSize = 1;
      elemsPerCore = (tileElems + groupSize - 1) / groupSize;
    }

    auto allocNumElems = elemsPerCore;
    // XPU2 FP16: align to 32 + double space for fp16tofp32
    if (static_cast<XPUArch>(targetInfo.getXPUArch()) == XPUArch::XPU2 &&
        llvmElemTy.isF16()) {
      allocNumElems = (allocNumElems + 31) / 32 * 32;
      allocNumElems *= 2;
    }

    // 64 bytes aligned for LM
    allocNumElems = align(allocNumElems, llvmElemTy, 64);

    // A `#triton_xpu.smem` buffer is not an allocation at all: SM is statically
    // partitioned, so the "alloc" is just a compile-time offset into the
    // cluster-wide block. The region holds one slice per core, laid out with
    // the per-core size as the stride, so core `c` owns [smem_offset +
    // c*perCoreBytes, +perCoreBytes).
    if (triton::xpu::isSharedMemDesc(memDescTy)) {
      if (static_cast<XPUArch>(targetInfo.getXPUArch()) == XPUArch::XPU2)
        return op.emitError("#triton_xpu.smem buffers need arch >= 3 (there is "
                            "no XPU2 GM->SM DMA intrinsic)");
      auto offAttr =
          op->getAttrOfType<IntegerAttr>(triton::xpu::kSharedMemOffsetAttrName);
      if (!offAttr)
        return op.emitError("#triton_xpu.smem buffer carries no ")
               << triton::xpu::kSharedMemOffsetAttrName
               << "; ConvertTritonXPUToLLVM stamps it before conversion";
      unsigned elemBytes =
          std::max<unsigned>(llvmElemTy.getIntOrFloatBitWidth(), 8) / 8;
      Value smBase = getGlobalSmemBase(loc, rewriter, op);
      Value sliceOff = mul(tid_val(), i32_val(allocNumElems * elemBytes));
      Value slice = gep(ptr_ty(ctx, 2), i8_ty, smBase, sliceOff);
      Value buf = gep(ptr_ty(ctx, 2), i8_ty, slice,
                      i32_val(static_cast<int32_t>(offAttr.getInt())));
      rewriter.replaceOp(op, buf);
      return success();
    }

    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    auto lmBuf = allocate(lmPtrTy, llvmElemTy, i32_val(allocNumElems));

    // For TLE, we store the base pointer as the lowered result.
    // The memdesc type will be converted to a struct containing the base ptr.
    rewriter.replaceOp(op, lmBuf);
    return success();
  }
};

/// Lower ttg::LocalLoadOp: load all elements from LM buffer into a tensor.
struct XPUTLELocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>,
      public LoadStoreConversionBase {
  XPUTLELocalLoadOpConversion(LLVMTypeConverter &converter,
                              const xpu::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    auto resTy = op.getResult().getType();
    Type llvmResultTy = typeConverter->convertType(resTy);
    Type elemTy = cast<RankedTensorType>(resTy).getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);
    unsigned numElems = getTotalElemsPerThread(resTy);

    // src is the lowered memdesc, i.e. a pointer into the buffer's own space:
    // LM (0) for `#ttg.shared_memory`, SM (2) for `#triton_xpu.smem`.
    Value lmBase = adaptor.getSrc();
    auto lmPtrTy = LLVM::LLVMPointerType::get(
        ctx, triton::xpu::getMemDescAddrSpace(op.getSrc().getType()));

    SmallVector<Value> loadedVals;
    for (unsigned i = 0; i < numElems; ++i) {
      Value elemPtr = gep(lmPtrTy, llvmElemTy, lmBase, i32_val(i));
      Value val = load(llvmElemTy, elemPtr);
      loadedVals.push_back(val);
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, loadedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower ttg::LocalStoreOp: store tensor elements into LM buffer.
struct XPUTLELocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>,
      public LoadStoreConversionBase {
  XPUTLELocalStoreOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value srcVal = adaptor.getSrc();
    Value dstBuf = adaptor.getDst();
    // Writing a cluster-shared buffer is not supported: `dstBuf` would be the
    // dead ptr<0> placeholder from the alloc SM branch (see
    // XPUTLELocalAllocOpConversion), i.e. a wild store. SM buffers are
    // stage-once read-only caches.
    assert(!isTLESmemBuffer(op.getDst()) &&
           "cannot store into a scope=smem TLE buffer");

    auto srcTy = op.getSrc().getType();
    Type elemTy = cast<RankedTensorType>(srcTy).getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);
    unsigned numElems = getTotalElemsPerThread(srcTy);

    auto lmPtrTy = LLVM::LLVMPointerType::get(
        ctx, triton::xpu::getMemDescAddrSpace(op.getDst().getType()));
    auto srcVals = unpackLLElements(loc, srcVal, rewriter);

    for (unsigned i = 0; i < numElems; ++i) {
      Value elemPtr = gep(lmPtrTy, llvmElemTy, dstBuf, i32_val(i));
      store(srcVals[i], elemPtr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// One contiguous GM<->LM DMA segment for a single core.
//   gmElemOffset : element offset from the GM tile base pointer.
//   lmElemOffset : compile-time element offset into the core's private LM
//   buffer
//                  (direction-agnostic: it is the dst offset for G2L and the
//                  src offset for L2G; 0 for the single-slice case, k*innermost
//                  for the k-th row of a whole-row-multi copy).
//   validCount   : i32 number of valid elements to transfer (may be 0 to skip
//   an
//                  out-of-bounds row/slice).
struct TileDmaSeg {
  Value gmElemOffset;
  unsigned lmElemOffset;
  Value validCount;
};

// Compile-time delinearize: map a linear index to multi-dim coords over
// `sizes`, iterating dims in `order` with order[0] the fastest-varying (matches
// mlir::delinearize used by emitOffsetForClusterLayout).
static SmallVector<unsigned> delinearizeConst(unsigned linear,
                                              ArrayRef<unsigned> sizes,
                                              ArrayRef<unsigned> order) {
  unsigned rank = sizes.size();
  SmallVector<unsigned> coord(rank, 0);
  for (unsigned orderIdx = 0; orderIdx < rank && orderIdx < order.size();
       ++orderIdx) {
    unsigned d = order[orderIdx];
    unsigned dimSize = (d < rank && sizes[d]) ? sizes[d] : 1;
    coord[d] = linear % dimSize;
    linear /= dimSize;
  }
  return coord;
}

// Runtime delinearize of an i32 Value over `sizes` by `order` (order[0]
// fastest).
static SmallVector<Value> delinearizeRT(ConversionPatternRewriter &rewriter,
                                        Location loc, Value linear,
                                        ArrayRef<unsigned> sizes,
                                        ArrayRef<unsigned> order) {
  unsigned rank = sizes.size();
  SmallVector<Value> coord(rank);
  for (unsigned d = 0; d < rank; ++d)
    coord[d] = i32_val(0);
  Value cur = linear;
  for (unsigned orderIdx = 0; orderIdx < rank && orderIdx < order.size();
       ++orderIdx) {
    unsigned d = order[orderIdx];
    unsigned dimSize = (d < rank && sizes[d]) ? sizes[d] : 1;
    coord[d] = urem(cur, i32_val(dimSize));
    cur = udiv(cur, i32_val(dimSize));
  }
  return coord;
}
// Compile-time enumeration of the LM slots a core owns and the per-slot
// within-tile coordinates, derived PURELY from the ClusterLayout and buffer
// shape (no rewriter / runtime values -> unit-testable in isolation). For slot
// n: coords[n][d] = tileId[d]*shapePerCTATile[d] + elemCoord[d], i.e. the
// nano-tile origin plus the intra-nano-tile element offset -- the SAME space as
// emitOffsetForClusterLayout (Utility.cpp).
static SmallVector<SmallVector<unsigned>>
enumerateSlotCoords(triton::xpu::ClusterLayoutAttr layout,
                    ArrayRef<int64_t> bufShape) {
  unsigned rank = bufShape.size();
  auto sizePerCore = layout.getSizePerCore();
  auto coresPerGroup = layout.getCoresPerGroup();
  auto groupsPerCluster = layout.getGroupsPerCluster();
  auto order = layout.getOrder();

  SmallVector<unsigned> shapePerCTATile(rank), tilesPerDim(rank);
  unsigned totalSizePerThread = 1, numElems = 1;
  for (unsigned d = 0; d < rank; ++d) {
    shapePerCTATile[d] =
        sizePerCore[d] * coresPerGroup[d] * groupsPerCluster[d];
    tilesPerDim[d] =
        shapePerCTATile[d]
            ? (bufShape[d] + shapePerCTATile[d] - 1) / shapePerCTATile[d]
            : 1;
    unsigned coresAlongDim = coresPerGroup[d] * groupsPerCluster[d];
    assert(coresAlongDim > 0 &&
           "cluster layout dim has zero cores (invalid ClusterLayout)");
    numElems *= (bufShape[d] + coresAlongDim - 1) / coresAlongDim;
    totalSizePerThread *= sizePerCore[d];
  }

  SmallVector<SmallVector<unsigned>> coords(numElems);
  for (unsigned n = 0; n < numElems; ++n) {
    unsigned nanoId = n / totalSizePerThread;
    unsigned elemId = n % totalSizePerThread;
    SmallVector<unsigned> tileId = delinearizeConst(nanoId, tilesPerDim, order);
    SmallVector<unsigned> elemCoord =
        delinearizeConst(elemId, sizePerCore, order);
    coords[n].resize(rank);
    for (unsigned d = 0; d < rank; ++d)
      coords[n][d] = tileId[d] * shapePerCTATile[d] + elemCoord[d];
  }
  return coords;
}

// Coalesce consecutive owned slots that map to a contiguous GM run (identical
// within-tile coords in every dim except the innermost, which increments by 1)
// into single DMA segments. `threadBase64` + `coordRT` describe this core's
// runtime base; `coords` are the compile-time within-tile slot coordinates from
// enumerateSlotCoords. Each segment's lmElemOffset is its start slot n (the LM
// buffer is filled in slot-enumeration order).
static SmallVector<TileDmaSeg>
coalesceRuns(ConversionPatternRewriter &rewriter, Location loc,
             ArrayRef<SmallVector<unsigned>> coords, unsigned rank,
             Value tileOriginElem64, Value threadBase64,
             ArrayRef<Value> coordRT, ArrayRef<Value> descStrides,
             ValueRange offsets, ValueRange realShapes, bool hasRealShape) {
  SmallVector<TileDmaSeg> segs;
  Value zeroI32 = i32_val(0);
  unsigned numElems = coords.size();
  unsigned last = rank - 1;
  unsigned n = 0;
  while (n < numElems) {
    unsigned runStart = n;
    unsigned runLen = 1;
    while (n + runLen < numElems) {
      bool contiguous = true;
      for (unsigned d = 0; d < rank; ++d) {
        unsigned expect = coords[runStart][d] + (d == last ? runLen : 0);
        if (coords[n + runLen][d] != expect) {
          contiguous = false;
          break;
        }
      }
      if (!contiguous)
        break;
      ++runLen;
    }
    // GM element offset of the run start.
    Value gmRun = add(tileOriginElem64, threadBase64);
    for (unsigned d = 0; d < rank; ++d) {
      Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
      gmRun = add(gmRun, mul(i64_val((int64_t)coords[runStart][d]), stride));
    }
    // Valid element count: clamp against realShapes on the innermost dim; zero
    // the whole run if any outer coord is out of the real shape. Global coord
    // along dim d = offset[d] + coordRT[d] + coords[runStart][d].
    Value validCount = i32_val(runLen);
    if (hasRealShape) {
      Value lastOff = (last < offsets.size()) ? offsets[last] : zeroI32;
      Value gLastStart =
          add(add(lastOff, coordRT[last]), i32_val(coords[runStart][last]));
      validCount =
          smin(validCount, smax(sub(realShapes[last], gLastStart), zeroI32));
      for (unsigned d = 0; d + 1 < rank; ++d) {
        Value dOff = (d < offsets.size()) ? offsets[d] : zeroI32;
        Value gCoord = add(add(dOff, coordRT[d]), i32_val(coords[runStart][d]));
        validCount =
            select(icmp_slt(gCoord, realShapes[d]), validCount, zeroI32);
      }
    }
    segs.push_back({gmRun, runStart, validCount});
    n += runLen;
  }
  return segs;
}

// LargeN 1D output writeback (rank==1, groupsPerCluster[0] > 1 AND
// coresPerGroup[0] > 1): the reduce result [M] is distributed cyclically over
// `numGroups` groups (group grp owns rows grp, grp+numGroups, ...). All
// `groupSize` col-cores of a group hold the same row-sums after the cross-core
// reduce, so only col-core 0 of each group (idInGroup == 0) writes them back to
// avoid redundant DMAs. Slot k of the private buffer maps to global row
// grp + k*numGroups.
static SmallVector<TileDmaSeg> planLargeNOutputSegments(
    ConversionPatternRewriter &rewriter, Location loc,
    triton::xpu::ClusterLayoutAttr layout, Value tileOriginElem64,
    ArrayRef<Value> descStrides, ValueRange offsets, ValueRange realShapes,
    bool hasRealShape, Value idInGroup, Value groupId, unsigned numGroups) {
  SmallVector<TileDmaSeg> segs;
  Value zeroI32 = i32_val(0);
  unsigned rowsPerGroup = layout.getSizePerCore()[0]; // == ceil(M, numGroups)
  Value writeGate = icmp_eq(idInGroup, zeroI32);
  Value rowOffsetBase = (offsets.size() > 0) ? offsets[0] : zeroI32;
  for (unsigned k = 0; k < rowsPerGroup; ++k) {
    Value globalRow = add(groupId, i32_val(k * numGroups));
    Value gmElemOffset =
        add(tileOriginElem64, mul(sext(i64_ty, globalRow), descStrides[0]));
    Value validCount = i32_val(1);
    if (hasRealShape) {
      Value rowIdx = add(rowOffsetBase, globalRow);
      validCount = select(icmp_slt(rowIdx, realShapes[0]), validCount, zeroI32);
    }
    validCount = select(writeGate, validCount, zeroI32);
    segs.push_back({gmElemOffset, k, validCount});
  }
  return segs;
}

// Layout-driven per-core DMA planning: the ClusterLayout is the SINGLE source
// of truth for which global (row,col) each LM slot n holds, mirroring
// emitOffsetForClusterLayout (Utility.cpp) so memory / elementwise / reduce
// agree. Returns coalesced contiguous GM runs. `coreId` is the FULL physical
// core id (tid), split into idInGroup (%groupSize) + groupId. Delegates to:
//   * planLargeNOutputSegments  -- LargeN 1D cyclic result writeback;
//   * enumerateSlotCoords       -- pure compile-time owned-slot coordinates;
//   * coalesceRuns              -- merge slots into contiguous GM DMA runs.
static SmallVector<TileDmaSeg>
planTileSegmentsFromLayout(ConversionPatternRewriter &rewriter, Location loc,
                           triton::xpu::ClusterLayoutAttr layout,
                           ArrayRef<int64_t> bufShape, Value tileOriginElem64,
                           ArrayRef<Value> descStrides, ValueRange offsets,
                           ValueRange realShapes, Value coreId) {
  unsigned rank = bufShape.size();
  bool hasRealShape = (!realShapes.empty() && realShapes.size() == rank);
  auto sizePerCore = layout.getSizePerCore();
  auto coresPerGroup = layout.getCoresPerGroup();
  auto groupsPerCluster = layout.getGroupsPerCluster();
  auto order = layout.getOrder();
  unsigned groupSize = 1, numGroups = 1;
  for (unsigned d = 0; d < rank; ++d) {
    groupSize *= coresPerGroup[d];
    numGroups *= groupsPerCluster[d];
  }
  Value idInGroup = urem(coreId, i32_val(groupSize));
  Value groupId = udiv(coreId, i32_val(groupSize));

  // LargeN output special-case (see planLargeNOutputSegments). The
  // coresPerGroup[0] > 1 guard distinguishes the g > 1 (LargeN) case from the
  // g == 1 (RowTiled) 1D output whose layout is cpg=[1]/gpc=[coreNum]: there
  // groupsPerCluster[0] > 1 but coresPerGroup[0] == 1, so it falls through to
  // the general BLOCK branch below (which is what RowTiled needs; cyclic would
  // be wrong for it).
  //
  // The SECOND disjunct covers m == 1 (a single row split across ALL cores,
  // ngroup == 1): the 1D result there is REPLICATED across the groupSize
  // col-cores (every core's slot 0 maps the same logical row 0), but its
  // layout has groupsPerCluster[0] == 1, so without this it fell into the
  // general BLOCK branch -- which treats each core's base as EXCLUSIVE
  // (origin + coreId * spc) and made every core write its own GM row. With
  // one program per row (the fused group-norm's [1, WT] configs) the 64
  // cores of program p then wrote rows [p, p+64) and the programs overwrote
  // each other -- measured as rows 4..7 of an 8-program run all holding
  // program 3's value, while the in-cluster y data stayed correct. The
  // cyclic formula itself is exact at ngroup == 1 (globalRow = groupId + k),
  // so route the replicated shape here: cpg > 1 and the cores along the
  // axis outnumber the rows (spc * cpg > len) means replicas, not owners.
  if (rank == 1 && coresPerGroup[0] > 1 &&
      (groupsPerCluster[0] > 1 ||
       sizePerCore[0] * coresPerGroup[0] > static_cast<unsigned>(bufShape[0])))
    return planLargeNOutputSegments(
        rewriter, loc, layout, tileOriginElem64, descStrides, offsets,
        realShapes, hasRealShape, idInGroup, groupId, numGroups);

  // ---- General branch: RowTiled / LargeN / 1D / default-1D. ----
  // Runtime per-core coords: coreCoordInGroup[d] = intra-group core coord
  // (idInGroup over coresPerGroup), groupCoord[d] = group coord (groupId over
  // groupsPerCluster), both delinearized by `order`. The per-core thread base
  // coordinate along dim d is
  //   coordRT[d] = groupCoord[d]*(sizePerCore[d]*coresPerGroup[d])
  //                + coreCoordInGroup[d]*sizePerCore[d].
  SmallVector<Value> coreCoordInGroup =
      delinearizeRT(rewriter, loc, idInGroup, coresPerGroup, order);
  SmallVector<Value> groupCoord =
      delinearizeRT(rewriter, loc, groupId, groupsPerCluster, order);
  SmallVector<Value> coordRT(rank);
  Value threadBase64 = i64_val(0);
  for (unsigned d = 0; d < rank; ++d) {
    coordRT[d] =
        add(mul(groupCoord[d], i32_val(sizePerCore[d] * coresPerGroup[d])),
            mul(coreCoordInGroup[d], i32_val(sizePerCore[d])));
    Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
    threadBase64 = add(threadBase64, mul(sext(i64_ty, coordRT[d]), stride));
  }

  // Enumerate owned slots (pure compile-time) then coalesce into GM runs.
  SmallVector<SmallVector<unsigned>> coords =
      enumerateSlotCoords(layout, bufShape);
  if (coords.empty())
    return {};
  return coalesceRuns(rewriter, loc, coords, rank, tileOriginElem64,
                      threadBase64, coordRT, descStrides, offsets, realShapes,
                      hasRealShape);
}

// ---- Legacy flat row-major cut (used when NO xpu.tile_layout is stamped).
// ---- Core c owns the flattened element range [c*epc, (c+1)*epc) of the tile.
// This is the pre-CoreTiling behavior, kept verbatim for TLE kernels that do
// not run tritonxpu-tle-core-tiling.
static SmallVector<TileDmaSeg>
planTileSegmentsLegacy(ConversionPatternRewriter &rewriter, Location loc,
                       ArrayRef<int64_t> bufShape, ValueRange offsets,
                       ValueRange realShapes, ArrayRef<Value> coreCoord,
                       Value coreBase, Value gmElemOffset,
                       unsigned elemsPerCore, unsigned tileElems) {
  unsigned rank = bufShape.size();
  bool hasRealShape = (!realShapes.empty() && realShapes.size() == rank);
  Value zeroI32 = i32_val(0);
  SmallVector<TileDmaSeg> segs;

  // This planner emits ONE contiguous per-core slice, so it is only correct
  // while that slice stays inside a single innermost row. Without a stamped
  // xpu.tile_layout the tile keeps the default column-distributed layout, so a
  // core never owns more than one whole innermost row here (XBLOCK=1 is the
  // only correct untiled 2D use); the former multi-whole-row branch was dead
  // and has been removed. Guard the invariant instead of trusting it silently:
  // a config that violates it would otherwise emit a slice spanning a GM row
  // boundary, i.e. silently wrong data. Fail loudly so it is caught at compile
  // time.
  assert((rank < 2 || bufShape[rank - 1] <= 0 ||
          elemsPerCore <= static_cast<unsigned>(bufShape[rank - 1])) &&
         "legacy (untiled) TLE tile planner requires the per-core slice to fit "
         "in one innermost row; got elemsPerCore > innermost, which needs one "
         "DMA per row (stamp an xpu.tile_layout and use "
         "planTileSegmentsFromLayout instead)");
  Value remainElems = smax(sub(i32_val(tileElems), coreBase), zeroI32);
  Value validCount = smin(remainElems, i32_val(elemsPerCore));
  if (hasRealShape) {
    unsigned last = rank - 1;
    Value colOff = (last < offsets.size()) ? offsets[last] : zeroI32;
    Value gColStart = add(colOff, coreCoord[last]);
    Value remCols = smax(sub(realShapes[last], gColStart), zeroI32);
    validCount = smin(validCount, remCols);
    for (unsigned d = 0; d + 1 < rank; ++d) {
      Value dOff = (d < offsets.size()) ? offsets[d] : zeroI32;
      Value gCoord = add(dOff, coreCoord[d]);
      Value inBound = icmp_slt(gCoord, realShapes[d]);
      validCount = select(inBound, validCount, zeroI32);
    }
  }
  segs.push_back({gmElemOffset, 0, validCount});
  return segs;
}

/// The mfence mask a TLE DMA needs: name the bit of the LM/SM side of the
/// transfer and nothing else -- never the GM bit. Measured in all four
/// directions (design doc §4.5): GM->LM 1, GM->SM 2, LM->GM 1, SM->GM 2.
/// Masks are NOT monotonic in their bits, so a wider mask is not a safe
/// substitute: mask 5 (LM|GM) drains GM2LM but does NOT drain GM2SM, and
/// mask 6 (SM|GM) needlessly drains the LM channel too.
static int32_t tleFenceMask(bool isSM) { return isSM ? 2 : 1; }

/// Lower triton_xpu.tle_copy_g2l to GM2LM DMA instruction.
struct XPUTLECopyG2LOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLECopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  XPUTLECopyG2LOpConversion(LLVMTypeConverter &converter,
                            const xpu::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLECopyGlobalToLocalOp>(converter,
                                                                    benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLECopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // --- SM (cluster-shared) cache branch -----------------------------------
    // If the destination buffer was allocated with scope=smem, the WHOLE array
    // is staged into global_smem once per cluster (the DMA is partitioned
    // across all cores, see emitPartitionedGM2SM) and shared by every core.
    // This skips the per-core tile-segment planning used for LM entirely.
    //
    // LIMITATION -- no tail clamp: exactly product(buffer_shape) elements are
    // read from GM, so `adaptor.getShapes()` (the descriptor's REAL extent,
    // used by the LM path below to clamp row/col tails) is deliberately unused
    // here. The caller must therefore pad the GM weight array up to the smem
    // buffer width (see test_layernorm_sm_stageonce.py's R_PAD), otherwise the
    // staging DMA reads past the tensor. Acceptable because scope=smem targets
    // loop-invariant weight vectors whose padding is free; a real clamp would
    // need per-dimension segment planning like planTileSegmentsFromLayout.
    if (Operation *allocOp = getTLESmemAlloc(op.getDstBuffer())) {
      auto memDescTy =
          cast<triton::gpu::MemDescType>(op.getDstBuffer().getType());
      auto bufShape = memDescTy.getShape();
      unsigned elemBytes =
          memDescTy.getElementType().getIntOrFloatBitWidth() / 8u;
      unsigned tileElems = 1;
      for (auto dim : bufShape)
        tileElems *= dim;

      // GM source base (desc lowers to ptr<1>) + tile-origin byte offset from
      // offsets . strides.
      Value gmBasePtr = adaptor.getDesc();
      SmallVector<Value> descStrides = getDescStridesOrRowMajor(
          loc, rewriter, adaptor.getStrides(), bufShape);
      auto offsets = adaptor.getOffsets();
      Value tileOriginElem64 = i64_val(0);
      for (unsigned i = 0; i < offsets.size() && i < descStrides.size(); ++i)
        tileOriginElem64 = add(tileOriginElem64,
                               mul(sext(i64_ty, offsets[i]), descStrides[i]));
      Value gmByteOff = mul(tileOriginElem64, i64_val(elemBytes));
      Value srcPtr = gep(ptr_ty(ctx, 1), i8_ty, gmBasePtr, gmByteOff);

      emitPartitionedGM2SM(loc, rewriter, ctx, op, srcPtr,
                           getTLESmemBase(loc, rewriter, op, allocOp),
                           i32_val((int32_t)tileElems),
                           i32_val((int32_t)elemBytes));

      rewriter.eraseOp(op);
      return success();
    }

    // Get the descriptor (TensorDescType is lowered to ptr<1> = GM base
    // pointer)
    Value desc = adaptor.getDesc();
    // Get dst buffer (LM base pointer)
    Value dstBuf = adaptor.getDstBuffer();
    // Get offsets
    auto offsets = adaptor.getOffsets();

    // desc is already a GM pointer (ptr<1>), no extraction needed.
    Value gmBasePtr = desc;

    // Strides are passed directly as op operands (desc.strides from Python),
    // with a row-major fallback when the descriptor carried none.
    SmallVector<Value> descStrides = getDescStridesOrRowMajor(
        loc, rewriter, adaptor.getStrides(),
        cast<triton::gpu::MemDescType>(op.getDstBuffer().getType()).getShape());

    // Get the memdesc type to know element type and shape. The buffer's space
    // decides the DMA flavour: `#ttg.shared_memory` is per-core LM, while only
    // `#triton_xpu.smem` is the cluster-shared block (see getMemDescAddrSpace).
    auto memDescTy = op.getDstBuffer().getType();
    auto bufShape = cast<triton::gpu::MemDescType>(memDescTy).getShape();
    Type elemTy = cast<triton::gpu::MemDescType>(memDescTy).getElementType();
    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;
    bool toShared = triton::xpu::isSharedMemDesc(memDescTy);
    unsigned dstSpace = triton::xpu::getMemDescAddrSpace(memDescTy);

    // Support arbitrary rank tensors. The memdesc shape is the WHOLE tile.
    unsigned rank = bufShape.size();
    unsigned tileElems = 1;
    for (unsigned d = 0; d < rank; ++d)
      tileElems *= bufShape[d];

    // --- SPMD per-core partition ---
    // The tile is split contiguously across `groupSize` cores. Core `c` owns
    // the flattened element range [c*elemsPerCore, (c+1)*elemsPerCore) and
    // copies it into its private LM buffer starting at local index 0.
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned groupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (groupSize == 0)
      groupSize = 1;
    unsigned elemsPerCore = (tileElems + groupSize - 1) / groupSize;

    // Tile-origin element offset in GM: sum(offset[i] * stride[i]).
    Value tileOriginElem64 = i64_val(0);
    for (unsigned i = 0; i < offsets.size() && i < descStrides.size(); ++i) {
      Value off_64 = sext(i64_ty, offsets[i]);
      tileOriginElem64 = add(tileOriginElem64, mul(off_64, descStrides[i]));
    }

    // Per-core flattened start index within the tile.
    Value coreId = tid_val();
    Value idInsideGroup = srem(coreId, i32_val(groupSize));
    Value coreBase = mul(idInsideGroup, i32_val(elemsPerCore));

    // Decompose coreBase into multi-dim coords (row-major over bufShape) and
    // compute the GM element offset of the core's first element.
    Value gmElemOffset = tileOriginElem64;
    Value rem = coreBase;
    SmallVector<Value> coreCoord(rank);
    for (unsigned d = 0; d < rank; ++d) {
      unsigned innerProd = 1;
      for (unsigned j = d + 1; j < rank; ++j)
        innerProd *= bufShape[j];
      Value coord = (innerProd > 1) ? udiv(rem, i32_val(innerProd)) : rem;
      if (d + 1 < rank)
        rem = urem(rem, i32_val(innerProd));
      coreCoord[d] = coord;
      Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
      gmElemOffset = add(gmElemOffset, mul(sext(i64_ty, coord), stride));
    }
    Value zeroI32 = i32_val(0);
    Value dstPtr = bitcast(dstBuf, ptr_ty(ctx, dstSpace));
    auto shapesG2L = adaptor.getShapes();

    // Plan the per-core DMA segments. When tritonxpu-tle-core-tiling stamped a
    // ClusterLayout on this op ("xpu.tile_layout"), the partition is fully
    // layout-driven; otherwise fall back to the legacy flat row-major cut.
    auto tileLayout =
        op->getAttrOfType<triton::xpu::ClusterLayoutAttr>("xpu.tile_layout");
    SmallVector<TileDmaSeg> segs =
        tileLayout
            ? planTileSegmentsFromLayout(rewriter, loc, tileLayout, bufShape,
                                         tileOriginElem64, descStrides, offsets,
                                         shapesG2L, coreId)
            : planTileSegmentsLegacy(rewriter, loc, bufShape, offsets,
                                     shapesG2L, coreCoord, coreBase,
                                     gmElemOffset, elemsPerCore, tileElems);

    for (auto &seg : segs) {
      Value gmByteOff = mul(seg.gmElemOffset, i64_val(elemBytes));
      Value gmAddr = gep(ptr_ty(ctx, 1), i8_ty, gmBasePtr, gmByteOff);
      Value srcPtr = bitcast(gmAddr, ptr_ty(ctx, 1));
      Value dstPtrSeg = seg.lmElemOffset == 0
                            ? dstPtr
                            : gep(ptr_ty(ctx, dstSpace), elemTy, dstPtr,
                                  i32_val(seg.lmElemOffset));
      Value copyBytes = mul(seg.validCount, i32_val(elemBytes));
      if (toShared)
        createGM2SMOp(rewriter, ctx, loc, srcPtr, dstPtrSeg, zeroI32,
                      copyBytes);
      else
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtrSeg, zeroI32,
                      copyBytes);
    }

    bool isSync = op.getIsSync();
    if (isSync)
      createMfenceOp(rewriter, loc, tleFenceMask(toShared));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_copy_l2g to LM2GM DMA instruction.
struct XPUTLECopyL2GOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLECopyLocalToGlobalOp>,
      public LoadStoreConversionBase {
  XPUTLECopyL2GOpConversion(LLVMTypeConverter &converter,
                            const xpu::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLECopyLocalToGlobalOp>(converter,
                                                                    benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLECopyLocalToGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value desc = adaptor.getDesc();
    Value srcBuf = adaptor.getSrcBuffer();
    auto offsets = adaptor.getOffsets();

    // --- SM (cluster-shared) writeback branch -------------------------------
    // scope=smem source: the buffer is ONE cluster-wide copy of the whole tile
    // (`srcBuf` is only the dead ptr<0> placeholder from the alloc SM branch,
    // so the ptr<2> base is recomputed here like every other smem consumer
    // does). There is no per-core ownership to honour, which is the point: this
    // is the route for a result row too narrow to partition across cores
    // without every core landing in the same GM cache line.
    if (Operation *allocOp = getTLESmemAlloc(op.getSrcBuffer())) {
      auto smMemDescTy =
          cast<triton::gpu::MemDescType>(op.getSrcBuffer().getType());
      auto smBufShape = smMemDescTy.getShape();
      unsigned smElemBytes =
          smMemDescTy.getElementType().getIntOrFloatBitWidth() / 8u;
      unsigned smTileElems = 1;
      for (auto dim : smBufShape)
        smTileElems *= dim;

      // Tile-origin byte offset in GM: sum(offset[i] * stride[i]), exactly as
      // the g2l SM branch computes its source.
      SmallVector<Value> smStrides = getDescStridesOrRowMajor(
          loc, rewriter, adaptor.getStrides(), smBufShape);
      Value smOriginElem64 = i64_val(0);
      for (unsigned i = 0; i < offsets.size() && i < smStrides.size(); ++i)
        smOriginElem64 =
            add(smOriginElem64, mul(sext(i64_ty, offsets[i]), smStrides[i]));
      Value smOriginByte64 = mul(smOriginElem64, i64_val(smElemBytes));
      Value gmDstBase = gep(ptr_ty(ctx, 1), i8_ty, desc, smOriginByte64);

      // Like the g2l SM branch this writes exactly product(buffer_shape)
      // elements with no tail clamp, so a descriptor whose real extent is
      // shorter than the buffer would write past the tensor. Callers of
      // scope=smem writeback must size the buffer to the extent they mean.
      emitCoalescedSM2GM(loc, rewriter, ctx, op,
                         getTLESmemBase(loc, rewriter, op, allocOp), gmDstBase,
                         smTileElems * smElemBytes, op.getIsSync());
      rewriter.eraseOp(op);
      return success();
    }

    // desc is a GM pointer (ptr<1>)
    Value gmBasePtr = desc;

    // Strides are passed directly as op operands (desc.strides from Python),
    // with a row-major fallback when the descriptor carried none.
    SmallVector<Value> descStrides = getDescStridesOrRowMajor(
        loc, rewriter, adaptor.getStrides(),
        cast<triton::gpu::MemDescType>(op.getSrcBuffer().getType()).getShape());

    // Get buffer shape/type info. The space decides the DMA flavour: per-core
    // LM vs cluster-shared SM (see getMemDescAddrSpace).
    auto memDescTy = op.getSrcBuffer().getType();
    auto bufShape = cast<triton::gpu::MemDescType>(memDescTy).getShape();
    Type elemTy = cast<triton::gpu::MemDescType>(memDescTy).getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    bool fromShared = triton::xpu::isSharedMemDesc(memDescTy);
    unsigned srcSpace = triton::xpu::getMemDescAddrSpace(memDescTy);

    // Support arbitrary rank tensors. The memdesc shape is the WHOLE tile.
    unsigned rank = bufShape.size();
    unsigned tileElems = 1;
    for (unsigned d = 0; d < rank; ++d)
      tileElems *= bufShape[d];

    // --- SPMD per-core partition ---
    // Core `c` writes back only the flattened element range
    // [c*elemsPerCore, (c+1)*elemsPerCore) from its private LM buffer to GM.
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned groupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (groupSize == 0)
      groupSize = 1;
    unsigned elemsPerCore = (tileElems + groupSize - 1) / groupSize;

    // Tile-origin element offset in GM: sum(offset[i] * stride[i]).
    Value tileOriginElem64 = i64_val(0);
    for (unsigned i = 0; i < offsets.size() && i < descStrides.size(); ++i) {
      Value off_64 = sext(i64_ty, offsets[i]);
      tileOriginElem64 = add(tileOriginElem64, mul(off_64, descStrides[i]));
    }

    // Per-core flattened start index within the tile.
    Value coreId = tid_val();
    Value idInsideGroup = srem(coreId, i32_val(groupSize));
    Value coreBase = mul(idInsideGroup, i32_val(elemsPerCore));

    // Decompose coreBase into multi-dim coords (row-major over bufShape) and
    // compute the GM element offset of the core's first element.
    Value gmElemOffset = tileOriginElem64;
    Value rem = coreBase;
    SmallVector<Value> coreCoord(rank);
    for (unsigned d = 0; d < rank; ++d) {
      unsigned innerProd = 1;
      for (unsigned j = d + 1; j < rank; ++j)
        innerProd *= bufShape[j];
      Value coord = (innerProd > 1) ? udiv(rem, i32_val(innerProd)) : rem;
      if (d + 1 < rank)
        rem = urem(rem, i32_val(innerProd));
      coreCoord[d] = coord;
      Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
      gmElemOffset = add(gmElemOffset, mul(sext(i64_ty, coord), stride));
    }
    Value zeroI32 = i32_val(0);
    Value srcLmPtr = bitcast(srcBuf, ptr_ty(ctx, srcSpace));
    auto shapesL2G = adaptor.getShapes();

    // Plan the per-core DMA using the SAME layout-driven dispatcher as copy_g2l
    // (segments are direction-agnostic: {gmElemOffset, lmElemOffset,
    // validCount}). When absent, the legacy flat cut applies (the reduce OUTPUT
    // buffer is 1D [XBLOCK], so the layout path enumerates one contiguous
    // per-core slice too).
    auto tileLayout =
        op->getAttrOfType<triton::xpu::ClusterLayoutAttr>("xpu.tile_layout");
    SmallVector<TileDmaSeg> segs =
        tileLayout
            ? planTileSegmentsFromLayout(rewriter, loc, tileLayout, bufShape,
                                         tileOriginElem64, descStrides, offsets,
                                         shapesL2G, coreId)
            : planTileSegmentsLegacy(rewriter, loc, bufShape, offsets,
                                     shapesL2G, coreCoord, coreBase,
                                     gmElemOffset, elemsPerCore, tileElems);

    // Ensure preceding CPU stores to the buffer are visible to the DMA engine.
    createMfenceOp(rewriter, loc, tleFenceMask(fromShared));
    for (auto &seg : segs) {
      Value gmByteOff = mul(seg.gmElemOffset, i64_val(elemBytes));
      Value gmAddr = gep(ptr_ty(ctx, 1), i8_ty, gmBasePtr, gmByteOff);
      Value dstPtr = bitcast(gmAddr, ptr_ty(ctx, 1));
      Value srcPtrSeg = seg.lmElemOffset == 0
                            ? srcLmPtr
                            : gep(ptr_ty(ctx, srcSpace), elemTy, srcLmPtr,
                                  i32_val(seg.lmElemOffset));
      Value copyBytesVal = mul(seg.validCount, i32_val(elemBytes));
      if (fromShared)
        createSM2GMOp(rewriter, ctx, loc, srcPtrSeg, dstPtr, zeroI32,
                      copyBytesVal);
      else
        createLM2GMOp(rewriter, ctx, loc, srcPtrSeg, dstPtr, zeroI32,
                      copyBytesVal);
    }
    // Completion fence. Dropped for is_sync=false so the write-back overlaps
    // with whatever follows; the caller owes a tle_dma_wait before it reuses
    // the source buffer.
    if (op.getIsSync())
      createMfenceOp(rewriter, loc, tleFenceMask(fromShared));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_dma_wait to a single mfence. The mask attribute selects
/// which memory classes to drain (bit0=LM, bit1=SM, bit2=GM), matching the
/// masks createMfenceOp already uses elsewhere in this file.
struct XPUTLEDmaWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLEDmaWaitOp>,
      public LoadStoreConversionBase {
  XPUTLEDmaWaitOpConversion(LLVMTypeConverter &converter,
                            const xpu::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLEDmaWaitOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLEDmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    createMfenceOp(rewriter, loc, static_cast<int32_t>(op.getMask()));
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_normcopy_g2l: element-wise (permuted) gather from a GM
/// pointer tensor into a local memory buffer. Element i of this core's lane set
/// is copied to local slot i, matching the tle_local_ptr layout.
struct XPUTLENormCopyG2LOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLENormCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  XPUTLENormCopyG2LOpConversion(LLVMTypeConverter &converter,
                                const xpu::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLENormCopyGlobalToLocalOp>(
            converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLENormCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy =
        cast<triton::gpu::MemDescType>(op.getDstBuffer().getType());
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = getTypeConverter()->convertType(elemTy);
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

    // Per-core GM pointer lanes (same distribution as tle_local_ptr).
    auto gmPtrs = unpackLLElements(loc, adaptor.getSrcPtrs(), rewriter);
    bool toShared = triton::xpu::isSharedMemDesc(memDescTy);
    unsigned dstSpace = triton::xpu::getMemDescAddrSpace(memDescTy);
    Value lmBase = bitcast(adaptor.getDstBuffer(), ptr_ty(ctx, dstSpace));
    Value zeroI32 = i32_val(0);
    Value szI32 = i32_val(elemBytes);

    for (unsigned i = 0; i < gmPtrs.size(); ++i) {
      Value gmPtr = bitcast(gmPtrs[i], ptr_ty(ctx, 1));
      Value lmSlot = gep(ptr_ty(ctx, dstSpace), llvmElemTy, lmBase, i32_val(i));
      if (toShared)
        createGM2SMOp(rewriter, ctx, loc, gmPtr, lmSlot, zeroI32, szI32);
      else
        createGM2LMOp(rewriter, ctx, loc, gmPtr, lmSlot, zeroI32, szI32);
    }
    // is_sync=false means the caller (tritonxpu-tle-pipeline) drains this fill
    // itself, one iteration later, with a triton_xpu.tle_wait. Fencing here as
    // well would defeat the whole point of the prefetch.
    if (op.getIsSync())
      createMfenceOp(rewriter, loc, tleFenceMask(toShared));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_normcopy_l2g: element-wise (permuted) scatter from a
/// local memory buffer to a GM pointer tensor. Scatter counterpart of the
/// gather above.
struct XPUTLENormCopyL2GOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLENormCopyLocalToGlobalOp>,
      public LoadStoreConversionBase {
  XPUTLENormCopyL2GOpConversion(LLVMTypeConverter &converter,
                                const xpu::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLENormCopyLocalToGlobalOp>(
            converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLENormCopyLocalToGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy =
        cast<triton::gpu::MemDescType>(op.getSrcBuffer().getType());
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = getTypeConverter()->convertType(elemTy);
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

    auto gmPtrs = unpackLLElements(loc, adaptor.getDstPtrs(), rewriter);
    bool fromShared = triton::xpu::isSharedMemDesc(memDescTy);
    unsigned srcSpace = triton::xpu::getMemDescAddrSpace(memDescTy);
    Value lmBase = bitcast(adaptor.getSrcBuffer(), ptr_ty(ctx, srcSpace));
    Value zeroI32 = i32_val(0);
    Value szI32 = i32_val(elemBytes);

    // Ensure preceding CPU stores to the buffer are visible to the DMA engine.
    createMfenceOp(rewriter, loc, tleFenceMask(fromShared));
    for (unsigned i = 0; i < gmPtrs.size(); ++i) {
      Value lmSlot = gep(ptr_ty(ctx, srcSpace), llvmElemTy, lmBase, i32_val(i));
      Value gmPtr = bitcast(gmPtrs[i], ptr_ty(ctx, 1));
      if (fromShared)
        createSM2GMOp(rewriter, ctx, loc, lmSlot, gmPtr, zeroI32, szI32);
      else
        createLM2GMOp(rewriter, ctx, loc, lmSlot, gmPtr, zeroI32, szI32);
    }
    // Only the trailing fence is optional: the leading one above guards the
    // CPU stores that produced this buffer and must always be there.
    if (op.getIsSync())
      createMfenceOp(rewriter, loc, tleFenceMask(fromShared));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_wait: drain the DMA channels named by `mask`.
/// The mask is an mfence mask (bit0=LM, bit1=SM, bit2=GM), so this is a bulk
/// drain, not a counting wait -- which is exactly why tritonxpu-tle-pipeline
/// caps its depth at one outstanding transfer per memory space.
struct XPUTLEWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLEWaitOp>,
      public LoadStoreConversionBase {
  XPUTLEWaitOpConversion(LLVMTypeConverter &converter,
                         const xpu::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLEWaitOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLEWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    createMfenceOp(rewriter, loc, static_cast<int32_t>(op.getMask()));
    rewriter.eraseOp(op);
    return success();
  }
};

/// LM slot that element `i` of a sliced TLE access touches. Shared by
/// tle_local_ptr (slots are elements) and tle_vload / tle_vstore (slots are
/// whole vectors) so the two cannot drift apart on the same buffer.
///
/// A contiguous slice -- one row per core, and the unsliced whole-buffer access
/// -- is `loopIndex * perSlice + i`, a null `loopIndex` meaning slot 0. With
/// more than one row per core the slice is a column band instead, because
/// unroll control divides only the last dimension: element `i` sits at `(i /
/// sliceCols) * rowStride + loopIndex * sliceCols + i % sliceCols`. Both
/// constants come from `stampSliceGeometry`; their absence selects the
/// contiguous form.
static Value tleSlotIndex(Location loc, Operation *op, Value loopIndex,
                          unsigned perSlice, unsigned i,
                          ConversionPatternRewriter &rewriter) {
  auto sliceColsAttr = op->getAttrOfType<IntegerAttr>("xpu.slice_cols");
  auto rowStrideAttr = op->getAttrOfType<IntegerAttr>("xpu.row_stride");
  if (sliceColsAttr && rowStrideAttr) {
    unsigned sliceCols =
        std::max<unsigned>(sliceColsAttr.getValue().getZExtValue(), 1);
    unsigned rowStride = rowStrideAttr.getValue().getZExtValue();
    Value fixed = i32_val((i / sliceCols) * rowStride + i % sliceCols);
    if (!loopIndex)
      return fixed;
    return add(fixed, mul(loopIndex, i32_val(sliceCols)));
  }
  if (!loopIndex)
    return i32_val(i);
  return add(mul(loopIndex, i32_val(perSlice)), i32_val(i));
}

/// Buffer slot that element `i` of a sliced TLE access touches in a
/// CLUSTER-SHARED (scope=smem) buffer.
///
/// The LM form above is enough for a per-core buffer, where the staging DMA
/// already dropped this core's slice at the base. An smem buffer instead holds
/// the whole array once, so two things change:
///
///  * the core's own column block has to be added: cores tile the axis, each
///    owning `rowStride` columns (the PRE-slice per-core row), so the base is
///    `gCoord[axis] * rowStride`. Cores past the axis extent are replicas and
///    wrap, matching XPUTLETritonMakeRangeOpConversion's `% shape[axis]`.
///
///  * the ROW component of the LM form must be dropped. A weight read that a
///    broadcast collapsed has one row per tile row, and every row reads the
///    SAME columns; keeping `(i / sliceCols) * rowStride` would walk into the
///    next core's columns instead. This is what made 1024x4096 XB16 RI4 read
///    wrong data while every one-row-per-core config was fine.
///
/// Unsliced (no stamp) collapses to `gCoord * colsPerCore + i`.
static Value tleSmemSlotIndex(Location loc, Operation *op, Value loopIndex,
                              Value coreColBase, unsigned colsPerCore,
                              unsigned i, ConversionPatternRewriter &rewriter) {
  auto sliceColsAttr = op->getAttrOfType<IntegerAttr>("xpu.slice_cols");
  unsigned sliceCols =
      sliceColsAttr
          ? std::max<unsigned>(sliceColsAttr.getValue().getZExtValue(), 1)
          : std::max<unsigned>(colsPerCore, 1);
  Value slot = add(coreColBase, i32_val(i % sliceCols));
  if (loopIndex)
    slot = add(slot, mul(loopIndex, i32_val(sliceCols)));
  return slot;
}

/// Lower triton_xpu.tle_local_ptr: compute LM pointers from buffer + indices.
struct XPUTLELocalPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLELocalPtrOp>,
      public LoadStoreConversionBase {
  XPUTLELocalPtrOpConversion(LLVMTypeConverter &converter,
                             const xpu::TargetInfo &targetInfo,
                             ModuleAxisInfoAnalysis &axisAnalysisPass,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLELocalPtrOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLELocalPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // Get buffer base pointer (lowered from memdesc)
    Value lmBase = adaptor.getBuffer();

    // Get the result type to determine shape.
    // The buffer's own space decides the pointer space: `#ttg.shared_memory`
    // means per-core LM, which on XPU is the flat space (0) -- Triton's
    // addrspace 3 never reaches the target. Only `#triton_xpu.smem` is the
    // cluster-shared block (2).
    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    unsigned addrSpace =
        triton::xpu::getMemDescAddrSpace(op.getBuffer().getType());
    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, addrSpace);

    Type llvmResultTy = typeConverter->convertType(resTy);

    // Get buffer element type from the memdesc
    auto memDescTy = cast<triton::gpu::MemDescType>(op.getBuffer().getType());
    auto bufShape = memDescTy.getShape();
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);

    // Get index operands
    auto indices = adaptor.getIndices();
    unsigned rank = bufShape.size();

    // Calculate the number of result pointers (per-thread elements).
    // Since the result type is an unencoded pointer tensor, use the LLVM
    // struct type from the type converter (which computes proper numElems).
    auto llvmResTy = cast<LLVM::LLVMStructType>(llvmResultTy);
    unsigned numElems = llvmResTy.getBody().size();

    // --- SM (cluster-shared) cache branch -----------------------------------
    // A cluster-shared buffer holds the FULL array, one copy for all cores.
    // Unlike the per-core LM path, the index operands MUST be used: register k
    // reads global element `index[k]` from the shared copy. The index tensor
    // carries the ClusterLayout, so index[k] is exactly the logical element
    // this lane owns (see XPUTLETritonMakeRangeOpConversion).
    if (Operation *allocOp = getTLESmemAlloc(op.getBuffer())) {
      unsigned resAddrSpace = 0;
      if (auto ptrETy =
              dyn_cast<triton::PointerType>(resTensorTy.getElementType()))
        resAddrSpace = ptrETy.getAddressSpace();
      assert(
          resAddrSpace == 2 &&
          "local_ptr into a scope=smem buffer must have an addrspace-2 result "
          "pointer type (set by the TLE frontend)");
      (void)resAddrSpace;
      Value smByteBase = getTLESmemBase(loc, rewriter, op, allocOp);

      // Unpack each index tensor into per-register i32 values. Each index
      // tensor must carry the SAME ClusterLayout as the result pointer tensor,
      // i.e. exactly one index per pointer register -- TLECoreTiling stamps the
      // indices for smem buffers precisely to guarantee this. If it did not,
      // the loop below would read past the unpacked index vector.
      SmallVector<SmallVector<Value>> idxUnpacked;
      for (auto idx : indices)
        idxUnpacked.push_back(unpackLLElements(loc, idx, rewriter));
      // Normally one index per pointer. A VECTORIZED SM gather (xpu.sm_gather,
      // see XPUTLETritonLoadOpConversion) keeps the index SCALAR -- laneCount
      // indices per vector pointer -- while the result pointer tensor is
      // vectorized. Its consumer recomputes every lane address from the full
      // index and never dereferences these pointers, so emitting one
      // representative (lane-0) pointer per vector keeps this conversion valid.
      unsigned idxCount =
          idxUnpacked.empty() ? numElems : idxUnpacked[0].size();
      assert(numElems > 0 && idxCount % numElems == 0 &&
             "smem local_ptr index count must be a whole multiple of the "
             "pointer count (TLECoreTiling must encode the index tensors)");
      unsigned idxStride = numElems ? idxCount / numElems : 1;

      // Row-major strides over the buffer shape to linearize multi-dim indices.
      SmallVector<int64_t> bufStrides(rank, 1);
      for (int d = static_cast<int>(rank) - 2; d >= 0; --d)
        bufStrides[d] = bufStrides[d + 1] * bufShape[d + 1];

      auto smPtrTy = LLVM::LLVMPointerType::get(ctx, 2);
      SmallVector<Value> resultPtrs;
      for (unsigned k = 0; k < numElems; ++k) {
        Value lin = i32_val(0);
        for (unsigned d = 0; d < rank && d < idxUnpacked.size(); ++d) {
          Value iv = idxUnpacked[d][k * idxStride];
          Value contrib = (bufStrides[d] == 1)
                              ? iv
                              : mul(iv, i32_val((int32_t)bufStrides[d]));
          lin = add(lin, contrib);
        }
        resultPtrs.push_back(gep(smPtrTy, llvmElemTy, smByteBase, lin));
      }
      Value resultStruct = packLLElements(loc, typeConverter, resultPtrs,
                                          rewriter, llvmResultTy);
      rewriter.replaceOp(op, {resultStruct});
      return success();
    }

    // --- SPMD per-core partition ---
    // The LM buffer is per-core (elemsPerCore elements). make_range distributes
    // the tile so this core's i-th element is global tile position
    // coreBase + i, and CopyG2L placed that element at LM-local slot i.
    // Therefore the i-th pointer is simply lmBase + i (local index). The raw
    // global index operands are not needed here.
    //
    // That argument is about the whole tile. On a segment unroll control
    // sliced, `numElems` is the slice and this core's i-th slice element sits
    // at sliceBase + i, so the slice base is the one piece the index tensors
    // cannot stand in for -- it arrives through $loopIndex.
    (void)indices;
    (void)rank;
    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    // One pointer per VECTOR slot once tritonxpu-vectorize retyped this op (see
    // makeVectorTLELocalPtr): a slot is W buffer elements wide, which is not
    // the register width when the registers are f32 over a bf16 buffer.
    Type slotTy = llvmElemTy;
    if (auto ptrETy =
            dyn_cast<triton::PointerType>(resTensorTy.getElementType()))
      if (auto vecETy = dyn_cast<VectorType>(ptrETy.getPointeeType()))
        slotTy = VectorType::get(vecETy.getNumElements(), llvmElemTy);
    SmallVector<Value> resultPtrs;
    for (unsigned i = 0; i < numElems; ++i) {
      // `slotTy` fixes the width of one step, `tleSlotIndex` which step this
      // core and tile iteration takes; both are in slot units.
      Value ptr = gep(
          basePtrTy, slotTy, lmBase,
          tleSlotIndex(loc, op, adaptor.getLoopIndex(), numElems, i, rewriter));
      resultPtrs.push_back(ptr);
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, resultPtrs, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton::LoadOp in TLE path: simple element-wise load from LM pointers.
/// This handles the pattern: tle_local_ptr → tl.load (LM pointer load).
struct XPUTLETritonLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::LoadOp>,
      public LoadStoreConversionBase {
  XPUTLETritonLoadOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Type resTy = op.getResult().getType();
    Type llvmResultTy = typeConverter->convertType(resTy);
    Type elemTy = getElementTypeOrSelf(resTy);
    Type llvmElemTy = typeConverter->convertType(elemTy);

    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    // Unpack pointer values (each is an LM pointer from tle_local_ptr)
    auto llPtrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    unsigned numElems = llPtrs.size();

    SmallVector<Value> loadedVals;

    // ---- vectorized SM gather (xpu.sm_gather) -> one vgathers per vector ----
    // The vectorizer kept this SM-buffer load a vector tt.load (marker set in
    // Vectorize.cpp) so the affine chain around it stays vector. The index is
    // still SCALAR, one per lane; build a per-lane BYTE-offset vector from it
    // (channel index * elem bytes) and issue one SM vgather (`vgathers`) per
    // result vector -- the whole per-channel weight read becomes a single
    // vector instruction, no candidate loads, no select, no LM staging.
    if (auto vecElemTy = dyn_cast<VectorType>(elemTy)) {
      if (op->hasAttr("xpu.sm_gather")) {
        triton::xpu::TLELocalPtrOp lp = tleLocalPtrThroughLayout(op.getPtr());
        Operation *allocOp = lp ? getTLESmemAlloc(lp.getBuffer()) : nullptr;
        Value idxLowered =
            lp ? rewriter.getRemappedValue(lp.getIndices().back()) : Value();
        unsigned laneCount = vecElemTy.getNumElements();
        if (lp && allocOp && idxLowered) {
          SmallVector<Value> idxRegs =
              unpackLLElements(loc, idxLowered, rewriter);
          Type bufElemTy = getElementTypeOrSelf(lp.getBuffer().getType());
          Value smBase = getTLESmemBase(loc, rewriter, lp, allocOp);
          Type llvmVecTy = typeConverter->convertType(elemTy);
          if (bufElemTy.isBF16()) {
            // bf16 buffer, f32-promoted result. XPU3 has no bf16 SM load, but
            // the HF vgather returns raw 16-bit lanes (<32 x i16>); fold two
            // 32-lane gathers into f32 result vectors with VecBF16ToFP32 (the
            // same helper the LM bf16 vector load uses). resVecSize == 16 (f32
            // lanes); tleSmemGather guaranteed numElems is even.
            unsigned resVecSize = laneCount;
            if (resVecSize == 16 && idxRegs.size() == numElems * resVecSize &&
                numElems % 2 == 0) {
              VectorType i16x32 = VectorType::get(32, int_ty(16));
              SmallVector<Value> bf16vecs;
              for (unsigned g = 0; g < numElems / 2; ++g) {
                Value offVec = rewriter.create<LLVM::UndefOp>(loc, i16x32);
                for (unsigned li = 0; li < 32; ++li) {
                  unsigned e = g * 32 + li;
                  Value byteOff =
                      rewriter.create<LLVM::MulOp>(loc, idxRegs[e], i32_val(2));
                  byteOff =
                      rewriter.create<LLVM::TruncOp>(loc, int_ty(16), byteOff);
                  offVec = insert_element(i16x32, offVec, byteOff,
                                          i32_val((int32_t)li));
                }
                bf16vecs.push_back(
                    rewriter.create<mlir::LLVM::XPU::VGatherSMHFOp>(
                        loc, i16x32, smBase, offVec));
              }
              Type resElemTy = llvmVecTy;
              VecBF16ToFP32(ctx, loc, rewriter, resElemTy, (int)numElems,
                            (int)resVecSize, 32, bf16vecs);
              rewriter.replaceOp(op,
                                 {packLLElements(loc, typeConverter, bf16vecs,
                                                 rewriter, llvmResultTy)});
              return success();
            }
          } else {
            Type scalarTy = vecElemTy.getElementType();
            unsigned scalarBits = scalarTy.getIntOrFloatBitWidth();
            if ((scalarBits == 16 || scalarBits == 32) && !idxRegs.empty() &&
                idxRegs.size() == numElems * laneCount) {
              Type offElemTy = int_ty(scalarBits);
              VectorType offVecTy = VectorType::get(laneCount, offElemTy);
              int32_t elemBytes = (int32_t)(scalarBits / 8);
              SmallVector<Value> vecVals;
              for (unsigned vi = 0; vi < numElems; ++vi) {
                Value offVec = rewriter.create<LLVM::UndefOp>(loc, offVecTy);
                for (unsigned li = 0; li < laneCount; ++li) {
                  unsigned e = vi * laneCount + li;
                  Value byteOff = rewriter.create<LLVM::MulOp>(
                      loc, idxRegs[e], i32_val(elemBytes));
                  if (scalarBits == 16)
                    byteOff =
                        rewriter.create<LLVM::TruncOp>(loc, offElemTy, byteOff);
                  offVec = insert_element(offVecTy, offVec, byteOff,
                                          i32_val((int32_t)li));
                }
                // The intrinsic returns an INTEGER vector of the same width
                // (<16 x i32> / <32 x i16>); bitcast to the fp result vector.
                Value g;
                if (scalarBits == 32)
                  g = rewriter.create<mlir::LLVM::XPU::VGatherSMFOp>(
                      loc, offVecTy, smBase, offVec);
                else
                  g = rewriter.create<mlir::LLVM::XPU::VGatherSMHFOp>(
                      loc, offVecTy, smBase, offVec);
                vecVals.push_back(bitcast(g, llvmVecTy));
              }
              rewriter.replaceOp(op,
                                 {packLLElements(loc, typeConverter, vecVals,
                                                 rewriter, llvmResultTy)});
              return success();
            }
          }
        }
      }
    }

    // ---- segment-constant SM index: load the few candidates, select ------
    // A group-norm channel map is `ch = grp*group_size + cols // HW`: a
    // MONOTONE STEP along the registers that advances by at most one per HW
    // columns. When this core owns a single row (layout sizePerCore[0] == 1)
    // its spc columns are consecutive, so every register's channel lies in
    // [ch0, ch0 + ceil(spc / HW)] -- a handful of values at most, yet the
    // generic path below issues one shared-memory load PER REGISTER. Load each
    // candidate once and select per register against the register's own index,
    // which is already in registers, so the compare is far cheaper than the SM
    // load it replaces. Measured on FlagGems native_group_norm: this scalar
    // chain was 57-79% of the fused kernel's runtime on the official shapes.
    //
    // Only the exact `[+uniform] divsi(cols-derived, splat(C))` shape is taken.
    // Anything else keeps the generic path, so an unrecognized pattern costs
    // nothing but the missed optimization.
    if (numElems > 1 && !isa<VectorType>(elemTy) && !elemTy.isBF16()) {
      if (triton::xpu::TLELocalPtrOp lp =
              tleLocalPtrThroughLayout(op.getPtr())) {
        if (Operation *allocOp = getTLESmemAlloc(lp.getBuffer())) {
          auto ptrTensorTy =
              dyn_cast<RankedTensorType>(lp.getResult().getType());
          auto ptrElemTy =
              ptrTensorTy
                  ? dyn_cast<triton::PointerType>(ptrTensorTy.getElementType())
                  : nullptr;
          auto cl = ptrTensorTy
                        ? dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
                              ptrTensorTy.getEncoding())
                        : nullptr;
          if (ptrElemTy && ptrElemTy.getAddressSpace() == 2 && cl &&
              lp.getIndices().size() == 1 && cl.getSizePerCore().size() == 2 &&
              cl.getSizePerCore()[0] == 1) {
            auto strip = [](Value v) -> Value {
              while (Operation *d = v.getDefiningOp()) {
                if (isa<triton::xpu::ConvertLayoutOp, triton::ExpandDimsOp,
                        triton::xpu::BroadcastOp>(d)) {
                  v = d->getOperand(0);
                  continue;
                }
                break;
              }
              return v;
            };
            Value core = strip(lp.getIndices().back());
            // A TAIL row clamps the channel map with `min(expr, splat(CBLK-1))`
            // so the padding columns do not read past the [CBLK] buffer. The
            // clamped lanes are exactly the discards (the reduce masks them and
            // copy_l2g never writes them), so the step structure of the VALID
            // columns is untouched -- peel the clamp and look inside.
            for (unsigned depth = 0; depth < 3; ++depth) {
              auto mn = core.getDefiningOp<arith::MinSIOp>();
              if (!mn)
                break;
              // Keep the EXPRESSION side; the clamp constant is the other one
              // (picking by "is it a divsi" fails here because the expression
              // side is the addi that CONTAINS the divsi).
              auto constLike = [](Value v) {
                if (v.getDefiningOp<arith::ConstantOp>())
                  return true;
                if (auto sp = v.getDefiningOp<triton::SplatOp>())
                  return sp.getSrc().getDefiningOp<arith::ConstantOp>() !=
                         nullptr;
                return false;
              };
              Value l = strip(mn.getLhs()), r = strip(mn.getRhs());
              core = constLike(l) ? r : l;
            }
            if (auto add = core.getDefiningOp<arith::AddIOp>()) {
              Value l = strip(add.getLhs()), r = strip(add.getRhs());
              core = l.getDefiningOp<arith::DivSIOp>() ? l : r;
            }
            if (auto div = core.getDefiningOp<arith::DivSIOp>()) {
              int64_t divisor = 0;
              Value den = div.getRhs();
              if (auto sp = den.getDefiningOp<triton::SplatOp>())
                den = sp.getSrc();
              if (auto cst = den.getDefiningOp<arith::ConstantOp>())
                if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
                  divisor = ia.getInt();
              bool colDerived = false;
              {
                Value v = strip(div.getLhs());
                for (int depth = 0; depth < 6 && v; ++depth) {
                  if (v.getDefiningOp<triton::MakeRangeOp>()) {
                    colDerived = true;
                    break;
                  }
                  Operation *d = v.getDefiningOp();
                  if (!d)
                    break;
                  if (isa<triton::xpu::BroadcastOp, triton::ExpandDimsOp,
                          triton::xpu::ConvertLayoutOp, arith::AddIOp,
                          arith::MulIOp>(d)) {
                    v = d->getOperand(0);
                    continue;
                  }
                  break;
                }
              }
              // The per-register index values, already lowered: this load's
              // local_ptr operand has been converted (its result is what we are
              // reading), so its indices are remapped too.
              Value idxLowered =
                  rewriter.getRemappedValue(lp.getIndices().back());
              if (divisor >= 2 && colDerived && idxLowered) {
                SmallVector<Value> idxRegs =
                    unpackLLElements(loc, idxLowered, rewriter);
                if (idxRegs.size() == numElems) {
                  unsigned spc = std::max<unsigned>(cl.getSizePerCore()[1], 1);
                  unsigned steps = (spc + static_cast<unsigned>(divisor) - 1) /
                                   static_cast<unsigned>(divisor);
                  auto smPtrTy = LLVM::LLVMPointerType::get(ctx, 2);
                  Value smBase = getTLESmemBase(loc, rewriter, lp, allocOp);
                  SmallVector<Value> cands;
                  for (unsigned j = 0; j <= steps; ++j) {
                    Value off = (j == 0) ? idxRegs[0]
                                         : add(idxRegs[0], i32_val((int32_t)j));
                    cands.push_back(load(
                        llvmElemTy, gep(smPtrTy, llvmElemTy, smBase, off)));
                  }
                  SmallVector<Value> vals;
                  for (unsigned k = 0; k < numElems; ++k) {
                    Value v = cands[0];
                    for (unsigned j = 1; j < cands.size(); ++j)
                      v = select(icmp_eq(idxRegs[k],
                                         add(idxRegs[0], i32_val((int32_t)j))),
                                 cands[j], v);
                    vals.push_back(v);
                  }
                  rewriter.replaceOp(op,
                                     {packLLElements(loc, typeConverter, vals,
                                                     rewriter, llvmResultTy)});
                  return success();
                }
              }
            }
          }
        }
      }
    }

    // bf16 buffer read as f32 registers (`xpu.lm_bf16`, stamped by
    // tritonxpu-vectorize): XPU3 has no bf16->f32 convert instruction, so the
    // 16 bits are placed into the f32 pattern here, the same way
    // XPULoadOpConversion does it on the GM staging buffer.
    if (op->hasAttr("xpu.lm_bf16")) {
      // Chain left scalar (not vectorized): one bf16 element per register, so
      // the placement is a plain per-element fpext -- the Vec* helpers need
      // whole 512-bit registers.
      if (!isa<VectorType>(elemTy)) {
        // A cluster-shared (SM, addrspace 2) pointer has NO bf16 load on XPU3
        // -- SelectionDAG fails with "Cannot select: bf16 load from addrspace
        // 2" (an f16 SM load of the same 16-bit width selects fine, which is
        // how the fp16 SM-gather path works). Load the two bytes through an
        // f16 container and place the bits into the f32 pattern directly,
        // instead of fpext-ing a bf16 value that cannot be loaded.
        unsigned ptrAddrSpace = 0;
        if (auto ptrTensorTy =
                dyn_cast<RankedTensorType>(op.getPtr().getType()))
          if (auto ptrElemTy =
                  dyn_cast<triton::PointerType>(ptrTensorTy.getElementType()))
            ptrAddrSpace = ptrElemTy.getAddressSpace();
        for (unsigned i = 0; i < numElems; ++i) {
          if (ptrAddrSpace == 2) {
            Value bits = bitcast(load(f16_ty, llPtrs[i]), i16_ty);
            bits = zext(i32_ty, bits);
            loadedVals.push_back(bitcast(shl(bits, i32_val(16)), llvmElemTy));
          } else {
            Value bf16Val = load(bf16_ty, llPtrs[i]);
            loadedVals.push_back(
                rewriter.create<LLVM::FPExtOp>(loc, llvmElemTy, bf16Val));
          }
        }
        rewriter.replaceOp(op, {packLLElements(loc, typeConverter, loadedVals,
                                               rewriter, llvmResultTy)});
        return success();
      }
      auto resVecTy = cast<VectorType>(elemTy);
      unsigned resVecSize = resVecTy.getNumElements();
      unsigned ptrDataVecSize = resVecSize * 2; // 32 bf16 lanes -> 16 f32 lanes
      Value lmBase = llPtrs[0];
      if (op->hasAttr("xpu.bf16_unordered")) {
        VecBF16ToFP32Unordered(ctx, loc, rewriter, llvmElemTy, numElems,
                               resVecSize, ptrDataVecSize, lmBase, loadedVals);
      } else {
        VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
        VectorType halfVecBf16Ty = VectorType::get(resVecSize, bf16_ty);
        Value bf16Base = bitcast(lmBase, lmPtrTy);
        for (unsigned i = 0; i < numElems / 2; ++i)
          loadedVals.push_back(
              load(vecBf16Ty, gep(lmPtrTy, vecBf16Ty, bf16Base, i32_val(i))));
        if (numElems % 2 == 1)
          loadedVals.push_back(
              load(halfVecBf16Ty, gep(lmPtrTy, halfVecBf16Ty, bf16Base,
                                      i32_val(numElems - 1))));
        VecBF16ToFP32(ctx, loc, rewriter, llvmElemTy, numElems, resVecSize,
                      ptrDataVecSize, loadedVals);
      }
      rewriter.replaceOp(op, {packLLElements(loc, typeConverter, loadedVals,
                                             rewriter, llvmResultTy)});
      return success();
    }

    for (unsigned i = 0; i < numElems; ++i) {
      Value val = load(llvmElemTy, llPtrs[i]);
      loadedVals.push_back(val);
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, loadedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton::StoreOp in TLE path: simple element-wise store to LM pointers.
struct XPUTLETritonStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  XPUTLETritonStoreOpConversion(LLVMTypeConverter &converter,
                                const xpu::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto llPtrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto llVals = unpackLLElements(loc, adaptor.getValue(), rewriter);

    // NOTE on bf16 into cluster-shared SM (addrspace 2): XPU3 cannot select a
    // 16-bit float store there ("Cannot select: store (s16) ... addrspace 2"),
    // the mirror of the bf16 SM LOAD gap that the load lowering works around
    // with an f16 container. A container does NOT work on the store side:
    // both `store (bitcast bf16 to half)` and `store (bitcast bf16 to i16)`
    // are folded straight back to `store bfloat` by InstCombine in the
    // OPTIMIZE_O3 pass backend/compiler.py runs before llc (both verified --
    // the bitcast is emitted and is absent from the dumped llir). Defeating the
    // fold would mean emitting the store as inline asm, or building the i16
    // bits arithmetically from f32 the way VecFP32ToBF16* do so no bf16 value
    // exists to fold back to. The proper fix is a selection pattern in the XPU
    // LLVM backend, whose source is not part of this repo (only the prebuilt
    // backend/llvm19/bin/llc). Until then bf16 callers keep small statistics on
    // the LM strip (FlagGems native_group_norm gates this with SM_STATS).

    // f32 registers written into a bf16 buffer (`xpu.lm_bf16`): XPU3 has
    // neither an f32->bf16 convert nor a 16-bit pack, so the rounding and the
    // placement of each register's 16 halves are fused into the store, exactly
    // like XPUStoreOpConversion does on the GM staging buffer.
    if (op->hasAttr("xpu.lm_bf16")) {
      Type valElemTy = getElementTypeOrSelf(op.getValue().getType());
      // Chain left scalar: per-element fptrunc, for the same reason as the
      // load.
      if (!isa<VectorType>(valElemTy)) {
        for (unsigned i = 0; i < llPtrs.size(); ++i) {
          Value bf16Val =
              rewriter.create<LLVM::FPTruncOp>(loc, bf16_ty, llVals[i]);
          store(bf16Val, llPtrs[i]);
        }
        rewriter.eraseOp(op);
        return success();
      }
      auto valVecTy = cast<VectorType>(valElemTy);
      unsigned valueVecSize = valVecTy.getNumElements();
      unsigned ptrDataVecSize = valueVecSize * 2; // 16 f32 lanes -> 32 bf16
      Value lmBase = llPtrs[0];
      if (op->hasAttr("xpu.bf16_unordered"))
        VecFP32ToBF16Unordered(ctx, loc, rewriter, llVals.size(), valueVecSize,
                               ptrDataVecSize, llVals, lmBase);
      else if (isBf16Fast)
        VecFP32ToBF16(op, ctx, loc, rewriter, llVals.size(), valueVecSize,
                      ptrDataVecSize, llVals, lmBase);
      else
        VecFP32ToBF16Slow(ctx, loc, rewriter, llVals.size(), valueVecSize,
                          ptrDataVecSize, llVals, lmBase);
      rewriter.eraseOp(op);
      return success();
    }

    for (unsigned i = 0; i < llPtrs.size(); ++i) {
      store(llVals[i], llPtrs[i]);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_vload: vectorized load from a per-core LM buffer.
/// The result tensor element type is a VectorType<W x elem>; each struct slot
/// i loads W contiguous elements from LM[(base+i)*W : (base+i+1)*W] as one
/// <W x elem>, where base is loopIndex * numVecs (0 without a loopIndex).
struct XPUTLEVLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLEVLoadOp>,
      public LoadStoreConversionBase {
  XPUTLEVLoadOpConversion(LLVMTypeConverter &converter,
                          const xpu::TargetInfo &targetInfo,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLEVLoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLEVLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = adaptor.getBuffer();

    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    Type vecElemTy = resTensorTy.getElementType(); // vector<W x elem>
    Type llvmVecTy = typeConverter->convertType(vecElemTy);
    Type llvmResultTy = typeConverter->convertType(resTy);
    auto llvmStructTy = cast<LLVM::LLVMStructType>(llvmResultTy);
    unsigned numVecs = llvmStructTy.getBody().size();

    // --- SM (cluster-shared) cache branch -----------------------------------
    // All cores share ONE copy at global_smem + xpu.sm_offset;
    // adaptor.getBuffer() is only the dead ptr<0> placeholder from the alloc SM
    // branch, so the real ptr<2> base is recomputed here. Reading the shared
    // copy VECTORIZED (numVecs 64B-aligned vector loads) is the whole point:
    // the prior scalar-per-register SM read (256 ptr<2> loads/core) serialized
    // on bank conflicts; keeping the load vectorized (the result stays a vector
    // fed straight into broadcast/compute, never extracted back to scalar)
    // avoids both the contention and the extract-to-stack OOB.
    if (Operation *allocOp = getTLESmemAlloc(op.getBuffer())) {
      Value smByteBase = getTLESmemBase(loc, rewriter, op, allocOp);
      // Optional per-tile slice offset (element offset) for a stage-once SM
      // buffer read one [R0_BLOCK] slice at a time (roff = tile*R0_BLOCK). roff
      // is a multiple of the SIMD width, so the numVecs consecutive vector GEPs
      // below stay 64B aligned.
      if (Value off = adaptor.getSmElemOffset()) {
        Type smElemTy = typeConverter->convertType(
            cast<VectorType>(vecElemTy).getElementType());
        smByteBase = gep(ptr_ty(ctx, 2), smElemTy, smByteBase, off);
      }
      auto smPtrTy = LLVM::LLVMPointerType::get(ctx, 2);
      // Per-core column base in the cluster-shared buffer.
      //
      // Cores tile the axis, each owning the PRE-slice per-core row: that is
      // `xpu.row_stride` when unroll control sliced the segment (stamped for
      // every smem read, see stampSliceGeometry), else the layout's own
      // sizePerCore on the axis. `numVecs` is NOT that number -- it counts this
      // op's rows times columns, and for a broadcast-collapsed weight read the
      // rows are replicas of the same columns.
      Value coreColBase = i32_val(0);
      unsigned colsPerCore = 1;
      if (auto clusterLayout =
              mlir::dyn_cast_if_present<triton::xpu::ClusterLayoutAttr>(
                  resTensorTy.getEncoding())) {
        auto coresPerGroup = clusterLayout.getCoresPerGroup();
        auto groupsPerCluster = clusterLayout.getGroupsPerCluster();
        unsigned axis = resTensorTy.getRank() - 1;
        colsPerCore =
            std::max<unsigned>(clusterLayout.getSizePerCore()[axis], 1);
        int64_t rowStride = colsPerCore;
        if (auto rs = op->getAttrOfType<IntegerAttr>("xpu.row_stride"))
          rowStride = rs.getInt();
        // Same derivation the scalar SM path uses (make_range), shared so the
        // two cannot drift: gCoord[axis] * unitsPerCore, unsigned throughout.
        coreColBase = mlir::LLVM::XPU::getClusterLayoutAxisBase(
            rewriter, loc, clusterLayout, axis, rowStride);
        // Replica cores wrap, same as the make_range they mirror. The buffer's
        // column count in vector units is rowStride times the cores that own
        // unique data, which is the axis extent of the PRE-slice tensor.
        int64_t coresAlongAxis =
            int64_t(coresPerGroup[axis]) * groupsPerCluster[axis];
        if (int64_t uniqueCols = coresAlongAxis * rowStride)
          if (int64_t axisVecs = resTensorTy.getShape()[axis])
            if (uniqueCols > axisVecs && !op->hasAttr("xpu.row_stride"))
              coreColBase = urem(coreColBase, i32_val(axisVecs));
      }
      SmallVector<Value> smVals;
      for (unsigned i = 0; i < numVecs; ++i) {
        Value vecIdx = tleSmemSlotIndex(loc, op, adaptor.getLoopIndex(),
                                        coreColBase, colsPerCore, i, rewriter);
        smVals.push_back(
            load(llvmVecTy, gep(smPtrTy, llvmVecTy, smByteBase, vecIdx)));
      }
      rewriter.replaceOp(op, {packLLElements(loc, typeConverter, smVals,
                                             rewriter, llvmResultTy)});
      return success();
    }

    // A segment unroll control sliced reads one slice per iteration: the tile
    // loop's index arrives through $loopIndex and the base advances by a whole
    // slice (numVecs vectors, already the sliced count because the result type
    // was sliced with it). Absent index == whole-buffer load.
    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    SmallVector<Value> loadedVals;
    for (unsigned i = 0; i < numVecs; ++i) {
      Value ptr = gep(
          basePtrTy, llvmVecTy, lmBase,
          tleSlotIndex(loc, op, adaptor.getLoopIndex(), numVecs, i, rewriter));
      Value v = load(llvmVecTy, ptr);
      loadedVals.push_back(v);
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, loadedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton_xpu.tle_vstore: vectorized store to a per-core LM buffer.
struct XPUTLEVStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLEVStoreOp>,
      public LoadStoreConversionBase {
  XPUTLEVStoreOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLEVStoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLEVStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = adaptor.getBuffer();
    // Same restriction as XPUTLELocalStoreOpConversion: an smem buffer's
    // lowered base is only a dead ptr<0> placeholder, so storing through it
    // would be a wild store.
    assert(!isTLESmemBuffer(op.getBuffer()) &&
           "cannot vstore into a scope=smem TLE buffer");
    auto valTensorTy = cast<RankedTensorType>(op.getValue().getType());
    Type vecElemTy = valTensorTy.getElementType(); // vector<W x elem>
    Type llvmVecTy = typeConverter->convertType(vecElemTy);

    auto llVals = unpackLLElements(loc, adaptor.getValue(), rewriter);
    auto basePtrTy = LLVM::LLVMPointerType::get(
        ctx, triton::xpu::getMemDescAddrSpace(op.getBuffer().getType()));
    for (unsigned i = 0; i < llVals.size(); ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase,
                      tleSlotIndex(loc, op, adaptor.getLoopIndex(),
                                   llVals.size(), i, rewriter));
      store(llVals[i], ptr);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Shared helper for the vector<->scalar boundary ops: recover the LM base
/// pointer from the bufPtr operand. XPUAllocaOpConversion materializes an
/// alloca as a struct of per-element pointers, so the buffer base is the
/// first element of that struct.
static Value getBoundaryLMBase(Location loc, Value bufPtr,
                               ConversionPatternRewriter &rewriter) {
  if (!bufPtr)
    return Value();
  if (isa<LLVM::LLVMStructType>(bufPtr.getType()))
    return unpackLLElements(loc, bufPtr, rewriter)[0];
  return bufPtr;
}

/// Lower triton_xpu.pack: scalar tensor -> vector tensor, through LM.
/// The scalar elements are stored contiguously into the buffer and read back
/// as whole vectors, one vector load per result slot. Deliberately no
/// per-lane insertelement chain: the point of this op is to keep the
/// reinterpretation at one memory op per slot.
struct XPUPackOpConversion : public ConvertOpToLLVMPattern<triton::xpu::PackOp>,
                             public LoadStoreConversionBase {
  XPUPackOpConversion(LLVMTypeConverter &converter,
                      const xpu::TargetInfo &targetInfo,
                      ModuleAxisInfoAnalysis &axisAnalysisPass,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::PackOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = getBoundaryLMBase(loc, adaptor.getBufPtr(), rewriter);
    if (!lmBase)
      return op.emitError("triton_xpu.pack reached lowering without a bufPtr; "
                          "tritonxpu-alloca must attach one");

    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    Type llvmScalarTy =
        typeConverter->convertType(srcTensorTy.getElementType());

    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    Type llvmVecTy =
        typeConverter->convertType(resTensorTy.getElementType()); // <W x elem>
    Type llvmResultTy = typeConverter->convertType(resTy);
    unsigned numVecs =
        cast<LLVM::LLVMStructType>(llvmResultTy).getBody().size();

    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    for (unsigned i = 0; i < srcVals.size(); ++i) {
      Value ptr = gep(basePtrTy, llvmScalarTy, lmBase, i32_val(i));
      store(srcVals[i], ptr);
    }
    // The scalar stores and the vector loads below hit the same LM bytes from
    // two different pipes, and the hardware does not order them: without this
    // fence the widest load picks up stale bytes for whichever elements were
    // stored last. Measured on `add` (8 Mi f32) -- exactly the tail of each
    // 64-scalar group came back wrong (offsets 61..63 and 125..127 of the
    // 128-element per-core slice), nondeterministically, ~2% of elements.
    // Mask 1 is LM only; the GM bit (4) that createMfenceOp uses is not needed
    // here and would make every boundary pay for a DMA fence it never issues.
    createMfenceLMOp(rewriter, loc);

    SmallVector<Value> packedVals;
    for (unsigned i = 0; i < numVecs; ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase, i32_val(i));
      packedVals.push_back(load(llvmVecTy, ptr));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, packedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton_xpu.unpack: vector tensor -> scalar tensor, through LM.
/// Inverse of XPUPackOpConversion: one vector store per source slot, then a
/// scalar load per result element.
struct XPUUnpackOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::UnpackOp>,
      public LoadStoreConversionBase {
  XPUUnpackOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::UnpackOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = getBoundaryLMBase(loc, adaptor.getBufPtr(), rewriter);
    if (!lmBase)
      return op.emitError("triton_xpu.unpack reached lowering without a "
                          "bufPtr; tritonxpu-alloca must attach one");

    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    Type llvmVecTy =
        typeConverter->convertType(srcTensorTy.getElementType()); // <W x elem>

    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    Type llvmScalarTy =
        typeConverter->convertType(resTensorTy.getElementType());
    Type llvmResultTy = typeConverter->convertType(resTy);
    unsigned numElems =
        cast<LLVM::LLVMStructType>(llvmResultTy).getBody().size();

    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    // Volatile in both directions, and not for ordering -- the fence below does
    // that. Without it LLVM forwards the vector store into the scalar loads
    // (the LM buffer SROAs away entirely) and re-expresses the reinterpretation
    // as `bitcast <16 x i32> to <64 x i8>` + 16 `extractelement`s. The XPU3
    // backend lowers that by spilling the vector register to the *stack*
    // (`vstore_mask64.mz vr1{mr1}, -704(r30)`) and reading bytes back out, and
    // that faults on device: `truncint` at any size died with
    // cudaErrorLaunchFailure until these accesses became volatile. The LM round
    // trip is the semantics of this op, not an implementation detail -- it is
    // also what P4's `N + ceil(N/W) + 1` price counts, so letting it vanish
    // would make the boundary unmeasurable even where it does not fault.
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    for (unsigned i = 0; i < srcVals.size(); ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase, i32_val(i));
      store(srcVals[i], ptr, /*alignment=*/0, /*isVolatile=*/true);
    }
    // Same hazard as in the pack direction, stores and loads swapped.
    createMfenceLMOp(rewriter, loc);

    SmallVector<Value> scalarVals;
    for (unsigned i = 0; i < numElems; ++i) {
      Value ptr = gep(basePtrTy, llvmScalarTy, lmBase, i32_val(i));
      scalarVals.push_back(
          load(llvmScalarTy, ptr, /*alignment=*/0, /*isVolatile=*/true));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, scalarVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns
      .add<XPULoadOpConversion, XPUStoreOpConversion, XPUAllocaOpConversion,
           XPULoadScalarIndexedOpConversion, XPUStageSMOpConversion,
           XPUGM2LMMaskOpConversion, XPULM2GMMaskOpConversion,
           XPUAtomicRMWOpConversion, XPUGM2LMOpConversion, XPULM2GMOpConversion,
           // TLE patterns
           XPUTLELocalAllocOpConversion, XPUTLECopyG2LOpConversion,
           XPUTLECopyL2GOpConversion, XPUTLEDmaWaitOpConversion,
           XPUTLENormCopyG2LOpConversion, XPUTLENormCopyL2GOpConversion,
           XPUTLEWaitOpConversion, XPUTLELocalPtrOpConversion,
           XPUTLETritonLoadOpConversion, XPUTLETritonStoreOpConversion,
           XPUTLEVLoadOpConversion, XPUTLEVStoreOpConversion,
           // vector<->scalar boundary
           XPUPackOpConversion, XPUUnpackOpConversion>(
          typeConverter, targetInfo, axisInfoAnalysis, benefit);
}
