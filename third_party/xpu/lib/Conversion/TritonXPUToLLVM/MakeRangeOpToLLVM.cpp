#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only

template <typename SourceOp>
struct RangeOpConversionBase : public ConvertOpToLLVMPattern<SourceOp> {
  explicit RangeOpConversionBase(LLVMTypeConverter &converter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern<SourceOp>(converter, benefit),
        converter(converter), targetInfo(targetInfo), benefit(benefit) {}

  using ConvertOpToLLVMPattern<SourceOp>::getTypeConverter;

protected:
  LLVMTypeConverter &converter;
  const TargetInfoBase &targetInfo;
  PatternBenefit benefit;
};

struct XPUMakeRangeOpConversion
    : public RangeOpConversionBase<triton::xpu::MakeRangeOp> {
  XPUMakeRangeOpConversion(LLVMTypeConverter &converter,
                           const TargetInfoBase &targetInfo,
                           PatternBenefit benefit)
      : RangeOpConversionBase<triton::xpu::MakeRangeOp>(converter, targetInfo,
                                                        benefit) {}

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.
  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const TargetInfoBase &target, Attribute layout,
              RankedTensorType type) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerCore = clusterLayout.getTotalElemsPerThread(shape, type);
    SmallVector<SmallVector<Value>> indices(elemsPerCore,
                                            SmallVector<Value>(rank));

    // offset = idInsideGroup * elemsPerCore + n
    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    unsigned groupSize = product(clusterLayout.getCoresPerGroup());
    Value idInsideGroup = srem(coreId, i32_val(groupSize));
    Value base = mul(idInsideGroup, i32_val(elemsPerCore));

    for (unsigned n = 0; n < elemsPerCore; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, idx_val(n));
      }
    }
    return indices;
  }

  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const TargetInfoBase &target, Attribute layout,
              RankedTensorType type, const Value &loopIdx) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerCore = clusterLayout.getTotalElemsPerThread(shape, type);
    SmallVector<SmallVector<Value>> indices(elemsPerCore,
                                            SmallVector<Value>(rank));

    // const int nthreads = core_num() * cluster_num();
    // const int tid = cluster_id() * core_num() + core_id();
    // for (int i = 0; i < iterCount; ++i) {
    //     const int idx = tid + nthreads * i;
    //     const int indice = idx * buf_len;
    Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
    auto coresPerGroup = clusterLayout.getCoresPerGroup();
    auto groupsPerCluster = clusterLayout.getGroupsPerCluster();
    bool atomicSim = (llvm::find_if(coresPerGroup,
                                    [](unsigned int num) {
                                      return num != 1;
                                    }) == coresPerGroup.end()) &&
                     (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                        return num != 1;
                      }) == groupsPerCluster.end());

    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value bufLen = i32_val(elemsPerCore);
    Value base;
    if (atomicSim) {
      base = mul(loopIdx, bufLen);
    } else {
      base = mul(add(coreId, mul(loopIdx, coreNum)), bufLen);
    }

    for (unsigned n = 0; n < elemsPerCore; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, idx_val(n));
      }
    }
    return indices;
  }

  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const TargetInfoBase &target, Attribute layout,
              RankedTensorType type, const Value &loopIdx,
              const Value &unrollIdx, uint32_t range) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    unsigned rank = shape.size();

    unsigned _unrollNum = clusterLayout.getTotalElemsPerThread(shape, type);
    auto coresPerGroup = clusterLayout.getCoresPerGroup().back();
    unsigned _elemsPerCore = range / coresPerGroup;
    SmallVector<SmallVector<Value>> indices(_unrollNum,
                                            SmallVector<Value>(rank));

    // (idInsideGroup + loopIdx * groupSize) * elemsPerCore + unrollIdx *
    // unrollNum + (0, unrollNum)
    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value elemsPerCore = i32_val(_elemsPerCore);
    Value unrollNum = i32_val(_unrollNum);
    Value _loopIdx = loopIdx ? loopIdx : i32_val(0);
    Value groupSize = i32_val(coresPerGroup);
    Value idInsideGroup = srem(coreId, groupSize);
    Value base =
        mul(add(idInsideGroup, mul(_loopIdx, groupSize)), elemsPerCore);
    base = add(base, mul(unrollIdx, unrollNum));

    for (unsigned n = 0; n < _unrollNum; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, idx_val(n));
      }
    }
    return indices;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType ty = op.getType();
    auto shape = ty.getShape();
    auto layout = ty.getEncoding();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));
    uint32_t _start = op.getStart();
    uint32_t _end = op.getEnd();
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, _start);
    uint32_t _range = _end - _start;

    auto loopIndex =
        adaptor.getLoopIndex(); // TODO[dyq]: check loopIndex Lowering Logic
    auto unrollIndex = adaptor.getUnrollIndex();
    SmallVector<SmallVector<Value>> idxs;
    if (unrollIndex) {
      idxs = emitIndices(loc, rewriter, targetInfo, layout, ty, loopIndex,
                         unrollIndex, _range);
    } else if (loopIndex) {
      idxs = emitIndices(loc, rewriter, targetInfo, layout, ty, loopIndex);
    } else {
      idxs = emitIndices(loc, rewriter, targetInfo, layout, ty);
    }

    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct XPUOutRangeOpConversion
    : public RangeOpConversionBase<triton::xpu::OutRangeOp> {

  SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter, const Attribute &layout,
              RankedTensorType type, int groupsize, int rowspercore,
              const Value &index) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerCore = clusterLayout.getTotalElemsPerThread(shape, type);
    SmallVector<SmallVector<Value>> indices(elemsPerCore,
                                            SmallVector<Value>(rank));

    // offset = (idx * group_num + group_id) * rowspercore + (0 ...
    // rowspercore-1)
    unsigned ngroup = product(clusterLayout.getGroupsPerCluster());

    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value groupId = sdiv(coreId, i32_val(groupsize));
    Value base =
        mul(add(mul(index, i32_val(ngroup)), groupId), i32_val(rowspercore));

    for (unsigned n = 0; n < elemsPerCore; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, idx_val(n));
      }
    }
    return indices;
  }

  XPUOutRangeOpConversion(LLVMTypeConverter &converter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : RangeOpConversionBase<triton::xpu::OutRangeOp>(converter, targetInfo,
                                                       benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::OutRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType ty = op.getType();
    auto shape = ty.getShape();
    auto layout = ty.getEncoding();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));

    auto groupsize = adaptor.getGroupsize();
    auto rowspercore = adaptor.getRowspercore();
    auto index = adaptor.getIndex();

    auto idxs =
        emitIndices(loc, rewriter, layout, ty, groupsize, rowspercore, index);

    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = multiDim.value()[0];
    }

    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct XPUInterleaveOpConversion
    : public RangeOpConversionBase<triton::xpu::InterleaveOp> {

  XPUInterleaveOpConversion(LLVMTypeConverter &converter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit)
      : RangeOpConversionBase<triton::xpu::InterleaveOp>(converter, targetInfo,
                                                         benefit) {}

  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const TargetInfoBase &target, Attribute layout,
              RankedTensorType type, const Value &loopIdx) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    auto elemBit = type.getElementType().getIntOrFloatBitWidth();
    unsigned rank = shape.size();
    unsigned elemsPerCore = clusterLayout.getTotalElemsPerThread(shape, type);

    // const int nthreads = core_num() * cluster_num();
    // const int tid = core_id() * cluster_num() + cluster_id();
    // for (int i = 0; i < iterCount; ++i) {
    //     const int idx = tid + nthreads * i;
    //     const int offset = idx * buf_len;
    Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
    auto coresPerGroup = clusterLayout.getCoresPerGroup();
    auto groupsPerCluster = clusterLayout.getGroupsPerCluster();

    bool atomicSim = (llvm::find_if(coresPerGroup,
                                    [](unsigned int num) {
                                      return num != 1;
                                    }) == coresPerGroup.end()) &&
                     (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                        return num != 1;
                      }) == groupsPerCluster.end());

    Value base;
    if (atomicSim) {
      base = mul(loopIdx, i32_val(elemsPerCore));
    } else {
      Value clusterNum = mlir::LLVM::XPU::getGridDim(rewriter, loc);
      Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
      Value clusterId = mlir::LLVM::XPU::getBlockId(rewriter, loc);
      Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
      Value bufLen = i32_val(elemsPerCore);
      Value _loopIdx = loopIdx;
      if (elemBit == 64) {
        bufLen = i64_val(elemsPerCore);
        clusterNum = rewriter.create<LLVM::SExtOp>(loc, i64_ty, clusterNum);
        coreNum = rewriter.create<LLVM::SExtOp>(loc, i64_ty, coreNum);
        clusterId = rewriter.create<LLVM::SExtOp>(loc, i64_ty, clusterId);
        coreId = rewriter.create<LLVM::SExtOp>(loc, i64_ty, coreId);
        _loopIdx = rewriter.create<LLVM::SExtOp>(loc, i64_ty, loopIdx);
      }
      Value nThread = mul(clusterNum, coreNum);
      Value tid = add(mul(coreId, clusterNum), clusterId);
      Value idx = add(tid, mul(nThread, _loopIdx));
      base = mul(idx, bufLen);
    }

    SmallVector<SmallVector<Value>> indices(elemsPerCore,
                                            SmallVector<Value>(rank));
    for (unsigned n = 0; n < elemsPerCore; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, int_val(elemBit, n));
      }
    }
    return indices;
  }

  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const TargetInfoBase &target, Attribute layout,
              RankedTensorType type, const Value &loopIdx,
              const Value &unrollIdx, uint32_t range) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    auto elemBit = type.getElementType().getIntOrFloatBitWidth();
    unsigned rank = shape.size();

    unsigned _unrollNum = clusterLayout.getTotalElemsPerThread(shape, type);
    auto unrollNum = i32_val(_unrollNum);
    auto _coresPerGroup = product(clusterLayout.getCoresPerGroup());
    auto _groupsPerCluster = product(clusterLayout.getGroupsPerCluster());
    auto _coreNum = _coresPerGroup * _groupsPerCluster;
    unsigned _elemsPerCore = ceil<unsigned>(range, _coreNum);

    // const int nthreads = core_num() * cluster_num();
    // const int tid = core_id() * cluster_num() + cluster_id();
    // for (int loopIdx = 0; loopIdx < iterCount; ++loopIdx) {
    //   for (int unrollIdx = 0; unrollIdx < unrollNum; ++unrollIdx) {
    //     const int idx = tid + nthreads * loopIdx;
    //     const int offset = idx * bufLen + unrollIdx * unrollNum;
    Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
    auto coresPerGroup = clusterLayout.getCoresPerGroup();
    auto groupsPerCluster = clusterLayout.getGroupsPerCluster();

    bool atomicSim = (llvm::find_if(coresPerGroup,
                                    [](unsigned int num) {
                                      return num != 1;
                                    }) == coresPerGroup.end()) &&
                     (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                        return num != 1;
                      }) == groupsPerCluster.end());

    Value base;
    Value newUnrollIdx = unrollIdx;
    Value bufLen = int_val(elemBit, _elemsPerCore);
    if (atomicSim) {
      base = mul(loopIdx, bufLen);
    } else {
      Value clusterNum = mlir::LLVM::XPU::getGridDim(rewriter, loc);
      Value coreNum = mlir::LLVM::XPU::getBlockDim(rewriter, loc);
      Value clusterId = mlir::LLVM::XPU::getBlockId(rewriter, loc);
      Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
      Value _loopIdx = loopIdx;
      if (elemBit == 64) {
        clusterNum = rewriter.create<LLVM::SExtOp>(loc, i64_ty, clusterNum);
        coreNum = rewriter.create<LLVM::SExtOp>(loc, i64_ty, coreNum);
        clusterId = rewriter.create<LLVM::SExtOp>(loc, i64_ty, clusterId);
        coreId = rewriter.create<LLVM::SExtOp>(loc, i64_ty, coreId);
        _loopIdx = rewriter.create<LLVM::SExtOp>(loc, i64_ty, loopIdx);
        newUnrollIdx = rewriter.create<LLVM::SExtOp>(loc, i64_ty, unrollIdx);
        unrollNum = rewriter.create<LLVM::SExtOp>(loc, i64_ty, unrollNum);
      }
      Value nThread = mul(clusterNum, coreNum);
      Value tid = add(mul(coreId, clusterNum), clusterId);
      Value idx = add(tid, mul(nThread, _loopIdx));
      base = add(mul(idx, bufLen), mul(newUnrollIdx, unrollNum));
    }

    SmallVector<SmallVector<Value>> indices(_unrollNum,
                                            SmallVector<Value>(rank));
    for (unsigned n = 0; n < _unrollNum; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, int_val(elemBit, n));
      }
    }
    return indices;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::InterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType ty = op.getType();
    auto shape = ty.getShape();
    auto layout = ty.getEncoding();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32) || elemTy.isInteger(64));
    uint32_t _start = op.getStart();
    uint32_t _end = op.getEnd();
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, _start);
    uint32_t _range = _end - _start;

    auto loopIndex =
        adaptor.getLoopIndex(); // TODO[dyq]: check loopIndex Lowering Logic
    auto unrollIndex = adaptor.getUnrollIndex();
    SmallVector<SmallVector<Value>> idxs;
    if (unrollIndex) {
      idxs = emitIndices(loc, rewriter, targetInfo, layout, ty, loopIndex,
                         unrollIndex, _range);
    } else {
      idxs = emitIndices(loc, rewriter, targetInfo, layout, ty, loopIndex);
    }

    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// TLE path lowers the upstream triton::MakeRangeOp directly to LLVM (mirroring
// XPUTLETritonLoadOpConversion for triton::LoadOp): in the TLE pipeline
// tt.make_range survives to this stage (the normcopy path rewrites it to
// xpu::MakeRangeOp earlier), so this pattern only fires for TLE kernels and
// leaves the shared xpu::MakeRangeOp lowering untouched.
struct XPUTLETritonMakeRangeOpConversion
    : public RangeOpConversionBase<triton::MakeRangeOp> {
  XPUTLETritonMakeRangeOpConversion(LLVMTypeConverter &converter,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit)
      : RangeOpConversionBase<triton::MakeRangeOp>(converter, targetInfo,
                                                   benefit) {}

  // Axis-aware, hierarchical per-core offset (scheme A).
  //
  // The per-core base along the (innermost) axis is
  // getClusterLayoutAxisBase(.., elemsPerCore) -- see Utility.h for the
  // gCoord derivation. The TLE make_range emits an iota off it:
  // offset = base + n.
  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter, Attribute layout,
              RankedTensorType type) const {
    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerCore = clusterLayout.getTotalElemsPerThread(shape, type);
    SmallVector<SmallVector<Value>> indices(elemsPerCore,
                                            SmallVector<Value>(rank));

    unsigned axis = rank - 1;
    unsigned axisSize = static_cast<unsigned>(shape[axis]);

    // 1D cyclic-replicated layout -- the LargeN row vector that
    // getCluster1DForSliceDim(d == 1) / getOutput1D build when g > 1
    // (cpg=[g] > 1, gpc=[ngroup] > 1). Its authoritative element map is
    // planLargeNOutputSegments': "group grp owns rows grp, grp+numGroups,
    // ...; slot k maps to global row grp + k*numGroups", and the g col-cores
    // of a group are REPLICAS holding identical values. The block base below
    // (gCoord * unitsPerCore == coreId here) instead hands every col-core its
    // own element, which nothing else agrees with -- observed as a
    // row-shifted arange (`row r` reading rid `r*g`) the first time a kernel
    // actually CONSUMES the row arange values (the fused group-norm's
    // SM-gather weight map). Plain tle kernels never noticed because
    // rows/cols aranges only ever fed local_ptr indices, which the LM path
    // drops.
    if (rank == 1 && clusterLayout.getCoresPerGroup()[0] > 1 &&
        clusterLayout.getGroupsPerCluster()[0] > 1) {
      unsigned groupSize = product(clusterLayout.getCoresPerGroup());
      unsigned numGroups = clusterLayout.getGroupsPerCluster()[0];
      Value groupId =
          udiv(mlir::LLVM::XPU::getThreadId(rewriter, loc), i32_val(groupSize));
      for (unsigned n = 0; n < elemsPerCore; ++n) {
        Value idx = add(groupId, mul(i32_val(n), i32_val(numGroups)));
        if (axisSize > 0)
          idx = urem(idx, i32_val(axisSize));
        for (unsigned k = 0; k < rank; ++k) {
          indices[n][k] = idx;
        }
      }
      return indices;
    }

    Value base = mlir::LLVM::XPU::getClusterLayoutAxisBase(
        rewriter, loc, clusterLayout, axis, elemsPerCore);

    // When the axis is distributed over more cores than it has unique elements
    // (shapePerCTATile[axis] = elemsPerCore * nCores[axis] > shape[axis]), the
    // surplus cores are data REPLICAS, not new indices. Without wrapping, a
    // degenerate axis (e.g. XBLOCK=1 spread over 64 cores) would make each core
    // produce its own coreId as the "index", corrupting any value derived from
    // this range (observed as an extra 24*core_id in permuted addresses). Wrap
    // the per-core index modulo the axis size so replicas mirror the unique
    // data, matching emitOffsetForClusterLayout's `% shapePerCTA` semantics.

    for (unsigned n = 0; n < elemsPerCore; ++n) {
      Value idx = add(base, idx_val(n));
      if (axisSize > 0)
        idx = urem(idx, i32_val(axisSize));
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = idx;
      }
    }
    return indices;
  }

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType ty = op.getType();
    auto layout = ty.getEncoding();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));
    uint32_t _start = op.getStart();
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, _start);

    auto idxs = emitIndices(loc, rewriter, layout, ty);

    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }
};

void mlir::triton::xpu::populateMakeRangeOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {

  patterns.add<XPUMakeRangeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<XPUTLETritonMakeRangeOpConversion>(typeConverter, targetInfo,
                                                  benefit);
  patterns.add<XPUOutRangeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<XPUInterleaveOpConversion>(typeConverter, targetInfo, benefit);
}
