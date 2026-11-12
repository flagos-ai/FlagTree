#include "AsyncUtility.h"
#include "AccumulatorFence.h"
#include "FiniteSharedReadPlan.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "WarpSpecializeUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using ::mlir::triton::gpu::MemDescType;

namespace {
// Keep the complete, actual accumulator cohort tied across the boundary.
// Inline assembly here is a target lowering detail, not the source-level
// semantic representation. In particular, no fixed 128-register ABI leaks
// into the operator or the IR operation.
class AccumulatorFenceOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::AccumulatorFenceOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(triton::hcugpu::AccumulatorFenceOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto values = unpackLLElements(op.getLoc(), adaptor.getAcc(), rewriter);
    if (values.empty())
      return op.emitOpError("requires a nonempty per-thread accumulator cohort");
    // Only cross the immediately following pure dot, with an exact, single
    // accumulator use. Keep a tied empty memory fence here and transfer the
    // priority update to the first MMAC's full accumulator cohort. No memory,
    // synchronization, conversion, control-flow or unrelated user is crossed.
    bool defer = false;
    auto module = op->getParentOfType<ModuleOp>();
    auto enabled = module->getAttrOfType<BoolAttr>(
        "ttg.hcu.priority_after_first_mmac");
    if (op.getPriority() == 1 && enabled && enabled.getValue() &&
        op.getResult().hasOneUse()) {
      auto dot = dyn_cast_or_null<triton::DotOp>(op->getNextNode());
      auto tensor = dyn_cast<RankedTensorType>(op.getType());
      auto mma = tensor ? dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(
                              tensor.getEncoding())
                        : triton::gpu::AMDMfmaEncodingAttr{};
      if (dot && dot.getC() == op.getResult() && mma && mma.isHCUMmac()) {
        dot->setAttr("hcu.priority_after_first_mmac", rewriter.getUnitAttr());
        defer = true;
      }
    }
    auto outputs = triton::HCU::emitAccumulatorFence(
        op.getLoc(), values,
        defer ? "" : op.getPriority() ? "s_setprio 1" : "s_setprio 0",
        rewriter);
    rewriter.replaceOp(op, packLLElements(op.getLoc(), getTypeConverter(),
                                        outputs, rewriter, op.getType()));
    return success();
  }
};

static bool hasAlignedMatrixReadAllocation(
    Value source, const LLVMTypeConverter &converter,
    ArrayRef<int64_t> offsetAlignment = {});

static uint32_t matrixReadAddressBits(const LinearLayout &layout,
                                     StringAttr dim, unsigned inputMask) {
  uint32_t bits = 0;
  for (unsigned i = 0; i < layout.getInDimSizeLog2(dim); ++i)
    if (inputMask & (1u << i))
      bits |= uint32_t(layout.getBasis(dim, i)[0]);
  return bits;
}

// The chart combines coordinates with XOR. Bits in the constant row that
// cannot occur in the variable address can instead be added by the instruction.
static uint32_t matrixReadImmediate(uint32_t row, uint32_t variableBits,
                                    unsigned elemBytes) {
  uint64_t offset = uint64_t(row & ~variableBits) * elemBytes;
  if (row > INT32_MAX || offset > UINT16_MAX || offset % 16)
    return 0;
  return offset;
}

// The legacy gfx936 matrix read gathers eight 16-bit values per lane. Within
// each 16-lane group, output registers select producer addresses in the order
// [0, 4, 8, 12, 2, 6, 10, 14] plus ((outputLane >> 3) & 1). The
// output lane selects element (outputLane & 7) at each producer address.
// This implements an N-contiguous B operand without scalar LDS reads and
// register packing.
class LegacyMatrixLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LegacyMatrixLocalLoadOpConversion(const LLVMTypeConverter &converter,
                                   const HCU::TargetInfo &targetInfo,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (targetInfo.getArch() != "gfx936")
      return failure();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mma = dot ? dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(dot.getParent())
                   : triton::gpu::AMDMfmaEncodingAttr{};
    bool legacyDot =
        mma && mma.getMmacLayout() == triton::gpu::MmacLayout::LEGACY;
    if (!(legacyDot ||
          isa<triton::gpu::LinearEncodingAttr>(dstTy.getEncoding())) ||
        (dstTy.getRank() != 2 && dstTy.getRank() != 3) ||
        srcTy.getRank() != dstTy.getRank() ||
        srcTy.getShape() != dstTy.getShape() ||
        srcTy.getElementType() != dstTy.getElementType() ||
        !(dstTy.getElementType().isF16() || dstTy.getElementType().isBF16()) ||
        !isa<triton::gpu::SwizzledSharedEncodingAttr,
             triton::gpu::SharedLinearEncodingAttr>(srcTy.getEncoding()))
      return failure();

    // All encodings use the same physical gather proof below. Retain the
    // established rank-two legacy-dot path for affine views. Other layouts
    // need a single CTA and an independently proven asynchronous wait.
    auto module = op->getParentOfType<ModuleOp>();
    auto numCTAs = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
    auto numWarps = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
    auto threads = module->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
    bool legacyAffineView = legacyDot && dstTy.getRank() == 2;
    bool fullLayout =
        numCTAs && numCTAs.getInt() == 1 &&
        numWarps && numWarps.getInt() > 0 && numWarps.getInt() <= 8 &&
        llvm::isPowerOf2_64(numWarps.getInt()) &&
        threads && threads.getInt() == 64 &&
        !LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(srcTy) &&
        (!op.getToken() || HCU::isSyncedViaAsyncWait(op));
    if (!legacyAffineView && !fullLayout)
      return failure();
    // Explicit allocation alignment overrides the encoding's default. Prove
    // the actual root, including forwarded loop views, for either encoding.
    if (!hasAlignedMatrixReadAllocation(op.getSrc(), *getTypeConverter()))
      return failure();
    if (auto shared = dyn_cast<triton::gpu::SharedLinearEncodingAttr>(
            srcTy.getEncoding())) {
      if (srcTy.getRank() != 2 || shared.getAlignment() < 16)
        return failure();
    }

    auto *ctx = rewriter.getContext();
    auto reg = StringAttr::get(ctx, "register");
    auto lane = StringAttr::get(ctx, "lane");
    auto warp = StringAttr::get(ctx, "warp");
    auto block = StringAttr::get(ctx, "block");
    auto offset = StringAttr::get(ctx, "offset");
    auto cvt = triton::gpu::toLinearLayout(dstTy).invertAndCompose(
        triton::gpu::toLinearLayout(srcTy));
    if (!cvt.isTrivialOver({block}))
      return failure();
    cvt = cvt.sublayout({reg, lane, warp}, {offset});
    unsigned count = cvt.getInDimSize(reg);
    if (count < 8 || cvt.getInDimSize(lane) != 64)
      return failure();
    for (unsigned bit = 0; bit < 4; ++bit)
      if (cvt.getBasis(lane, bit)[0] != (1 << bit))
        return failure();
    // Every producer must address an aligned eight-element row. Smaller swizzles
    // or misaligned subviews cannot be represented by this instruction.
    for (auto dim : {reg, warp})
      for (unsigned bit = 0; bit < cvt.getInDimSizeLog2(dim); ++bit)
        if (cvt.getBasis(dim, bit)[0] & 7)
          return failure();
    for (unsigned bit = 4; bit < 6; ++bit)
      if (cvt.getBasis(lane, bit)[0] & 7)
        return failure();
    if (SharedMemoryObject::getMaskSpanOffsets(srcTy) & 7)
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto smem = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elemTy, rewriter);
    auto affine = smem.getShmemOffset(loc, rewriter, srcTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto producerReg = b.or_(b.and_(b.lshr(laneId, b.i32_val(2)), b.i32_val(3)),
                            b.shl(b.and_(b.lshr(laneId, b.i32_val(1)), b.i32_val(1)),
                                  b.i32_val(2)));
    auto producerLane = b.or_(b.and_(laneId, b.i32_val(48)),
                              b.shl(b.and_(laneId, b.i32_val(1)), b.i32_val(3)));
    constexpr llvm::StringLiteral readName =
        "llvm.hcu.ds.read.m32x16.b16.v4i32.p3";
    auto readFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(readName);
    if (!readFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto fnTy = LLVM::LLVMFunctionType::get(
          vec_ty(i32_ty, 4), {ptr_ty(ctx, 3), i32_ty}, false);
      readFunc = LLVM::LLVMFuncOp::create(rewriter, loc, readName, fnTy);
      readFunc.setConvergent(true);
      readFunc.setPassthroughAttr(rewriter.getArrayAttr({
          rewriter.getStringAttr("nounwind"), rewriter.getStringAttr("willreturn"),
          rewriter.getStringAttr("nocallback"), rewriter.getStringAttr("nofree")}));
    }
    // The instruction address uses register[0:2] and lane[3:5]; the
    // remaining register bits select whole 16-byte rows. A constant row bit
    // disjoint from every dynamic address/subview bit can be added rather
    // than XORed, so it is an exact byte immediate with the same AS3 origin.
    // This proof can simplify address registers, but the SDK currently adds
    // extra VM waits for the resulting pointer aliases on some pipelines.
    // Keep it opt-in while those pointer aliases still affect scheduling.
    auto foldAttr = module->getAttrOfType<IntegerAttr>(
        "ttg.hcu.matrix_read_immediate");
    bool foldImmediate = foldAttr && foldAttr.getInt() == 1;
    uint64_t dynamicBits = foldImmediate
        ? SharedMemoryObject::getMaskSpanOffsets(srcTy) : ~uint64_t(0);
    // A span mask alone does not prove arbitrary affine subview offsets.
    // Keep those reads unchanged until their actual offset bits are known.
    if (SharedMemoryObject::isAffineSharedMemoryAccess(srcTy))
      dynamicBits = ~uint64_t(0);
    dynamicBits |= matrixReadAddressBits(cvt, reg, 7) |
                   matrixReadAddressBits(cvt, lane, 0x38) |
                   matrixReadAddressBits(cvt, warp, ~0u);
    unsigned elemBytes = dstTy.getElementTypeBitWidth() / 8;
    llvm::DenseMap<uint64_t, Value> rowPointers;
    Value zeroRowAddress;
    SmallVector<Value> values;
    for (unsigned i = 0; i < count; i += 8) {
      uint64_t row = uint32_t(cvt.apply(
          {{reg, int32_t(i)}, {lane, 0}, {warp, 0}})[0].second);
      uint32_t byteOffset = matrixReadImmediate(row, dynamicBits, elemBytes);
      Value ptr;
      if (byteOffset) {
        uint64_t baseRow = row ^ (byteOffset / elemBytes);
        auto &cached = rowPointers[baseRow];
        if (!cached) {
          Value addr = b.xor_(zeroRowAddress, b.i32_val(baseRow));
          cached = b.gep(ptr_ty(ctx, 3), elemTy, smem.getBase(), addr);
        }
        ptr = cached;
      } else {
        auto regId = b.or_(producerReg, b.i32_val(i));
        Value addr = applyLinearLayout(loc, rewriter, cvt,
                                      {{reg, regId}, {lane, producerLane},
                                       {warp, warpId}})[0].second;
        addr = b.xor_(addr, affine);
        if (i == 0)
          zeroRowAddress = addr;
        ptr = b.gep(ptr_ty(ctx, 3), elemTy, smem.getBase(), addr);
        rowPointers.try_emplace(row, ptr);
      }
      // CallOp carries alias metadata, unlike CallIntrinsicOp. Preserve the
      // existing async-wait contract so LLVM does not insert redundant waits.
      auto read = LLVM::createLLVMCallOp(rewriter, loc, readFunc,
                                        {ptr, b.i32_val(byteOffset)});
      HCU::addLocalLoadNoAliasScope(
          op, cast<LLVM::AliasAnalysisOpInterface>(read.getOperation()));
      auto packed = b.bitcast(read.getResult(), vec_ty(elemTy, 8));
      for (unsigned j = 0; j < 8; ++j)
        values.push_back(b.extract_element(elemTy, packed, b.i32_val(j)));
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), values,
                                          rewriter, dstTy));
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

// Follow the same forwarded-successor operand interface used by upstream's
// TensorMemoryAllocation::getAlloc. Check every incoming edge, including both
// edges of a conditional branch to the same block, and reject unknown roots.
// When offsetAlignment is present, also prove the accumulated logical offsets
// are aligned to each dimension of the result tile.
static bool hasAlignedMatrixReadAllocation(
    Value source, const LLVMTypeConverter &converter,
    ArrayRef<int64_t> offsetAlignment) {
  llvm::DenseSet<Value> seen;
  llvm::DenseMap<Value, SmallVector<Value>> dependents;
  SmallVector<Value> worklist{source}, anchors;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    auto follow = [&](Value incoming) {
      dependents[incoming].push_back(value);
      worklist.push_back(incoming);
    };
    if (auto alloc = value.getDefiningOp<triton::gpu::LocalAllocOp>()) {
      auto alignment = alloc.getAlignmentOrDefault();
      if (!alloc.isSharedMemoryAlloc() || alignment < 16 || alignment % 16)
        return false;
      anchors.push_back(value);
    } else if (auto index =
                   value.getDefiningOp<triton::gpu::MemDescIndexOp>()) {
      // A leading buffer index must preserve the root's 16-byte alignment.
      int64_t strideModulo8 = 1;
      for (int64_t size : triton::gpu::getAllocationShapePerCTA(index.getType()))
        strideModulo8 = (strideModulo8 * (size % 8)) % 8;
      if (strideModulo8)
        return false;
      follow(index.getSrc());
    } else if (auto slice =
                   value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
      // Like the common ViewOp lowering, a subslice keeps the allocation base
      // and only adds logical offsets. Native-read patterns separately prove
      // the register/lane map and affine offset alignment via getMaskSpanOffsets.
      auto srcTy = slice.getSrc().getType();
      auto dstTy = slice.getType();
      if (srcTy.getElementType() != dstTy.getElementType() ||
          srcTy.getEncoding() != dstTy.getEncoding() ||
          srcTy.getAllocShape() != dstTy.getAllocShape() ||
          srcTy.getMemorySpace() != dstTy.getMemorySpace())
        return false;
      if (!offsetAlignment.empty()) {
        ArrayRef<int32_t> offsets = slice.getOffsets();
        if (offsets.size() < offsetAlignment.size())
          return false;
        for (auto [offset, alignment] :
             llvm::zip(offsets.take_back(offsetAlignment.size()),
                       offsetAlignment))
          if (offset < 0 || offset % alignment)
            return false;
      }
      follow(slice.getSrc());
    } else if (auto view =
                   value.getDefiningOp<triton::gpu::MemDescReinterpretOp>()) {
      // Reinterpret materializes the source's affine base, then resets logical
      // offsets. The root alignment survives only when that source offset is
      // itself a multiple of a 16-byte matrix-read row. This is independent of
      // either view's rank or logical tensor shape.
      auto from = view.getSrc().getType();
      auto to = view.getType();
      if (!offsetAlignment.empty() ||
          from.getElementType() != to.getElementType() ||
          from.getMemorySpace() != to.getMemorySpace() ||
          from.getElementTypeBitWidth() != 16 ||
          (SharedMemoryObject::getMaskSpanOffsets(from) & 7))
        return false;
      follow(view.getSrc());
    } else if (auto select = value.getDefiningOp<arith::SelectOp>()) {
      // Match TensorMemoryAllocation::getAlloc: a descriptor may forward either
      // select operand. Keep the existing alignment proof for every source.
      if (!isa<triton::gpu::MemDescType>(value.getType()))
        return false;
      follow(select.getTrueValue());
      follow(select.getFalseValue());
    } else if (auto bridge =
                   value.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Function/CF conversion precedes the memory patterns. Accept only the
      // exact one-to-one memdesc materialization introduced by that conversion.
      if (bridge->getNumOperands() != 1 || bridge->getNumResults() != 1)
        return false;
      Value input = bridge->getOperand(0);
      Type from = input.getType(), to = value.getType();
      auto fromMem = dyn_cast<triton::gpu::MemDescType>(from);
      auto toMem = dyn_cast<triton::gpu::MemDescType>(to);
      if (!((fromMem && converter.convertType(from) == to) ||
            (toMem && converter.convertType(to) == from)))
        return false;
      follow(input);
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      Block *block = arg.getOwner();
      Value capture = HCU::resolveWarpSpecializeCapture(arg);
      if (capture != value) {
        follow(capture);
        continue;
      }
      if (block->isEntryBlock())
        return false;
      bool hasIncoming = false;
      for (Block *pred : block->getPredecessors()) {
        auto terminator = pred->getTerminator();
        auto branch = dyn_cast<BranchOpInterface>(terminator);
        if (!branch ||
            !isa<cf::BranchOp, cf::CondBranchOp, LLVM::BrOp, LLVM::CondBrOp>(
                terminator))
          return false;
        for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
          if (branch->getSuccessor(i) != block)
            continue;
          auto operands = branch.getSuccessorOperands(i);
          unsigned produced = operands.getProducedOperandCount();
          unsigned number = arg.getArgNumber();
          if (number < produced ||
              number - produced >= operands.getForwardedOperands().size())
            return false;
          follow(operands.getForwardedOperands()[number - produced]);
          hasIncoming = true;
        }
      }
      if (!hasIncoming)
        return false;
    } else {
      return false;
    }
  }
  // Visiting a cycle is not evidence of alignment. Every visited value must
  // reach an actual aligned allocation; an unanchored SCC remains rejected.
  llvm::DenseSet<Value> anchored;
  while (!anchors.empty()) {
    Value value = anchors.pop_back_val();
    if (!anchored.insert(value).second)
      continue;
    for (Value dependent : dependents[value])
      anchors.push_back(dependent);
  }
  return !seen.empty() && anchored.size() == seen.size();
}

// Standard SharedLinear memory uses the same finite instruction proof as the
// legacy producer, with the address relation derived solely from LinearLayout.
// Failure leaves generic tracked local loads in place.
class LinearMatrixLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LinearMatrixLocalLoadOpConversion(const LLVMTypeConverter &converter,
                                  const HCU::TargetInfo &targetInfo,
                                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto shared = dyn_cast<triton::gpu::SharedLinearEncodingAttr>(srcTy.getEncoding());
    auto module = op->getParentOfType<ModuleOp>();
    auto ctas = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
    if (targetInfo.getArch() != "gfx936" || !shared ||
        srcTy.getRank() != 2 || dstTy.getRank() != 2 ||
        srcTy.getShape() != dstTy.getShape() ||
        srcTy.getElementType() != dstTy.getElementType() ||
        !(dstTy.getElementType().isF16() || dstTy.getElementType().isBF16()) ||
        !ctas || ctas.getInt() != 1 ||
        (op.getToken() && !HCU::isSyncedViaAsyncWait(op)) ||
        !hasAlignedMatrixReadAllocation(op.getSrc(), *getTypeConverter()) ||
        (SharedMemoryObject::getMaskSpanOffsets(srcTy) & 7))
      return failure();
    auto *ctx = rewriter.getContext();
    auto reg = StringAttr::get(ctx, "register");
    auto lane = StringAttr::get(ctx, "lane");
    auto warp = StringAttr::get(ctx, "warp");
    auto block = StringAttr::get(ctx, "block");
    auto offset = StringAttr::get(ctx, "offset");
    auto cvt = triton::gpu::toLinearLayout(dstTy).invertAndCompose(
        triton::gpu::toLinearLayout(srcTy));
    if (!cvt.isTrivialOver({block}))
      return failure();
    cvt = cvt.sublayout({reg, lane, warp}, {offset});
    unsigned count = cvt.getInDimSize(reg), warps = cvt.getInDimSize(warp);
    if (cvt.getInDimSize(lane) != 64 ||
        uint64_t(count) * warps * 64 > (uint64_t(1) << 20))
      return failure();
    SmallVector<unsigned> physical;
    for (unsigned r = 0; r < count; ++r)
      for (unsigned w = 0; w < warps; ++w)
        for (unsigned l = 0; l < 64; ++l)
          physical.push_back(cvt.apply({{reg, int32_t(r)}, {lane, int32_t(l)},
                                       {warp, int32_t(w)}})[0].second);
    auto plan = HCU::planFiniteSharedReads(count, warps,
        cvt.getOutDimSize(offset), physical, true, true, true);
    if (!plan || !plan->matrix || plan->optimizationBudgetExceeded)
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto smem = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elemTy, rewriter);
    auto affine = smem.getShmemOffset(loc, rewriter, srcTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    StringRef name = plan->matrixAlt
        ? "llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3"
        : "llvm.hcu.ds.read.m32x16.b16.v4i32.p3";
    auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!fn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto fnTy = LLVM::LLVMFunctionType::get(
          vec_ty(i32_ty, 4), {ptr_ty(ctx, 3), i32_ty}, false);
      fn = LLVM::LLVMFuncOp::create(rewriter, loc, name, fnTy);
      fn.setConvergent(true);
      fn.setPassthroughAttr(rewriter.getArrayAttr({
          rewriter.getStringAttr("nounwind"), rewriter.getStringAttr("willreturn"),
          rewriter.getStringAttr("nocallback"), rewriter.getStringAttr("nofree")}));
    }
    Value sourceReg = b.and_(b.lshr(laneId, b.i32_val(2)), b.i32_val(3));
    if (!plan->matrixAlt)
      sourceReg = b.or_(sourceReg, b.shl(
          b.and_(b.lshr(laneId, b.i32_val(1)), b.i32_val(1)), b.i32_val(2)));
    Value sourceLane = b.or_(b.and_(laneId, b.i32_val(48)),
        b.shl(b.and_(laneId, b.i32_val(plan->matrixAlt ? 3 : 1)),
              b.i32_val(plan->matrixAlt ? 2 : 3)));
    SmallVector<Value> values(count);
    for (unsigned r : plan->groups) {
      Value regId = b.i32_val(r);
      if (plan->repartition) {
        for (unsigned bit = 0; bit < 3; ++bit) {
          auto digit = b.and_(b.lshr(sourceReg, b.i32_val(bit)), b.i32_val(1));
          regId = b.or_(regId, b.shl(digit, b.i32_val(plan->registerAxes[bit])));
        }
      } else {
        regId = b.or_(regId, sourceReg);
      }
      auto addr = applyLinearLayout(loc, rewriter, cvt,
          {{reg, regId}, {lane, sourceLane}, {warp, warpId}})[0].second;
      addr = b.xor_(addr, affine);
      auto ptr = b.gep(ptr_ty(ctx, 3), elemTy, smem.getBase(), addr);
      // A subview offset is XORed with the chart. Do not import additive
      // immediates until they are separately proved disjoint from that offset.
      auto read = LLVM::createLLVMCallOp(rewriter, loc, fn, {ptr, b.i32_val(0)});
      HCU::addLocalLoadNoAliasScope(
          op, cast<LLVM::AliasAnalysisOpInterface>(read.getOperation()));
      auto packed = b.bitcast(read.getResult(), vec_ty(elemTy, 8));
      for (unsigned j = 0; j < 8; ++j) {
        unsigned output = plan->repartition ? plan->matrixByAnchor[r][j] : r + j;
        values[output] = b.extract_element(elemTy, packed, b.i32_val(j));
      }
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), values,
                                         rewriter, dstTy));
    return success();
  }
private:
  const HCU::TargetInfo &targetInfo;
};

// Private, default-off ALT experiment. Its continuous hardware tile is
// reg[chosen] -> offset1, lane[0:2] -> offset[2,4]. All remaining coordinates
// select aligned eight-element producer rows, checked before emitting IR.
class LegacyAltMatrixLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LegacyAltMatrixLocalLoadOpConversion(const LLVMTypeConverter &converter,
                                      const HCU::TargetInfo &targetInfo,
                                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto attr = module->getAttrOfType<IntegerAttr>("ttg.hcu.alt_matrix");
    if (targetInfo.getArch() != "gfx936")
      return failure();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto swizzled = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
        srcTy.getEncoding());
    // A distributed Linear result can already encode the ALT read's exact
    // value ownership, independently of any dot parent or scheduling hint.
    // The conversion below proves the physical lane/register tile. A tokened
    // read must satisfy the existing pre-conversion async-wait analysis used
    // by generic local loads and the existing normal matrix-read lowering.
    // Nonflat existing SwizzledShared is admitted only if the complete
    // low-bit composition, allocation alignment and affine guards below pass.
    bool natural = swizzled &&
        isa<triton::gpu::LinearEncodingAttr>(dstTy.getEncoding()) &&
        srcTy.getRank() == 2 && dstTy.getRank() == 2 &&
        srcTy.getShape() == dstTy.getShape() &&
        srcTy.getElementType() == dstTy.getElementType() &&
        (dstTy.getElementType().isF16() || dstTy.getElementType().isBF16()) &&
        swizzled.getOrder() == ArrayRef<unsigned>({1, 0}) &&
        llvm::all_of(swizzled.getCTALayout().getCTAsPerCGA(),
                     [](unsigned n) { return n == 1; }) &&
        llvm::all_of(swizzled.getCTALayout().getCTASplitNum(),
                     [](unsigned n) { return n == 1; }) &&
        (!op.getToken() || HCU::isSyncedViaAsyncWait(op));
    if (!natural && (!attr || (attr.getInt() != 2 && attr.getInt() != 3)))
      return failure();
    auto numCTAs = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
    auto numWarps = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
    auto threadsPerWarp =
        module->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
    auto prefetch = module->getAttrOfType<IntegerAttr>("ttg.hcu.local_prefetch");
    if (natural &&
        (!numCTAs || !numCTAs.getType().isInteger(32) ||
         !numWarps || !numWarps.getType().isInteger(32) ||
         !threadsPerWarp || !threadsPerWarp.getType().isInteger(32)))
      return failure();
    if (!numCTAs || numCTAs.getInt() != 1 || !numWarps ||
        !threadsPerWarp || threadsPerWarp.getInt() != 64 ||
        (numWarps.getInt() != 4 && numWarps.getInt() != 8) ||
        (!natural && prefetch && prefetch.getInt() != 0 &&
         prefetch.getInt() != 2))
      return failure();
    auto shared = dyn_cast<triton::gpu::SharedLinearEncodingAttr>(
        srcTy.getEncoding());
    auto dot = dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mma = dot ? dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(dot.getParent())
                   : triton::gpu::AMDMfmaEncodingAttr{};
    if (!natural && (!shared || shared.getAlignment() < 16 || !mma ||
        mma.getMmacLayout() != triton::gpu::MmacLayout::LEGACY ||
        mma.getIsTransposed() ||
        dot.getOpIdx() != 1 || dstTy.getRank() != 2 || srcTy.getRank() != 2 ||
        !(dstTy.getElementType().isF16() || dstTy.getElementType().isBF16()) ||
        (dot.getKWidth() != 4 && dot.getKWidth() != 8) ||
        mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
        llvm::any_of(mma.getTilesPerWarp(), [](unsigned n) { return n != 1; })))
      return failure();
    // Encoding alignment can be overridden by LocalAlloc's explicit alignment.
    if (!hasAlignedMatrixReadAllocation(op.getSrc(), *getTypeConverter()))
      return failure();

    auto *ctx = rewriter.getContext();
    auto reg = StringAttr::get(ctx, "register");
    auto lane = StringAttr::get(ctx, "lane");
    auto warp = StringAttr::get(ctx, "warp");
    auto block = StringAttr::get(ctx, "block");
    auto offset = StringAttr::get(ctx, "offset");
    auto cvt = triton::gpu::toLinearLayout(dstTy).invertAndCompose(
        triton::gpu::toLinearLayout(srcTy));
    if (!cvt.isTrivialOver({block}))
      return failure();
    cvt = cvt.sublayout({reg, lane, warp}, {offset});
    unsigned chosen = natural ? cvt.getInDimSizeLog2(reg) : attr.getInt();
    if (natural) {
      // Register bits 0/1 supply producerReg; select only the third ALT bit.
      for (unsigned bit = 2; bit < cvt.getInDimSizeLog2(reg); ++bit) {
        if (cvt.getBasis(reg, bit)[0] == 1) {
          chosen = bit;
          break;
        }
      }
    }
    unsigned count = cvt.getInDimSize(reg);
    if (count < 8 || chosen >= cvt.getInDimSizeLog2(reg) ||
        cvt.getInDimSize(lane) != 64 || cvt.getBasis(reg, chosen)[0] != 1 ||
        cvt.getBasis(lane, 0)[0] != 2 || cvt.getBasis(lane, 1)[0] != 4)
      return failure();
    for (unsigned bit = 0; bit < cvt.getInDimSizeLog2(reg); ++bit)
      if (bit != chosen && (cvt.getBasis(reg, bit)[0] & 7))
        return failure();
    for (unsigned bit = 2; bit < 6; ++bit)
      if (cvt.getBasis(lane, bit)[0] & 7)
        return failure();
    for (unsigned bit = 0; bit < cvt.getInDimSizeLog2(warp); ++bit)
      if (cvt.getBasis(warp, bit)[0] & 7)
        return failure();
    if (SharedMemoryObject::getMaskSpanOffsets(srcTy) & 7)
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto smem = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elemTy, rewriter);
    auto affine = smem.getShmemOffset(loc, rewriter, srcTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto producerReg = b.and_(b.lshr(laneId, b.i32_val(2)), b.i32_val(3));
    auto producerLane = b.or_(b.and_(laneId, b.i32_val(48)),
                              b.shl(b.and_(laneId, b.i32_val(3)), b.i32_val(2)));
    constexpr llvm::StringLiteral readName =
        "llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3";
    auto readFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(readName);
    if (!readFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto fnTy = LLVM::LLVMFunctionType::get(
          vec_ty(i32_ty, 4), {ptr_ty(ctx, 3), i32_ty}, false);
      readFunc = LLVM::LLVMFuncOp::create(rewriter, loc, readName, fnTy);
      readFunc.setConvergent(true);
      readFunc.setPassthroughAttr(rewriter.getArrayAttr({
          rewriter.getStringAttr("nounwind"), rewriter.getStringAttr("willreturn"),
          rewriter.getStringAttr("nocallback"), rewriter.getStringAttr("nofree")}));
    }
    SmallVector<Value> values(count);
    unsigned selectedMask = 3u | (1u << chosen);
    // An affine subview may occupy only its allocation-span bits if every
    // reaching subslice is aligned to the final tile. Check the offsets,
    // including identity subslices, instead of assuming this from the type.
    uint32_t dynamicBits = SharedMemoryObject::getMaskSpanOffsets(srcTy);
    bool foldImmediate = natural &&
        (!dynamicBits || hasAlignedMatrixReadAllocation(
                            op.getSrc(), *getTypeConverter(), srcTy.getShape()));
    // producerReg uses bits 0–1 and producerLane uses bits 2–5.
    if (foldImmediate)
      dynamicBits |= matrixReadAddressBits(cvt, reg, 3) |
                     matrixReadAddressBits(cvt, lane, 0x3c) |
                     matrixReadAddressBits(cvt, warp, ~0u);
    unsigned elemBytes = dstTy.getElementTypeBitWidth() / 8;
    Value zeroRowAddress;
    if (foldImmediate) {
      auto addr = applyLinearLayout(loc, rewriter, cvt,
                                    {{reg, producerReg}, {lane, producerLane},
                                     {warp, warpId}})[0].second;
      zeroRowAddress = b.xor_(addr, affine);
    }
    for (unsigned base = 0; base < count; ++base) {
      if (base & selectedMask)
        continue;
      uint32_t byteOffset = 0;
      Value addr;
      if (foldImmediate) {
        uint32_t row = cvt.apply(
            {{reg, int32_t(base)}, {lane, 0}, {warp, 0}})[0].second;
        byteOffset = matrixReadImmediate(row, dynamicBits, elemBytes);
        addr = b.xor_(zeroRowAddress, b.i32_val(row ^ (byteOffset / elemBytes)));
      } else {
        auto regId = b.or_(producerReg, b.i32_val(base));
        addr = applyLinearLayout(loc, rewriter, cvt,
                                 {{reg, regId}, {lane, producerLane},
                                  {warp, warpId}})[0].second;
        addr = b.xor_(addr, affine);
      }
      auto ptr = b.gep(ptr_ty(ctx, 3), elemTy, smem.getBase(), addr);
      auto read = LLVM::createLLVMCallOp(rewriter, loc, readFunc,
                                        {ptr, b.i32_val(byteOffset)});
      HCU::addLocalLoadNoAliasScope(
          op, cast<LLVM::AliasAnalysisOpInterface>(read.getOperation()));
      auto packed = b.bitcast(read.getResult(), vec_ty(elemTy, 8));
      for (unsigned j = 0; j < 8; ++j) {
        unsigned original = base | (j & 3u) | ((j >> 2) << chosen);
        values[original] = b.extract_element(elemTy, packed, b.i32_val(j));
      }
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), values,
                                          rewriter, dstTy));
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

class TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const HCU::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::LocalLoadOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitWidth != 16 && bitWidth != 8) {
      return failure();
    }
    auto ldsParams = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (!ldsParams)
      return failure();

    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout cvtDstLL = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvtDstLL = triton::gpu::toLinearLayout(dstTy).invertAndCompose(sharedLL);
      if (paddedEnc.getMinInterval() < ldsParams->tileSize)
        return failure();
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      cvtDstLL = triton::gpu::toLinearLayout(dstTy).invertAndCompose(sharedLL);
    }
    auto kBlock = StringAttr::get(ctx, "block");
    auto maybeSublayout = cvtDstLL.quotient({kBlock});
    if (!maybeSublayout) {
      return failure();
    }
    cvtDstLL = maybeSublayout.value();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    auto smemBase = smemObj.getBase();
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto calcPaddedOffset = [&](Value smemOffset) {
      TritonLLVMOpBuilder b(loc, rewriter);
      if (paddedEnc) {
        // Apply the offset needed for padding.
        Value padOffset = emitPadding(loc, rewriter, paddedEnc, bitWidth,
                                      smemOffset, /*offsetInBytes=*/true);
        smemOffset = b.add(smemOffset, padOffset);
      }
      return smemOffset;
    };

    llvm::SmallVector<Value> values;
    auto result =
        lowerDsReadTr(op, ldsParams.value(), loc, cvtDstLL, values, smemBase,
                      affineOffset, maskSpanAffineOffset, calcPaddedOffset,
                      llvmElemTy, rewriter, targetInfo);
    if (failed(result)) {
      return failure();
    }

    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(values.size(), llvmElemTy));
    auto value = packLLElements(loc, typeConverter, values, rewriter, structTy);

    rewriter.replaceOp(op, value);
    return success();
  }

private:
  LogicalResult lowerDsReadTr(
      triton::gpu::LocalLoadOp op,
      ::triton::HCU::TargetInfo::LDSTransLoadParams ldsParams, Location loc,
      LinearLayout cvt,
      SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
      Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
      std::function<Value(Value)> calcPaddedOffset, Type llvmElemTy,
      ConversionPatternRewriter &rewriter,
      const ::triton::HCU::TargetInfo &targetInfo) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto kOffset = S("offset");
    auto kAddr = S("addr");
    auto smemPtrTy = ptr_ty(ctx, 3);
    auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);

    // Map onto offsets (contiguous part) and addr (non-contiguous part)
    LinearLayout fullTile;
    // Contiguous tile
    LinearLayout tile;
    // ds_read_tr*_b64 performs a cooperative transposed load across 16
    // threads. The instruction processes an Nx16 tile (N=4 for 16-bit, N=8 for
    // 8-bit). The loaded tile is re-packed/transposed where lane i will
    // receive the i-th column.
    //
    // Loaded tile layout (input):     Register layout (output after transpose):
    //     K0  K1  ... K15               R0  R1  R2  R3
    // M0[ ............... ]    =>  T0 [ .   .   .   . ]
    // M1[ ............... ]        T1 [ .   .   .   . ]
    // M2[ ............... ]        ...
    // M3[ ............... ]        T15[ .   .   .   . ]
    //
    // Each lane loads 64 contiguous bits from LDS. After the transpose,
    // lane i receives column i from the input (elements strided by 16
    // the loaded tile).
    //
    // For example with N=4 (16-bit):
    // - Lane 0 receives elements from column 0: originally at [t0,t4,t8,t12]
    // - Lane 1 receives elements from column 1: originally at [t0,t4,t8,t12]
    //   These are the second 16 bits loaded by the same lanes before repacking
    // - Lane 4 receives elements from column 4: originally at [t1,t5,t9,t13]
    //
    // Note that there is no restriction on where elements are loaded
    // from, only that each lane needs to load 64 contiguous bits from shared
    // memory. We require N number of lanes to be contiguous since they read
    // consecutive 64 bits loaded from the same lanes.
    tile = LinearLayout::identity1D(ldsParams.tileSize, kLane, kOffset);

    const auto isaFamily = targetInfo.getISAFamily();
    // B8 types on gfx1250 require a different tile with double the contiguity
    bool doubleB8Contiguity =
        isaFamily == HCU::ISAFamily::GFX1250 && bitWidth == 8;
    const unsigned missingLanes =
        targetInfo.getWarpSize() / tile.getInDimSize(kLane);
    unsigned otherLanes = 1;
    if (isaFamily == HCU::ISAFamily::CDNA4) {
      otherLanes = (bitWidth == 8) ? 2 : 4;
    } else if (doubleB8Contiguity) {
      otherLanes = 2;
    }

    if (doubleB8Contiguity) {
      fullTile =
          tile * LinearLayout::identity1D(ldsParams.tileSize / 2, kReg, kAddr) *
          LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(2, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
    } else {
      fullTile =
          tile * LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(ldsParams.tileSize, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
    }
    // Add warp dimension so we can invert and compose with reps later
    fullTile *= LinearLayout::identity1D(1, kWarp, kAddr);

    if (cvt.getInDimSize(kReg) < fullTile.getInDimSize(kReg)) {
      return failure();
    }

    auto maybeQuot = divideLeft(cvt, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }

    // From here on we perform the lowering
    auto reps = zerosLike(tile) * maybeQuot.value();

    // Sanity check
    assert(fullTile.getInDimSize(kReg) * bitWidth == ldsParams.instBitWidth);

    // If we are lowering a subslice, the subslice offsets shall not touch the
    // contiguous part of the tile
    if (maskSpanAffineOffset & (tile.getOutDimSizeLog2(kOffset) - 1)) {
      return failure();
    }

    // fullTile.invert() is a map from kOffset, kAddr into kReg, kLane, kWarp
    // addrToOffset gives us a map from kAddr into kOffset, which is the map of
    // the addresses each lane should hold
    auto addrToOffset = fullTile.invert().compose(reps);
    // sanity check
    assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
           addrToOffset.getInDimSizeLog2(kAddr) <= 6);

    LinearLayout addrLayout =
        LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                      {kWarp, reps.getBases().lookup(kWarp)}},
                     {{kOffset, reps.getOutDimSize(kOffset)}}, false);

    // Compute the bits that are moved by one instruction
    // Compute elements for which we can swap the xor by an add
    auto [nAdditive, permStrides] =
        actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset);
    reps = permStrides.apply(reps);

    // Perform computation in bytes, LLVM optimises this better
    assert(bitWidth >= 8);
    auto i8Tile =
        zerosLike(LinearLayout::identity1D(bitWidth / 8, kReg, kOffset));
    auto i8AddrLayout = i8Tile * addrLayout;

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto regBase =
        applyLinearLayout(
            loc, rewriter, i8AddrLayout,
            {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
            .second;

    // It's fine that we don't compute the offset in bytes as affineOffset
    // will be folded into a constant
    auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitWidth / 8));
    regBase = b.xor_(regBase, affineOffsetI8);

    auto lowerInst = [&](RewriterBase &rewriter, Location loc, Value vecAddr,
                         int idx, VectorType vTy) -> SmallVector<Value> {
      assert(bitWidth == 16 || bitWidth == 8);
      Value dsReadTr;
      // tr16 instructions return vectors of bf16/f16 while "tr8" instructions
      // return vectors of i32. Generate the corresponding i32 vector
      auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
      auto vTyI32 = VectorType::get(numElemsI32, i32_ty);
      switch (targetInfo.getISAFamily()) {
      case HCU::ISAFamily::GFX1250: {
        if (bitWidth == 16) {
          dsReadTr = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, "llvm.amdgcn.ds.load.tr16.b128", {vTy},
                         {vecAddr})
                         .getResult(0);
        } else
          dsReadTr = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, "llvm.amdgcn.ds.load.tr8.b64", {vTyI32},
                         {vecAddr})
                         .getResult(0);
        break;
      }
      case HCU::ISAFamily::CDNA4: {
        if (bitWidth == 16) {
          dsReadTr =
              ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
        } else {
          dsReadTr =
              ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
        }
        break;
      }
      default:
        return {};
      }
      // GFX1250 is currently using LLVM intrinsics so it cannot cast it to
      // AliasAnalysisOpInterface
      if (targetInfo.getISAFamily() != HCU::ISAFamily::GFX1250)
        HCU::addLocalLoadNoAliasScope(
            op, cast<LLVM::AliasAnalysisOpInterface>(dsReadTr.getDefiningOp()));
      Value vecVal = b.bitcast(dsReadTr, vTy);
      SmallVector<Value> loadedVals;
      for (int v = 0; v < vTy.getNumElements(); v++) {
        loadedVals.push_back(
            b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
      }

      return loadedVals;
    };

    // Elements per op
    auto elemsPerInstr = fullTile.getInDimSize(kReg);
    auto elemsPerVec = ldsParams.instBitWidth / bitWidth;
    auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
    for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
      auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxI8 = regIdx * (bitWidth / 8);
      Value offset = b.xor_(regBase, b.i32_val(regIdxI8));
      for (int i2 = 0; i2 < nAdditive; i2 += elemsPerInstr) {
        // all these constants will go as immediate values to ds_read_tr
        auto regIdxAdd =
            reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}})[0].second;
        auto regIdxAddI8 = regIdxAdd * (bitWidth / 8);
        Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
        auto vecAddr =
            b.gep(smemPtrTy, i8_ty, smemBase, calcPaddedOffset(innerOffset),
                  LLVM::GEPNoWrapFlags::inbounds);
        llvm::append_range(vals,
                           lowerInst(rewriter, loc, vecAddr, i + i2, vecTy));
      }
    }
    // apply all the inverse permutations in the reverse order
    assert(vals.size() == cvt.getInDimSize(kReg));
    vals = permStrides.inverse().apply(vals);

    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

class LocalLoadPackedTransposedOpConversion
    : public ConvertOpToLLVMPattern<
          triton::hcugpu::LocalLoadPackedTransposedOp> {
public:
  LocalLoadPackedTransposedOpConversion(const LLVMTypeConverter &converter,
                                        const HCU::TargetInfo &targetInfo,
                                        PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::hcugpu::LocalLoadPackedTransposedOp>(
            converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor =
      typename triton::hcugpu::LocalLoadPackedTransposedOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::hcugpu::LocalLoadPackedTransposedOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and
    if (bitWidth != 8) {
      return failure();
    }
    // FP4 packed along M/N are not supported yet on GFX1250
    if (targetInfo.getISAFamily() == HCU::ISAFamily::GFX1250) {
      return failure();
    }

    return lowerSharedToDotOperandTransLL(op, adaptor, typeConverter, rewriter);
  }

private:
  LogicalResult
  lowerSharedToDotOperandTransLL(triton::hcugpu::LocalLoadPackedTransposedOp op,
                                 OpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    mlir::Type retTy = dstTy;
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto calcPaddedOffset = [&](Value smemOffset) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();
      if (auto paddedLayout = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
              srcTy.getEncoding())) {
        // Apply the offset needed for padding.
        Value padOffset = emitPadding(loc, rewriter, paddedLayout, bitWidth,
                                      smemOffset, /*offsetInBytes=*/true);
        smemOffset = b.add(smemOffset, padOffset);
      }
      return smemOffset;
    };

    auto shape = srcTy.getShape();
    auto ldsTransLoadParams = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (!ldsTransLoadParams)
      return failure();
    // FP4 are packed into i8 so the real bitWidth is different
    auto llBitWidth = 4;
    auto ldsTransLayout = triton::gpu::chooseDsReadTrLayout(
        dstTy.getEncoding(), shape, llBitWidth,
        ldsTransLoadParams->instBitWidth,
        ldsTransLoadParams->numLanesInShuffleGroup);

    // Check that we have computed a layout
    if (!ldsTransLayout) {
      return failure();
    }

    auto smemPtrTy = ptr_ty(ctx, 3);
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = ldsTransLayout->invertAndCompose(sharedLL);
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      cvt = ldsTransLayout->invertAndCompose(sharedLL);
    }
    // Check that we will be able to vectorize the load.
    // Need to have exactly ldsTransLoadParams->tileSize,
    // otherwise we can't use ds_read_tr
    auto [elemsPerVec, permutation] =
        largestVectorisation(ctx, cvt, bitWidth, ldsTransLoadParams->tileSize);

    if (paddedEnc)
      elemsPerVec = std::min<int>(elemsPerVec, paddedEnc.getMinInterval());

    if (elemsPerVec != ldsTransLoadParams->tileSize)
      return failure();

    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
    auto lowerInst = [&](RewriterBase &rewriter, Location loc,
                         ArrayRef<Value> inVals, Value vecAddr, int idx,
                         VectorType vTy) -> SmallVector<Value> {
      auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
      auto vTyI32 = VectorType::get(numElemsI32, i32_ty);
      Value dsReadTr =
          ROCDL::ds_read_tr4_b64::create(rewriter, loc, vTyI32, vecAddr);
      Value vecVal = b.bitcast(dsReadTr, vTy);
      SmallVector<Value> loadedVals;
      for (int v = 0; v < vTy.getNumElements(); v++) {
        loadedVals.push_back(
            b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
      }

      return loadedVals;
    };

    SmallVector<Value> outVals = lowerLdSt(
        loc, rewriter.getContext(), cvt, {}, // Input for store, output for load
        llvmElemTy, smemObj.getBase(), calcPaddedOffset, affineOffset,
        maskSpanAffineOffset, laneId, warpId, rewriter, targetInfo,
        ldsTransLoadParams->tileSize, lowerInst);
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, retTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

class LocalBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp> {
public:
  LocalBarrierOpConversion(const LLVMTypeConverter &converter,
                           const HCU::TargetInfo &targetInfo,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalBarrierOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::LocalBarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCDNA(targetInfo.getISAFamily()))
      return failure();
    // In CDNA we can lower local_barrier to s_waitcnt + s_barrier
    // - s_waitcnt specifies how many operations to VMEM/LDS can be outstanding
    //   when the instruction completes.
    //   In this case we require 0 outstanding LDS operations
    //   hcugpu::MemoryCounterWaitOp will lower s_waitcnt
    // - s_barrier syncronizes the execution for the CTA
    auto dsAttr = rewriter.getI32IntegerAttr(0);
    rewriter.create<hcugpu::MemoryCounterWaitOp>(
        op->getLoc(), /* load= */ nullptr, /* store= */ nullptr,
        /* ds= */ dsAttr);
    auto barrier = rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);
    if (targetInfo.getArch() == "gfx936") {
      // Keep the intrinsic through LLVM optimization and the existing LDS
      // ordering/scheduler workarounds. This marker identifies only local
      // barriers, whose contract excludes global-memory synchronization.
      auto module = barrier->getParentOfType<ModuleOp>();
      constexpr llvm::StringLiteral name =
          "__triton_hcu_local_barrier_marker";
      auto marker = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
      if (!marker) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto type = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(rewriter.getContext()), {}, false);
        marker = LLVM::LLVMFuncOp::create(rewriter, barrier.getLoc(), name, type);
        marker.setConvergent(true);
        marker.setPassthroughAttr(rewriter.getArrayAttr({
            rewriter.getStringAttr("nounwind"),
            rewriter.getStringAttr("willreturn"),
            rewriter.getStringAttr("nocallback"),
            rewriter.getStringAttr("nofree")}));
      }
      rewriter.setInsertionPointAfter(barrier);
      LLVM::createLLVMCallOp(rewriter, barrier.getLoc(), marker, {});
    }

    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

/// Encodes the waitcnt value for HCUGPU architectures.
///
/// Note: This function duplicates the bitpacking logic from HCUGPU backend
/// (llvm/lib/Target/HCUGPU/Utils/HCUGPUBaseInfo.h), as it's not accessible from
/// llvm/include. The logic handles different encoding schemes across
/// various GPU architecture versions (pre-gfx9 to gfx11).
///
/// The waitcnt encoding uses different bit positions for each counter
/// based on the ISA version:
/// - Vmcnt (vector memory counter): tracks pending vector memory operations
/// - Expcnt (export counter): tracks pending export operations
/// - Lgkmcnt (LDS/GDS/scalar memory counter): tracks pending LDS/GDS/scalar
/// memory ops
///
/// Each architecture version has its own bit layout, Vmcnt, Expcnt and Lgkmcnt
/// are decoded as follows:
///     Vmcnt = Waitcnt[3:0]        (pre-gfx9)
///     Vmcnt = Waitcnt[15:14,3:0]  (gfx9,10)
///     Vmcnt = Waitcnt[15:10]      (gfx11)
///     Expcnt = Waitcnt[6:4]       (pre-gfx11)
///     Expcnt = Waitcnt[2:0]       (gfx11)
///     Lgkmcnt = Waitcnt[11:8]     (pre-gfx10)
///     Lgkmcnt = Waitcnt[13:8]     (gfx10)
///     Lgkmcnt = Waitcnt[9:4]      (gfx11)
static FailureOr<unsigned> encodeWaitcnt(llvm::AMDGPU::IsaVersion isaVersion,
                                         unsigned vmcnt, unsigned lgkmcnt) {
  if (isaVersion.Major == 9) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(15u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 10) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 11) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    return (vmcnt << 10) | expcnt | (lgkmcnt << 4);
  }
  return failure();
}

struct MemoryCounterWaitOpConversion
    : public ConvertOpToLLVMPattern<hcugpu::MemoryCounterWaitOp> {
  MemoryCounterWaitOpConversion(const LLVMTypeConverter &converter,
                                const HCU::TargetInfo &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(hcugpu::MemoryCounterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto isaVersion = targetInfo.getIsaVersion();

    /// If major version >= fgx12, lower  to
    ///   * ROCDL::WaitDscntOp if ds is present
    ///   * ROCDL::WaitLoadcntOp if load is present
    ///   * ROCDL::WaitStorecntOp if store is present
    if (isaVersion.Major >= 12) {
      Location loc = op.getLoc();
      if (std::optional<int> ds = adaptor.getDs())
        ROCDL::WaitDscntOp::create(rewriter, loc, *ds);

      if (std::optional<int> load = adaptor.getLoad())
        ROCDL::WaitLoadcntOp::create(rewriter, loc, *load);

      if (std::optional<int> store = adaptor.getStore())
        ROCDL::WaitStorecntOp::create(rewriter, loc, *store);

      rewriter.eraseOp(op);
      return success();
    }

    /// Otherwise, lower to ROCDL::SWaitcntOp
    auto getVal = [](Attribute attr) -> unsigned {
      if (attr)
        return cast<IntegerAttr>(attr).getInt();

      // This value will be clamped to the maximum value for the target version.
      return 1024;
    };
    unsigned ds = getVal(adaptor.getDsAttr());

    unsigned vmcnt = 1024;
    Attribute load = adaptor.getLoadAttr();
    Attribute store = adaptor.getStoreAttr();
    if (load && store) {
      vmcnt = getVal(load) + getVal(store);
    } else if (load) {
      vmcnt = getVal(load);
    } else if (store) {
      vmcnt = getVal(store);
    }

    FailureOr<unsigned> waitcnt = encodeWaitcnt(isaVersion, vmcnt, ds);
    if (failed(waitcnt))
      return op.emitOpError("unsupported chipset");

    rewriter.replaceOpWithNewOp<ROCDL::SWaitcntOp>(op, *waitcnt);
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::HCU::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<AccumulatorFenceOpConversion>(typeConverter, benefit);
  patterns.add<LinearMatrixLocalLoadOpConversion>(typeConverter, targetInfo,
      PatternBenefit(benefit.getBenefit() + 2));
  patterns.add<TransLocalLoadOpConversion>(typeConverter, targetInfo,
                                           transBenefit);
  patterns.add<LegacyMatrixLocalLoadOpConversion>(typeConverter, targetInfo,
                                                 transBenefit);
  patterns.add<LegacyAltMatrixLocalLoadOpConversion>(typeConverter, targetInfo,
                                                    transBenefit);
  patterns.add<LocalLoadPackedTransposedOpConversion>(typeConverter, targetInfo,
                                                      benefit);
  patterns.add<LocalBarrierOpConversion, MemoryCounterWaitOpConversion>(
      typeConverter, targetInfo, barrierBenefit);
}
