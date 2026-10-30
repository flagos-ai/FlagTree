#include "PatternTritonXPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only
#include "CombineEmit.h"
#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include <cfloat>
namespace {

using ::mlir::triton::gpu::getTotalElemsPerThread;

struct XPUReduceOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ReduceOp> {

  XPUReduceOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ReduceOp>(converter, benefit),
        targetInfo(targetInfo) {}

  inline SmallVector<SmallVector<Value>>
  emitIndices(Location loc, RewriterBase &rewriter,
              const xpu::TargetInfo &target, Attribute layout,
              RankedTensorType type, const Value &loopIdx) const {
    SmallVector<int64_t> shape(type.getShape());

    // for sliceEncoding, we need to set layout as its parent
    if (auto slice = dyn_cast<SliceEncodingAttr>(layout)) {
      layout = slice.getParent();
      shape.insert(shape.begin() + slice.getDim(), 1);
    }

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
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
    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value bufLen = i32_val(elemsPerCore);
    Value base = mul(add(coreId, mul(loopIdx, coreNum)), bufLen);

    for (unsigned n = 0; n < elemsPerCore; ++n) {
      for (unsigned k = 0; k < rank; ++k) {
        indices[n][k] = add(base, idx_val(n));
      }
    }
    return indices;
  }

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(triton::xpu::ReduceOp op, int i) const {
    auto ty = getElementTypeOrSelf(op.getInputTypes()[i].getElementType());
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  std::pair<SmallVector<Value>, SmallVector<Value>>
  getSmemBases(triton::xpu::ReduceOp op, unsigned elems,
               ConversionPatternRewriter &rewriter) const {
    ReduceOpHelper helper(op);
    SmallVector<int64_t> offsets;
    // auto curIdx = helper.getReduceId();
    // getSMOffsets() returns null when the predecessor id has no recorded
    // offsets. That is the normal case when the predecessor is
    // core-synchronous: assignReduceScanIds() numbers *every* reduce, but only
    // a non-core-synchronous one reaches initSharedMemory() and so calls
    // setSMOffsets(). setSMOffsets() itself already treats a missing
    // predecessor as a zero offset; do the same here rather than dereferencing
    // null (which segfaults the compiler and, under the crash reproducer, is
    // reported as a bare "Failures have been detected ..." with no diagnostic).
    unsigned prevSMOffset = 0;
    if (helper.getReduceId() != 0)
      if (const auto *prev = helper.getSMOffsets(helper.getReduceId() - 1))
        prevSMOffset = prev->endOffset;
    // op->dump();
    // LLVM_DEBUG(llvm::dbgs() << "\nprevSMOffset = " << prevSMOffset << "\n");

    auto loc = op.getLoc();
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands() - 1); // skip loopIndex
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      if (i == op.getNumOperands() - 1 ||
          j == op.getNumOperands() - 1) { // skip loopIndex
        return false;
      }
      // On the vectorized path the element type is vector<NxT>, which has no
      // bitwidth; the smem slot is sized from the scalar type either way (see
      // getElementType above). Only multi-operand reduces reach this
      // comparator, which is why the single-output vector reduces never hit it.
      return getElementTypeOrSelf(op.getElementTypes()[i])
                 .getIntOrFloatBitWidth() >
             getElementTypeOrSelf(op.getElementTypes()[j])
                 .getIntOrFloatBitWidth();
    });

    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    indexToBase[indices[0]] =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    // add prev reduceOp used sm bytes offset
    indexToBase[indices[0]] =
        gep(ptr_ty(rewriter.getContext(), 2), getElementType(op, indices[0]),
            indexToBase[indices[0]], i32_val(prevSMOffset));

    offsets.push_back(
        (getElementType(op, indices[0]).getIntOrFloatBitWidth() * elems) / 8);
    for (unsigned i = 1; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      indexToBase[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 2), getElementType(op, indices[i - 1]),
          indexToBase[indices[i - 1]], i32_val(elems));
      offsets.push_back(
          (getElementType(op, indices[i - 1]).getIntOrFloatBitWidth() * elems) /
          8);
    }

    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands() - 1);     // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      smemBases[i] = indexToBase[i];
    }

    // loopResCacheSmemBases[k] is the base pointer for the k-th operand which
    // is the prev loop reduce result
    SmallVector<Value> loopResCacheSmemBases(op.getNumOperands() -
                                             1); // skip loopIndex
    std::map<unsigned, Value> indexToBaseForLoopCache;
    indexToBaseForLoopCache[indices[0]] = gep(
        ptr_ty(rewriter.getContext(), 2), getElementType(op, indices.back()),
        smemBases.back(), i32_val(elems));
    offsets.push_back(
        (getElementType(op, indices[0]).getIntOrFloatBitWidth() * 1) / 8);
    for (unsigned i = 1; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      indexToBaseForLoopCache[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 2), getElementType(op, indices[i - 1]),
          indexToBaseForLoopCache[indices[i - 1]], i32_val(1));
      offsets.push_back(
          (getElementType(op, indices[0]).getIntOrFloatBitWidth() * 1) / 8);
    }

    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      loopResCacheSmemBases[i] = indexToBaseForLoopCache[i];
    }

    helper.setSMOffsets(helper.getReduceId(), offsets);
    return {smemBases, loopResCacheSmemBases};
  }

  LogicalResult
  matchAndRewrite(triton::xpu::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    assert(cast<triton::xpu::ClusterLayoutAttr>(helper.getSrcLayout()) &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();
    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);

    // A reduce that lives inside the core-tiling slicing loop carries its
    // accumulation across iterations through the shared-memory loop cache
    // (see accumulatePartialReductions / loadReductionAndPackResult, guarded by
    // isNeedLoopCacheResult).  packResults() is register-only and has no such
    // cache, so taking the core-synchronous shortcut for a loop-carried reduce
    // silently returns the *current* iteration's partial instead of the
    // accumulated result: the value result degenerates to the value operand
    // and, for a multi-operand combine such as tl.argmax, the index result
    // degenerates to the index operand -- which is the loop index itself when
    // the index operand is the loop's out_range.  A consumer comparing against
    // that operand then folds to a constant.
    bool coreSync = helper.isCoreSynchronous();
    bool loopCarried = op.getLoopNum() > 1 && isNeedLoopCacheResult(op);

    // Init shared memory for mask.
    if (!targetInfo.getXPUIsUseMaskZero() && !(coreSync && !loopCarried)) {
      initSharedMemory(helper, rewriter, loc);
    }
    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    if (coreSync && !loopCarried) {
      LLVM_DEBUG(llvm::dbgs() << "\nisCoreSynchronous=True\n");

      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "\nisCoreSynchronous=False\n");

    // Then reduce across threads within a group.
    reduceWithinGroups(helper, accs, rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "\n After matchAndRewrite xpu::ReduceOp:"
    //              << op->getParentOfType<ModuleOp>() << "\n");
    // helper.dumpSMOffsets();
    return success();
  }

private:
  const xpu::TargetInfo &targetInfo;

  inline bool isNeedLoopCacheResult(triton::xpu::ReduceOp op) const {
    if (op.getAxis() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "\nDont Need SM Loop Result Cache");
      return false;
    }

    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto resultShape = resultTy.getShape();
      auto rank = resultShape.size();

      if (rank == 1) { // TODO[dyq]: add [op.getAxis() == None] logic
        LLVM_DEBUG(llvm::dbgs() << "\nNeed SM Loop Result Cache");
        return true;
      }

      LLVM_DEBUG(llvm::dbgs() << "\nDont Need SM Loop Result Cache");
      return false;
    }

    // scalar
    LLVM_DEBUG(llvm::dbgs() << "\nNeed SM Loop Result Cache");
    return true;
  }

  SmallVector<Value> initVals(triton::xpu::ReduceOp &op,
                              ConversionPatternRewriter &rewriter,
                              Location loc) const {

    auto naiveInit = [&](Type elemTy, int value) {
      Value val;
      if (elemTy.isInteger(1)) {
        val = int_val(1, value);
      } else if (elemTy.isInteger(8)) {
        val = int_val(8, value);
      } else if (elemTy.isInteger(16)) {
        val = int_val(16, value);
      } else if (elemTy.isInteger(32)) {
        val = i32_val(value);
      } else if (elemTy.isInteger(64)) {
        val = int_val(64, value);
      } else if (elemTy.isF16()) {
        val = f16_val(value);
      } else if (elemTy.isF32()) {
        val = f32_val(value);
      } else if (elemTy.isF64()) {
        val = f64_val(value);
      } else {
        LLVM_DEBUG(elemTy.dump());
        llvm_unreachable("[Reduce Init]: Unsupported ElemTy in Naive Init");
      }
      return val;
    };

    auto maxInit = [&](Type elemTy) {
      Value val;
      if (elemTy.isInteger(1)) {
        val = int_val(1, 1);
      } else if (elemTy.isInteger(8)) {
        val = int_val(8, INT8_MAX);
      } else if (elemTy.isInteger(16)) {
        val = int_val(16, INT16_MAX);
      } else if (elemTy.isInteger(32)) {
        val = i32_val(INT32_MAX);
      } else if (elemTy.isInteger(64)) {
        val = int_val(64, INT64_MAX);
      } else if (elemTy.isF16()) {
        val = f16_val(65504);
      } else if (elemTy.isF32()) {
        val = f32_val(FLT_MAX);
      } else if (elemTy.isF64()) {
        val = f64_val(DBL_MAX);
      } else {
        LLVM_DEBUG(elemTy.dump());
        llvm_unreachable("[Reduce Init]: Unsupported ElemTy in Max Init");
      }
      return val;
    };

    auto minInit = [&](Type elemTy) {
      Value val;
      if (elemTy.isInteger(1)) {
        val = int_val(1, 0);
      } else if (elemTy.isInteger(8)) {
        val = int_val(8, -INT8_MAX);
      } else if (elemTy.isInteger(16)) {
        val = int_val(16, -INT16_MAX);
      } else if (elemTy.isInteger(32)) {
        val = i32_val(-INT32_MAX);
      } else if (elemTy.isInteger(64)) {
        val = int_val(64, -INT64_MAX);
      } else if (elemTy.isF16()) {
        val = f16_val(-65504);
      } else if (elemTy.isF32()) {
        val = f32_val(-FLT_MAX);
      } else if (elemTy.isF64()) {
        val = f64_val(-DBL_MAX);
      } else {
        LLVM_DEBUG(elemTy.dump());
        llvm_unreachable("[Reduce Init]: Unsupported ElemTy in Min Init");
      }
      return val;
    };

    auto &combineBlock = op.getCombineOp().getBlocks().front();
    SmallVector<Operation *> blockArgDefOps;
    for (int i = 0; i < combineBlock.getArguments().size() / 2; ++i) {
      auto arg = combineBlock.getArgument(i);
      bool isBreak = false;
      for (auto user : arg.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<arith::AndIOp>([&](auto andIOp) {
              blockArgDefOps.emplace_back(andIOp);
              isBreak = true;
            })
            .Case<arith::OrIOp>([&](auto orIOp) {
              blockArgDefOps.emplace_back(orIOp);
              isBreak = true;
            })
            .Case<arith::XOrIOp>([&](auto xorIOp) {
              blockArgDefOps.emplace_back(xorIOp);
              isBreak = true;
            })
            .Case<arith::AddFOp>([&](auto addFOp) {
              blockArgDefOps.emplace_back(addFOp);
              isBreak = true;
            })
            .Case<arith::AddIOp>([&](auto addIOp) {
              blockArgDefOps.emplace_back(addIOp);
              isBreak = true;
            })
            .Case<arith::SubFOp>([&](auto subFOp) {
              blockArgDefOps.emplace_back(subFOp);
              isBreak = true;
            })
            .Case<arith::SubIOp>([&](auto subIOp) {
              blockArgDefOps.emplace_back(subIOp);
              isBreak = true;
            })
            .Case<arith::MulFOp>([&](auto mulFOp) {
              blockArgDefOps.emplace_back(mulFOp);
              isBreak = true;
            })
            .Case<arith::MulIOp>([&](auto mulIOp) {
              blockArgDefOps.emplace_back(mulIOp);
              isBreak = true;
            })
            .Case<arith::DivFOp>([&](auto divFOp) {
              blockArgDefOps.emplace_back(divFOp);
              isBreak = true;
            })
            .Case<arith::DivSIOp>([&](auto divSIOp) {
              blockArgDefOps.emplace_back(divSIOp);
              isBreak = true;
            })
            .Case<arith::DivUIOp>([&](auto divUIOp) {
              blockArgDefOps.emplace_back(divUIOp);
              isBreak = true;
            })
            .Case<arith::MaxNumFOp>([&](auto maxNumFOp) {
              blockArgDefOps.emplace_back(maxNumFOp);
              isBreak = true;
            })
            .Case<arith::MaxSIOp>([&](auto maxSIOp) {
              blockArgDefOps.emplace_back(maxSIOp);
              isBreak = true;
            })
            .Case<arith::MaxUIOp>([&](auto maxUIOp) {
              blockArgDefOps.emplace_back(maxUIOp);
              isBreak = true;
            })
            .Case<arith::MinNumFOp>([&](auto minNumFOp) {
              blockArgDefOps.emplace_back(minNumFOp);
              isBreak = true;
            })
            .Case<arith::MinSIOp>([&](auto minSIOp) {
              blockArgDefOps.emplace_back(minSIOp);
              isBreak = true;
            })
            .Case<arith::MinUIOp>([&](auto minUIOp) {
              blockArgDefOps.emplace_back(minUIOp);
              isBreak = true;
            })
            .Case<arith::MaximumFOp>([&](auto maximumFOp) {
              blockArgDefOps.emplace_back(maximumFOp);
              isBreak = true;
            })
            .Case<arith::MinimumFOp>([&](auto minimumFOp) {
              blockArgDefOps.emplace_back(minimumFOp);
              isBreak = true;
            })
            .Case<arith::SelectOp>([&](auto selectOp) {
              // arith::SelectOp appears as the index combiner in argmax/argmin
              // reduce patterns (e.g. tl.max with return_indices=True).
              blockArgDefOps.emplace_back(selectOp);
              isBreak = true;
            })
            .Case<arith::CmpFOp>([&](auto cmpFOp) {
              if (cmpFOp.getPredicate() == arith::CmpFPredicate::OGT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::OGE ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::OLT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::OLE ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::UGT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::UGE ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::ULT ||
                  cmpFOp.getPredicate() == arith::CmpFPredicate::ULE) {
                blockArgDefOps.emplace_back(cmpFOp);
                isBreak = true;
              }
            })
            .Case<arith::CmpIOp>([&](auto cmpIOp) {
              if (cmpIOp.getPredicate() == arith::CmpIPredicate::slt ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::sle ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::sgt ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::sge ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::ult ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::ule ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::ugt ||
                  cmpIOp.getPredicate() == arith::CmpIPredicate::uge) {
                blockArgDefOps.emplace_back(cmpIOp);
                isBreak = true;
              }
            });
        if (isBreak) {
          break;
        }
      }
    }

    auto types = op.getInputTypes();
    assert(blockArgDefOps.size() == types.size() &&
           "[Reduce Init]: BlockArgDefOps Size() != Types Size");
    SmallVector<Value> vals;
    for (int i = 0; i < blockArgDefOps.size(); ++i) {
      auto elemTy = getElementTypeOrSelf(types[i]);
      if (auto vecTy = dyn_cast<mlir::VectorType>(elemTy)) {
        elemTy = vecTy.getElementType();
      }
      Value val = naiveInit(elemTy, 0);
      auto blockArgDefOp = blockArgDefOps[i];
      TypeSwitch<Operation *>(blockArgDefOp)
          .Case<arith::AndIOp>([&](auto andIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::OrIOp>([&](auto orIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::XOrIOp>([&](auto xorIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::AddFOp>([&](auto addFOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::AddIOp>([&](auto addIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::SubFOp>([&](auto subFOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::SubIOp>([&](auto subIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::MulFOp>([&](auto mulFOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::MulIOp>([&](auto mulIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::DivFOp>([&](auto divFOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::DivSIOp>(
              [&](auto divSIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::DivUIOp>(
              [&](auto divUIOp) { val = naiveInit(elemTy, 1); })
          .Case<arith::MaxNumFOp>(
              [&](auto maxNumFOp) { val = minInit(elemTy); })
          .Case<arith::MaxSIOp>([&](auto maxSIOp) { val = minInit(elemTy); })
          .Case<arith::MaxUIOp>(
              [&](auto maxUIOp) { val = naiveInit(elemTy, 0); })
          .Case<arith::MinNumFOp>(
              [&](auto minNumFOp) { val = maxInit(elemTy); })
          .Case<arith::MinSIOp>([&](auto minSIOp) { val = maxInit(elemTy); })
          .Case<arith::MinUIOp>([&](auto minUIOp) { val = maxInit(elemTy); })
          .Case<arith::MaximumFOp>(
              [&](auto maximumFOp) { val = minInit(elemTy); })
          .Case<arith::MinimumFOp>(
              [&](auto minimumFOp) { val = maxInit(elemTy); })
          .Case<arith::SelectOp>([&](auto selectOp) {
            // Index combiner in argmax/argmin patterns; init to 0.
            val = naiveInit(elemTy, 0);
          })
          .Case<arith::CmpFOp>([&](auto cmpFOp) {
            if (cmpFOp.getPredicate() == arith::CmpFPredicate::OGT ||
                cmpFOp.getPredicate() == arith::CmpFPredicate::OGE ||
                cmpFOp.getPredicate() == arith::CmpFPredicate::UGT ||
                cmpFOp.getPredicate() == arith::CmpFPredicate::UGE) {
              val = minInit(elemTy);
            } else if (cmpFOp.getPredicate() == arith::CmpFPredicate::OLT ||
                       cmpFOp.getPredicate() == arith::CmpFPredicate::OLE ||
                       cmpFOp.getPredicate() == arith::CmpFPredicate::ULT ||
                       cmpFOp.getPredicate() == arith::CmpFPredicate::ULE) {
              val = maxInit(elemTy);
            } else {
              llvm_unreachable(
                  "[Reduce Init]: Unsupported CmpFPredicate in CmpFOp");
            }
          })
          .Case<arith::CmpIOp>([&](auto cmpIOp) {
            if (cmpIOp.getPredicate() == arith::CmpIPredicate::sgt ||
                cmpIOp.getPredicate() == arith::CmpIPredicate::sge ||
                cmpIOp.getPredicate() == arith::CmpIPredicate::ugt ||
                cmpIOp.getPredicate() == arith::CmpIPredicate::uge) {
              val = minInit(elemTy);
            } else if (cmpIOp.getPredicate() == arith::CmpIPredicate::slt ||
                       cmpIOp.getPredicate() == arith::CmpIPredicate::sle ||
                       cmpIOp.getPredicate() == arith::CmpIPredicate::ult ||
                       cmpIOp.getPredicate() == arith::CmpIPredicate::ule) {
              val = maxInit(elemTy);
            } else {
              llvm_unreachable(
                  "[Reduce Init]: Unsupported CmpFPredicate in CmpIOp");
            }
          })
          .Default([&](auto defaultOp) {
            LLVM_DEBUG(defaultOp->dump());
            llvm_unreachable(
                "[Reduce Init]: Unsupported Operation in Reduce Output");
          });
      vals.emplace_back(val);
    }

    return vals;
  }

  void initSharedMemory(ReduceOpHelper &helper,
                        ConversionPatternRewriter &rewriter,
                        Location loc) const {
    // Init shared memory
    ConversionPatternRewriter::InsertionGuard guard(
        rewriter); // save reduceOpPtr to restore
    auto reduceOpPtr = rewriter.saveInsertionPoint();
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    auto func = op->template getParentOfType<LLVM::LLVMFuncOp>();
    rewriter.setInsertionPointToStart(&(func.front()));
    // Compute a shared memory base per operand.
    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    SmallVector<Value> vals = initVals(op, rewriter, loc);

    auto smemShape = helper.getXPUScratchConfig();
    auto [smemBases, loopResCacheSmemBases] =
        getSmemBases(op, product<unsigned>(smemShape), rewriter);
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      Value initPtr =
          gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i], coreId);
      store_sm(vals[i], initPtr);
    }
    xpu_barrier();
    rewriter.restoreInsertionPoint(reduceOpPtr); // restore reduceOpPtr
  }

  void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                  SmallVector<Value> &acc, ValueRange cur, bool isFirst) const {
    if (isFirst) {
      acc = SmallVector<Value>(cur.begin(), cur.end());
      return;
    }

    // Create a new copy of the reduce block, and inline it
    Block *currentBlock = rewriter.getBlock();
    Region &parent = *currentBlock->getParent();
    rewriter.cloneRegionBefore(combineOp, &parent.front());
    auto &newReduce = parent.front();
    auto returnOp =
        dyn_cast<triton::xpu::ReduceReturnOp>(newReduce.getTerminator());

    llvm::SmallVector<Value> combineArgs(2 * acc.size());
    for (unsigned i = 0; i < acc.size(); ++i) {
      combineArgs[i] = acc[i];
      combineArgs[acc.size() + i] = cur[i];
    }

    rewriter.inlineBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
                               combineArgs);

    auto results = returnOp.getResult();
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }

    // Delete the terminator, which is no longer used
    rewriter.eraseOp(returnOp);
  }

  LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) const {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }

  LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) const {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return LLVM::ICmpPredicate::eq;
    case arith::CmpIPredicate::ne:
      return LLVM::ICmpPredicate::ne;
    case arith::CmpIPredicate::sgt:
      return LLVM::ICmpPredicate::sgt;
    case arith::CmpIPredicate::sge:
      return LLVM::ICmpPredicate::sge;
    case arith::CmpIPredicate::slt:
      return LLVM::ICmpPredicate::slt;
    case arith::CmpIPredicate::sle:
      return LLVM::ICmpPredicate::sle;
    case arith::CmpIPredicate::ugt:
      return LLVM::ICmpPredicate::ugt;
    case arith::CmpIPredicate::uge:
      return LLVM::ICmpPredicate::uge;
    case arith::CmpIPredicate::ult:
      return LLVM::ICmpPredicate::ult;
    case arith::CmpIPredicate::ule:
      return LLVM::ICmpPredicate::ule;
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }

  void calculate(ConversionPatternRewriter &rewriter, const Location &loc,
                 Operation *op, Value &acc, const Value &cur) const {
    TypeSwitch<Operation *>(op)
        .Case<arith::AddFOp>([&](auto addfOp) { acc = fadd(acc, cur); })
        .Case<arith::MulFOp>([&](auto mulfOp) { acc = fmul(acc, cur); })
        .Case<arith::MaxNumFOp>([&](auto maxfOp) { acc = fmax(acc, cur); })
        .Case<arith::MinNumFOp>([&](auto minfOp) { acc = fmin(acc, cur); })
        .Case<arith::OrIOp>([&](auto oriOp) { acc = or_(acc, cur); })
        .Case<arith::XOrIOp>([&](auto xoriOp) { acc = xor_(acc, cur); })
        .Case<arith::AndIOp>([&](auto andOp) { acc = and_(acc, cur); })
        .Case<arith::CmpFOp>([&](auto cmpfOp) {
          acc = rewriter.create<LLVM::FCmpOp>(
              loc, acc.getType(),
              ArithCmpFPredicateToLLVM(cmpfOp.getPredicate()), acc, cur);
        })
        .Default([&](auto defaultOp) {
          LLVM_DEBUG(defaultOp->dump());
          llvm_unreachable("[Vectorization]: Unsupported Operation Type "
                           "To VecType in Reduce");
        });
  }

  // ---- region-interpreting combine (3.1b prototype) ------------------------
  //
  // `calculate` applies exactly *one* op per output -- the defining op of that
  // output's return value (`helper.getReturnDefOps()`). That equals the combine
  // region only when the combine is separable: one op per output, no data flow
  // between outputs. A welford combine is not separable (the mean update reads
  // the count), so no whitelist extension can make the getReturnDefOps path
  // correct for it; it would silently apply the return's defining op alone.
  //
  // The region also cannot just be cloned the way `accumulate` does it: on the
  // vector path arith-to-llvm re-types the cloned arith ops back to f32 (see
  // the note in reduceWithinThreads). So walk the region and emit LLVM ops
  // directly, which needs no change between scalar and vector<NxT> operands.
  //
  // The one-day-on / off-again history and the current default live with the
  // predicate in VectorizabilityAnalysis.h, because the analysis, Vectorize's
  // retyping and this lowering all have to read the same answer.
  //
  // What was blamed on this file for a while (the within-core fold below losing
  // the lane-0 contribution of most cores on a vectorized welford,
  // nondeterministically) was not a defect here: `collapseVectorsJointly` is
  // exact, and the loss came from a write-after-read hazard on an LM buffer
  // that MemoryInplace had folded across three loads. Fixed on the load side;
  // see the fence in XPULoadOpConversion and findings 1.74. Multi-operand
  // combines are simply the first shape that keeps three loaded vectors live at
  // once, which is what made that reuse observable.
  static bool regionCombineEnabled() {
    return mlir::triton::xpu::reduceCombineRegionEnabled();
  }

  // Must stay in sync with the TypeSwitch in emitCombineOp.
  static bool isSupportedCombineOp(Operation *op) {
    // Bodies live in CombineEmit.cpp now, shared with the scan lowering.
    return mlir::triton::xpu::isSupportedCombineOp(op);
  }

  // All rejection happens here, before anything is emitted, so a region we
  // cannot handle leaves no half-built IR behind.
  // A constant the region reads is usable at any stage only if one scalar value
  // describes it: the same region is replayed on vector<NxT> accumulators
  // (within-core) and on scalars (across cores), so the constant has to be
  // narrowable and widenable at will.
  static bool isUniformConstant(Value v) {
    return mlir::triton::xpu::isUniformConstant(v);
  }

  bool canInterpretCombine(Region &combine) const {
    return mlir::triton::xpu::canInterpretCombine(combine);
  }

  // Widen a scalar to `width` lanes *keeping its own element type*: a cmpf mask
  // must become vector<Nxi1>, not vector<NxF>. Deriving the whole vector type
  // from a neighbouring operand gets that wrong.
  Value splatToWidth(ConversionPatternRewriter &rewriter, Location loc, Value v,
                     int64_t width) const {
    return mlir::triton::xpu::splatToWidth(rewriter, loc, v, width);
  }

  // Always emit the constant scalar first and widen afterwards: the region may
  // carry it as vector<NxT> (Vectorize rematerializes it that way) while the
  // stage being lowered is scalar, or the other way round.
  Value materializeConstant(ConversionPatternRewriter &rewriter, Location loc,
                            arith::ConstantOp cstOp, int64_t width) const {
    return mlir::triton::xpu::materializeConstant(rewriter, loc, cstOp, width);
  }

  // `width` is the lane count of the stage: > 0 while combining vector
  // accumulators, 0 once the values are scalars (across-core, loop cache).
  Value emitCombineOp(ConversionPatternRewriter &rewriter, Location loc,
                      Operation *op, SmallVector<Value> &args,
                      int64_t width) const {
    return mlir::triton::xpu::emitCombineOp(rewriter, loc, op, args, width);
  }

  // One combine step: acc = combine(acc, cur), emitted from the region itself.
  // `canInterpretCombine` must have accepted the region first.
  void interpretCombine(ConversionPatternRewriter &rewriter, Location loc,
                        Region &combine, SmallVector<Value> &acc,
                        ValueRange cur) const {
    mlir::triton::xpu::interpretCombine(rewriter, loc, combine, acc, cur);
  }

  // Horizontal collapse to lane 0, done *jointly* across outputs: one combine
  // step consumes lane i of every accumulator. accmulateWithinVector collapses
  // each output on its own, which again only holds for a separable combine.
  void collapseVectorsJointly(ConversionPatternRewriter &rewriter, Location loc,
                              Region &combine,
                              SmallVector<Value> &accVecs) const {
    auto laneOf = [&](Value v, int64_t lane) {
      return extract_element(getElementTypeOrSelf(v.getType()), v,
                             i32_val(lane));
    };
    int64_t vecSize = cast<VectorType>(accVecs[0].getType()).getNumElements();
    for (Value v : accVecs) {
      assert(cast<VectorType>(v.getType()).getNumElements() == vecSize &&
             "accumulators of one reduce disagree on vector width");
      (void)v;
    }

    // Network path -- the same Hillis-Steele lane-prefix the scan lowering
    // uses: fold the k-lane-shifted vector into the running one via
    // interpretCombine, and for any combine whose identity is not the shift's
    // zero fill (min/max/...), keep lane<k at its pre-step value with the vvor
    // masked select. The reduction is lane (vecSize-1) of the resulting
    // inclusive prefix. Restricted to what the packed shift + masked select
    // model: any interpretable combine at one register width; the mask is
    // float-only, so plain add also runs for integer (svsllp_s, maskless --
    // 0 is its identity), but an integer min/max keeps the serial replay
    // below, as does a mixed-width or otherwise un-interpretable region.
    bool netEligible = combine.getBlocks().size() == 1 &&
                       mlir::triton::xpu::canInterpretCombine(combine);
    Type netElemTy;
    bool plainAdd = false;
    if (netEligible) {
      Block &body = combine.front();
      netElemTy = getElementTypeOrSelf(accVecs[0].getType());
      for (Value v : accVecs)
        if (getElementTypeOrSelf(v.getType()) != netElemTy)
          netEligible = false;
      for (Value arg : body.getArguments())
        if (getElementTypeOrSelf(arg.getType()) != netElemTy)
          netEligible = false;
      if (std::distance(body.without_terminator().begin(),
                        body.without_terminator().end()) == 1)
        plainAdd = isa<arith::AddFOp, arith::AddIOp>(body.front());
      if (!plainAdd && !isa<FloatType>(netElemTy))
        netEligible = false; // masked (non-add) collapse needs the float vvor;
                             // i32 max/min stay serial (network net-negative,
                             // KTC-148 asm A/B). Plain int add still nets below.
    }
    if (netEligible) {
      // Shared masked lane-prefix network (with the scan lowering); the total
      // is lane (vecSize-1) of the resulting inclusive prefix.
      mlir::triton::xpu::emitMaskedLanePrefix(rewriter, loc, combine, accVecs,
                                              /*needsMask=*/!plainAdd);
      // lane (vecSize-1) of the inclusive prefix holds the reduction total.
      for (Value &v : accVecs)
        v = extract_element(netElemTy, v, i32_val(vecSize - 1));
      return;
    }

    SmallVector<Value> acc;
    for (Value v : accVecs)
      acc.push_back(laneOf(v, 0));
    for (int64_t lane = 1; lane < vecSize; ++lane) {
      SmallVector<Value> cur;
      for (Value v : accVecs)
        cur.push_back(laneOf(v, lane));
      interpretCombine(rewriter, loc, combine, acc, cur);
    }
    accVecs = acc;
  }

  void accmulateWithinVector(ConversionPatternRewriter &rewriter,
                             const Location &loc, Operation *op,
                             Value &accVec) const {
    auto accTy = cast<VectorType>(accVec.getType());
    size_t vecSize = accTy.getNumElements();
    Type elemTy = getElementTypeOrSelf(accTy);
    Value acc = extract_element(elemTy, accVec, i32_val(0));
    for (size_t i = 1; i < vecSize; ++i) {
      auto cur = extract_element(elemTy, accVec, i32_val(i));
      calculate(rewriter, loc, op, acc, cur);
    }
    accVec = acc;
  }

  void accmulateNaive(ConversionPatternRewriter &rewriter, const Location &loc,
                      SmallVector<Operation *> &ops, SmallVector<Value> &accs,
                      SmallVector<Value> &curs, bool isFirst) const {
    for (unsigned i = 0; i < accs.size(); ++i) {
      if (isFirst) {
        accs[i] = curs[i];
      } else {
        calculate(rewriter, loc, ops[i], accs[i], curs[i]);
      }
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::xpu::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto values = unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }

    return srcValues;
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> _offsets =
        mlir::LLVM::XPU::emitOffsetForLayoutXPU(helper.getSrcLayout(),
                                                operandType);

    // Thread X might hold the same input value in two registers.  Get the
    // indices in `offsets` that hold unique values, and only accumualte over
    // those.
    SmallVector<SmallVector<unsigned>> offsets(_offsets);
    auto shape = operandType.getShape();
    auto layout =
        cast<triton::xpu::ClusterLayoutAttr>(operandType.getEncoding());
    unsigned rowsPerCore = layout.getSizePerCore()[0];
    // The emitted per-core offsets are col-major only when the layout order
    // puts the row dim first (order[0] == 0). In that case, when a core holds
    // multiple rows, transpose col-major -> row-major to match the row-major
    // register slots that the memory ops fill. When order == [1, 0] (order[0]
    // == 1) the offsets are ALREADY row-major, so skip the transpose to avoid a
    // double-flip.
    auto order = layout.getOrder();
    bool colMajorEmitted = (order.size() == 2 && order[0] == 0);
    bool coreDealMultiRows =
        (shape.size() == 2 && rowsPerCore > 1 && colMajorEmitted);
    if (coreDealMultiRows) {
      // Col major to row major
      unsigned col = shape[1];
      unsigned row = offsets.size() / col;
      assert(offsets.size() % col == 0 &&
             "Offsets size is not equal to col mul row");
      for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
          offsets[i * col + j] = _offsets[j * row + i];
        }
      }
    }

    unsigned srcElems = getTotalElemsPerThread(operandType);
    auto *combineOp = &op.getCombineOp();
    auto srcIndices =
        emitIndices(op.getLoc(), rewriter, targetInfo, helper.getSrcLayout(),
                    operandType, op.getLoopIndex());

    // LLVM22: When the reduce is vectorized, the cloned arith ops in the
    // combine region get re-typed to scalar (f32) by arith-to-llvm patterns
    // even though the cloned ops carry vector<NxT> result types, producing
    // invalid llvm.fadd ops with mismatched operand/result types. To avoid
    // that, emit the combine directly as LLVM ops via accmulateNaive when
    // vectorized (mirrors what storeCoreReduceToSharedMemory and
    // accumulatePartialReductions already do for the same reason).
    bool vectorized = helper.isVectorized();
    SmallVector<Operation *> returnDefOpsForAccum;
    if (vectorized)
      returnDefOpsForAccum = helper.getReturnDefOps();
    // Decided once per reduce so the three combine sites below cannot end up on
    // different paths within one op.
    bool useRegion =
        vectorized && regionCombineEnabled() && canInterpretCombine(*combineOp);

    // reduce within threads
    for (int i = 0; i < offsets.size(); ++i) {
      SmallVector<unsigned> key = offsets[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      if (vectorized) {
        if (isFirst) {
          accs[key] =
              SmallVector<Value>(srcValues[i].begin(), srcValues[i].end());
        } else if (useRegion) {
          interpretCombine(rewriter, op.getLoc(), *combineOp, accs[key],
                           srcValues[i]);
        } else {
          accmulateNaive(rewriter, op.getLoc(), returnDefOpsForAccum, accs[key],
                         srcValues[i], /*isFirst=*/false);
        }
      } else {
        accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      }
      if (isFirst)
        indices[key] = srcIndices[i];
    }

    // Accumulate within Vector
    if (useRegion) {
      for (auto &it : accs)
        collapseVectorsJointly(rewriter, op.getLoc(), *combineOp, it.second);
    } else if (helper.isVectorized()) {
      SmallVector<Operation *> returnDefOps = helper.getReturnDefOps();
      for (auto &it : accs) {
        SmallVector<Value> &accVecs = it.second;
        assert(accVecs.size() == returnDefOps.size() &&
               "accVecs.size() !=returnDefOps.size()");
        for (unsigned i = 0; i < returnDefOps.size(); ++i) {
          accmulateWithinVector(rewriter, op.getLoc(), returnDefOps[i],
                                accVecs[i]);
        }
      }
    }
  }

  void storeCoreReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);

    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      for (unsigned i = 0; i < (op.getNumOperands() - 1);
           ++i) { // skip loopIndex
        auto elemTy = getElementType(op, i);
        Value writePtr =
            gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i], coreId);
        store_sm(acc[i], writePtr);

        // Dump smem[i][0-63] Data Which Will Be Accmulated By Core0
        if (false) { // modify true to enable dump
          for (int core_id_offset = 0; core_id_offset < 64; ++core_id_offset) {
            Value loadPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                                smemBases[i], i32_val(core_id_offset));
            Value loadVal = load_sm(elemTy, loadPtr);
            std::string calFunc = "_ZN3xpu10printFloatEfi";
            ValueRange operandValueRange(
                {loadVal, i32_val(i * 100 + core_id_offset)});
            mlir::LLVM::XPU::createDeviceCall(calFunc, rewriter, op,
                                              operandValueRange, loc);
          }
        }
      }
    }
  }

  mlir::RewriterBase::InsertPoint
  getPreviousInsertionPoint(PatternRewriter &rewriter) const {
    auto currentInsertionPoint = rewriter.getInsertionPoint();
    auto oldInsertionPoint = currentInsertionPoint;
    Block *currentBlock_0 = rewriter.getInsertionBlock();
    // Move the iterator one step backward
    if (currentInsertionPoint != currentBlock_0->begin()) {
      --currentInsertionPoint;
    }
    // Set the insertion point to the adjusted position and save it
    rewriter.setInsertionPoint(currentBlock_0, currentInsertionPoint);
    auto startInsertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(currentBlock_0, oldInsertionPoint);
    return startInsertionPoint;
  }

  void createCondBr(PatternRewriter &rewriter, Location loc, Value condition,
                    Block *&trueDest, Block *&falseDest) const {
    Block *currentBlock = rewriter.getInsertionBlock();
    falseDest = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    trueDest = rewriter.createBlock(falseDest);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, condition, trueDest, falseDest);
    rewriter.setInsertionPointToStart(trueDest);

    rewriter.create<LLVM::BrOp>(loc, falseDest);
    rewriter.setInsertionPointToStart(falseDest);
  }

  void moveOpsBetweenInsertionPoints(ReduceOpHelper &helper,
                                     PatternRewriter &rewriter, Block *trueDest,
                                     Block *falseDest, Block::iterator start,
                                     Block::iterator end) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();

    auto terminator = &*(trueDest->begin());

    // LLVM_DEBUG(llvm::dbgs() << "\n Before moveBefore:" <<
    // op->getParentOfType<ModuleOp>()
    //              << "\n");

    // LLVM_DEBUG(llvm::dbgs() << "\n movedOp: " << "\n");
    start++; // skip the previous op
    while (start != end) {
      Operation *op = &*start++;
      //   op->dump();
      op->moveBefore(trueDest, trueDest->end());
    }

    // deal the previous op
    Operation *endOp = &*end;
    endOp->moveBefore(trueDest, trueDest->end());
    terminator->moveBefore(trueDest, trueDest->end());
  }

  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   SmallVector<Value> &loopResCacheSmemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    auto srcLayout = helper.getSrcLayout();
    Location loc = op.getLoc();
    unsigned groupSizeInt = helper.getIntraGroupSizeWithUniqueData();
    unsigned operandNum = op.getInputTypes().size();

    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value groupSize = i32_val(groupSizeInt);
    Value coreIdInGroup = urem(coreId, groupSize);
    Value groupId = udiv(coreId, groupSize);
    Value groupSkip = mul(groupId, groupSize);
    // Value laneId = add(mul(groupId, groupSize), coreIdInGroup);
    Value zero = i32_val(0);
    Value coreIdInGroupZero = icmp_eq(coreIdInGroup, zero);

    auto startInsertionPoint = getPreviousInsertionPoint(rewriter);

    SmallVector<Value> acc(operandNum); // skip loopIndex
    SmallVector<SmallVector<Value>> readValues(groupSizeInt);
    for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      Value groupBasePtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                               smemBases[i], groupSkip);
      for (unsigned readOffset = 0; readOffset < groupSizeInt; ++readOffset) {
        Value readPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                            groupBasePtr, i32_val(readOffset));
        readValues[readOffset].push_back(load_sm(elemTy, readPtr));
      }
    }

    SmallVector<Operation *> returnDefOps = helper.getReturnDefOps();
    // The values read back from smem are scalars (getElementType strips the
    // vector), so the interpreter runs on scalar operands here.
    bool useRegion = helper.isVectorized() && regionCombineEnabled() &&
                     canInterpretCombine(op.getCombineOp());
    for (auto [i, v] : llvm::enumerate(readValues)) {
      if (useRegion) {
        if (i == 0)
          acc = SmallVector<Value>(v.begin(), v.end());
        else
          interpretCombine(rewriter, loc, op.getCombineOp(), acc, v);
      } else if (helper.isVectorized()) {
        accmulateNaive(rewriter, loc, returnDefOps, acc, v, i == 0);
      } else {
        accumulate(rewriter, op.getCombineOp(), acc, v, i == 0);
      }
    }

    SmallVector<Value> writePtrs(operandNum);   // skip loopIndex
    for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      // TODO[dyq]: check writeOffset == i32_val(0)
      Value writeOffset = groupSkip;
      writePtrs[i] = gep(ptr_ty(rewriter.getContext(), 2), elemTy, smemBases[i],
                         /*writeOffset*/ writeOffset);
    }

    for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
      store_sm(acc[i], writePtrs[i]);
    }

    // reduce calcution(cur loop) finsh, now move it to the trueDest
    auto endInsertionPoint = getPreviousInsertionPoint(rewriter);

    Block *trueDest = nullptr;
    Block *falseDest = nullptr;
    createCondBr(rewriter, op->getLoc(), coreIdInGroupZero, trueDest,
                 falseDest);

    moveOpsBetweenInsertionPoints(helper, rewriter, trueDest, falseDest,
                                  startInsertionPoint.getPoint(),
                                  endInsertionPoint.getPoint());

    if (isNeedLoopCacheResult(op)) {
      // reduce with prev loopResult
      Value laneId = coreId;
      Value zero = i32_val(0);
      Value laneZero = icmp_eq(laneId, zero);
      Value loopIndex = op.getLoopIndex();
      Value loopNonZero = icmp_ne(loopIndex, zero);
      Value cond = and_(laneZero, loopNonZero);

      Block *loopReduceTrueDest = nullptr;
      Block *loopReduceFalseDest = nullptr;
      createCondBr(rewriter, op->getLoc(), cond, loopReduceTrueDest,
                   loopReduceFalseDest);

      auto curInsertionPoint = rewriter.getInsertionPoint();
      Block *curBlock = rewriter.getInsertionBlock();

      rewriter.setInsertionPointToStart(loopReduceTrueDest);
      SmallVector<Value> curResSmemValues(operandNum);
      SmallVector<Value> loopResCacheSmemValues(operandNum);
      for (unsigned i = 0; i < (operandNum); ++i) { // skip loopIndex
        auto elemTy = getElementType(op, i);
        curResSmemValues[i] = load_sm(elemTy, smemBases[i]);
        loopResCacheSmemValues[i] = load_sm(elemTy, loopResCacheSmemBases[i]);
      }

      SmallVector<Operation *> returnDefOps = helper.getReturnDefOps();
      if (useRegion) {
        interpretCombine(rewriter, loc, op.getCombineOp(), curResSmemValues,
                         loopResCacheSmemValues);
      } else if (helper.isVectorized()) {
        accmulateNaive(rewriter, loc, returnDefOps, loopResCacheSmemValues,
                       loopResCacheSmemValues, false);
      } else {
        accumulate(rewriter, op.getCombineOp(), curResSmemValues,
                   loopResCacheSmemValues, false);
      }

      // store the final result
      for (unsigned i = 0; i < operandNum; ++i) { // skip loopIndex
        store_sm(curResSmemValues[i], smemBases[i]);
      }

      rewriter.setInsertionPoint(curBlock, curInsertionPoint);
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  SmallVector<Value> &loopResCacheSmemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    unsigned groupSizeInt = helper.getIntraGroupSizeWithUniqueData();

    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value groupSize = i32_val(groupSizeInt);
    Value groupId = udiv(coreId, groupSize);
    Value groupSkip = mul(groupId, groupSize);
    SmallVector<Value> results(op.getNumOperands() - 1);       // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        // nd-tensor where n >= 1
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices =
            emitIndices(loc, rewriter, targetInfo, resultLayout, resultTy,
                        op.getLoopIndex());
        auto resultShape = resultTy.getShape();
        // getShapePerCTATile was removed upstream; resultCTATile usage is
        // already commented out below, so drop this dead line.
        // auto resultCTATile = getShapePerCTATile(resultLayout, resultShape);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          //   readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          //   for (size_t resultIdx = 0, resultDim = resultShape.size();
          //        resultIdx < resultDim; ++resultIdx) {
          //     auto smemIdx = resultIdx < op.getAxis() ? resultIdx :
          //     resultIdx
          //     + 1; if (resultCTATile[resultIdx] > smemShape[smemIdx] ||
          //         resultShape[resultIdx] > smemShape[smemIdx]) {
          //       // When srcShape smaller then src sizePerThread, only
          //       srcShape
          //       // elements is accumulated in smem. Modulo smemShape
          //       effectively
          //       // replicates srcShape elements to src sizePerThread.
          //       readIdx[smemIdx] =
          //           urem(readIdx[smemIdx], i32_val(smemShape[smemIdx]));
          //     }
          //   }

          Value readOffset = groupSkip;
          //   Value readOffset = linearize(rewriter, loc, readIdx, smemShape,
          //   smemOrder);
          Value readPtr = gep(ptr_ty(rewriter.getContext(), 2), elemTy,
                              smemBases[i], readOffset);
          resultVals[j] = load_sm(elemTy, readPtr);
        }

        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load_sm(elemTy, smemBases[i]);
        // save reduce result in cur loop
        if (isNeedLoopCacheResult(op))
          store_sm(results[i], loopResCacheSmemBases[i]);
      }
    }

    rewriter.replaceOp(op, results);
  }

  // Reduce across threads within each group.
  void
  reduceWithinGroups(ReduceOpHelper &helper,
                     std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                     ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();

    // unsigned sizeIntraGroups = helper.getIntraGroupSizeWithUniqueData();
    // unsigned threadOffsetOnReductionAxis =
    // helper.getThreadOffsetOnReductionAxis(); LLVM_DEBUG(llvm::dbgs() <<
    // "\nsizeIntraGroups=" << sizeIntraGroups << "\n"); LLVM_DEBUG(llvm::dbgs()
    // <<
    // "\nthreadOffsetOnReductionAxis="
    //              << threadOffsetOnReductionAxis << "\n");

    // Compute a shared memory base per operand.
    auto smemShape = helper.getXPUScratchConfig();

    auto [smemBases, loopResCacheSmemBases] =
        getSmemBases(op, product<unsigned>(smemShape), rewriter);

    storeCoreReduceToSharedMemory(helper, accs, smemBases, rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "\n After storeCoreReduceToSharedMemory:"
    //              << op->getParentOfType<ModuleOp>() << "\n");

    xpu_barrier();

    accumulatePartialReductions(helper, smemBases, loopResCacheSmemBases,
                                rewriter);
    // LLVM_DEBUG(llvm::dbgs() << "\n After accumulatePartialReductions:"
    //              << op->getParentOfType<ModuleOp>() << "\n");

    xpu_barrier();

    loadReductionAndPackResult(helper, smemShape, smemBases,
                               loopResCacheSmemBases, rewriter);

    // llvm_unreachable("Not Supported");
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::xpu::ReduceOp op = helper.getXPUOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results((op.getNumOperands() - 1));     // skip loopIndex
    for (unsigned i = 0; i < (op.getNumOperands() - 1); ++i) { // skip loopIndex
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        auto resultLayout = cast<SliceEncodingAttr>(resultTy.getEncoding());
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            mlir::LLVM::XPU::emitOffsetForLayoutXPU(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = packLLElements(loc, getTypeConverter(), resultVals,
                                    rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }
};

} // namespace

void mlir::triton::xpu::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<XPUReduceOpConversion>(typeConverter, targetInfo, benefit);
}
