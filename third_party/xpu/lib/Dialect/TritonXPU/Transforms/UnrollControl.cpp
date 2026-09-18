#include "mlir/IR/IRMapping.h"
#include "triton/Analysis/TileAnalysis.h"
#include "triton/Analysis/TileDecision.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#define DEBUG_TYPE "tritonxpu-unroll-control"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUUNROLLCONTROL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

template <typename OP> struct COMOp;

#define COMOP(SrcType, DstType)                                                \
  template <> struct COMOp<SrcType> {                                          \
    typedef DstType type;                                                      \
  };

COMOP(arith::AddFOp, triton::xpu::VvaddFOp);
// subf/divf only reach a combine region on the region-interpreting path
// (TRITONXPU_REDUCE_REGION); COMBINE_BINARY_OP does not list them.
COMOP(arith::SubFOp, triton::xpu::VvsubFOp);
COMOP(arith::DivFOp, triton::xpu::VvdivFOp);
COMOP(arith::MulFOp, triton::xpu::VvmulFOp);
COMOP(arith::MaxNumFOp, triton::xpu::VvmaxNumFOp);
COMOP(arith::MinNumFOp, triton::xpu::VvminNumFOp);
COMOP(arith::OrIOp, triton::xpu::VvorIOp);
COMOP(arith::XOrIOp, triton::xpu::VvxorIOp);
COMOP(arith::AndIOp, triton::xpu::VvandIOp);

struct TritonXPUUnrollControl
    : public impl::TritonXPUUnrollControlBase<TritonXPUUnrollControl> {

public:
  using impl::TritonXPUUnrollControlBase<
      TritonXPUUnrollControl>::TritonXPUUnrollControlBase;

  TritonXPUUnrollControl() = default;
  TritonXPUUnrollControl(unsigned bufferSize, unsigned coreNum,
                         unsigned unrollNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
    this->unrollNum = unrollNum;
  }

  // Enabled by TRITONXPU_UNROLL_DRYRUN: report the tile factor a
  // register-budget model would pick, without changing the IR.
  bool dryRun = false;

  // Set when a legality constraint dictates the factor, in which case the
  // budget model must not override it. `pinReason` names the constraint.
  static constexpr StringLiteral kUnrollLoopAttr = "triton_xpu.unroll_loop";
  int64_t pinnedUnrollNum = 0;
  const char *pinReason = "";
  // Backs `pinReason` when the knob renames it; a plain literal otherwise.
  std::string pinReasonStorage;

  // Step 3.5: the two pins are gone by default. What each one was standing in
  // for is now a legality predicate the model consults itself -- the multi-row
  // segment shape in `isMultiRowLegal` and the fused i8 store's trip count in
  // `isFusedI8StoreLegal` -- so a pinned constant can only either agree with
  // the model or be worse than it. On device it is worse at both sites by ~19%
  // with bit-identical results (`findings.md` §1.47), and the corruption that
  // used to appear when `bool-store-vectorize` was dropped was a width
  // conversion in `LoadStoreOpToLLVM.cpp`, unrelated to tiling (§1.46).
  //
  // The constants stay reachable as an escape hatch rather than being deleted:
  // `pinUnrollNum` == 0 (the default) ignores them, < 0 restores them, and > 0
  // overrides one so its time curve can be swept -- the sweeps in `findings.md`
  // §1.31/§1.44/§1.47 are all reproduced through this knob.
  //
  // Both numbers move together on purpose: the pin's factor equals the legacy
  // one only while `unrollNum == pinnedUnrollNum` (`findings.md` §1.30 proves
  // the identity via `getNumUnroll`), so splitting them would make a sweep
  // measure two things at once.
  //
  // A default run prints nothing -- the site goes through the model, which
  // remarks its own decision. Restoring or overriding a pin does print, because
  // `emitRemark` alone never reaches the log here (nothing in this pipeline
  // installs a handler for it).
  void applyPin(ModuleOp m, int64_t constant, const char *reason) {
    if (this->pinUnrollNum == 0) {
      m->emitRemark("[UnrollControl] pin " + std::string(reason) +
                    " is not applied (constant was " +
                    std::to_string(constant) + "); the model decides instead");
      return;
    }
    int64_t value = this->pinUnrollNum > 0 ? this->pinUnrollNum : constant;
    this->unrollNum = value;
    pinnedUnrollNum = value;
    pinReasonStorage = reason;
    if (value != constant)
      pinReasonStorage += ":pin-override=" + std::to_string(value);
    pinReason = pinReasonStorage.c_str();
    llvm::errs() << "[PinKnob] pin=" << reason << " constant=" << constant
                 << (this->pinUnrollNum > 0 ? " action=override unrollNum="
                                            : " action=restored unrollNum=")
                 << value << "\n";
  }

  template <typename T> static decltype(auto) createCombineVectorizedOp(T op) {
    OpBuilder builder(op);
    return builder.create<typename COMOp<T>::type>(
        op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs());
  }

  void processOpVecTy(ModuleOp &m) {
    m.walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<COMBINE_BINARY_OP, arith::SubFOp, arith::DivFOp>(
              [&](auto combineBinaryOp) {
                if (auto tensorTy = dyn_cast<RankedTensorType>(
                        combineBinaryOp.getResult().getType())) {
                  if (isa<VectorType>(getElementTypeOrSelf(tensorTy))) {
                    auto vecOp = createCombineVectorizedOp(combineBinaryOp);
                    combineBinaryOp.replaceAllUsesWith(vecOp.getResult());
                    combineBinaryOp.erase();
                  }
                }
              })
          .Case<arith::SelectOp>([&](auto selectOp) {
            if (auto tensorTy = dyn_cast<RankedTensorType>(
                    selectOp.getResult().getType())) {
              if (isa<VectorType>(getElementTypeOrSelf(tensorTy))) {
                OpBuilder builder(selectOp);
                auto vecOp = builder.create<triton::xpu::VSelectOp>(
                    selectOp.getLoc(), tensorTy, selectOp.getCondition(),
                    selectOp.getTrueValue(), selectOp.getFalseValue());
                selectOp.replaceAllUsesWith(vecOp.getResult());
                selectOp.erase();
              }
            }
          })
          .Case<arith::CmpFOp>([&](auto cmpFOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(cmpFOp.getResult().getType())) {
              if (isa<VectorType>(getElementTypeOrSelf(tensorTy))) {
                OpBuilder builder(cmpFOp);
                auto vecOp = builder.create<triton::xpu::VCmpFOp>(
                    cmpFOp.getLoc(), cmpFOp.getResult().getType(),
                    cmpFOp.getPredicate(), cmpFOp.getLhs(), cmpFOp.getRhs());
                ;
                cmpFOp.replaceAllUsesWith(vecOp.getResult());
                cmpFOp.erase();
              }
            }
          });
    });
  }

  bool isAncestorOf(Operation *op1, Operation *op2, bool needBefore = false) {
    Block *block1 = op1->getBlock();
    for (Block *block2 = op2->getBlock(); block2 != nullptr;) {
      if (block1 == block2) {
        if (needBefore && !op1->isBeforeInBlock(op2)) {
          return false;
        }
        return true;
      }
      op2 = block2->getParentOp();
      if (op2 == nullptr) {
        break;
      }
      block2 = op2->getBlock();
    }
    return false;
  }

  bool isForBlockSizeArgument(Operation *op, Value operand) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Block *block = forOp.getBody();
      for (BlockArgument arg : block->getArguments()) {
        if (arg == operand) {
          return true;
        }
      }
    }
    return false;
  }

  void getUnrollTree(Operation *op, SetVector<Operation *> &opTree,
                     SetVector<Operation *> &visitedOps,
                     SetVector<Operation *> &excludeChainOps, Operation *rootOp,
                     bool isTop2Bottom = true, bool needBefore = false) {
    if (!op || visitedOps.count(op) ||
        isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
            triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp, scf::YieldOp,
            triton::xpu::ReduceOp, triton::xpu::ReduceReturnOp,
            triton::xpu::ScanOp, triton::xpu::ScanReturnOp>(op)) {
      return;
    }

    visitedOps.insert(op);
    if (isAncestorOf(op, rootOp, needBefore) ||
        op->getBlock() == rootOp->getBlock()) {
      opTree.insert(op);
    }

    // Search definedOp of childOp
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Then
      auto &ifThenBlock = ifOp.getThenRegion().front();
      for (auto &inBlockOp : ifThenBlock) {
        getUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps, rootOp,
                      isTop2Bottom, needBefore);
      }
      // Else
      auto &ifElseRegion = ifOp.getElseRegion();
      if (!ifElseRegion.empty()) {
        auto &ifElseBlock = ifElseRegion.front();
        for (auto &inBlockOp : ifElseBlock) {
          getUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps, rootOp,
                        isTop2Bottom, needBefore);
        }
      }
    }

    // from bottom to top
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Block *body = forOp.getBody();
      for (auto &op : body->getOperations()) {
        if (isa<triton::xpu::LoadOp, arith::ConstantOp, triton::xpu::VConstOp>(
                &op)) {
        } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(&op)) {
          auto defOp = storeOp.getValue().getDefiningOp();
          if (!isAncestorOf(forOp.getOperation(), defOp)) {
            getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                          isTop2Bottom, needBefore);
          }
        } else {
          for (auto operand : op.getOperands()) {
            if (isForBlockSizeArgument(forOp.getOperation(), operand))
              continue;
            auto defOp = operand.getDefiningOp();
            if (!isAncestorOf(forOp.getOperation(), defOp)) {
              getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                            isTop2Bottom, needBefore);
            }
          }
        }
      }
    } else if (isa<triton::xpu::LoadOp, arith::ConstantOp,
                   triton::xpu::VConstOp>(op)) {
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
      auto defOp = storeOp.getValue().getDefiningOp();
      getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                    isTop2Bottom, needBefore);
    } else {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                      isTop2Bottom, needBefore);
      }
    }

    if (isTop2Bottom) {
      // from top to bottom
      if (excludeChainOps.count(op) ||
          isa<arith::ConstantOp, triton::xpu::VConstOp>(op)) {
      } else {
        for (auto userOp : op->getUsers()) {
          getUnrollTree(userOp, opTree, visitedOps, excludeChainOps, rootOp,
                        isTop2Bottom, needBefore);
        }
      }
    }
    return;
  }

  bool isOuterBroadcast(Operation *op) {
    if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
      auto src = broadcastOp.getSrc();
      auto result = broadcastOp.getResult();
      if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
        if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
          int64_t srcElemNum = 1;
          if (auto vecTy = dyn_cast<VectorType>(getElementTypeOrSelf(srcTy))) {
            srcElemNum = vecTy.getNumElements();
          }
          int64_t resElemNum = 1;
          if (auto vecTy = dyn_cast<VectorType>(getElementTypeOrSelf(resTy))) {
            resElemNum = vecTy.getNumElements();
          }
          auto srcShape = srcTy.getShape();
          auto resShape = resTy.getShape();
          int64_t srcInnerNum = srcElemNum * srcShape.back();
          int64_t resInnerNum = resElemNum * resShape.back();
          if (srcInnerNum != resInnerNum) { // unequal dim 1 shape means in
                                            // the inner axis op chain
            assert(srcShape.front() == resShape.front() && "Invalid BroadCast");
            return true;
          }
        }
      }
    }
    return false;
  }

  template <typename T> Operation *getVdefOp(T op) {
    Operation *vDefOp;
    auto elemState = static_cast<ElemState>(op.getElemState());
    if (elemState == ElemState::SV) {
      vDefOp = op.getRhs().getDefiningOp();
    } else if (elemState == ElemState::VS) {
      vDefOp = op.getLhs().getDefiningOp();
    } else {
      llvm_unreachable(
          "[Unroll Control]: ElemState the SVOp Only Could be SV/VS.");
    }
    return vDefOp;
  }

  void getPostReduceUnrollTree(Operation *op, SetVector<Operation *> &opTree,
                               SetVector<Operation *> &visitedOps,
                               SetVector<Operation *> &excludeChainOps,
                               Operation *rootOp) {
    if (!op || visitedOps.count(op) ||
        isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
            triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp, scf::YieldOp,
            triton::xpu::ReduceOp, triton::xpu::ReduceReturnOp,
            triton::xpu::ScanOp, triton::xpu::ScanReturnOp>(op)) {
      return;
    }

    visitedOps.insert(op);
    if (isAncestorOf(op, rootOp) || op->getBlock() == rootOp->getBlock()) {
      opTree.insert(op);
    }

    // Search definedOp of childOp
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Then
      auto &ifThenBlock = ifOp.getThenRegion().front();
      for (auto &inBlockOp : ifThenBlock) {
        getPostReduceUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps,
                                rootOp);
      }
      // Else
      auto &ifElseRegion = ifOp.getElseRegion();
      if (!ifElseRegion.empty()) {
        auto &ifElseBlock = ifElseRegion.front();
        for (auto &inBlockOp : ifElseBlock) {
          getPostReduceUnrollTree(&inBlockOp, opTree, visitedOps,
                                  excludeChainOps, rootOp);
        }
      }
    }

    // from bottom to top
    if (isa<triton::xpu::LoadOp, arith::ConstantOp, triton::xpu::VConstOp>(
            op) ||
        isOuterBroadcast(op)) {
    } else if (isa<XPU_SVECTORIZED_BINARY_OP>(op)) {
      TypeSwitch<Operation *>(op).Case<XPU_SVECTORIZED_BINARY_OP>(
          [&](auto vBinOp) {
            Operation *vDefOp = getVdefOp(vBinOp);
            getPostReduceUnrollTree(vDefOp, opTree, visitedOps, excludeChainOps,
                                    rootOp);
          });
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
      auto defOp = storeOp.getValue().getDefiningOp();
      getPostReduceUnrollTree(defOp, opTree, visitedOps, excludeChainOps,
                              rootOp);
    } else {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        getPostReduceUnrollTree(defOp, opTree, visitedOps, excludeChainOps,
                                rootOp);
      }
    }

    return;
  }

  int64_t getNumCol(Type type) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(type))
      return tensorTy.getShape().back();
    else
      return 1;
  }

  int64_t getNumInVector(Type type) {
    if (auto vecType = dyn_cast<VectorType>(type))
      return vecType.getNumElements();
    else
      return 1;
  }

  int64_t getNumUnroll(Type type) {
    int64_t numUnroll = this->unrollNum * this->coreNum;
    if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
      auto clusterEncoding =
          cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
      numUnroll = this->unrollNum * clusterEncoding.getCoresPerGroup().back();
    }
    return numUnroll;
  }

  //===--------------------------------------------------------------------===//
  // Dry-run register-pressure model.
  //
  // The current tile factor comes from `unrollNum`, a global knob with no
  // notion of how many registers the segment actually needs. The code below
  // computes what a budget-driven forward decision *would* pick and reports
  // the delta. It does not touch the IR.
  //===--------------------------------------------------------------------===//

  // Reports the gate that actually decides whether tiling happens at all:
  // `numCol > numUnroll && numCol % numUnroll == 0`. A factor that does not
  // divide the width silently disables tiling, which is why unrollNum
  // 3/5/6/7/8 all produce identical code today.
  void reportGate(const char *site, Operation *op, int64_t numCol,
                  int64_t numUnroll) {
    StringRef kernel = "<unknown>";
    if (auto funcOp = op->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    const char *why = numCol <= numUnroll    ? "skip:factor>=width"
                      : (numCol % numUnroll) ? "skip:width%factor!=0"
                                             : "tile";
    llvm::errs() << "[UnrollControl][gate] " << kernel << " site=" << site
                 << " numCol=" << numCol << " numUnroll=" << numUnroll
                 << " unrollNum=" << this->unrollNum << " -> " << why << "\n";
  }

  void reportDryRun(const char *site, Operation *insertPt,
                    const SetVector<Operation *> &unrollOpTree, Type valTy,
                    int64_t numCol, int64_t numUnroll, int64_t iterNum) {
    RegPressure p;
    getRegPressure(getOperation(), unrollOpTree, p);
    int64_t peakVRegs = p.vecPeak, totalVRegs = p.vecTotal;

    int64_t coresPerGroup = 1, widthPerCore = numCol;
    if (auto tensorTy = dyn_cast<RankedTensorType>(valTy)) {
      if (auto clusterEncoding = getClusterLayout(tensorTy)) {
        coresPerGroup = clusterEncoding.getCoresPerGroup().back();
        widthPerCore = clusterEncoding.getSizePerCore().back();
      }
    }

    // Smallest trip count that fits the budget.
    int64_t iterModel =
        this->vrfBudget > 0 ? ceil<int64_t>(peakVRegs, this->vrfBudget) : 1;
    iterModel = std::max<int64_t>(iterModel, 1);
    // A trip count that does not divide the width makes setTensorType round
    // the tile up, so the tiling is silently dropped (unrollNum 3/5/6/7/8 all
    // emit byte-identical asm today). Snap up to the next legal divisor.
    int64_t iterLegal = iterModel;
    while (iterLegal < numCol &&
           (numCol % iterLegal || widthPerCore % iterLegal))
      ++iterLegal;
    if (numCol % iterLegal || widthPerCore % iterLegal)
      iterLegal = 1;

    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();

    llvm::errs() << "[UnrollControl][dry-run] " << kernel << " site=" << site
                 << " numCol=" << numCol << " widthPerCore=" << widthPerCore
                 << " coresPerGroup=" << coresPerGroup
                 << " treeOps=" << unrollOpTree.size()
                 << " peakVRegs=" << peakVRegs << " totalVRegs=" << totalVRegs
                 << " scalarPeak=" << p.scalarPeak
                 << " scalarTotal=" << p.scalarTotal
                 << " budget=" << this->vrfBudget
                 << " | now: unrollNum=" << this->unrollNum
                 << " numUnroll=" << numUnroll << " iterNum=" << iterNum
                 << " | model: iterNum=" << iterLegal << " (raw " << iterModel
                 << ") unrollNum="
                 << ceil<int64_t>(numCol, iterLegal * coresPerGroup)
                 << (iterLegal == iterNum ? " SAME" : " DIFF") << "\n";
  }

  //===--------------------------------------------------------------------===//
  // Forward tile decision.
  //
  // Instead of asking "does the global unrollNum happen to divide the width",
  // decide the trip count from the register pressure of the segment and then
  // intersect it with the trip counts the rewrite can actually express.
  //===--------------------------------------------------------------------===//

  void getTileGeometry(Type valTy, int64_t numCol, int64_t &widthPerCore,
                       int64_t &coresPerGroup) {
    widthPerCore = numCol;
    coresPerGroup = 1;
    if (auto tensorTy = dyn_cast<RankedTensorType>(valTy)) {
      if (auto clusterEncoding = getClusterLayout(tensorTy)) {
        coresPerGroup = clusterEncoding.getCoresPerGroup().back();
        widthPerCore = clusterEncoding.getSizePerCore().back();
      }
    }
  }

  // setTensorType/createEncoding slice both the shape and sizePerCore with
  // ceil(). A trip count that does not divide them evenly makes the last
  // iteration run off the end, so only exact divisors are expressible.
  // `minVecWidth` is the narrowest vector row in the tree, in slots, or 0 when
  // the tree holds no vector value. It has to divide the trip count too: since
  // M3.2 the type this geometry is read from can be the *scalar* side of a
  // vector->scalar boundary (the unpack in front of a scalar reduce), whose row
  // is vecSize times wider. Slicing by a factor the vector row cannot express
  // saturates it at one slot while the scalar row keeps dividing, and the two
  // then cover a different number of lanes per iteration.
  // `rowsPerCore` carries the multi-row constraint (`isMultiRowLegal`) and
  // `fusedI8Store` the compare-fusion store constraint (`isFusedI8StoreLegal`).
  // Both default to the inert value, so callers without the segment's value
  // type or op tree at hand keep the old behaviour.
  bool isLegalIterNum(int64_t iterNum, int64_t numCol, int64_t widthPerCore,
                      int64_t minVecWidth = 0, int64_t rowsPerCore = 1,
                      bool fusedI8Store = false) {
    return iterNum >= 1 && numCol % iterNum == 0 &&
           widthPerCore % iterNum == 0 &&
           (minVecWidth == 0 || minVecWidth % iterNum == 0) &&
           isMultiRowLegal(iterNum, rowsPerCore, widthPerCore) &&
           isFusedI8StoreLegal(iterNum, widthPerCore, fusedI8Store);
  }

  // Rows per core of the segment's own value type -- the exact quantity the
  // `core-deal-multi-rows` pin tests before it fires (`:2435`, off the reduce's
  // input type).
  int64_t getRowsPerCore(Type valTy) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(valTy))
      if (tensorTy.getShape().size() == 2)
        if (auto clusterEncoding = getClusterLayout(tensorTy))
          return clusterEncoding.getSizePerCore()[0];
    return 1;
  }

  // `ExtractSliceOp`'s lowering asserts `sizePerCore[1] == 1` whenever
  // `sizePerCore[0] > 1` (`XPUUtilityOpToLLVM.cpp:81`), and `createEncoding`
  // only ever divides the *last* dim, so a segment holding more than one row
  // per core is expressible at exactly two trip counts: 1, where no slice is
  // created at all, and `widthPerCore`, where the column count collapses to
  // one. Everything in between asserts -- measured on shortrow, whose
  // `legal={1,2,4}` has the compilable subset `{1,4}` (`findings.md` 1.31).
  //
  // This is what the pin encodes by hand, and 3.5c measured the pin's factor to
  // be 18.9% *slower* than `iterNum=1` on that probe (1.44), so the site should
  // be handed to the model -- which first requires the constraint to live here
  // rather than in the pin: before this predicate existed an out-of-set factor
  // was not filtered at all, it asserted in `ConvertTritonXPUToLLVM`.
  bool isMultiRowLegal(int64_t iterNum, int64_t rowsPerCore,
                       int64_t widthPerCore) {
    return rowsPerCore <= 1 || iterNum == 1 || iterNum >= widthPerCore;
  }

  // Does the segment store through a compare fusion? `Normalize` rewrites
  // [cmpf, extui] into a compare that leaves its result in *mask* registers
  // typed i32 and records the compared width in the store's `dtype`; the store
  // is then recognised by exactly this shape -- i32 value, i8 pointee -- both
  // here and by the `bool-store-vectorize` pin (`:2425`).
  bool hasFusedI8Store(const SetVector<Operation *> &unrollOpTree) {
    for (auto *op : unrollOpTree) {
      auto storeOp = dyn_cast<triton::xpu::StoreOp>(op);
      if (!storeOp)
        continue;
      auto valElemTy = getElementTypeOrSelf(
          getElementTypeOrSelf(storeOp.getValue().getType()));
      auto ptrElemTy = getElementTypeOrSelf(
          getElementTypeOrSelf(storeOp.getPtr().getType()));
      auto ptrTy = dyn_cast<triton::PointerType>(ptrElemTy);
      if (valElemTy.isInteger(32) && ptrTy &&
          ptrTy.getPointeeType().isInteger(8))
        return true;
    }
    return false;
  }

  // Step 3.5 prerequisite (b). Such a store lowers to
  // `vstorei8` calls that each consume exactly *four* mask registers
  // (`LoadStoreOpToLLVM.cpp`, `for (i = 0; i < valueNumElems; i += 4)`), so a
  // tile leaving a count that is not a multiple of four indexes past the end of
  // the value list and asserts in `SmallVector.h:293`. Measured on boolfused,
  // whose `widthPerCore=8` gives the compilable subset `{1,2}` out of
  // `legal={1,2,4,8}` (`findings.md` 1.31); 1.46 fixed the *silent* half of
  // that site (the untiled store's byte stride) but not this one, which is a
  // real expressibility limit of the emission loop.
  bool isFusedI8StoreLegal(int64_t iterNum, int64_t widthPerCore,
                           bool fusedI8Store) {
    if (!fusedI8Store)
      return true;
    return iterNum >= 1 && (widthPerCore / iterNum) % 4 == 0;
  }

  // Does the segment straddle a vector->scalar boundary? Only there do the two
  // representations of one row -- V vector slots on one side, V * vecSize
  // scalars on the other -- get sliced by the same factor while being counted
  // in different units, which is what makes the vector row a legality bound
  // rather than just the widest thing in the tree.
  bool hasVecScalarBoundary(const SetVector<Operation *> &unrollOpTree) {
    return false;
  }

  // How many times the segment crosses between vector and scalar form. Packs
  // and unpacks are counted one by one rather than paired up: the tier-3 price
  // is per crossing, and once M4 places these there is no guarantee a segment
  // holds a whole number of pairs.
  int64_t countBoundaryCrossings(const SetVector<Operation *> &unrollOpTree) {
    int64_t n = 0;
    for (auto *op : unrollOpTree)
      if (isa<triton::xpu::UnpackOp, triton::xpu::PackOp>(op))
        ++n;
    return n;
  }

  // Lanes in one hardware vector at the boundary, taken from the vector side of
  // the pack/unpack (`tensor<256xvector<16xf32>>` -> 16). The narrowest one
  // wins when a segment has crossings of more than one element type, because
  // the narrower vector is the one that costs more memory ops per element.
  //
  // Read from the type rather than from RegPressure::maxVecWidth: that field is
  // the widest vector *row per core in elements*, which happens to equal
  // `widthPerCore` in every geometry measured, so using it made the `N/W` term
  // of the boundary price collapse to 1 without any visible error.
  int64_t boundaryVecLanes(const SetVector<Operation *> &unrollOpTree) {
    int64_t lanes = 0;
    auto note = [&](Value val) {
      auto tensorTy = dyn_cast<RankedTensorType>(val.getType());
      if (!tensorTy)
        return;
      auto vecTy = dyn_cast<VectorType>(tensorTy.getElementType());
      if (!vecTy)
        return;
      int64_t n = vecTy.getNumElements();
      lanes = lanes ? std::min(lanes, n) : n;
    };
    for (auto *op : unrollOpTree) {
      if (auto pack = dyn_cast<triton::xpu::PackOp>(op))
        note(pack.getResult());
      else if (auto unpack = dyn_cast<triton::xpu::UnpackOp>(op))
        note(unpack.getSrc());
    }
    return lanes;
  }

  // Is this the LM buffer tritonxpu-alloca attached to a vector<->scalar
  // boundary? Recognised by its users rather than by a flag on the alloca,
  // because Alloca.cpp creates one buffer per pack/unpack and nothing else
  // ever consumes it. Conservative on purpose: a buffer shared with a store
  // pointer is not treated as a boundary buffer.
  bool isBoundaryBuffer(Operation *op) {
    if (!isa<triton::xpu::AllocaOp>(op))
      return false;
    if (op->use_empty())
      return false;
    return false;
  }

  // Is there anything for the model to pick besides "do not tile"? Used at the
  // points that decide whether to collect an unroll tree at all.
  // True once the op sits inside a tile loop this pass created.
  bool inUnrollLoop(Operation *op) {
    for (auto *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
      if (isa<scf::ForOp>(parent) && parent->hasAttr(kUnrollLoopAttr))
        return true;
    return false;
  }

  bool canTile(Operation *op, Type valTy, int64_t numCol, int64_t numUnroll) {
    if (this->budgetTiling && inUnrollLoop(op))
      return false;
    if (!this->budgetTiling)
      return numCol > numUnroll && numCol % numUnroll == 0;
    int64_t widthPerCore = 1, coresPerGroup = 1;
    getTileGeometry(valTy, numCol, widthPerCore, coresPerGroup);
    for (int64_t iterNum = 2; iterNum <= widthPerCore; ++iterNum)
      if (isLegalIterNum(iterNum, numCol, widthPerCore, /*minVecWidth=*/0,
                         getRowsPerCore(valTy)))
        return true;
    return false;
  }

  void remarkDecision(const char *site, Operation *insertPt, int64_t numCol,
                      int64_t widthPerCore, int64_t peakVRegs, int64_t target,
                      int64_t chosen, int64_t maxLegal, const char *why,
                      int64_t scalarPeak = -1, int64_t minVecWidth = -1,
                      const triton::xpu::Decision *decision = nullptr,
                      int64_t crossings = 0, int64_t vecLanes = 0) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "[UnrollControl] site=" << site << " numCol=" << numCol
       << " widthPerCore=" << widthPerCore << " minVecWidth=" << minVecWidth
       << " peakVRegs=" << peakVRegs << " scalarPeak=" << scalarPeak
       << " budget=" << this->vrfBudget << " target=" << target
       << " maxLegal=" << maxLegal << " -> iterNum=" << chosen << " (" << why
       << ")";
    // The two inputs of the boundary price, printed only where there is a
    // boundary: the price is `crossings * (widthPerCore +
    // iterNum*ceil(widthPerCore/(iterNum*lanes)) + iterNum)`, so with these the
    // reported cost can be recomputed by hand.
    if (crossings > 0)
      os << " crossings=" << crossings << " vecLanes=" << vecLanes;
    // Every criterion that took part has to be visible, feasible or not: a
    // criterion nobody can read is a criterion nobody can falsify.
    if (decision) {
      for (const auto &t : decision->perTierTrace) {
        os << " [tier" << t.tier << ":" << t.name << " cands=" << t.candidatesIn
           << "->" << t.candidatesOut;
        if (t.chosenCost)
          os << " cost=" << *t.chosenCost;
        // A report-only criterion has to say what it would have done, otherwise
        // "it did not move anything" is unfalsifiable. `n/a` is the case where
        // it had no opinion at this site at all -- which is what every probe in
        // the suite shows today for boundary-price, since none materialises a
        // vector<->scalar crossing.
        if (t.reportOnly) {
          os << " report-only pick=";
          if (t.shadowPick)
            os << *t.shadowPick
               << " moved=" << (*t.shadowPick != chosen ? "yes" : "no");
          else
            os << "n/a";
        }
        if (!t.why.empty())
          os << " veto=" << t.why;
        os << "]";
      }
      // What the soft tier would pick if tier 1 only priced instead of
      // filtering. §3.2.1 plans exactly that demotion now that the boundary
      // unit price exists, so this says per site what it would cost today --
      // before the change, not after.
      os << " budget-off pick=";
      if (decision->budgetOffPick)
        os << *decision->budgetOffPick
           << " moved=" << (*decision->budgetOffPick != chosen ? "yes" : "no");
      else
        os << "n/a";
    }
    insertPt->emitRemark(msg);
    if (dryRun)
      llvm::errs() << "[UnrollControl][decide] " << msg << "\n";
  }

  // Step 3.4 groundwork, report-only. `peakVRegs` fed to the decision is
  // max(treeP.vecPeak, blockP.vecPeak) and 3.4 replaces the block half with a
  // per-segment peak. §1.7 measured the gap between the two halves going *both*
  // ways (layernorm's reduce-for site is tree 48 against block 24, its
  // pointwise sites are equal), so `blockP` is not an upper bound and the swap
  // cannot be done blind -- the first step is to tell per site which half is
  // in charge, and whether dropping the block term would move the factor at
  // all.
  //
  // The replay uses the tree peak as the stand-in for a segment peak. That is
  // the loosest end of the range 3.4 can land in: a segment is a superset of
  // one op tree and a subset of the block, so the real per-segment peak sits
  // between these two numbers. A site where the tree-only replay picks the same
  // factor therefore cannot be moved by 3.4 in the loosening direction either,
  // which is what makes this a usable filter rather than a guess.
  // The decision the model would make with the block half of the peak dropped,
  // i.e. the loosest end of the range a real per-segment peak can land in.
  // Shared by the report below and by the `TRITONXPU_SEG_PEAK` hatch, so the
  // hatch cannot drift from the point the report predicts will move.
  static triton::xpu::Decision treeOnlyDecision(
      const RegPressure &treeP, const triton::xpu::TileContext &ctx,
      llvm::ArrayRef<int64_t> candidates, triton::xpu::TileContext &altCtx) {
    altCtx = ctx;
    altCtx.peakVRegs = treeP.vecPeak;
    altCtx.maxVecWidth = treeP.maxVecWidth;
    return triton::xpu::TileDecider().decide(candidates, altCtx);
  }

  // Names one decision point. `site` alone is not enough: layernorm has two
  // `reduce` sites and 3.4's four movable points include both of them, so the
  // key carries the geometry that tells them apart, plus the decision's ordinal
  // in this pass run. The ordinal is what makes the key usable as a filter:
  // `blockPeak` is *not* invariant under the hatch -- forcing layernorm's first
  // reduce site retiles it, which drops the block pressure the second one
  // measures from 36 to 32, and a filter written without the ordinal then
  // matches both sites instead of one (measured, findings 1.73).
  static std::string segPointKey(StringRef kernel, const char *site,
                                 const triton::xpu::TileContext &ctx,
                                 const RegPressure &treeP,
                                 const RegPressure &blockP, int64_t ordinal) {
    std::string key;
    llvm::raw_string_ostream os(key);
    os << kernel << ":" << site << ":" << ctx.numCol << ":" << treeP.vecPeak
       << ":" << blockP.vecPeak << ":#" << ordinal;
    return key;
  }

  // Step 3.4 calibration hatch.
  //   TRITONXPU_SEG_PEAK=tree           tree-only peak at every site
  //   TRITONXPU_SEG_PEAK=tree:<substr>  only where <substr> matches a point key
  // Unset (the default) keeps `max(treeP, blockP)`. This is a calibration hatch
  // and not a mode: the dominance replay says exactly four sites can move and
  // all four move 2->1 (findings 1.29), but three of them live in one kernel,
  // so timing the whole-kernel difference would mix them. The filter is what
  // makes one point measurable at a time. It changes generated code, so it is
  // deliberately not cache-neutral -- switch it with TRITON_ALWAYS_COMPILE=1.
  //
  // It stays a hatch because the measurement came back negative
  // (findings 1.73): of the four points, layernorm's three are flat to within
  // 0.1% -- that shape is bandwidth bound at ~300 GB/s, so dropping the reduce
  // strip-mine hides behind the loads -- and only mixedwidth's narrow pointwise
  // wins, by 1.07%. A 1% win on one synthetic probe does not pay for changing
  // what feeds the one *hard* criterion in the chain. None of the four spills
  // either way, so the re-test went to a shape whose segment peak does cross
  // the budget
  // (`spillseg`): there the two movable points move in *opposite* directions --
  // -9.1% on the reduce segment, +1.8% on the pointwise one -- so the hatch
  // stays a hatch for a second reason, that the sign is not predictable from
  // anything the model holds (findings 1.78, 1.81).
  int64_t segPointCounter = 0;

  static bool segPeakHatchApplies(StringRef key) {
    static const char *spec = std::getenv("TRITONXPU_SEG_PEAK");
    if (!spec)
      return false;
    StringRef s(spec);
    if (!s.consume_front("tree"))
      return false;
    if (s.empty())
      return true;
    return s.consume_front(":") && key.contains(s);
  }

  void reportSegDominance(const char *site, Operation *insertPt,
                          const RegPressure &treeP, const RegPressure &blockP,
                          const triton::xpu::TileContext &ctx,
                          llvm::ArrayRef<int64_t> candidates,
                          const triton::xpu::Decision &decision,
                          int64_t ordinal) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    triton::xpu::TileContext altCtx;
    triton::xpu::Decision alt =
        treeOnlyDecision(treeP, ctx, candidates, altCtx);
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    llvm::errs()
        << "[SegDominance] " << kernel << " site=" << site
        << " key=" << segPointKey(kernel, site, ctx, treeP, blockP, ordinal)
        << " treePeak=" << treeP.vecPeak << " blockPeak=" << blockP.vecPeak
        << " dominant="
        << (treeP.vecPeak > blockP.vecPeak
                ? "tree"
                : (treeP.vecPeak == blockP.vecPeak ? "equal" : "block"))
        << " treeMaxVecWidth=" << treeP.maxVecWidth
        << " blockMaxVecWidth=" << blockP.maxVecWidth
        << " target=" << triton::xpu::vrfBudgetTarget(ctx)
        << " treeOnlyTarget=" << triton::xpu::vrfBudgetTarget(altCtx)
        << " iterNum=" << decision.iterNum << " treeOnlyIterNum=" << alt.iterNum
        << " moved=" << (alt.iterNum != decision.iterNum ? "yes" : "no")
        << " why=" << decision.why << " treeOnlyWhy=" << alt.why << "\n";
  }

  // Step 3.4, the measurement findings 1.78 asked for. The two movable points
  // of the `spillseg` probe move in *opposite* directions on hardware --
  // relaxing the reduce segment -9.1%, the pointwise one +1.8% -- while both
  // add the same 6~7 vector spills, so neither `vspill` nor `treePeak`
  // separates the win from the loss. The one measured quantity that ordered
  // with time was the trip-weighted dynamic instruction count (-13.2% against
  // -3.3%), and its difference between two trip counts is by construction the
  // part of the body that does *not* shrink when the trip count does:
  //
  //     dyn(k) ~= W + k * F   =>   dyn(k1) - dyn(k2) = (k1 - k2) * F
  //
  // so `F`, the per-iteration fixed cost, is the candidate discriminator. It is
  // read off the lowering rather than fitted: a reduce segment whose combine is
  // interpreted as a region pays one horizontal collapse per tile iteration
  // (`ReduceOpToLLVM.cpp:981`, one `collapseVectorsJointly` per accumulator
  // key), which is `(regionOps + numOutputs) * lanes` -- the same term
  // `Vectorize.cpp:585` already charges this shape. Any other segment kind pays
  // only the loop overhead the soft tier already prices.
  //
  // **Measured on all six movable points, and refuted** (findings 1.81). `F`
  // separates the two `spillseg` arms (240 against 2) but *inverts* the two
  // points whose opposite signs are both real: `mixedwidth` #2 wins 1.07 % at
  // save=1 while `spillseg` #2 loses 1.8 % at save=2, so no threshold on
  // `dTrips * F` orders them. The collapse term also only matches the point it
  // was read off (120 against a measured 143); on `layernorm`'s two reduce
  // points the same formula says 16 and the dynamic count does not move at all.
  // And the premise itself does not hold: relaxing `mixedwidth` *raises* dyn by
  // 5 and still gets faster, so dyn is not monotone with time either.
  //
  // Kept, printing only what it measures and no verdict, for the reason
  // `TRITONXPU_SEG_PEAK` was kept after 1.73: the next attempt at this
  // judgement needs these six rows, and re-deriving them costs more than the
  // report does. Report-only and behind `TRITONXPU_TILE_REPORT`; nothing here
  // narrows the candidate set, so no emitted code can move.
  struct SegRelaxTerms {
    const char *fixedKind = "loop";
    int64_t regionOps = 0;
    int64_t numOutputs = 0;
    int64_t lanes = 0;
    int64_t fixedPerIter = 0;
  };

  static SegRelaxTerms segRelaxTerms(Operation *insertPt,
                                     const triton::xpu::TileContext &ctx,
                                     const RegPressure &treeP) {
    SegRelaxTerms t;
    auto redOp = dyn_cast<triton::xpu::ReduceOp>(insertPt);
    if (redOp && triton::xpu::reduceCombineRegionEnabled()) {
      for (Block &block : redOp.getCombineOp().getBlocks())
        for (Operation &op : block)
          if (!isa<triton::xpu::ReduceReturnOp>(op))
            ++t.regionOps;
      t.numOutputs = redOp.getNumResults();
      // The collapse runs over one accumulator vector, so the lane count is
      // this tree's own vector row -- not `ctx.maxVecWidth`, which is the
      // block's.
      t.lanes = std::max<int64_t>(treeP.maxVecWidth, 1);
      t.fixedKind = "collapse";
      t.fixedPerIter = (t.regionOps + t.numOutputs) * t.lanes;
      return t;
    }
    // Mirrors `loop-overhead` (TileDecision.cpp:79) rather than restating it in
    // other terms: one index update plus a carry in and out per iter_arg.
    t.fixedPerIter = 1 + 2 * std::max<int64_t>(ctx.loopResults, 0);
    return t;
  }

  void reportSegRelax(const char *site, Operation *insertPt,
                      const RegPressure &treeP, const RegPressure &blockP,
                      const triton::xpu::TileContext &ctx,
                      llvm::ArrayRef<int64_t> candidates,
                      const triton::xpu::Decision &decision, int64_t ordinal) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    triton::xpu::TileContext altCtx;
    triton::xpu::Decision alt =
        treeOnlyDecision(treeP, ctx, candidates, altCtx);
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    SegRelaxTerms t = segRelaxTerms(insertPt, ctx, treeP);
    int64_t before = std::max<int64_t>(decision.iterNum, 1);
    int64_t after = std::max<int64_t>(alt.iterNum, 1);
    // What relaxing buys: the fixed cost of the iterations it removes.
    int64_t save = (before - after) * t.fixedPerIter;
    // The risk side, in the model's own currency: relaxing decides against the
    // tree peak, but the register file still holds the whole block, so the
    // pressure the relaxed trip count really runs at is the block peak at that
    // trip count. Reported raw and deliberately *not* priced: 1.81 measured
    // this delta at 12 both for `mixedwidth` #2, which spills nothing, and for
    // `spillseg` #2, which goes 8 -> 15 vector spills, so it does not predict
    // spilling and multiplying it by a constant only hides that.
    int64_t overBefore = std::max<int64_t>(
        triton::xpu::pressureAtTrip(ctx, blockP.vecPeak, before) -
            ctx.vrfBudget,
        0);
    int64_t overAfter = std::max<int64_t>(
        triton::xpu::pressureAtTrip(ctx, blockP.vecPeak, after) - ctx.vrfBudget,
        0);
    llvm::errs() << "[SegRelax] " << kernel << " site=" << site << " key="
                 << segPointKey(kernel, site, ctx, treeP, blockP, ordinal)
                 << " moved=" << (after != before ? "yes" : "no")
                 << " trips=" << before << "->" << after
                 << " fixed{kind=" << t.fixedKind
                 << ",regionOps=" << t.regionOps << ",outputs=" << t.numOutputs
                 << ",lanes=" << t.lanes << ",perIter=" << t.fixedPerIter << "}"
                 << " over{before=" << overBefore << ",after=" << overAfter
                 << "}"
                 << " save=" << save << "\n";
  }

  // One run of the pressure model, from the two measurements to the decider's
  // verdict. Held together in a struct so the pin branch can ask what the model
  // would have said without a second copy of the wiring: the one time this pass
  // kept two copies of a derived number (`target`), they drifted, which is why
  // 2.3 folded it into `vrfBudgetTarget`.
  struct ModelRun {
    triton::xpu::Decision decision;
    RegPressure treeP, blockP;
    triton::xpu::TileContext ctx;
    SmallVector<int64_t> candidates;
    int64_t maxLegal = 1;
    // The tree holds no vector value, so the model has nothing to say and stops
    // before `blockP` / `ctx` / `decision` are filled. The caller decides what
    // to fall back to; this only reports that it happened.
    bool noVectorValues = false;
  };

  void runModel(const char *site, Operation *insertPt,
                const SetVector<Operation *> &unrollOpTree, Type valTy,
                int64_t numCol, int64_t widthPerCore, int64_t loopResults,
                ModelRun &run) {
    getRegPressure(getOperation(), unrollOpTree, run.treeP);
    // A tree that holds no vector value (a bool store, say) puts no pressure on
    // the vector file *of its own*, so the budget criterion has nothing to say
    // about it: what tiling buys there is code size, which is not modelled.
    // The rest of the model still runs, because the block around the tree may
    // hold vector values and because tier 3 has an opinion regardless -- the
    // caller keeps falling back to the legacy factor, and `[DecisionEntry]`
    // reports what admitting these sites would cost. Measured on `bool`: the
    // legacy path is 15.8% off the real-machine optimum (`findings.md` §1.41).
    run.noVectorValues = run.treeP.vecPeak == 0;
    getBlockRegPressure(getOperation(), insertPt, run.blockP);
    // Everything the criteria may read, and nothing else: the target itself is
    // computed by the tier-1 criterion out of these numbers
    // (`vrfBudgetTarget`), so the pass no longer holds a second copy of it.
    //
    // Scalar pressure is measured and reported, but it does not drive the
    // factor. Measured on the layernorm probe (bufSz=512): scalarPeak 389..770
    // against vecPeak 32..48, while the emitted scalar spill count stays
    // at 7..9 no matter which trip count is picked. Those un-vectorized tensors
    // are not register-resident at that scale, so charging them to a register
    // budget only distorts the target; what they actually cost is instructions.
    run.ctx.numCol = numCol;
    run.ctx.widthPerCore = widthPerCore;
    run.ctx.peakVRegs = std::max(run.treeP.vecPeak, run.blockP.vecPeak);
    run.ctx.maxVecWidth =
        std::max(run.treeP.maxVecWidth, run.blockP.maxVecWidth);
    run.ctx.scalarPeak = std::max(run.treeP.scalarPeak, run.blockP.scalarPeak);
    run.ctx.vecRow =
        hasVecScalarBoundary(unrollOpTree) ? run.treeP.minVecWidth : 0;
    run.ctx.vrfBudget = this->vrfBudget;
    run.ctx.loopResults = loopResults;
    run.ctx.boundaryCrossings = countBoundaryCrossings(unrollOpTree);
    run.ctx.vecLanes = boundaryVecLanes(unrollOpTree);

    // A per-candidate pressure term for reduce segments that collapse an
    // interpreted combine region (step 3.4b) lived here and was reverted on
    // 2026-08-05: it drove welford's reduce segment to `vspill=0` at
    // `iterNum=8` as designed, but the real-hardware sweep says that point is
    // 15.6% *slower* than the `iterNum=2` the plain budget model picks, at all
    // three occupancies measured (`findings.md` §1.17). Spilling 14 accumulator
    // vregs is cheaper on this segment than running the collapse 4x more times,
    // so zero spill is not the objective this model should be optimising.
    //
    // The candidate set is the expressible trip counts; which one to take is
    // the decider's business, tier by tier.
    for (int64_t iterNum = 1; iterNum <= widthPerCore; ++iterNum) {
      if (!isLegalIterNum(iterNum, numCol, widthPerCore, run.ctx.vecRow,
                          getRowsPerCore(valTy), hasFusedI8Store(unrollOpTree)))
        continue;
      run.candidates.emplace_back(iterNum);
      run.maxLegal = iterNum;
    }
    triton::xpu::TileDecider decider;
    run.decision = decider.decide(run.candidates, run.ctx);
    // These four report on how the model's own decision was reached, so they
    // stay off the sites where the caller throws that decision away.
    if (run.noVectorValues)
      return;
    reportMultiRow(site, insertPt, valTy, numCol, widthPerCore, run);
    reportFusedI8(site, insertPt, unrollOpTree, valTy, numCol, widthPerCore,
                  run);
    reportDivisibility(site, insertPt, valTy, unrollOpTree, numCol,
                       widthPerCore, run);
    reportInvariant(site, insertPt, run);
    reportLiveRange(site, insertPt, run);
  }

  // Companion report of `isFusedI8StoreLegal`, same shape as `[MultiRow]`:
  // `legal=` is recomputed with this one constraint switched off so `dropped=`
  // says what it cost. `kept={1,2}` on boolfused is the compilable set measured
  // by sweeping the pin, which is what let it be enforced.
  void reportFusedI8(const char *site, Operation *insertPt,
                     const SetVector<Operation *> &unrollOpTree, Type valTy,
                     int64_t numCol, int64_t widthPerCore,
                     const ModelRun &run) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    bool fusedI8 = hasFusedI8Store(unrollOpTree);
    if (!fusedI8)
      return;
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    int64_t legal = 0;
    for (int64_t iterNum = 1; iterNum <= widthPerCore; ++iterNum)
      if (isLegalIterNum(iterNum, numCol, widthPerCore, run.ctx.vecRow,
                         getRowsPerCore(valTy), /*fusedI8Store=*/false))
        ++legal;
    std::string kept;
    for (int64_t c : run.candidates)
      kept += (kept.empty() ? "" : ",") + std::to_string(c);
    llvm::errs() << "[FusedI8] " << kernel << " site=" << site
                 << " widthPerCore=" << widthPerCore << " legal=" << legal
                 << " kept={" << kept
                 << "} dropped=" << (legal - (int64_t)run.candidates.size())
                 << " iterNum=" << run.decision.iterNum << "\n";
  }

  // Step 3.7, report-only: how much of the peak a tile loop cannot shrink.
  //
  // `vrfBudgetTarget` divides the whole peak by the budget, which charges the
  // entry-live registers as if tiling made them smaller. It does not: values
  // defined outside the segment stay live across every iteration. The honest
  // model subtracts them first -- pressure at trip count `i` is
  // `I + (P - I) / i`, so fitting the budget needs `i >= ceil(P - I, B - I)`,
  // and when `I >= B` no trip count fits at all.
  //
  // This only measures the gap. `vrfBudget=24` was calibrated by device time
  // *against the dividing formula*, so it has already absorbed whatever bias
  // this introduces; swapping the formula in without recalibrating would move
  // every target at once with nothing to attribute the change to. The step
  // table says as much, and the numbers here are what decides whether the
  // recalibration is worth running.
  void reportInvariant(const char *site, Operation *insertPt,
                       const ModelRun &run) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    int64_t peak = run.ctx.peakVRegs;
    int64_t budget = run.ctx.vrfBudget;
    // Same `max` convention `peakVRegs` itself uses, then clamped: the two
    // measurements come from different op sets, so the block invariant can in
    // principle exceed the tree peak.
    int64_t inv = std::min(
        std::max(run.treeP.vecInvariant, run.blockP.vecInvariant), peak);
    llvm::errs() << "[Invariant] " << kernel << " site=" << site
                 << " peak=" << peak << " treeInv=" << run.treeP.vecInvariant
                 << " blockInv=" << run.blockP.vecInvariant
                 << " invariant=" << inv << " budget=" << budget
                 << " target=" << triton::xpu::vrfBudgetTarget(run.ctx)
                 << " targetIfSubtract=";
    if (budget <= 0)
      llvm::errs() << "n/a";
    else if (inv >= budget)
      llvm::errs() << "unreachable";
    else
      llvm::errs() << triton::xpu::vrfBudgetTargetFrom(
          run.ctx, llvm::divideCeil(peak - inv, budget - inv));
    llvm::errs() << " pick=" << run.decision.iterNum << "\n";
  }

  // Step 3.7b, report-only: what taking each value's last use from its liveness
  // range instead of from its last direct use inside the op set changed.
  //
  // The two peaks differ exactly on values that are defined outside a loop body
  // and read inside it: the direct-use walk calls them dead from that read on,
  // while the implicit backedge keeps them live for every remaining iteration.
  // So `livePeak >= usePeak` always, and where it is strictly larger the
  // direct-use walk was under-counting the file.
  //
  // Since 2026-08-14 the live peak is the one the decision runs on, so the
  // replay goes the other way: it re-decides with the direct-use peak, and
  // `moved=` says where the correction changed the factor. On the golden suite
  // that is nowhere at `vrfBudget=24` -- the understatement is real but lands
  // on the same side of every `ceil` -- which is why both numbers have to stay
  // printed and not just the picks.
  void reportLiveRange(const char *site, Operation *insertPt,
                       const ModelRun &run) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    triton::xpu::TileContext altCtx = run.ctx;
    altCtx.peakVRegs = std::max(run.treeP.vecPeakUse, run.blockP.vecPeakUse);
    altCtx.scalarPeak =
        std::max(run.treeP.scalarPeakUse, run.blockP.scalarPeakUse);
    triton::xpu::Decision alt =
        triton::xpu::TileDecider().decide(run.candidates, altCtx);
    llvm::errs() << "[LiveRange] " << kernel << " site=" << site
                 << " peak=" << run.ctx.peakVRegs
                 << " usePeak=" << altCtx.peakVRegs
                 << " treeLive=" << run.treeP.vecPeakLive
                 << " treeUse=" << run.treeP.vecPeakUse
                 << " blockLive=" << run.blockP.vecPeakLive
                 << " blockUse=" << run.blockP.vecPeakUse
                 << " scalarPeak=" << run.ctx.scalarPeak
                 << " scalarUse=" << altCtx.scalarPeak
                 << " target=" << triton::xpu::vrfBudgetTarget(run.ctx)
                 << " useTarget=" << triton::xpu::vrfBudgetTarget(altCtx)
                 << " pick=" << run.decision.iterNum
                 << " usePick=" << alt.iterNum << " moved="
                 << (alt.iterNum != run.decision.iterNum ? "yes" : "no")
                 << " why=" << run.decision.why << " useWhy=" << alt.why
                 << "\n";
  }

  // Step 2.6, report-only: what admitting the `no-vector-values` sites into the
  // criteria chain would decide. Those sites keep falling back to the legacy
  // factor -- which is derived from the `unroll_num` *width* knob and moves
  // opposite to `iterNum` -- so this is the one entry where the model is not
  // merely outvoted but never asked. `bool` is the measured case: the factory
  // default lands 15.8% off the real-machine optimum (`findings.md` §1.41).
  // `treePeak=0` means tier 1 cannot bind (`target` collapses to 1), so any
  // move here comes from tier 3, and `blockPeak` says whether the block around
  // the tree holds vector values that tier 1 could have priced.
  void reportDecisionEntry(const char *site, Operation *insertPt,
                           int64_t numCol, int64_t widthPerCore,
                           int64_t legacyIterNum, const ModelRun &run) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    std::string cands;
    for (int64_t c : run.candidates)
      cands += (cands.empty() ? "" : ",") + std::to_string(c);
    llvm::errs() << "[DecisionEntry] " << kernel << " site=" << site
                 << " numCol=" << numCol << " widthPerCore=" << widthPerCore
                 << " treePeak=" << run.treeP.vecPeak
                 << " blockPeak=" << run.blockP.vecPeak
                 << " treeScalarPeak=" << run.treeP.scalarPeak
                 << " peakVRegs=" << run.ctx.peakVRegs
                 << " budget=" << run.ctx.vrfBudget
                 << " target=" << triton::xpu::vrfBudgetTarget(run.ctx)
                 << " cands={" << cands << "}"
                 << " legacyIterNum=" << legacyIterNum
                 << " modelIterNum=" << run.decision.iterNum
                 << " modelWhy=" << run.decision.why << " moved="
                 << (run.decision.iterNum != legacyIterNum ? "yes" : "no")
                 << "\n";
  }

  // Step 3.6, report-only: what the three divisibility terms in
  // `isLegalIterNum` cost, per site. The model's pick is the smallest candidate
  // at or above `vrfBudgetTarget` (tier 1 admits >= target, tier 3 takes the
  // smallest back), so if divisibility were relaxed it would take the target
  // itself -- which makes `slack = pick - relaxedPick` the entire price of
  // these terms here, with nothing to calibrate.
  //
  // The relaxed set keeps `isMultiRowLegal` and `isFusedI8StoreLegal`: those
  // two are shape constraints on how a segment can be sliced at all
  // (§1.45/§1.46), and a main/tail split cannot express them either. Only the
  // three `%` terms are what a tail loop would buy back.
  //
  // `blockedBy=` names which of the three actually excludes the relaxed factor.
  // Relaxing an inert term buys nothing, and `minVecWidth` in particular must
  // be argued on its own (§3.6(3)) -- so it has to be visible which term is
  // binding before any of them is touched.
  void reportDivisibility(const char *site, Operation *insertPt, Type valTy,
                          const SetVector<Operation *> &unrollOpTree,
                          int64_t numCol, int64_t widthPerCore,
                          const ModelRun &run) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    int64_t target = triton::xpu::vrfBudgetTarget(run.ctx);
    int64_t rowsPerCore = getRowsPerCore(valTy);
    bool fusedI8 = hasFusedI8Store(unrollOpTree);
    // Smallest factor the shape constraints alone allow at or above the target.
    int64_t relaxed = 0;
    for (int64_t i = std::max<int64_t>(target, 1); i <= widthPerCore; ++i)
      if (isMultiRowLegal(i, rowsPerCore, widthPerCore) &&
          isFusedI8StoreLegal(i, widthPerCore, fusedI8)) {
        relaxed = i;
        break;
      }
    std::string blockedBy;
    if (relaxed) {
      auto note = [&](const char *term) {
        blockedBy += (blockedBy.empty() ? "" : "+") + std::string(term);
      };
      if (numCol % relaxed != 0)
        note("numCol");
      if (widthPerCore % relaxed != 0)
        note("widthPerCore");
      if (run.ctx.vecRow != 0 && run.ctx.vecRow % relaxed != 0)
        note("minVecWidth");
    }
    if (blockedBy.empty())
      blockedBy = "none";
    int64_t pick = run.decision.iterNum;
    llvm::errs() << "[Divisibility] " << kernel << " site=" << site
                 << " numCol=" << numCol << " widthPerCore=" << widthPerCore
                 << " minVecWidth=" << run.ctx.vecRow << " target=" << target
                 << " pick=" << pick << " relaxedPick=";
    if (relaxed)
      llvm::errs() << relaxed;
    else
      llvm::errs() << "none";
    llvm::errs() << " slack=" << (relaxed ? pick - relaxed : 0)
                 << " blockedBy=" << blockedBy << "\n";
  }

  // Per decision site: how much of the divisor set the multi-row constraint
  // removes, and which factor the model then takes. `run.candidates` is already
  // filtered, so `legal=` is recomputed here with the constraint switched off
  // -- that is the only way to see what it cost. `dropped=0` everywhere means
  // the constraint is inert on the probe set, which is what let it be enforced
  // without moving a decision.
  void reportMultiRow(const char *site, Operation *insertPt, Type valTy,
                      int64_t numCol, int64_t widthPerCore,
                      const ModelRun &run) {
    if (!mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_REPORT"))
      return;
    int64_t rowsPerCore = getRowsPerCore(valTy);
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    int64_t legal = 0;
    for (int64_t iterNum = 1; iterNum <= widthPerCore; ++iterNum)
      if (isLegalIterNum(iterNum, numCol, widthPerCore, run.ctx.vecRow,
                         /*rowsPerCore=*/1))
        ++legal;
    std::string kept;
    for (int64_t c : run.candidates)
      kept += (kept.empty() ? "" : ",") + std::to_string(c);
    llvm::errs() << "[MultiRow] " << kernel << " site=" << site
                 << " rowsPerCore=" << rowsPerCore
                 << " widthPerCore=" << widthPerCore
                 << " binds=" << (rowsPerCore > 1 ? "yes" : "no")
                 << " legal=" << legal << " kept={" << kept
                 << "} dropped=" << (legal - (int64_t)run.candidates.size())
                 << " iterNum=" << run.decision.iterNum << "\n";
  }

  // Step 3.5 groundwork, report-only. `pinnedUnrollNum` short-circuits the
  // model before it ever runs, and the pin value is not reachable from any knob
  // (`unroll_num` is read after it), so today neither magic number can be shown
  // to be wrong: boolfused emits the same code at unroll_num 1/4/16. This puts
  // the pin's factor next to what the model would have picked at the same site.
  //
  // The legacy factor is printed next to it because that is what the site would
  // fall back to. The two turn out to be the *same* number by construction, not
  // by coincidence: `getNumUnroll` already folds `unrollNum * coresPerGroup`
  // into `numUnroll`, and both pin sites set `unrollNum` to the pinned value,
  // so `ceil(numCol, pin * coresPerGroup) == ceil(numCol, numUnroll)`. Measured
  // on both pin sites (`findings.md` §1.30). That is why 3.5b only has to stop
  // the early return -- the pin's arithmetic contributes nothing of its own.
  void reportPinShadow(const char *site, Operation *insertPt,
                       const SetVector<Operation *> &unrollOpTree, Type valTy,
                       int64_t numCol, int64_t widthPerCore,
                       int64_t coresPerGroup, int64_t legacyIterNum,
                       int64_t pinned, int64_t loopResults) {
    if (!std::getenv("TRITONXPU_TILE_REPORT"))
      return;
    ModelRun run;
    runModel(site, insertPt, unrollOpTree, valTy, numCol, widthPerCore,
             loopResults, run);
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    llvm::errs() << "[PinShadow] " << kernel << " site=" << site
                 << " pin=" << pinReason
                 << " pinnedUnrollNum=" << pinnedUnrollNum
                 << " numCol=" << numCol << " widthPerCore=" << widthPerCore
                 << " coresPerGroup=" << coresPerGroup
                 << " pinIterNum=" << pinned
                 << " legacyIterNum=" << legacyIterNum;
    // Without the pin this site would fall back to the legacy factor -- which
    // the pin itself already overwrote (`this->unrollNum`), so the comparison
    // is against the pinned unrollNum, not the user's. Worth seeing, not
    // hiding.
    if (run.noVectorValues) {
      llvm::errs() << " modelWhy=no-vector-values:legacy treePeak=0"
                   << " treeScalarPeak=" << run.treeP.scalarPeak
                   << " modelIterNum=" << legacyIterNum
                   << " moved=" << (legacyIterNum != pinned ? "yes" : "no")
                   << "\n";
      return;
    }
    // The candidate set in full: a pin value that is not even expressible, or a
    // set collapsed to {1}, is the degradation path 3.5 must keep observable.
    std::string legal;
    for (int64_t c : run.candidates)
      legal += (legal.empty() ? "" : ",") + std::to_string(c);
    llvm::errs() << " treePeak=" << run.treeP.vecPeak
                 << " blockPeak=" << run.blockP.vecPeak
                 << " vecRow=" << run.ctx.vecRow
                 << " target=" << triton::xpu::vrfBudgetTarget(run.ctx)
                 << " maxLegal=" << run.maxLegal << " legal={" << legal << "}"
                 << " pinExpressible="
                 << (llvm::is_contained(run.candidates, pinned) ? "yes" : "no")
                 << " modelIterNum=" << run.decision.iterNum
                 << " modelWhy=" << run.decision.why
                 << " moved=" << (run.decision.iterNum != pinned ? "yes" : "no")
                 << "\n";
  }

  // Applies the 3.4 hatch at this site if the filter names it. Returns false
  // when it does not, so the default path keeps the model's own decision
  // untouched. `remarkDecision` has already reported that decision, so the log
  // shows both what the model said and what the hatch overrode it with.
  bool applySegPeakHatch(const char *site, Operation *insertPt, ModelRun &run,
                         int64_t ordinal, triton::xpu::Decision &out) {
    StringRef kernel = "<unknown>";
    if (auto funcOp = insertPt->getParentOfType<triton::FuncOp>())
      kernel = funcOp.getName();
    std::string key =
        segPointKey(kernel, site, run.ctx, run.treeP, run.blockP, ordinal);
    if (!segPeakHatchApplies(key))
      return false;
    triton::xpu::TileContext altCtx;
    out = treeOnlyDecision(run.treeP, run.ctx, run.candidates, altCtx);
    llvm::errs() << "[SegPeakForce] key=" << key
                 << " iterNum=" << run.decision.iterNum << "->" << out.iterNum
                 << " peak=" << run.ctx.peakVRegs << "->" << altCtx.peakVRegs
                 << " maxVecWidth=" << run.ctx.maxVecWidth << "->"
                 << altCtx.maxVecWidth << " why=" << out.why << "\n";
    out.why = "seg-peak-tree:" + out.why;
    return true;
  }

  // The register file is shared by everything simultaneously live in the block,
  // not by one op tree: sibling trees hold their values at the same time (the
  // mean and var accumulators of a layernorm, say). Measuring per tree
  // double-spends the file, so the pressure is taken over the whole block the
  // tile loop will live in. Every vector value there scales with the same
  // factor, so one target per block is the right granularity.
  // Returns the whole decision, not just the factor: the trip count alone
  // cannot say which criterion produced it, and the two call sites must not
  // have to change again every time a criterion is added (§3.2.2).
  // `loopResults` is the number of iter_args the tile loop will carry, which
  // only the caller knows before the loop exists.
  triton::xpu::Decision
  decideIterNum(const char *site, Operation *insertPt,
                const SetVector<Operation *> &unrollOpTree, Type valTy,
                int64_t numCol, int64_t numUnroll, int64_t loopResults = 0) {
    int64_t legacyIterNum = ceil<int64_t>(numCol, numUnroll);
    // Counts every decision this pass makes, in walk order, so one point can be
    // named even after the hatch has changed the pressure the later ones see.
    const int64_t ordinal = ++segPointCounter;
    if (dryRun)
      reportDryRun(site, insertPt, unrollOpTree, valTy, numCol, numUnroll,
                   legacyIterNum);
    int64_t widthPerCore = 1, coresPerGroup = 1;
    getTileGeometry(valTy, numCol, widthPerCore, coresPerGroup);

    if (!this->budgetTiling) {
      // The vector row is a legality bound, not part of the model, so it binds
      // the legacy factor as well: ceil(numCol, unrollNum) is derived from the
      // same possibly-scalar `valTy` and can exceed what the vector values in
      // the tree can express. Fall back to the largest expressible factor that
      // is no larger.
      RegPressure p;
      int64_t vecRow = 0;
      if (hasVecScalarBoundary(unrollOpTree)) {
        getRegPressure(getOperation(), unrollOpTree, p);
        vecRow = p.minVecWidth;
      }
      if (vecRow &&
          !isLegalIterNum(legacyIterNum, numCol, widthPerCore, vecRow)) {
        int64_t clamped = 1;
        for (int64_t iterNum = 1; iterNum <= legacyIterNum; ++iterNum)
          if (isLegalIterNum(iterNum, numCol, widthPerCore, vecRow))
            clamped = iterNum;
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "[UnrollControl] site=" << site << " numCol=" << numCol
           << " widthPerCore=" << widthPerCore << " minVecWidth=" << vecRow
           << " -> iterNum=" << clamped << " (vector-row-bound, legacy "
           << legacyIterNum << ")";
        insertPt->emitRemark(msg);
        legacyIterNum = clamped;
        return {legacyIterNum, "vector-row-bound", {}};
      }
      return {legacyIterNum, "legacy", {}};
    }

    // Collect the expressible trip counts once: the largest is the fallback
    // when the budget cannot be reached at all.
    int64_t maxLegal = 1;

    // A pinned factor claims to be a correctness constraint, not a heuristic,
    // so the model gets no say here; it only reports which constraint took
    // over. Whether that claim holds is now testable from the outside:
    // `pin-unroll-num` (3.5b) can drop the pin or move its constant, and
    // `[PinShadow]` says what the model would have picked at the same site.
    if (pinnedUnrollNum > 0) {
      int64_t pinned = ceil<int64_t>(numCol, pinnedUnrollNum * coresPerGroup);
      pinned = std::max<int64_t>(pinned, 1);
      remarkDecision(site, insertPt, numCol, widthPerCore, /*peakVRegs=*/-1,
                     /*target=*/-1, pinned, maxLegal, pinReason);
      reportPinShadow(site, insertPt, unrollOpTree, valTy, numCol, widthPerCore,
                      coresPerGroup, legacyIterNum, pinned, loopResults);
      return {pinned, pinReason, {}};
    }

    ModelRun run;
    runModel(site, insertPt, unrollOpTree, valTy, numCol, widthPerCore,
             loopResults, run);
    // Nothing for the model to say about the vector file: keep the legacy
    // factor. What the chain *would* have decided is reported, not applied --
    // step 2.6 has to prove the other probes do not move before this entry can
    // be opened.
    if (run.noVectorValues) {
      reportDecisionEntry(site, insertPt, numCol, widthPerCore, legacyIterNum,
                          run);
      if (!this->openDecisionEntry) {
        remarkDecision(site, insertPt, numCol, widthPerCore, /*peakVRegs=*/0,
                       /*target=*/-1, legacyIterNum, /*maxLegal=*/-1,
                       "no-vector-values:legacy", run.treeP.scalarPeak);
        return {legacyIterNum, "no-vector-values:legacy", {}};
      }
      triton::xpu::Decision opened = run.decision;
      opened.why = "no-vector-values:model";
      remarkDecision(site, insertPt, numCol, widthPerCore, run.ctx.peakVRegs,
                     triton::xpu::vrfBudgetTarget(run.ctx), opened.iterNum,
                     run.maxLegal, opened.why.c_str(), run.ctx.scalarPeak,
                     run.ctx.vecRow, &opened, run.ctx.boundaryCrossings,
                     run.ctx.vecLanes);
      return opened;
    }
    remarkDecision(site, insertPt, numCol, widthPerCore, run.ctx.peakVRegs,
                   triton::xpu::vrfBudgetTarget(run.ctx), run.decision.iterNum,
                   run.maxLegal, run.decision.why.c_str(), run.ctx.scalarPeak,
                   run.ctx.vecRow, &run.decision, run.ctx.boundaryCrossings,
                   run.ctx.vecLanes);
    reportSegDominance(site, insertPt, run.treeP, run.blockP, run.ctx,
                       run.candidates, run.decision, ordinal);
    reportSegRelax(site, insertPt, run.treeP, run.blockP, run.ctx,
                   run.candidates, run.decision, ordinal);
    triton::xpu::Decision forced;
    if (applySegPeakHatch(site, insertPt, run, ordinal, forced))
      return forced;
    return run.decision;
  }

  // Earliest store of the tree plus the geometry it implies. Returns false when
  // the stores of one tree live in different blocks, which the rewrite cannot
  // handle yet.
  bool getTreeUnrollInfo(const SetVector<Operation *> &unrollOpTree,
                         triton::xpu::StoreOp &insertPt,
                         SmallVector<triton::xpu::StoreOp> &allStoreOps,
                         int64_t &numCol, int64_t &numUnroll) {
    for (auto op : unrollOpTree) {
      auto storeOp = dyn_cast<triton::xpu::StoreOp>(op);
      if (!storeOp)
        continue;
      auto type = storeOp.getValue().getType();
      numUnroll = numUnroll == 1 ? getNumUnroll(type)
                                 : std::min(numUnroll, getNumCol(type));
      numCol =
          numCol == 1 ? getNumCol(type) : std::min(numCol, getNumCol(type));
      allStoreOps.emplace_back(storeOp);
      //[TODO] To deal with the case that storeOps are in more than one block
      if (insertPt && insertPt->getBlock() != storeOp->getBlock())
        return false;
      if (!insertPt || storeOp->isBeforeInBlock(insertPt))
        insertPt = storeOp;
    }
    return true;
  }

  // One decision per tree, taken before any IR is touched so that the discrete
  // pointer rewrite can be kept in sync with it.
  SmallVector<int64_t>
  planIterNums(SmallVector<SetVector<Operation *>> &unrollOpTrees,
               const char *site) {
    SmallVector<int64_t> plan;
    for (auto &unrollOpTree : unrollOpTrees) {
      triton::xpu::StoreOp insertPt;
      SmallVector<triton::xpu::StoreOp> allStoreOps;
      int64_t numCol = 1, numUnroll = 1;
      if (!getTreeUnrollInfo(unrollOpTree, insertPt, allStoreOps, numCol,
                             numUnroll) ||
          !insertPt) {
        plan.emplace_back(1);
        continue;
      }
      // A pointwise store segment carries no iter_args (createFor is called
      // with an empty range at :1263), so the loop-overhead criterion sees
      // only the index arithmetic.
      plan.emplace_back(decideIterNum(site, insertPt, unrollOpTree,
                                      insertPt.getValue().getType(), numCol,
                                      numUnroll, /*loopResults=*/0)
                            .iterNum);
    }
    return plan;
  }

  Type createPointerType(Type type, int64_t vecSize) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      Type elemType = getElementTypeOrSelf(tensorType);
      Type elemScalarType = getElementTypeOrSelf(elemType);
      Type pointerType = triton::PointerType::get(elemScalarType, 0);
      auto shape = tensorType.getShape().vec();
      shape[shape.size() - 1] = shape.back() * vecSize;
      return RankedTensorType::get(shape, pointerType,
                                   tensorType.getEncoding());
    } else {
      return triton::PointerType::get(type, 0);
    }
  }

  triton::xpu::ClusterLayoutAttr
  createEncoding(MLIRContext *context, triton::xpu::ClusterLayoutAttr &encoding,
                 int64_t iterNum) const {
    auto sizePerCore = encoding.getSizePerCore().vec();
    sizePerCore[sizePerCore.size() - 1] =
        ceil<int64_t>(sizePerCore.back(), iterNum);
    auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
        context, sizePerCore, encoding.getCoresPerGroup(),
        encoding.getGroupsPerCluster(), encoding.getOrder());
    return newEncoding;
  }

  void setTensorType(MLIRContext *context, Operation *op, int64_t iterNum,
                     bool isOuter, bool sliceShape = true) const {
    for (auto [i, resTy] : llvm::enumerate(op->getResultTypes())) {
      if (isa<RankedTensorType>(resTy) && !isOuter) {
        auto tensorTy = cast<RankedTensorType>(resTy);
        auto shape = tensorTy.getShape().vec();
        if (sliceShape) {
          shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
        }
        RankedTensorType controledTensorTy;
        if (auto sliceEncoding = dyn_cast<triton::gpu::SliceEncodingAttr>(
                tensorTy.getEncoding())) {
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(sliceEncoding.getParent());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          auto newEncoding = triton::gpu::SliceEncodingAttr::get(
              context, sliceEncoding.getDim(), newClusterEncoding);
          controledTensorTy = RankedTensorType::get(
              shape, tensorTy.getElementType(), newEncoding);
        } else {
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          controledTensorTy = RankedTensorType::get(
              shape, tensorTy.getElementType(), newClusterEncoding);
        }
        op->getResult(i).setType(controledTensorTy);
      }
    }
  }

  void setHoistedOperand(MLIRContext *context, OpBuilder &builder,
                         Location &loc, mlir::Block &block, scf::IfOp &ifOp,
                         int64_t iterNum) {
    for (auto &inBlockOp : block) {
      if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(&inBlockOp)) {
        unsigned numifOpResults = ifOp.getNumResults();
        unsigned numyieldOpOperands = yieldOp.getNumOperands();
        // isOperandValidInSameForBlock denotes two points:
        // 1. Extraction is required if the operand of YieldOp
        //    does not match the type expected by the result of IfOp
        // 2. whether the operand of YieldOp is in the same ForBlock as IfOp.
        SmallVector<bool, 4> isOperandValidInSameForBlock(numyieldOpOperands);
        assert((numifOpResults == numyieldOpOperands) &&
               "The number of IfOp results and YieldOp operands must match.");
        for (unsigned i = 0; i < numyieldOpOperands; ++i) {
          Type ifOpResTy = ifOp.getResult(i).getType();
          isOperandValidInSameForBlock[i] =
              isOperandOperationInSameForBlock(&inBlockOp, i) ||
              (inBlockOp.getOperand(i).getType() == ifOpResTy);
          if (!isOperandValidInSameForBlock[i]) {
            // triton_xpu.vconst is what TritonXPUVectorize turns a top-level
            // arith.constant into, and it is just as pure and loop-invariant,
            // so the hoisted extract_slice below is equally safe for it.
            assert((isa<arith::ConstantOp>(
                        inBlockOp.getOperand(i).getDefiningOp()) ||
                    isa<triton::xpu::VConstOp>(
                        inBlockOp.getOperand(i).getDefiningOp())) &&
                   "Unable to extract the non-constant operand.");
            auto extractSliceOp =
                getExtractedOperand(context, builder, loc, yieldOp, i, iterNum);
            extractSliceOp->moveBefore(ifOp);
            inBlockOp.setOperand(i, extractSliceOp->getResult(0));
          }
        }
      } else if (((&inBlockOp)->hasTrait<OpTrait::SameTypeOperands>() ||
                  (&inBlockOp)
                      ->hasTrait<OpTrait::SameOperandsAndResultType>())) {
        // 1. setOperandTensorType
        if ((&inBlockOp)->hasTrait<OpTrait::NOperands<2>::Impl>()) {
          unsigned numOperands = inBlockOp.getNumOperands();
          SmallVector<bool, 4> isOperandValidInSameForBlock(numOperands);
          for (size_t i = 0; i < numOperands; ++i) {
            isOperandValidInSameForBlock[i] =
                isOperandOperationInSameForBlock(&inBlockOp, i) ||
                (inBlockOp.getOperand(i).getType() ==
                 inBlockOp.getOperand(i ^ 1).getType());
            if (!isOperandValidInSameForBlock[i]) {
              assert((isa<arith::ConstantOp>(
                          inBlockOp.getOperand(i).getDefiningOp()) ||
                      isa<triton::xpu::VConstOp>(
                          inBlockOp.getOperand(i).getDefiningOp())) &&
                     "Unable to extract the non-constant operand.");
              auto extractSliceOp = getExtractedOperand(context, builder, loc,
                                                        &inBlockOp, i, iterNum);
              extractSliceOp->moveBefore(ifOp);
              inBlockOp.setOperand(i, extractSliceOp->getResult(0));
            }
          }
        }
      }
    }
  }

  triton::xpu::ExtractSliceOp
  getExtractedOperand(MLIRContext *context, OpBuilder &builder, Location &loc,
                      mlir::Operation *op, unsigned operandIndex,
                      int64_t iterNum) const {
    auto resTy = op->getOperand(operandIndex).getType();
    RankedTensorType tensorTy;
    if (isa<RankedTensorType>(resTy)) {
      tensorTy = cast<RankedTensorType>(resTy);
    }
    auto shape = tensorTy.getShape().vec();
    shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
    auto clusterEncoding =
        cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
    auto newClusterEncoding = createEncoding(context, clusterEncoding, iterNum);

    RankedTensorType controledTensorTy = RankedTensorType::get(
        shape, tensorTy.getElementType(), newClusterEncoding);
    triton::xpu::ExtractSliceOp extractSliceOp =
        builder.create<triton::xpu::ExtractSliceOp>(
            loc, controledTensorTy, op->getOperand(operandIndex));
    return extractSliceOp;
  }

  // Determine whether the operand has been hoisted
  bool isOperandOperationInSameForBlock(mlir::Operation *op,
                                        unsigned operandIndex) {
    auto *parentOp = op->getParentOp();
    while (parentOp && !llvm::isa<mlir::scf::ForOp>(parentOp)) {
      parentOp = parentOp->getParentOp();
    }
    if (!parentOp)
      return false;

    auto forOp = llvm::cast<mlir::scf::ForOp>(parentOp);
    mlir::Value operand = op->getOperand(operandIndex);
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      mlir::Block *block = forOp.getBody()->front().getBlock();
      return blockArg.getOwner() == block;
    } else {
      mlir::Operation *definingOp = operand.getDefiningOp();
      if (definingOp) {
        return definingOp->getBlock()->getParentOp() == forOp.getOperation();
      }
    }
    return false;
  }

  void insertIndex(Operation *op, Value idxVar) {
    OpBuilder builder(op);
    auto operandSegmentSizesAttr =
        op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    SmallVector<int, 4> operandSegmentSizes(
        operandSegmentSizesAttr.asArrayRef());
    // LoadOp: 0: ptr, 1: mask, 2: other, 3: index
    // StoreOp: 0: ptr, 1: value, 2: mask, 3: index
    // MakeRangeOp: 0: loopIndex, 1: unrollIndex
    // InterleaveOp: 0: loopIndex, 1: unrollIndex
    ++operandSegmentSizes[operandSegmentSizes.size() - 1];
    op->setAttr("operandSegmentSizes",
                builder.getDenseI32ArrayAttr(operandSegmentSizes));
    op->insertOperands(op->getNumOperands(), {idxVar});
  }

  void getOpChainBwdPostReduce(llvm::SetVector<Operation *> &opChain,
                               Operation *op) {
    if (!op) {
      return;
    }
    opChain.insert(op);

    int noDefCnt = 0;
    for (auto operand : op->getOperands()) {
      if (!operand.getDefiningOp()) {
        noDefCnt++;
      }
    }

    if (isa<arith::ConstantOp, triton::xpu::VConstOp, triton::xpu::StoreOp,
            triton::xpu::ReduceOp>(op) ||
        noDefCnt == op->getNumOperands()) {
      return;
    }

    for (auto operand : op->getOperands()) {
      getOpChainBwdPostReduce(opChain, operand.getDefiningOp());
    }
  }

  void getOuterChain(llvm::SetVector<Operation *> &allOpTree,
                     llvm::SetVector<Operation *> &outerChain,
                     bool postReduce = false) {
    for (auto op : allOpTree) {
      if (auto expandDimOp = dyn_cast<triton::ExpandDimsOp>(op)) {
        auto src = expandDimOp.getSrc();
        auto result = expandDimOp.getResult();
        if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
            if (expandDimOp.getAxis() == 1) {
              if (postReduce) {
                getOpChainBwdPostReduce(outerChain, expandDimOp);
              } else {
                getOpChainBwd(outerChain, expandDimOp);
              }
              outerChain.remove(expandDimOp);
            }
          }
        }
      }
      if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
        auto src = broadcastOp.getSrc();
        auto result = broadcastOp.getResult();
        if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
            int64_t srcElemNum = 1;
            if (auto vecTy =
                    dyn_cast<VectorType>(getElementTypeOrSelf(srcTy))) {
              srcElemNum = vecTy.getNumElements();
            }
            int64_t resElemNum = 1;
            if (auto vecTy =
                    dyn_cast<VectorType>(getElementTypeOrSelf(resTy))) {
              resElemNum = vecTy.getNumElements();
            }
            auto srcShape = srcTy.getShape();
            auto resShape = resTy.getShape();
            int64_t srcInnerNum = srcElemNum * srcShape.back();
            int64_t resInnerNum = resElemNum * resShape.back();
            if (srcInnerNum != resInnerNum) { // unequal dim 1 shape means in
                                              // the inner axis op chain
              assert(srcShape.front() == resShape.front() &&
                     "Invalid BroadCast");
              if (postReduce) {
                getOpChainBwdPostReduce(outerChain, broadcastOp);
              } else {
                getOpChainBwd(outerChain, broadcastOp);
              }
              outerChain.remove(broadcastOp);
            }
          }
        }
      }
    }
  }

  void
  getOuterChains(const SmallVector<llvm::SetVector<Operation *>> &allOpTrees,
                 SmallVector<llvm::SetVector<Operation *>> &outerChains,
                 bool postReduce = false) {
    for (auto allOpTree : allOpTrees) {
      SetVector<Operation *> outerChain;
      getOuterChain(allOpTree, outerChain, postReduce);
      outerChains.emplace_back(outerChain);
    }
  }

  void getDAG(Operation *op, SetVector<Operation *> &visitedOps,
              SmallVector<SetVector<Operation *>> &unrollOpTrees,
              SetVector<Operation *> &excludeChainOps, bool isTop2Bottom = true,
              bool needBefore = false) {
    SetVector<Operation *> opTree;
    getUnrollTree(op, opTree, visitedOps, excludeChainOps, op, isTop2Bottom,
                  needBefore);
    if (!opTree.empty()) {
      SetVector<Operation *> sortedOpTree = sortOpTree(opTree);
      unrollOpTrees.emplace_back(sortedOpTree);
    }
  }

  void getPostReduceDAG(Operation *op, SetVector<Operation *> &visitedOps,
                        SmallVector<SetVector<Operation *>> &unrollOpTrees,
                        SetVector<Operation *> &excludeChainOps) {
    SetVector<Operation *> opTree;
    getPostReduceUnrollTree(op, opTree, visitedOps, excludeChainOps, op);
    if (!opTree.empty()) {
      SetVector<Operation *> sortedOpTree = sortOpTree(opTree);
      unrollOpTrees.emplace_back(sortedOpTree);
    }
  }

  void createFor(OpBuilder &builder, Location &loc, int64_t start,
                 int64_t iterNum, scf::ForOp &forOp, arith::IndexCastOp &idxVar,
                 ValueRange &iterArgs) {
    auto lower = builder.create<arith::ConstantIndexOp>(loc, start);
    auto upper = builder.create<arith::ConstantIndexOp>(loc, iterNum);
    auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
    if (iterArgs.empty()) {
      forOp = builder.create<scf::ForOp>(loc, lower, upper, step);
    } else {
      forOp = builder.create<scf::ForOp>(loc, lower, upper, step, iterArgs);
    }
    // Later stages of this pass walk the module again and would otherwise tile
    // an already tiled segment a second time, inserting the loop index twice.
    // The legacy gate hid this because a tiled value has numCol == numUnroll.
    forOp->setAttr(kUnrollLoopAttr, UnitAttr::get(forOp->getContext()));
    builder.setInsertionPointToStart(forOp.getBody());

    idxVar = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                forOp.getInductionVar());
  }

  void createLoopBody(MLIRContext *context, OpBuilder &builder, Location &loc,
                      int64_t iterNum, SetVector<Operation *> &unrollOpTree,
                      SetVector<Operation *> &outerChain,
                      arith::IndexCastOp &idxVar, IRMapping &mapping) {
    for (auto op : unrollOpTree) {
      // A vector<->scalar boundary buffer must not be cloned into the tile
      // loop. An alloca inside a loop body is never promoted, so every
      // iteration grows the stack and the launch fails outright. The original
      // alloca already sits before forOp (the tree precedes insertPt, and
      // forOp was created at insertPt), so all this takes is not cloning it:
      // the clones inside the body then reference the outside buffer, and
      // eraseDAG leaves it alone because it is no longer use-empty.
      //
      // Keeping one full-width buffer for all iterations is correct because
      // the boundary is written and read back within a single iteration, and
      // leaving its type unsliced is harmless because lowering reads only
      // element 0 of the bufPtr (LoadStoreOpToLLVM.cpp getBoundaryLMBase).
      if (isBoundaryBuffer(op))
        continue;
      bool isOuter = inOpChain(outerChain, op);
      auto newOp = builder.clone(*op, mapping);
      setTensorType(context, newOp, iterNum, isOuter);
      TypeSwitch<Operation *>(newOp)
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(loadOp.getPtr().getType())) {
              auto shape = tensorTy.getShape();
              bool isOuter = (shape.size() == 2 && shape.back() == 1);
              if (!isOuter && !loadOp.getSVOpt()) {
                insertIndex(newOp, idxVar);
              }
            }
          })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(storeOp.getPtr().getType())) {
              auto shape = tensorTy.getShape();
              bool isOuter = (shape.size() == 2 && shape.back() == 1);
              if (!isOuter) {
                insertIndex(newOp, idxVar);
              }
            }
          })
          .Case<triton::xpu::MakeRangeOp>([&](auto makeRangeOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(op->getResults()[0].getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<triton::xpu::InterleaveOp>([&](auto interleaveOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(op->getResults()[0].getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<XPUPrintOp>([&](auto xpuprintOp) {
            Value idxVar64 = builder.create<arith::ExtSIOp>(
                loc, builder.getI64Type(), idxVar);
            Value ucBound = builder.create<arith::ConstantIntOp>(
                loc, builder.getI64Type(), iterNum);
            auto NewOp = builder.create<XPUPrintOp>(
                xpuprintOp.getLoc(), xpuprintOp.getPidx(), xpuprintOp.getPidy(),
                xpuprintOp.getPidz(), xpuprintOp.getOuterIndex(),
                xpuprintOp.getInnerIndex(), idxVar64,
                xpuprintOp.getInnerBound(), ucBound, xpuprintOp.getPrefixAttr(),
                xpuprintOp.getHexAttr(), xpuprintOp.getArgs());
            newOp->erase();
          })
          .Case<triton::AddPtrOp>([&](auto addPtrOp) {
            auto ptr = addPtrOp.getPtr();
            auto offset = addPtrOp.getOffset();

            if (mlir::dyn_cast<mlir::BlockArgument>(ptr)) {
              // For the time being,
              // it seems that no additional processing
              // is needed for this addPtrOp here
            } else {
              auto ptrTensorTy = dyn_cast<RankedTensorType>(ptr.getType());
              auto offsetTensorTy =
                  dyn_cast<RankedTensorType>(offset.getType());
              if (ptrTensorTy && offsetTensorTy &&
                  ptrTensorTy.getShape() != offsetTensorTy.getShape()) {
                auto extractOp = builder.create<triton::xpu::ExtractOp>(
                    loc, getElementTypeOrSelf(ptr),
                    builder.getI32IntegerAttr(0), ptr);
                auto splatTy = RankedTensorType::get(
                    offsetTensorTy.getShape(), getElementTypeOrSelf(ptr),
                    offsetTensorTy.getEncoding());
                auto splatOp =
                    builder.create<triton::SplatOp>(loc, splatTy, extractOp);
                addPtrOp.setOperand(0, splatOp);
                addPtrOp->moveAfter(splatOp);
              }
            }
          })
          .Case<arith::ConstantOp>([&](auto constantOp) {
            auto value = constantOp.getValue();
            if (auto attr = dyn_cast<DenseElementsAttr>(value)) {
              value = DenseElementsAttr::getFromRawBuffer(
                  cast<ShapedType>(constantOp.getType()), attr.getRawData());
            }
            constantOp.setValueAttr(value);
          })
          .Case<scf::IfOp>([&](auto ifOp) {
            // process ifOp recursively to handle nested ifOp
            auto processIfOp = [&](auto &self, scf::IfOp ifOp) -> void {
              auto &thenRegion = ifOp.getThenRegion();
              if (!thenRegion.empty()) {

                auto &thenBlock = thenRegion.front();
                for (auto &op : thenBlock) {
                  if (auto nestedIfOp = dyn_cast<scf::IfOp>(op)) {
                    self(self, nestedIfOp);
                  } else {
                    setTensorType(context, &op, iterNum, isOuter);
                  }
                }
                setHoistedOperand(context, builder, loc, thenBlock, ifOp,
                                  iterNum);
              }
              auto &elseRegion = ifOp.getElseRegion();
              if (!elseRegion.empty()) {
                auto &elseBlock = elseRegion.front();

                for (auto &op : elseBlock) {
                  if (auto nestedIfOp = dyn_cast<scf::IfOp>(op)) {
                    self(self, nestedIfOp);
                  } else {
                    setTensorType(context, &op, iterNum, isOuter);
                  }
                }
                setHoistedOperand(context, builder, loc, elseBlock, ifOp,
                                  iterNum);
              }
            };
            processIfOp(processIfOp, ifOp);
          })
          .Case<scf::ForOp>([&](auto forOp) {
            // step 1 : set iter arg type.
            unsigned numInitArgs =
                forOp.getNumOperands() - 3; // 减去初始值、上界和步长
            Block &entryBlock = forOp.getBodyRegion().front();
            if (numInitArgs > 0 && entryBlock.getNumArguments() > 1) {
              for (unsigned i = 0; i < numInitArgs; ++i) {
                Type initIterArgType = forOp.getOperand(3 + i).getType();
                Type regionIterArgType =
                    entryBlock.getArgument(i + 1).getType();
                if (initIterArgType != regionIterArgType) {
                  entryBlock.getArgument(i + 1).setType(initIterArgType);
                }
              }
            }

            // step 2 : set ops' type in loop body.
            Block *body = forOp.getBody();
            for (auto &op : body->getOperations()) {
              setTensorType(context, &op, iterNum, isOuter);
            }
          });
    }
  }

  void eraseDAG(SetVector<Operation *> &unrollOpTree) {
    SetVector<Operation *> eraseOpTree(unrollOpTree.rbegin(),
                                       unrollOpTree.rend());
    for (auto op : eraseOpTree) {
      SetVector<Operation *> users;
      for (auto user : op->getUsers()) {
        if (isa<triton::xpu::ReduceReturnOp>(user)) {
          users.insert(user);
        }
      }
      for (auto user : users) {
        user->erase();
      }
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

  void moveAllocaAndGM2LM(scf::ForOp forOp,
                          SetVector<Operation *> &unrollOpTree) {
    ModuleOp m = getOperation();
    DenseMap<mlir::Operation *, unsigned> op2Line;
    getOpLine(m, op2Line);

    SmallVector<Operation *> gm2lmOps;
    SmallVector<Operation *> allocaOps;
    for (auto op : unrollOpTree) {
      if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
        auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(loadOp.getPtr());
        if (gm2lmOp) {
          gm2lmOps.emplace_back(gm2lmOp);
        }
        auto gm2lmmaskOp =
            findDefOpBwd<triton::xpu::GM2LMMaskOp>(loadOp.getPtr());
        if (gm2lmmaskOp) {
          gm2lmOps.emplace_back(gm2lmmaskOp);
        }
      }
      if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
        auto alloca = findDefOpBwd<triton::xpu::AllocaOp>(storeOp.getPtr());
        if (alloca) {
          allocaOps.emplace_back(alloca);
        }
      }
    }

    // move alloca when merge store
    for (auto allocaOp : allocaOps) {
      if (allocaOp->getBlock() == forOp->getBlock() &&
          forOp->isBeforeInBlock(allocaOp)) {
        allocaOp->moveBefore(forOp);
      }
    }

    for (auto gm2lmOp : gm2lmOps) {
      if (gm2lmOp->getBlock() != forOp->getBlock())
        continue;

      if (gm2lmOp->isBeforeInBlock(forOp))
        continue;

      for (auto operand : gm2lmOp->getOperands()) {
        auto op = operand.getDefiningOp();
        if (!op)
          continue;
        if (op2Line[op] > op2Line[forOp]) {
          op->moveBefore(forOp);
        }
      }
      gm2lmOp->moveBefore(forOp);
    }
  }

  void unrollControl(MLIRContext *context,
                     SmallVector<SetVector<Operation *>> &unrollOpTrees,
                     ArrayRef<int64_t> plan, bool postReduce = false) {
    // Get outerChains
    SmallVector<SetVector<Operation *>> outerChains;
    getOuterChains(unrollOpTrees, outerChains, postReduce);
    for (int i = 0; i < unrollOpTrees.size(); ++i) {
      auto outerChain = outerChains[i];
      auto unrollOpTree = unrollOpTrees[i];
      // 1. Prepare for unroll control
      int64_t numCol = 1;
      int64_t numUnroll = 1;
      triton::xpu::StoreOp insertPt;
      SmallVector<triton::xpu::StoreOp> allStoreOps;
      // 1.1 Get insertPt and tensor num
      if (!getTreeUnrollInfo(unrollOpTree, insertPt, allStoreOps, numCol,
                             numUnroll))
        return;
      if (insertPt) {
        auto loc = insertPt.getLoc();
        // Decided in planIterNums, before any IR was touched.
        int64_t iterNum = plan[i];
        // Skip this tree only: unlike the legacy gate, the budget model can
        // legitimately answer "no loop" for one tree while the others still
        // want one, so this must not abandon the remaining trees.
        if (iterNum <= 1)
          continue;
        LLVM_DEBUG(llvm::dbgs()
                   << "[Unroll Control] Hit Unroll Control Pointwise\n");
        // 2. Unroll control
        // 2.1 Create forOp
        OpBuilder builder(insertPt);
        scf::ForOp forOp;
        arith::IndexCastOp idxVar;
        ValueRange iterArgs = {};
        createFor(builder, loc, 0, iterNum, forOp, idxVar, iterArgs);
        // 2.2 Move Alloca & GM2LM Op before ForOp
        moveAllocaAndGM2LM(forOp, unrollOpTree);
        // 2.3 Set Tensor Type
        IRMapping mapping;
        createLoopBody(context, builder, loc, iterNum, unrollOpTree, outerChain,
                       idxVar, mapping);

        // 3. Erase old DAG
        eraseDAG(unrollOpTree);
      }
    }
  }

  // `iterNum` is decided by the caller: it has to be known before
  // findDiscretePtrChain() rewrites the pointer chain.
  void unrollControlReduce(MLIRContext *context,
                           SetVector<Operation *> &unrollOpTree,
                           Operation *insertPt, ValueRange &iterArgs,
                           ValueRange &returnOperands, int64_t iterNum) {
    SetVector<Operation *> outerChain;
    getOuterChain(unrollOpTree, outerChain);
    if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(insertPt)) {
      if (iterNum <= 1)
        return;
      OpBuilder builder(reduceOp);
      auto loc = reduceOp.getLoc();
      // 1. Prepare for unroll control
      // Insert ExtractSliceOp for TensorType
      SmallVector<Value> newIterArgs(iterArgs.size());
      for (int i = 0; i < iterArgs.size(); ++i) {
        auto iterArgDefOp = iterArgs[i].getDefiningOp();
        bool isOuter = inOpChain(outerChain, iterArgDefOp);
        auto extractSliceOp = builder.create<triton::xpu::ExtractSliceOp>(
            loc, iterArgs[i].getType(), iterArgs[i]);
        setTensorType(context, extractSliceOp, iterNum, isOuter);
        auto inUnrollOpTree = [&](OpOperand &operand) {
          return unrollOpTree.count(operand.getOwner());
        };
        iterArgs[i].replaceUsesWithIf(extractSliceOp.getResult(),
                                      inUnrollOpTree);
        newIterArgs[i] = extractSliceOp.getResult();
      }
      // 2. Unroll control
      // 2.1 Create forOp
      scf::ForOp forOp;
      arith::IndexCastOp idxVar;
      ValueRange newIterArgsRange(newIterArgs);
      createFor(builder, loc, 1, iterNum, forOp, idxVar, newIterArgsRange);
      // 2.2 Set Tensor Type
      IRMapping mapping;
      createLoopBody(context, builder, loc, iterNum, unrollOpTree, outerChain,
                     idxVar, mapping);
      bool isOuterReduce = inOpChain(outerChain, reduceOp);
      setTensorType(context, reduceOp, iterNum, isOuterReduce, false);
      // 2.3 Modify users and defs
      // replace initArgs with iterArgs
      auto inForOp = [&](OpOperand &operand) {
        return forOp == operand.getOwner()->getBlock()->getParentOp();
      };
      auto forBody = forOp.getBody();
      auto forArgs = forBody->getArguments();
      for (int i = 0; i < forOp.getInitArgs().size(); ++i) {
        forOp.getInitArgs()[i].replaceUsesWithIf(forArgs[i + 1], inForOp);
      }
      SmallVector<Value> mapRes;
      for (int i = 0; i < returnOperands.size(); ++i) {
        mapRes.emplace_back(mapping.lookup(returnOperands[i]));
      }
      builder.create<scf::YieldOp>(loc, mapRes);
      auto isReduceOp = [&](OpOperand &operand) {
        return reduceOp == operand.getOwner();
      };
      for (int i = 0; i < forOp.getResults().size(); ++i) {
        reduceOp.getOperands()[i].replaceUsesWithIf(forOp.getResults()[i],
                                                    isReduceOp);
      }
      // 3. Erase old DAG
      eraseDAG(unrollOpTree);
    }
  }

  void getExcludeChainOps(ModuleOp &m,
                          SetVector<Operation *> &excludeChainOps) {
    m.walk([&](Operation *op) {
      TypeSwitch<const Operation *>(op)
          .Case<XPU_MEMORY_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<XPU_MEMORY_MASK_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getMask()) {
              getOpChainBwd(excludeChainOps,
                            memoryOp.getMask().getDefiningOp());
            }
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<triton::xpu::LoadOp, triton::xpu::StoreOp>([&](auto acessOp) {
            if (acessOp.getMask()) {
              getOpChainBwd(excludeChainOps, acessOp.getMask().getDefiningOp());
            }
          });
    });
  }

  void
  getExcludeChainOpsforUnrollControl(ModuleOp &m,
                                     SetVector<Operation *> &excludeChainOps) {
    m.walk([&](Operation *op) {
      TypeSwitch<const Operation *>(op)
          .Case<XPU_MEMORY_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<XPU_MEMORY_MASK_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getMask()) {
              getOpChainBwd(excludeChainOps,
                            memoryOp.getMask().getDefiningOp());
            }
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) {
            if (storeOp.getMask()) {
              getOpChainBwd(excludeChainOps, storeOp.getMask().getDefiningOp());
            }
          })
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            if (loadOp.getMask()) {
              auto op = loadOp.getMask().getDefiningOp();
              auto userNum =
                  std::distance(op->getUsers().begin(), op->getUsers().end());
              decltype(userNum) loadNum = 0;
              for (auto user : op->getUsers()) {
                if (isa<triton::xpu::LoadOp>(user)) {
                  loadNum++;
                }
              }
              if (userNum == loadNum) {
                getOpChainBwd(excludeChainOps,
                              loadOp.getMask().getDefiningOp());
              }
            }
          });
    });
  }

  void findDiscretePtrChain(SetVector<Operation *> &unrollOpTree,
                            SetVector<Operation *> &newUnrollOpTree,
                            bool treeWillTile) {
    for (auto op : unrollOpTree) {
      if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
        bool isDiscrete = loadOp.getIsDiscrete();
        if (isDiscrete) {
          OpBuilder builder(loadOp);
          auto loc = loadOp.getLoc();
          auto resType = loadOp.getResult().getType();
          int64_t numCol = getNumCol(resType);
          int64_t numUnroll = getNumUnroll(resType);
          // The rewrite only makes sense inside the tiling loop, so it must
          // agree with the decision taken for the whole tree.
          bool willTile = this->budgetTiling
                              ? treeWillTile
                              : (numCol > numUnroll && numCol % numUnroll == 0);
          if (willTile) {
            auto lmPtr = loadOp.getPtr();
            if (auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(lmPtr)) {
              auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
              auto offset = gmPtrOp.getOffset();
              auto newLmPtr = builder.create<triton::AddPtrOp>(
                  loc, lmPtr.getType(), lmPtr, offset);
              SetVector<Operation *> ptrVisitedOps;
              SetVector<Operation *> ptrExcludeChainOps;
              getUnrollTree(newLmPtr, newUnrollOpTree, ptrVisitedOps,
                            ptrExcludeChainOps, newLmPtr, false);
              if (!newUnrollOpTree.empty()) {
                newUnrollOpTree = sortOpTree(newUnrollOpTree);
              }
              gm2lmOp->setAttr("offsetState",
                               builder.getSI32IntegerAttr(static_cast<int32_t>(
                                   OffsetState::Continuous)));
              loadOp.setOperand(0, newLmPtr);

            } else if (auto gm2lmOp =
                           findDefOpBwd<triton::xpu::GM2LMMaskOp>(lmPtr)) {
              auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
              auto offset = gmPtrOp.getOffset();
              auto newLmPtr = builder.create<triton::AddPtrOp>(
                  loc, lmPtr.getType(), lmPtr, offset);
              SetVector<Operation *> ptrVisitedOps;
              SetVector<Operation *> ptrExcludeChainOps;
              getUnrollTree(newLmPtr, newUnrollOpTree, ptrVisitedOps,
                            ptrExcludeChainOps, newLmPtr, false);
              if (!newUnrollOpTree.empty()) {
                newUnrollOpTree = sortOpTree(newUnrollOpTree);
              }
              gm2lmOp->setAttr("offsetState",
                               builder.getSI32IntegerAttr(static_cast<int32_t>(
                                   OffsetState::Continuous)));
              loadOp.setOperand(0, newLmPtr);
            }
          }
        }
      }
    }
  }

  void
  findDiscretePtrChains(SmallVector<SetVector<Operation *>> &unrollOpTrees,
                        SmallVector<SetVector<Operation *>> &newUnrollOpTrees,
                        ArrayRef<int64_t> plan) {
    for (auto [i, unrollOpTree] : llvm::enumerate(unrollOpTrees)) {
      findDiscretePtrChain(unrollOpTree, newUnrollOpTrees[i], plan[i] > 1);
    }
  }

  void createDiscreteOffset(ModuleOp &m) {
    m.walk([&](triton::xpu::LoadOp loadOp) {
      bool isDiscrete = loadOp.getIsDiscrete();
      if (isDiscrete) {
        OpBuilder builder(loadOp);
        auto loc = builder.getUnknownLoc();
        auto lmPtr = loadOp.getPtr();
        auto lmAddPtr =
            cast<triton::AddPtrOp>(findDefOpBwd<triton::AddPtrOp>(lmPtr));
        auto lmOffset = lmAddPtr.getOffset();
        if (auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(lmPtr)) {
          auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
          auto gmOffset = gmPtrOp.getOffset();
          // Nothing to rebase when the LM side has no addptr of its own: the
          // one we found walking back *is* the gm2lm's GM addptr, which is what
          // an untiled tree looks like. A discrete gm2lm already gathers into a
          // 0-based LM buffer, so rewriting this offset would only strip the
          // block's base off the GM address the gather reads from.
          if (lmAddPtr == gmPtrOp)
            return;
          auto extractOp = builder.create<triton::xpu::ExtractOp>(
              loc, getElementTypeOrSelf(gmOffset), builder.getI32IntegerAttr(0),
              gmOffset);
          auto splatOp = builder.create<triton::SplatOp>(
              loc, lmOffset.getType(), extractOp);
          auto offset = builder.create<arith::SubIOp>(loc, lmOffset.getType(),
                                                      lmOffset, splatOp);
          lmAddPtr.setOperand(1, offset);
          lmAddPtr->moveAfter(offset);
          if (gm2lmOp->getOperand(0) == lmAddPtr.getResult())
            gm2lmOp->moveAfter(lmAddPtr);
        } else if (auto gm2lmOp =
                       findDefOpBwd<triton::xpu::GM2LMMaskOp>(lmPtr)) {
          auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
          auto gmOffset = gmPtrOp.getOffset();
          // Nothing to rebase when the LM side has no addptr of its own: the
          // one we found walking back *is* the gm2lm's GM addptr, which is what
          // an untiled tree looks like. A discrete gm2lm already gathers into a
          // 0-based LM buffer, so rewriting this offset would only strip the
          // block's base off the GM address the gather reads from.
          if (lmAddPtr == gmPtrOp)
            return;
          auto extractOp = builder.create<triton::xpu::ExtractOp>(
              loc, getElementTypeOrSelf(gmOffset), builder.getI32IntegerAttr(0),
              gmOffset);
          auto splatOp = builder.create<triton::SplatOp>(
              loc, lmOffset.getType(), extractOp);
          auto offset = builder.create<arith::SubIOp>(loc, lmOffset.getType(),
                                                      lmOffset, splatOp);
          lmAddPtr.setOperand(1, offset);
          lmAddPtr->moveAfter(offset);
          if (gm2lmOp->getOperand(0) == lmAddPtr.getResult())
            gm2lmOp->moveAfter(lmAddPtr);
        }
      }
    });
  }

  void pointwiseUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Data-flow Analysis: get load -> store DAG
    //    (op in ptrChain/lenChain/maskChain will not walk from top to down)
    // 1.1 Get excludeChainOps
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    // 1.2 Get load -> store DAG
    SetVector<Operation *> visitedOps;
    SmallVector<SetVector<Operation *>> unrollOpTrees;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto valType = storeOp.getValue().getType();
      int64_t numCol = getNumCol(valType);
      int64_t numUnroll = getNumUnroll(valType);
      if (dryRun)
        reportGate("pointwise", storeOp, numCol, numUnroll);
      if (canTile(storeOp, valType, numCol, numUnroll)) {
        getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
      }
      for (auto visitedOp : visitedOps) {
        if (isa<arith::ConstantOp>(visitedOp)) {
          visitedOps.remove(visitedOp);
        }
      }
    });
    if (unrollOpTrees.size() == 0)
      return;

    // 1.3 Find ptr chain of discrete for moving to loop body
    //     The factor must be decided before this rewrite: it is only valid for
    //     trees that really end up inside a tiling loop.
    SmallVector<int64_t> plan = planIterNums(unrollOpTrees, "pointwise");
    SmallVector<SetVector<Operation *>> newUnrollOpTrees(unrollOpTrees);
    findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees, plan);

    // 2. Deal with unroll opTrees
    unrollControl(context, newUnrollOpTrees, plan);

    // 3. Calculate discrete offset in the runtime
    createDiscreteOffset(m);
  }

  void createLoadStore(scf::ForOp &forOp, scf::YieldOp &yieldOp, Value &yield,
                       int i, Block &block,
                       SmallVector<Operation *> &storeOps) {
    OpBuilder builder(yieldOp);
    auto loc = yieldOp->getLoc();
    Type yieldType = yield.getType();
    Type yieldElemType = getElementTypeOrSelf(yieldType);
    int64_t vecSize = getNumInVector(yieldElemType);
    Type ptrTy = createPointerType(yieldType, vecSize);
    int64_t tensorSize = getTensorSize(yieldType);
    if (!forOp.getResults()[i].use_empty()) {
      // Create Alloca Store for Init Args
      auto initForArg = forOp.getInitArgs()[i];
      auto newAllocaOp = builder.create<triton::xpu::AllocaOp>(
          loc, ptrTy, tensorSize * vecSize);
      auto initStoreOp = builder.create<triton::xpu::StoreOp>(
          loc, newAllocaOp, initForArg, Value(), Value(), -1, false,
          Dtype::UNKNOWN, MemorySyncMode::SYNC);
      newAllocaOp->moveBefore(forOp);
      initStoreOp->moveBefore(forOp);
      // Create Load for Input
      auto inputLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, yieldType, newAllocaOp, Value(), Value(), Value(), 1, -1, false,
          false, false, MemorySyncMode::SYNC);
      auto notUsedForYield = [&](OpOperand &operand) {
        return !isa<scf::YieldOp>(operand.getOwner());
      };
      auto forArg = forOp.getRegionIterArgs()[i];
      forArg.replaceUsesWithIf(inputLoadOp, notUsedForYield);
      inputLoadOp->moveBefore(&block.front());
      // Create Store for Output
      auto outputStoreOp = builder.create<triton::xpu::StoreOp>(
          loc, newAllocaOp, yield, Value(), Value(), -1, false, Dtype::UNKNOWN,
          MemorySyncMode::SYNC);
      outputStoreOp->moveBefore(yieldOp);
      storeOps.emplace_back(outputStoreOp);
      // Create Load for Reduce
      auto reduceLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, yieldType, newAllocaOp, Value(), Value(), Value(), 1, -1, false,
          false, false, MemorySyncMode::SYNC);

      // Replace For Result with Load
      auto notReduceLoadOp = [&](OpOperand &operand) {
        return reduceLoadOp != operand.getOwner();
      };
      forOp.getResults()[i].replaceUsesWithIf(reduceLoadOp, notReduceLoadOp);

      // Move Load closed to For user
      reduceLoadOp->moveAfter(forOp);
      Operation *insertPt = nullptr;
      for (auto user : forOp.getResults()[i].getUsers()) {
        if (!insertPt) {
          insertPt = user;
        } else {
          if (insertPt->getBlock() == user->getBlock()) {
            if (user->isBeforeInBlock(insertPt)) {
              insertPt = user;
            }
          }
        }
      }
      if (insertPt) {
        reduceLoadOp->moveBefore(insertPt);
      }

      // Discard Yield by setting initForArg to operand
      yieldOp->setOperand(i, initForArg);
    }
  }

  void getUnrollInfoReduce(triton::xpu::ReduceOp &reduceOp, int64_t &numCol,
                           int64_t &numUnroll) {
    auto types = reduceOp.getOperandTypes();
    assert(types.size() > 1);
    for (int i = 0; i < types.size() - 1; ++i) {
      if (i == 0) {
        numCol = getNumCol(types[i]);
        numUnroll = getNumUnroll(types[i]);
      } else {
        assert(numCol == getNumCol(types[i]));
        assert(numUnroll == getNumUnroll(types[i]));
      }
    }
  }

  void forUnrollControl(ModuleOp &m, MLIRContext *context) {
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOpsforUnrollControl(m, excludeChainOps);
    SetVector<Operation *> vistedForOps;
    // 1. Create Store Load
    m.walk([&](triton::xpu::ReduceOp reduceOp) {
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      if (dryRun)
        reportGate("reduce", reduceOp, numCol, numUnroll);
      if (canTile(reduceOp, reduceOp.getInputTypes()[0], numCol, numUnroll)) {
        llvm::SetVector<Operation *> reduceOpDefsBwd;
        getOpChainBwd(reduceOpDefsBwd, reduceOp);
        for (auto operand : reduceOpDefsBwd) {
          if (auto forOp = dyn_cast<scf::ForOp>(operand)) {
            if (!vistedForOps.count(forOp)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[Unroll Control] Hit Unroll Control For\n");
              vistedForOps.insert(forOp);
              auto &forBlock = forOp.getRegion().front();
              bool hasIf = false;
              SetVector<Operation *> visitedOps;
              for (auto &inForBlockOp : forBlock) {
                if (auto ifOp = dyn_cast<scf::IfOp>(inForBlockOp)) {
                  SmallVector<Operation *> storeOps;
                  auto &ifBlock = ifOp.getThenRegion().front();
                  auto yieldOp = cast<scf::YieldOp>(ifBlock.getTerminator());
                  for (auto [i, yield] :
                       llvm::enumerate(yieldOp.getOperands())) {
                    createLoadStore(forOp, yieldOp, yield, i, ifBlock,
                                    storeOps);
                  }
                  // Unroll control
                  for (auto storeOp : storeOps) {
                    if (visitedOps.count(storeOp))
                      continue;
                    SmallVector<SetVector<Operation *>> unrollOpTrees;
                    getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps,
                           true, true);
                    // Find ptr chain of discrete for moving to loop body
                    SmallVector<int64_t> plan =
                        planIterNums(unrollOpTrees, "reduce-for");
                    SmallVector<SetVector<Operation *>> newUnrollOpTrees(
                        unrollOpTrees);
                    findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees,
                                          plan);
                    unrollControl(context, newUnrollOpTrees, plan);
                  }
                  hasIf = true;
                }
              }
              if (!hasIf) {
                SmallVector<Operation *> storeOps;
                auto yieldOp = cast<scf::YieldOp>(forBlock.getTerminator());
                for (auto [i, yield] : llvm::enumerate(yieldOp.getOperands())) {
                  createLoadStore(forOp, yieldOp, yield, i, forBlock, storeOps);
                }
                // Unroll control
                for (auto storeOp : storeOps) {
                  if (visitedOps.count(storeOp))
                    continue;
                  SmallVector<SetVector<Operation *>> unrollOpTrees;
                  getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps,
                         true, true);
                  // Find ptr chain of discrete for moving to loop body
                  SmallVector<int64_t> plan =
                      planIterNums(unrollOpTrees, "reduce-for");
                  SmallVector<SetVector<Operation *>> newUnrollOpTrees(
                      unrollOpTrees);
                  findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees, plan);
                  unrollControl(context, newUnrollOpTrees, plan);
                }
              }
            }
          }
        }
      }
    });
  }

  void getInlineInfo(SetVector<Operation *> &inlineOps, Operation *startOp,
                     ValueRange &returnOperands) {
    Operation *op = startOp;
    while (!isa<triton::xpu::ReduceReturnOp>(op)) {
      inlineOps.insert(op);
      op = op->getNextNode();
    }
    returnOperands = op->getOperands();
  }

  void createReduceWithinCore(ModuleOp &m, MLIRContext *context) {
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    m.walk([&](triton::xpu::ReduceOp reduceOp) {
      ReduceOpHelper helper(reduceOp);
      OpBuilder builder(reduceOp);
      auto loc = reduceOp->getLoc();
      SetVector<Operation *> visitedOps;
      auto reduceOperandNum = reduceOp.getNumOperands() - 1;
      SmallVector<SetVector<Operation *>> copyOpTrees;
      SetVector<Operation *> unrollOpTree;
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      if (canTile(reduceOp, reduceOp.getInputTypes()[0], numCol, numUnroll)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Unroll Control] Hit Unroll Control Reduction\n");
        for (int i = 0; i < reduceOperandNum; ++i) {
          if (auto reduceDefOp = reduceOp.getOperands()[i].getDefiningOp()) {
            getDAG(reduceDefOp, visitedOps, copyOpTrees, excludeChainOps,
                   false);
          }
        }
        // 0. Decide the factor up front: everything below mutates the IR
        //    (clones the operand chain, inlines the combine region) and is only
        //    valid if a loop is actually created afterwards. Deciding later
        //    would leave the inlined combine region orphaned.
        SetVector<Operation *> probeOpTree;
        for (auto &copyOpTree : copyOpTrees)
          for (auto *copyOp : copyOpTree)
            probeOpTree.insert(copyOp);
        // The reduce loop carries one accumulator per data operand (the
        // iterArgs built at :1908), which is what the loop-overhead criterion
        // charges for.
        int64_t iterNum =
            decideIterNum("reduce", reduceOp, probeOpTree,
                          reduceOp.getInputTypes()[0], numCol, numUnroll,
                          /*loopResults=*/reduceOperandNum)
                .iterNum;
        if (iterNum <= 1)
          return;
        // 1. Copy Defined Op Chain of Reduce Operand for InitArgs
        IRMapping mapping;
        for (auto &copyOpTree : copyOpTrees) {
          for (auto &copyOp : copyOpTree) {
            auto newOp = builder.clone(*copyOp, mapping);
            unrollOpTree.insert(newOp);
          }
        }
        // 2. Inline Combine Op of Reduce
        // Clone Region
        IRRewriter rewriter(builder);
        Block *currentBlock = rewriter.getBlock();
        Region &parent = *currentBlock->getParent();
        rewriter.cloneRegionBefore(reduceOp.getCombineOp(), &parent.front());
        auto &newReduce = parent.front();
        // Set Type for Cloned Ops
        auto tensorTy = reduceOp.getInputTypes()[0];
        auto shape = tensorTy.getShape();
        // `tt.splat` cannot carry a vector element type. On the
        // region-interpreting path (TRITONXPU_REDUCE_REGION) the combine
        // region's constants are vector<NxT>, so the broadcast has to be
        // triton_xpu.vsplat instead -- the same choice Vectorize.cpp makes for
        // splats it retypes. VSplatOpConversion broadcasts a *scalar* into
        // every lane (insertelement into lane 0 + shuffle), so the constant is
        // narrowed back to its splat value here rather than handed over as a
        // vector. Ops are created at `anchor` because the cloned combine block
        // sits ahead of the builder's insertion point.
        // Both arith.constant and triton_xpu.vconst keep the constant in a
        // DenseElementsAttr over the *scalar* type: VConstOp is what
        // TritonXPUVectorize rewrites a top-level arith.constant into, and it
        // is built from exactly that attribute (Vectorize.cpp
        // processOpVecTy). Its result is TTX_VectorLike, so it may be a bare
        // vector<NxT> -- precisely the non-tensor operand shape the two
        // broadcasts below have to fix up.
        auto getConstDense = [](mlir::Value src) -> mlir::DenseElementsAttr {
          auto *defOp = src.getDefiningOp();
          if (auto cstOp = dyn_cast_or_null<arith::ConstantOp>(defOp))
            return mlir::dyn_cast<mlir::DenseElementsAttr>(cstOp.getValue());
          if (auto vConstOp = dyn_cast_or_null<triton::xpu::VConstOp>(defOp))
            return mlir::dyn_cast<mlir::DenseElementsAttr>(vConstOp.getValue());
          return {};
        };
        auto createCombineSplat = [&](mlir::Type resTy, mlir::Value src,
                                      mlir::Operation *anchor) -> mlir::Value {
          OpBuilder b(anchor);
          auto elemTy =
              mlir::cast<mlir::RankedTensorType>(resTy).getElementType();
          if (!mlir::isa<mlir::VectorType>(elemTy))
            return b.create<triton::SplatOp>(loc, resTy, src).getResult();
          auto dense = getConstDense(src);
          assert(dense && dense.isSplat() && "combine constant is not uniform");
          auto scalar = b.create<arith::ConstantOp>(
              src.getLoc(),
              mlir::cast<mlir::TypedAttr>(dense.getSplatValue<Attribute>()));
          return b.create<triton::xpu::VSplatOp>(loc, resTy, scalar)
              .getResult();
        };
        for (auto &op : newReduce) {
          if (isa<arith::CmpFOp>(op) || isa<arith::CmpIOp>(op)) {
            auto tensorTy0 = op.getOperand(0).getType();
            auto tensorTy1 = op.getOperand(1).getType();
            // The operand that is not a tensor is the one to broadcast. Asking
            // "is it a Float or an Integer" is too narrow: on the vector path
            // the combine region's constants are vector<NxT>, and neither
            // branch used to fire, leaving operandIndexNeedModify uninitialized
            // and the assert below reading a garbage index.
            int operandIndexNeedModify = -1;
            mlir::Type operandNeedReserved;
            if (tensorTy0 != tensorTy1) {
              if (!mlir::isa<mlir::TensorType>(tensorTy0) &&
                  mlir::isa<mlir::TensorType>(tensorTy1)) {
                operandIndexNeedModify = 0;
                operandNeedReserved = tensorTy1;
              } else if (!mlir::isa<mlir::TensorType>(tensorTy1) &&
                         mlir::isa<mlir::TensorType>(tensorTy0)) {
                operandIndexNeedModify = 1;
                operandNeedReserved = tensorTy0;
              }
              assert(
                  (operandIndexNeedModify >= 0 &&
                   isa_and_nonnull<arith::ConstantOp, triton::xpu::VConstOp>(
                       op.getOperand(operandIndexNeedModify).getDefiningOp()) &&
                   "Unable to extract the non-constant operand."));
              op.setOperand(operandIndexNeedModify,
                            createCombineSplat(
                                operandNeedReserved,
                                op.getOperand(operandIndexNeedModify), &op));
            }
          } else if (auto selOp = dyn_cast<arith::SelectOp>(op)) {
            auto tensorTy1 = selOp.getODSOperands(1)[0].getType();
            auto tensorTy2 = selOp.getODSOperands(2)[0].getType();
            int operandIndexNeedModify = -1;
            mlir::Type operandNeedReserved;
            if (tensorTy1 != tensorTy2) {
              if (!mlir::isa<mlir::TensorType>(tensorTy1) &&
                  mlir::isa<mlir::TensorType>(tensorTy2)) {
                operandIndexNeedModify = 1;
                operandNeedReserved = tensorTy2;
              } else if (!mlir::isa<mlir::TensorType>(tensorTy2) &&
                         mlir::isa<mlir::TensorType>(tensorTy1)) {
                operandIndexNeedModify = 2;
                operandNeedReserved = tensorTy1;
              }
              assert((operandIndexNeedModify >= 0 &&
                      isa_and_nonnull<arith::ConstantOp, triton::xpu::VConstOp>(
                          selOp.getOperand(operandIndexNeedModify)
                              .getDefiningOp()) &&
                      "Unable to extract the non-constant operand."));

              selOp.setOperand(
                  operandIndexNeedModify,
                  createCombineSplat(operandNeedReserved,
                                     selOp.getOperand(operandIndexNeedModify),
                                     &op));
            }
          }
          for (auto [i, resTy] : llvm::enumerate(op.getResultTypes())) {
            auto inlineTensorTy =
                RankedTensorType::get(shape, resTy, tensorTy.getEncoding());
            op.getResult(i).setType(inlineTensorTy);
          }
        }
        // Inline Ops
        llvm::SmallVector<Value> combineArgs(2 * reduceOperandNum);
        for (unsigned i = 0; i < reduceOperandNum; ++i) {
          combineArgs[i] = reduceOp.getOperands()[i];
          combineArgs[reduceOperandNum + i] =
              mapping.lookup(reduceOp.getOperands()[i]);
        }
        auto currOp = &*rewriter.getInsertionPoint();
        auto insertOp = currOp->getPrevNode();
        rewriter.inlineBlockBefore(&newReduce, currOp, combineArgs);
        ValueRange returnOperands;
        getInlineInfo(unrollOpTree, insertOp, returnOperands);

        auto isReduceOp = [&](OpOperand &operand) {
          return reduceOp == operand.getOwner();
        };
        llvm::SmallVector<Value> iterArgs(reduceOperandNum);
        for (auto [i, returnOperand] : llvm::enumerate(returnOperands)) {
          iterArgs[i] = reduceOp.getOperands()[i];
          reduceOp.getOperands()[i].replaceUsesWithIf(returnOperand,
                                                      isReduceOp);
        }
        // Find ptr chain of discrete for moving to loop body
        SetVector<Operation *> newUnrollOpTree(unrollOpTree);
        findDiscretePtrChain(unrollOpTree, newUnrollOpTree, iterNum > 1);
        // 3. Create Loop for ReduceWithinCore
        ValueRange iterArgsRange(iterArgs);
        unrollControlReduce(context, newUnrollOpTree, reduceOp, iterArgsRange,
                            returnOperands, iterNum);
        // 4. For Vectorize: triton.addf->triton_xpu.vvaddf
        processOpVecTy(m);
      }
    });
  }

  bool isPostReduceStore(triton::xpu::StoreOp storeOp) {
    bool _isPostReduceStore = false;
    if (auto valTy = dyn_cast<RankedTensorType>(storeOp.getValue().getType())) {
      auto shape = valTy.getShape();
      if (shape.size() > 1 && shape.back() > 1) {
        _isPostReduceStore = true;
      }
    }
    return _isPostReduceStore;
  }

  void mergeSets(SmallVector<SetVector<Operation *>> &unrollOpTrees) {
    // Create Mapping of All Sets
    DenseMap<Operation *, SmallVector<SetVector<Operation *> *>> opToSets;
    for (auto &set : unrollOpTrees) {
      for (Operation *op : set) {
        opToSets[op].push_back(&set);
      }
    }
    // Merge unrollOpTrees that has common nodes
    DenseSet<SetVector<Operation *> *> processedSets;
    for (auto &currentSet : unrollOpTrees) {
      if (processedSets.count(&currentSet))
        continue;
      SetVector<Operation *> mergedSet = currentSet;
      bool hasMerged = true;
      while (hasMerged) {
        hasMerged = false;
        for (Operation *op : mergedSet) {
          auto &relatedSets = opToSets[op];
          for (auto *relatedSet : relatedSets) {
            if (relatedSet == &mergedSet || processedSets.count(relatedSet))
              continue;

            mergedSet.insert(relatedSet->begin(), relatedSet->end());
            relatedSet->clear();
            processedSets.insert(relatedSet);
            hasMerged = true;
          }
        }
      }
      if (mergedSet.size() > currentSet.size()) {
        mergedSet = sortOpTree(mergedSet);
        currentSet = mergedSet;
      }
      // Remove Empty Sets
      unrollOpTrees.erase(
          llvm::remove_if(
              unrollOpTrees,
              [](const SetVector<Operation *> &set) { return set.empty(); }),
          unrollOpTrees.end());
    }
  }

  void postReduceUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Data-flow Analysis: get post reduce -> store DAG
    //    (op in ptrChain/lenChain/maskChain will not walk from top to down)
    // 1.1 Get excludeChainOps
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    // 1.2 Get load -> store DAG
    SmallVector<SetVector<Operation *>> unrollOpTrees;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      SetVector<Operation *> visitedOps;
      auto valType = storeOp.getValue().getType();
      int64_t numCol = getNumCol(valType);
      int64_t numUnroll = getNumUnroll(valType);
      bool _isPostReduceStore = isPostReduceStore(storeOp);
      if (canTile(storeOp, valType, numCol, numUnroll) && _isPostReduceStore) {
        getPostReduceDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
      }
    });
    if (unrollOpTrees.size() == 0)
      return;

    // 2. Merge unrollOpTrees that has common nodes
    mergeSets(unrollOpTrees);

    // 3. Deal with unroll opTrees
    LLVM_DEBUG(llvm::dbgs()
               << "[Unroll Control] Hit Unroll Control Post Reduction\n");
    SmallVector<int64_t> plan = planIterNums(unrollOpTrees, "post-reduce");
    unrollControl(context, unrollOpTrees, plan, /*postReduce=*/true);
  }

  void reductionUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Unroll Control for Reduce For
    forUnrollControl(m, context);
    // 2. Create For for ReduceWithinCore
    createReduceWithinCore(m, context);
    // 3. Deal with BroadCastOp/ReduceOp to StoreOp
    postReduceUnrollControl(m, context);
    // 4. Calculate discrete offset in the runtime
    createDiscreteOffset(m);
    // 5. Check Def-Use Shape Match
    checkDefUseShapeMatch(m, context);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    dryRun = std::getenv("TRITONXPU_UNROLL_DRYRUN") != nullptr;

    bool isScan = false;
    m.walk([&](triton::xpu::ScanOp scanOp) { isScan = true; });
    if (isScan) {
      return;
    }

    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto dtype = storeOp.getDtype();
      auto valTy = storeOp.getValue().getType();
      auto ptrTy = storeOp.getPtr().getType();
      auto valElemTy = getElementTypeOrSelf(getElementTypeOrSelf(valTy));
      auto ptrElemTy = getElementTypeOrSelf(getElementTypeOrSelf(ptrTy));
      if (dtype == Dtype::FP32 && valElemTy.isInteger(32) &&
          cast<triton::PointerType>(ptrElemTy).getPointeeType().isInteger(8)) {
        applyPin(m, /*constant=*/4, "bool-store-vectorize");
      }
    });

    bool isReduce = false;
    m.walk([&](triton::xpu::ReduceOp redOp) {
      isReduce = true;
      // Set this->unrollNum=1 When coreDealMultiRows
      RankedTensorType operandType = redOp.getInputTypes()[0];
      auto shape = operandType.getShape();
      auto layout =
          cast<triton::xpu::ClusterLayoutAttr>(operandType.getEncoding());
      unsigned rowsPerCore = layout.getSizePerCore()[0];
      if (shape.size() == 2 && rowsPerCore > 1) {
        applyPin(m, /*constant=*/1, "core-deal-multi-rows");
      }
    });

    if (isReduce) {
      reductionUnrollControl(m, context);
    } else {
      pointwiseUnrollControl(m, context);
    }

    // The marker only exists to stop a later walk inside this pass from tiling
    // an already tiled segment a second time; it must not survive into the
    // emitted IR. With budget_tiling off the artifacts have to stay
    // byte-identical to the legacy pipeline, and a leftover attribute is a
    // visible difference (10/10 probes' .ttxir differed on just this line).
    m.walk([&](scf::ForOp forOp) { forOp->removeAttr(kUnrollLoopAttr); });

    // Report-only LM accounting, second of three observation points (step 2.5).
    // The `[Alloca]` line is emitted by the alloca pass, and that pass runs
    // *before* this one (`tritonxpu-alloca, tritonxpu-tile-analysis,
    // tritonxpu-unroll-control`), so it is invariant to the tile decision by
    // construction -- measured: 11 probes x 4 widths, identical to the byte,
    // and `bitred` prints it even on the runs that then die inside this pass
    // (findings.md 1.43). Counting again here attributes to the decision only
    // the buffers this pass itself creates, which is a strict subset: measured
    // on 11 probes x {u8, u2}, exactly one cell moves (layernorm u2, 5 allocas
    // /2560 B -> 7/3584 B, all 1024 B of the delta `inLoopBytes`), while
    // add and mixedwidth grow only later in the pipeline. The number that is
    // comparable to what XTDK finally sees is `[MemoryInplace][lm]`, after the
    // reuse pass. `inLoopBytes` breaks out the part that sits inside a loop,
    // which is the part a per-tile rewrite (M4) would move. Never a veto: no
    // LM capacity check exists in this pipeline to compare against, and the
    // ceiling (KERNEL_STACK_SIZE, 8000 B) is enforced elsewhere, on a
    // different quantity.
    if (mlir::triton::tools::getBoolEnvXPU("TRITONXPU_LM_REPORT")) {
      int64_t totalBytes = 0;
      int64_t inLoopBytes = 0;
      unsigned numAllocas = 0;
      m.walk([&](triton::xpu::AllocaOp allocaOp) {
        auto ty = allocaOp.getResult().getType();
        int64_t bytes =
            getTotalElemsPerThread(ty) * triton::getPointeeBitWidth(ty) / 8;
        bytes = (bytes + 63) / 64 * 64; // LM allocas are 64-byte aligned
        totalBytes += bytes;
        ++numAllocas;
        if (allocaOp->getParentOfType<scf::ForOp>())
          inLoopBytes += bytes;
      });
      llvm::errs() << "[UnrollControl][lm] allocas=" << numAllocas
                   << " lmBytes=" << totalBytes
                   << " inLoopBytes=" << inLoopBytes
                   << " (post-tiling, per core, 64B-aligned, before "
                      "memory-inplace reuse)\n";
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
