#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace musa = mlir::triton::musa;
namespace musa_tle = mlir::triton::musa_tle;

namespace {

constexpr llvm::StringLiteral
    kAutoSharedLayoutAttr("musa_tle.auto_shared_layout");
constexpr llvm::StringLiteral kExplicitSqmmaAttr("musa_tle.explicit_sqmma");
constexpr llvm::StringLiteral kEnableEncodingRematerializationAttr(
    "tle.enable_encoding_rematerialization");

struct SelectedSqmmaConfig {
  SmallVector<unsigned, 3> instrShape;
  SmallVector<unsigned, 2> warpsPerCTA;
};

static std::optional<SelectedSqmmaConfig>
selectSqmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                  musa::SQMMAEltType operandType, int computeCapability) {
  if (numWarps < 4 || numWarps % 4 != 0)
    return std::nullopt;
  static constexpr unsigned kMN[] = {128, 64, 32, 16};
  static constexpr unsigned kK[] = {128, 64, 32, 16};

  std::optional<SelectedSqmmaConfig> best;
  unsigned bestCount = std::numeric_limits<unsigned>::max();
  unsigned bestVolume = 0;
  for (unsigned instM : kMN) {
    if (m % instM != 0)
      continue;
    for (unsigned instN : kMN) {
      if (n % instN != 0 ||
          !musa::isSupportedSqmmaInstrMN(operandType, instM, instN))
        continue;
      for (unsigned instK : kK) {
        if (k % instK != 0 ||
            !musa::isSupportedSqmmaForCapability(
                operandType, operandType, musa::SQMMAEltType::f32, instM,
                instN, instK, computeCapability))
          continue;
        for (unsigned warpsM = 4; warpsM <= numWarps; warpsM *= 2) {
          if (numWarps % warpsM != 0)
            continue;
          unsigned warpsN = numWarps / warpsM;
          // PH1 shared/SQMMA layouts encode each warp axis as a power-of-two
          // split.  Reject unsupported decompositions here so a future
          // 24-warp request (for example 8x3) falls back before layout
          // construction rather than failing in LinearLayout verification.
          if (!llvm::isPowerOf2_32(warpsM) ||
              !llvm::isPowerOf2_32(warpsN))
            continue;
          unsigned tileM = instM * (warpsM / 4);
          unsigned tileN = instN * warpsN;
          if (m % tileM != 0 || n % tileN != 0)
            continue;
          unsigned count = (m / tileM) * (n / tileN) * (k / instK);
          unsigned volume = instM * instN * instK;
          if (!best || count < bestCount ||
              (count == bestCount && volume > bestVolume)) {
            best = SelectedSqmmaConfig{{instM, instN, instK}, {warpsM, warpsN}};
            bestCount = count;
            bestVolume = volume;
          }
        }
      }
    }
  }
  return best;
}

static ttg::CGAEncodingAttr prependBufferDim(ttg::CGAEncodingAttr cgaLayout) {
  auto prependOne = [](SmallVector<unsigned> values) {
    values.insert(values.begin(), 1);
    return values;
  };
  SmallVector<unsigned> order{0};
  for (unsigned dim : cgaLayout.getCTAOrder())
    order.push_back(dim + 1);
  return ttg::CGAEncodingAttr::fromSplitParams(
      cgaLayout.getContext(), prependOne(cgaLayout.getCTAsPerCGA()),
      prependOne(cgaLayout.getCTASplitNum()), order);
}

static ttg::SwizzledSharedEncodingAttr
prependBufferDim(ttg::SwizzledSharedEncodingAttr encoding) {
  SmallVector<unsigned> order;
  for (unsigned dim : encoding.getOrder())
    order.push_back(dim + 1);
  order.push_back(0);
  return ttg::SwizzledSharedEncodingAttr::get(
      encoding.getContext(), encoding.getVec(), encoding.getPerPhase(),
      encoding.getMaxPhase(), order, prependBufferDim(encoding.getCGALayout()));
}

static ttg::LocalAllocOp findRootAlloc(Value value) {
  llvm::SmallPtrSet<void *, 16> visited;
  while (value && visited.insert(value.getAsOpaquePointer()).second) {
    if (auto alloc = value.getDefiningOp<ttg::LocalAllocOp>())
      return alloc;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Region *region = argument.getOwner()->getParent();
      auto partitions =
          dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
              region ? region->getParentOp() : nullptr);
      if (!partitions ||
          argument.getArgNumber() >= partitions.getExplicitCaptures().size())
        break;
      value = partitions.getExplicitCaptures()[argument.getArgNumber()];
      continue;
    }
    Operation *def = value.getDefiningOp();
    if (auto index = dyn_cast_or_null<ttg::MemDescIndexOp>(def))
      value = index.getSrc();
    else if (auto subslice = dyn_cast_or_null<ttg::MemDescSubsliceOp>(def))
      value = subslice.getSrc();
    else if (auto reinterpret =
                 dyn_cast_or_null<ttg::MemDescReinterpretOp>(def))
      value = reinterpret.getSrc();
    else if (auto reshape = dyn_cast_or_null<ttg::MemDescReshapeOp>(def))
      value = reshape.getSrc();
    else if (auto trans = dyn_cast_or_null<ttg::MemDescTransOp>(def))
      value = trans.getSrc();
    else
      break;
  }
  return {};
}

static musa::SQMMALayout inferLayout(ttg::MemDescType type) {
  auto order = ttg::getOrder(type);
  bool rowMajor = !order.empty() && order.front() + 1 == type.getRank();
  return rowMajor ? musa::SQMMALayout::row : musa::SQMMALayout::col;
}

static FailureOr<ttg::MemDescType>
inferTransposeSourceType(ttg::MemDescTransOp trans, ttg::MemDescType targetTy) {
  auto sourceTy = dyn_cast<ttg::MemDescType>(trans.getSrc().getType());
  Attribute targetEncoding = targetTy.getEncoding();
  ArrayRef<int32_t> order = trans.getOrder();
  if (!sourceTy || !targetEncoding || order.size() != targetTy.getRank() ||
      order.size() != sourceTy.getRank())
    return failure();

  SmallVector<int32_t> inverseOrder(order.size());
  SmallVector<int64_t> expectedSourceShape(order.size());
  for (auto [targetDim, sourceDim] : llvm::enumerate(order)) {
    if (sourceDim < 0 || static_cast<size_t>(sourceDim) >= order.size())
      return failure();
    inverseOrder[sourceDim] = targetDim;
    expectedSourceShape[sourceDim] = targetTy.getShape()[targetDim];
  }
  if (expectedSourceShape != sourceTy.getShape())
    return failure();

  Dialect &dialect = targetEncoding.getDialect();
  auto inferLayoutInterface =
      dyn_cast<tt::DialectInferLayoutInterface>(&dialect);
  if (!inferLayoutInterface)
    return failure();

  Attribute sourceEncoding;
  if (failed(inferLayoutInterface->inferTransOpEncoding(
          targetEncoding, targetTy.getShape(), inverseOrder, sourceEncoding,
          trans.getLoc())))
    return failure();

  SmallVector<int64_t> sourceAllocShape;
  ArrayRef<int64_t> targetAllocShape = targetTy.getAllocShape();
  if (!targetAllocShape.empty()) {
    if (targetAllocShape.size() < order.size())
      return failure();
    size_t prefixSize = targetAllocShape.size() - order.size();
    sourceAllocShape.append(targetAllocShape.begin(),
                            targetAllocShape.begin() + prefixSize);
    SmallVector<int64_t> sourceTail(order.size());
    ArrayRef<int64_t> targetTail = targetAllocShape.take_back(order.size());
    for (auto [targetDim, sourceDim] : llvm::enumerate(order))
      sourceTail[sourceDim] = targetTail[targetDim];
    sourceAllocShape.append(sourceTail.begin(), sourceTail.end());
  }

  return ttg::MemDescType::get(sourceTy.getShape(), sourceTy.getElementType(),
                               sourceEncoding, sourceTy.getMemorySpace(),
                               sourceTy.getMutableMemory(), sourceAllocShape);
}

static LogicalResult updateOperandLayout(musa_tle::SqmmaOp op,
                                         unsigned operandIdx,
                                         ttg::MUSASqmmaEncodingAttr mmaEnc) {
  Value operand = operandIdx == 0 ? op.getA() : op.getB();
  auto operandTy = cast<ttg::MemDescType>(operand.getType());
  int64_t elemBytes =
      std::max<int64_t>(1, (operandTy.getElementTypeBitWidth() + 7) / 8);
  ttg::LocalAllocOp root = findRootAlloc(operand);
  if (!root)
    return op.emitOpError("requires SQMMA operands rooted at ttg.local_alloc");
  if (!root->hasAttr(kAutoSharedLayoutAttr))
    return op.emitOpError(
        "requires layout=None and nv_mma_shared_layout=True for initial "
        "mthreads TLE SQMMA operands");

  auto order = ttg::getOrder(operandTy);
  auto cga = ttg::getCGALayout(operandTy.getEncoding());
  auto dotEncoding = ttg::DotOperandEncodingAttr::get(
      op.getContext(), operandIdx, mmaEnc, operandTy.getElementType());
  SmallVector<int64_t> physicalShape = musa::getMemDescPhysicalShape(operandTy);
  auto shared = musa::composeMusaOperandSharedLayout(
      dotEncoding, physicalShape, order, cga, operandTy.getElementType(),
      /*needTrans=*/false);
  if (!shared)
    return op.emitOpError(
               "failed to infer PH1 SQMMA shared layout for operand ")
           << operandIdx;

  auto desiredTy = ttg::MemDescType::get(
      operandTy.getShape(), operandTy.getElementType(), *shared,
      operandTy.getMemorySpace(), operandTy.getMutableMemory(),
      operandTy.getAllocShape());

  // trans_a/trans_b are represented by a descriptor-only memdesc_trans view.
  // Select the SQMMA layout for the final logical view, then propagate that
  // layout backwards so TME still writes a single physical landing buffer.
  Value landing = operand;
  ttg::MemDescType landingTy = desiredTy;
  if (auto trans = operand.getDefiningOp<ttg::MemDescTransOp>()) {
    auto sourceTy = inferTransposeSourceType(trans, desiredTy);
    if (failed(sourceTy))
      return op.emitOpError(
                 "failed to infer the physical SQMMA landing type through "
                 "ttg.memdesc_trans for operand ")
             << operandIdx;
    landing = trans.getSrc();
    landingTy = *sourceTy;
  }

  SmallVector<int64_t> landingPhysicalShape =
      musa::getMemDescPhysicalShape(landingTy);
  auto landingOrder =
      musa::getSharedOrder(landingTy.getEncoding(), landingPhysicalShape);
  if (failed(musa::resolveTMESwizzleConfigFromMatrixView(
          landingTy, landingPhysicalShape, landingOrder)))
    return op.emitOpError(
               "inferred physical SQMMA landing layout is not uniquely "
               "TME-compatible for operand ")
           << operandIdx;

  auto rootTy = cast<ttg::MemDescType>(root.getType());
  auto landingEncoding =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(landingTy.getEncoding());
  if (!landingEncoding)
    return op.emitOpError("requires a swizzled physical SQMMA landing layout");
  Attribute rootEncoding = landingEncoding;
  if (rootTy.getRank() == landingTy.getRank() + 1)
    rootEncoding = prependBufferDim(landingEncoding);
  else if (rootTy.getRank() != landingTy.getRank())
    return op.emitOpError("unsupported staged SQMMA allocation rank");

  auto newRootTy =
      ttg::MemDescType::get(rootTy.getShape(), rootTy.getElementType(),
                            rootEncoding, rootTy.getMemorySpace(),
                            rootTy.getMutableMemory(), rootTy.getAllocShape());
  ttg::MemDescType stageTy = landingTy;
  if (rootTy.getRank() == landingTy.getRank() + 1) {
    SmallVector<int64_t> stageShape(rootTy.getShape().drop_front());
    SmallVector<int64_t> stageAllocShape;
    ArrayRef<int64_t> rootAllocShape = rootTy.getAllocShape();
    if (rootAllocShape.size() == rootTy.getRank()) {
      ArrayRef<int64_t> trailingAllocShape = rootAllocShape.drop_front();
      stageAllocShape.append(trailingAllocShape.begin(),
                             trailingAllocShape.end());
    }
    stageTy = ttg::MemDescType::get(
        stageShape, rootTy.getElementType(), landingEncoding,
        rootTy.getMemorySpace(), rootTy.getMutableMemory(), stageAllocShape);
  }
  root.getResult().setType(newRootTy);
  landing.setType(landingTy);
  operand.setType(desiredTy);

  // Explicit warp-specialize captures are represented by operands on the
  // isolated partitions container and corresponding block arguments in every
  // partition. Changing the captured root value type does not update those
  // block arguments automatically, so keep the isolation boundary consistent
  // with the inferred TME/SQMMA layout.
  SmallVector<Value, 4> rootAliases{root.getResult()};
  root->getParentOfType<tt::FuncOp>().walk(
      [&](ttg::WarpSpecializePartitionsOp partitions) {
        for (auto [index, capture] :
             llvm::enumerate(partitions.getExplicitCaptures())) {
          if (capture != root.getResult())
            continue;
          for (Region &partition : partitions.getPartitionRegions()) {
            partition.getArgument(index).setType(newRootTy);
            rootAliases.push_back(partition.getArgument(index));
          }
        }
      });

  bool rowMajor = inferLayout(desiredTy) == musa::SQMMALayout::row;
  auto checkAndSet = [&](Operation *target) -> LogicalResult {
    if (!target)
      return success();
    if (auto oldIdx = musa::getSqmmaOpIdx(target)) {
      auto oldBytes = musa::getSqmmaElemBytes(target);
      bool oldRow = musa::getSqmmaRowMajor(target, rowMajor);
      if (*oldIdx != static_cast<int64_t>(operandIdx) || !oldBytes ||
          *oldBytes != elemBytes || oldRow != rowMajor)
        return target->emitOpError("conflicting TLE SQMMA consumer contract");
    }
    musa::setSqmmaAttrs(target, operandIdx, elemBytes, rowMajor);
    return success();
  };
  if (failed(checkAndSet(root.getOperation())) ||
      failed(checkAndSet(landing.getDefiningOp())) ||
      failed(checkAndSet(operand.getDefiningOp())))
    return failure();

  // Keep stage views and warp-specialize explicit captures type-consistent.
  bool changed = true;
  while (changed) {
    changed = false;
    root->getParentOfType<tt::FuncOp>().walk([&](ttg::MemDescIndexOp index) {
      auto srcTy = cast<ttg::MemDescType>(index.getSrc().getType());
      auto dstTy = cast<ttg::MemDescType>(index.getType());
      if (srcTy.getRank() != dstTy.getRank() + 1 ||
          !llvm::is_contained(rootAliases, index.getSrc()))
        return;
      if (index.getResult().getType() != stageTy) {
        index.getResult().setType(stageTy);
        (void)checkAndSet(index.getOperation());
        changed = true;
      }
    });
  }
  return success();
}

// A predicate around an async SQMMA is represented as an scf.if result.  The
// result is still safe when the branch values converge at a TLE wait (or are
// passed as the accumulator C operand of another SQMMA).  Keep the analysis
// deliberately narrow: arbitrary tensor users and branches producing an
// unrelated value remain rejected rather than risking an uncompleted async
// dependency.
static bool isAllowedAsyncIfValue(
    Value value, musa_tle::SqmmaOp root,
    llvm::SmallPtrSetImpl<void *> &visited) {
  if (!value || !visited.insert(value.getAsOpaquePointer()).second)
    return false;
  if (value == root.getD() || value == root.getC())
    return true;

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  Operation *def = result.getOwner();
  if (auto dot = dyn_cast<musa_tle::SqmmaOp>(def))
    return dot.getC() == root.getC();
  if (auto wait = dyn_cast<musa_tle::SqmmaWaitOp>(def)) {
    unsigned idx = result.getResultNumber();
    return idx == 0 &&
           isAllowedAsyncIfValue(wait.getInput(), root, visited);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned idx = result.getResultNumber();
    auto checkRegion = [&](Region &region) {
      if (region.empty())
        return true;
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (!yield || idx >= yield.getNumOperands())
        return false;
      llvm::SmallPtrSet<void *, 16> branchVisited;
      return isAllowedAsyncIfValue(yield.getOperand(idx), root,
                                   branchVisited);
    };
    return checkRegion(ifOp.getThenRegion()) &&
           (!ifOp.elseBlock() || checkRegion(ifOp.getElseRegion()));
  }
  if (def->getNumOperands() == 1 && def->getNumResults() == 1)
    return isAllowedAsyncIfValue(def->getOperand(0), root, visited);
  return false;
}

static bool hasAsyncCompletionBoundary(
    Value value, musa_tle::SqmmaOp root,
    llvm::SmallPtrSetImpl<void *> &visited) {
  if (!value || !visited.insert(value.getAsOpaquePointer()).second)
    return false;

  bool foundBoundary = false;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (isa<musa_tle::SqmmaWaitOp>(user)) {
      foundBoundary = true;
      continue;
    }
    // Feeding the C operand keeps an asynchronous accumulator chain intact;
    // the consuming dot is checked independently by verifyAsyncUses.
    if (isa<musa_tle::SqmmaOp>(user)) {
      if (use.getOperandNumber() == 2)
        continue;
      return false;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp());
      if (!ifOp) {
        auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
        unsigned idx = use.getOperandNumber();
        if (forOp && idx < forOp.getNumRegionIterArgs() &&
            root.getC() == forOp.getRegionIterArg(idx))
          return true;
        continue;
      }
      unsigned idx = use.getOperandNumber();
      if (idx >= ifOp.getNumResults())
        return false;
      llvm::SmallPtrSet<void *, 16> valueVisited;
      if (!isAllowedAsyncIfValue(ifOp.getResult(idx), root, valueVisited))
        return false;
      llvm::SmallPtrSet<void *, 16> nestedVisited;
      if (hasAsyncCompletionBoundary(ifOp.getResult(idx), root,
                                     nestedVisited))
        foundBoundary = true;
      continue;
    }
    return false;
  }
  return foundBoundary;
}

static LogicalResult verifyAsyncUses(musa_tle::SqmmaOp op) {
  for (OpOperand &use : op.getD().getUses()) {
    Operation *user = use.getOwner();
    if (auto next = dyn_cast<musa_tle::SqmmaOp>(user)) {
      if (use.getOperandNumber() == 2)
        continue;
    }
    // A loop-carried async accumulator may be yielded directly.  The
    // ConvertSqmmaToMTGPU pass rewrites that pattern to an opaque native
    // carrier and inserts one completion wait after the loop.  Restrict this
    // exception to the exact induction-argument pattern handled by that pass;
    // yielding an async value through any other region would still let it
    // escape without a completion boundary.
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      // A dot in a predicated branch may converge at an outer TLE wait.  This
      // is the branch-local async form emitted for a partial final K group.
      if (auto ifOp = dyn_cast<scf::IfOp>(yield->getParentOp())) {
        unsigned idx = use.getOperandNumber();
        if (idx < ifOp.getNumResults()) {
          llvm::SmallPtrSet<void *, 16> valueVisited;
          llvm::SmallPtrSet<void *, 16> boundaryVisited;
          if (isAllowedAsyncIfValue(ifOp.getResult(idx), op, valueVisited) &&
              hasAsyncCompletionBoundary(ifOp.getResult(idx), op,
                                         boundaryVisited))
            continue;
        }
      }
      auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
      unsigned resultIndex = use.getOperandNumber();
      if (!forOp || resultIndex >= forOp.getNumRegionIterArgs() ||
          op.getC() != forOp.getRegionIterArg(resultIndex) ||
          op->getNextNode() != yield.getOperation())
        return op.emitOpError(
            "async result must be consumed by musa_tle.sqmma_wait before "
            "non-loop scf.yield");
      continue;
    }
    if (isa<musa_tle::SqmmaWaitOp, scf::ForOp>(user))
      continue;
    return op.emitOpError(
        "async result must be consumed by musa_tle.sqmma_wait before "
        "ordinary tensor use");
  }
  return success();
}

// Materialize completion inside every predicated branch that yields an async
// SQMMA result.  A native accumulator cannot cross an SCF region as an
// ordinary tensor: even when the branch result eventually feeds another
// predicated SQMMA, the next region has no way to carry the native value.  A
// branch-local wait converts it back to a regular tensor before the region
// exit.  This is deliberately conservative and only applies to direct dot
// results; arbitrary region values remain rejected by verifyAsyncUses.
static void materializeBranchLocalWaits(
    ArrayRef<musa_tle::SqmmaOp> dots,
    llvm::DenseSet<Value> &synchronousIfResults,
    SmallVectorImpl<Operation *> &newWaits) {
  for (musa_tle::SqmmaOp dot : dots) {
    SmallVector<OpOperand *, 4> branchUses;
    for (OpOperand &use : dot.getD().getUses()) {
      auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
      if (!yield || !isa<scf::IfOp>(yield->getParentOp()))
        continue;
      branchUses.push_back(&use);
    }
    for (OpOperand *use : branchUses) {
      auto yield = cast<scf::YieldOp>(use->getOwner());
      auto ifOp = cast<scf::IfOp>(yield->getParentOp());
      unsigned idx = use->getOperandNumber();
      if (idx >= ifOp.getNumResults())
        continue;
      Value ifResult = ifOp.getResult(idx);
      // Do not duplicate a wait if a previous lowering round already
      // materialized one immediately before this yield.
      if (auto previous = dyn_cast_or_null<musa_tle::SqmmaWaitOp>(
              yield->getPrevNode())) {
        if (previous.getInput() == dot.getD()) {
          synchronousIfResults.insert(ifResult);
          continue;
        }
      }

      OpBuilder builder(yield);
      auto wait = musa_tle::SqmmaWaitOp::create(
          builder, dot.getLoc(), dot.getD().getType(), dot.getD(),
          builder.getI32IntegerAttr(0));
      yield->setOperand(idx, wait.getOutput());
      // Mark the converged value as synchronous so a later outer wait does
      // not issue a second native completion for this branch-local result.
      synchronousIfResults.insert(ifResult);
      newWaits.push_back(wait.getOperation());
    }
  }
}

static bool isMmaEncoded(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type &&
         isa_and_nonnull<ttg::MUSASqmmaEncodingAttr>(type.getEncoding());
}

// Convert a direct loop-carried accumulator to the SQMMA tensor encoding
// before replacing the TLE dot.  The native-carrier conversion runs later in
// the TTGPU pipeline, so the loop must already have a type-consistent SSA
// boundary when this pass erases the source musa_tle.sqmma operation.
//
// Keep this deliberately strict: the only supported form is
//
//   %next = musa_tle.sqmma ..., %iter
//   scf.yield %next
//
// and %iter may not have any other users.  More general control-flow paths
// need path-aware carrier analysis and must remain fail-closed.
static LogicalResult prepareDirectLoopCarriedSqmmaTypes(
    ArrayRef<musa_tle::SqmmaOp> dots,
    DenseMap<musa_tle::SqmmaOp, ttg::MUSASqmmaEncodingAttr> &encodings) {
  for (musa_tle::SqmmaOp dot : dots) {
    auto mmaEncoding = encodings.find(dot);
    if (mmaEncoding == encodings.end())
      continue;

    for (OpOperand &use : dot.getD().getUses()) {
      auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
      if (!yield || dot->getNextNode() != yield.getOperation())
        continue;
      auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!forOp)
        continue;


      unsigned idx = use.getOperandNumber();
      if (idx >= forOp.getNumRegionIterArgs() ||
          dot.getC() != forOp.getRegionIterArg(idx))
        continue;

      // The loop-carried value must be used solely as the next dot's C
      // operand and as the corresponding yield operand.  Mutating its type
      // in the presence of another tensor consumer would be unsound.
      bool isolated = true;
      for (OpOperand &iterUse : forOp.getRegionIterArg(idx).getUses()) {
        if (iterUse.getOwner() == dot.getOperation() &&
            iterUse.getOperandNumber() == 2)
          continue;
        isolated = iterUse.getOwner() == yield.getOperation() &&
                   iterUse.getOperandNumber() == idx;
        if (!isolated)
          break;
      }
      if (!isolated)
        return dot.emitOpError(
            "direct loop-carried accumulator has unsupported additional "
            "users");

      auto oldAccTy = dyn_cast<RankedTensorType>(dot.getC().getType());
      if (!oldAccTy)
        return dot.emitOpError(
            "direct loop-carried accumulator must be a ranked tensor");
      auto oldLoopResultTy =
          dyn_cast<RankedTensorType>(forOp.getResult(idx).getType());
      if (!oldLoopResultTy)
        return dot.emitOpError(
            "direct loop-carried loop result must be a ranked tensor");
      auto nativeTy = RankedTensorType::get(
          oldAccTy.getShape(), oldAccTy.getElementType(), mmaEncoding->second);

      Value init = forOp.getInitArgs()[idx];
      if (init.getType() != nativeTy) {
        OpBuilder builder(forOp);
        Value converted = ttg::ConvertLayoutOp::create(
            builder, dot.getLoc(), nativeTy, init);
        forOp.getInitArgsMutable()[idx].assign(converted);
      }

      // Keep all three sides of the SCF carried-value contract in sync.  The
      // TLE dot is replaced below by a MUSA SQMMA dot with this same result
      // type, allowing the later ConvertSqmmaToMTGPU pass to pack the loop
      // argument into its opaque native carrier.
      forOp.getRegionIterArg(idx).setType(nativeTy);
      forOp.getResult(idx).setType(nativeTy);
      dot.getResult().setType(nativeTy);

      // Preserve the original tensor contract for consumers outside the
      // loop.  The later MTGPU conversion replaces the loop result with an
      // unpacked carrier, and this conversion then restores the layout that
      // pointer/store users were type-checked against.  Keep a direct TLE
      // wait input in native form so the canonical external wait remains
      // discoverable by that conversion pass.
      SmallVector<OpOperand *> externalUses;
      for (OpOperand &resultUse : forOp.getResult(idx).getUses()) {
        if (isa<musa_tle::SqmmaWaitOp>(resultUse.getOwner()))
          continue;
        externalUses.push_back(&resultUse);
      }
      if (!externalUses.empty()) {
        OpBuilder builder(forOp);
        builder.setInsertionPointAfter(forOp);
        Value converted = ttg::ConvertLayoutOp::create(
            builder, dot.getLoc(), oldLoopResultTy, forOp.getResult(idx));
        for (OpOperand *resultUse : externalUses)
          resultUse->set(converted);
      }
    }
  }
  return success();
}

// Prepare a loop-carried SQMMA chain whose completion wait is required by a
// pipe slot release.  The direct-dot helper above intentionally handles only
// `dot -> scf.yield`; pipelined consumers instead form
//
//   iter -> dot -> dot ... -> wait -> scf.yield
//
// (the waits are interleaved between groups in the source, but the final
// accumulator value follows this single SSA chain).  Keeping the SCF value
// in the SQMMA encoding lets ConvertSqmmaToMTGPU carry it as an opaque native
// accumulator while retaining the waits needed before shared-memory reuse.
// Do not accept branches, arithmetic, or a chain with extra accumulator uses:
// those cases need path-aware lifetime analysis and remain fail-closed.
static LogicalResult prepareWaitLoopCarriedSqmmaTypes(
    ArrayRef<musa_tle::SqmmaOp> dots,
    DenseMap<musa_tle::SqmmaOp, ttg::MUSASqmmaEncodingAttr> &encodings) {
  for (musa_tle::SqmmaOp rootDot : dots) {
    auto mmaEncoding = encodings.find(rootDot);
    if (mmaEncoding == encodings.end())
      continue;

    // A root dot is discovered by walking backwards from each loop yield.
    // This naturally handles MMA_GROUP > 1, where the body has several dots
    // and only the last one is directly consumed by the final wait.
    for (Operation *ancestor = rootDot->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp()) {
      auto forOp = dyn_cast<scf::ForOp>(ancestor);
      if (!forOp || rootDot->getBlock() != forOp.getBody())
        continue;

      auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yield)
        continue;

      for (unsigned idx = 0; idx < forOp.getNumRegionIterArgs(); ++idx) {
        Value iterArg = forOp.getRegionIterArg(idx);
        auto oldAccTy = dyn_cast<RankedTensorType>(iterArg.getType());
        if (!oldAccTy || isMmaEncoded(iterArg))
          continue;
        auto sourceLoopResultTy =
            dyn_cast<RankedTensorType>(forOp.getResult(idx).getType());
        if (!sourceLoopResultTy)
          continue;

        Value current = yield.getOperand(idx);
        SmallVector<musa_tle::SqmmaOp, 8> chainDots;
        SmallVector<musa_tle::SqmmaWaitOp, 8> chainWaits;
        llvm::SmallPtrSet<void *, 32> visited;
        bool valid = true;
        while (current != iterArg) {
          if (!visited.insert(current.getAsOpaquePointer()).second) {
            valid = false;
            break;
          }
          auto result = dyn_cast<OpResult>(current);
          if (!result) {
            valid = false;
            break;
          }
          Operation *def = result.getOwner();
          if (auto wait = dyn_cast<musa_tle::SqmmaWaitOp>(def)) {
            if (result.getResultNumber() != 0 ||
                wait->getBlock() != forOp.getBody()) {
              valid = false;
              break;
            }
            chainWaits.push_back(wait);
            current = wait.getInput();
            continue;
          }
          auto dot = dyn_cast<musa_tle::SqmmaOp>(def);
          if (!dot || result.getResultNumber() != 0 ||
              dot->getBlock() != forOp.getBody()) {
            valid = false;
            break;
          }
          chainDots.push_back(dot);
          current = dot.getC();
        }
        if (!valid || chainDots.empty() || chainDots.back() != rootDot)
          continue;

        // Every value in the backwards path must have exactly one consumer,
        // and the only iter-arg consumer is the root dot's C operand.  This
        // prevents changing the type of a tensor that also feeds a normal
        // store or an unrelated branch.
        for (musa_tle::SqmmaOp dot : chainDots) {
          unsigned uses = std::distance(dot.getD().use_begin(),
                                        dot.getD().use_end());
          if (uses != 1) {
            valid = false;
            break;
          }
          auto dotEncoding = encodings.find(dot);
          if (dotEncoding == encodings.end() ||
              dotEncoding->second != mmaEncoding->second) {
              valid = false;
              break;
          }
        }
        for (musa_tle::SqmmaWaitOp wait : chainWaits) {
          if (std::distance(wait.getOutput().use_begin(),
                            wait.getOutput().use_end()) != 1) {
            valid = false;
            break;
          }
        }
        if (!valid)
          continue;
        unsigned iterUses = 0;
        for (OpOperand &use : iterArg.getUses()) {
          if (use.getOwner() == chainDots.back().getOperation() &&
              use.getOperandNumber() == 2)
            ++iterUses;
          else {
            valid = false;
            break;
          }
        }
        if (!valid || iterUses != 1)
          continue;

        // The wait at the end of the path is a real completion boundary.  It
        // may be followed by pipe release, so we keep it in the native loop;
        // only its tensor layout conversion is removed.
        auto nativeTy = RankedTensorType::get(
            oldAccTy.getShape(), oldAccTy.getElementType(),
            mmaEncoding->second);
        Value init = forOp.getInitArgs()[idx];
        if (init.getType() != nativeTy) {
          OpBuilder builder(forOp);
          Value converted = ttg::ConvertLayoutOp::create(
              builder, rootDot.getLoc(), nativeTy, init);
          forOp.getInitArgsMutable()[idx].assign(converted);
        }
        forOp.getRegionIterArg(idx).setType(nativeTy);
        forOp.getResult(idx).setType(nativeTy);
        for (musa_tle::SqmmaOp dot : chainDots)
          dot.getResult().setType(nativeTy);
        for (musa_tle::SqmmaWaitOp wait : chainWaits) {
          wait.getInput().setType(nativeTy);
          wait.getOutput().setType(nativeTy);
        }

        SmallVector<OpOperand *> externalUses;
        for (OpOperand &use : forOp.getResult(idx).getUses()) {
          if (!isa<musa_tle::SqmmaWaitOp>(use.getOwner()))
            externalUses.push_back(&use);
        }
        if (!externalUses.empty()) {
          OpBuilder builder(forOp);
          builder.setInsertionPointAfter(forOp);
          Value converted = ttg::ConvertLayoutOp::create(
              builder, rootDot.getLoc(), sourceLoopResultTy,
              forOp.getResult(idx));
          for (OpOperand *use : externalUses)
            use->set(converted);
        }
        // One loop can only be prepared once.  The source loop now has the
        // carrier encoding and subsequent root-dot scans will skip it.
        break;
      }
      break;
    }
  }
  return success();
}

// An async SQMMA result can be fed directly into the accumulator operand of
// another async SQMMA.  Waiting in between those two operations only adds a
// device barrier and a layout round-trip; retain waits with ordinary or dead
// results because they may be the final async completion boundary.
static bool isElidableAccumulatorWait(musa_tle::SqmmaWaitOp wait) {
  // MTT SQMMA completion state is not carried safely across an SCF loop (or
  // a pipe stage represented by a loop-carried value).  Removing such a wait
  // lets the next iteration observe an accumulator before the hardware has
  // committed it, which produces silent numerical corruption.  Keep the
  // conservative boundary until loop-carried native carrier conversion has
  // proved the dependency explicitly.
  if (wait->getParentOfType<scf::ForOp>())
    return false;

  Value output = wait.getOutput();
  if (output.use_empty())
    // A dead result can still carry the kernel's final async completion
    // boundary; keep that wait even though there is no SSA consumer.
    return false;
  return llvm::all_of(output.getUses(), [&](OpOperand &use) {
    auto dot = dyn_cast<musa_tle::SqmmaOp>(use.getOwner());
    return dot && use.getOperandNumber() == 2 &&
           wait->getBlock() == dot->getBlock() &&
           wait->getNextNode() == dot.getOperation();
  });
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERSQMMA
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUTLELowerSqmmaPass
    : impl::TritonMUSAGPUTLELowerSqmmaBase<TritonMUSAGPUTLELowerSqmmaPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<musa_tle::SqmmaOp> dots;
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *candidate) {
      if (auto dot = dyn_cast<musa_tle::SqmmaOp>(candidate))
        dots.push_back(dot);
      if (isa<musa_tle::SqmmaOp, musa_tle::SqmmaWaitOp>(candidate))
        worklist.push_back(candidate);
    });
    if (!dots.empty())
      module->setAttr(kEnableEncodingRematerializationAttr,
                      UnitAttr::get(module.getContext()));

    // An async accumulator cannot cross an SCF region boundary as an ordinary
    // tensor.  Materialize a TLE wait immediately before every direct branch
    // yield, before verification, so the verifier sees an explicit completion
    // boundary even when the source has no outer wgmma_wait.  The helper is
    // deliberately limited to a direct sqmma result yielded by scf.if; it
    // does not accept arbitrary region values or tensor users.
    llvm::DenseSet<Value> synchronousIfResults;
    SmallVector<Operation *> branchLocalWaits;
    materializeBranchLocalWaits(dots, synchronousIfResults,
                                branchLocalWaits);
    worklist.append(branchLocalWaits.begin(), branchLocalWaits.end());

    DenseMap<musa_tle::SqmmaOp, ttg::MUSASqmmaEncodingAttr> encodings;
    for (musa_tle::SqmmaOp dot : dots) {
      if (failed(verifyAsyncUses(dot)))
        return signalPassFailure();
      auto aTy = cast<ttg::MemDescType>(dot.getA().getType());
      auto bTy = cast<ttg::MemDescType>(dot.getB().getType());
      auto operandType = musa::getWmmaEltType(aTy.getElementType());
      if (!operandType) {
        dot.emitOpError("cannot map operand dtype to an SQMMA element type");
        return signalPassFailure();
      }
      unsigned numWarps = ttg::maybeLookupNumWarps(dot).value_or(0);
      auto config =
          selectSqmmaConfig(aTy.getShape()[0], bTy.getShape()[1],
                            aTy.getShape()[1], numWarps, *operandType,
                            musa::getMusaComputeCapability(dot.getOperation()));
      if (!config) {
        dot.emitOpError("cannot select a supported SQMMA configuration");
        return signalPassFailure();
      }
      auto accTy = cast<RankedTensorType>(dot.getC().getType());
      auto cga = ttg::getCGALayout(accTy.getEncoding());
      auto mmaEnc = ttg::MUSASqmmaEncodingAttr::get(
          dot.getContext(), 3, 1, config->warpsPerCTA, cga, config->instrShape);
      encodings[dot] = mmaEnc;
    }

    // Infer all physical layouts before replacing any TLE SQMMA op.
    for (musa_tle::SqmmaOp dot : dots) {
      if (failed(updateOperandLayout(dot, 0, encodings[dot])) ||
          failed(updateOperandLayout(dot, 1, encodings[dot])))
        return signalPassFailure();
    }

    // Prepare strict direct loop-carried dots before replacing any TLE
    // operation.  This prevents the old dot result from remaining attached to
    // scf.yield while the source operation is erased.
    if (failed(prepareDirectLoopCarriedSqmmaTypes(dots, encodings)))
      return signalPassFailure();
    if (failed(prepareWaitLoopCarriedSqmmaTypes(dots, encodings)))
      return signalPassFailure();

    // Keep the asynchronous accumulator chain intact.  The generic MUSA
    // SQMMA pipeline already relies on this contract, and TLE's verifier
    // explicitly permits a dot result to be consumed as the next dot's C
    // operand without an intervening wait.  Only the first non-C consumer
    // needs a real wait, which is normally the final store/conversion.
    for (unsigned worklistIndex = 0; worklistIndex < worklist.size();
         ++worklistIndex) {
      Operation *candidate = worklist[worklistIndex];
      if (!candidate)
        continue;
      auto wait = dyn_cast<musa_tle::SqmmaWaitOp>(candidate);
      if (!wait || !isElidableAccumulatorWait(wait))
        continue;
      wait.getOutput().replaceAllUsesWith(wait.getInput());
      wait.erase();
      // Keep the pre-collected worklist free of dangling operation pointers.
      worklist[worklistIndex] = nullptr;
    }

    DenseMap<Value, Value> nativeAccumulators;
    llvm::SmallPtrSet<Operation *, 8> loweredWaits;
    for (Operation *candidate : worklist) {
      if (!candidate || loweredWaits.contains(candidate))
        continue;
      OpBuilder builder(candidate);
      if (auto dot = dyn_cast<musa_tle::SqmmaOp>(candidate)) {
        auto oldAccTy = cast<RankedTensorType>(dot.getC().getType());
        auto nativeTy = RankedTensorType::get(
            oldAccTy.getShape(), oldAccTy.getElementType(), encodings[dot]);
        Value nativeAcc = nativeAccumulators.lookup(dot.getC());
        if (!nativeAcc) {
          if (isMmaEncoded(dot.getC()))
            nativeAcc = dot.getC();
          else
            nativeAcc = ttg::ConvertLayoutOp::create(builder, dot.getLoc(),
                                                     nativeTy, dot.getC());
        } else if (nativeAcc.getType() != nativeTy) {
          // A wait result can be reused directly only when the next SQMMA
          // uses the same accumulator encoding.  Keep the fast path for the
          // common K-loop case, while preserving correctness for a change of
          // tile/layout between consecutive dots.
          nativeAcc = ttg::ConvertLayoutOp::create(
              builder, dot.getLoc(), nativeTy, nativeAcc);
        }
        Value useC = arith::ConstantIntOp::create(builder, dot.getLoc(), 1, 1);
        auto config = cast<ttg::MUSASqmmaEncodingAttr>(nativeTy.getEncoding());
        auto instr = config.getInstrShape();
        auto operandType = musa::getWmmaEltType(
            cast<ttg::MemDescType>(dot.getA().getType()).getElementType());
        assert(operandType && "SQMMA operand type must be verified");
        auto nativeDot = musa::SquadDotOp::create(
            builder, dot.getLoc(), nativeTy, dot.getA(), dot.getB(), nativeAcc,
            useC, instr[0], instr[1], instr[2], musa::SQMMAEltType::f32,
            *operandType, *operandType,
            inferLayout(cast<ttg::MemDescType>(dot.getA().getType())),
            inferLayout(cast<ttg::MemDescType>(dot.getB().getType())), true,
            musa::SQMMAAccumulationMode::hardware,
            static_cast<int32_t>(dot.getInputPrecision()), 0);
        nativeDot->setAttr(kExplicitSqmmaAttr, builder.getUnitAttr());
        nativeAccumulators[dot.getD()] = nativeDot.getD();

        // Most TLE dot results are intentionally kept in the source SSA map
        // until their explicit wait is lowered below.  A strict direct
        // loop-carried dot has no wait, however; its result is consumed by
        // scf.yield and the source operation is erased at the end of this
        // pass.  Rewrite that yield now so it cannot retain a dangling use.
        for (OpOperand &use : dot.getD().getUses()) {
          auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
          if (!yield || dot->getNextNode() != yield.getOperation())
            continue;
          auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
          unsigned idx = use.getOperandNumber();
          if (forOp && idx < forOp.getNumRegionIterArgs() &&
              dot.getC() == forOp.getRegionIterArg(idx))
            use.set(nativeDot.getD());
        }
        continue;
      }

      auto wait = cast<musa_tle::SqmmaWaitOp>(candidate);
      SmallVector<musa_tle::SqmmaWaitOp, 4> adjacentWaits{wait};
      for (Operation *next = wait->getNextNode(); next;) {
        auto nextWait = dyn_cast<musa_tle::SqmmaWaitOp>(next);
        if (!nextWait || nextWait.getPendings() != wait.getPendings())
          break;
        adjacentWaits.push_back(nextWait);
        next = next->getNextNode();
      }

      SmallVector<Value, 4> nativeInputs;
      for (musa_tle::SqmmaWaitOp adjacentWait : adjacentWaits) {
        if (synchronousIfResults.contains(adjacentWait.getInput()))
          continue;
        Value nativeInput = nativeAccumulators.lookup(adjacentWait.getInput());
        if (!nativeInput) {
          adjacentWait.emitOpError(
              "input must be the async result of musa_tle.sqmma");
          return signalPassFailure();
        }
        nativeInputs.push_back(nativeInput);
      }
      triton::musa::SquadDotWaitOp nativeWait;
      if (!nativeInputs.empty()) {
        nativeWait = musa::SquadDotWaitOp::create(builder, wait.getLoc(),
                                                  nativeInputs);
        nativeWait->setAttr(kExplicitSqmmaAttr, builder.getUnitAttr());
      }
      unsigned nativeResultIndex = 0;
      for (musa_tle::SqmmaWaitOp adjacentWait : adjacentWaits) {
        Value released;
        if (synchronousIfResults.contains(adjacentWait.getInput())) {
          // This input was completed in the producing branch; preserve its
          // tensor value and do not issue a second native wait.
          released = adjacentWait.getInput();
        } else {
          Value nativeResult = nativeWait.getResult(nativeResultIndex++);
          released = ttg::ConvertLayoutOp::create(
              builder, adjacentWait.getLoc(),
              adjacentWait.getOutput().getType(),
              nativeResult);
          nativeAccumulators[released] = nativeResult;
        }
        adjacentWait.getOutput().replaceAllUsesWith(released);
        // Keep the native value available when a subsequent SQMMA consumes
        // this wait result as its accumulator.  Without this mapping the
        // next dot would convert the just-released value back to native form.
        loweredWaits.insert(adjacentWait.getOperation());
      }
      for (musa_tle::SqmmaWaitOp adjacentWait : adjacentWaits)
        adjacentWait.erase();
    }

    for (musa_tle::SqmmaOp dot : llvm::reverse(dots))
      dot.erase();
  }
};

} // namespace mlir

#endif // __TLE__
