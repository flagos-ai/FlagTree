#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "MUSATLE/Transforms/TransformAttrs.h"
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
constexpr llvm::StringLiteral kEnableEncodingRematerializationAttr(
    "tle.enable_encoding_rematerialization");
constexpr llvm::StringLiteral
    kDisableMSplitPreferenceAttr("musa_tle.disable_sqmma_m_split_preference");

struct SelectedSqmmaConfig {
  SmallVector<unsigned, 3> instrShape;
  SmallVector<unsigned, 2> warpsPerCTA;
};

static std::optional<SelectedSqmmaConfig>
selectSqmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                  musa::SQMMAEltType operandType,
                  bool disableMSplitPreference) {
  if (numWarps < 4 || numWarps % 4 != 0)
    return std::nullopt;
  const auto *sqTraits = musa::getMusaSqmmaArchTraits(musa::MusaArch::PH1);
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
          !musa::isSupportedSqmmaInstrMN(operandType, instM, instN, *sqTraits))
        continue;
      for (unsigned instK : kK) {
        if (k % instK != 0 ||
            !musa::isSupportedSqmmaWithK128Compatibility(
                operandType, operandType, musa::SQMMAEltType::f32, instM, instN,
                instK, *sqTraits))
          continue;
        for (unsigned warpsM = 4; warpsM <= numWarps; warpsM *= 2) {
          if (numWarps % warpsM != 0)
            continue;
          unsigned warpsN = numWarps / warpsM;
          unsigned tileM = instM * (warpsM / 4);
          unsigned tileN = instN * warpsN;
          if (m % tileM != 0 || n % tileN != 0)
            continue;
          unsigned count = (m / tileM) * (n / tileN) * (k / instK);
          unsigned volume = instM * instN * instK;
          // On a full tie prefer a pure M-split (warpsN == 1) as long as
          // the instruction M stays >= 32: attention-style kernels reduce the
          // dot result along axis=1, and an M-split keeps every output row
          // inside one squad so the reduction never crosses warps. Finer M
          // splits (instM 16) measure slower and are never preferred.
          bool preferMSplit = !disableMSplitPreference && warpsN == 1 &&
                              instM >= 32 && best && best->warpsPerCTA[1] != 1;
          if (!best || count < bestCount ||
              (count == bestCount &&
               (volume > bestVolume ||
                (volume == bestVolume && preferMSplit)))) {
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
    // A worker's shared descriptor is an explicit capture, not a new
    // allocation. Follow it back across the isolated partition boundary.
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
          argument.getOwner()->getParentOp());
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

  SmallVector<int64_t> physicalShape = musa::getMemDescPhysicalShape(operandTy);
  auto order = musa::getSharedOrder(operandTy.getEncoding(), physicalShape);
  auto cga = ttg::getCGALayout(operandTy.getEncoding());
  // Use the current SQMMA attribute API directly.  A DotOperandEncodingAttr
  // describes a register operand and is intentionally rejected for TLE
  // SQMMA, whose operands must remain shared-memory memdescs.
  auto shared = mmaEnc.composeSharedLayoutForOperand(
      cga, operandIdx, physicalShape, order,
      /*kWidth=*/0, operandTy.getElementTypeBitWidth(),
      /*needTrans=*/false);

  auto desiredTy = ttg::MemDescType::get(
      operandTy.getShape(), operandTy.getElementType(), shared,
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
  // A rank-preserving operand slice (for example B[64, 256] viewed as two
  // B[64, 128] tiles) must not shrink the stage returned by memdesc_index.
  // Build that stage from the root allocation and reserve landingTy for the
  // final view only.
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
    stageTy = ttg::MemDescType::get(stageShape, rootTy.getElementType(),
                                    landingEncoding, rootTy.getMemorySpace(),
                                    rootTy.getMutableMemory(), stageAllocShape);
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

// A predicate around an async SQMMA is represented as an scf.if result. The
// result is safe only when every branch stays on the same accumulator path and
// the converged value reaches a completion boundary.
static bool isAllowedAsyncIfValue(Value value, musa_tle::SqmmaOp root,
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
    return idx == 0 && isAllowedAsyncIfValue(wait.getInput(), root, visited);
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
      return isAllowedAsyncIfValue(yield.getOperand(idx), root, branchVisited);
    };
    return checkRegion(ifOp.getThenRegion()) &&
           (!ifOp.elseBlock() || checkRegion(ifOp.getElseRegion()));
  }
  return false;
}

static bool hasAsyncCompletionBoundary(Value value, musa_tle::SqmmaOp root,
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
      if (hasAsyncCompletionBoundary(ifOp.getResult(idx), root, nestedVisited))
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
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
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

// Complete a direct branch-local async result before it crosses the SCF
// boundary. This remains deliberately conservative and does not accept
// arbitrary branch values.
static void
materializeBranchLocalWaits(ArrayRef<musa_tle::SqmmaOp> dots,
                            SmallVectorImpl<Operation *> &newWaits) {
  for (musa_tle::SqmmaOp dot : dots) {
    SmallVector<OpOperand *, 4> branchUses;
    for (OpOperand &use : dot.getD().getUses()) {
      auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
      if (yield && isa<scf::IfOp>(yield->getParentOp()))
        branchUses.push_back(&use);
    }
    for (OpOperand *use : branchUses) {
      auto yield = cast<scf::YieldOp>(use->getOwner());
      auto ifOp = cast<scf::IfOp>(yield->getParentOp());
      unsigned idx = use->getOperandNumber();
      if (idx >= ifOp.getNumResults())
        continue;
      if (auto previous =
              dyn_cast_or_null<musa_tle::SqmmaWaitOp>(yield->getPrevNode())) {
        if (previous.getInput() == dot.getD())
          continue;
      }

      OpBuilder builder(yield);
      auto wait = musa_tle::SqmmaWaitOp::create(
          builder, dot.getLoc(), dot.getD().getType(), dot.getD(),
          builder.getI32IntegerAttr(0));
      yield->setOperand(idx, wait.getOutput());
      newWaits.push_back(wait.getOperation());
    }
  }
}

// A nested scf.if result is synchronous when every branch yields either an
// already-completed inner if result, an explicit SQMMA wait result, or a block
// argument which carries no newly-issued asynchronous work. Propagate that
// proof outwards to a fixed point so an outer source-level wait is erased
// rather than incorrectly lowered as another hardware wait.
static void
propagateSynchronousIfResults(Operation *root,
                              llvm::DenseSet<Value> &synchronousIfResults) {
  auto isKnownSynchronous = [&](Value value) {
    if (synchronousIfResults.contains(value))
      return true;
    if (value.getDefiningOp<musa_tle::SqmmaWaitOp>())
      return true;
    if (value.getDefiningOp<musa_tle::SqmmaOp>())
      return false;
    if (auto result = dyn_cast<OpResult>(value)) {
      if (isa<scf::IfOp, scf::ForOp>(result.getOwner()))
        return false;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return !isa_and_nonnull<scf::ForOp>(blockArg.getOwner()->getParentOp());
    return value.getDefiningOp() != nullptr;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    root->walk([&](scf::ForOp forOp) {
      for (unsigned idx = 0; idx < forOp.getNumRegionIterArgs(); ++idx) {
        Value iterArg = forOp.getRegionIterArg(idx);
        if (!synchronousIfResults.contains(iterArg) &&
            isKnownSynchronous(forOp.getInitArgs()[idx])) {
          synchronousIfResults.insert(iterArg);
          changed = true;
        }
      }
    });
    root->walk([&](scf::IfOp ifOp) {
      for (unsigned idx = 0; idx < ifOp.getNumResults(); ++idx) {
        Value result = ifOp.getResult(idx);
        if (synchronousIfResults.contains(result))
          continue;

        bool sawCompletedPath = false;
        auto isCompletedRegion = [&](Region &region) {
          if (region.empty())
            return false;
          auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
          if (!yield || idx >= yield.getNumOperands())
            return false;
          Value yielded = yield.getOperand(idx);
          if (synchronousIfResults.contains(yielded)) {
            sawCompletedPath = true;
            return true;
          }
          if (yielded.getDefiningOp<musa_tle::SqmmaWaitOp>()) {
            sawCompletedPath = true;
            return true;
          }
          return isKnownSynchronous(yielded);
        };

        if (isCompletedRegion(ifOp.getThenRegion()) && ifOp.elseBlock() &&
            isCompletedRegion(ifOp.getElseRegion()) && sawCompletedPath) {
          synchronousIfResults.insert(result);
          changed = true;
        }
      }
    });
    root->walk([&](scf::ForOp forOp) {
      auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yield)
        return;
      for (unsigned idx = 0; idx < forOp.getNumResults(); ++idx) {
        Value result = forOp.getResult(idx);
        if (synchronousIfResults.contains(result) ||
            !synchronousIfResults.contains(forOp.getRegionIterArg(idx)) ||
            !isKnownSynchronous(yield.getOperand(idx)))
          continue;
        synchronousIfResults.insert(result);
        changed = true;
      }
    });
  }
}

static bool isMmaEncoded(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type &&
         isa_and_nonnull<ttg::MUSASqmmaEncodingAttr>(type.getEncoding());
}

// Prepare the strict `%next = sqmma ..., %iter; yield %next` form before the
// TLE operation is replaced. The later MTGPU conversion then carries the
// accumulator through SCF as an opaque native value.
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
      auto oldLoopResultTy =
          dyn_cast<RankedTensorType>(forOp.getResult(idx).getType());
      if (!oldAccTy || !oldLoopResultTy)
        return dot.emitOpError(
            "direct loop-carried accumulator must be a ranked tensor");
      auto nativeTy = RankedTensorType::get(
          oldAccTy.getShape(), oldAccTy.getElementType(), mmaEncoding->second);

      Value init = forOp.getInitArgs()[idx];
      if (init.getType() != nativeTy) {
        OpBuilder builder(forOp);
        Value converted =
            ttg::ConvertLayoutOp::create(builder, dot.getLoc(), nativeTy, init);
        forOp.getInitArgsMutable()[idx].assign(converted);
      }
      forOp.getRegionIterArg(idx).setType(nativeTy);
      forOp.getResult(idx).setType(nativeTy);
      dot.getResult().setType(nativeTy);

      SmallVector<OpOperand *> externalUses;
      for (OpOperand &resultUse : forOp.getResult(idx).getUses()) {
        if (!isa<musa_tle::SqmmaWaitOp>(resultUse.getOwner()))
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
// pipe slot release:
//
//   iter -> dot -> dot ... -> wait -> scf.yield
//
// Every value on the path must be single-use and every dot must select the
// same accumulator encoding. Other control-flow and ordinary tensor users
// remain fail-closed.
static LogicalResult prepareWaitLoopCarriedSqmmaTypes(
    ArrayRef<musa_tle::SqmmaOp> dots,
    DenseMap<musa_tle::SqmmaOp, ttg::MUSASqmmaEncodingAttr> &encodings) {
  for (musa_tle::SqmmaOp rootDot : dots) {
    auto mmaEncoding = encodings.find(rootDot);
    if (mmaEncoding == encodings.end())
      continue;

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

        for (musa_tle::SqmmaOp dot : chainDots) {
          unsigned uses =
              std::distance(dot.getD().use_begin(), dot.getD().use_end());
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

        auto nativeTy = RankedTensorType::get(oldAccTy.getShape(),
                                              oldAccTy.getElementType(),
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
        break;
      }
      break;
    }
  }
  return success();
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERSQMMA
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUTLELowerSqmmaPass
    : impl::TritonMUSAGPUTLELowerSqmmaBase<TritonMUSAGPUTLELowerSqmmaPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool disableMSplitPreference =
        module->hasAttr(kDisableMSplitPreferenceAttr);
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

    // A native asynchronous accumulator cannot cross an SCF branch as an
    // ordinary tensor. Complete direct branch-local values before verifying
    // and lowering the surrounding chain.
    llvm::DenseSet<Value> synchronousIfResults;
    SmallVector<Operation *> branchLocalWaits;
    materializeBranchLocalWaits(dots, branchLocalWaits);
    propagateSynchronousIfResults(module, synchronousIfResults);
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
      auto config = selectSqmmaConfig(aTy.getShape()[0], bTy.getShape()[1],
                                      aTy.getShape()[1], numWarps, *operandType,
                                      disableMSplitPreference);
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

    if (failed(prepareDirectLoopCarriedSqmmaTypes(dots, encodings)) ||
        failed(prepareWaitLoopCarriedSqmmaTypes(dots, encodings)))
      return signalPassFailure();

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
        }
        // An explicit SQMMA layout can describe another instruction shape.
        // Being MMA-encoded alone does not make its register distribution
        // compatible with this dot's selected accumulator type.
        if (nativeAcc.getType() != nativeTy)
          nativeAcc = ttg::ConvertLayoutOp::create(builder, dot.getLoc(),
                                                   nativeTy, nativeAcc);
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
        nativeDot->setAttr(musa_tle::kMUSATLEExplicitSqmmaAttr,
                           builder.getUnitAttr());
        nativeAccumulators[dot.getD()] = nativeDot.getD();

        // A direct loop-carried dot has no source wait. Redirect its yield to
        // the native result before the TLE operation is erased below.
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
        Value nativeInput;
        if (synchronousIfResults.contains(adjacentWait.getInput()))
          nativeInput = adjacentWait.getInput();
        else
          nativeInput = nativeAccumulators.lookup(adjacentWait.getInput());
        if (!nativeInput) {
          adjacentWait.emitOpError(
              "input must be the async result of musa_tle.sqmma");
          return signalPassFailure();
        }
        nativeInputs.push_back(nativeInput);
      }
      // A MUSA SQMMA wait is a global completion boundary, not an SSA-local
      // wait for only the listed accumulators.  Even when every source value
      // was completed by a branch-local wait, preserve the source wait as an
      // additional hardware boundary so unrelated outstanding SQMMA cannot
      // escape it.  Thread synchronous values through that wait as ordinary
      // SSA passthroughs: the native verifier deliberately rejects a wait
      // without dependencies.
      auto nativeWait =
          musa::SquadDotWaitOp::create(builder, wait.getLoc(), nativeInputs);
      nativeWait->setAttr(musa_tle::kMUSATLEExplicitSqmmaAttr,
                          builder.getUnitAttr());
      for (auto [nativeResult, adjacentWait] :
           llvm::zip_equal(nativeWait.getOutputs(), adjacentWaits)) {
        Value released = nativeResult;
        // Loop-carried SQMMA chains keep their accumulator in the native MMA
        // layout across iterations.  A source wait after such a loop still
        // promises its original result type to ordinary tensor consumers
        // (for example tt.store).  Do not leak the native accumulator type
        // merely because the loop result is already known to be synchronous.
        if (nativeResult.getType() != adjacentWait.getOutput().getType()) {
          released = ttg::ConvertLayoutOp::create(
              builder, adjacentWait.getLoc(),
              adjacentWait.getOutput().getType(), nativeResult);
        }
        // Preserve the native form as well.  A later SQMMA or another explicit
        // global wait may consume the released source value without forcing a
        // spurious round trip through its blocked layout.
        nativeAccumulators[released] = nativeResult;
        if (Attribute explicitEncoding =
                getTleExplicitValueEncoding(adjacentWait.getOutput())) {
          auto releasedResult = dyn_cast<OpResult>(released);
          if (!releasedResult) {
            adjacentWait.emitOpError(
                "lowered SQMMA wait must release an operation result");
            return signalPassFailure();
          }
          setTleExplicitResultEncoding(releasedResult, explicitEncoding);
        }
        adjacentWait.getOutput().replaceAllUsesWith(released);
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
