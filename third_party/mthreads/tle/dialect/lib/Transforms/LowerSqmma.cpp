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
constexpr llvm::StringLiteral kExplicitSqmmaAttr("musa_tle.explicit_sqmma");
constexpr llvm::StringLiteral kEnableEncodingRematerializationAttr(
    "tle.enable_encoding_rematerialization");

struct SelectedSqmmaConfig {
  SmallVector<unsigned, 3> instrShape;
  SmallVector<unsigned, 2> warpsPerCTA;
};

static std::optional<SelectedSqmmaConfig>
selectSqmmaConfig(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                  musa::SQMMAEltType operandType) {
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
            !musa::isSupportedSqmma(operandType, operandType,
                                    musa::SQMMAEltType::f32, instM, instN,
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
          bool preferMSplit =
              warpsN == 1 && instM >= 32 && best && best->warpsPerCTA[1] != 1;
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
  auto shared = mmaEnc.composeSharedLayoutForOperand(
      cga, operandIdx, operandTy.getShape(), order,
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

  if (failed(musa::resolveTMESwizzleConfigFromEncoding(landingTy)))
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
      if (index.getResult().getType() != landingTy) {
        index.getResult().setType(landingTy);
        (void)checkAndSet(index.getOperation());
        changed = true;
      }
    });
  }
  return success();
}

static LogicalResult verifyAsyncUses(musa_tle::SqmmaOp op) {
  for (OpOperand &use : op.getD().getUses()) {
    Operation *user = use.getOwner();
    if (auto next = dyn_cast<musa_tle::SqmmaOp>(user)) {
      if (use.getOperandNumber() == 2)
        continue;
    }
    if (isa<musa_tle::SqmmaWaitOp, scf::YieldOp, scf::ForOp>(user))
      continue;
    return op.emitOpError(
        "async result must be consumed by musa_tle.sqmma_wait before "
        "ordinary tensor use");
  }
  return success();
}

static bool isMmaEncoded(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type &&
         isa_and_nonnull<ttg::MUSASqmmaEncodingAttr>(type.getEncoding());
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
                            aTy.getShape()[1], numWarps, *operandType);
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

    DenseMap<Value, Value> nativeAccumulators;
    for (Operation *candidate : worklist) {
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
        continue;
      }

      auto wait = cast<musa_tle::SqmmaWaitOp>(candidate);
      Value nativeInput = nativeAccumulators.lookup(wait.getInput());
      if (!nativeInput) {
        wait.emitOpError("input must be the async result of musa_tle.sqmma");
        return signalPassFailure();
      }
      auto nativeWait =
          musa::SquadDotWaitOp::create(builder, wait.getLoc(), nativeInput);
      nativeWait->setAttr(kExplicitSqmmaAttr, builder.getUnitAttr());
      Value released = ttg::ConvertLayoutOp::create(builder, wait.getLoc(),
                                                    wait.getOutput().getType(),
                                                    nativeWait.getResult(0));
      if (Attribute explicitEncoding =
              getTleExplicitValueEncoding(wait.getOutput()))
        setTleExplicitResultEncoding(released.getDefiningOp(), 0,
                                     explicitEncoding);
      wait.getOutput().replaceAllUsesWith(released);
      wait.erase();
    }

    for (musa_tle::SqmmaOp dot : llvm::reverse(dots))
      dot.erase();
  }
};

} // namespace mlir

#endif // __TLE__
