#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "MUSATLE/Transforms/PipeAnalysis.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERPIPE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace tle = triton::tle;
namespace ttmg = triton::musa;
namespace musa_tle = triton::musa_tle;

constexpr StringLiteral kPipeLocalStoreWaitGroupsAttr =
    "musa_tle.pipe_local_store_wait_groups";
constexpr StringLiteral kPipeLocalStoreGroupAttr =
    "musa_tle.pipe_local_store_group";
constexpr StringLiteral kPipeBarrierRingAttr = "musa_tle.pipe_barrier_ring";
static bool isPipeOp(Operation *op) {
  return isa<musa_tle::PipeCreateOp, musa_tle::PipeWriterAcquireOp,
             musa_tle::PipeWriterCommitOp, musa_tle::PipeWriterCloseOp,
             musa_tle::PipeReaderWaitOp, musa_tle::PipeReaderReleaseOp>(op);
}

static std::optional<std::pair<ttg::WarpSpecializeOp, Region *>>
getEnclosingPartition(Operation *op) {
  FailureOr<std::optional<musa_tle::PipeStaticPartitionInfo>> resolved =
      musa_tle::resolvePipeStaticPartition(op);
  if (failed(resolved) || !*resolved || !(*resolved)->owner ||
      (*resolved)->kind != musa_tle::PipePartitionKind::WarpSpecializeWorker ||
      !(*resolved)->region)
    return std::nullopt;
  return std::make_pair((*resolved)->owner, (*resolved)->region);
}

static bool isDefinedInside(Value value, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return region->isAncestor(blockArg.getOwner()->getParent());
  Operation *def = value.getDefiningOp();
  return def && region->isAncestor(def->getParentRegion());
}

static Value captureForUse(Operation *use, Value value) {
  auto partition = getEnclosingPartition(use);
  if (!partition || isDefinedInside(value, partition->second))
    return value;

  ttg::WarpSpecializeOp ws = partition->first;
  ttg::WarpSpecializePartitionsOp partitions = ws.getPartitionOp();
  Region *region = partition->second;
  for (auto [index, capture] :
       llvm::enumerate(partitions.getExplicitCaptures())) {
    if (capture == value)
      return region->getArgument(index);
  }

  partitions->insertOperands(partitions->getNumOperands(), value);
  unsigned captureIndex = partitions->getNumOperands() - 1;
  for (Region *partitionRegion : ws.getPartitionRegions())
    partitionRegion->addArgument(value.getType(), value.getLoc());
  return region->getArgument(captureIndex);
}

struct PipeLoweringArtifacts {
  Value fullBase;
  Value emptyBase;
  Value closeTags;
  ttg::MemDescType closeTagSlotType;
  RankedTensorType closeTagTensorType;
  RankedTensorType closeTagStoreTensorType;
  RankedTensorType closeTagInitTensorType;
  RankedTensorType closeTagInitArrayTensorType;
  int32_t closeTagInitWarpCount = 0;
  SmallVector<std::optional<int64_t>> localStoreGroupByField;
  musa_tle::PipeBarrierParticipant writerParticipant;
  SmallVector<musa_tle::PipeBarrierParticipant> readerParticipants;
};

static bool sameParticipant(const musa_tle::PipeBarrierParticipant &lhs,
                            const musa_tle::PipeBarrierParticipant &rhs) {
  return lhs.endpointIndex == rhs.endpointIndex &&
         lhs.partitionIndex == rhs.partitionIndex &&
         lhs.partition == rhs.partition && lhs.warpBegin == rhs.warpBegin &&
         lhs.warpCount == rhs.warpCount;
}

static FailureOr<musa_tle::PipeBarrierParticipant> getReaderBarrierParticipant(
    const musa_tle::PipeState &state, const PipeLoweringArtifacts &artifacts,
    const musa_tle::PipeEndpointState *endpoint, Operation *operation) {
  if (!endpoint || endpoint->role != musa_tle::PipeEndpointRole::Reader)
    return operation->emitOpError(
        "internal MUSA TLE pipe lowering lost reader endpoint");
  if (endpoint->index >= state.endpoints.size())
    return operation->emitOpError(
        "internal MUSA TLE pipe lowering received an invalid reader endpoint");
  for (const musa_tle::PipeBarrierParticipant &participant :
       artifacts.readerParticipants) {
    if (participant.endpointIndex == endpoint->index) {
      musa_tle::PipeBarrierParticipant expected{
          endpoint->index, endpoint->partitionIndex, endpoint->partition,
          endpoint->warpBegin, endpoint->warpCount};
      if (!sameParticipant(participant, expected))
        return operation->emitOpError(
            "internal MUSA TLE pipe lowering received an unstable reader "
            "barrier participant");
      return participant;
    }
  }
  return operation->emitOpError(
      "internal MUSA TLE pipe lowering lost reader barrier participant");
}

static SmallVector<int64_t>
getAllLocalStoreGroups(const PipeLoweringArtifacts &artifacts) {
  SmallVector<int64_t> groups;
  for (std::optional<int64_t> group : artifacts.localStoreGroupByField) {
    if (group)
      groups.push_back(*group);
  }
  llvm::sort(groups);
  groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
  return groups;
}

static FailureOr<SmallVector<int64_t>>
getReaderLocalStoreGroups(const PipeLoweringArtifacts &artifacts,
                          const musa_tle::PipeEndpointState *endpoint,
                          Operation *operation) {
  if (!endpoint || endpoint->role != musa_tle::PipeEndpointRole::Reader)
    return operation->emitOpError(
        "internal MUSA TLE pipe lowering lost reader endpoint");
  SmallVector<int64_t> groups;
  for (unsigned fieldIndex : endpoint->subscribedFields) {
    if (fieldIndex >= artifacts.localStoreGroupByField.size())
      return operation->emitOpError(
          "internal MUSA TLE pipe lowering received an invalid reader field "
          "subscription");
    if (std::optional<int64_t> group =
            artifacts.localStoreGroupByField[fieldIndex])
      groups.push_back(*group);
  }
  llvm::sort(groups);
  groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
  return groups;
}

static void setLocalStoreWaitGroups(Operation *wait, OpBuilder &builder,
                                    ArrayRef<int64_t> groups) {
  if (!groups.empty())
    wait->setAttr(kPipeLocalStoreWaitGroupsAttr,
                  builder.getDenseI64ArrayAttr(groups));
}

static Attribute getCloseTagEncoding(MLIRContext *context, int64_t rank) {
  SmallVector<unsigned> order;
  for (int64_t dim = rank - 1; dim >= 0; --dim)
    order.push_back(static_cast<unsigned>(dim));
  auto ctaLayout = ttg::CGAEncodingAttr::get1CTALayout(context, rank);
  return ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, order,
                                              ctaLayout);
}

static RankedTensorType getCloseTagTensorType(Operation *op, OpBuilder &builder,
                                              ArrayRef<int64_t> shape,
                                              int numWarps = -1) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (numWarps < 0)
    numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(module);
  Attribute encoding = ttg::getDefaultBlockedEncoding(
      op->getContext(), shape, numWarps, threadsPerWarp, numCTAs);
  return RankedTensorType::get(shape, builder.getI32Type(), encoding);
}

static Value createCloseTagTensor(Operation *op, OpBuilder &builder,
                                  Location loc, RankedTensorType type,
                                  bool value, int numWarps = -1) {
  // Constants cannot directly carry a shared-memory encoding. Materialize
  // the value in a regular distributed layout and convert it to the close-tag
  // layout after broadcasting. This also avoids the power-of-two restriction
  // on tt.splat for capacities such as three.
  auto blockedType =
      getCloseTagTensorType(op, builder, type.getShape(), numWarps);
  Value scalarTensor;
  if (type.getRank() > 1) {
    auto sliceEncoding = ttg::SliceEncodingAttr::get(
        op->getContext(), /*dim=*/0,
        cast<ttg::DistributedEncodingTrait>(blockedType.getEncoding()));
    auto scalarRangeType =
        RankedTensorType::get({1}, type.getElementType(), sliceEncoding);
    scalarTensor = tt::MakeRangeOp::create(builder, loc, scalarRangeType, 0, 1);
    scalarTensor =
        tt::ExpandDimsOp::create(builder, loc, blockedType, scalarTensor, 0);
  } else {
    auto scalarRangeType = getCloseTagTensorType(op, builder, {1}, numWarps);
    scalarTensor = tt::MakeRangeOp::create(builder, loc, scalarRangeType, 0, 1);
  }
  auto scalarType = cast<RankedTensorType>(scalarTensor.getType());
  if (value) {
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value oneTensor = tt::SplatOp::create(builder, loc, scalarType, one);
    scalarTensor = arith::AddIOp::create(builder, loc, scalarTensor, oneTensor);
  }
  Value valueTensor = scalarTensor;
  if (cast<RankedTensorType>(valueTensor.getType()) != type) {
    if (blockedType != scalarType)
      valueTensor =
          tt::BroadcastOp::create(builder, loc, blockedType, scalarTensor);
    valueTensor = ttg::ConvertLayoutOp::create(builder, loc, type, valueTensor);
  }
  return valueTensor;
}

static Value createCloseTagSlot(OpBuilder &builder, Location loc,
                                const PipeLoweringArtifacts &artifacts,
                                Value stage) {
  Value index = stage;
  return ttg::MemDescIndexOp::create(builder, loc, artifacts.closeTagSlotType,
                                     artifacts.closeTags, index);
}

static Value loadCloseTag(OpBuilder &builder, Location loc,
                          const PipeLoweringArtifacts &artifacts,
                          Operation *use, Value stage,
                          RankedTensorType loadTensorType = {}) {
  Value closeTags = captureForUse(use, artifacts.closeTags);
  PipeLoweringArtifacts captured = artifacts;
  captured.closeTags = closeTags;
  Value slot = createCloseTagSlot(builder, loc, captured, stage);
  if (!loadTensorType)
    loadTensorType = artifacts.closeTagTensorType;
  Value tagTensor =
      ttg::LocalLoadOp::create(builder, loc, loadTensorType, slot);
  Value tagI32 =
      tt::UnsplatOp::create(builder, loc, builder.getI32Type(), tagTensor);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, tagI32,
                               zero);
}

static void storeCloseTag(OpBuilder &builder, Location loc,
                          const PipeLoweringArtifacts &artifacts,
                          Operation *use, Value stage, bool value,
                          RankedTensorType storeTensorType = {},
                          int numWarps = -1) {
  Value closeTags = captureForUse(use, artifacts.closeTags);
  PipeLoweringArtifacts captured = artifacts;
  captured.closeTags = closeTags;
  Value slot = createCloseTagSlot(builder, loc, captured, stage);
  if (!storeTensorType)
    storeTensorType = artifacts.closeTagStoreTensorType;
  Value tag =
      createCloseTagTensor(use, builder, loc, storeTensorType, value, numWarps);
  ttg::LocalStoreOp::create(builder, loc, tag, slot);
}

static void initializeCloseTags(OpBuilder &builder, Location loc,
                                const PipeLoweringArtifacts &artifacts,
                                Operation *use, int32_t capacity,
                                bool useInitTensorType = false) {
  if (useInitTensorType && artifacts.closeTagInitArrayTensorType) {
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value tag = tt::SplatOp::create(
        builder, loc, artifacts.closeTagInitArrayTensorType, zero);
    ttg::LocalStoreOp::create(builder, loc, tag, artifacts.closeTags);
    return;
  }
  RankedTensorType storeTensorType = useInitTensorType
                                         ? artifacts.closeTagInitTensorType
                                         : artifacts.closeTagStoreTensorType;
  if (!storeTensorType)
    return;
  int numWarps = useInitTensorType ? artifacts.closeTagInitWarpCount
                                   : artifacts.writerParticipant.warpCount;
  for (int32_t index = 0; index < capacity; ++index) {
    Value stage = arith::ConstantIntOp::create(builder, loc, index, 32);
    storeCloseTag(builder, loc, artifacts, use, stage, /*value=*/false,
                  storeTensorType, numWarps);
  }
}

static LogicalResult getWriterIssueThread(Operation *op, int32_t &issueThread) {
  FailureOr<int32_t> startThread = musa_tle::getPipePartitionStartThread(op);
  if (failed(startThread))
    return failure();
  issueThread = *startThread;
  return success();
}

static FailureOr<int64_t> getStaticFieldBytes(const musa_tle::PipeState &state,
                                              unsigned fieldIndex) {
  if (fieldIndex >= state.fields.size())
    return failure();
  auto type =
      dyn_cast<ttg::MemDescType>(state.fields[fieldIndex].memdesc.getType());
  if (!type || type.getShape().empty())
    return failure();
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.front() == state.capacity)
    shape = shape.drop_front();
  if (shape.empty())
    return failure();
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
      return failure();
    elements *= dim;
  }
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 ||
      elements > std::numeric_limits<int64_t>::max() / bitWidth)
    return failure();
  int64_t bits = elements * bitWidth;
  if (bits % 8 != 0)
    return failure();
  int64_t bytes = bits / 8;
  return bytes > 0 ? FailureOr<int64_t>(bytes) : FailureOr<int64_t>(failure());
}

static bool isValidRegion(const musa_tle::PipeCoveredRegion &region) {
  return region.exact && region.byteOffset && region.byteSize &&
         *region.byteOffset >= 0 && *region.byteSize > 0 &&
         *region.byteOffset <=
             std::numeric_limits<int64_t>::max() - *region.byteSize;
}

static LogicalResult
verifyCompletionCoverage(const musa_tle::PipeState &state,
                         Operation *diagnosticOp,
                         ArrayRef<musa_tle::PipeCompletionSource> sources) {
  for (unsigned fieldIndex = 0; fieldIndex < state.fields.size();
       ++fieldIndex) {
    FailureOr<int64_t> fieldBytes = getStaticFieldBytes(state, fieldIndex);
    if (failed(fieldBytes))
      return diagnosticOp->emitOpError(
          "MUSA TLE pipe requires a statically provable contiguous field "
          "region");
    SmallVector<musa_tle::PipeByteInterval> intervals;
    std::optional<musa_tle::PipeTransportKind> fieldTransport;
    for (const musa_tle::PipeCompletionSource &source : sources) {
      if (source.destinationField != fieldIndex)
        continue;
      if (fieldTransport && *fieldTransport != source.kind)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe does not support mixed TME and local-store "
            "sources for one payload field");
      fieldTransport = source.kind;
      if (!isValidRegion(source.coveredRegion))
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe requires a statically provable contiguous field "
            "region");
      intervals.push_back(
          {*source.coveredRegion.byteOffset, *source.coveredRegion.byteSize});
    }
    if (intervals.empty())
      return diagnosticOp->emitOpError(
          "MUSA TLE pipe commit does not cover every payload field region");
    llvm::sort(intervals, [](const musa_tle::PipeByteInterval &lhs,
                             const musa_tle::PipeByteInterval &rhs) {
      return lhs.byteOffset < rhs.byteOffset;
    });
    int64_t cursor = 0;
    for (const musa_tle::PipeByteInterval &interval : intervals) {
      if (interval.byteOffset < cursor)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe completion sources for one field must not "
            "overlap");
      if (interval.byteOffset != cursor)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe commit does not cover every payload field "
            "region");
      cursor += interval.byteSize;
    }
    if (cursor != *fieldBytes)
      return diagnosticOp->emitOpError(
          "MUSA TLE pipe commit does not cover every payload field region");
  }
  return success();
}

class LowerPipePass
    : public impl::TritonMUSAGPUTLELowerPipeBase<LowerPipePass> {
  static LogicalResult
  verifyLoweringCapabilities(musa_tle::PipeAnalysisResult &analysis) {
    llvm::DenseMap<Operation *, int64_t> pipeBarrierSlots;
    llvm::DenseMap<Operation *, musa_tle::PipeCreateOp> pipeBarrierAnchors;
    for (const std::unique_ptr<musa_tle::PipeState> &ownedState :
         analysis.getPipes()) {
      musa_tle::PipeState &state = *ownedState;
      bool oneShot =
          state.lifecycle.mode == musa_tle::PipeLifecycleMode::OneShot;
      bool hasClose = !state.closeGenerations.empty();
      if (state.create->getAttrOfType<ArrayAttr>("readers")) {
        if (!state.barrierPlan.writerParticipant ||
            state.barrierPlan.readerParticipants.size() !=
                state.endpoints.size() - 1)
          return state.create.emitOpError(
              "internal MUSA TLE pipe lowering received an incomplete named "
              "reader barrier participant ledger");
        int64_t readerWarps = 0;
        for (const musa_tle::PipeBarrierParticipant &participant :
             state.barrierPlan.readerParticipants) {
          readerWarps += participant.warpCount;
          if (readerWarps > std::numeric_limits<int32_t>::max())
            return state.create.emitOpError(
                "MUSA TLE pipe reader arrival count exceeds the positive "
                "i32 range");
        }
        if (readerWarps <= 0 ||
            (oneShot && state.barrierPlan.empty.has_value()) ||
            (!oneShot &&
             (!state.barrierPlan.empty ||
              state.barrierPlan.empty->arrivalCount != readerWarps)))
          return state.create.emitOpError(
              "internal MUSA TLE pipe lowering received inconsistent named "
              "reader arrival accounting");
        if (hasClose && state.closeWaits.size() != state.endpoints.size() - 1)
          return state.create.emitOpError(
              "internal MUSA TLE pipe lowering received an incomplete "
              "terminal close broadcast");
      }
      if (state.fields.empty() ||
          (!state.barrierPlan.full.transactionBytes && !hasClose))
        return state.create.emitOpError(
            "internal MUSA TLE pipe lowering requires non-empty payload "
            "fields and static full-barrier transaction bytes");
      bool externalFull = state.barrierPlan.full.storageOwner ==
                          musa_tle::PipeBarrierStorageOwner::External;
      if ((externalFull != (state.barrierPlan.fullBarrierStorageOwner ==
                            musa_tle::PipeBarrierStorageOwner::External)) ||
          (externalFull && (!state.barrierPlan.externalFull ||
                            !state.barrierPlan.full.externalStorage)))
        return state.create.emitOpError(
            "internal MUSA TLE pipe lowering received an invalid external "
            "completion barrier plan");
      if ((oneShot &&
           (state.barrierPlan.empty || state.barrierPlan.closeTagPlan ||
            state.barrierPlan.hasCloseState || hasClose ||
            state.barrierPlan.phasePolicy !=
                musa_tle::PipePhasePolicy::OneShotFixed)) ||
          (!oneShot && (!state.barrierPlan.empty ||
                        state.barrierPlan.phasePolicy !=
                            musa_tle::PipePhasePolicy::CyclicAlternating)))
        return state.create.emitOpError(
            "internal MUSA TLE pipe lowering received an invalid lifecycle "
            "barrier plan");
      if (hasClose) {
        musa_tle::PipeCloseGeneration &close = *state.closeGenerations.front();
        bool hasTME = false;
        bool hasLocalStore = false;
        for (const auto &group : state.commitGroups) {
          hasTME |= group->tmeGroupArrivalCount != 0;
          hasLocalStore |= group->localStoreArrivalCount != 0;
        }
        if (!state.barrierPlan.writerParticipant)
          return state.create.emitOpError(
              "internal MUSA TLE pipe lowering lost writer barrier "
              "participant");
        int32_t writerWarps = state.barrierPlan.writerParticipant->warpCount;
        int32_t expectedControl = (hasTME || !hasLocalStore) ? 1 : 0;
        int32_t expectedLocal = hasLocalStore ? writerWarps : 0;
        if (!state.barrierPlan.hasCloseState ||
            !state.barrierPlan.closeTagPlan ||
            state.barrierPlan.closeTagPlan->capacity != state.capacity ||
            state.barrierPlan.closeTagPlan->initialValue ||
            state.barrierPlan.closeTagPlan->storageOwner !=
                musa_tle::PipeBarrierStorageOwner::Pipe ||
            close.transactionBytes != 0 ||
            close.controlArrivalCount != expectedControl ||
            close.localStoreArrivalCount != expectedLocal ||
            close.fullArrivalCount != expectedControl + expectedLocal ||
            close.fullArrivalCount != state.barrierPlan.full.arrivalCount)
          return close.close.emitOpError(
              "MUSA TLE pipe close plan has inconsistent arrival shape");
      }
      for (const std::unique_ptr<musa_tle::PipeCommitGroup> &ownedGroup :
           state.commitGroups) {
        musa_tle::PipeCommitGroup &group = *ownedGroup;
        if (!group.logicalGeneration)
          return group.commit.emitOpError(
              "MUSA TLE pipe lowering received no logical generation "
              "mapping");
        if (group.completionSources.empty())
          return group.commit.emitOpError(
              "MUSA TLE pipe lowering requires completion coverage for "
              "every payload field");

        int64_t totalBytes = 0;
        bool hasTME = false;
        bool hasLocalStore = false;
        for (const musa_tle::PipeCompletionSource &source :
             group.completionSources) {
          if (!source.operation ||
              source.destinationField >= state.fields.size() ||
              source.stage != group.stage || source.phase != group.phase ||
              source.coveredRegion.fieldIndex != source.destinationField ||
              source.coveredRegion.memdescRoot !=
                  state.fields[source.destinationField].memdescRoot ||
              !isValidRegion(source.coveredRegion) ||
              (source.kind == musa_tle::PipeTransportKind::TME &&
               (source.barrierStorageOwner ==
                musa_tle::PipeBarrierStorageOwner::External) != externalFull) ||
              (source.kind == musa_tle::PipeTransportKind::TME &&
               externalFull &&
               source.externalBarrierRoot !=
                   state.barrierPlan.full.externalStorage) ||
              ((source.kind != musa_tle::PipeTransportKind::TME ||
                !externalFull) &&
               source.externalBarrierRoot))
            return group.commit.emitOpError(
                "MUSA TLE pipe lowering received an invalid completion "
                "group");
          if (source.kind == musa_tle::PipeTransportKind::TME) {
            if (!isa<ttg::TMACopyOp>(source.operation) ||
                source.transactionBytes <= 0 ||
                source.coveredRegion.byteSize !=
                    std::optional<int64_t>(source.transactionBytes))
              return group.commit.emitOpError(
                  "MUSA TLE pipe lowering received an invalid TME "
                  "completion source");
            hasTME = true;
          } else if (source.kind == musa_tle::PipeTransportKind::LocalStore) {
            if (!isa<triton::StoreOp, ttg::LocalStoreOp>(source.operation) ||
                source.transactionBytes != 0)
              return group.commit.emitOpError(
                  "MUSA TLE pipe lowering received an invalid local-store "
                  "completion source");
            FailureOr<int64_t> fieldBytes =
                getStaticFieldBytes(state, source.destinationField);
            if (source.coveredRegion.byteOffset != std::optional<int64_t>(0) ||
                failed(fieldBytes) || !source.coveredRegion.byteSize ||
                *source.coveredRegion.byteSize != *fieldBytes)
              return group.commit.emitOpError(
                  "MUSA TLE pipe lowering received an invalid local-store "
                  "completion source");
            hasLocalStore = true;
          } else {
            return group.commit.emitOpError(
                "MUSA TLE pipe lowering received an unknown completion "
                "transport");
          }
          totalBytes += source.transactionBytes;
          if (totalBytes > std::numeric_limits<int32_t>::max())
            return group.commit.emitOpError(
                "MUSA TLE pipe completion bytes exceed the positive i32 "
                "range");
        }
        if (failed(verifyCompletionCoverage(state, group.commit.getOperation(),
                                            group.completionSources)))
          return failure();
        if (!state.barrierPlan.writerParticipant)
          return group.commit.emitOpError(
              "internal MUSA TLE pipe lowering lost writer barrier "
              "participant");
        int32_t writerWarps = state.barrierPlan.writerParticipant->warpCount;
        bool validTMEGroup = hasTME && !hasLocalStore &&
                             group.tmeGroupArrivalCount == 1 &&
                             group.localStoreArrivalCount == 0 &&
                             group.fullArrivalCount == 1 && totalBytes > 0;
        bool validLocalStoreGroup =
            !hasTME && hasLocalStore && state.fields.size() == 1 &&
            group.tmeGroupArrivalCount == 0 &&
            group.localStoreArrivalCount == writerWarps &&
            group.fullArrivalCount == writerWarps && totalBytes == 0;
        bool validMixedGroup =
            hasTME && hasLocalStore && group.tmeGroupArrivalCount == 1 &&
            group.localStoreArrivalCount == writerWarps &&
            group.fullArrivalCount == 1 + writerWarps && totalBytes > 0;
        if ((!validTMEGroup && !validLocalStoreGroup && !validMixedGroup) ||
            totalBytes != group.totalTransactionBytes ||
            totalBytes != *state.barrierPlan.full.transactionBytes ||
            group.fullArrivalCount != state.barrierPlan.full.arrivalCount)
          return group.commit.emitOpError(
              "MUSA TLE pipe lowering received inconsistent completion "
              "accounting");
      }

      tt::FuncOp function = state.create->getParentOfType<tt::FuncOp>();
      if (!function)
        return state.create.emitOpError(
            "MUSA TLE pipe requires an enclosing function for barrier "
            "allocation");
      int64_t slots = externalFull ? 0 : state.barrierPlan.full.capacity;
      if (state.barrierPlan.empty)
        slots += state.barrierPlan.empty->capacity;
      pipeBarrierSlots[function.getOperation()] += slots;
      pipeBarrierAnchors.try_emplace(function.getOperation(), state.create);
    }

    for (const auto &entry : pipeBarrierSlots) {
      Operation *functionOp = entry.first;
      auto function = cast<FunctionOpInterface>(functionOp);
      int64_t current = 0;
      if (auto next =
              function->getAttrOfType<IntegerAttr>(ttmg::kNextBarrierIdAttr))
        current = next.getInt();
      else if (auto max = function->getAttrOfType<IntegerAttr>(
                   ttmg::kMaxBarrierIdAttr))
        current = max.getInt();
      current = std::max<int64_t>(current,
                                  ttmg::getImplicitAsyncBarrierFloor(function));
      if (entry.second > ttmg::kMaxBarrierId ||
          current > ttmg::kMaxBarrierId - entry.second) {
        musa_tle::PipeCreateOp anchor = pipeBarrierAnchors.lookup(functionOp);
        return anchor.emitOpError(
            "MUSA TLE pipe barrier allocation exceeds hardware barrier id "
            "limit");
      }
    }
    return success();
  }

  static Value toI32Phase(OpBuilder &builder, Location loc, Value phase,
                          bool invert) {
    Value value = phase;
    if (invert) {
      Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
      value = arith::XOrIOp::create(builder, loc, value, one);
    }
    return arith::ExtUIOp::create(builder, loc, builder.getI32Type(), value);
  }

  static Value createIndex(OpBuilder &builder, Location loc, Operation *use,
                           Value base, Value stage) {
    Value capturedBase = captureForUse(use, base);
    return musa_tle::BarrierIndexOp::create(builder, loc, capturedBase, stage);
  }

  static void lowerLocalStoreArrival(OpBuilder &builder, Location loc,
                                     Operation *use, Value fullBase,
                                     const musa_tle::PipeCommitGroup &group) {
    Value barrier = createIndex(builder, loc, use, fullBase, group.stage);
    Value phase = toI32Phase(builder, loc, group.phase, false);
    musa_tle::BarrierArriveOp::create(builder, loc, barrier, phase,
                                      builder.getI32IntegerAttr(1));
  }

  static bool usesLocalStoreTransport(const musa_tle::PipeState &state) {
    return !state.commitGroups.empty() &&
           llvm::any_of(state.commitGroups, [](const auto &group) {
             return !group->completionSources.empty() &&
                    llvm::any_of(
                        group->completionSources, [](const auto &source) {
                          return source.kind ==
                                 musa_tle::PipeTransportKind::LocalStore;
                        });
           });
  }

  static LogicalResult materializePipeArtifacts(
      const musa_tle::PipeAnalysisResult &analysis,
      llvm::DenseMap<const musa_tle::PipeState *, PipeLoweringArtifacts>
          &artifacts,
      int64_t &nextLocalStoreGroupId) {
    for (const std::unique_ptr<musa_tle::PipeState> &ownedState :
         analysis.getPipes()) {
      const musa_tle::PipeState *state = ownedState.get();
      musa_tle::PipeCreateOp create = state->create;
      OpBuilder builder(create);
      Location loc = create.getLoc();
      const musa_tle::PipeBarrierPlan &plan = state->barrierPlan;
      bool oneShot =
          state->lifecycle.mode == musa_tle::PipeLifecycleMode::OneShot;
      if ((!plan.full.transactionBytes && !plan.closeTagPlan) ||
          (oneShot ? plan.empty.has_value() : !plan.empty.has_value()) ||
          (!oneShot && plan.empty->storageOwner !=
                           musa_tle::PipeBarrierStorageOwner::Pipe) ||
          !plan.writerParticipant ||
          (plan.full.storageOwner ==
               musa_tle::PipeBarrierStorageOwner::External &&
           (!plan.externalFull || !plan.full.externalStorage)) ||
          (plan.full.storageOwner == musa_tle::PipeBarrierStorageOwner::Pipe &&
           (plan.externalFull || plan.full.externalStorage)))
        return create.emitOpError(
            "internal MUSA TLE pipe lowering requires a valid full "
            "barrier plan");

      auto capacity = builder.getI32IntegerAttr(plan.full.capacity);
      auto fullArrivals = builder.getI32IntegerAttr(plan.full.arrivalCount);
      auto pending = builder.getI32IntegerAttr(0);
      auto ready = builder.getI32IntegerAttr(1);
      IntegerAttr bytes;
      if (*plan.full.transactionBytes > 0)
        bytes = builder.getI32IntegerAttr(*plan.full.transactionBytes);
      Value fullBase;
      if (plan.full.storageOwner ==
          musa_tle::PipeBarrierStorageOwner::External) {
        fullBase = plan.full.externalStorage;
      } else {
        fullBase = musa_tle::BarrierAllocOp::create(
            builder, loc, capacity, fullArrivals, pending, bytes);
        fullBase.getDefiningOp()->setAttr(kPipeBarrierRingAttr,
                                          builder.getUnitAttr());
      }
      Value emptyBase;
      if (plan.empty) {
        emptyBase = musa_tle::BarrierAllocOp::create(
            builder, loc, builder.getI32IntegerAttr(plan.empty->capacity),
            builder.getI32IntegerAttr(plan.empty->arrivalCount), ready,
            IntegerAttr());
        emptyBase.getDefiningOp()->setAttr(kPipeBarrierRingAttr,
                                           builder.getUnitAttr());
      }

      Value closeTags;
      ttg::MemDescType closeTagSlotType;
      RankedTensorType closeTagTensorType;
      RankedTensorType closeTagStoreTensorType;
      RankedTensorType closeTagInitTensorType;
      RankedTensorType closeTagInitArrayTensorType;
      int32_t closeTagInitWarpCount = 0;
      if (plan.closeTagPlan) {
        MLIRContext *context = create->getContext();
        auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
        Attribute arrayEncoding = getCloseTagEncoding(context, 2);
        Attribute slotEncoding = getCloseTagEncoding(context, 1);
        auto arrayType = ttg::MemDescType::get(
            {plan.closeTagPlan->capacity, 1}, builder.getI32Type(),
            arrayEncoding, sharedMemorySpace, /*mutableMemory=*/true);
        closeTags = ttg::LocalAllocOp::create(builder, loc, arrayType);
        closeTagSlotType = ttg::MemDescType::get(
            {1}, builder.getI32Type(), slotEncoding, sharedMemorySpace,
            /*mutableMemory=*/true);
        ModuleOp module = create->getParentOfType<ModuleOp>();
        int moduleWarps =
            module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName).getInt();
        closeTagTensorType =
            getCloseTagTensorType(create, builder, {1}, moduleWarps);
        closeTagStoreTensorType = getCloseTagTensorType(
            create, builder, {1}, plan.writerParticipant->warpCount);
        closeTagInitTensorType =
            getCloseTagTensorType(create, builder, {1}, moduleWarps);
        // Triton distributed tensors require a power-of-two element count for
        // a single vector store.  Use that compact initialization when it is
        // representable; non-power-of-two rings fall back to the existing
        // per-slot stores below.
        int32_t tagCapacity = plan.closeTagPlan->capacity;
        if (tagCapacity > 0 && (tagCapacity & (tagCapacity - 1)) == 0)
          closeTagInitArrayTensorType = getCloseTagTensorType(
              create, builder, {tagCapacity, 1}, moduleWarps);
        closeTagInitWarpCount = moduleWarps;
      }

      SmallVector<std::optional<int64_t>> localStoreGroupByField(
          state->fields.size());
      if (usesLocalStoreTransport(*state)) {
        for (const musa_tle::PipeFieldState &field : state->fields) {
          if (field.transportKind != musa_tle::PipeTransportKind::LocalStore)
            continue;
          int64_t group = nextLocalStoreGroupId++;
          localStoreGroupByField[field.index] = group;
          IntegerAttr groupAttr = builder.getI64IntegerAttr(group);
          Operation *root = field.memdescRoot.getDefiningOp();
          if (!root)
            return create.emitOpError(
                "local-store pipe field must have a materialized shared "
                "allocation root");
          root->setAttr(kPipeLocalStoreGroupAttr, groupAttr);
        }
      }

      PipeLoweringArtifacts pipeArtifacts;
      pipeArtifacts.fullBase = fullBase;
      pipeArtifacts.emptyBase = emptyBase;
      pipeArtifacts.closeTags = closeTags;
      pipeArtifacts.closeTagSlotType = closeTagSlotType;
      pipeArtifacts.closeTagTensorType = closeTagTensorType;
      pipeArtifacts.closeTagStoreTensorType = closeTagStoreTensorType;
      pipeArtifacts.closeTagInitTensorType = closeTagInitTensorType;
      pipeArtifacts.closeTagInitArrayTensorType = closeTagInitArrayTensorType;
      pipeArtifacts.closeTagInitWarpCount = closeTagInitWarpCount;
      pipeArtifacts.localStoreGroupByField = std::move(localStoreGroupByField);
      pipeArtifacts.writerParticipant = *plan.writerParticipant;
      pipeArtifacts.readerParticipants = plan.readerParticipants;

      // A static warp-specialized close tag is shared by partitions.  Publish
      // its initial false values before dispatch so every reader partition
      // observes the same initialized ring without adding a CTA barrier.
      if (plan.closeTagPlan &&
          state->executionMode ==
              musa_tle::PipeExecutionMode::StaticWarpSpecialized) {
        if (Operation *alloc = closeTags.getDefiningOp())
          builder.setInsertionPointAfter(alloc);
        initializeCloseTags(builder, loc, pipeArtifacts, create,
                            state->capacity, /*useInitTensorType=*/true);
      }
      artifacts[state] = std::move(pipeArtifacts);
    }
    return success();
  }

  LogicalResult rewrite(ModuleOp module,
                        const musa_tle::PipeAnalysisResult &analysis) {
    llvm::DenseMap<const musa_tle::PipeState *, PipeLoweringArtifacts>
        artifacts;
    llvm::DenseSet<const musa_tle::PipeState *> initializedCloseTags;
    int64_t nextCompletionGroupId = 0;
    int64_t nextLocalStoreGroupId = 0;

    if (failed(materializePipeArtifacts(analysis, artifacts,
                                        nextLocalStoreGroupId)))
      return failure();

    for (const std::unique_ptr<musa_tle::PipeState> &ownedState :
         analysis.getPipes()) {
      const musa_tle::PipeState *state = ownedState.get();
      if (state->executionMode ==
              musa_tle::PipeExecutionMode::StaticWarpSpecialized &&
          state->barrierPlan.closeTagPlan)
        initializedCloseTags.insert(state);
    }

    for (Operation *op : analysis.getLifecycleOps()) {
      const musa_tle::PipeState *state = analysis.lookupPipe(op);
      if (!state)
        return op->emitOpError(
            "internal MUSA TLE pipe lowering lost analyzed pipe state");

      OpBuilder builder(op);
      Location loc = op->getLoc();

      if (isa<musa_tle::PipeCreateOp>(op)) {
        if (artifacts.find(state) == artifacts.end())
          return op->emitOpError(
              "internal MUSA TLE pipe lowering lost barrier artifacts");
        op->erase();
        continue;
      }

      auto artifactsIt = artifacts.find(state);
      if (artifactsIt == artifacts.end())
        return op->emitOpError(
            "internal MUSA TLE pipe lowering lost barrier artifacts");
      const PipeLoweringArtifacts &pipeArtifacts = artifactsIt->second;

      if (auto acquire = dyn_cast<musa_tle::PipeWriterAcquireOp>(op)) {
        if (state->lifecycle.mode == musa_tle::PipeLifecycleMode::OneShot) {
          acquire.erase();
          continue;
        }
        if (pipeArtifacts.closeTags &&
            initializedCloseTags.insert(state).second) {
          initializeCloseTags(builder, loc, pipeArtifacts, acquire,
                              state->capacity);
        }
        Value barrier = createIndex(builder, loc, op, pipeArtifacts.emptyBase,
                                    acquire.getStage());
        Value phase = toI32Phase(builder, loc, acquire.getPhase(), true);
        auto wait =
            musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        setLocalStoreWaitGroups(wait, builder,
                                getAllLocalStoreGroups(pipeArtifacts));
        acquire.erase();
        continue;
      }

      if (auto commit = dyn_cast<musa_tle::PipeWriterCommitOp>(op)) {
        const musa_tle::PipeCommitGroup *group =
            analysis.lookupCommitGroup(commit);
        if (!group || group->completionSources.empty())
          return commit.emitOpError("lost the analyzed pipe completion group");

        bool hasTME = llvm::any_of(
            group->completionSources,
            [](const musa_tle::PipeCompletionSource &source) {
              return source.kind == musa_tle::PipeTransportKind::TME;
            });
        bool hasLocalStore = llvm::any_of(
            group->completionSources,
            [](const musa_tle::PipeCompletionSource &source) {
              return source.kind == musa_tle::PipeTransportKind::LocalStore;
            });
        if (hasLocalStore) {
          lowerLocalStoreArrival(builder, loc, op, pipeArtifacts.fullBase,
                                 *group);
        }

        if (hasTME) {
          ttg::TMACopyOp firstCopy;
          for (const musa_tle::PipeCompletionSource &source :
               group->completionSources) {
            if (source.kind == musa_tle::PipeTransportKind::TME) {
              firstCopy = dyn_cast_or_null<ttg::TMACopyOp>(source.operation);
              break;
            }
          }
          if (!firstCopy)
            return commit.emitOpError("lost an analyzed pipe TME copy");

          OpBuilder groupBuilder(firstCopy);
          Value barrier =
              createIndex(groupBuilder, firstCopy.getLoc(), firstCopy,
                          pipeArtifacts.fullBase, group->stage);
          int64_t groupId = nextCompletionGroupId++;
          int64_t memberCount = llvm::count_if(
              group->completionSources,
              [](const musa_tle::PipeCompletionSource &source) {
                return source.kind == musa_tle::PipeTransportKind::TME;
              });
          int64_t memberIndex = 0;
          for (const musa_tle::PipeCompletionSource &source :
               group->completionSources) {
            if (source.kind != musa_tle::PipeTransportKind::TME)
              continue;
            ttg::TMACopyOp copy =
                dyn_cast_or_null<ttg::TMACopyOp>(source.operation);
            if (!copy)
              return commit.emitOpError("lost an analyzed pipe TME copy");

            OpBuilder copyBuilder(copy);
            auto replacement = ttg::TMACopyOp::create(
                copyBuilder, copy.getLoc(), copy.getSrc(), copy.getDst(),
                copy.getIndices(), barrier);
            replacement->setDiscardableAttrs(
                copy->getDiscardableAttrDictionary());
            replacement->setAttr("expect_bytes", copyBuilder.getI32IntegerAttr(
                                                     source.transactionBytes));
            replacement->setAttr(ttmg::kTLECompletionGroupAttr,
                                 copyBuilder.getDenseI64ArrayAttr(
                                     {groupId, memberIndex++, memberCount,
                                      group->totalTransactionBytes}));
            copy.erase();
          }
        }
        commit.erase();
        continue;
      }

      if (auto wait = dyn_cast<musa_tle::PipeReaderWaitOp>(op)) {
        FailureOr<musa_tle::PipeBarrierParticipant> readerParticipant =
            getReaderBarrierParticipant(*state, pipeArtifacts,
                                        analysis.lookupEndpoint(wait), wait);
        if (failed(readerParticipant))
          return failure();
        Value barrier = createIndex(builder, loc, op, pipeArtifacts.fullBase,
                                    wait.getStage());
        Value phase =
            state->lifecycle.mode == musa_tle::PipeLifecycleMode::OneShot
                ? arith::ConstantIntOp::create(builder, loc, 0, 32)
                : toI32Phase(builder, loc, wait.getPhase(), false);
        auto loweredWait =
            musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        FailureOr<SmallVector<int64_t>> waitGroups = getReaderLocalStoreGroups(
            pipeArtifacts, analysis.lookupEndpoint(wait), wait);
        if (failed(waitGroups))
          return failure();
        setLocalStoreWaitGroups(loweredWait, builder, *waitGroups);
        if (const musa_tle::PipeReaderDrainGroup *drain =
                analysis.lookupReaderDrainGroup(wait)) {
          for (const musa_tle::PipeReaderDrainSource &source :
               drain->drainSources) {
            auto copy = dyn_cast_or_null<ttg::TMACopyOp>(source.operation);
            if (!copy)
              return wait.emitOpError(
                  "internal MUSA TLE pipe lowering lost reader TME store");
            FailureOr<int32_t> issueThread =
                musa_tle::getPipePartitionStartThread(source.operation);
            if (failed(issueThread))
              return failure();
            copy->setAttr(ttmg::kTMEIssueThreadAttr,
                          builder.getI32IntegerAttr(*issueThread));
          }
        }
        if (!wait.getIsClosed().use_empty()) {
          Value isClosed;
          if (state->barrierPlan.closeTagPlan) {
            builder.setInsertionPointAfter(loweredWait);
            RankedTensorType loadTensorType = getCloseTagTensorType(
                wait, builder, {1}, readerParticipant->warpCount);
            isClosed = loadCloseTag(builder, loc, pipeArtifacts, op,
                                    wait.getStage(), loadTensorType);
          } else {
            isClosed = arith::ConstantIntOp::create(builder, loc, 0, 1);
          }
          wait.getIsClosed().replaceAllUsesWith(isClosed);
        }
        wait.erase();
        continue;
      }

      if (auto close = dyn_cast<musa_tle::PipeWriterCloseOp>(op)) {
        if (state->lifecycle.mode == musa_tle::PipeLifecycleMode::OneShot)
          return close.emitOpError(
              "MUSA TLE one-shot pipe does not support writer.close");
        const musa_tle::PipeCloseGeneration *generation =
            analysis.lookupCloseGeneration(close);
        if (!generation || !pipeArtifacts.closeTags)
          return close.emitOpError(
              "internal MUSA TLE pipe lowering lost close generation");

        if (initializedCloseTags.insert(state).second)
          initializeCloseTags(builder, loc, pipeArtifacts, close,
                              state->capacity);

        Value emptyBarrier = createIndex(
            builder, loc, op, pipeArtifacts.emptyBase, generation->stage);
        Value emptyPhase = toI32Phase(builder, loc, generation->phase, true);
        auto emptyWait = musa_tle::BarrierWaitOp::create(
            builder, loc, emptyBarrier, emptyPhase);
        setLocalStoreWaitGroups(emptyWait, builder,
                                getAllLocalStoreGroups(pipeArtifacts));

        builder.setInsertionPointAfter(emptyWait);
        storeCloseTag(builder, loc, pipeArtifacts, op, generation->stage,
                      /*value=*/true, pipeArtifacts.closeTagStoreTensorType,
                      pipeArtifacts.writerParticipant.warpCount);

        Operation *tagStore = op->getPrevNode();
        builder.setInsertionPointAfter(tagStore);
        Value fullBarrier = createIndex(
            builder, loc, op, pipeArtifacts.fullBase, generation->stage);
        Value predicate = arith::ConstantIntOp::create(builder, loc, 1, 1);
        Operation *controlOp = nullptr;
        if (generation->controlArrivalCount == 1) {
          auto control = ttmg::ArriveBarrierNoRetOp::create(
              builder, loc, fullBarrier, predicate);
          controlOp = control.getOperation();
          int32_t issueThread = 0;
          if (failed(getWriterIssueThread(op, issueThread)))
            return failure();
          control->setAttr(ttmg::kTMEIssueThreadAttr,
                           builder.getI32IntegerAttr(issueThread));
          control->setAttr(ttmg::kTMEExplicitCompletionAttr,
                           builder.getUnitAttr());
        }
        if (generation->localStoreArrivalCount > 0) {
          builder.setInsertionPointAfter(controlOp ? controlOp
                                                   : predicate.getDefiningOp());
          Value phase = toI32Phase(builder, loc, generation->phase, false);
          musa_tle::BarrierArriveOp::create(builder, loc, fullBarrier, phase,
                                            builder.getI32IntegerAttr(1));
        }
        close.erase();
        continue;
      }

      auto release = cast<musa_tle::PipeReaderReleaseOp>(op);
      if (state->lifecycle.mode == musa_tle::PipeLifecycleMode::OneShot) {
        release.erase();
        continue;
      }
      const musa_tle::PipeReaderDrainGroup *drain =
          analysis.lookupReaderDrainGroup(release);
      if (failed(getReaderBarrierParticipant(*state, pipeArtifacts,
                                             analysis.lookupEndpoint(release),
                                             release)))
        return failure();
      if (!drain)
        return release.emitOpError("lost the analyzed pipe reader drain group");
      Value barrier = createIndex(builder, loc, op, pipeArtifacts.emptyBase,
                                  release.getStage());
      Value phase = toI32Phase(builder, loc, drain->phase, false);
      musa_tle::BarrierArriveOp::create(builder, loc, barrier, phase,
                                        builder.getI32IntegerAttr(1));
      release.erase();
    }

    bool hasPipeOps = false;
    module.walk([&](Operation *op) { hasPipeOps |= isPipeOp(op); });
    if (hasPipeOps)
      return module.emitError("MUSA TLE pipe lowering left lifecycle ops");
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<std::unique_ptr<musa_tle::PipeAnalysisResult>> analysis =
        musa_tle::analyzeMUSAPipes(module);
    if (failed(analysis) || failed(verifyLoweringCapabilities(**analysis)) ||
        failed(rewrite(module, **analysis)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
