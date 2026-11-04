#include "IR/Dialect.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>
#include <optional>

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLELOWERPIPETOBARRIERS
#include "Transforms/Passes.h.inc"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct PipeState {
  // Per-stage barriers signaled by writers after a slot is full and readable.
  Value fullBars;
  // Per-stage barriers signaled by readers after a slot is empty and reusable.
  Value emptyBars;
  // Per-stage close markers: true means the slot carries a close signal.
  Value closeTags;
  ttg::MemDescType barrierSlotType;
  ttg::MemDescType closeTagSlotType;
  Value drainBarrier;
  SmallVector<std::string> readerNames;
  bool oneShot;
  std::optional<int32_t> writerThreadCount;
  std::optional<int32_t> writerFullCount;
  std::map<std::string, std::pair<int32_t, int32_t>> readerTasks;
};

static int64_t getPipeCapacity(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("capacity").getInt();
}

static OperandRange getPipeFields(Operation *op) {
  if (auto pipeOp = dyn_cast<PipeCreateOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<PipeWriterAcquireOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<PipeWriterCommitOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<PipeWriterCloseOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<PipeReaderWaitOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<PipeReaderReleaseOp>(op))
    return pipeOp.getFields();
  return cast<PipeDrainOp>(op).getFields();
}

static bool isPipeLifecycleOp(Operation *op) {
  return isa<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
             PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp, PipeDrainOp>(op);
}

static bool containsPipeLifecycleOp(tt::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isPipeLifecycleOp(op))
      found = true;
  });
  return found;
}

static Value canonicalizePipeField(Value field) {
  while (auto blockArg = dyn_cast<BlockArgument>(field)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (partitions) {
      auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
      if (!wsOp)
        break;
      unsigned argNo = blockArg.getArgNumber();
      OperandRange captures = wsOp.getExplicitCaptures();
      if (argNo >= captures.size())
        break;
      field = captures[argNo];
      continue;
    }
    break;
  }
  return field;
}

static std::optional<std::pair<ttg::WarpSpecializeOp, Region *>>
getEnclosingWarpSpecializePartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
      return std::make_pair(
          cast<ttg::WarpSpecializeOp>(partitions->getParentOp()), region);
    region = parent->getParentRegion();
  }
  return std::nullopt;
}

static bool isDefinedInsideRegion(Value value, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return region->isAncestor(blockArg.getOwner()->getParent());
  Operation *def = value.getDefiningOp();
  return def && region->isAncestor(def->getParentRegion());
}

static Value getWarpSpecializeCaptureForUse(Operation *useOp, Value value) {
  auto partition = getEnclosingWarpSpecializePartition(useOp);
  if (!partition)
    return value;

  ttg::WarpSpecializeOp wsOp = partition->first;
  Region *region = partition->second;
  if (isDefinedInsideRegion(value, region))
    return value;

  OperandRange captures = wsOp.getExplicitCaptures();
  for (auto indexed : llvm::enumerate(captures)) {
    if (indexed.value() == value)
      return region->getArgument(indexed.index());
  }

  wsOp->insertOperands(wsOp.getNumOperands(), value);
  unsigned captureIndex = wsOp.getNumOperands() - 1;
  for (Region *partitionRegion : wsOp.getPartitionRegions())
    partitionRegion->addArgument(value.getType(), value.getLoc());
  return region->getArgument(captureIndex);
}

static std::string getPipeKey(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << getPipeCapacity(op) << "|";
  op->getAttr("scope").print(os);
  os << "|";
  if (Attribute pipeName = op->getAttr("pipe_name"))
    pipeName.print(os);
  os << "|";
  op->getAttr("field_names").print(os);
  os << "|";
  for (Value field : getPipeFields(op))
    os << canonicalizePipeField(field).getAsOpaquePointer() << ",";
  return key;
}

static void setAsyncTaskId(Operation *op, int32_t id) {
  SmallVector<int32_t, 1> ids{id};
  op->setAttr("async_task_id", DenseI32ArrayAttr::get(op->getContext(), ids));
}

static void setRoleTaskId(Operation *source, Operation *created,
                          int32_t defaultTaskId) {
  if (Attribute existing = source->getAttr("async_task_id")) {
    created->setAttr("async_task_id", existing);
    return;
  }
  setAsyncTaskId(created, defaultTaskId);
}

static int32_t getEnclosingDefaultTaskId(Operation *op,
                                         int32_t nonWarpSpecializeDefault) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &wsOp.getDefaultRegion())
        return 0;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      for (auto indexed : llvm::enumerate(partitions.getRegions())) {
        if (region == indexed.value())
          return static_cast<int32_t>(indexed.index()) + 1;
      }
    }
    region = parent->getParentRegion();
  }
  return nonWarpSpecializeDefault;
}

static FailureOr<int32_t> getSingleTaskId(Operation *op,
                                          int32_t defaultTaskId) {
  auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id");
  if (!attr)
    return defaultTaskId;
  ArrayRef<int32_t> ids = attr.asArrayRef();
  if (ids.size() != 1) {
    op->emitOpError("requires exactly one async_task_id for pipe lifecycle "
                    "ops");
    return failure();
  }
  return ids.front();
}

static FailureOr<int32_t> getTaskThreadCount(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) {
    op->emitOpError("requires enclosing module to infer pipe task "
                    "thread count");
    return failure();
  }
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  if (numWarps <= 0 || threadsPerWarp <= 0) {
    op->emitOpError("requires positive num_warps and threads_per_warp "
                    "to infer pipe task thread count");
    return failure();
  }
  return numWarps * threadsPerWarp;
}

static LogicalResult recordWriterTask(PipeState &state, Operation *op,
                                      int32_t taskId, int32_t threadCount) {
  if (state.writerThreadCount && *state.writerThreadCount != threadCount)
    return op->emitOpError("uses writer thread count ")
           << threadCount << " but pipe already has writer thread count "
           << *state.writerThreadCount;
  state.writerThreadCount = threadCount;
  return success();
}

static LogicalResult setWriterFullCount(PipeState &state, Operation *op,
                                        int32_t count) {
  if (state.writerFullCount && *state.writerFullCount != count)
    return op->emitOpError("requires pipe full barrier count ")
           << count << " but pipe already uses full barrier count "
           << *state.writerFullCount
           << "; local-store pipe commits on one pipe must have one proven "
              "writer participant contract";
  state.writerFullCount = count;
  return success();
}

static LogicalResult recordReaderTask(PipeState &state, Operation *op,
                                      StringRef readerName, int32_t taskId,
                                      int32_t threadCount) {
  std::string key =
      readerName.empty() ? std::string("<default>") : readerName.str();
  auto [it, inserted] =
      state.readerTasks.emplace(key, std::make_pair(taskId, threadCount));
  if (!inserted) {
    if (it->second.first != taskId || it->second.second != threadCount)
      return op->emitOpError("inconsistent reader task metadata for reader ")
             << key;
  }
  return success();
}

static Attribute getBarrierEncoding(MLIRContext *context) {
  auto barrierCTALayout = ttg::CTAEncodingAttr::getDefault(context, 1);
  return ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0},
                                              barrierCTALayout);
}

static Attribute getCloseTagEncoding(MLIRContext *context, int64_t rank) {
  SmallVector<unsigned> order;
  for (int64_t dim = rank - 1; dim >= 0; --dim)
    order.push_back(static_cast<unsigned>(dim));
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(context, rank);
  return ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, order,
                                              ctaLayout);
}

static RankedTensorType getCloseTagTensorType(Operation *op, OpBuilder &builder,
                                              ArrayRef<int64_t> shape) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(module);
  Attribute encoding = ttg::getDefaultBlockedEncoding(context, shape, numWarps,
                                                      threadsPerWarp, numCTAs);
  return RankedTensorType::get(shape, builder.getI32Type(), encoding);
}

static Value createCloseTagTensor(OpBuilder &builder, Location loc,
                                  RankedTensorType tensorType, bool value) {
  Value scalar = arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 32);
  return tt::SplatOp::create(builder, loc, tensorType, scalar);
}

static Value createCloseTagSlot(OpBuilder &builder, Location loc,
                                const PipeState &state, Value closeTags,
                                Value stage) {
  return ttg::MemDescIndexOp::create(builder, loc, state.closeTagSlotType,
                                     closeTags, stage);
}

static Value createBarrierArray(OpBuilder &builder, Location loc,
                                int64_t capacity) {
  auto *context = builder.getContext();
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto encoding = getBarrierEncoding(context);
  auto arrayType =
      ttg::MemDescType::get({capacity, 1}, builder.getI64Type(), encoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(builder, loc, arrayType, Value());
}

static Value getBarrierView(OpBuilder &builder, Location loc,
                            const PipeState &state, Value array, Value stage) {
  return ttg::MemDescIndexOp::create(builder, loc, state.barrierSlotType, array,
                                     stage);
}

static void initBarrierArray(OpBuilder &builder, Location loc,
                             const PipeState &state, Value array,
                             int64_t capacity, int32_t arriveCount) {
  for (int64_t i = 0; i < capacity; ++i) {
    Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
    Value view = getBarrierView(builder, loc, state, array, idx);
    InitBarrierOp::create(builder, loc, view, arriveCount);
  }
}

static Value xorPhaseForEmpty(OpBuilder &builder, Location loc, Value phase) {
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
  return arith::XOrIOp::create(builder, loc, phase, one);
}

static Value phaseAsI32(OpBuilder &builder, Location loc, Value phaseI1) {
  return arith::ExtUIOp::create(builder, loc, builder.getI32Type(), phaseI1);
}

struct PipeDrainBuildState {
  SmallVector<std::string> readerNames;
  bool oneShot = false;
  std::optional<std::pair<int32_t, int32_t>> writerCloseTask;
  std::map<std::string, std::pair<int32_t, int32_t>> readerReleaseTasks;
  std::map<int32_t, int32_t> readyTasks;
  std::map<int32_t, int32_t> drainTasks;
  DenseSet<int32_t> drainedTasks;
};

static LogicalResult recordDrainReadyTask(PipeDrainBuildState &state,
                                          Operation *op, int32_t taskId,
                                          int32_t threadCount) {
  auto it = state.readyTasks.find(taskId);
  if (it != state.readyTasks.end() && it->second != threadCount)
    return op->emitOpError("uses pipe drain participant task ")
           << taskId << " with inconsistent thread counts " << it->second
           << " and " << threadCount;
  state.readyTasks[taskId] = threadCount;
  return success();
}

static LogicalResult recordDrainArrival(PipeDrainBuildState &state,
                                        Operation *op, int32_t taskId,
                                        int32_t threadCount) {
  if (state.oneShot)
    return op->emitOpError("does not support wait_drained on one_shot pipe");
  if (!state.readyTasks.count(taskId))
    return op->emitOpError(
        "requires a preceding pipe.writer_close or pipe.reader_release in the "
        "same async task before wait_drained");
  if (state.drainedTasks.contains(taskId))
    return op->emitOpError("has duplicate wait_drained in async_task_id ")
           << taskId;
  state.drainedTasks.insert(taskId);
  state.drainTasks[taskId] = threadCount;
  return success();
}

static LogicalResult verifyNoUseAfterDrain(PipeDrainBuildState &state,
                                           Operation *op, int32_t taskId) {
  if (!state.drainedTasks.contains(taskId))
    return success();
  return op->emitOpError("uses pipe after wait_drained in async_task_id ")
         << taskId;
}

static LogicalResult
verifyRequiredDrainParticipant(PipeDrainBuildState &state, Operation *op,
                               StringRef role,
                               std::optional<std::pair<int32_t, int32_t>> task) {
  if (!task)
    return op->emitOpError("wait_drained requires ") << role
                                                     << " to reach drain";
  int32_t taskId = task->first;
  auto it = state.drainTasks.find(taskId);
  if (it == state.drainTasks.end())
    return op->emitOpError("wait_drained requires ") << role
                                                     << " async_task_id "
                                                     << taskId
                                                     << " to call drain";
  if (it->second != task->second)
    return op->emitOpError("wait_drained participant thread count mismatch "
                           "for ")
           << role;
  return success();
}

static LogicalResult analyzePipeDrains(
    ArrayRef<Operation *> ops, std::map<std::string, PipeCreateOp> &pipes,
    std::map<std::string, int32_t> &drainParticipantCounts) {
  std::map<std::string, PipeDrainBuildState> states;

  for (const auto &entry : pipes) {
    PipeDrainBuildState state;
    if (auto readers = entry.second->getAttrOfType<ArrayAttr>("readers"))
      for (Attribute reader : readers)
        state.readerNames.push_back(cast<StringAttr>(reader).getValue().str());
    if (auto oneShot = entry.second->getAttrOfType<BoolAttr>("one_shot"))
      state.oneShot = oneShot.getValue();
    states.emplace(entry.first, std::move(state));
  }

  for (Operation *op : ops) {
    if (isa<PipeCreateOp>(op))
      continue;

    std::string key = getPipeKey(op);
    auto defIt = pipes.find(key);
    if (defIt == pipes.end())
      return op->emitOpError("requires a preceding matching pipe.create");
    PipeDrainBuildState &state = states[key];

    auto defaultTask = [&]() -> int32_t {
      if (isa<PipeWriterAcquireOp, PipeWriterCommitOp, PipeWriterCloseOp>(op))
        return getEnclosingDefaultTaskId(op, /*writer=*/0);
      if (isa<PipeDrainOp>(op))
        return getEnclosingDefaultTaskId(op, /*nonWarpSpecializeDefault=*/0);
      return getEnclosingDefaultTaskId(op, /*reader=*/1);
    };
    auto taskId = getSingleTaskId(op, defaultTask());
    if (failed(taskId))
      return failure();
    auto threadCount = getTaskThreadCount(op);
    if (failed(threadCount))
      return failure();

    if (isa<PipeDrainOp>(op)) {
      if (failed(recordDrainArrival(state, op, *taskId, *threadCount)))
        return failure();
      continue;
    }

    if (failed(verifyNoUseAfterDrain(state, op, *taskId)))
      return failure();

    if (auto close = dyn_cast<PipeWriterCloseOp>(op)) {
      if (state.writerCloseTask && state.writerCloseTask->first != *taskId)
        return close.emitOpError("uses writer async_task_id ")
               << *taskId << " but pipe close already has writer "
               << "async_task_id " << state.writerCloseTask->first;
      state.writerCloseTask = std::make_pair(*taskId, *threadCount);
      if (failed(recordDrainReadyTask(state, op, *taskId, *threadCount)))
        return failure();
      continue;
    }

    if (auto release = dyn_cast<PipeReaderReleaseOp>(op)) {
      std::string readerName;
      if (auto attr = release->getAttrOfType<StringAttr>("reader_name"))
        readerName = attr.getValue().str();
      auto it = state.readerReleaseTasks.find(readerName);
      if (it != state.readerReleaseTasks.end() &&
          it->second.first != *taskId)
        return release.emitOpError("uses reader ")
               << readerName << " async_task_id " << *taskId
               << " but that reader already released from async_task_id "
               << it->second.first;
      state.readerReleaseTasks[readerName] =
          std::make_pair(*taskId, *threadCount);
      if (failed(recordDrainReadyTask(state, op, *taskId, *threadCount)))
        return failure();
    }
  }

  for (auto &[key, state] : states) {
    if (state.drainTasks.empty())
      continue;

    Operation *diagnosticOp = pipes[key].getOperation();
    if (failed(verifyRequiredDrainParticipant(
            state, diagnosticOp, "writer close", state.writerCloseTask)))
      return failure();

    if (state.readerNames.empty()) {
      auto it = state.readerReleaseTasks.find("");
      std::optional<std::pair<int32_t, int32_t>> defaultReader;
      if (it != state.readerReleaseTasks.end())
        defaultReader = it->second;
      if (failed(verifyRequiredDrainParticipant(state, diagnosticOp,
                                                "default reader release",
                                                defaultReader)))
        return failure();
    } else {
      for (const std::string &readerName : state.readerNames) {
        auto it = state.readerReleaseTasks.find(readerName);
        std::optional<std::pair<int32_t, int32_t>> readerTask;
        if (it != state.readerReleaseTasks.end())
          readerTask = it->second;
        std::string role = (Twine("reader ") + readerName + " release").str();
        if (failed(verifyRequiredDrainParticipant(state, diagnosticOp, role,
                                                  readerTask)))
          return failure();
      }
    }

    int64_t participantCount = 0;
    for (auto &[taskId, threads] : state.drainTasks)
      participantCount += threads;
    if (participantCount <= 0 ||
        participantCount > std::numeric_limits<int32_t>::max())
      return diagnosticOp->emitOpError(
          "wait_drained participant count is out of range");
    drainParticipantCounts[key] = static_cast<int32_t>(participantCount);
  }

  return success();
}

static PipeState createPipeState(PipeCreateOp op, int32_t provisionalCount,
                                 std::optional<int32_t> drainCount) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *context = op->getContext();
  int64_t capacity = getPipeCapacity(op);
  bool oneShot = false;
  if (auto attr = op->getAttrOfType<BoolAttr>("one_shot"))
    oneShot = attr.getValue();

  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  auto barrierEncoding = getBarrierEncoding(context);
  auto barrierSlotType =
      ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);

  Value fullBars = createBarrierArray(builder, loc, capacity);
  Value emptyBars;
  if (!oneShot)
    emptyBars = createBarrierArray(builder, loc, capacity);

  PipeState state;
  state.fullBars = fullBars;
  state.emptyBars = emptyBars;
  state.barrierSlotType = barrierSlotType;
  state.oneShot = oneShot;

  initBarrierArray(builder, loc, state, fullBars, capacity, provisionalCount);
  if (emptyBars)
    initBarrierArray(builder, loc, state, emptyBars, capacity,
                     provisionalCount);

  if (!oneShot) {
    Attribute closeTagArrayEncoding = getCloseTagEncoding(context, 2);
    Attribute closeTagSlotEncoding = getCloseTagEncoding(context, 1);
    auto closeTagArrayType =
        ttg::MemDescType::get({capacity, 1}, builder.getI32Type(),
                              closeTagArrayEncoding, sharedMemorySpace,
                              /*mutableMemory=*/true);
    state.closeTagSlotType =
        ttg::MemDescType::get({1}, builder.getI32Type(), closeTagSlotEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    RankedTensorType closeTagArrayTensorType =
        getCloseTagTensorType(op, builder, {capacity, 1});
    Value initialCloseTags =
        createCloseTagTensor(builder, loc, closeTagArrayTensorType,
                             /*value=*/false);
    state.closeTags = ttg::LocalAllocOp::create(builder, loc, closeTagArrayType,
                                                initialCloseTags);

  }

  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    state.readerNames.reserve(readersAttr.size());
    for (Attribute attr : readersAttr)
      state.readerNames.push_back(cast<StringAttr>(attr).getValue().str());
  }

  if (drainCount) {
    state.drainBarrier = ttg::LocalAllocOp::create(builder, loc, barrierSlotType,
                                                  Value());
    InitBarrierOp::create(builder, loc, state.drainBarrier, *drainCount);
  }
  ::mlir::gpu::BarrierOp::create(builder, loc);
  return state;
}

static void storeCloseTag(OpBuilder &builder, Location loc,
                          const PipeState &state, Value stage, bool value,
                          Operation *source, int32_t taskId) {
  Value closeTags = getWarpSpecializeCaptureForUse(source, state.closeTags);
  Value slot = createCloseTagSlot(builder, loc, state, closeTags, stage);
  RankedTensorType tagType = getCloseTagTensorType(source, builder, {1});
  Value tag =
      createCloseTagTensor(builder, loc, tagType, value);
  auto store = ttg::LocalStoreOp::create(builder, loc, tag, slot);
  setRoleTaskId(source, slot.getDefiningOp(), taskId);
  setRoleTaskId(source, tag.getDefiningOp(), taskId);
  setRoleTaskId(source, store.getOperation(), taskId);
}

static Value loadCloseTag(OpBuilder &builder, Location loc,
                          const PipeState &state, Value stage,
                          Operation *source, int32_t taskId) {
  Value closeTags = getWarpSpecializeCaptureForUse(source, state.closeTags);
  Value slot = createCloseTagSlot(builder, loc, state, closeTags, stage);
  RankedTensorType tagType = getCloseTagTensorType(source, builder, {1});
  Value tagTensor =
      ttg::LocalLoadOp::create(builder, loc, tagType, slot);
  Value tagI32 =
      tt::UnsplatOp::create(builder, loc, builder.getI32Type(), tagTensor);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value tag = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                    tagI32, zero);
  setRoleTaskId(source, slot.getDefiningOp(), taskId);
  setRoleTaskId(source, tagTensor.getDefiningOp(), taskId);
  setRoleTaskId(source, tagI32.getDefiningOp(), taskId);
  setRoleTaskId(source, tag.getDefiningOp(), taskId);
  return tag;
}

static LogicalResult inlinePipeCall(tt::CallOp call, tt::FuncOp callee) {
  if (callee.isExternal())
    return call.emitOpError(
        "cannot inline external callee containing pipe ops");
  Region &body = callee.getBody();
  if (!body.hasOneBlock())
    return call.emitOpError("cannot inline multi-block callee containing pipe "
                            "ops before pipe lowering");

  Block &block = body.front();
  auto returnOp = dyn_cast<tt::ReturnOp>(block.getTerminator());
  if (!returnOp)
    return call.emitOpError("callee containing pipe ops must terminate with "
                            "tt.return before pipe lowering");
  if (returnOp.getNumOperands() != call.getNumResults())
    return call.emitOpError("callee return count does not match call results");

  IRMapping mapping;
  for (auto [arg, operand] :
       llvm::zip(block.getArguments(), call.getOperands()))
    mapping.map(arg, operand);

  OpBuilder builder(call);
  for (Operation &op : block.getOperations()) {
    if (&op == returnOp.getOperation())
      continue;
    builder.clone(op, mapping);
  }

  for (auto [result, returned] :
       llvm::zip(call.getResults(), returnOp.getOperands()))
    result.replaceAllUsesWith(mapping.lookupOrDefault(returned));
  call.erase();
  return success();
}

static LogicalResult inlinePipeHelperCalls(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<tt::CallOp> calls;
    module.walk([&](tt::CallOp call) {
      auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee());
      if (callee && containsPipeLifecycleOp(callee))
        calls.push_back(call);
    });

    for (tt::CallOp call : calls) {
      if (!call->getBlock())
        continue;
      auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee());
      if (!callee || !containsPipeLifecycleOp(callee))
        continue;
      if (failed(inlinePipeCall(call, callee)))
        return failure();
      changed = true;
    }
  }

  for (tt::FuncOp func :
       llvm::make_early_inc_range(module.getOps<tt::FuncOp>())) {
    if (!containsPipeLifecycleOp(func))
      continue;
    if (func.getVisibility() != SymbolTable::Visibility::Public &&
        SymbolTable::symbolKnownUseEmpty(func, module)) {
      func.erase();
      continue;
    }
    if (func.getVisibility() != SymbolTable::Visibility::Public)
      return func.emitOpError("contains pipe ops but still has call sites "
                              "after pipe helper inlining");
  }

  return success();
}

} // namespace

class TritonIluvatarTleLowerPipeToBarriersPass
    : public impl::TritonIluvatarTleLowerPipeToBarriersBase<
          TritonIluvatarTleLowerPipeToBarriersPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(inlinePipeHelperCalls(module))) {
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      if (isPipeLifecycleOp(op))
        ops.push_back(op);
    });
    if (ops.empty())
      return;

    // Provisional arrive count from module default warp count. Refined after
    // scanning lifecycle ops that record partition thread counts.
    int provisionalCount = 128;
    if (auto numWarps = module->getAttrOfType<IntegerAttr>("ttg.num-warps")) {
      int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
      provisionalCount = numWarps.getInt() * threadsPerWarp;
    }

    std::map<std::string, PipeCreateOp> definitions;
    for (Operation *op : ops)
      if (auto create = dyn_cast<PipeCreateOp>(op))
        definitions.emplace(getPipeKey(op), create);
    std::map<std::string, int32_t> drainCounts;
    if (failed(analyzePipeDrains(ops, definitions, drainCounts))) {
      signalPassFailure();
      return;
    }
    std::map<std::string, PipeState> pipes;
    SmallVector<PipeCreateOp> creates;
    for (Operation *op : ops) {
      std::string key = getPipeKey(op);
      if (auto create = dyn_cast<PipeCreateOp>(op)) {
        std::optional<int32_t> drainCount;
        if (auto it = drainCounts.find(key); it != drainCounts.end())
          drainCount = it->second;
        pipes.emplace(key, createPipeState(create, provisionalCount, drainCount));
        creates.push_back(create);
        continue;
      }

      auto it = pipes.find(key);
      if (it == pipes.end()) {
        op->emitOpError("requires a preceding matching pipe.create");
        signalPassFailure();
        return;
      }
      PipeState &state = it->second;
      OpBuilder builder(op);
      Location loc = op->getLoc();

      if (auto acquire = dyn_cast<PipeWriterAcquireOp>(op)) {
        if (state.oneShot) {
          acquire.erase();
          continue;
        }
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*writer=*/0));
        if (failed(taskId)) {
          signalPassFailure();
          return;
        }
        auto threadCount = getTaskThreadCount(op);
        if (failed(threadCount) ||
            failed(recordWriterTask(state, op, *taskId, *threadCount))) {
          signalPassFailure();
          return;
        }
        Value emptyBars = getWarpSpecializeCaptureForUse(op, state.emptyBars);
        Value view =
            getBarrierView(builder, loc, state, emptyBars, acquire.getStage());
        Value phase = phaseAsI32(
            builder, loc, xorPhaseForEmpty(builder, loc, acquire.getPhase()));
        auto wait = WaitBarrierOp::create(builder, loc, view, phase);
        setRoleTaskId(op, view.getDefiningOp(), *taskId);
        setRoleTaskId(op, wait.getOperation(), *taskId);
        acquire.erase();
        continue;
      }

      if (auto commit = dyn_cast<PipeWriterCommitOp>(op)) {
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*writer=*/0));
        auto threadCount = getTaskThreadCount(op);
        if (failed(taskId) || failed(threadCount) ||
            failed(recordWriterTask(state, op, *taskId, *threadCount)) ||
            failed(setWriterFullCount(state, op, *threadCount))) {
          signalPassFailure();
          return;
        }
        Value fullBars = getWarpSpecializeCaptureForUse(op, state.fullBars);
        Value view =
            getBarrierView(builder, loc, state, fullBars, commit.getStage());
        auto arrive = ArriveBarrierOp::create(builder, loc, view, *threadCount);
        setRoleTaskId(op, view.getDefiningOp(), *taskId);
        setRoleTaskId(op, arrive.getOperation(), *taskId);
        commit.erase();
        continue;
      }

      if (auto close = dyn_cast<PipeWriterCloseOp>(op)) {
        if (state.oneShot) {
          close.emitOpError("one_shot pipes do not support close");
          signalPassFailure();
          return;
        }
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*writer=*/0));
        auto threadCount = getTaskThreadCount(op);
        if (failed(taskId) || failed(threadCount) ||
            failed(recordWriterTask(state, op, *taskId, *threadCount)) ||
            failed(setWriterFullCount(state, op, *threadCount))) {
          signalPassFailure();
          return;
        }
        // Close = acquire empty + store close tag + commit full.
        Value emptyBars = getWarpSpecializeCaptureForUse(op, state.emptyBars);
        Value emptyView =
            getBarrierView(builder, loc, state, emptyBars, close.getStage());
        Value emptyPhase = phaseAsI32(
            builder, loc, xorPhaseForEmpty(builder, loc, close.getPhase()));
        auto wait = WaitBarrierOp::create(builder, loc, emptyView, emptyPhase);
        setRoleTaskId(op, emptyView.getDefiningOp(), *taskId);
        setRoleTaskId(op, wait.getOperation(), *taskId);
        storeCloseTag(builder, loc, state, close.getStage(), /*value=*/true, op,
                      *taskId);
        Value fullBars = getWarpSpecializeCaptureForUse(op, state.fullBars);
        Value fullView =
            getBarrierView(builder, loc, state, fullBars, close.getStage());
        auto arrive =
            ArriveBarrierOp::create(builder, loc, fullView, *threadCount);
        setRoleTaskId(op, fullView.getDefiningOp(), *taskId);
        setRoleTaskId(op, arrive.getOperation(), *taskId);
        close.erase();
        continue;
      }

      if (auto waitOp = dyn_cast<PipeReaderWaitOp>(op)) {
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*reader=*/1));
        auto threadCount = getTaskThreadCount(op);
        StringRef readerName;
        if (auto attr = waitOp->getAttrOfType<StringAttr>("reader_name"))
          readerName = attr.getValue();
        if (failed(taskId) || failed(threadCount) ||
            failed(recordReaderTask(state, op, readerName, *taskId,
                                    *threadCount))) {
          signalPassFailure();
          return;
        }
        Value fullBars = getWarpSpecializeCaptureForUse(op, state.fullBars);
        Value view =
            getBarrierView(builder, loc, state, fullBars, waitOp.getStage());
        Value phase = phaseAsI32(builder, loc, waitOp.getPhase());
        auto wait = WaitBarrierOp::create(builder, loc, view, phase);
        setRoleTaskId(op, view.getDefiningOp(), *taskId);
        setRoleTaskId(op, wait.getOperation(), *taskId);

        Value isClosed;
        if (state.oneShot) {
          isClosed = arith::ConstantIntOp::create(builder, loc, 0, 1);
          setRoleTaskId(op, isClosed.getDefiningOp(), *taskId);
        } else {
          isClosed =
              loadCloseTag(builder, loc, state, waitOp.getStage(), op, *taskId);
        }
        waitOp.getIsClosed().replaceAllUsesWith(isClosed);
        waitOp.erase();
        continue;
      }

      if (auto drain = dyn_cast<PipeDrainOp>(op)) {
        auto taskId = getSingleTaskId(
            op, getEnclosingDefaultTaskId(op, /*nonWarpSpecializeDefault=*/0));
        auto threadCount = getTaskThreadCount(op);
        if (failed(taskId) || failed(threadCount)) {
          signalPassFailure();
          return;
        }
        Value barrier =
            getWarpSpecializeCaptureForUse(op, state.drainBarrier);
        auto arrive =
            ArriveBarrierOp::create(builder, loc, barrier, *threadCount);
        Value phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
        auto wait = WaitBarrierOp::create(builder, loc, barrier, phase);
        setRoleTaskId(op, arrive.getOperation(), *taskId);
        setRoleTaskId(op, phase.getDefiningOp(), *taskId);
        setRoleTaskId(op, wait.getOperation(), *taskId);
        drain.erase();
        continue;
      }

      if (auto release = dyn_cast<PipeReaderReleaseOp>(op)) {
        if (state.oneShot) {
          release.erase();
          continue;
        }
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*reader=*/1));
        auto threadCount = getTaskThreadCount(op);
        StringRef readerName;
        if (auto attr = release->getAttrOfType<StringAttr>("reader_name"))
          readerName = attr.getValue();
        if (failed(taskId) || failed(threadCount) ||
            failed(recordReaderTask(state, op, readerName, *taskId,
                                    *threadCount))) {
          signalPassFailure();
          return;
        }
        Value emptyBars = getWarpSpecializeCaptureForUse(op, state.emptyBars);
        Value view =
            getBarrierView(builder, loc, state, emptyBars, release.getStage());
        auto arrive = ArriveBarrierOp::create(builder, loc, view, *threadCount);
        setRoleTaskId(op, view.getDefiningOp(), *taskId);
        setRoleTaskId(op, arrive.getOperation(), *taskId);
        release.erase();
        continue;
      }
    }

    // Patch init counts now that writer/reader metadata is known. Re-walk
    // InitBarrierOps under each pipe's barrier arrays.
    for (auto &[key, state] : pipes) {
      (void)key;
      if (!state.writerFullCount && state.writerThreadCount)
        state.writerFullCount = state.writerThreadCount;
      int32_t fullCount = state.writerFullCount.value_or(provisionalCount);
      int32_t emptyCount = 0;
      if (!state.oneShot) {
        if (state.readerTasks.empty())
          emptyCount = state.writerThreadCount.value_or(fullCount);
        else {
          for (auto &entry : state.readerTasks)
            emptyCount += entry.second.second;
        }
        if (emptyCount <= 0)
          emptyCount = provisionalCount;
      }

      auto patchInits = [&](Value array, int32_t count) {
        if (!array)
          return;
        for (Operation *user : array.getUsers()) {
          if (auto index = dyn_cast<ttg::MemDescIndexOp>(user)) {
            for (Operation *idxUser : index.getResult().getUsers()) {
              if (auto init = dyn_cast<InitBarrierOp>(idxUser))
                init.setCount(count);
            }
          }
        }
      };
      patchInits(state.fullBars, fullCount);
      if (!state.oneShot)
        patchInits(state.emptyBars, emptyCount);
    }

    // Pipe handles can remain captured by warp-specialize regions while the
    // lifecycle operations are lowered. Erasing a live create invalidates
    // those capture operands and corrupts subsequent capture insertion.
    for (PipeCreateOp create : creates) {
      OpBuilder builder(create);
      Value loweredHandle =
          arith::ConstantIntOp::create(builder, create.getLoc(), 0, 32);
      create.getResult().replaceAllUsesWith(loweredHandle);
      create.erase();
    }

    // Ensure no pipe lifecycle ops remain.
    bool leftover = false;
    module.walk([&](Operation *op) {
      if (isPipeLifecycleOp(op)) {
        op->emitOpError("failed to lower pipe lifecycle op");
        leftover = true;
      }
    });
    if (leftover)
      signalPassFailure();
  }
};

} // namespace mlir::triton::iluvatar_tle
