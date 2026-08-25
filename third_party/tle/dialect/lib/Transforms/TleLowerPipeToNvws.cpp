#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "tle/dialect/include/Analysis/TlePipeEffectAnalysis.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <map>
#include <optional>

namespace mlir::triton::tle {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttnvws = mlir::triton::nvws;

#define GEN_PASS_DEF_TRITONTLELOWERPIPETONVWS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral
    kTleInferArriveCountAttr("tle.infer_arrive_count");
constexpr llvm::StringLiteral
    kTleInferFullCountOffsetAttr("tle.infer_full_count_offset");
constexpr llvm::StringLiteral
    kTleParticipantConsumerReleaseAttr("tle.participant_consumer_release");

enum class PipeCommitTransport {
  LocalStore,
  CpAsync,
  TmaCopy,
  MixedTmaLocalStore,
  MixedTmaCpAsync,
};

struct PipeState {
  Value token;
  Value closeTags;
  ttg::MemDescType closeTagSlotType;
  SmallVector<std::string> readerNames;
  bool oneShot;
  std::optional<int32_t> writerTaskId;
  std::optional<int32_t> writerThreadCount;
  std::optional<int32_t> writerFullCount;
  std::map<std::string, std::pair<int32_t, int32_t>> readerTasks;
  std::optional<PipeCommitTransport> dataTransport;
  Value drainBarrier;
};

struct PipeDefinition {
  PipeCreateOp create;
  bool oneShot = false;
};

enum class PipeProvenanceKind {
  Unknown,
  PipeFieldWhole,
  PipeFieldExactRange,
  PipeSlotWhole,
  Escaped,
};

struct ByteRange {
  int64_t offset = 0;
  int64_t size = 0;
  bool exact = false;
};

struct PipeFieldAccess {
  PipeProvenanceKind kind = PipeProvenanceKind::Unknown;
  Value pipeRoot;
  Value slot;
  unsigned fieldIndex = 0;
  ByteRange range;
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

static Value getPipeIdentity(Operation *op) {
  if (auto pipeOp = dyn_cast<PipeCreateOp>(op))
    return pipeOp.getIdentity();
  if (auto pipeOp = dyn_cast<PipeWriterAcquireOp>(op))
    return pipeOp.getIdentity();
  if (auto pipeOp = dyn_cast<PipeWriterCommitOp>(op))
    return pipeOp.getIdentity();
  if (auto pipeOp = dyn_cast<PipeWriterCloseOp>(op))
    return pipeOp.getIdentity();
  if (auto pipeOp = dyn_cast<PipeReaderWaitOp>(op))
    return pipeOp.getIdentity();
  if (auto pipeOp = dyn_cast<PipeReaderReleaseOp>(op))
    return pipeOp.getIdentity();
  return cast<PipeDrainOp>(op).getIdentity();
}

static bool isPipeLifecycleOp(Operation *op) {
  return isa<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
             PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp,
             PipeDrainOp>(op);
}

static bool containsPipeLifecycleOp(tt::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isPipeLifecycleOp(op))
      found = true;
  });
  return found;
}

static LogicalResult inlinePipeCall(tt::CallOp call, tt::FuncOp callee,
                                    int64_t callId) {
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

  OpBuilder builder(call);
  SmallVector<Type> argumentTypes = llvm::to_vector(
      llvm::map_range(call.getOperands(), [](Value value) {
        return value.getType();
      }));
  auto begin = PipeCallBeginOp::create(
      builder, call.getLoc(), argumentTypes, call.getOperands(),
      callee.getName(), callId);
  IRMapping mapping;
  for (auto [arg, alias] :
       llvm::zip(block.getArguments(), begin.getAliases()))
    mapping.map(arg, alias);
  for (Operation &op : block.getOperations()) {
    if (&op == returnOp.getOperation())
      continue;
    builder.clone(op, mapping);
  }
  PipeCallEndOp::create(builder, call.getLoc(), callee.getName(), callId);

  for (auto [result, returned] :
       llvm::zip(call.getResults(), returnOp.getOperands()))
    result.replaceAllUsesWith(mapping.lookupOrDefault(returned));
  call.erase();
  return success();
}

static LogicalResult inlinePipeHelperCalls(ModuleOp module) {
  bool changed = true;
  int64_t nextCallId = 0;
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
      if (failed(inlinePipeCall(call, callee, nextCallId++)))
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

static Value canonicalizePipeField(Value field) {
  while (true) {
    if (auto result = dyn_cast<OpResult>(field)) {
      if (auto begin = dyn_cast<PipeCallBeginOp>(result.getOwner())) {
        field = begin.getArguments()[result.getResultNumber()];
        continue;
      }
    }
    if (auto blockArg = dyn_cast<BlockArgument>(field)) {
      Block *block = blockArg.getOwner();
      auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
          block->getParentOp());
      if (!partitions)
        break;
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

static bool isNvwsTokenValue(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType && isa<ttnvws::TokenType>(tensorType.getElementType());
}

static Value getMemDescRoot(Value value) {
  Value current = canonicalizePipeField(value);
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = canonicalizePipeField(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = canonicalizePipeField(subslice.getSrc());
      continue;
    }
    if (auto alias = current.getDefiningOp<MemDescAliasOp>()) {
      current = canonicalizePipeField(alias.getSrc());
      continue;
    }
    break;
  }
  return current;
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
  os << canonicalizePipeField(getPipeIdentity(op)).getAsOpaquePointer();
  os << "|";
  // pipe_name is diagnostic only. Structural identity is sufficient after
  // endpoint arguments substitute the concrete payload memdescs at a call.
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

static void setTokenCount(Value token, StringRef attrName, int32_t count) {
  auto createToken = cast<ttnvws::CreateTokenOp>(token.getDefiningOp());
  createToken->setAttr(
      attrName,
      IntegerAttr::get(IntegerType::get(createToken.getContext(), 32), count));
}

static LogicalResult recordWriterTask(PipeState &state, Operation *op,
                                      int32_t taskId, int32_t threadCount) {
  if (state.writerTaskId && *state.writerTaskId != taskId)
    return op->emitOpError("uses writer async_task_id ")
           << taskId << " but pipe already has writer async_task_id "
           << *state.writerTaskId;
  if (state.writerThreadCount && *state.writerThreadCount != threadCount)
    return op->emitOpError("uses writer thread count ")
           << threadCount << " but pipe already has writer thread count "
           << *state.writerThreadCount;
  state.writerTaskId = taskId;
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
  setTokenCount(state.token, "full_count", count);
  return success();
}

static bool hasDeclaredReader(const PipeState &state, StringRef readerName) {
  return llvm::any_of(state.readerNames, [&](const std::string &declared) {
    return StringRef(declared) == readerName;
  });
}

static FailureOr<std::string> getPipeReaderName(PipeState &state,
                                                Operation *op) {
  std::string readerName;
  if (auto attr = op->getAttrOfType<StringAttr>("reader_name"))
    readerName = attr.getValue().str();

  if (state.readerNames.empty()) {
    if (!readerName.empty()) {
      op->emitOpError("uses named reader ")
          << readerName << " but pipe was created without readers";
      return failure();
    }
    return readerName;
  }

  if (readerName.empty()) {
    op->emitOpError("requires reader_name because pipe was created "
                    "with explicit readers");
    return failure();
  }
  if (!hasDeclaredReader(state, readerName)) {
    op->emitOpError("uses undeclared pipe reader ") << readerName;
    return failure();
  }
  return readerName;
}

static void updateTokenEmptyCount(PipeState &state) {
  int32_t emptyCount = 0;
  for (const auto &reader : state.readerTasks)
    emptyCount += reader.second.second;
  if (emptyCount > 0)
    setTokenCount(state.token, "empty_count", emptyCount);
}

static LogicalResult recordReaderTask(PipeState &state, Operation *op,
                                      StringRef readerName, int32_t taskId,
                                      int32_t threadCount,
                                      bool updateEmptyCountForReader = true) {
  auto it = state.readerTasks.find(readerName.str());
  if (it != state.readerTasks.end()) {
    if (it->second.first != taskId)
      return op->emitOpError("uses reader ")
             << readerName << " async_task_id " << taskId
             << " but that reader already has async_task_id "
             << it->second.first;
    if (it->second.second != threadCount)
      return op->emitOpError("uses reader ")
             << readerName << " thread count " << threadCount
             << " but that reader already has thread count "
             << it->second.second;
  }
  state.readerTasks[readerName.str()] = {taskId, threadCount};
  if (updateEmptyCountForReader)
    updateTokenEmptyCount(state);
  return success();
}

static SmallVector<std::string> getDeclaredReaderNames(PipeCreateOp op) {
  SmallVector<std::string> readerNames;
  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    readerNames.reserve(readersAttr.size());
    for (Attribute attr : readersAttr)
      readerNames.push_back(cast<StringAttr>(attr).getValue().str());
  }
  return readerNames;
}

static bool hasDeclaredReader(ArrayRef<std::string> readerNames,
                              StringRef readerName) {
  return llvm::any_of(readerNames, [&](const std::string &declared) {
    return StringRef(declared) == readerName;
  });
}

static FailureOr<std::string>
getPipeReaderNameForDefinition(ArrayRef<std::string> readerNames,
                               Operation *op) {
  std::string readerName;
  if (auto attr = op->getAttrOfType<StringAttr>("reader_name"))
    readerName = attr.getValue().str();

  if (readerNames.empty()) {
    if (!readerName.empty()) {
      op->emitOpError("uses named reader ")
          << readerName << " but pipe was created without readers";
      return failure();
    }
    return readerName;
  }

  if (readerName.empty()) {
    op->emitOpError("requires reader_name because pipe was created "
                    "with explicit readers");
    return failure();
  }
  if (!hasDeclaredReader(readerNames, readerName)) {
    op->emitOpError("uses undeclared pipe reader ") << readerName;
    return failure();
  }
  return readerName;
}

static FailureOr<bool> hasMatchingParticipantRelease(
    PipeReaderWaitOp wait, const PipeDefinition &definition) {
  if (definition.oneShot || wait.getFields().empty())
    return false;

  FailureOr<std::string> readerName = getPipeReaderNameForDefinition(
      getDeclaredReaderNames(definition.create), wait.getOperation());
  if (failed(readerName))
    return failure();
  auto waitTaskId = getSingleTaskId(
      wait.getOperation(),
      getEnclosingDefaultTaskId(wait.getOperation(), /*reader=*/1));
  auto waitThreadCount = getTaskThreadCount(wait.getOperation());
  if (failed(waitTaskId) || failed(waitThreadCount))
    return failure();

  std::string pipeKey = getPipeKey(wait.getOperation());
  for (Operation *next = wait->getNextNode(); next;
       next = next->getNextNode()) {
    if (!isa<PipeReaderWaitOp, PipeReaderReleaseOp>(next) ||
        getPipeKey(next) != pipeKey)
      continue;

    FailureOr<std::string> nextReaderName = getPipeReaderNameForDefinition(
        getDeclaredReaderNames(definition.create), next);
    if (failed(nextReaderName))
      return failure();
    if (*nextReaderName != *readerName)
      continue;

    // A second wait before releasing the current stage does not establish a
    // participant-counted cyclic protocol for either wait.
    if (isa<PipeReaderWaitOp>(next))
      return false;

    auto release = cast<PipeReaderReleaseOp>(next);
    if (release.getFields().empty() ||
        !sameIndexValue(wait.getStage(), release.getStage()))
      return false;

    auto releaseTaskId = getSingleTaskId(
        release.getOperation(),
        getEnclosingDefaultTaskId(release.getOperation(), /*reader=*/1));
    auto releaseThreadCount = getTaskThreadCount(release.getOperation());
    if (failed(releaseTaskId) || failed(releaseThreadCount))
      return failure();
    return *releaseTaskId == *waitTaskId &&
           *releaseThreadCount == *waitThreadCount;
  }

  return false;
}

static LogicalResult analyzeParticipantCountedWaits(
    ArrayRef<Operation *> ops,
    const std::map<std::string, PipeDefinition> &pipes,
    DenseSet<Operation *> &participantCountedWaits) {
  for (Operation *op : ops) {
    auto wait = dyn_cast<PipeReaderWaitOp>(op);
    if (!wait)
      continue;
    auto definition = pipes.find(getPipeKey(op));
    if (definition == pipes.end())
      return wait.emitOpError("requires a preceding matching pipe.create");
    FailureOr<bool> matched =
        hasMatchingParticipantRelease(wait, definition->second);
    if (failed(matched))
      return failure();
    if (*matched)
      participantCountedWaits.insert(op);
  }
  return success();
}

static StringRef getTransportName(PipeCommitTransport transport) {
  switch (transport) {
  case PipeCommitTransport::LocalStore:
    return "local-store";
  case PipeCommitTransport::CpAsync:
    return "cp.async";
  case PipeCommitTransport::TmaCopy:
    return "TMA copy";
  case PipeCommitTransport::MixedTmaLocalStore:
    return "mixed TMA/local-store";
  case PipeCommitTransport::MixedTmaCpAsync:
    return "mixed TMA/cp.async";
  }
  llvm_unreachable("unknown pipe commit transport");
}

static LogicalResult
recordDataTransport(std::optional<PipeCommitTransport> &dataTransport,
                    Operation *op, PipeCommitTransport transport) {
  if (dataTransport &&
      *dataTransport == PipeCommitTransport::MixedTmaLocalStore &&
      transport == PipeCommitTransport::MixedTmaLocalStore)
    return success();
  if (dataTransport && *dataTransport == PipeCommitTransport::MixedTmaCpAsync &&
      transport == PipeCommitTransport::MixedTmaCpAsync)
    return success();
  if (dataTransport && *dataTransport != transport)
    return op->emitOpError("mixes ")
           << getTransportName(*dataTransport) << " and "
           << getTransportName(transport)
           << " payload commits on the same pipe; pipe full-barrier count is "
              "a per-pipe contract";
  dataTransport = transport;
  return success();
}

static LogicalResult recordDataTransport(PipeState &state, Operation *op,
                                         PipeCommitTransport transport) {
  return recordDataTransport(state.dataTransport, op, transport);
}

static std::optional<int32_t> inferPrefixParticipants(Type valueType,
                                                      int32_t taskThreadCount) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueType);
  if (!tensorTy || !tensorTy.hasStaticShape() || !tensorTy.getEncoding())
    return std::nullopt;

  int64_t numElements = tensorTy.getNumElements();
  if (numElements <= 0)
    return std::nullopt;
  unsigned elemsPerThread = ttg::getTotalElemsPerThread(tensorTy);
  if (elemsPerThread == 0)
    return std::nullopt;

  int64_t participants = (numElements + elemsPerThread - 1) / elemsPerThread;
  if (participants <= 0)
    return std::nullopt;
  return static_cast<int32_t>(std::min<int64_t>(participants, taskThreadCount));
}

static std::optional<Value>
getCommitFieldRootForStore(Value memdesc, PipeWriterCommitOp commit) {
  Value current = canonicalizePipeField(memdesc);
  bool sawStageIndex = false;
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      if (!sameIndexValue(index.getIndex(), commit.getStage()))
        return std::nullopt;
      sawStageIndex = true;
      current = canonicalizePipeField(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = canonicalizePipeField(subslice.getSrc());
      continue;
    }
    if (auto alias = current.getDefiningOp<MemDescAliasOp>()) {
      current = canonicalizePipeField(alias.getSrc());
      continue;
    }
    break;
  }

  if (!sawStageIndex && getPipeCapacity(commit.getOperation()) != 1)
    return std::nullopt;
  return current;
}

static bool canInterleaveBeforeLocalStorePipeCommit(Operation *op) {
  return canInterleaveBeforePipeMetadataOp(op);
}

static std::optional<int32_t> inferLocalStoreParticipantCountForRoots(
    PipeWriterCommitOp commit, ArrayRef<Value> roots, int32_t taskThreadCount,
    Operation *windowBegin,
    const llvm::DenseSet<Value> *allowedInterleavedTmaRoots = nullptr) {
  llvm::DenseSet<Value> fieldRoots;
  for (Value rootValue : roots) {
    Value root = getMemDescRoot(rootValue);
    // If multiple logical fields share one root allocation, root-only alias
    // reasoning cannot distinguish which field was written. Keep the full
    // partition contract rather than publish a partially observed payload.
    if (!fieldRoots.insert(root).second)
      return std::nullopt;
  }

  llvm::DenseSet<Value> storedRoots;
  CompletedAsyncCopyState completedAsyncCopies;
  std::optional<int32_t> participants;
  bool sawLocalStore = false;
  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (prev == windowBegin)
      break;
    if (isa<ttnvws::ProducerAcquireOp>(prev))
      return std::nullopt;
    if (isPipeLifecycleOp(prev))
      return std::nullopt;
    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(prev)) {
      if (!allowedInterleavedTmaRoots)
        return std::nullopt;
      Value dstRoot = getMemDescRoot(tmaCopy.getDst());
      if (!allowedInterleavedTmaRoots->contains(dstRoot))
        return std::nullopt;
      continue;
    }
    if (auto wait = dyn_cast<ttg::AsyncWaitOp>(prev)) {
      recordCompletedAsyncWait(wait, completedAsyncCopies);
      continue;
    }
    if (auto asyncCommit = dyn_cast<ttg::AsyncCommitGroupOp>(prev)) {
      propagateCompletedAsyncCommitGroup(asyncCommit, completedAsyncCopies);
      continue;
    }
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(prev)) {
      std::optional<LocalStoreTarget> target = getAsyncCopyTarget(prev);
      assert(target && "async copy must have a local destination");
      std::optional<Value> root =
          getCommitFieldRootForStore(target->memdesc, commit);
      if (!root)
        return std::nullopt;
      if (fieldRoots.contains(*root)) {
        if (!isAsyncCopyComplete(asyncCopy, completedAsyncCopies))
          return std::nullopt;
        std::optional<int32_t> count =
            inferPrefixParticipants(target->valueType, taskThreadCount);
        if (!count)
          return std::nullopt;
        participants = participants ? std::max(*participants, *count) : *count;
        storedRoots.insert(*root);
        sawLocalStore = true;
      }
      continue;
    }

    std::optional<LocalStoreTarget> target = getLocalStoreTarget(prev);
    if (target) {
      std::optional<Value> root =
          getCommitFieldRootForStore(target->memdesc, commit);
      if (!root)
        return std::nullopt;
      if (fieldRoots.contains(*root)) {
        std::optional<int32_t> count =
            inferPrefixParticipants(target->valueType, taskThreadCount);
        if (!count)
          return std::nullopt;
        participants = participants ? std::max(*participants, *count) : *count;
        storedRoots.insert(*root);
        sawLocalStore = true;
      }
      continue;
    }

    if (auto store = dyn_cast<tt::StoreOp>(prev)) {
      if (isSharedPointer(store.getPtr()))
        return std::nullopt;
    }

    if (!canInterleaveBeforeLocalStorePipeCommit(prev))
      return std::nullopt;
  }

  if (!sawLocalStore || !participants)
    return std::nullopt;
  for (Value root : fieldRoots) {
    if (!storedRoots.contains(root))
      return std::nullopt;
  }
  return participants;
}

static std::optional<int32_t>
inferLocalStoreParticipantCount(PipeWriterCommitOp commit,
                                int32_t taskThreadCount,
                                Operation *windowBegin) {
  SmallVector<Value> roots(commit.getFields().begin(),
                           commit.getFields().end());
  return inferLocalStoreParticipantCountForRoots(commit, roots, taskThreadCount,
                                                 windowBegin);
}

// Conservative root-level implementation: accept only whole-field or
// root-disjoint mixed payloads. Exact byte-range/source-order repair is not
// implemented here and must fail fast.
static bool verifyRootLevelLocalStoreCoverage(
    PipeWriterCommitOp commit, ArrayRef<Value> roots, Operation *windowBegin,
    const llvm::DenseSet<Value> *allowedInterleavedTmaRoots = nullptr,
    std::string *failureReason = nullptr) {
  auto fail = [&](Twine reason) {
    if (failureReason)
      *failureReason = reason.str();
    return false;
  };

  llvm::DenseSet<Value> fieldRoots;
  for (Value rootValue : roots) {
    Value root = getMemDescRoot(rootValue);
    if (!fieldRoots.insert(root).second)
      return fail("multiple non-TMA fields share one root allocation");
  }

  llvm::DenseSet<Value> storedRoots;
  CompletedAsyncCopyState completedAsyncCopies;
  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (prev == windowBegin)
      break;
    if (isa<ttnvws::ProducerAcquireOp>(prev))
      return fail("encountered an unrelated NVWS producer acquire");
    if (isPipeLifecycleOp(prev))
      return fail(Twine("encountered pipe lifecycle op ") +
                  prev->getName().getStringRef());
    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(prev)) {
      if (!allowedInterleavedTmaRoots)
        return fail("encountered TMA copy in a non-mixed local-store commit");
      Value dstRoot = getMemDescRoot(tmaCopy.getDst());
      if (!allowedInterleavedTmaRoots->contains(dstRoot))
        return fail("encountered TMA copy for a field outside this commit");
      continue;
    }
    if (auto wait = dyn_cast<ttg::AsyncWaitOp>(prev)) {
      recordCompletedAsyncWait(wait, completedAsyncCopies);
      continue;
    }
    if (auto asyncCommit = dyn_cast<ttg::AsyncCommitGroupOp>(prev)) {
      propagateCompletedAsyncCommitGroup(asyncCommit, completedAsyncCopies);
      continue;
    }
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(prev)) {
      std::optional<LocalStoreTarget> target = getAsyncCopyTarget(prev);
      assert(target && "async copy must have a local destination");
      std::optional<Value> root =
          getCommitFieldRootForStore(target->memdesc, commit);
      if (!root)
        return fail("encountered an async copy that is not staged for this "
                    "commit slot");
      if (fieldRoots.contains(*root)) {
        if (!isAsyncCopyComplete(asyncCopy, completedAsyncCopies))
          return fail("encountered an async copy for a non-TMA field without "
                      "a proven async_wait before the pipe commit");
        storedRoots.insert(*root);
      }
      continue;
    }

    std::optional<LocalStoreTarget> target = getLocalStoreTarget(prev);
    if (target) {
      std::optional<Value> root =
          getCommitFieldRootForStore(target->memdesc, commit);
      if (!root)
        return fail("encountered a local store that is not staged for this "
                    "commit slot");
      if (fieldRoots.contains(*root))
        storedRoots.insert(*root);
      continue;
    }

    if (auto store = dyn_cast<tt::StoreOp>(prev)) {
      if (isSharedPointer(store.getPtr()))
        return fail("encountered an opaque shared-memory store");
    }

    if (!canInterleaveBeforeLocalStorePipeCommit(prev))
      return fail(Twine("encountered unsupported interleaved op ") +
                  prev->getName().getStringRef());
  }

  for (Value root : fieldRoots) {
    if (!storedRoots.contains(root))
      return fail("missing a local store for at least one non-TMA field");
  }
  return true;
}

// Conservative root-level implementation: accept only whole-field or
// root-disjoint mixed payloads. Exact byte-range/source-order repair is not
// implemented here and must fail fast.
static bool verifyRootLevelAsyncCopyCoverage(
    PipeWriterCommitOp commit, ArrayRef<Value> roots, Operation *windowBegin,
    const llvm::DenseSet<Value> *allowedInterleavedTmaRoots,
    std::string *failureReason = nullptr) {
  auto fail = [&](Twine reason) {
    if (failureReason)
      *failureReason = reason.str();
    return false;
  };

  llvm::DenseSet<Value> fieldRoots;
  for (Value rootValue : roots) {
    Value root = getMemDescRoot(rootValue);
    if (!fieldRoots.insert(root).second)
      return fail("multiple non-TMA fields share one root allocation");
  }

  llvm::DenseSet<Value> copiedRoots;
  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (prev == windowBegin)
      break;
    if (isa<ttnvws::ProducerAcquireOp>(prev))
      return fail("encountered an unrelated NVWS producer acquire");
    if (isPipeLifecycleOp(prev))
      return fail(Twine("encountered pipe lifecycle op ") +
                  prev->getName().getStringRef());
    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(prev)) {
      if (!allowedInterleavedTmaRoots)
        return fail("encountered TMA copy in a non-mixed cp.async commit");
      Value dstRoot = getMemDescRoot(tmaCopy.getDst());
      if (!allowedInterleavedTmaRoots->contains(dstRoot))
        return fail("encountered TMA copy for a field outside this commit");
      continue;
    }
    if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(prev))
      continue;
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(prev)) {
      std::optional<LocalStoreTarget> target = getAsyncCopyTarget(prev);
      assert(target && "async copy must have a local destination");
      std::optional<Value> root =
          getCommitFieldRootForStore(target->memdesc, commit);
      if (!root)
        return fail("encountered an async copy that is not staged for this "
                    "commit slot");
      if (fieldRoots.contains(*root))
        copiedRoots.insert(*root);
      continue;
    }

    if (auto target = getLocalStoreTarget(prev)) {
      std::optional<Value> root =
          getCommitFieldRootForStore(target->memdesc, commit);
      if (!root)
        return fail("encountered a local store that is not staged for this "
                    "commit slot");
      if (fieldRoots.contains(*root))
        return fail("encountered a non-TMA field local store that is not "
                    "covered by cp.async mbarrier tracking");
      continue;
    }

    if (auto store = dyn_cast<tt::StoreOp>(prev)) {
      if (isSharedPointer(store.getPtr()))
        return fail("encountered an opaque shared-memory store");
    }

    if (!canInterleaveBeforeLocalStorePipeCommit(prev))
      return fail(Twine("encountered unsupported interleaved op ") +
                  prev->getName().getStringRef());
  }

  for (Value root : fieldRoots) {
    if (!copiedRoots.contains(root))
      return fail("missing a cp.async copy for at least one non-TMA field");
  }
  return true;
}

static LogicalResult verifyTmaCopyTypes(ttg::TMACopyOp op) {
  auto descTy = dyn_cast<tt::TensorDescType>(op.getSrc().getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(op.getDst().getType());
  if (!descTy || !memDescTy)
    return op.emitOpError("used by a pipe TMA commit must be a "
                          "tensor-descriptor to memdesc copy");

  RankedTensorType blockTy = descTy.getSignlessBlockType();
  ArrayRef<int64_t> blockShape = blockTy.getShape();
  ArrayRef<int64_t> memShape = memDescTy.getShape();

  if (blockShape.size() > memShape.size()) {
    unsigned rankDiff = blockShape.size() - memShape.size();
    for (unsigned i = 0; i < rankDiff; ++i) {
      if (blockShape[i] != 1) {
        return op.emitOpError("used by a pipe TMA commit requires tensor "
                              "descriptor block shape ")
               << blockShape << " to match memdesc shape " << memShape
               << " except for unit leading dimensions";
      }
    }
    blockShape = blockShape.take_back(memShape.size());
  }

  if (blockShape.size() != memShape.size())
    return op.emitOpError("used by a pipe TMA commit requires tensor "
                          "descriptor rank ")
           << blockShape.size() << " to match memdesc rank " << memShape.size();

  if (blockShape != memShape)
    return op.emitOpError("used by a pipe TMA commit requires tensor "
                          "descriptor block shape ")
           << blockShape << " to match memdesc shape " << memShape;

  if (blockTy.getElementType() != memDescTy.getElementType())
    return op.emitOpError("used by a pipe TMA commit requires tensor "
                          "descriptor element type ")
           << blockTy.getElementType() << " to match memdesc element type "
           << memDescTy.getElementType();

  if (op.getIndices().size() != descTy.getBlockType().getRank())
    return op.emitOpError("used by a pipe TMA commit requires ")
           << descTy.getBlockType().getRank() << " TMA coordinates, but got "
           << op.getIndices().size();

  return success();
}

static bool canInterleaveBeforeTmaPipeCommit(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isCtaInvariantSpecialRegisterRead(op))
    return true;
  if (isMemoryEffectFree(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile() && isNonSharedPointer(load.getPtr());
  if (auto store = dyn_cast<tt::StoreOp>(op))
    return isNonSharedPointer(store.getPtr());
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
    return true;
  return false;
}

struct TmaPipeCommitInfo {
  bool sawPipeTmaCopy = false;
  llvm::DenseSet<Value> copiedRoots;
};

struct PipeCommitAnalysis {
  TmaPipeCommitInfo tmaInfo;
  SmallVector<Value> uniqueFieldRoots;
  SmallVector<Value> localStoreRoots;
  PipeCommitTransport transport = PipeCommitTransport::LocalStore;
  std::optional<int32_t> participantCount;
};

static FailureOr<TmaPipeCommitInfo>
getRootLevelTmaPipeCommitInfo(PipeWriterCommitOp commit,
                              Operation *windowBegin) {
  llvm::DenseSet<Value> fieldRoots;
  for (Value field : commit.getFields())
    fieldRoots.insert(getMemDescRoot(field));

  TmaPipeCommitInfo info;
  llvm::DenseSet<Value> interleavedLocalRoots;
  for (Operation *op = windowBegin->getNextNode(); op && op != commit;
       op = op->getNextNode()) {
    bool sawPayload = info.sawPipeTmaCopy || !interleavedLocalRoots.empty();
    if (isa<ttnvws::ProducerAcquireOp>(op)) {
      if (info.sawPipeTmaCopy)
        return commit.emitOpError("has an unrelated NVWS producer acquire "
                                  "between pipe payload TMA copies and "
                                  "commit");
      return info;
    }
    if (auto acquire = dyn_cast<PipeWriterAcquireOp>(op)) {
      if (sawPayload)
        return commit.emitOpError("has an unrelated pipe writer acquire "
                                  "between pipe payload TMA copies and "
                                  "commit");
      continue;
    }
    if (auto create = dyn_cast<PipeCreateOp>(op)) {
      if (sawPayload)
        return commit.emitOpError("has an unrelated pipe create between pipe "
                                  "payload TMA copies and commit");
      continue;
    }
    if (isPipeLifecycleOp(op)) {
      if (sawPayload)
        return commit.emitOpError("has pipe lifecycle op ")
               << op->getName() << " between pipe payload TMA copies and "
               << "commit";
      continue;
    }

    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(op)) {
      if (failed(verifyTmaCopyTypes(tmaCopy)))
        return failure();

      Value dstRoot = getMemDescRoot(tmaCopy.getDst());
      if (!fieldRoots.contains(dstRoot)) {
        if (info.sawPipeTmaCopy)
          return commit.emitOpError("has an unrelated ttg.tma_copy between "
                                    "pipe payload TMA copies and commit");
        return info;
      }
      if (interleavedLocalRoots.contains(dstRoot))
        return commit.emitOpError("has a ttg.tma_copy and a local-store "
                                  "payload targeting the same memdesc root");
      info.copiedRoots.insert(dstRoot);
      info.sawPipeTmaCopy = true;
      continue;
    }

    auto recordLocalRoot = [&](Value memdesc) -> LogicalResult {
      std::optional<Value> root = getCommitFieldRootForStore(memdesc, commit);
      if (!root) {
        if (info.sawPipeTmaCopy)
          return commit.emitOpError("has an unrelated local-store payload "
                                    "between pipe payload TMA copies and "
                                    "commit");
        return failure();
      }
      if (info.copiedRoots.contains(*root))
        return commit.emitOpError("has a ttg.tma_copy and a local-store "
                                  "payload targeting the same memdesc root");
      interleavedLocalRoots.insert(*root);
      return success();
    };

    if (auto target = getAsyncCopyTarget(op)) {
      if (failed(recordLocalRoot(target->memdesc))) {
        if (info.sawPipeTmaCopy)
          return failure();
        continue;
      }
      continue;
    }

    if (auto target = getLocalStoreTarget(op)) {
      if (failed(recordLocalRoot(target->memdesc))) {
        if (info.sawPipeTmaCopy)
          return failure();
        continue;
      }
      continue;
    }

    if (auto store = dyn_cast<tt::StoreOp>(op)) {
      if (isSharedPointer(store.getPtr())) {
        if (sawPayload)
          return commit.emitOpError("has an opaque shared-memory store "
                                    "between pipe payload TMA copies and "
                                    "commit");
        continue;
      }
    }

    if (canInterleaveBeforeTmaPipeCommit(op))
      continue;
    if (sawPayload)
      return commit.emitOpError("has unsupported interleaved op ")
             << op->getName() << " between pipe payload TMA copies and "
             << "commit";
  }

  return info;
}

static bool isOneShotPipe(PipeCreateOp op) {
  if (auto oneShotAttr = op->getAttrOfType<BoolAttr>("one_shot"))
    return oneShotAttr.getValue();
  return false;
}

static FailureOr<Operation *>
getPipePayloadWindowBegin(PipeWriterCommitOp commit,
                          PipeDefinition &definition) {
  std::string commitKey = getPipeKey(commit.getOperation());
  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    auto acquire = dyn_cast<PipeWriterAcquireOp>(prev);
    if (!acquire || getPipeKey(acquire.getOperation()) != commitKey)
      continue;
    if (sameIndexValue(acquire.getStage(), commit.getStage()))
      return acquire.getOperation();
  }

  if (definition.create->getBlock() != commit->getBlock()) {
    commit.emitOpError("without a matching writer acquire must be in the same "
                       "block as its pipe.create so the payload window is "
                       "explicit");
    return failure();
  }
  return definition.create.getOperation();
}

static FailureOr<PipeCommitAnalysis>
analyzePipeCommit(PipeWriterCommitOp commit, PipeDefinition &definition) {
  // A fieldless one-shot pipe is a pure execution-order edge. There is no
  // payload window or transport to infer; every writer task participant
  // arrives at commit and the matching reader waits on the control token.
  if (commit.getFields().empty()) {
    if (!definition.oneShot)
      return commit.emitOpError(
          "without payload fields requires a one_shot pipe");
    return PipeCommitAnalysis{};
  }

  FailureOr<Operation *> windowBegin =
      getPipePayloadWindowBegin(commit, definition);
  if (failed(windowBegin))
    return failure();

  auto threadCount = getTaskThreadCount(commit.getOperation());
  if (failed(threadCount))
    return failure();

  FailureOr<TmaPipeCommitInfo> tmaInfo =
      getRootLevelTmaPipeCommitInfo(commit, *windowBegin);
  if (failed(tmaInfo))
    return failure();

  PipeCommitAnalysis analysis;
  analysis.tmaInfo = std::move(*tmaInfo);

  llvm::DenseSet<Value> seenFieldRoots;
  for (Value field : commit.getFields()) {
    Value root = getMemDescRoot(field);
    if (seenFieldRoots.insert(root).second)
      analysis.uniqueFieldRoots.push_back(root);
  }

  for (Value root : analysis.uniqueFieldRoots) {
    if (!analysis.tmaInfo.copiedRoots.contains(root))
      analysis.localStoreRoots.push_back(root);
  }

  bool hasTmaPayload = analysis.tmaInfo.sawPipeTmaCopy;
  bool hasLocalPayload = !analysis.localStoreRoots.empty();
  bool hasCpAsyncPayload = commit->hasAttr(kTlePipeCommitCpAsyncAttr);

  if (hasCpAsyncPayload)
    analysis.transport = PipeCommitTransport::CpAsync;
  if (hasTmaPayload && !hasLocalPayload)
    analysis.transport = PipeCommitTransport::TmaCopy;
  if (hasTmaPayload && hasLocalPayload && !hasCpAsyncPayload)
    analysis.transport = PipeCommitTransport::MixedTmaLocalStore;
  if (hasTmaPayload && hasLocalPayload && hasCpAsyncPayload)
    analysis.transport = PipeCommitTransport::MixedTmaCpAsync;
  if (hasTmaPayload && !hasLocalPayload && hasCpAsyncPayload) {
    commit.emitOpError("marks cp.async payload but all commit fields are "
                       "already covered by TMA");
    return failure();
  }

  if (analysis.transport == PipeCommitTransport::LocalStore)
    analysis.participantCount =
        inferLocalStoreParticipantCount(commit, *threadCount, *windowBegin);
  if (analysis.transport == PipeCommitTransport::MixedTmaLocalStore) {
    analysis.participantCount = inferLocalStoreParticipantCountForRoots(
        commit, analysis.localStoreRoots, *threadCount, *windowBegin,
        &analysis.tmaInfo.copiedRoots);
    std::string coverageFailure;
    if (!analysis.participantCount &&
        !verifyRootLevelLocalStoreCoverage(
            commit, analysis.localStoreRoots, *windowBegin,
            &analysis.tmaInfo.copiedRoots, &coverageFailure)) {
      commit.emitOpError("mixed TMA/local-store pipe commit requires proven "
                         "local-store writes for the non-TMA fields: ")
          << coverageFailure;
      return failure();
    }
  }
  if (analysis.transport == PipeCommitTransport::MixedTmaCpAsync) {
    std::string coverageFailure;
    if (!verifyRootLevelAsyncCopyCoverage(
            commit, analysis.localStoreRoots, *windowBegin,
            &analysis.tmaInfo.copiedRoots, &coverageFailure)) {
      commit.emitOpError("mixed TMA/cp.async pipe commit requires proven "
                         "cp.async copies for the non-TMA fields: ")
          << coverageFailure;
      return failure();
    }
  }

  return analysis;
}

static LogicalResult
analyzePipeCommits(ArrayRef<Operation *> ops,
                   std::map<std::string, PipeDefinition> &pipes,
                   std::map<Operation *, PipeCommitAnalysis> &commitAnalyses) {
  std::map<std::string, std::optional<PipeCommitTransport>> transports;

  for (Operation *op : ops) {
    std::string key = getPipeKey(op);
    if (auto create = dyn_cast<PipeCreateOp>(op)) {
      if (pipes.count(key))
        return create.emitOpError("duplicates an existing pipe.create");
      pipes.emplace(key, PipeDefinition{create, isOneShotPipe(create)});
      continue;
    }

    auto it = pipes.find(key);
    if (it == pipes.end())
      return op->emitOpError("requires a preceding matching pipe.create");

    if (auto commit = dyn_cast<PipeWriterCommitOp>(op)) {
      FailureOr<PipeCommitAnalysis> analysis =
          analyzePipeCommit(commit, it->second);
      if (failed(analysis))
        return failure();
      if (failed(recordDataTransport(transports[key], op, analysis->transport)))
        return failure();
      commitAnalyses.emplace(op, std::move(*analysis));
    }
  }

  return success();
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
    ArrayRef<Operation *> ops, std::map<std::string, PipeDefinition> &pipes,
    std::map<std::string, int32_t> &drainParticipantCounts) {
  std::map<std::string, PipeDrainBuildState> states;

  for (const auto &entry : pipes) {
    PipeDrainBuildState state;
    state.readerNames = getDeclaredReaderNames(entry.second.create);
    state.oneShot = entry.second.oneShot;
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

    if (auto drain = dyn_cast<PipeDrainOp>(op)) {
      if (failed(recordDrainArrival(state, op, *taskId, *threadCount)))
        return failure();
      continue;
    }

    if (failed(verifyNoUseAfterDrain(state, op, *taskId)))
      return failure();

    if (auto close = dyn_cast<PipeWriterCloseOp>(op)) {
      if (state.writerCloseTask &&
          state.writerCloseTask->first != *taskId)
        return close.emitOpError("uses writer async_task_id ")
               << *taskId << " but pipe close already has writer "
               << "async_task_id " << state.writerCloseTask->first;
      state.writerCloseTask = std::make_pair(*taskId, *threadCount);
      if (failed(recordDrainReadyTask(state, op, *taskId, *threadCount)))
        return failure();
      continue;
    }

    if (auto release = dyn_cast<PipeReaderReleaseOp>(op)) {
      auto readerName =
          getPipeReaderNameForDefinition(state.readerNames, release);
      if (failed(readerName))
        return failure();
      auto it = state.readerReleaseTasks.find(*readerName);
      if (it != state.readerReleaseTasks.end() &&
          it->second.first != *taskId)
        return release.emitOpError("uses reader ")
               << *readerName << " async_task_id " << *taskId
               << " but that reader already released from async_task_id "
               << it->second.first;
      state.readerReleaseTasks[*readerName] =
          std::make_pair(*taskId, *threadCount);
      if (failed(recordDrainReadyTask(state, op, *taskId, *threadCount)))
        return failure();
      continue;
    }
  }

  for (auto &[key, state] : states) {
    if (state.drainTasks.empty())
      continue;

    Operation *diagnosticOp = pipes[key].create.getOperation();
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

static void setTokenLoadType(Value token, ttnvws::TokenLoadType loadType) {
  auto createToken = cast<ttnvws::CreateTokenOp>(token.getDefiningOp());
  createToken->setAttr(
      createToken.getLoadTypeAttrName(),
      ttnvws::TokenLoadTypeAttr::get(createToken.getContext(), loadType));
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
                                  RankedTensorType tensorType, bool value);

static Value createDrainBarrier(OpBuilder &builder, Location loc,
                                PipeCreateOp op, int32_t participantCount) {
  MLIRContext *context = op->getContext();
  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  Attribute barrierEncoding = getCloseTagEncoding(context, 1);
  auto barrierType =
      ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrier = ttg::LocalAllocOp::create(builder, loc, barrierType);
  ttng::InitBarrierOp::create(builder, loc, barrier, participantCount);
  mlir::gpu::BarrierOp::create(builder, loc);
  return barrier;
}

static PipeState createPipeState(PipeCreateOp op,
                                 std::optional<int32_t> drainCount) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *context = op->getContext();
  int64_t capacity = getPipeCapacity(op);
  // NVWS control state is represented by Triton tensors, so its physical
  // extent must remain a power of two. Stage arithmetic and payload memdescs
  // continue to use the logical capacity; padded control slots are unreachable.
  int64_t physicalControlCapacity =
      static_cast<int64_t>(llvm::PowerOf2Ceil(capacity));
  bool oneShot = isOneShotPipe(op);

  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  Value closeTags;
  ttg::MemDescType closeTagSlotType;
  if (!oneShot) {
    Attribute closeTagArrayEncoding = getCloseTagEncoding(context, 2);
    Attribute closeTagSlotEncoding = getCloseTagEncoding(context, 1);
    auto closeTagArrayType =
        ttg::MemDescType::get({physicalControlCapacity, 1},
                              builder.getI32Type(),
                              closeTagArrayEncoding, sharedMemorySpace,
                              /*mutableMemory=*/true);
    closeTagSlotType =
        ttg::MemDescType::get({1}, builder.getI32Type(), closeTagSlotEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);

    RankedTensorType closeTagArrayTensorType =
        getCloseTagTensorType(op, builder, {physicalControlCapacity, 1});
    Value initialCloseTags =
        createCloseTagTensor(builder, loc, closeTagArrayTensorType,
                             /*value=*/false);
    closeTags = ttg::LocalAllocOp::create(builder, loc, closeTagArrayType,
                                          initialCloseTags);
  }
  Value token = ttnvws::CreateTokenOp::create(
      builder, loc, static_cast<uint32_t>(physicalControlCapacity),
      ttnvws::TokenLoadType::LocalStoreOp);

  SmallVector<std::string> readerNames = getDeclaredReaderNames(op);
  Value drainBarrier;
  if (drainCount)
    drainBarrier = createDrainBarrier(builder, loc, op, *drainCount);

  PipeState state{token,
                  closeTags,
                  closeTagSlotType,
                  readerNames,
                  oneShot,
                  /*writerTaskId=*/std::nullopt,
                  /*writerThreadCount=*/std::nullopt,
                  /*writerFullCount=*/std::nullopt,
                  /*readerTasks=*/{},
                  /*dataTransport=*/std::nullopt,
                  drainBarrier};
  return state;
}

static Value createCloseTagSlot(OpBuilder &builder, Location loc,
                                const PipeState &state, Value closeTags,
                                Value stage) {
  return ttg::MemDescIndexOp::create(builder, loc, state.closeTagSlotType,
                                     closeTags, stage);
}

static Value createCloseTagTensor(OpBuilder &builder, Location loc,
                                  RankedTensorType tensorType, bool value) {
  Value scalar = arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 32);
  return tt::SplatOp::create(builder, loc, tensorType, scalar);
}

static void storeCloseTag(OpBuilder &builder, Location loc,
                          const PipeState &state, Value stage, bool value,
                          Operation *source, int32_t taskId) {
  Value closeTags = getWarpSpecializeCaptureForUse(source, state.closeTags);
  Value slot = createCloseTagSlot(builder, loc, state, closeTags, stage);
  RankedTensorType tensorType = getCloseTagTensorType(source, builder, {1});
  Value tag =
      createCloseTagTensor(builder, loc, tensorType, value);
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
  RankedTensorType tensorType = getCloseTagTensorType(source, builder, {1});
  Value tagTensor =
      ttg::LocalLoadOp::create(builder, loc, tensorType, slot);
  Value tagI32 =
      tt::UnsplatOp::create(builder, loc, builder.getI32Type(), tagTensor);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value tag = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                    tagI32, zero);
  setRoleTaskId(source, slot.getDefiningOp(), taskId);
  setRoleTaskId(source, tagTensor.getDefiningOp(), taskId);
  setRoleTaskId(source, tagI32.getDefiningOp(), taskId);
  setRoleTaskId(source, zero.getDefiningOp(), taskId);
  setRoleTaskId(source, tag.getDefiningOp(), taskId);
  return tag;
}

class TritonTleLowerPipeToNvwsPass
    : public impl::TritonTleLowerPipeToNvwsBase<TritonTleLowerPipeToNvwsPass> {
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
    if (!ops.empty())
      module->setAttr(kTleEnableEncodingRematerializationAttr,
                      UnitAttr::get(module.getContext()));

    std::map<std::string, PipeDefinition> pipeDefinitions;
    std::map<Operation *, PipeCommitAnalysis> commitAnalyses;
    if (failed(analyzePipeCommits(ops, pipeDefinitions, commitAnalyses))) {
      signalPassFailure();
      return;
    }
    std::map<std::string, int32_t> drainParticipantCounts;
    if (failed(analyzePipeDrains(ops, pipeDefinitions,
                                 drainParticipantCounts))) {
      signalPassFailure();
      return;
    }
    DenseSet<Operation *> participantCountedWaits;
    if (failed(analyzeParticipantCountedWaits(
            ops, pipeDefinitions, participantCountedWaits))) {
      signalPassFailure();
      return;
    }

    std::map<std::string, PipeState> pipes;
    SmallVector<PipeCreateOp> creates;
    for (Operation *op : ops) {
      std::string key = getPipeKey(op);
      if (auto create = dyn_cast<PipeCreateOp>(op)) {
        std::optional<int32_t> drainCount;
        auto drainIt = drainParticipantCounts.find(key);
        if (drainIt != drainParticipantCounts.end())
          drainCount = drainIt->second;
        pipes.emplace(key, createPipeState(create, drainCount));
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
        Value token = getWarpSpecializeCaptureForUse(op, state.token);
        auto nvwsOp = ttnvws::ProducerAcquireOp::create(
            builder, loc, token, acquire.getStage(), acquire.getPhase());
        setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
        acquire.erase();
        continue;
      }

      if (auto commit = dyn_cast<PipeWriterCommitOp>(op)) {
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*writer=*/0));
        if (failed(taskId)) {
          signalPassFailure();
          return;
        }
        auto threadCount = getTaskThreadCount(op);
        Value token = getWarpSpecializeCaptureForUse(op, state.token);
        auto analysisIt = commitAnalyses.find(op);
        if (analysisIt == commitAnalyses.end()) {
          commit.emitOpError("is missing precomputed pipe commit analysis");
          signalPassFailure();
          return;
        }
        const PipeCommitAnalysis &analysis = analysisIt->second;
        PipeCommitTransport transport = analysis.transport;
        std::optional<int32_t> participantCount = analysis.participantCount;
        bool hasTmaPayload = analysis.tmaInfo.sawPipeTmaCopy;
        bool hasCpAsyncPayload = commit->hasAttr(kTlePipeCommitCpAsyncAttr);
        if (failed(recordDataTransport(state, op, transport))) {
          signalPassFailure();
          return;
        }
        if (failed(threadCount) ||
            failed(recordWriterTask(state, op, *taskId, *threadCount))) {
          signalPassFailure();
          return;
        }
        if (transport == PipeCommitTransport::LocalStore ||
            transport == PipeCommitTransport::CpAsync) {
          int32_t fullCount = participantCount.value_or(*threadCount);
          if (failed(setWriterFullCount(state, op, fullCount))) {
            signalPassFailure();
            return;
          }
        }
        if (transport == PipeCommitTransport::MixedTmaLocalStore) {
          setTokenLoadType(state.token, ttnvws::TokenLoadType::LocalStoreOp);
          if (participantCount &&
              failed(setWriterFullCount(state, op, *participantCount + 1))) {
            signalPassFailure();
            return;
          }
          if (!participantCount) {
            auto createToken =
                cast<ttnvws::CreateTokenOp>(state.token.getDefiningOp());
            createToken->setAttr(kTleInferFullCountOffsetAttr,
                                 builder.getI32IntegerAttr(1));
          }
        }
        if (transport == PipeCommitTransport::MixedTmaCpAsync) {
          setTokenLoadType(state.token, ttnvws::TokenLoadType::LocalStoreOp);
          if (failed(setWriterFullCount(state, op, *threadCount + 1))) {
            signalPassFailure();
            return;
          }
        }

        auto createCommit = [&](ttnvws::ProducerCommitKind kind) {
          auto nvwsOp = ttnvws::ProducerCommitOp::create(builder, loc, token,
                                                         commit.getStage());
          nvwsOp->setAttr(
              nvwsOp.getCommitKindAttrName(),
              ttnvws::ProducerCommitKindAttr::get(builder.getContext(), kind));
          setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
          return nvwsOp;
        };

        if (transport == PipeCommitTransport::MixedTmaLocalStore) {
          createCommit(ttnvws::ProducerCommitKind::TmaCopyBarrierArrive);
          auto localCommit = createCommit(
              ttnvws::ProducerCommitKind::ParticipantBarrierArrive);
          if (participantCount) {
            localCommit->setAttr("arrive_count",
                                 builder.getI32IntegerAttr(*participantCount));
          } else {
            localCommit->setAttr(kTleInferArriveCountAttr,
                                 builder.getUnitAttr());
          }
        } else if (transport == PipeCommitTransport::MixedTmaCpAsync) {
          createCommit(ttnvws::ProducerCommitKind::TmaCopyBarrierArrive);
          createCommit(ttnvws::ProducerCommitKind::AsyncCopyMbarrierArrive);
        } else if (transport == PipeCommitTransport::TmaCopy) {
          setTokenLoadType(state.token, ttnvws::TokenLoadType::TMALoadOp);
          createCommit(ttnvws::ProducerCommitKind::TmaCopyBarrierArrive);
        } else if (hasCpAsyncPayload) {
          auto nvwsOp = ttnvws::ProducerCommitOp::create(builder, loc, token,
                                                         commit.getStage());
          setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
          nvwsOp->setAttr(
              nvwsOp.getCommitKindAttrName(),
              ttnvws::ProducerCommitKindAttr::get(
                  builder.getContext(),
                  ttnvws::ProducerCommitKind::AsyncCopyMbarrierArrive));
        } else if (participantCount) {
          auto nvwsOp = ttnvws::ProducerCommitOp::create(builder, loc, token,
                                                         commit.getStage());
          setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
          nvwsOp->setAttr(
              nvwsOp.getCommitKindAttrName(),
              ttnvws::ProducerCommitKindAttr::get(
                  builder.getContext(),
                  ttnvws::ProducerCommitKind::ParticipantBarrierArrive));
        } else {
          auto nvwsOp = ttnvws::ProducerCommitOp::create(builder, loc, token,
                                                         commit.getStage());
          setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
        }
        commit.erase();
        continue;
      }

      if (auto close = dyn_cast<PipeWriterCloseOp>(op)) {
        if (state.oneShot) {
          close.emitOpError("does not support close on one_shot pipe");
          signalPassFailure();
          return;
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
        if (failed(setWriterFullCount(state, op, *threadCount))) {
          signalPassFailure();
          return;
        }
        Value token = getWarpSpecializeCaptureForUse(op, state.token);
        auto acquireOp = ttnvws::ProducerAcquireOp::create(
            builder, loc, token, close.getStage(), close.getPhase());
        setRoleTaskId(op, acquireOp.getOperation(), *taskId);
        storeCloseTag(builder, loc, state, close.getStage(), /*value=*/true, op,
                      *taskId);
        auto commitOp = ttnvws::ProducerCommitOp::create(builder, loc, token,
                                                         close.getStage());
        setRoleTaskId(op, commitOp.getOperation(), *taskId);
        close.erase();
        continue;
      }

      if (auto wait = dyn_cast<PipeReaderWaitOp>(op)) {
        auto taskId =
            getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*reader=*/1));
        if (failed(taskId)) {
          signalPassFailure();
          return;
        }
        auto threadCount = getTaskThreadCount(op);
        auto readerName = getPipeReaderName(state, op);
        if (failed(threadCount) || failed(readerName) ||
            failed(recordReaderTask(
                state, op, *readerName, *taskId, *threadCount,
                /*updateEmptyCountForReader=*/!state.oneShot))) {
          signalPassFailure();
          return;
        }
        Value token = getWarpSpecializeCaptureForUse(op, state.token);
        auto nvwsOp = ttnvws::ConsumerWaitOp::create(
            builder, loc, token, wait.getStage(), wait.getPhase());
        if (participantCountedWaits.contains(op))
          nvwsOp.setWaitMode(ttnvws::ConsumerWaitMode::ParticipantCounted);
        setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
        if (!wait.getIsClosed().use_empty()) {
          Value isClosed;
          if (state.oneShot) {
            isClosed = arith::ConstantIntOp::create(builder, loc, 0, 1);
            setRoleTaskId(op, isClosed.getDefiningOp(), *taskId);
          } else {
            isClosed =
                loadCloseTag(builder, loc, state, wait.getStage(), op, *taskId);
          }
          wait.getIsClosed().replaceAllUsesWith(isClosed);
        }
        wait.erase();
        continue;
      }

      if (auto drain = dyn_cast<PipeDrainOp>(op)) {
        if (!state.drainBarrier) {
          drain.emitOpError("is missing precomputed drain barrier");
          signalPassFailure();
          return;
        }
        auto taskId = getSingleTaskId(
            op, getEnclosingDefaultTaskId(op,
                                          /*nonWarpSpecializeDefault=*/0));
        if (failed(taskId)) {
          signalPassFailure();
          return;
        }
        auto threadCount = getTaskThreadCount(op);
        if (failed(threadCount)) {
          signalPassFailure();
          return;
        }
        Value barrier = getWarpSpecializeCaptureForUse(op, state.drainBarrier);
        auto arrive =
            ttng::ArriveBarrierOp::create(builder, loc, barrier, *threadCount);
        arrive.setParticipantArrive(true);
        arrive.setReleaseFence(true);
        setRoleTaskId(op, arrive.getOperation(), *taskId);
        Value phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
        setRoleTaskId(op, phase.getDefiningOp(), *taskId);
        auto wait =
            ttng::WaitBarrierOp::create(builder, loc, barrier, phase);
        setRoleTaskId(op, wait.getOperation(), *taskId);
        drain.erase();
        continue;
      }

      auto release = cast<PipeReaderReleaseOp>(op);
      if (state.oneShot) {
        auto readerName = getPipeReaderName(state, op);
        if (failed(readerName)) {
          signalPassFailure();
          return;
        }
        release.erase();
        continue;
      }
      auto taskId =
          getSingleTaskId(op, getEnclosingDefaultTaskId(op, /*reader=*/1));
      if (failed(taskId)) {
        signalPassFailure();
        return;
      }
      auto threadCount = getTaskThreadCount(op);
      auto readerName = getPipeReaderName(state, op);
      if (failed(threadCount) || failed(readerName) ||
          failed(recordReaderTask(state, op, *readerName, *taskId,
                                  *threadCount))) {
        signalPassFailure();
        return;
      }
      Value token = getWarpSpecializeCaptureForUse(op, state.token);
      auto releaseCountAttr = builder.getI32IntegerAttr(*threadCount);
      SmallVector<Value> releasedFields;
      for (Value field : release.getFields())
        releasedFields.push_back(getWarpSpecializeCaptureForUse(op, field));
      auto nvwsOp = ttnvws::ConsumerReleaseOp::create(
          builder, loc, token, release.getStage(), releasedFields,
          releaseCountAttr);
      if (!releasedFields.empty())
        nvwsOp->setAttr(kTleParticipantConsumerReleaseAttr,
                        builder.getUnitAttr());
      setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
      release.erase();
    }

    // The call marker owns the canonical lowered ABI used to restore one
    // shared noinline helper. Replace original pipe identities in that ABI by
    // the NVWS tokens created for the concrete pipe instance. The token
    // lowering pass will subsequently replace those token parameters by the
    // required barrier arrays across the restored call boundary.
    SmallVector<PipeCallBeginOp> callBegins;
    module.walk([&](PipeCallBeginOp begin) { callBegins.push_back(begin); });
    for (PipeCallBeginOp begin : llvm::reverse(callBegins)) {
      for (auto [index, operand] :
           llvm::enumerate(begin->getOpOperands())) {
        Value identity = canonicalizePipeField(operand.get());
        if (identity != operand.get() && isNvwsTokenValue(identity)) {
          operand.set(identity);
          begin.getAliases()[index].setType(identity.getType());
          continue;
        }
        for (auto &[key, definition] : pipeDefinitions) {
          if (definition.create.getIdentity() != identity)
            continue;
          auto state = pipes.find(key);
          assert(state != pipes.end() && "lowered pipe must have state");
          Value token =
              getWarpSpecializeCaptureForUse(begin, state->second.token);
          operand.set(token);
          begin.getAliases()[index].setType(token.getType());
          break;
        }
      }
    }

    // Pipe identities are explicit warp-specialize captures so lifecycle ops
    // in isolated partition regions retain structural instance identity. Once
    // those ops are lowered, remove their dead block arguments and captures
    // before erasing the defining pipe.create ops.
    RewritePatternSet patterns(&getContext());
    ttg::WarpSpecializeOp::getCanonicalizationPatterns(patterns,
                                                        &getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      module.emitError("failed to canonicalize lowered pipe captures");
      signalPassFailure();
      return;
    }

    for (PipeCreateOp create : creates) {
      if (!create.getIdentity().use_empty()) {
        create.emitOpError("SSA identity still has uses after pipe lowering");
        signalPassFailure();
        return;
      }
      create.erase();
    }
  }
};

} // namespace

} // namespace mlir::triton::tle
