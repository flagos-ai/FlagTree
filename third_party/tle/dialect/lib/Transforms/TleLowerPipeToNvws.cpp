#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <optional>

namespace mlir::triton::tle {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttnvws = mlir::triton::nvws;

#define GEN_PASS_DEF_TRITONTLELOWERPIPETONVWS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

enum class PipeCommitTransport {
  LocalStore,
  CpAsync,
  TmaCopy,
};

struct PipeState {
  Value token;
  Value closeTags;
  ttg::MemDescType closeTagSlotType;
  RankedTensorType closeTagTensorType;
  SmallVector<std::string> readerNames;
  bool oneShot;
  std::optional<int32_t> writerTaskId;
  std::optional<int32_t> writerThreadCount;
  std::optional<int32_t> writerFullCount;
  std::map<std::string, std::pair<int32_t, int32_t>> readerTasks;
  std::optional<PipeCommitTransport> dataTransport;
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
  return cast<PipeReaderReleaseOp>(op).getFields();
}

static bool isPipeLifecycleOp(Operation *op) {
  return isa<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
             PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp>(op);
}

static bool containsPipeLifecycleOp(tt::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isPipeLifecycleOp(op))
      found = true;
  });
  return found;
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

static Value canonicalizePipeField(Value field) {
  while (auto blockArg = dyn_cast<BlockArgument>(field)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
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
  }
  return field;
}

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
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

static StringRef getTransportName(PipeCommitTransport transport) {
  switch (transport) {
  case PipeCommitTransport::LocalStore:
    return "local-store";
  case PipeCommitTransport::CpAsync:
    return "cp.async";
  case PipeCommitTransport::TmaCopy:
    return "TMA copy";
  }
  llvm_unreachable("unknown pipe commit transport");
}

static LogicalResult recordDataTransport(PipeState &state, Operation *op,
                                         PipeCommitTransport transport) {
  if (state.dataTransport && *state.dataTransport != transport)
    return op->emitOpError("mixes ")
           << getTransportName(*state.dataTransport) << " and "
           << getTransportName(transport)
           << " payload commits on the same pipe; pipe full-barrier count is "
              "a per-pipe contract";
  state.dataTransport = transport;
  return success();
}

static bool isSharedPointer(Value value) {
  Type type = value.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    type = tensorTy.getElementType();
  auto ptrTy = dyn_cast<tt::PointerType>(type);
  return ptrTy && ptrTy.getAddressSpace() == 3;
}

static bool isGlobalPointer(Value value) {
  Type type = value.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    type = tensorTy.getElementType();
  auto ptrTy = dyn_cast<tt::PointerType>(type);
  return ptrTy && ptrTy.getAddressSpace() == 1;
}

static bool sameIndexValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsCst = lhs.getDefiningOp<arith::ConstantIntOp>();
  auto rhsCst = rhs.getDefiningOp<arith::ConstantIntOp>();
  if (lhsCst && rhsCst)
    return lhsCst.value() == rhsCst.value();
  auto lhsIdx = lhs.getDefiningOp<arith::ConstantIndexOp>();
  auto rhsIdx = rhs.getDefiningOp<arith::ConstantIndexOp>();
  if (lhsIdx && rhsIdx)
    return lhsIdx.value() == rhsIdx.value();
  return false;
}

static std::optional<int32_t> inferPrefixParticipants(Type valueType,
                                                      int32_t taskThreadCount) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueType);
  if (!tensorTy || !tensorTy.hasStaticShape() ||
      !isa<ttg::BlockedEncodingAttr>(tensorTy.getEncoding()))
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

struct LocalStoreTarget {
  Value memdesc;
  Type valueType;
};

static std::optional<LocalStoreTarget> getLocalStoreTarget(Operation *op) {
  if (auto localStore = dyn_cast<ttg::LocalStoreOp>(op))
    return LocalStoreTarget{localStore.getDst(), localStore.getSrc().getType()};

  auto store = dyn_cast<tt::StoreOp>(op);
  if (!store)
    return std::nullopt;

  Value ptr = stripConvertLayouts(store.getPtr());
  while (auto addPtr = ptr.getDefiningOp<tt::AddPtrOp>())
    ptr = stripConvertLayouts(addPtr.getPtr());
  auto localPointers = ptr.getDefiningOp<LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;
  return LocalStoreTarget{localPointers.getSrc(), store.getValue().getType()};
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
    break;
  }

  if (!sawStageIndex && getPipeCapacity(commit.getOperation()) != 1)
    return std::nullopt;
  return current;
}

static bool canInterleaveBeforeLocalStorePipeCommit(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isMemoryEffectFree(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile() && isGlobalPointer(load.getPtr());
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    if (getLocalStoreTarget(op))
      return true;
    return isGlobalPointer(store.getPtr());
  }
  if (isa<ttg::LocalStoreOp>(op))
    return true;
  return false;
}

static std::optional<int32_t>
inferLocalStoreParticipantCount(PipeWriterCommitOp commit,
                                int32_t taskThreadCount, Value token) {
  llvm::DenseSet<Value> fieldRoots;
  for (Value field : commit.getFields()) {
    Value root = getMemDescRoot(field);
    // If multiple logical fields share one root allocation, root-only alias
    // reasoning cannot distinguish which field was written. Keep the full
    // partition contract rather than publish a partially observed payload.
    if (!fieldRoots.insert(root).second)
      return std::nullopt;
  }

  llvm::DenseSet<Value> storedRoots;
  std::optional<int32_t> participants;
  bool sawLocalStore = false;
  std::string commitKey = getPipeKey(commit.getOperation());

  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (prev == token.getDefiningOp())
      break;
    if (auto acquire = dyn_cast<ttnvws::ProducerAcquireOp>(prev)) {
      if (acquire.getToken() == token &&
          sameIndexValue(acquire.getIdx(), commit.getStage()))
        break;
      return std::nullopt;
    }
    if (auto acquire = dyn_cast<PipeWriterAcquireOp>(prev)) {
      if (getPipeKey(acquire.getOperation()) == commitKey)
        break;
      return std::nullopt;
    }
    if (auto create = dyn_cast<PipeCreateOp>(prev)) {
      if (getPipeKey(create.getOperation()) == commitKey)
        break;
      return std::nullopt;
    }
    if (isPipeLifecycleOp(prev))
      return std::nullopt;

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
  return isMemoryEffectFree(op);
}

static FailureOr<bool> isTmaPipeCommit(PipeWriterCommitOp commit) {
  llvm::DenseSet<Value> fieldRoots;
  for (Value field : commit.getFields())
    fieldRoots.insert(getMemDescRoot(field));

  llvm::DenseSet<Value> copiedRoots;
  bool sawPipeTmaCopy = false;
  for (Operation *prev = commit->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(prev)) {
      if (failed(verifyTmaCopyTypes(tmaCopy)))
        return failure();

      Value dstRoot = getMemDescRoot(tmaCopy.getDst());
      if (!fieldRoots.contains(dstRoot)) {
        if (sawPipeTmaCopy)
          return commit.emitOpError("has an unrelated ttg.tma_copy between "
                                    "pipe payload TMA copies and commit");
        return false;
      }
      copiedRoots.insert(dstRoot);
      sawPipeTmaCopy = true;
      continue;
    }

    if (canInterleaveBeforeTmaPipeCommit(prev))
      continue;
    break;
  }

  if (!sawPipeTmaCopy)
    return false;

  if (copiedRoots.size() != fieldRoots.size())
    return commit.emitOpError("TMA pipe commit must be immediately preceded by "
                              "TMA copies covering every pipe field");

  return true;
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

static PipeState createPipeState(PipeCreateOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *context = op->getContext();
  int64_t capacity = getPipeCapacity(op);
  bool oneShot = false;
  if (auto oneShotAttr = op->getAttrOfType<BoolAttr>("one_shot"))
    oneShot = oneShotAttr.getValue();

  auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  Value closeTags;
  ttg::MemDescType closeTagSlotType;
  RankedTensorType closeTagTensorType;
  if (!oneShot) {
    Attribute closeTagArrayEncoding = getCloseTagEncoding(context, 2);
    Attribute closeTagSlotEncoding = getCloseTagEncoding(context, 1);
    auto closeTagArrayType =
        ttg::MemDescType::get({capacity, 1}, builder.getI32Type(),
                              closeTagArrayEncoding, sharedMemorySpace,
                              /*mutableMemory=*/true);
    closeTagSlotType =
        ttg::MemDescType::get({1}, builder.getI32Type(), closeTagSlotEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);

    RankedTensorType closeTagArrayTensorType =
        getCloseTagTensorType(op, builder, {capacity, 1});
    Value initialCloseTags =
        createCloseTagTensor(builder, loc, closeTagArrayTensorType,
                             /*value=*/false);
    closeTags = ttg::LocalAllocOp::create(builder, loc, closeTagArrayType,
                                          initialCloseTags);
    closeTagTensorType = getCloseTagTensorType(op, builder, {1});
  }
  Value token = ttnvws::CreateTokenOp::create(
      builder, loc, static_cast<uint32_t>(capacity),
      ttnvws::TokenLoadType::LocalStoreOp);

  SmallVector<std::string> readerNames;
  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    readerNames.reserve(readersAttr.size());
    for (Attribute attr : readersAttr)
      readerNames.push_back(cast<StringAttr>(attr).getValue().str());
  }

  PipeState state{token,
                  closeTags,
                  closeTagSlotType,
                  closeTagTensorType,
                  readerNames,
                  oneShot,
                  /*writerTaskId=*/std::nullopt,
                  /*writerThreadCount=*/std::nullopt,
                  /*writerFullCount=*/std::nullopt,
                  /*readerTasks=*/{},
                  /*dataTransport=*/std::nullopt};
  op.erase();
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
  Value tag =
      createCloseTagTensor(builder, loc, state.closeTagTensorType, value);
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
  Value tagTensor =
      ttg::LocalLoadOp::create(builder, loc, state.closeTagTensorType, slot);
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

    std::map<std::string, PipeState> pipes;
    SmallVector<Operation *> ops;

    module.walk([&](Operation *op) {
      if (isPipeLifecycleOp(op))
        ops.push_back(op);
    });

    for (Operation *op : ops) {
      std::string key = getPipeKey(op);
      if (auto create = dyn_cast<PipeCreateOp>(op)) {
        if (pipes.count(key)) {
          create.emitOpError("duplicates an existing pipe.create");
          signalPassFailure();
          return;
        }
        pipes.emplace(key, createPipeState(create));
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
        FailureOr<bool> tmaCommit = isTmaPipeCommit(commit);
        if (failed(tmaCommit)) {
          signalPassFailure();
          return;
        }
        PipeCommitTransport transport = PipeCommitTransport::LocalStore;
        if (commit->hasAttr(kTlePipeCommitCpAsyncAttr))
          transport = PipeCommitTransport::CpAsync;
        if (*tmaCommit)
          transport = PipeCommitTransport::TmaCopy;
        if (failed(recordDataTransport(state, op, transport))) {
          signalPassFailure();
          return;
        }
        if (failed(threadCount) ||
            failed(recordWriterTask(state, op, *taskId, *threadCount))) {
          signalPassFailure();
          return;
        }

        Value token = getWarpSpecializeCaptureForUse(op, state.token);
        std::optional<int32_t> participantCount;
        if (transport == PipeCommitTransport::LocalStore)
          participantCount =
              inferLocalStoreParticipantCount(commit, *threadCount, token);
        if (transport == PipeCommitTransport::LocalStore ||
            transport == PipeCommitTransport::CpAsync) {
          int32_t fullCount = participantCount.value_or(*threadCount);
          if (failed(setWriterFullCount(state, op, fullCount))) {
            signalPassFailure();
            return;
          }
        }

        auto nvwsOp = ttnvws::ProducerCommitOp::create(builder, loc, token,
                                                       commit.getStage());
        setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
        if (*tmaCommit) {
          setTokenLoadType(state.token, ttnvws::TokenLoadType::TMALoadOp);
          nvwsOp->setAttr(
              nvwsOp.getCommitKindAttrName(),
              ttnvws::ProducerCommitKindAttr::get(
                  builder.getContext(),
                  ttnvws::ProducerCommitKind::TmaCopyBarrierArrive));
        } else if (commit->hasAttr(kTlePipeCommitCpAsyncAttr)) {
          nvwsOp->setAttr(
              nvwsOp.getCommitKindAttrName(),
              ttnvws::ProducerCommitKindAttr::get(
                  builder.getContext(),
                  ttnvws::ProducerCommitKind::AsyncCopyMbarrierArrive));
        } else if (participantCount) {
          nvwsOp->setAttr(
              nvwsOp.getCommitKindAttrName(),
              ttnvws::ProducerCommitKindAttr::get(
                  builder.getContext(),
                  ttnvws::ProducerCommitKind::ParticipantBarrierArrive));
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
      setRoleTaskId(op, nvwsOp.getOperation(), *taskId);
      release.erase();
    }
  }
};

} // namespace

} // namespace mlir::triton::tle
