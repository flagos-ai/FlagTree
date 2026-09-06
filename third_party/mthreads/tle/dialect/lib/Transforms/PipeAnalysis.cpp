#ifdef __TLE__

#include "MUSATLE/Transforms/PipeAnalysis.h"

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

namespace mlir::triton::musa_tle {

namespace tt = triton;
namespace ttg = triton::gpu;

namespace {

static int64_t getCapacity(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("capacity").getInt();
}

static OperandRange getFields(Operation *op) {
  if (auto pipe = dyn_cast<PipeCreateOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<PipeWriterAcquireOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<PipeWriterCommitOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<PipeWriterCloseOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<PipeReaderWaitOp>(op))
    return pipe.getFields();
  return cast<PipeReaderReleaseOp>(op).getFields();
}

static bool isPipeOp(Operation *op) {
  return isa<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
             PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp>(op);
}

static Value canonicalizePipeField(Value field) {
  while (auto blockArg = dyn_cast<BlockArgument>(field)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!partitions)
      break;
    auto ws = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!ws ||
        blockArg.getArgNumber() >= partitions.getExplicitCaptures().size())
      break;
    field = partitions.getExplicitCaptures()[blockArg.getArgNumber()];
  }
  return field;
}

// Barrier bases and slots can be captured by a warp-specialize partition in
// exactly the same way as pipe memdesc values.  Resolve those captures before
// inspecting the defining BarrierIndexOp/BarrierAllocOp so external barrier
// identity does not depend on the partition being visited first.
static Value canonicalizePipeValue(Value value) {
  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!partitions ||
        blockArg.getArgNumber() >= partitions.getExplicitCaptures().size())
      break;
    value = partitions.getExplicitCaptures()[blockArg.getArgNumber()];
  }
  return value;
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
    if (auto alias = current.getDefiningOp<tle::MemDescAliasOp>()) {
      current = canonicalizePipeField(alias.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = canonicalizePipeField(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      current = canonicalizePipeField(reshape.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      current = canonicalizePipeField(reinterpret.getSrc());
      continue;
    }
    if (auto wgmmaView = current.getDefiningOp<tle::MemDescWGMMAViewOp>()) {
      current = canonicalizePipeField(wgmmaView.getSrc());
      continue;
    }
    return current;
  }
}

static bool sameIndex(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  APInt lhsValue;
  APInt rhsValue;
  return matchPattern(lhs, m_ConstantInt(&lhsValue)) &&
         matchPattern(rhs, m_ConstantInt(&rhsValue)) && lhsValue == rhsValue;
}

struct PipeExternalBarrierUse {
  BarrierAllocOp allocation;
  Value base;
  Value slot;
};

static FailureOr<PipeExternalBarrierUse>
resolvePipeExternalBarrier(Value barrier, Value stage,
                           Operation *diagnosticOp) {
  Value canonicalBarrier = canonicalizePipeValue(barrier);
  auto index = canonicalBarrier.getDefiningOp<BarrierIndexOp>();
  if (!index)
    return diagnosticOp->emitOpError(
        "MUSA TLE pipe external completion barrier must be a stage-indexed "
        "barrier array");

  Value indexValue = canonicalizePipeValue(index.getIndex());
  Value canonicalStage = canonicalizePipeValue(stage);
  if (!sameIndex(indexValue, canonicalStage))
    return diagnosticOp->emitOpError(
        "MUSA TLE pipe external completion barrier stage must match the "
        "payload stage");

  Value base = canonicalizePipeValue(index.getBaseId());
  auto allocation = base.getDefiningOp<BarrierAllocOp>();
  if (!allocation)
    return diagnosticOp->emitOpError(
        "MUSA TLE pipe external completion barrier must be a stage-indexed "
        "barrier array");
  ModuleOp module = diagnosticOp->getParentOfType<ModuleOp>();
  if (!module)
    return diagnosticOp->emitOpError(
        "MUSA TLE pipe external completion barrier must dominate all pipe "
        "operations");
  DominanceInfo dominance(module);
  if (allocation->getParentOfType<tt::FuncOp>() !=
          diagnosticOp->getParentOfType<tt::FuncOp>() ||
      allocation->getParentOfType<ttg::WarpSpecializeOp>() ||
      !dominance.dominates(allocation.getOperation(), diagnosticOp))
    return diagnosticOp->emitOpError(
        "MUSA TLE pipe external completion barrier must dominate all pipe "
        "operations");

  APInt constantStage;
  if (matchPattern(indexValue, m_ConstantInt(&constantStage)) &&
      (constantStage.isNegative() ||
       constantStage.getZExtValue() >= allocation.getNumBarriers()))
    return diagnosticOp->emitOpError(
        "MUSA TLE pipe external completion barrier stage must match the "
        "payload stage");

  return PipeExternalBarrierUse{allocation, base, canonicalBarrier};
}

static Value stripPointerLayoutConversions(Value value) {
  Value current = value;
  while (auto convert = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = convert.getSrc();
  return current;
}

static Value stripIndexWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto setLayout = current.getDefiningOp<SetLayoutOp>()) {
      current = setLayout.getSrc();
      continue;
    }
    if (auto convert = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = convert.getSrc();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

// Structured control-flow can introduce equivalent values through casts,
// branch arguments, or loop-carried values.  Keep this deliberately
// conservative: when no proof is available the normal lifecycle diagnostics
// remain preferable to guessing that two generations are the same.
static Value canonicalizePipeIndex(Value value) {
  Value current = canonicalizePipeValue(value);
  while (true) {
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = canonicalizePipeValue(cast.getIn());
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = canonicalizePipeValue(ext.getIn());
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = canonicalizePipeValue(ext.getIn());
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = canonicalizePipeValue(trunc.getIn());
      continue;
    }
    if (auto convert = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = canonicalizePipeValue(convert.getSrc());
      continue;
    }
    break;
  }
  return current;
}

static bool equivalentPipeValueImpl(Value lhs, Value rhs, unsigned depth) {
  if (depth > 16)
    return false;
  lhs = canonicalizePipeIndex(lhs);
  rhs = canonicalizePipeIndex(rhs);
  if (lhs == rhs)
    return true;
  APInt lhsValue;
  APInt rhsValue;
  if (matchPattern(lhs, m_ConstantInt(&lhsValue)) &&
      matchPattern(rhs, m_ConstantInt(&rhsValue)))
    return lhsValue == rhsValue;

  auto lhsBlockArg = dyn_cast<BlockArgument>(lhs);
  auto rhsBlockArg = dyn_cast<BlockArgument>(rhs);
  if (lhsBlockArg && rhsBlockArg) {
    if (lhsBlockArg.getArgNumber() != rhsBlockArg.getArgNumber())
      return false;
    // Induction/iter arguments of the same structured operation are the
    // canonical representative of a loop-carried expression.  Region branch
    // arguments with the same ordinal are likewise equivalent when both arms
    // belong to one parent structured operation.
    return lhsBlockArg.getOwner()->getParentOp() ==
           rhsBlockArg.getOwner()->getParentOp();
  }

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || lhsDef->getName() != rhsDef->getName() ||
      lhsDef->getNumOperands() != rhsDef->getNumOperands() ||
      lhsDef->getAttrDictionary() != rhsDef->getAttrDictionary())
    return false;
  for (auto [leftOperand, rightOperand] :
       llvm::zip(lhsDef->getOperands(), rhsDef->getOperands())) {
    if (!equivalentPipeValueImpl(leftOperand, rightOperand, depth + 1))
      return false;
  }
  return true;
}

static bool equivalentPipeValue(Value lhs, Value rhs) {
  return equivalentPipeValueImpl(lhs, rhs, 0);
}

static bool matchZeroStartRange(Value value, int64_t extent) {
  auto range = stripIndexWrappers(value).getDefiningOp<tt::MakeRangeOp>();
  return range && range.getStart() == 0 && range.getEnd() == extent;
}

static bool matchWholeFieldIndex(Value index, size_t axis,
                                 ArrayRef<int64_t> shape) {
  auto indexTy = dyn_cast<RankedTensorType>(index.getType());
  if (!indexTy || !indexTy.getElementType().isInteger() ||
      indexTy.getShape() != shape)
    return false;

  Value current = stripIndexWrappers(index);
  if (shape.size() == 1)
    return matchZeroStartRange(current, shape.front());

  auto broadcast = current.getDefiningOp<tt::BroadcastOp>();
  if (!broadcast)
    return false;
  auto sourceTy = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
  if (!sourceTy || sourceTy.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [dimension, extent] : llvm::enumerate(shape)) {
    int64_t expected = dimension == axis ? extent : 1;
    if (sourceTy.getShape()[dimension] != expected)
      return false;
  }

  current = stripIndexWrappers(broadcast.getSrc());
  while (auto expand = current.getDefiningOp<tt::ExpandDimsOp>())
    current = stripIndexWrappers(expand.getSrc());
  auto rangeTy = dyn_cast<RankedTensorType>(current.getType());
  return rangeTy && rangeTy.getRank() == 1 &&
         rangeTy.getShape().front() == shape[axis] &&
         matchZeroStartRange(current, shape[axis]);
}

struct LocalStoreTarget {
  Value memdesc;
  bool exactWholeField = false;
};

static std::optional<LocalStoreTarget>
getLocalStoreTarget(Operation *operation) {
  if (auto localStore = dyn_cast<ttg::LocalStoreOp>(operation))
    return LocalStoreTarget{localStore.getDst(), true};

  auto store = dyn_cast<tt::StoreOp>(operation);
  if (!store)
    return std::nullopt;

  Value pointer = stripPointerLayoutConversions(store.getPtr());
  bool hasPointerOffset = false;
  while (auto addPtr = pointer.getDefiningOp<tt::AddPtrOp>()) {
    hasPointerOffset = true;
    pointer = stripPointerLayoutConversions(addPtr.getPtr());
  }
  auto localPointers = pointer.getDefiningOp<LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  bool exact =
      !hasPointerOffset && !store.getMask() && store.getBoundaryCheck().empty();
  auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
  auto pointerTy = dyn_cast<RankedTensorType>(localPointers.getType());
  auto memdescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!valueTy || !pointerTy || !memdescTy ||
      valueTy.getShape() != memdescTy.getShape() ||
      pointerTy.getShape() != memdescTy.getShape() ||
      valueTy.getElementType() != memdescTy.getElementType())
    exact = false;

  if (exact) {
    ValueRange indices = localPointers.getIndices();
    if (!indices.empty()) {
      if (indices.size() != static_cast<size_t>(memdescTy.getRank())) {
        exact = false;
      } else {
        for (auto [axis, index] : llvm::enumerate(indices)) {
          if (!matchWholeFieldIndex(index, axis, memdescTy.getShape())) {
            exact = false;
            break;
          }
        }
      }
    }
  }
  return LocalStoreTarget{localPointers.getSrc(), exact};
}

struct PipeIdentity {
  Operation *enclosingFunction = nullptr;
  int64_t capacity = 0;
  StringAttr scope;
  StringAttr pipeName;
  SmallVector<StringAttr> fieldNames;
  SmallVector<Value> fieldRoots;
};

static PipeIdentity getPipeIdentity(Operation *op) {
  PipeIdentity identity;
  if (auto func = op->getParentOfType<tt::FuncOp>())
    identity.enclosingFunction = func.getOperation();
  identity.capacity = getCapacity(op);
  identity.scope = op->getAttrOfType<StringAttr>("scope");
  identity.pipeName = op->getAttrOfType<StringAttr>("pipe_name");
  auto fieldNames = op->getAttrOfType<ArrayAttr>("field_names");
  identity.fieldNames.reserve(fieldNames.size());
  for (Attribute name : fieldNames)
    identity.fieldNames.push_back(cast<StringAttr>(name));
  identity.fieldRoots.reserve(getFields(op).size());
  for (Value field : getFields(op))
    identity.fieldRoots.push_back(getMemDescRoot(field));
  return identity;
}

static bool sameIdentity(const PipeIdentity &lhs, const PipeIdentity &rhs) {
  return lhs.enclosingFunction == rhs.enclosingFunction &&
         lhs.capacity == rhs.capacity && lhs.scope == rhs.scope &&
         lhs.pipeName == rhs.pipeName && lhs.fieldNames == rhs.fieldNames &&
         lhs.fieldRoots == rhs.fieldRoots;
}

static LogicalResult verifyCommonContract(Operation *op) {
  if (getCapacity(op) <= 0)
    return op->emitOpError("requires positive capacity");
  auto scope = op->getAttrOfType<StringAttr>("scope");
  if (!scope || scope.getValue() != "cta")
    return op->emitOpError("MUSA TLE pipe supports only scope='cta'");
  OperandRange fields = getFields(op);
  if (fields.empty())
    return op->emitOpError("requires at least one payload field");
  auto fieldNames = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNames || fieldNames.size() != fields.size())
    return op->emitOpError("requires one field name for every payload field");
  for (unsigned index = 0; index < fields.size(); ++index) {
    auto name = dyn_cast<StringAttr>(fieldNames[index]);
    if (!name)
      return op->emitOpError("requires string payload field names");
    Value root = getMemDescRoot(fields[index]);
    for (unsigned previous = 0; previous < index; ++previous) {
      if (name == dyn_cast<StringAttr>(fieldNames[previous]))
        return op->emitOpError("requires unique payload field names");
      if (root == getMemDescRoot(fields[previous]))
        return op->emitOpError("requires distinct payload field memdesc roots");
    }
  }
  return success();
}

static FailureOr<int32_t> getTransactionBytes(ttg::TMACopyOp copy) {
  bool globalToLocal = isa<tt::TensorDescType>(copy.getSrc().getType()) &&
                       isa<ttg::MemDescType>(copy.getDst().getType());
  Value descriptor = globalToLocal ? copy.getSrc() : copy.getDst();
  Value memdesc = globalToLocal ? copy.getDst() : copy.getSrc();
  auto descTy = dyn_cast<tt::TensorDescType>(descriptor.getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(memdesc.getType());
  if (!descTy || !memDescTy) {
    copy.emitOpError("MUSA TLE pipe TME copy requires a tensor-descriptor "
                     "and a shared-memory memdesc");
    return failure();
  }
  auto blockTy = descTy.getSignlessBlockType();
  if (blockTy.getShape() != memDescTy.getShape() ||
      blockTy.getElementType() != memDescTy.getElementType()) {
    copy.emitOpError("pipe TME descriptor block must match the destination "
                     "slot shape and element type");
    return failure();
  }

  int64_t elements = 1;
  for (int64_t dim : blockTy.getShape()) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim) {
      copy.emitOpError("cannot infer a positive static TME transaction size");
      return failure();
    }
    elements *= dim;
  }
  unsigned bitWidth = blockTy.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 ||
      elements > std::numeric_limits<int64_t>::max() / bitWidth ||
      (elements * bitWidth) % 8 != 0) {
    copy.emitOpError("TME transaction size must be a whole number of bytes");
    return failure();
  }
  int64_t bytes = elements * bitWidth / 8;
  if (bytes <= 0 || bytes > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("TME transaction bytes exceed the positive i32 range");
    return failure();
  }
  return static_cast<int32_t>(bytes);
}

static FailureOr<PipeStaticPartitionInfo> getEndpointPlacement(Operation *op) {
  FailureOr<std::optional<PipeStaticPartitionInfo>> resolved =
      resolvePipeStaticPartition(op);
  if (failed(resolved))
    return failure();
  if (!*resolved)
    return op->emitOpError(
        "MUSA TLE pipe operation is not in a recognized execution partition");
  return **resolved;
}

static LogicalResult
recordExecutionMode(PipeState &state, Operation *op,
                    const PipeStaticPartitionInfo &placement) {
  PipeExecutionMode mode = placement.kind == PipePartitionKind::CTA
                               ? PipeExecutionMode::NonWarpSpecialized
                               : PipeExecutionMode::StaticWarpSpecialized;
  if (mode == PipeExecutionMode::StaticWarpSpecialized) {
    if (state.staticWarpSpecialize &&
        state.staticWarpSpecialize != placement.owner)
      return op->emitOpError(
          "MUSA TLE pipe endpoint operations must use one execution mode and "
          "warp-specialize owner");
    state.staticWarpSpecialize = placement.owner;
  }
  if (state.executionMode != PipeExecutionMode::StaticWarpSpecialized)
    state.executionMode = mode;
  return success();
}

static bool samePlacement(const PipeEndpointState &endpoint,
                          const PipeStaticPartitionInfo &placement) {
  return endpoint.warpSpecialize == placement.owner &&
         endpoint.partitionIndex == placement.partitionIndex &&
         endpoint.partition == placement.kind &&
         endpoint.worker == placement.workerIndex &&
         endpoint.warpBegin == placement.warpBegin &&
         endpoint.warpCount == placement.warpCount;
}

static std::optional<Value> getSingleStageIndex(Value value) {
  Value current = canonicalizePipeField(value);
  std::optional<Value> stage;
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      if (stage)
        return std::nullopt;
      stage = index.getIndex();
      current = canonicalizePipeField(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = canonicalizePipeField(subslice.getSrc());
      continue;
    }
    if (auto alias = current.getDefiningOp<tle::MemDescAliasOp>()) {
      current = canonicalizePipeField(alias.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = canonicalizePipeField(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      current = canonicalizePipeField(reshape.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      current = canonicalizePipeField(reinterpret.getSrc());
      continue;
    }
    if (auto wgmmaView = current.getDefiningOp<tle::MemDescWGMMAViewOp>()) {
      current = canonicalizePipeField(wgmmaView.getSrc());
      continue;
    }
    break;
  }
  return stage;
}

} // namespace

PipeState *PipeAnalysisResult::lookupPipe(Operation *op) {
  return pipeByOperation.lookup(op);
}

const PipeState *PipeAnalysisResult::lookupPipe(Operation *op) const {
  return pipeByOperation.lookup(op);
}

PipeEndpointState *PipeAnalysisResult::lookupEndpoint(Operation *op) {
  PipeState *state = lookupPipe(op);
  auto endpoint = endpointIndexByOperation.find(op);
  if (!state || endpoint == endpointIndexByOperation.end() ||
      endpoint->second >= state->endpoints.size())
    return nullptr;
  return &state->endpoints[endpoint->second];
}

const PipeEndpointState *
PipeAnalysisResult::lookupEndpoint(Operation *op) const {
  const PipeState *state = lookupPipe(op);
  auto endpoint = endpointIndexByOperation.find(op);
  if (!state || endpoint == endpointIndexByOperation.end() ||
      endpoint->second >= state->endpoints.size())
    return nullptr;
  return &state->endpoints[endpoint->second];
}

const PipeCommitGroup *
PipeAnalysisResult::lookupCommitGroup(PipeWriterCommitOp op) const {
  return commitGroupByOperation.lookup(op.getOperation());
}

const PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainGroup(PipeReaderWaitOp op) const {
  return readerDrainGroupByWait.lookup(op.getOperation());
}

const PipeCloseGeneration *
PipeAnalysisResult::lookupCloseGeneration(PipeWriterCloseOp op) const {
  return closeGenerationByOperation.lookup(op.getOperation());
}

class PipeAnalysisBuilder {
public:
  explicit PipeAnalysisBuilder(ModuleOp module)
      : module(module), dominance(module) {}

  FailureOr<std::unique_ptr<PipeAnalysisResult>> run() {
    if (failed(collectLifecycleOps()) || failed(createPipeDefinitions()) ||
        failed(bindLifecycleOwnership()) || failed(initializeEndpoints()) ||
        failed(analyzeCommits()) || failed(analyzeReaders()) ||
        failed(finalizePipeStates()) || failed(validateExternalBarrierUses()))
      return failure();
    return std::move(result);
  }

private:
  struct PipeDefinition {
    PipeIdentity identity;
    PipeState *state = nullptr;
  };

  LogicalResult collectLifecycleOps() {
    module.walk([&](Operation *op) {
      if (isPipeOp(op))
        result->lifecycleOps.push_back(op);
    });
    for (Operation *op : result->lifecycleOps) {
      if (failed(verifyCommonContract(op)))
        return failure();
    }
    return success();
  }

  LogicalResult createPipeDefinitions() {
    for (Operation *op : result->lifecycleOps) {
      auto create = dyn_cast<PipeCreateOp>(op);
      if (!create)
        continue;
      if (!create->getParentOfType<tt::FuncOp>() ||
          create->getParentOfType<ttg::WarpSpecializeOp>())
        return create.emitOpError(
            "requires pipe.create outside warp_specialize");

      PipeIdentity identity = getPipeIdentity(op);
      auto candidates =
          definitionsByFirstRoot.lookup(identity.fieldRoots.front());
      for (unsigned index : candidates) {
        if (sameIdentity(definitions[index].identity, identity))
          return create.emitOpError("duplicates an existing pipe identity");
      }

      auto state = std::make_unique<PipeState>();
      state->create = create;
      state->capacity = static_cast<int32_t>(getCapacity(op));
      bool oneShot = false;
      if (auto oneShotAttr = create->getAttrOfType<BoolAttr>("one_shot"))
        oneShot = oneShotAttr.getValue();
      state->lifecycle.mode =
          oneShot ? PipeLifecycleMode::OneShot : PipeLifecycleMode::Cyclic;
      state->barrierPlan.full = {state->capacity,
                                 1,
                                 PipeBarrierInitialState::Pending,
                                 std::nullopt,
                                 PipeBarrierStorageOwner::Pipe,
                                 Value()};
      if (!oneShot)
        state->barrierPlan.empty =
            PipeBarrierRingPlan{state->capacity,
                                0,
                                PipeBarrierInitialState::Ready,
                                0,
                                PipeBarrierStorageOwner::Pipe,
                                Value()};
      state->barrierPlan.hasCloseState = false;

      auto fieldNames = create->getAttrOfType<ArrayAttr>("field_names");
      for (auto [index, field] : llvm::enumerate(create.getFields())) {
        auto name = cast<StringAttr>(fieldNames[index]).getValue().str();
        state->fields.push_back(PipeFieldState{static_cast<unsigned>(index),
                                               name,
                                               field,
                                               getMemDescRoot(field),
                                               field.getType(),
                                               {}});
      }

      PipeState *statePtr = state.get();
      result->pipes.push_back(std::move(state));
      unsigned definitionIndex = definitions.size();
      definitions.push_back(PipeDefinition{std::move(identity), statePtr});
      definitionsByFirstRoot[definitions.back().identity.fieldRoots.front()]
          .push_back(definitionIndex);
      result->pipeByOperation[op] = statePtr;
    }
    return success();
  }

  FailureOr<PipeState *> resolvePipe(Operation *op) {
    PipeIdentity identity = getPipeIdentity(op);
    if (identity.fieldRoots.empty()) {
      op->emitOpError("requires a preceding matching pipe.create");
      return failure();
    }

    PipeState *match = nullptr;
    auto candidates =
        definitionsByFirstRoot.lookup(identity.fieldRoots.front());
    for (unsigned index : candidates) {
      PipeDefinition &definition = definitions[index];
      if (!sameIdentity(definition.identity, identity) ||
          !dominance.dominates(definition.state->create.getOperation(), op))
        continue;
      if (match) {
        op->emitOpError("matches multiple dominating pipe.create operations");
        return failure();
      }
      match = definition.state;
    }
    if (!match) {
      op->emitOpError("requires a preceding matching pipe.create");
      return failure();
    }
    return match;
  }

  LogicalResult bindLifecycleOwnership() {
    for (Operation *op : result->lifecycleOps) {
      if (isa<PipeCreateOp>(op))
        continue;
      FailureOr<PipeState *> resolved = resolvePipe(op);
      if (failed(resolved))
        return failure();
      PipeState &state = **resolved;
      result->pipeByOperation[op] = &state;

      if (auto acquire = dyn_cast<PipeWriterAcquireOp>(op))
        state.acquires.push_back(acquire);
      else if (auto commit = dyn_cast<PipeWriterCommitOp>(op))
        state.commits.push_back(commit);
      else if (auto close = dyn_cast<PipeWriterCloseOp>(op))
        state.closes.push_back(close);
      else if (auto wait = dyn_cast<PipeReaderWaitOp>(op))
        state.waits.push_back(wait);
      else
        state.releases.push_back(cast<PipeReaderReleaseOp>(op));
    }
    return success();
  }

  struct ReaderSubscription {
    PipeReaderSubscriptionKind kind = PipeReaderSubscriptionKind::AllFields;
    SmallVector<unsigned> fields;
  };

  FailureOr<ReaderSubscription> getReaderSubscription(PipeState &state,
                                                      Operation *op) {
    ReaderSubscription subscription;
    if (auto readerFields = op->getAttrOfType<ArrayAttr>("reader_fields")) {
      if (readerFields.empty())
        return op->emitOpError(
            "MUSA TLE pipe reader field subscription must not be empty");
      SmallVector<bool> selected(state.fields.size(), false);
      for (Attribute fieldAttr : readerFields) {
        auto fieldName = dyn_cast<StringAttr>(fieldAttr);
        if (!fieldName)
          return op->emitOpError(
              "MUSA TLE pipe reader field subscription requires string "
              "field names");
        auto field = llvm::find_if(state.fields, [&](const PipeFieldState &f) {
          return f.name == fieldName.getValue();
        });
        if (field == state.fields.end())
          return op->emitOpError(
              "MUSA TLE pipe reader field subscription references an "
              "unknown payload field");
        if (selected[field->index])
          return op->emitOpError(
              "MUSA TLE pipe reader field subscription contains a duplicate "
              "payload field");
        selected[field->index] = true;
      }
      for (unsigned fieldIndex = 0; fieldIndex < selected.size();
           ++fieldIndex) {
        if (selected[fieldIndex])
          subscription.fields.push_back(fieldIndex);
      }
      subscription.kind = subscription.fields.size() == state.fields.size()
                              ? PipeReaderSubscriptionKind::AllFields
                              : PipeReaderSubscriptionKind::ExplicitSubset;
      return subscription;
    }

    subscription.fields.reserve(state.fields.size());
    for (unsigned fieldIndex = 0; fieldIndex < state.fields.size();
         ++fieldIndex)
      subscription.fields.push_back(fieldIndex);
    return subscription;
  }

  LogicalResult
  recordEndpoint(PipeState &state, Operation *op, PipeEndpointRole endpointRole,
                 unsigned expectedIndex, StringRef name,
                 std::optional<PipeReaderSubscriptionKind> readerSubscription,
                 ArrayRef<unsigned> subscribedFields) {
    FailureOr<PipeStaticPartitionInfo> placement = getEndpointPlacement(op);
    if (failed(placement))
      return failure();

    if (failed(recordExecutionMode(state, op, *placement)))
      return failure();

    if (expectedIndex < state.endpoints.size()) {
      PipeEndpointState &endpoint = state.endpoints[expectedIndex];
      if (endpoint.index != expectedIndex || endpoint.role != endpointRole ||
          endpoint.name != name ||
          endpoint.readerSubscription != readerSubscription ||
          endpoint.subscribedFields != subscribedFields)
        return op->emitOpError(
            "MUSA TLE pipe endpoint operations require one stable field "
            "subscription");
      if (!samePlacement(endpoint, *placement))
        return op->emitOpError(
            "MUSA TLE pipe endpoint operations must remain in one static "
            "warp-specialize partition");
      result->endpointIndexByOperation[op] = expectedIndex;
      return success();
    }

    if (state.endpoints.size() != expectedIndex)
      return op->emitOpError(
          "internal MUSA TLE pipe analysis created unstable endpoint indices");

    state.endpoints.push_back(PipeEndpointState{
        expectedIndex, name.str(), endpointRole, readerSubscription,
        SmallVector<unsigned>(subscribedFields.begin(), subscribedFields.end()),
        placement->owner, placement->partitionIndex, placement->kind,
        placement->workerIndex, placement->warpBegin, placement->warpCount});
    result->endpointIndexByOperation[op] = expectedIndex;
    return success();
  }

  LogicalResult recordWriter(PipeState &state, Operation *op) {
    SmallVector<unsigned> fields;
    fields.reserve(state.fields.size());
    for (unsigned fieldIndex = 0; fieldIndex < state.fields.size();
         ++fieldIndex)
      fields.push_back(fieldIndex);
    return recordEndpoint(state, op, PipeEndpointRole::Writer,
                          /*expectedIndex=*/0, "writer", std::nullopt, fields);
  }

  LogicalResult recordReader(PipeState &state, Operation *op,
                             unsigned endpointIndex, StringRef readerName) {
    FailureOr<ReaderSubscription> subscription =
        getReaderSubscription(state, op);
    if (failed(subscription) ||
        failed(recordEndpoint(state, op, PipeEndpointRole::Reader,
                              endpointIndex, readerName, subscription->kind,
                              subscription->fields)))
      return failure();
    return success();
  }

  LogicalResult verifyReaderName(Operation *op,
                                 ArrayRef<StringAttr> declaredReaders) {
    auto readerName = op->getAttrOfType<StringAttr>("reader_name");
    if (declaredReaders.empty()) {
      if (readerName)
        return op->emitOpError(
            "MUSA TLE pipe operation uses a named reader without an "
            "explicit reader declaration");
      return success();
    }
    if (!readerName)
      return op->emitOpError(
          "MUSA TLE pipe requires reader_name for an explicitly declared "
          "reader");
    if (!llvm::is_contained(declaredReaders, readerName))
      return op->emitOpError(
          "MUSA TLE pipe operation uses an undeclared reader");
    return success();
  }

  LogicalResult initializeEndpoints() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      for (PipeWriterAcquireOp acquire : state.acquires) {
        if (failed(recordWriter(state, acquire)))
          return failure();
      }
      for (PipeWriterCommitOp commit : state.commits) {
        if (failed(recordWriter(state, commit)))
          return failure();
      }
      for (PipeWriterCloseOp close : state.closes) {
        if (failed(recordWriter(state, close)))
          return failure();
      }

      SmallVector<StringAttr> declaredReaders;
      if (auto readers = state.create->getAttrOfType<ArrayAttr>("readers")) {
        declaredReaders.reserve(readers.size());
        for (Attribute reader : readers)
          declaredReaders.push_back(cast<StringAttr>(reader));
      }
      for (PipeReaderWaitOp wait : state.waits) {
        if (failed(verifyReaderName(wait, declaredReaders)))
          return failure();
      }
      for (PipeReaderReleaseOp release : state.releases) {
        if (failed(verifyReaderName(release, declaredReaders)))
          return failure();
      }

      if (declaredReaders.empty()) {
        for (PipeReaderWaitOp wait : state.waits) {
          if (failed(recordReader(state, wait, /*endpointIndex=*/1, "")))
            return failure();
        }
        for (PipeReaderReleaseOp release : state.releases) {
          if (failed(recordReader(state, release, /*endpointIndex=*/1, "")))
            return failure();
        }
      } else {
        for (auto [readerOffset, readerName] :
             llvm::enumerate(declaredReaders)) {
          unsigned endpointIndex = static_cast<unsigned>(readerOffset) + 1;
          bool found = false;
          for (PipeReaderWaitOp wait : state.waits) {
            if (wait->getAttrOfType<StringAttr>("reader_name") != readerName)
              continue;
            found = true;
            if (failed(recordReader(state, wait, endpointIndex,
                                    readerName.getValue())))
              return failure();
          }
          for (PipeReaderReleaseOp release : state.releases) {
            if (release->getAttrOfType<StringAttr>("reader_name") != readerName)
              continue;
            found = true;
            if (failed(recordReader(state, release, endpointIndex,
                                    readerName.getValue())))
              return failure();
          }
          if (!found)
            return state.create.emitOpError(
                "MUSA TLE pipe declared reader has no lifecycle "
                "operations");
        }
      }

      if (state.barrierPlan.empty) {
        int64_t totalReaderWarps = 0;
        for (const PipeEndpointState &endpoint : state.endpoints) {
          if (endpoint.role != PipeEndpointRole::Reader)
            continue;
          totalReaderWarps += endpoint.warpCount;
          if (totalReaderWarps > std::numeric_limits<int32_t>::max())
            return state.create.emitOpError(
                "MUSA TLE pipe reader arrival count exceeds the positive "
                "i32 range");
        }
        if (totalReaderWarps > 0)
          state.barrierPlan.empty->arrivalCount =
              static_cast<int32_t>(totalReaderWarps);
      }
    }
    return success();
  }

  // A window is local to a commit. It is not a proof of the matching
  // consumer's control flow or of initialization of every payload byte.
  LogicalResult collectCommitEffects(PipeState &state, PipeCommitGroup &group) {
    llvm::DenseSet<Operation *> seen;
    auto record = [&](Operation *op) -> LogicalResult {
      if (!seen.insert(op).second)
        return success();
      Value target;
      PipeTransportKind kind;
      bool exactStore = true;
      if (auto copy = dyn_cast<ttg::TMACopyOp>(op)) {
        if (!isa<tt::TensorDescType>(copy.getSrc().getType()))
          return success();
        target = copy.getDst();
        kind = PipeTransportKind::TME;
      } else if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        target = copy.getResult();
        kind = PipeTransportKind::AsyncCopy;
      } else if (auto store = getLocalStoreTarget(op)) {
        target = store->memdesc;
        exactStore = store->exactWholeField;
        kind = PipeTransportKind::LocalStore;
      } else {
        return success();
      }
      Value root = getMemDescRoot(target);
      auto field = llvm::find_if(state.fields, [&](const PipeFieldState &f) {
        return f.memdescRoot == root;
      });
      if (field == state.fields.end())
        return success();
      auto stage = getSingleStageIndex(target);
      // An imprecise synchronous write can be published by the whole writer
      // partition. Async operations need a definite slot to bind completion.
      if (stage && !equivalentPipeValue(*stage, group.stage))
        return success();
      if (!stage && kind != PipeTransportKind::LocalStore)
        return op->emitOpError("cannot associate asynchronous pipe copy with "
                               "the commit stage");
      auto placement = getEndpointPlacement(op);
      if (failed(placement))
        return failure();
      if (!samePlacement(state.endpoints.front(), *placement))
        return op->emitOpError("pipe payload must be published in its writer "
                               "partition");
      PipeCoveredRegion region{field->index, root, std::nullopt, std::nullopt,
                               false};
      auto resolved = resolvePipeMemDescRegion(target);
      if (succeeded(resolved)) {
        region.byteOffset = resolved->interval.byteOffset;
        region.byteSize = resolved->interval.byteSize;
        region.exact = resolved->exact && exactStore;
      }
      int32_t bytes = 0;
      Value externalRoot;
      auto owner = PipeBarrierStorageOwner::Pipe;
      if (kind == PipeTransportKind::TME) {
        auto copy = cast<ttg::TMACopyOp>(op);
        auto transactionBytes = getTransactionBytes(copy);
        if (failed(transactionBytes))
          return failure();
        bytes = *transactionBytes;
        if (copy.getCompletionBarrier()) {
          auto external = resolvePipeExternalBarrier(
              copy.getCompletionBarrier(), *stage, copy);
          if (failed(external))
            return failure();
          if (external->allocation.getNumBarriers() != state.capacity)
            return copy.emitOpError(
                "MUSA TLE pipe external completion barrier capacity must match "
                "pipe capacity");
          externalRoot = external->base;
          owner = PipeBarrierStorageOwner::External;
          Operation *allocation = external->allocation;
          if (!dominance.dominates(allocation, state.create))
            return copy.emitOpError("external completion barrier must "
                                    "dominate pipe.create");
          auto [it, inserted] =
              externalBarrierOwners.try_emplace(allocation, &state);
          if (!inserted && it->second != &state)
            return copy.emitOpError("MUSA TLE external completion barrier "
                                    "cannot be shared by multiple pipes");
          pipeExternalTMECopies.insert(op);
        }
      }
      group.completionSources.push_back({kind, op, field->index,
                                         stage.value_or(group.stage), region,
                                         bytes, owner, externalRoot});
      return success();
    };

    // Walk backwards to the nearest local boundary. Nested structured regions
    // contribute effects, not enumerated lifecycle paths. Each TME operation
    // accounts its own bytes when executed; the commit supplies the arrival.
    std::function<LogicalResult(Operation *)> visit =
        [&](Operation *op) -> LogicalResult {
      if (isa<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp>(op))
        return success();
      if (failed(record(op)))
        return failure();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            if (failed(visit(&nested)))
              return failure();
      return success();
    };
    Operation *cursor = group.commit.getOperation();
    while (cursor) {
      for (Operation *previous = cursor->getPrevNode(); previous;
           previous = previous->getPrevNode()) {
        if (result->lookupPipe(previous) == &state) {
          if (auto acquire = dyn_cast<PipeWriterAcquireOp>(previous)) {
            if (equivalentPipeValue(acquire.getStage(), group.stage)) {
              return success();
            }
          }
          if (auto commit = dyn_cast<PipeWriterCommitOp>(previous)) {
            if (equivalentPipeValue(commit.getStage(), group.stage))
              return success();
          }
          if (isa<PipeCreateOp, PipeReaderWaitOp>(previous))
            return success();
        }
        if (failed(visit(previous)))
          return failure();
      }
      Operation *parent = cursor->getParentOp();
      if (!parent || isa<tt::FuncOp, ttg::WarpSpecializeOp,
                         ttg::WarpSpecializePartitionsOp>(parent))
        break;
      cursor = parent;
    }
    return success();
  }

  LogicalResult analyzeCommits() {
    for (auto &owned : result->pipes) {
      PipeState &state = *owned;
      for (PipeWriterCommitOp commit : state.commits) {
        auto group = std::make_unique<PipeCommitGroup>();
        group->commit = commit;
        group->stage = commit.getStage();
        if (failed(collectCommitEffects(state, *group)))
          return failure();
        // Reverse the backwards scan to retain lexical order on straight-line
        // fast paths. Lowering never relies on this order across regions.
        std::reverse(group->completionSources.begin(),
                     group->completionSources.end());
        int64_t bytes = 0;
        for (const auto &source : group->completionSources) {
          bytes += source.transactionBytes;
          if (bytes > std::numeric_limits<int32_t>::max())
            return commit.emitOpError("aggregate pipe transaction bytes "
                                      "exceed the positive i32 range");
          if (source.kind == PipeTransportKind::TME)
            group->tmeGroupArrivalCount = 1;
          else
            group->localStoreArrivalCount = state.endpoints.front().warpCount;
        }
        group->totalTransactionBytes = bytes;
        // Empty/unknown synchronous effects still have a well-defined ready
        // edge. Publishing does not imply initializing the entire payload.
        if (group->completionSources.empty())
          group->localStoreArrivalCount = state.endpoints.front().warpCount;
        result->commitGroupByOperation[commit] = group.get();
        state.commitGroups.push_back(std::move(group));
      }
    }
    return success();
  }

  static bool mayPrecede(Operation *before, Operation *after) {
    if (before == after)
      return false;
    Operation *left = before;
    while (left) {
      Operation *right = after;
      while (right) {
        if (left->getBlock() == right->getBlock())
          return left == right || left->isBeforeInBlock(right);
        right = right->getParentOp();
      }
      left = left->getParentOp();
    }
    return false;
  }

  LogicalResult analyzeReaders() {
    for (auto &owned : result->pipes) {
      PipeState &state = *owned;
      for (PipeReaderWaitOp wait : state.waits) {
        auto group = std::make_unique<PipeReaderDrainGroup>();
        group->wait = wait;
        group->stage = wait.getStage();
        group->phase = wait.getPhase();
        group->readerEndpoint = result->lookupEndpoint(wait)->index;
        result->readerDrainGroupByWait[wait] = group.get();
        state.readerDrainGroups.push_back(std::move(group));
      }
    }
    WalkResult walked = module.walk([&](ttg::TMACopyOp copy) -> WalkResult {
      if (!isa<ttg::MemDescType>(copy.getSrc().getType()))
        return WalkResult::advance();
      Value root = getMemDescRoot(copy.getSrc());
      auto stage = getSingleStageIndex(copy.getSrc());
      auto placement = getEndpointPlacement(copy);
      if (failed(placement))
        return WalkResult::interrupt();
      PipeReaderDrainGroup *matched = nullptr;
      const PipeFieldState *matchedField = nullptr;
      bool unsubscribed = false;
      for (auto &owned : result->pipes) {
        PipeState &state = *owned;
        for (const auto &field : state.fields) {
          if (field.memdescRoot != root)
            continue;
          for (auto &group : state.readerDrainGroups) {
            auto &endpoint = state.endpoints[group->readerEndpoint];
            if (!samePlacement(endpoint, *placement) ||
                (stage && !equivalentPipeValue(*stage, group->stage)) ||
                !dominance.dominates(group->wait.getOperation(), copy))
              continue;
            if (!llvm::is_contained(endpoint.subscribedFields, field.index)) {
              unsubscribed = true;
              continue;
            }
            if (!matched || dominance.dominates(matched->wait.getOperation(),
                                                group->wait.getOperation())) {
              matched = group.get();
              matchedField = &field;
            }
          }
        }
      }
      if (!matched) {
        if (unsubscribed) {
          copy.emitOpError("reader TME store source is not included in the "
                           "reader field subscription");
          return WalkResult::interrupt();
        }
        return WalkResult::advance(); // standalone TME store
      }
      PipeCoveredRegion region{matchedField->index, root, std::nullopt,
                               std::nullopt, false};
      auto resolved = resolvePipeMemDescRegion(copy.getSrc());
      if (succeeded(resolved)) {
        region.byteOffset = resolved->interval.byteOffset;
        region.byteSize = resolved->interval.byteSize;
        region.exact = resolved->exact;
      }
      bool modified = false;
      auto function = copy->getParentOfType<tt::FuncOp>();
      function.walk([&](Operation *op) {
        if (!mayPrecede(matched->wait, op))
          return;
        auto *state = result->lookupPipe(matched->wait);
        if (llvm::any_of(state->releases, [&](auto release) {
              return result->lookupEndpoint(release)->index ==
                         matched->readerEndpoint &&
                     equivalentPipeValue(release.getStage(), matched->stage) &&
                     dominance.dominates(matched->wait.getOperation(),
                                         release) &&
                     dominance.dominates(release.getOperation(), op);
            }))
          return;
        Value destination;
        if (auto store = getLocalStoreTarget(op))
          destination = store->memdesc;
        else if (auto async = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
          destination = async.getResult();
        else if (auto tme = dyn_cast<ttg::TMACopyOp>(op)) {
          if (isa<ttg::MemDescType>(tme.getDst().getType()))
            destination = tme.getDst();
        }
        if (destination && getMemDescRoot(destination) == root)
          modified = true;
        if (!destination) {
          if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
            SmallVector<MemoryEffects::EffectInstance> instances;
            effects.getEffects(instances);
            for (const auto &effect : instances) {
              if (!isa<MemoryEffects::Write, MemoryEffects::Free>(
                      effect.getEffect()))
                continue;
              Value value = effect.getValue();
              if (!value) {
                modified = true;
                continue;
              }
              Type type = value.getType();
              if (auto tensor = dyn_cast<RankedTensorType>(type))
                type = tensor.getElementType();
              if (auto pointer = dyn_cast<tt::PointerType>(type))
                modified |= pointer.getAddressSpace() == 3;
              else if (isa<ttg::MemDescType>(type))
                modified |= getMemDescRoot(value) == root;
            }
          } else if (op->getNumRegions() == 0 && !isMemoryEffectFree(op) &&
                     !isPipeOp(op) && !op->hasTrait<OpTrait::IsTerminator>()) {
            modified = true;
          }
        }
      });
      matched->sourceModifiedAfterWait |= modified;
      matched->drainSources.push_back({PipeReaderDrainKind::TMEStore, copy,
                                       matchedField->index, region,
                                       copy.getDst()});
      return WalkResult::advance();
    });
    return failure(walked.wasInterrupted());
  }

  LogicalResult finalizePipeStates() {
    for (auto &owned : result->pipes) {
      PipeState &state = *owned;
      bool oneShot = state.lifecycle.mode == PipeLifecycleMode::OneShot;
      if (oneShot && !state.closes.empty())
        return state.create.emitOpError(
            "MUSA TLE one-shot pipe does not support writer.close");
      if (state.endpoints.size() < 2 ||
          (state.commits.empty() && state.closes.empty()))
        return state.create.emitOpError("pipe requires a writer publication "
                                        "and a reader endpoint");
      if (state.staticWarpSpecialize) {
        int32_t totalWarps =
            module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName).getInt();
        for (int32_t warps : state.staticWarpSpecialize.getPartitionNumWarps())
          totalWarps += warps;
        for (auto &endpoint : state.endpoints) {
          if (endpoint.partition != PipePartitionKind::CTA)
            continue;
          for (Operation *op : result->lifecycleOps)
            if (result->lookupPipe(op) == &state &&
                result->lookupEndpoint(op) == &endpoint &&
                !dominance.dominates(op, state.staticWarpSpecialize))
              return op->emitOpError(
                  "CTA pipe endpoints must precede static warp dispatch");
          // Before dispatch all physical warps execute the CTA endpoint,
          // including warps subsequently assigned to workers.
          endpoint.warpCount = totalWarps;
        }
        if (state.barrierPlan.empty) {
          int64_t readers = 0;
          for (const auto &endpoint : state.endpoints)
            if (endpoint.role == PipeEndpointRole::Reader)
              readers += endpoint.warpCount;
          if (readers > std::numeric_limits<int32_t>::max())
            return state.create.emitOpError(
                "pipe reader arrival count exceeds i32");
          state.barrierPlan.empty->arrivalCount = readers;
        }
      }
      int32_t tmeArrivals = 0;
      int32_t localArrivals = 0;
      Value external;
      bool internalTME = false;
      for (auto &group : state.commitGroups) {
        tmeArrivals = std::max(tmeArrivals, group->tmeGroupArrivalCount);
        if (group->localStoreArrivalCount)
          localArrivals = state.endpoints.front().warpCount;
        for (const auto &source : group->completionSources) {
          if (source.kind != PipeTransportKind::TME)
            continue;
          if (!source.externalBarrierRoot) {
            internalTME = true;
          } else {
            if (external && external != source.externalBarrierRoot)
              return group->commit.emitOpError("external completion barrier "
                                               "must be used consistently");
            external = source.externalBarrierRoot;
          }
        }
      }
      if (!tmeArrivals && !localArrivals)
        tmeArrivals = 1; // close-only control publication
      if (external && internalTME)
        return state.create.emitOpError("external completion barrier must be "
                                        "bound to every TME source");
      state.barrierPlan.full.arrivalCount = tmeArrivals + localArrivals;
      // Transaction bytes are added at the actual copy, not fixed for a ring.
      state.barrierPlan.full.transactionBytes = 0;
      for (auto &group : state.commitGroups) {
        group->tmeGroupArrivalCount = tmeArrivals;
        group->localStoreArrivalCount = localArrivals;
        group->fullArrivalCount = tmeArrivals + localArrivals;
        group->externalBarrierRoot = external;
      }
      if (external) {
        auto allocation = external.getDefiningOp<BarrierAllocOp>();
        if (allocation.getNumBarriers() != state.capacity)
          return state.create.emitOpError("external completion barrier "
                                          "capacity must match pipe capacity");
        if (allocation.getInitPolarity() != 0)
          return state.create.emitOpError("external completion barrier must "
                                          "start in PENDING state");
        if (allocation.getArriveCount() != tmeArrivals + localArrivals)
          return state.create.emitOpError("external completion barrier "
                                          "arrival count must match transport");
        auto bytes = allocation->getAttrOfType<IntegerAttr>("expect_bytes");
        if (!bytes || bytes.getInt() <= 0)
          return state.create.emitOpError("external completion barrier "
                                          "requires positive expect_bytes");
        for (auto &group : state.commitGroups)
          if (group->totalTransactionBytes != bytes.getInt())
            return group->commit.emitOpError("external completion barrier "
                                             "expect_bytes must match "
                                             "aggregate TME bytes");
        state.barrierPlan.full.storageOwner = PipeBarrierStorageOwner::External;
        state.barrierPlan.full.externalStorage = external;
        state.barrierPlan.externalFull =
            PipeExternalBarrierBinding{allocation,
                                       external,
                                       state.capacity,
                                       tmeArrivals + localArrivals,
                                       PipeBarrierInitialState::Pending,
                                       static_cast<int32_t>(bytes.getInt())};
      }
      for (const auto &endpoint : state.endpoints) {
        PipeBarrierParticipant participant{
            endpoint.index, endpoint.partitionIndex, endpoint.partition,
            endpoint.warpBegin, endpoint.warpCount};
        if (endpoint.role == PipeEndpointRole::Writer)
          state.barrierPlan.writerParticipant = participant;
        else {
          state.barrierPlan.readerParticipants.push_back(participant);
          for (unsigned field : endpoint.subscribedFields)
            state.fields[field].subscribedReaders.push_back(endpoint.index);
        }
      }
      for (auto close : state.closes) {
        auto group = std::make_unique<PipeCloseGeneration>();
        group->close = close;
        group->stage = close.getStage();
        group->phase = close.getPhase();
        group->controlArrivalCount = tmeArrivals;
        group->localStoreArrivalCount = localArrivals;
        group->fullArrivalCount = tmeArrivals + localArrivals;
        result->closeGenerationByOperation[close] = group.get();
        state.closeGenerations.push_back(std::move(group));
      }
      if (!state.closes.empty()) {
        state.barrierPlan.hasCloseState = true;
        state.barrierPlan.closeTagPlan = PipeCloseTagPlan{
            state.capacity, false, PipeBarrierStorageOwner::Pipe};
      }
    }
    return success();
  }
  LogicalResult validateExternalBarrierUses() {
    // An external full barrier is a pipe-owned completion resource.  A
    // barrier operation or an unrelated explicit TME copy using the same
    // allocation would race the pipe's transaction accounting, so reject it
    // before LowerPipe materializes or rewrites anything.
    bool valid = true;
    module.walk([&](ttg::TMACopyOp copy) {
      if (!copy.getCompletionBarrier() ||
          pipeExternalTMECopies.contains(copy.getOperation()))
        return;
      Value barrier = canonicalizePipeValue(copy.getCompletionBarrier());
      auto index = barrier.getDefiningOp<BarrierIndexOp>();
      if (!index)
        return;
      Value base = canonicalizePipeValue(index.getBaseId());
      auto allocation = base.getDefiningOp<BarrierAllocOp>();
      auto owner = allocation
                       ? externalBarrierOwners.find(allocation.getOperation())
                       : externalBarrierOwners.end();
      if (owner != externalBarrierOwners.end()) {
        copy.emitOpError(
            "MUSA TLE external completion barrier cannot be used by a "
            "standalone TME copy");
        valid = false;
      }
    });
    if (!valid)
      return failure();

    module.walk([&](BarrierWaitOp wait) {
      Value barrier = canonicalizePipeValue(wait.getBarId());
      auto index = barrier.getDefiningOp<BarrierIndexOp>();
      if (!index)
        return;
      Value base = canonicalizePipeValue(index.getBaseId());
      auto allocation = base.getDefiningOp<BarrierAllocOp>();
      if (allocation &&
          externalBarrierOwners.count(allocation.getOperation())) {
        wait.emitOpError(
            "MUSA TLE external completion barrier is exclusively owned by "
            "the pipe");
        valid = false;
      }
    });
    module.walk([&](BarrierArriveOp arrive) {
      Value barrier = canonicalizePipeValue(arrive.getBarId());
      auto index = barrier.getDefiningOp<BarrierIndexOp>();
      if (!index)
        return;
      Value base = canonicalizePipeValue(index.getBaseId());
      auto allocation = base.getDefiningOp<BarrierAllocOp>();
      if (allocation &&
          externalBarrierOwners.count(allocation.getOperation())) {
        arrive.emitOpError(
            "MUSA TLE external completion barrier is exclusively owned by "
            "the pipe");
        valid = false;
      }
    });
    return success(valid);
  }

  ModuleOp module;
  DominanceInfo dominance;
  std::unique_ptr<PipeAnalysisResult> result =
      std::make_unique<PipeAnalysisResult>();
  SmallVector<PipeDefinition> definitions;
  llvm::DenseMap<Value, SmallVector<unsigned>> definitionsByFirstRoot;
  llvm::DenseMap<Operation *, PipeState *> externalBarrierOwners;
  llvm::DenseSet<Operation *> pipeExternalTMECopies;
};

FailureOr<std::unique_ptr<PipeAnalysisResult>>
analyzeMUSAPipes(ModuleOp module) {
  return PipeAnalysisBuilder(module).run();
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
