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

static std::optional<int64_t> getStaticMemDescBytes(Value value) {
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    elements *= dim;
  }
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 ||
      elements > std::numeric_limits<int64_t>::max() / bitWidth ||
      (elements * bitWidth) % 8 != 0)
    return std::nullopt;
  return elements * bitWidth / 8;
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

static bool hasPipeLifecycleDescendant(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (isPipeOp(nested))
      found = true;
  });
  return found;
}

static bool hasStructuredPipeAncestor(Operation *op) {
  if (!op)
    return false;
  return op->getParentOfType<scf::IfOp>() || op->getParentOfType<scf::ForOp>();
}

static void collectPipePathExits(Operation *anchor,
                                 SmallVectorImpl<Operation *> &exits) {
  if (!anchor)
    return;
  Region *region = anchor->getParentRegion();
  if (!region)
    return;
  for (Block &block : *region)
    if (Operation *terminator = block.getTerminator())
      exits.push_back(terminator);
}

static bool hasStructuredPipeAfter(Operation *op) {
  if (!op)
    return false;
  if (hasStructuredPipeAncestor(op))
    return true;
  Block *block = op->getBlock();
  if (!block)
    return false;
  // A lifecycle operation can be visited after a structured region (for
  // example a commit after an if whose acquire is branch-local).  Treat both
  // preceding and following structured regions as path-sensitive context so
  // we report the stable dominance/path diagnostic instead of a misleading
  // same-block matching error.
  for (Operation &sibling : *block) {
    if (&sibling == op)
      continue;
    if (isa<scf::IfOp, scf::ForOp>(&sibling) &&
        hasPipeLifecycleDescendant(&sibling))
      return true;
  }
  return false;
}

static bool isPipeOperationInRegion(Operation *op, Region &region) {
  return op && region.isAncestor(op->getParentRegion());
}

static scf::IfOp getNearestPipeIf(Operation *op) {
  return op ? op->getParentOfType<scf::IfOp>() : scf::IfOp();
}

static bool areIfAlternatives(Operation *lhs, Operation *rhs) {
  scf::IfOp lhsIf = getNearestPipeIf(lhs);
  scf::IfOp rhsIf = getNearestPipeIf(rhs);
  if (!lhsIf || !rhsIf || lhsIf != rhsIf)
    return false;
  bool lhsThen = lhsIf.getThenRegion().isAncestor(lhs->getParentRegion());
  bool rhsThen = lhsIf.getThenRegion().isAncestor(rhs->getParentRegion());
  bool lhsElse = lhsIf.elseBlock() &&
                 lhsIf.getElseRegion().isAncestor(lhs->getParentRegion());
  bool rhsElse = lhsIf.elseBlock() &&
                 lhsIf.getElseRegion().isAncestor(rhs->getParentRegion());
  return (lhsThen && rhsElse) || (lhsElse && rhsThen);
}

static bool isUniformPipeCondition(scf::IfOp ifOp) {
  Value condition = ifOp.getCondition();
  APInt constant;
  if (matchPattern(condition, m_ConstantInt(&constant)))
    return true;
  if (auto reduce = condition.getDefiningOp<tt::ReduceOp>()) {
    Operation *combiner = reduce.getSingleCombiner();
    if (combiner && isa<arith::OrIOp>(combiner))
      return true;
  }
  // Scalar values defined outside the structured operation are CTA-uniform
  // by construction.  Tensor predicates, and values produced in either arm,
  // are not safe for a barrier protocol.
  if (isa<RankedTensorType>(condition.getType()))
    return false;
  Operation *def = condition.getDefiningOp();
  return !def || !ifOp->isAncestor(def);
}

static FailureOr<int32_t> getOneShotStage(const PipeState &state, Value stage,
                                          Operation *operation) {
  APInt value;
  if (!matchPattern(stripIndexWrappers(stage), m_ConstantInt(&value)) ||
      !value.isSignedIntN(32)) {
    operation->emitOpError(
        "MUSA TLE one-shot pipe requires a statically known stage within "
        "capacity");
    return failure();
  }
  int64_t stageValue = value.getSExtValue();
  if (stageValue < 0 || stageValue >= state.capacity) {
    operation->emitOpError(
        "MUSA TLE one-shot pipe requires a statically known stage within "
        "capacity");
    return failure();
  }
  return static_cast<int32_t>(stageValue);
}

static LogicalResult verifyOneShotPhase(Value phase, Operation *operation) {
  APInt value;
  if (!matchPattern(stripIndexWrappers(phase), m_ConstantInt(&value)) ||
      !value.isZero())
    return operation->emitOpError(
        "MUSA TLE one-shot pipe does not support phase changes or stage "
        "reuse");
  return success();
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
  if (state.executionMode != PipeExecutionMode::Unset &&
      state.executionMode != mode)
    return op->emitOpError(
        "MUSA TLE pipe endpoint operations must use one execution mode and "
        "warp-specialize owner");
  if (mode == PipeExecutionMode::StaticWarpSpecialized) {
    if (state.staticWarpSpecialize &&
        state.staticWarpSpecialize != placement.owner)
      return op->emitOpError(
          "MUSA TLE pipe endpoint operations must use one execution mode and "
          "warp-specialize owner");
    state.staticWarpSpecialize = placement.owner;
  }
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

static LogicalResult verifyEndpointPlacement(PipeState &state,
                                             unsigned endpointIndex,
                                             Operation *operation,
                                             StringRef diagnostic) {
  if (endpointIndex >= state.endpoints.size())
    return operation->emitOpError(
        "internal MUSA TLE pipe analysis lost endpoint placement");
  FailureOr<PipeStaticPartitionInfo> placement =
      getEndpointPlacement(operation);
  if (failed(placement))
    return failure();
  if (!samePlacement(state.endpoints[endpointIndex], *placement))
    return operation->emitOpError(diagnostic);
  return success();
}

static std::optional<unsigned> findEndpoint(const PipeState &state,
                                            PipeEndpointRole role) {
  for (const PipeEndpointState &endpoint : state.endpoints) {
    if (endpoint.role == role)
      return endpoint.index;
  }
  return std::nullopt;
}

struct PipeFieldOwner {
  PipeState *pipe = nullptr;
  unsigned fieldIndex = 0;
};

struct ResolvedPipeFieldAccess {
  PipeState *pipe = nullptr;
  unsigned fieldIndex = 0;
  Value memdescRoot;
  Value stage;
  PipeCoveredRegion coveredRegion;
  bool exactWholeSlot = false;
};

struct PendingCompletionSource {
  PipeTransportKind kind = PipeTransportKind::Unknown;
  Operation *operation = nullptr;
  ResolvedPipeFieldAccess access;
  PipeBarrierStorageOwner barrierStorageOwner = PipeBarrierStorageOwner::Pipe;
  Value externalBarrierRoot;
};

struct OpenWriterGeneration {
  PipeState *pipe = nullptr;
  unsigned writerEndpoint = 0;
  Value stage;
  Value phase;
  PipeWriterAcquireOp acquire;
  SmallVector<PendingCompletionSource> completionSources;
};

struct OpenReaderGeneration {
  PipeState *pipe = nullptr;
  unsigned readerEndpoint = 0;
  Value stage;
  Value phase;
  PipeReaderWaitOp wait;
  SmallVector<PipeReaderDrainSource> drainSources;
  SmallVector<unsigned> modifiedFields;
  SmallVector<PipeCoveredRegion> modifiedRegions;
  Operation *firstModification = nullptr;
};

static void recordModifiedField(OpenReaderGeneration &generation,
                                unsigned fieldIndex, Operation *operation) {
  if (!llvm::is_contained(generation.modifiedFields, fieldIndex))
    generation.modifiedFields.push_back(fieldIndex);
  if (!generation.firstModification)
    generation.firstModification = operation;
}

static void recordModifiedRegion(OpenReaderGeneration &generation,
                                 const PipeCoveredRegion &region,
                                 Operation *operation) {
  recordModifiedField(generation, region.fieldIndex, operation);
  generation.modifiedRegions.push_back(region);
  if (!generation.firstModification)
    generation.firstModification = operation;
}

static FailureOr<int64_t> getStaticFieldBytes(const PipeState &state,
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

static bool isValidRegion(const PipeCoveredRegion &region) {
  return region.exact && region.byteOffset && region.byteSize &&
         *region.byteOffset >= 0 && *region.byteSize > 0 &&
         *region.byteOffset <=
             std::numeric_limits<int64_t>::max() - *region.byteSize;
}

static bool sameRegion(const PipeCoveredRegion &lhs,
                       const PipeCoveredRegion &rhs) {
  return lhs.fieldIndex == rhs.fieldIndex &&
         lhs.memdescRoot == rhs.memdescRoot &&
         lhs.byteOffset == rhs.byteOffset && lhs.byteSize == rhs.byteSize &&
         lhs.exact == rhs.exact;
}

static LogicalResult
verifyRegionCoverage(const PipeState &state, Operation *diagnosticOp,
                     ArrayRef<PipeCompletionSource> sources) {
  for (unsigned fieldIndex = 0; fieldIndex < state.fields.size();
       ++fieldIndex) {
    FailureOr<int64_t> fieldBytes = getStaticFieldBytes(state, fieldIndex);
    if (failed(fieldBytes))
      return diagnosticOp->emitOpError(
          "MUSA TLE pipe requires a statically provable contiguous field "
          "region");
    SmallVector<PipeByteInterval> intervals;
    std::optional<PipeTransportKind> fieldTransport;
    for (const PipeCompletionSource &source : sources) {
      if (source.destinationField != fieldIndex)
        continue;
      if (fieldTransport && *fieldTransport != source.kind)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe does not support mixed transport "
            "sources for one payload field");
      fieldTransport = source.kind;
      if (!isValidRegion(source.coveredRegion))
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe requires a statically provable contiguous field "
            "region");
      if (source.coveredRegion.memdescRoot !=
          state.fields[fieldIndex].memdescRoot)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe completion source has an invalid field root");
      intervals.push_back(
          {*source.coveredRegion.byteOffset, *source.coveredRegion.byteSize});
      if (source.kind == PipeTransportKind::TME &&
          source.transactionBytes != *source.coveredRegion.byteSize)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe TME transaction bytes must equal the covered "
            "region size");
    }
    if (intervals.empty())
      return diagnosticOp->emitOpError(
          "MUSA TLE pipe commit does not cover every payload field region");
    llvm::sort(intervals,
               [](const PipeByteInterval &lhs, const PipeByteInterval &rhs) {
                 return lhs.byteOffset < rhs.byteOffset;
               });
    int64_t cursor = 0;
    for (size_t index = 0; index < intervals.size(); ++index) {
      const PipeByteInterval &interval = intervals[index];
      if (interval.byteOffset < cursor)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe completion sources for one field must not "
            "overlap");
      if (interval.byteOffset != cursor)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe commit does not cover every payload field region");
      if (interval.byteSize > std::numeric_limits<int64_t>::max() - cursor)
        return diagnosticOp->emitOpError(
            "MUSA TLE pipe requires a statically provable contiguous field "
            "region");
      cursor += interval.byteSize;
    }
    if (cursor != *fieldBytes)
      return diagnosticOp->emitOpError(
          "MUSA TLE pipe commit does not cover every payload field region");
  }
  return success();
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

static LogicalResult verifyAnalysisState(PipeAnalysisResult &result) {
  for (Operation *op : result.getLifecycleOps()) {
    if (!result.lookupPipe(op))
      return op->emitOpError(
          "internal MUSA TLE pipe analysis lost lifecycle ownership");
    if (!isa<PipeCreateOp>(op) && !result.lookupEndpoint(op))
      return op->emitOpError(
          "internal MUSA TLE pipe analysis lost endpoint ownership");
  }

  for (const std::unique_ptr<PipeState> &ownedState : result.getPipes()) {
    PipeState &state = *ownedState;
    bool oneShot = state.lifecycle.mode == PipeLifecycleMode::OneShot;
    bool closeOnly =
        !state.closeGenerations.empty() && state.commitGroups.empty();
    auto declaredReaders = state.create->getAttrOfType<ArrayAttr>("readers");
    size_t expectedReaderCount = declaredReaders ? declaredReaders.size() : 1;
    if (state.fields.empty() ||
        state.endpoints.size() != expectedReaderCount + 1 ||
        state.endpoints[0].role != PipeEndpointRole::Writer ||
        state.endpoints[0].index != 0 || state.endpoints[0].name != "writer" ||
        state.endpoints[0].readerSubscription.has_value() ||
        (oneShot ? state.barrierPlan.empty.has_value()
                 : !state.barrierPlan.empty.has_value()) ||
        state.lifecycle.stagePhasePolicy !=
            (oneShot ? PipePhasePolicy::OneShotFixed
                     : PipePhasePolicy::CyclicAlternating) ||
        state.barrierPlan.phasePolicy != state.lifecycle.stagePhasePolicy ||
        (!state.barrierPlan.full.transactionBytes && !closeOnly))
      return state.create.getOperation()->emitOpError(
          "internal MUSA TLE pipe analysis produced an incomplete plan");

    std::optional<unsigned> writerEndpoint =
        findEndpoint(state, PipeEndpointRole::Writer);
    if (!writerEndpoint || *writerEndpoint >= state.endpoints.size())
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis lost writer endpoint");

    int64_t readerWarps = 0;
    if (state.partitionMapping.size() != state.endpoints.size())
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an incomplete endpoint "
          "mapping");
    for (unsigned endpointIndex = 0; endpointIndex < state.endpoints.size();
         ++endpointIndex) {
      const PipeEndpointState &endpoint = state.endpoints[endpointIndex];
      const PipePartitionMapping &mapping =
          state.partitionMapping[endpointIndex];
      if (endpoint.index != endpointIndex ||
          mapping.endpoint != endpointIndex ||
          endpoint.warpSpecialize != mapping.warpSpecialize ||
          endpoint.partitionIndex != mapping.partitionIndex ||
          endpoint.partition != mapping.partition ||
          endpoint.worker != mapping.worker ||
          endpoint.warpBegin != mapping.warpBegin ||
          endpoint.warpCount != mapping.warpCount)
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an unstable endpoint "
            "mapping");
      if (endpointIndex == 0)
        continue;
      StringRef expectedName =
          declaredReaders
              ? cast<StringAttr>(declaredReaders[endpointIndex - 1]).getValue()
              : StringRef();
      if (endpoint.role != PipeEndpointRole::Reader ||
          endpoint.name != expectedName ||
          !endpoint.readerSubscription.has_value() ||
          endpoint.subscribedFields.empty() || endpoint.warpCount <= 0)
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an invalid reader "
            "endpoint");
      unsigned previousField = 0;
      bool firstField = true;
      for (unsigned fieldIndex : endpoint.subscribedFields) {
        if (fieldIndex >= state.fields.size() ||
            (!firstField && fieldIndex <= previousField))
          return state.create.emitOpError(
              "internal MUSA TLE pipe analysis produced an invalid field "
              "subscription");
        firstField = false;
        previousField = fieldIndex;
      }
      bool subscribesAllFields =
          endpoint.subscribedFields.size() == state.fields.size();
      if ((*endpoint.readerSubscription ==
           PipeReaderSubscriptionKind::AllFields) != subscribesAllFields)
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an inconsistent field "
            "subscription");
      readerWarps += endpoint.warpCount;
    }
    if (readerWarps <= 0 ||
        (!oneShot && readerWarps != state.barrierPlan.empty->arrivalCount))
      return state.create.getOperation()->emitOpError(
          "internal MUSA TLE pipe analysis produced an invalid empty "
          "barrier count");

    bool externalFull = state.barrierPlan.full.storageOwner ==
                        PipeBarrierStorageOwner::External;
    if (externalFull != (state.barrierPlan.fullBarrierStorageOwner ==
                         PipeBarrierStorageOwner::External) ||
        (externalFull && (!state.barrierPlan.externalFull ||
                          !state.barrierPlan.full.externalStorage ||
                          state.barrierPlan.externalFull->base !=
                              state.barrierPlan.full.externalStorage)) ||
        (!externalFull && (state.barrierPlan.externalFull ||
                           state.barrierPlan.full.externalStorage)))
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an invalid external "
          "completion barrier plan");

    if (!state.barrierPlan.writerParticipant ||
        state.barrierPlan.writerParticipant->endpointIndex != 0 ||
        state.barrierPlan.readerParticipants.size() != expectedReaderCount)
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an incomplete barrier "
          "participant ledger");
    unsigned readerParticipant = 0;
    for (const PipeEndpointState &endpoint : state.endpoints) {
      if (endpoint.role != PipeEndpointRole::Reader)
        continue;
      const PipeBarrierParticipant &participant =
          state.barrierPlan.readerParticipants[readerParticipant++];
      if (participant.endpointIndex != endpoint.index ||
          participant.partitionIndex != endpoint.partitionIndex ||
          participant.partition != endpoint.partition ||
          participant.warpBegin != endpoint.warpBegin ||
          participant.warpCount != endpoint.warpCount)
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an unstable barrier "
            "participant ledger");
    }
    const PipeEndpointState &writer = state.endpoints[*writerEndpoint];
    if (state.barrierPlan.writerParticipant->partitionIndex !=
            writer.partitionIndex ||
        state.barrierPlan.writerParticipant->partition != writer.partition ||
        state.barrierPlan.writerParticipant->warpBegin != writer.warpBegin ||
        state.barrierPlan.writerParticipant->warpCount != writer.warpCount)
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an unstable writer "
          "barrier participant");

    for (const PipeFieldState &field : state.fields) {
      SmallVector<unsigned> expectedSubscribers;
      for (const PipeEndpointState &endpoint : state.endpoints) {
        if (endpoint.role == PipeEndpointRole::Reader &&
            llvm::is_contained(endpoint.subscribedFields, field.index))
          expectedSubscribers.push_back(endpoint.index);
      }
      if (field.subscribedReaders != expectedSubscribers)
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an inconsistent "
            "reader subscription index");
    }

    SmallVector<int32_t> verifiedOneShotStages;
    for (const std::unique_ptr<PipeCommitGroup> &ownedGroup :
         state.commitGroups) {
      PipeCommitGroup &group = *ownedGroup;
      bool externalFull = state.barrierPlan.full.storageOwner ==
                          PipeBarrierStorageOwner::External;
      if (oneShot) {
        FailureOr<int32_t> stage =
            getOneShotStage(state, group.stage, group.commit);
        if (failed(stage) ||
            failed(verifyOneShotPhase(group.phase, group.commit)))
          return failure();
        if (llvm::is_contained(verifiedOneShotStages, *stage))
          return group.commit.emitOpError(
              "MUSA TLE one-shot pipe stage may be published at most once");
        verifiedOneShotStages.push_back(*stage);
      }
      int64_t totalBytes = 0;
      bool hasTME = false;
      bool hasLocalStore = false;
      bool hasAsyncCopy = false;
      for (const PipeCompletionSource &source : group.completionSources) {
        if (!source.operation ||
            source.destinationField >= state.fields.size() ||
            !equivalentPipeValue(source.stage, group.stage) ||
            !equivalentPipeValue(source.phase, group.phase) ||
            source.coveredRegion.fieldIndex != source.destinationField ||
            source.coveredRegion.memdescRoot !=
                state.fields[source.destinationField].memdescRoot ||
            !isValidRegion(source.coveredRegion) ||
            (source.kind == PipeTransportKind::TME &&
             (source.barrierStorageOwner ==
              PipeBarrierStorageOwner::External) != externalFull) ||
            (source.kind == PipeTransportKind::TME && externalFull &&
             source.externalBarrierRoot !=
                 state.barrierPlan.full.externalStorage) ||
            ((source.kind != PipeTransportKind::TME || !externalFull) &&
             source.externalBarrierRoot))
          return group.commit.getOperation()->emitOpError(
              "internal MUSA TLE pipe analysis produced an invalid "
              "completion source");
        if (!result.lookupLogicalGeneration(source.operation) ||
            !group.logicalGeneration ||
            result.lookupCompletionSourceGroup(source.operation) != &group)
          return group.commit.getOperation()->emitOpError(
              "internal MUSA TLE pipe analysis lost logical completion "
              "generation");
        if (source.kind == PipeTransportKind::TME) {
          if (!isa<ttg::TMACopyOp>(source.operation) ||
              source.transactionBytes <= 0 ||
              source.coveredRegion.byteSize !=
                  std::optional<int64_t>(source.transactionBytes))
            return group.commit.getOperation()->emitOpError(
                "internal MUSA TLE pipe analysis produced an invalid TME "
                "completion source");
          hasTME = true;
        } else if (source.kind == PipeTransportKind::LocalStore) {
          std::optional<LocalStoreTarget> target =
              getLocalStoreTarget(source.operation);
          if (!target || source.transactionBytes != 0)
            return group.commit.getOperation()->emitOpError(
                "internal MUSA TLE pipe analysis produced an invalid "
                "local-store completion source");
          FailureOr<int64_t> fieldBytes =
              getStaticFieldBytes(state, source.destinationField);
          if (!target->exactWholeField ||
              source.coveredRegion.byteOffset != std::optional<int64_t>(0) ||
              failed(fieldBytes) || !source.coveredRegion.byteSize ||
              *source.coveredRegion.byteSize != *fieldBytes)
            return group.commit.getOperation()->emitOpError(
                "internal MUSA TLE pipe analysis produced an invalid "
                "local-store completion source");
          hasLocalStore = true;
        } else if (source.kind == PipeTransportKind::AsyncCopy) {
          if (!isa<ttg::AsyncCopyGlobalToLocalOp>(source.operation) ||
              source.transactionBytes != 0)
            return group.commit.getOperation()->emitOpError(
                "internal MUSA TLE pipe analysis produced an invalid "
                "async-copy completion source");
          hasAsyncCopy = true;
        } else {
          return group.commit.getOperation()->emitOpError(
              "internal MUSA TLE pipe analysis produced an unknown "
              "completion source");
        }
        if (totalBytes > std::numeric_limits<int32_t>::max() -
                             static_cast<int64_t>(source.transactionBytes))
          return group.commit.getOperation()->emitOpError(
              "MUSA TLE pipe aggregate TME transaction bytes exceed the "
              "positive i32 range");
        totalBytes += source.transactionBytes;
      }
      if (failed(verifyRegionCoverage(state, group.commit.getOperation(),
                                      group.completionSources)))
        return failure();
      int32_t writerWarps = state.barrierPlan.writerParticipant->warpCount;
      int32_t expectedTMEArrivals = hasTME ? 1 : 0;
      int32_t expectedLocalArrivals =
          (hasLocalStore || hasAsyncCopy) ? writerWarps : 0;
      if (totalBytes != group.totalTransactionBytes || totalBytes < 0 ||
          (!hasTME && !hasLocalStore && !hasAsyncCopy) ||
          (!hasTME && !hasAsyncCopy && hasLocalStore &&
           state.fields.size() != 1) ||
          group.tmeGroupArrivalCount != expectedTMEArrivals ||
          group.localStoreArrivalCount != expectedLocalArrivals ||
          group.fullArrivalCount !=
              group.tmeGroupArrivalCount + group.localStoreArrivalCount ||
          group.totalTransactionBytes !=
              *state.barrierPlan.full.transactionBytes ||
          group.externalBarrierRoot != state.barrierPlan.full.externalStorage ||
          group.fullArrivalCount != state.barrierPlan.full.arrivalCount)
        return group.commit.getOperation()->emitOpError(
            "internal MUSA TLE pipe analysis produced inconsistent "
            "completion accounting");
      if (result.lookupCommitGroup(group.commit) != &group)
        return group.commit.getOperation()->emitOpError(
            "internal MUSA TLE pipe analysis lost commit ownership");
    }

    for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
         state.readerDrainGroups) {
      PipeReaderDrainGroup &group = *ownedGroup;
      PipeReaderIssuePublicationPolicy expectedPolicy =
          state.executionMode == PipeExecutionMode::StaticWarpSpecialized
              ? PipeReaderIssuePublicationPolicy::PipeFullWait
              : PipeReaderIssuePublicationPolicy::NonWarpSpecializedCTA;
      Operation *groupOperation = group.release ? group.release.getOperation()
                                                : group.wait.getOperation();
      if (oneShot) {
        FailureOr<int32_t> stage =
            getOneShotStage(state, group.stage, groupOperation);
        if (failed(stage))
          return failure();
        if (!llvm::is_contained(state.oneShotPublishedStages, *stage))
          return groupOperation->emitOpError(
              "MUSA TLE one-shot pipe reader.wait requires a published "
              "stage");
      }
      if (group.readerEndpoint >= state.endpoints.size() ||
          state.endpoints[group.readerEndpoint].role !=
              PipeEndpointRole::Reader ||
          !result.lookupEndpoint(group.wait) ||
          result.lookupEndpoint(group.wait)->index != group.readerEndpoint ||
          (group.release && (!result.lookupEndpoint(group.release) ||
                             result.lookupEndpoint(group.release)->index !=
                                 group.readerEndpoint)) ||
          group.issuePublicationPolicy != expectedPolicy ||
          !group.sourceModifiedAfterWait.has_value() ||
          *group.sourceModifiedAfterWait ||
          result.lookupReaderDrainGroup(group.wait) != &group ||
          (group.release
               ? result.lookupReaderDrainGroup(group.release) != &group
               : !oneShot))
        return groupOperation->emitOpError(
            "internal MUSA TLE pipe analysis produced an invalid reader "
            "drain group");

      for (const PipeReaderDrainSource &source : group.drainSources) {
        auto copy = dyn_cast_or_null<ttg::TMACopyOp>(source.operation);
        if (!copy || source.kind != PipeReaderDrainKind::TMEStore ||
            !isa<ttg::MemDescType>(copy.getSrc().getType()) ||
            !isa<tt::TensorDescType>(copy.getDst().getType()) ||
            source.sourceField >= state.fields.size() ||
            !llvm::is_contained(
                state.endpoints[group.readerEndpoint].subscribedFields,
                source.sourceField) ||
            source.coveredRegion.fieldIndex != source.sourceField ||
            source.coveredRegion.memdescRoot !=
                state.fields[source.sourceField].memdescRoot ||
            !isValidRegion(source.coveredRegion) ||
            source.destinationDescriptor != copy.getDst())
          return groupOperation->emitOpError(
              "internal MUSA TLE pipe analysis produced an invalid reader "
              "drain source");
        if (result.lookupReaderDrainSourceGroup(source.operation) != &group)
          return groupOperation->emitOpError(
              "internal MUSA TLE pipe analysis lost reader drain "
              "ownership");
      }
      if (!group.logicalGeneration ||
          result.lookupLogicalGeneration(group.wait) !=
              group.logicalGeneration ||
          (group.release && result.lookupLogicalGeneration(group.release) !=
                                group.logicalGeneration))
        return groupOperation->emitOpError(
            "internal MUSA TLE pipe analysis lost logical reader generation");
    }

    if (oneShot &&
        (verifiedOneShotStages.size() != state.oneShotPublishedStages.size() ||
         llvm::any_of(verifiedOneShotStages, [&](int32_t stage) {
           return !llvm::is_contained(state.oneShotPublishedStages, stage);
         })))
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an invalid one-shot "
          "publication plan");

    size_t unreleasedCloseWaits = llvm::count_if(
        state.closeWaits, [](const auto &wait) { return !wait->release; });
    if (state.waits.size() !=
        state.readerDrainGroups.size() + unreleasedCloseWaits)
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an unmatched reader "
          "generation");
    llvm::DenseSet<unsigned> closeWaitEndpoints;
    for (const std::unique_ptr<PipeReaderCloseWait> &ownedWait :
         state.closeWaits) {
      PipeReaderCloseWait &closeWait = *ownedWait;
      PipeReaderDrainGroup *releaseGroup =
          closeWait.release ? result.lookupReaderDrainGroup(closeWait.release)
                            : nullptr;
      if (!closeWaitEndpoints.insert(closeWait.readerEndpoint).second ||
          closeWait.readerEndpoint >= state.endpoints.size() ||
          state.endpoints[closeWait.readerEndpoint].role !=
              PipeEndpointRole::Reader ||
          !closeWait.logicalGeneration ||
          result.lookupLogicalGeneration(closeWait.wait) !=
              closeWait.logicalGeneration ||
          !result.lookupEndpoint(closeWait.wait) ||
          result.lookupEndpoint(closeWait.wait)->index !=
              closeWait.readerEndpoint ||
          (closeWait.release &&
           (!result.lookupEndpoint(closeWait.release) ||
            result.lookupEndpoint(closeWait.release)->index !=
                closeWait.readerEndpoint)) ||
          state.closeGenerations.size() != 1 ||
          !sameIndex(closeWait.stage, state.closeGenerations.front()->stage) ||
          !sameIndex(closeWait.phase, state.closeGenerations.front()->phase) ||
          result.lookupCloseWait(closeWait.wait) != &closeWait ||
          (closeWait.release &&
           (!releaseGroup || releaseGroup->wait != closeWait.wait ||
            !releaseGroup->drainSources.empty())) ||
          (!closeWait.release && result.lookupReaderDrainGroup(closeWait.wait)))
        return closeWait.wait.emitOpError(
            "internal MUSA TLE pipe analysis produced an invalid terminal "
            "close wait");
    }

    if (!state.closeGenerations.empty() &&
        closeWaitEndpoints.size() != expectedReaderCount)
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an incomplete terminal "
          "close broadcast");

    if (oneShot && !state.closeWaits.empty())
      return state.create.emitOpError(
          "internal MUSA TLE pipe analysis produced an unexpected one-shot "
          "close wait");

    if (state.closeGenerations.empty()) {
      if (state.barrierPlan.hasCloseState || state.barrierPlan.closeTagPlan ||
          state.lifecycle.closePolicy != PipeClosePolicy::Unsupported)
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an unexpected close "
            "state");
    } else {
      if (!state.barrierPlan.hasCloseState || !state.barrierPlan.closeTagPlan ||
          state.barrierPlan.closeTagPlan->capacity != state.capacity ||
          state.barrierPlan.closeTagPlan->initialValue ||
          state.barrierPlan.closeTagPlan->storageOwner !=
              PipeBarrierStorageOwner::Pipe ||
          state.lifecycle.closePolicy != PipeClosePolicy::TaggedBroadcast ||
          state.closeGenerations.size() != 1 ||
          state.closeGenerations.front()->transactionBytes != 0 ||
          !state.closeGenerations.front()->implicitEmptyAcquire ||
          result.lookupCloseGeneration(state.closeGenerations.front()->close) !=
              state.closeGenerations.front().get())
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced an invalid close "
            "plan");

      PipeCloseGeneration &close = *state.closeGenerations.front();
      bool hasTME = false;
      bool hasLocalStore = false;
      for (const auto &group : state.commitGroups) {
        hasTME |= group->tmeGroupArrivalCount != 0;
        hasLocalStore |= group->localStoreArrivalCount != 0;
      }
      int32_t writerWarps = state.barrierPlan.writerParticipant->warpCount;
      int32_t expectedControl = (hasTME || !hasLocalStore) ? 1 : 0;
      int32_t expectedLocal = hasLocalStore ? writerWarps : 0;
      if (close.controlArrivalCount != expectedControl ||
          close.localStoreArrivalCount != expectedLocal ||
          close.fullArrivalCount != expectedControl + expectedLocal ||
          close.stage != close.close.getStage() ||
          close.phase != close.close.getPhase() ||
          (!state.commitGroups.empty() &&
           close.fullArrivalCount != state.barrierPlan.full.arrivalCount))
        return close.close.emitOpError(
            "MUSA TLE pipe close plan has inconsistent arrival shape");
    }
  }
  return success();
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

PipeCommitGroup *PipeAnalysisResult::lookupCommitGroup(PipeWriterCommitOp op) {
  return commitGroupByOperation.lookup(op.getOperation());
}

const PipeCommitGroup *
PipeAnalysisResult::lookupCommitGroup(PipeWriterCommitOp op) const {
  return commitGroupByOperation.lookup(op.getOperation());
}

PipeCommitGroup *
PipeAnalysisResult::lookupCompletionSourceGroup(Operation *op) {
  return completionSourceGroupByOperation.lookup(op);
}

const PipeCommitGroup *
PipeAnalysisResult::lookupCompletionSourceGroup(Operation *op) const {
  return completionSourceGroupByOperation.lookup(op);
}

PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainGroup(PipeReaderReleaseOp op) {
  return readerDrainGroupByOperation.lookup(op.getOperation());
}

const PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainGroup(PipeReaderReleaseOp op) const {
  return readerDrainGroupByOperation.lookup(op.getOperation());
}

PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainGroup(PipeReaderWaitOp op) {
  return readerDrainGroupByWait.lookup(op.getOperation());
}

const PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainGroup(PipeReaderWaitOp op) const {
  return readerDrainGroupByWait.lookup(op.getOperation());
}

PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainSourceGroup(Operation *op) {
  return readerDrainGroupBySource.lookup(op);
}

const PipeReaderDrainGroup *
PipeAnalysisResult::lookupReaderDrainSourceGroup(Operation *op) const {
  return readerDrainGroupBySource.lookup(op);
}

PipeCloseGeneration *
PipeAnalysisResult::lookupCloseGeneration(PipeWriterCloseOp op) {
  return closeGenerationByOperation.lookup(op.getOperation());
}

const PipeCloseGeneration *
PipeAnalysisResult::lookupCloseGeneration(PipeWriterCloseOp op) const {
  return closeGenerationByOperation.lookup(op.getOperation());
}

PipeReaderCloseWait *PipeAnalysisResult::lookupCloseWait(PipeReaderWaitOp op) {
  return closeWaitByOperation.lookup(op.getOperation());
}

const PipeReaderCloseWait *
PipeAnalysisResult::lookupCloseWait(PipeReaderWaitOp op) const {
  return closeWaitByOperation.lookup(op.getOperation());
}

PipeLogicalGeneration *
PipeAnalysisResult::lookupLogicalGeneration(Operation *op) {
  return logicalGenerationByOperation.lookup(op);
}

const PipeLogicalGeneration *
PipeAnalysisResult::lookupLogicalGeneration(Operation *op) const {
  return logicalGenerationByOperation.lookup(op);
}

class PipeAnalysisBuilder {
public:
  explicit PipeAnalysisBuilder(ModuleOp module)
      : module(module), dominance(module), postDominance(module) {}

  FailureOr<std::unique_ptr<PipeAnalysisResult>> run() {
    if (failed(collectLifecycleOps()) || failed(createPipeDefinitions()) ||
        failed(bindLifecycleOwnership()) || failed(initializeEndpoints()) ||
        failed(analyzeLifecycleBlocks()) ||
        failed(finalizeUnmatchedReaderGenerations()) ||
        failed(classifyTerminalCloseWaits()) ||
        failed(validateStructuredControlFlow()) ||
        failed(validatePathCompleteness()) ||
        failed(buildLogicalGenerations()) ||
        failed(validateComplexLifecyclePaths()) ||
        failed(finalizePipeStates()) ||
        failed(validateNamedReaderGenerations()) ||
        failed(validateExternalBarrierUses()) ||
        failed(verifyAnalysisState(*result)))
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
      state->lifecycle = {oneShot ? PipeLifecycleMode::OneShot
                                  : PipeLifecycleMode::Cyclic,
                          PipeClosePolicy::Unsupported,
                          oneShot ? PipePhasePolicy::OneShotFixed
                                  : PipePhasePolicy::CyclicAlternating};
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
      state->barrierPlan.fullBarrierStorageOwner =
          PipeBarrierStorageOwner::Pipe;
      state->barrierPlan.phasePolicy = oneShot
                                           ? PipePhasePolicy::OneShotFixed
                                           : PipePhasePolicy::CyclicAlternating;

      auto fieldNames = create->getAttrOfType<ArrayAttr>("field_names");
      for (auto [index, field] : llvm::enumerate(create.getFields())) {
        auto name = cast<StringAttr>(fieldNames[index]).getValue().str();
        state->fields.push_back(PipeFieldState{static_cast<unsigned>(index),
                                               name,
                                               field,
                                               getMemDescRoot(field),
                                               field.getType(),
                                               PipeTransportKind::Unknown,
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

    if (placement->kind != PipePartitionKind::CTA) {
      for (const PipeEndpointState &existing : state.endpoints) {
        if (existing.partitionIndex == placement->partitionIndex)
          return op->emitOpError(
              "MUSA TLE static warp-specialized pipe partitions must host at "
              "most one pipe endpoint");
      }
    }
    state.endpoints.push_back(PipeEndpointState{
        expectedIndex, name.str(), endpointRole, readerSubscription,
        SmallVector<unsigned>(subscribedFields.begin(), subscribedFields.end()),
        placement->owner, placement->partitionIndex, placement->kind,
        placement->workerIndex, placement->warpBegin, placement->warpCount});
    state.partitionMapping.push_back(PipePartitionMapping{
        expectedIndex, placement->owner, placement->partitionIndex,
        placement->kind, placement->workerIndex, placement->warpBegin,
        placement->warpCount});
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

  LogicalResult buildFieldOwnerIndex() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      for (const PipeFieldState &field : state.fields)
        fieldOwnersByRoot[field.memdescRoot].push_back(
            PipeFieldOwner{&state, field.index});
    }
    return success();
  }

  FailureOr<SmallVector<ResolvedPipeFieldAccess>>
  resolvePipeFieldAccess(Operation *operation, Value memdesc) {
    FailureOr<PipeResolvedRegion> resolved = resolvePipeMemDescRegion(memdesc);
    Value root;
    Value stage;
    PipeByteInterval interval;
    bool exact = false;
    if (succeeded(resolved)) {
      root = resolved->memdescRoot;
      stage = resolved->stage;
      interval = resolved->interval;
      exact = resolved->exact;
    } else {
      // A shared value which is not one of this pipe's roots must remain
      // invisible to pipe analysis.  Only issue the contiguous-region
      // diagnostic once a root owner is found below.
      root = getMemDescRoot(memdesc);
    }
    auto owners = fieldOwnersByRoot.lookup(root);
    if (owners.empty())
      return SmallVector<ResolvedPipeFieldAccess>{};
    if (failed(resolved)) {
      operation->emitOpError(
          "MUSA TLE pipe requires a statically provable contiguous field "
          "region");
      return failure();
    }
    SmallVector<ResolvedPipeFieldAccess> accesses;
    accesses.reserve(owners.size());
    for (const PipeFieldOwner &owner : owners) {
      FailureOr<int64_t> fieldBytes =
          getStaticFieldBytes(*owner.pipe, owner.fieldIndex);
      if (!isValidRegion(PipeCoveredRegion{owner.fieldIndex, root,
                                           interval.byteOffset,
                                           interval.byteSize, exact}) ||
          failed(fieldBytes) || interval.byteOffset > *fieldBytes ||
          interval.byteSize > *fieldBytes - interval.byteOffset)
        return operation->emitOpError(
            "MUSA TLE pipe requires a statically provable contiguous field "
            "region");
      bool exactWholeSlot = exact && interval.byteOffset == 0 &&
                            succeeded(fieldBytes) &&
                            *fieldBytes == interval.byteSize;
      accesses.push_back(ResolvedPipeFieldAccess{
          owner.pipe, owner.fieldIndex, root, stage,
          PipeCoveredRegion{owner.fieldIndex, root, interval.byteOffset,
                            interval.byteSize, exact},
          exactWholeSlot});
    }
    return accesses;
  }

  static std::optional<Value> getReaderMutationMemDesc(Operation *operation) {
    if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(operation))
      return asyncCopy.getResult();

    if (auto copy = dyn_cast<ttg::TMACopyOp>(operation)) {
      bool globalToLocal = isa<tt::TensorDescType>(copy.getSrc().getType()) &&
                           isa<ttg::MemDescType>(copy.getDst().getType());
      if (globalToLocal)
        return copy.getDst();
      return std::nullopt;
    }

    if (std::optional<LocalStoreTarget> target = getLocalStoreTarget(operation))
      return target->memdesc;

    if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(operation))
      return dealloc.getSrc();

    // Keep a conservative fallback for TLE operations which expose their
    // shared-memory write/free through MemoryEffectOpInterface instead of one
    // of the concrete store/dealloc ops above.  A read-only effect (including
    // the reader-side TME store itself) is deliberately ignored.
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const MemoryEffects::EffectInstance &instance : instances) {
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(
                instance.getEffect()))
          continue;
        Value value = instance.getValue();
        if (value && isa<ttg::MemDescType>(value.getType()))
          return value;
      }
    }

    return std::nullopt;
  }

  bool
  hasOpenReaderForMemDesc(Value memdesc,
                          SmallVectorImpl<OpenReaderGeneration> &openReaders) {
    Value root = getMemDescRoot(memdesc);
    auto owners = fieldOwnersByRoot.lookup(root);
    if (owners.empty())
      return false;
    std::optional<Value> stage = getSingleStageIndex(memdesc);
    for (const OpenReaderGeneration &generation : openReaders) {
      if (!generation.pipe ||
          generation.readerEndpoint >= generation.pipe->endpoints.size())
        continue;
      for (const PipeFieldOwner &owner : owners) {
        if (owner.pipe != generation.pipe ||
            !llvm::is_contained(
                generation.pipe->endpoints[generation.readerEndpoint]
                    .subscribedFields,
                owner.fieldIndex))
          continue;
        if (!stage || sameIndex(*stage, generation.stage))
          return true;
      }
    }
    return false;
  }

  static bool operationBeforeRelease(Operation *operation,
                                     PipeReaderReleaseOp release) {
    if (!release)
      return true;
    if (operation->getBlock() != release.getOperation()->getBlock())
      return true;
    return operation->isBeforeInBlock(release.getOperation());
  }

  PipeReaderDrainGroup *
  findDominatingReaderDrainGroup(const ResolvedPipeFieldAccess &access,
                                 Operation *operation,
                                 const PipeStaticPartitionInfo &placement) {
    for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
         access.pipe->readerDrainGroups) {
      PipeReaderDrainGroup *group = ownedGroup.get();
      if (group->readerEndpoint >= access.pipe->endpoints.size() ||
          !equivalentPipeValue(group->stage, access.stage) ||
          !dominance.dominates(group->wait.getOperation(), operation) ||
          !operationBeforeRelease(operation, group->release) ||
          (group->release && !postDominance.postDominates(
                                 group->release.getOperation(), operation)))
        continue;
      const PipeEndpointState &endpoint =
          access.pipe->endpoints[group->readerEndpoint];
      if (!llvm::is_contained(endpoint.subscribedFields, access.fieldIndex) ||
          !samePlacement(endpoint, placement))
        continue;
      return group;
    }
    return nullptr;
  }

  static void appendModifiedRegion(PipeReaderDrainGroup &group,
                                   const PipeCoveredRegion &region,
                                   Operation *operation) {
    group.modifiedRegions.push_back(region);
    for (const PipeReaderDrainSource &source : group.drainSources) {
      if (source.sourceField != region.fieldIndex ||
          source.coveredRegion.memdescRoot != region.memdescRoot)
        continue;
      bool overlaps = !region.exact || !source.coveredRegion.exact ||
                      !region.byteOffset || !region.byteSize ||
                      !source.coveredRegion.byteOffset ||
                      !source.coveredRegion.byteSize ||
                      intervalsOverlap({*region.byteOffset, *region.byteSize},
                                       {*source.coveredRegion.byteOffset,
                                        *source.coveredRegion.byteSize});
      if (overlaps) {
        group.sourceModifiedAfterWait = true;
        break;
      }
    }
    (void)operation;
  }

  LogicalResult
  recordReaderMutation(Operation *operation,
                       SmallVectorImpl<OpenReaderGeneration> &openReaders) {
    std::optional<Value> memdesc = getReaderMutationMemDesc(operation);
    if (!memdesc)
      return success();

    Value root = getMemDescRoot(*memdesc);
    auto owners = fieldOwnersByRoot.lookup(root);
    if (owners.empty())
      return success();

    if (isa<ttg::LocalDeallocOp>(operation) &&
        llvm::any_of(owners, [](const PipeFieldOwner &owner) {
          return owner.pipe &&
                 owner.pipe->lifecycle.mode == PipeLifecycleMode::OneShot;
        }))
      return operation->emitOpError(
          "MUSA TLE one-shot pipe payload is immutable after publication");

    FailureOr<PipeResolvedRegion> resolvedRegion =
        resolvePipeMemDescRegion(*memdesc);
    Value resolvedStage;
    PipeByteInterval resolvedInterval;
    bool hasResolvedRegion = succeeded(resolvedRegion);
    if (hasResolvedRegion) {
      resolvedStage = resolvedRegion->stage;
      resolvedInterval = resolvedRegion->interval;
    }
    for (OpenReaderGeneration &generation : openReaders) {
      if (!generation.pipe)
        continue;
      for (const PipeFieldOwner &owner : owners) {
        if (generation.readerEndpoint >= generation.pipe->endpoints.size() ||
            owner.pipe != generation.pipe ||
            owner.fieldIndex >= generation.pipe->fields.size() ||
            !llvm::is_contained(
                generation.pipe->endpoints[generation.readerEndpoint]
                    .subscribedFields,
                owner.fieldIndex))
          continue;
        if (hasResolvedRegion && !sameIndex(resolvedStage, generation.stage))
          continue;
        if (failed(verifyEndpointPlacement(
                *generation.pipe, generation.readerEndpoint, operation,
                "MUSA TLE pipe reader payload operation must execute in the "
                "reader partition")))
          return failure();
        PipeCoveredRegion region{
            owner.fieldIndex, owner.pipe->fields[owner.fieldIndex].memdescRoot,
            std::nullopt, std::nullopt, false};
        if (hasResolvedRegion) {
          region.byteOffset = resolvedInterval.byteOffset;
          region.byteSize = resolvedInterval.byteSize;
          region.exact = resolvedRegion->exact;
        }
        recordModifiedRegion(generation, region, operation);
      }
    }

    // A wait in an outer block can dominate a mutation nested in a structured
    // region.  The concrete drain group may already have been finalized by
    // the linear pass; retain the same path fact on that group so later
    // verification and lowering see the mutation as part of the wait window.
    for (const PipeFieldOwner &owner : owners) {
      PipeState *pipe = owner.pipe;
      for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
           pipe->readerDrainGroups) {
        PipeReaderDrainGroup &group = *ownedGroup;
        if (!dominance.dominates(group.wait.getOperation(), operation) ||
            !operationBeforeRelease(operation, group.release) ||
            group.readerEndpoint >= pipe->endpoints.size() ||
            !llvm::is_contained(
                pipe->endpoints[group.readerEndpoint].subscribedFields,
                owner.fieldIndex))
          continue;
        if (hasResolvedRegion &&
            !equivalentPipeValue(resolvedStage, group.stage))
          continue;
        if (failed(verifyEndpointPlacement(
                *pipe, group.readerEndpoint, operation,
                "MUSA TLE pipe reader payload operation must execute in the "
                "reader partition")))
          return failure();
        PipeCoveredRegion region{owner.fieldIndex,
                                 pipe->fields[owner.fieldIndex].memdescRoot,
                                 std::nullopt, std::nullopt, false};
        if (hasResolvedRegion) {
          region.byteOffset = resolvedInterval.byteOffset;
          region.byteSize = resolvedInterval.byteSize;
          region.exact = resolvedRegion->exact;
        }
        appendModifiedRegion(group, region, operation);
        if (group.sourceModifiedAfterWait.value_or(false))
          return operation->emitOpError(
              "MUSA TLE pipe reader TME store source must not be modified "
              "after reader.wait");
      }
    }

    return success();
  }

  static FailureOr<unsigned>
  findOpenWriter(ArrayRef<OpenWriterGeneration> openWriters, PipeState *state,
                 Value stage, Operation *operation, StringRef failureMessage) {
    std::optional<unsigned> match;
    for (auto [index, generation] : llvm::enumerate(openWriters)) {
      if (generation.pipe != state || !sameIndex(generation.stage, stage))
        continue;
      if (match) {
        operation->emitOpError(
            "matches multiple open writer generations for one pipe stage");
        return failure();
      }
      match = index;
    }
    if (!match) {
      operation->emitOpError(failureMessage);
      return failure();
    }
    return *match;
  }

  static FailureOr<unsigned>
  findOpenReader(ArrayRef<OpenReaderGeneration> openReaders, PipeState *state,
                 unsigned readerEndpoint, Value stage, Operation *operation,
                 StringRef failureMessage) {
    std::optional<unsigned> match;
    for (auto [index, generation] : llvm::enumerate(openReaders)) {
      if (generation.pipe != state ||
          generation.readerEndpoint != readerEndpoint ||
          !sameIndex(generation.stage, stage))
        continue;
      if (match) {
        operation->emitOpError(
            "matches multiple open reader generations for one pipe stage");
        return failure();
      }
      match = index;
    }
    if (!match) {
      operation->emitOpError(failureMessage);
      return failure();
    }
    return *match;
  }

  OpenWriterGeneration *
  findDominatingWriterGeneration(const ResolvedPipeFieldAccess &access,
                                 Operation *operation,
                                 const PipeStaticPartitionInfo &placement) {
    auto matches = [&](OpenWriterGeneration &generation) {
      if (generation.pipe != access.pipe ||
          !equivalentPipeValue(generation.stage, access.stage) ||
          !dominance.dominates(generation.acquire.getOperation(), operation) ||
          generation.writerEndpoint >= access.pipe->endpoints.size())
        return false;
      return samePlacement(access.pipe->endpoints[generation.writerEndpoint],
                           placement);
    };
    OpenWriterGeneration *matched = nullptr;
    for (OpenWriterGeneration &generation : unmatchedWriterGenerations) {
      if (!matches(generation))
        continue;
      if (matched) {
        operation->emitOpError(
            "MUSA TLE pipe writer acquire must dominate all payload sources "
            "and commit");
        return nullptr;
      }
      matched = &generation;
    }
    if (matched)
      return matched;

    // If traversal reached a dominated region before the outer block was
    // visited, synthesize the open window from its unique dominating acquire.
    PipeWriterAcquireOp dominatingAcquire;
    for (PipeWriterAcquireOp acquire : access.pipe->acquires) {
      if (!equivalentPipeValue(acquire.getStage(), access.stage) ||
          !dominance.dominates(acquire.getOperation(), operation))
        continue;
      if (dominatingAcquire) {
        operation->emitOpError(
            "MUSA TLE pipe writer acquire must dominate all payload sources "
            "and commit");
        return nullptr;
      }
      dominatingAcquire = acquire;
    }
    if (!dominatingAcquire)
      return nullptr;
    auto generation = std::make_unique<OpenWriterGeneration>();
    generation->pipe = access.pipe;
    generation->writerEndpoint = 0;
    generation->stage = dominatingAcquire.getStage();
    generation->phase = dominatingAcquire.getPhase();
    generation->acquire = dominatingAcquire;
    unmatchedWriterGenerations.push_back(std::move(*generation));
    return &unmatchedWriterGenerations.back();
  }

  LogicalResult recordTMECopy(ttg::TMACopyOp copy,
                              SmallVectorImpl<OpenWriterGeneration> &writers,
                              SmallVectorImpl<OpenReaderGeneration> &readers) {
    bool globalToLocal = isa<tt::TensorDescType>(copy.getSrc().getType()) &&
                         isa<ttg::MemDescType>(copy.getDst().getType());
    Value memdesc = globalToLocal ? copy.getDst() : copy.getSrc();
    FailureOr<SmallVector<ResolvedPipeFieldAccess>> accesses =
        resolvePipeFieldAccess(copy, memdesc);
    if (failed(accesses))
      return failure();
    if (accesses->empty())
      return success();
    FailureOr<PipeStaticPartitionInfo> copyPlacement =
        getEndpointPlacement(copy);
    if (failed(copyPlacement))
      return failure();

    std::optional<unsigned> matchedGeneration;
    std::optional<ResolvedPipeFieldAccess> matchedAccess;
    bool hasUnsubscribedReaderCandidate = false;
    bool hasWrongEndpointPlacement = false;
    for (const ResolvedPipeFieldAccess &access : *accesses) {
      if (globalToLocal) {
        for (auto [index, generation] : llvm::enumerate(writers)) {
          if (generation.pipe == access.pipe &&
              sameIndex(generation.stage, access.stage)) {
            if (failed(verifyEndpointPlacement(
                    *access.pipe, generation.writerEndpoint, copy,
                    "MUSA TLE pipe writer payload operation must execute in "
                    "the writer partition")))
              return failure();
            if (matchedGeneration) {
              copy.emitOpError(
                  "cannot uniquely associate TME load with a pipe field");
              return failure();
            }
            matchedGeneration = index;
            matchedAccess = access;
          }
        }
      } else {
        for (auto [index, generation] : llvm::enumerate(readers)) {
          if (generation.pipe != access.pipe ||
              !sameIndex(generation.stage, access.stage))
            continue;
          if (generation.readerEndpoint >= access.pipe->endpoints.size())
            return copy.emitOpError(
                "internal MUSA TLE pipe analysis lost reader endpoint");
          if (!samePlacement(access.pipe->endpoints[generation.readerEndpoint],
                             *copyPlacement)) {
            hasWrongEndpointPlacement = true;
            continue;
          }
          const PipeEndpointState &endpoint =
              access.pipe->endpoints[generation.readerEndpoint];
          if (!llvm::is_contained(endpoint.subscribedFields,
                                  access.fieldIndex)) {
            hasUnsubscribedReaderCandidate = true;
            continue;
          }
          if (matchedGeneration) {
            copy.emitOpError(
                "cannot uniquely associate TME store with a pipe field");
            return failure();
          }
          matchedGeneration = index;
          matchedAccess = access;
        }
      }
    }

    // Reader waits may live in an enclosing block while the actual drain is
    // nested in an scf.if/scf.for region.  Associate such a store with the
    // already-created dominating drain group instead of requiring both
    // operations to share a basic block.
    if (!globalToLocal && !matchedGeneration) {
      for (const ResolvedPipeFieldAccess &access : *accesses) {
        PipeReaderDrainGroup *group =
            findDominatingReaderDrainGroup(access, copy, *copyPlacement);
        if (!group)
          continue;
        if (!isValidRegion(access.coveredRegion))
          return copy.emitOpError(
              "MUSA TLE pipe requires a statically provable contiguous field "
              "region");
        FailureOr<int32_t> transactionBytes = getTransactionBytes(copy);
        if (failed(transactionBytes))
          return failure();
        if (access.coveredRegion.byteSize !=
            std::optional<int64_t>(*transactionBytes))
          return copy.emitOpError(
              "MUSA TLE pipe TME transaction bytes must equal the covered "
              "region size");
        group->drainSources.push_back(PipeReaderDrainSource{
            PipeReaderDrainKind::TMEStore, copy.getOperation(),
            access.fieldIndex, access.coveredRegion, copy.getDst()});
        for (const PipeCoveredRegion &modified : group->modifiedRegions) {
          if (modified.fieldIndex != access.fieldIndex ||
              modified.memdescRoot != access.coveredRegion.memdescRoot)
            continue;
          bool overlaps =
              !modified.exact || !access.coveredRegion.exact ||
              !modified.byteOffset || !modified.byteSize ||
              !access.coveredRegion.byteOffset ||
              !access.coveredRegion.byteSize ||
              intervalsOverlap({*modified.byteOffset, *modified.byteSize},
                               {*access.coveredRegion.byteOffset,
                                *access.coveredRegion.byteSize});
          if (overlaps)
            return copy.emitOpError(
                "MUSA TLE pipe reader TME store source must not be modified "
                "after reader.wait");
        }
        copy->setAttr(triton::musa::kTLEPipeReaderTMEStoreAttr,
                      UnitAttr::get(copy.getContext()));
        return success();
      }
    }

    if (globalToLocal && !matchedGeneration) {
      for (const ResolvedPipeFieldAccess &access : *accesses) {
        OpenWriterGeneration *generation =
            findDominatingWriterGeneration(access, copy, *copyPlacement);
        if (!generation)
          continue;
        if (failed(verifyEndpointPlacement(
                *access.pipe, generation->writerEndpoint, copy,
                "MUSA TLE pipe writer payload operation must execute in the "
                "writer partition")))
          return failure();
        if (!access.coveredRegion.exact || !access.coveredRegion.byteOffset ||
            !access.coveredRegion.byteSize)
          return copy.emitOpError(
              "MUSA TLE pipe requires a statically provable contiguous field "
              "region");
        PipeBarrierStorageOwner storageOwner = PipeBarrierStorageOwner::Pipe;
        Value externalRoot;
        if (Value explicitBarrier = copy.getCompletionBarrier()) {
          FailureOr<PipeExternalBarrierUse> external =
              resolvePipeExternalBarrier(explicitBarrier, access.stage, copy);
          if (failed(external))
            return failure();
          storageOwner = PipeBarrierStorageOwner::External;
          externalRoot = external->base;
          for (Operation *lifecycle : result->lifecycleOps) {
            if (result->lookupPipe(lifecycle) != access.pipe ||
                dominance.dominates(external->allocation.getOperation(),
                                    lifecycle))
              continue;
            return copy.emitOpError(
                "MUSA TLE pipe external completion barrier must dominate all "
                "pipe operations");
          }
          externalBarrierOwners[external->allocation.getOperation()] =
              access.pipe;
          pipeExternalTMECopies.insert(copy.getOperation());
        }
        generation->completionSources.push_back(
            PendingCompletionSource{PipeTransportKind::TME, copy.getOperation(),
                                    access, storageOwner, externalRoot});
        return success();
      }
    }

    if (!matchedGeneration) {
      if (!globalToLocal && hasWrongEndpointPlacement)
        return copy.emitOpError(
            "MUSA TLE pipe reader payload operation must execute in the "
            "reader partition");
      if (!globalToLocal && hasUnsubscribedReaderCandidate)
        return copy.emitOpError(
            "MUSA TLE pipe reader TME store source is not included in the "
            "reader field subscription");
      if (globalToLocal && hasOpenReaderForMemDesc(copy.getDst(), readers))
        return success();
      if (globalToLocal &&
          llvm::any_of(*accesses, [](const ResolvedPipeFieldAccess &access) {
            return access.pipe &&
                   access.pipe->lifecycle.mode == PipeLifecycleMode::OneShot;
          }))
        return copy.emitOpError(
            "MUSA TLE one-shot pipe payload is immutable after publication");
      if (!globalToLocal &&
          llvm::all_of(*accesses, [](const ResolvedPipeFieldAccess &access) {
            return access.pipe &&
                   access.pipe->lifecycle.mode != PipeLifecycleMode::OneShot;
          })) {
        for (const ResolvedPipeFieldAccess &access : *accesses) {
          for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
               access.pipe->readerDrainGroups) {
            const PipeReaderDrainGroup &group = *ownedGroup;
            if (!group.release ||
                group.readerEndpoint >= access.pipe->endpoints.size() ||
                !sameIndex(group.stage, access.stage) ||
                !dominance.dominates(const_cast<PipeReaderDrainGroup &>(group)
                                         .wait.getOperation(),
                                     copy.getOperation()) ||
                !samePlacement(access.pipe->endpoints[group.readerEndpoint],
                               *copyPlacement) ||
                !llvm::is_contained(access.pipe->endpoints[group.readerEndpoint]
                                        .subscribedFields,
                                    access.fieldIndex))
              continue;
            if (dominance.dominates(const_cast<PipeReaderDrainGroup &>(group)
                                        .release.getOperation(),
                                    copy.getOperation()) ||
                !postDominance.postDominates(
                    const_cast<PipeReaderDrainGroup &>(group)
                        .release.getOperation(),
                    copy.getOperation()))
              return copy.emitOpError(
                  "MUSA TLE pipe reader TME store must complete before every "
                  "release or lifecycle exit");
          }
        }
      }
      if (hasStructuredPipeAfter(copy.getOperation()))
        return copy.emitOpError(
            globalToLocal
                ? "MUSA TLE pipe writer acquire must dominate all payload "
                  "sources and commit"
                : "MUSA TLE pipe reader wait must dominate all drain sources "
                  "and release");
      copy.emitOpError(globalToLocal
                           ? "pipe TME load requires an open same-block "
                             "writer generation with the same stage"
                           : "pipe TME store requires an open same-block "
                             "reader generation with the same stage");
      return failure();
    }

    if (globalToLocal) {
      if (!matchedAccess->coveredRegion.exact ||
          !matchedAccess->coveredRegion.byteOffset ||
          !matchedAccess->coveredRegion.byteSize ||
          *matchedAccess->coveredRegion.byteSize <= 0)
        return copy.emitOpError(
            "MUSA TLE pipe requires a statically provable contiguous field "
            "region");
      PipeBarrierStorageOwner storageOwner = PipeBarrierStorageOwner::Pipe;
      Value externalRoot;
      if (Value explicitBarrier = copy.getCompletionBarrier()) {
        FailureOr<PipeExternalBarrierUse> external = resolvePipeExternalBarrier(
            explicitBarrier, matchedAccess->stage, copy);
        if (failed(external))
          return failure();
        storageOwner = PipeBarrierStorageOwner::External;
        externalRoot = external->base;
        Operation *allocation = external->allocation.getOperation();
        for (Operation *lifecycle : result->lifecycleOps) {
          if (result->lookupPipe(lifecycle) != matchedAccess->pipe ||
              dominance.dominates(allocation, lifecycle))
            continue;
          return copy.emitOpError(
              "MUSA TLE pipe external completion barrier must dominate all "
              "pipe operations");
        }
        auto owner = externalBarrierOwners.find(allocation);
        if (owner != externalBarrierOwners.end() &&
            owner->second != matchedAccess->pipe)
          return copy.emitOpError(
              "MUSA TLE external completion barrier cannot be shared by "
              "multiple pipes");
        externalBarrierOwners[allocation] = matchedAccess->pipe;
        pipeExternalTMECopies.insert(copy.getOperation());
      }
      writers[*matchedGeneration].completionSources.push_back(
          PendingCompletionSource{PipeTransportKind::TME, copy.getOperation(),
                                  *matchedAccess, storageOwner, externalRoot});
      return success();
    }

    if (!isValidRegion(matchedAccess->coveredRegion))
      return copy.emitOpError(
          "MUSA TLE pipe requires a statically provable contiguous field "
          "region");
    FailureOr<int32_t> transactionBytes = getTransactionBytes(copy);
    if (failed(transactionBytes))
      return failure();
    if (matchedAccess->coveredRegion.byteSize !=
        std::optional<int64_t>(*transactionBytes))
      return copy.emitOpError(
          "MUSA TLE pipe TME transaction bytes must equal the covered "
          "region size");

    OpenReaderGeneration &generation = readers[*matchedGeneration];
    generation.drainSources.push_back(
        PipeReaderDrainSource{PipeReaderDrainKind::TMEStore,
                              copy.getOperation(), matchedAccess->fieldIndex,
                              matchedAccess->coveredRegion, copy.getDst()});
    return success();
  }

  LogicalResult
  recordLocalStore(Operation *operation, const LocalStoreTarget &target,
                   SmallVectorImpl<OpenWriterGeneration> &writers,
                   SmallVectorImpl<OpenReaderGeneration> &readers) {
    FailureOr<SmallVector<ResolvedPipeFieldAccess>> accesses =
        resolvePipeFieldAccess(operation, target.memdesc);
    if (failed(accesses))
      return failure();
    if (accesses->empty())
      return success();

    FailureOr<PipeStaticPartitionInfo> operationPlacement =
        getEndpointPlacement(operation);
    if (failed(operationPlacement))
      return failure();

    std::optional<unsigned> matchedGeneration;
    std::optional<ResolvedPipeFieldAccess> matchedAccess;
    bool hasWrongEndpointPlacement = false;
    for (const ResolvedPipeFieldAccess &access : *accesses) {
      for (auto [index, generation] : llvm::enumerate(writers)) {
        if (generation.pipe != access.pipe ||
            !sameIndex(generation.stage, access.stage))
          continue;
        if (generation.writerEndpoint >= access.pipe->endpoints.size())
          return operation->emitOpError(
              "internal MUSA TLE pipe analysis lost writer endpoint");
        if (!samePlacement(access.pipe->endpoints[generation.writerEndpoint],
                           *operationPlacement)) {
          hasWrongEndpointPlacement = true;
          continue;
        }
        if (matchedGeneration) {
          operation->emitOpError(
              "cannot uniquely associate local store with a pipe field");
          return failure();
        }
        matchedGeneration = index;
        matchedAccess = access;
      }
    }

    if (!matchedGeneration) {
      if (hasOpenReaderForMemDesc(target.memdesc, readers))
        return success();
      if (hasWrongEndpointPlacement)
        return operation->emitOpError(
            "MUSA TLE pipe writer payload operation must execute in the "
            "writer partition");
      for (const ResolvedPipeFieldAccess &access : *accesses) {
        OpenWriterGeneration *generation = findDominatingWriterGeneration(
            access, operation, *operationPlacement);
        if (!generation)
          continue;
        if (!target.exactWholeField || !access.exactWholeSlot)
          return operation->emitOpError(
              "MUSA TLE pipe local-store transport requires one unmasked "
              "whole-field store");
        std::optional<int64_t> fieldBytes =
            getStaticMemDescBytes(target.memdesc);
        if (!fieldBytes || *fieldBytes <= 0)
          return operation->emitOpError(
              "MUSA TLE pipe local-store transport requires a positive "
              "static whole-field size");
        ResolvedPipeFieldAccess fallbackAccess = access;
        fallbackAccess.coveredRegion.byteOffset = 0;
        fallbackAccess.coveredRegion.byteSize = *fieldBytes;
        generation->completionSources.push_back(PendingCompletionSource{
            PipeTransportKind::LocalStore, operation, fallbackAccess,
            PipeBarrierStorageOwner::Pipe, Value()});
        return success();
      }
      if (llvm::any_of(*accesses, [](const ResolvedPipeFieldAccess &access) {
            return access.pipe &&
                   access.pipe->lifecycle.mode == PipeLifecycleMode::OneShot;
          }))
        return operation->emitOpError(
            "MUSA TLE one-shot pipe payload is immutable after publication");
      operation->emitOpError(
          "pipe local store requires an open same-block writer generation "
          "with the same stage");
      return failure();
    }
    if (!target.exactWholeField || !matchedAccess->exactWholeSlot)
      return operation->emitOpError(
          "MUSA TLE pipe local-store transport requires one unmasked "
          "whole-field store");

    std::optional<int64_t> fieldBytes = getStaticMemDescBytes(target.memdesc);
    if (!fieldBytes || *fieldBytes <= 0)
      return operation->emitOpError(
          "MUSA TLE pipe local-store transport requires a positive static "
          "whole-field size");
    matchedAccess->coveredRegion.byteOffset = 0;
    matchedAccess->coveredRegion.byteSize = *fieldBytes;
    writers[*matchedGeneration].completionSources.push_back(
        PendingCompletionSource{PipeTransportKind::LocalStore, operation,
                                *matchedAccess, PipeBarrierStorageOwner::Pipe,
                                Value()});
    return success();
  }

  // Records a ttg.async_copy_global_to_local whose destination is a pipe
  // payload field as a completion source.  The transport is
  // wait-then-arrive: the analysis requires a per-thread async wait between
  // the copy and the commit (checked in closeWriterGeneration), and lowering
  // publishes with one warp-collective arrival per writer warp, exactly like
  // the local-store transport.
  LogicalResult
  recordAsyncCopy(ttg::AsyncCopyGlobalToLocalOp copy,
                  SmallVectorImpl<OpenWriterGeneration> &writers,
                  SmallVectorImpl<OpenReaderGeneration> &readers) {
    FailureOr<SmallVector<ResolvedPipeFieldAccess>> accesses =
        resolvePipeFieldAccess(copy, copy.getResult());
    if (failed(accesses))
      return failure();
    if (accesses->empty())
      return success();

    FailureOr<PipeStaticPartitionInfo> operationPlacement =
        getEndpointPlacement(copy);
    if (failed(operationPlacement))
      return failure();

    std::optional<unsigned> matchedGeneration;
    std::optional<ResolvedPipeFieldAccess> matchedAccess;
    bool hasWrongEndpointPlacement = false;
    for (const ResolvedPipeFieldAccess &access : *accesses) {
      for (auto [index, generation] : llvm::enumerate(writers)) {
        if (generation.pipe != access.pipe ||
            !sameIndex(generation.stage, access.stage))
          continue;
        if (generation.writerEndpoint >= access.pipe->endpoints.size())
          return copy.emitOpError(
              "internal MUSA TLE pipe analysis lost writer endpoint");
        if (!samePlacement(access.pipe->endpoints[generation.writerEndpoint],
                           *operationPlacement)) {
          hasWrongEndpointPlacement = true;
          continue;
        }
        if (matchedGeneration) {
          copy.emitOpError(
              "cannot uniquely associate async copy with a pipe field");
          return failure();
        }
        matchedGeneration = index;
        matchedAccess = access;
      }
    }

    if (!matchedGeneration) {
      if (hasOpenReaderForMemDesc(copy.getResult(), readers))
        return success();
      if (hasWrongEndpointPlacement)
        return copy.emitOpError(
            "MUSA TLE pipe writer payload operation must execute in the "
            "writer partition");
      if (llvm::any_of(*accesses, [](const ResolvedPipeFieldAccess &access) {
            return access.pipe &&
                   access.pipe->lifecycle.mode == PipeLifecycleMode::OneShot;
          }))
        return copy.emitOpError(
            "MUSA TLE one-shot pipe payload is immutable after publication");
      return copy.emitOpError(
          "MUSA TLE pipe async-copy transport requires an open same-block "
          "writer generation with the same stage");
    }

    if (!matchedAccess->coveredRegion.exact)
      return copy.emitOpError(
          "MUSA TLE pipe async-copy transport requires a statically provable "
          "contiguous field region");
    writers[*matchedGeneration].completionSources.push_back(
        PendingCompletionSource{PipeTransportKind::AsyncCopy, copy,
                                *matchedAccess, PipeBarrierStorageOwner::Pipe,
                                Value()});
    return success();
  }

  bool isWriterOperationAfterClose(PipeState *state, Operation *operation) {
    if (!state || !operation)
      return false;
    for (const std::unique_ptr<PipeCloseGeneration> &generation :
         state->closeGenerations) {
      Operation *close = generation->close.getOperation();
      if ((close->getBlock() == operation->getBlock() &&
           close->isBeforeInBlock(operation)) ||
          dominance.dominates(close, operation))
        return true;
    }
    for (PipeWriterCloseOp closeOp : deferredCloseGenerations) {
      if (result->lookupPipe(closeOp) != state)
        continue;
      Operation *close = closeOp.getOperation();
      if ((close->getBlock() == operation->getBlock() &&
           close->isBeforeInBlock(operation)) ||
          dominance.dominates(close, operation))
        return true;
    }
    return false;
  }

  LogicalResult
  openWriterGeneration(PipeWriterAcquireOp acquire,
                       SmallVectorImpl<OpenWriterGeneration> &openWriters) {
    PipeState *state = result->lookupPipe(acquire);
    if (isWriterOperationAfterClose(state, acquire.getOperation()))
      return acquire.emitOpError(
          "MUSA TLE pipe writer operations are not allowed after writer.close");
    std::optional<unsigned> writerEndpoint =
        findEndpoint(*state, PipeEndpointRole::Writer);
    if (!writerEndpoint)
      return acquire.emitOpError(
          "internal MUSA TLE pipe analysis lost writer endpoint");
    bool oneShot = state->lifecycle.mode == PipeLifecycleMode::OneShot;
    if (oneShot &&
        (failed(getOneShotStage(*state, acquire.getStage(), acquire)) ||
         failed(verifyOneShotPhase(acquire.getPhase(), acquire))))
      return failure();
    for (const OpenWriterGeneration &generation : openWriters) {
      if (generation.pipe == state &&
          sameIndex(generation.stage, acquire.getStage())) {
        if (oneShot)
          return success();
        return acquire.emitOpError(
            "duplicates an open writer generation for the same pipe stage");
      }
    }
    openWriters.push_back(OpenWriterGeneration{state,
                                               *writerEndpoint,
                                               acquire.getStage(),
                                               acquire.getPhase(),
                                               acquire,
                                               {}});
    return success();
  }

  LogicalResult
  recordCloseGeneration(PipeWriterCloseOp close,
                        SmallVectorImpl<OpenWriterGeneration> &openWriters) {
    PipeState *state = result->lookupPipe(close);
    if (!state)
      return close.emitOpError(
          "internal MUSA TLE pipe analysis lost close ownership");
    if (state->lifecycle.mode == PipeLifecycleMode::OneShot)
      return close.emitOpError(
          "MUSA TLE one-shot pipe does not support writer.close");
    if (!state->closeGenerations.empty())
      return close.emitOpError(
          "MUSA TLE pipe supports at most one writer.close per pipe");
    if (deferredClosePipes.contains(state))
      return close.emitOpError(
          "MUSA TLE pipe supports at most one writer.close per pipe");
    for (PipeWriterCommitOp candidate : state->commits) {
      if (candidate.getOperation() == close.getOperation())
        continue;
      bool afterClose =
          (candidate->getBlock() == close->getBlock() &&
           close->isBeforeInBlock(candidate)) ||
          dominance.dominates(close.getOperation(), candidate.getOperation());
      if (afterClose)
        return close.emitOpError(
            "MUSA TLE pipe writer operations are not allowed after "
            "writer.close");
    }

    auto hasStructuredGap = [&](Operation *begin, Operation *end) {
      if (!begin || !end || begin->getBlock() != end->getBlock())
        return false;
      for (Operation *cursor = begin->getNextNode(); cursor && cursor != end;
           cursor = cursor->getNextNode()) {
        if ((isa<scf::IfOp, scf::ForOp>(cursor) &&
             hasPipeLifecycleDescendant(cursor)))
          return true;
      }
      return false;
    };
    bool openWriter =
        llvm::any_of(openWriters, [&](const OpenWriterGeneration &generation) {
          return generation.pipe == state;
        });
    bool unvisitedCommit = state->commitGroups.size() < state->commits.size();
    bool structuredGap = false;
    for (const OpenWriterGeneration &generation : openWriters) {
      if (generation.pipe == state &&
          hasStructuredGap(const_cast<PipeWriterAcquireOp &>(generation.acquire)
                               .getOperation(),
                           close.getOperation())) {
        structuredGap = true;
        break;
      }
    }
    if (!structuredGap) {
      for (Operation *cursor = close->getPrevNode(); cursor;
           cursor = cursor->getPrevNode()) {
        if (!isa<scf::IfOp, scf::ForOp>(cursor) ||
            !hasPipeLifecycleDescendant(cursor))
          continue;
        bool belongsToPipe = false;
        cursor->walk([&](Operation *nested) {
          belongsToPipe = belongsToPipe || result->lookupPipe(nested) == state;
        });
        if (belongsToPipe) {
          structuredGap = true;
          break;
        }
      }
    }
    if (openWriter && !structuredGap) {
      return close.emitOpError(
          "MUSA TLE pipe close requires all writer payload generations to "
          "commit");
    }
    if (unvisitedCommit && !structuredGap) {
      // There is no structured region between this close and the pending
      // commit operations.  Preserve the established diagnostic rather than
      // treating an ordinary same-block lifecycle error as deferred CFG.
      return close.emitOpError(
          "MUSA TLE pipe close requires all writer payload generations to "
          "commit");
    }
    if (openWriter || unvisitedCommit) {
      for (auto it = openWriters.begin(); it != openWriters.end();) {
        if (it->pipe != state) {
          ++it;
          continue;
        }
        unmatchedWriterGenerations.push_back(std::move(*it));
        it = openWriters.erase(it);
      }
      deferredClosePipes.insert(state);
      deferredCloseGenerations.push_back(close);
      return success();
    }

    bool hasTME = false;
    bool hasLocalStore = false;
    if (!state->commitGroups.empty()) {
      const PipeCommitGroup &group = *state->commitGroups.front();
      hasTME = group.tmeGroupArrivalCount != 0;
      hasLocalStore = group.localStoreArrivalCount != 0;
      if (group.fullArrivalCount !=
          group.tmeGroupArrivalCount + group.localStoreArrivalCount)
        return close.emitOpError(
            "MUSA TLE pipe close plan has inconsistent arrival shape");
    }

    int32_t writerWarps = 0;
    if (hasLocalStore) {
      std::optional<unsigned> writer =
          findEndpoint(*state, PipeEndpointRole::Writer);
      if (!writer || *writer >= state->endpoints.size() ||
          state->endpoints[*writer].warpCount <= 0)
        return close.emitOpError(
            "MUSA TLE pipe close requires a positive producer warp count");
      writerWarps = state->endpoints[*writer].warpCount;
    }

    auto generation = std::make_unique<PipeCloseGeneration>();
    generation->close = close;
    generation->stage = close.getStage();
    generation->phase = close.getPhase();
    generation->controlArrivalCount = (hasTME || !hasLocalStore) ? 1 : 0;
    generation->localStoreArrivalCount = hasLocalStore ? writerWarps : 0;
    generation->fullArrivalCount =
        generation->controlArrivalCount + generation->localStoreArrivalCount;
    generation->transactionBytes = 0;
    generation->implicitEmptyAcquire = true;

    if (!state->commitGroups.empty()) {
      const PipeBarrierPlan &plan = state->barrierPlan;
      if (plan.full.arrivalCount != generation->fullArrivalCount)
        return close.emitOpError(
            "MUSA TLE pipe close plan has inconsistent arrival shape");
    }

    state->barrierPlan.hasCloseState = true;
    state->barrierPlan.closeTagPlan =
        PipeCloseTagPlan{state->capacity, false, PipeBarrierStorageOwner::Pipe};
    state->lifecycle.closePolicy = PipeClosePolicy::TaggedBroadcast;
    PipeCloseGeneration *generationPtr = generation.get();
    state->closeGenerations.push_back(std::move(generation));
    result->closeGenerationByOperation[close] = generationPtr;
    return success();
  }

  LogicalResult
  closeWriterGeneration(PipeWriterCommitOp commit,
                        SmallVectorImpl<OpenWriterGeneration> &openWriters) {
    PipeState *state = result->lookupPipe(commit);
    if (state && isWriterOperationAfterClose(state, commit.getOperation()))
      return commit.emitOpError(
          "MUSA TLE pipe writer operations are not allowed after "
          "writer.close");
    std::optional<unsigned> localMatch;
    for (auto [index, candidate] : llvm::enumerate(openWriters)) {
      if (candidate.pipe == state &&
          equivalentPipeValue(candidate.stage, commit.getStage())) {
        if (localMatch)
          return commit.emitOpError(
              "matches multiple open writer generations for one pipe stage");
        localMatch = index;
      }
    }
    OpenWriterGeneration generation;
    if (localMatch) {
      generation = std::move(openWriters[*localMatch]);
      openWriters.erase(openWriters.begin() + *localMatch);
    } else {
      OpenWriterGeneration *dominating = nullptr;
      for (OpenWriterGeneration &candidate : unmatchedWriterGenerations) {
        if (candidate.pipe != state ||
            !equivalentPipeValue(candidate.stage, commit.getStage()) ||
            !dominance.dominates(candidate.acquire.getOperation(),
                                 commit.getOperation()))
          continue;
        if (dominating)
          return commit.emitOpError(
              "matches multiple open writer generations for one pipe stage");
        dominating = &candidate;
      }
      if (!dominating) {
        // Handle a traversal order in which a nested region is visited before
        // its containing block: recover the unique dominating acquire and
        // create the open window lazily.
        PipeWriterAcquireOp acquire;
        for (PipeWriterAcquireOp candidate : state->acquires) {
          if (!equivalentPipeValue(candidate.getStage(), commit.getStage()) ||
              !dominance.dominates(candidate.getOperation(),
                                   commit.getOperation()))
            continue;
          if (acquire)
            return commit.emitOpError(
                "matches multiple open writer generations for one pipe stage");
          acquire = candidate;
        }
        if (!acquire) {
          if (hasStructuredPipeAfter(commit.getOperation()))
            return commit.emitOpError(
                "MUSA TLE pipe writer acquire must dominate all payload "
                "sources and commit");
          return commit.emitOpError(
              "requires a same-block, same-stage matching writer.acquire");
        }
        generation.pipe = state;
        generation.writerEndpoint = 0;
        generation.stage = acquire.getStage();
        generation.phase = acquire.getPhase();
        generation.acquire = acquire;
      } else {
        generation = std::move(*dominating);
        unmatchedWriterGenerations.erase(
            llvm::find_if(unmatchedWriterGenerations,
                          [&](const OpenWriterGeneration &candidate) {
                            return &candidate == dominating;
                          }));
      }
    }

    if (state->lifecycle.mode == PipeLifecycleMode::OneShot) {
      FailureOr<int32_t> stage =
          getOneShotStage(*state, commit.getStage(), commit);
      if (failed(stage) || failed(verifyOneShotPhase(generation.phase, commit)))
        return failure();
      if (llvm::is_contained(state->oneShotPublishedStages, *stage))
        return commit.emitOpError(
            "MUSA TLE one-shot pipe stage may be published at most once");
      state->oneShotPublishedStages.push_back(*stage);
    }

    int64_t tmeSourceCount =
        llvm::count_if(generation.completionSources,
                       [](const PendingCompletionSource &source) {
                         return source.kind == PipeTransportKind::TME;
                       });
    int64_t localStoreSourceCount =
        llvm::count_if(generation.completionSources,
                       [](const PendingCompletionSource &source) {
                         return source.kind == PipeTransportKind::LocalStore;
                       });
    int64_t asyncCopySourceCount =
        llvm::count_if(generation.completionSources,
                       [](const PendingCompletionSource &source) {
                         return source.kind == PipeTransportKind::AsyncCopy;
                       });
    bool hasTME = tmeSourceCount != 0;
    bool hasLocalStore = localStoreSourceCount != 0;
    bool hasAsyncCopy = asyncCopySourceCount != 0;

    if (!hasTME && !hasLocalStore && !hasAsyncCopy) {
      if (hasStructuredPipeAfter(commit.getOperation()))
        return commit.emitOpError(
            "MUSA TLE pipe lifecycle generation alternatives must have "
            "identical completion regions");
      return commit.emitOpError(
          "MUSA TLE pipe commit requires a completion source for every "
          "payload field");
    }
    if (!hasTME && !hasAsyncCopy && hasLocalStore && state->fields.size() != 1)
      return commit.emitOpError(
          "MUSA TLE pipe local-store-only transport currently requires one "
          "payload field");
    if (!hasTME && !hasAsyncCopy && hasLocalStore && localStoreSourceCount != 1)
      return commit.emitOpError(
          "MUSA TLE pipe local-store transport requires one unmasked "
          "whole-field store");
    SmallVector<PipeCompletionSource> completionSources;
    completionSources.reserve(generation.completionSources.size());
    int64_t totalBytes = 0;
    Value externalBarrierRoot;
    for (PendingCompletionSource &pending : generation.completionSources) {
      unsigned fieldIndex = pending.access.fieldIndex;
      if (fieldIndex >= state->fields.size())
        return pending.operation->emitOpError(
            "internal MUSA TLE pipe analysis found an invalid field index");
      PipeCoveredRegion region = pending.access.coveredRegion;
      if (!isValidRegion(region) ||
          region.memdescRoot != state->fields[fieldIndex].memdescRoot)
        return pending.operation->emitOpError(
            "MUSA TLE pipe requires a statically provable contiguous field "
            "region");
      int32_t transactionBytes = 0;
      if (pending.kind == PipeTransportKind::TME) {
        auto copy = dyn_cast_or_null<ttg::TMACopyOp>(pending.operation);
        if (!copy)
          return commit.emitOpError(
              "internal MUSA TLE pipe analysis lost a TME completion "
              "operation");
        FailureOr<int32_t> bytes = getTransactionBytes(copy);
        if (failed(bytes))
          return failure();
        if (region.byteSize != std::optional<int64_t>(*bytes))
          return copy.emitOpError(
              "MUSA TLE pipe TME transaction bytes must equal the covered "
              "region size");
        if (totalBytes >
            std::numeric_limits<int32_t>::max() - static_cast<int64_t>(*bytes))
          return commit.emitOpError(
              "MUSA TLE pipe aggregate TME transaction bytes exceed the "
              "positive i32 range");
        totalBytes += *bytes;
        transactionBytes = *bytes;
        region.byteSize = *bytes;
      } else if (pending.kind == PipeTransportKind::LocalStore) {
        if (!getLocalStoreTarget(pending.operation))
          return commit.emitOpError(
              "internal MUSA TLE pipe analysis lost a local-store "
              "completion operation");
        FailureOr<int64_t> fieldBytes = getStaticFieldBytes(*state, fieldIndex);
        if (!pending.access.exactWholeSlot ||
            region.byteOffset != std::optional<int64_t>(0) ||
            failed(fieldBytes) || !region.byteSize ||
            *region.byteSize != *fieldBytes)
          return pending.operation->emitOpError(
              "MUSA TLE pipe local-store transport requires a positive "
              "static whole-field size");
      } else if (pending.kind == PipeTransportKind::AsyncCopy) {
        auto copy =
            dyn_cast_or_null<ttg::AsyncCopyGlobalToLocalOp>(pending.operation);
        if (!copy)
          return commit.emitOpError(
              "internal MUSA TLE pipe analysis lost an async-copy "
              "completion operation");
        // The transport is wait-then-arrive: the data only lands in shared
        // memory when the issuing thread waits, so require an async wait in
        // the same block between the copy and this commit.  The fusion pass
        // and the TLE async-load lowering always emit this wait.
        if (copy->getBlock() != commit->getBlock())
          return copy.emitOpError(
              "MUSA TLE pipe async-copy transport requires the copy and its "
              "commit in the same block");
        bool coveredByWait = false;
        for (Operation *cursor = copy->getNextNode(); cursor;
             cursor = cursor->getNextNode()) {
          if (cursor == commit.getOperation())
            break;
          if (isa<ttg::AsyncWaitOp>(cursor)) {
            coveredByWait = true;
            break;
          }
        }
        if (!coveredByWait)
          return copy.emitOpError(
              "MUSA TLE pipe async-copy transport requires an async wait "
              "between the copy and the commit");
      } else {
        return commit.emitOpError(
            "internal MUSA TLE pipe analysis found an unknown completion "
            "transport");
      }
      bool duplicate = llvm::any_of(
          completionSources, [&](const PipeCompletionSource &candidate) {
            return candidate.kind == pending.kind &&
                   candidate.operation == pending.operation &&
                   sameRegion(candidate.coveredRegion, region);
          });
      if (!duplicate)
        completionSources.push_back(PipeCompletionSource{
            pending.kind, pending.operation, fieldIndex, pending.access.stage,
            generation.phase, region, transactionBytes,
            pending.barrierStorageOwner, pending.externalBarrierRoot});
      if (pending.kind == PipeTransportKind::TME &&
          pending.barrierStorageOwner == PipeBarrierStorageOwner::External) {
        if (!pending.externalBarrierRoot)
          return pending.operation->emitOpError(
              "MUSA TLE pipe external completion barrier must be bound to "
              "every TME source");
        if (externalBarrierRoot &&
            externalBarrierRoot != pending.externalBarrierRoot)
          return pending.operation->emitOpError(
              "MUSA TLE pipe external completion barrier must be used "
              "consistently across all commits");
        externalBarrierRoot = pending.externalBarrierRoot;
      } else if (pending.kind == PipeTransportKind::TME &&
                 externalBarrierRoot) {
        return pending.operation->emitOpError(
            "MUSA TLE pipe external completion barrier must be bound to "
            "every TME source");
      }
    }

    if (externalBarrierRoot &&
        llvm::any_of(completionSources, [](const PipeCompletionSource &source) {
          return source.kind == PipeTransportKind::TME &&
                 source.barrierStorageOwner !=
                     PipeBarrierStorageOwner::External;
        }))
      return commit.emitOpError(
          "MUSA TLE pipe external completion barrier must be bound to every "
          "TME source");

    for (unsigned fieldIndex = 0; fieldIndex < state->fields.size();
         ++fieldIndex) {
      bool hasFieldTME = llvm::any_of(
          completionSources, [&](const PipeCompletionSource &source) {
            return source.destinationField == fieldIndex &&
                   source.kind == PipeTransportKind::TME;
          });
      bool hasFieldLocal = llvm::any_of(
          completionSources, [&](const PipeCompletionSource &source) {
            return source.destinationField == fieldIndex &&
                   source.kind == PipeTransportKind::LocalStore;
          });
      bool hasFieldAsync = llvm::any_of(
          completionSources, [&](const PipeCompletionSource &source) {
            return source.destinationField == fieldIndex &&
                   source.kind == PipeTransportKind::AsyncCopy;
          });
      if ((hasFieldTME && hasFieldLocal) || (hasFieldTME && hasFieldAsync) ||
          (hasFieldLocal && hasFieldAsync))
        return commit.emitOpError(
            "MUSA TLE pipe does not support mixed transport sources for one "
            "payload field");
    }

    if (failed(verifyRegionCoverage(*state, commit, completionSources))) {
      // A branch-local commit is an alternative of one logical generation.
      // Report the path-signature mismatch at the structured boundary instead
      // of exposing a branch-order-dependent per-field coverage error.
      if (hasStructuredPipeAfter(commit.getOperation()))
        return commit.emitOpError(
            "MUSA TLE pipe lifecycle generation alternatives must have "
            "identical completion regions");
      return failure();
    }

    hasTME = llvm::any_of(completionSources, [](const PipeCompletionSource &s) {
      return s.kind == PipeTransportKind::TME;
    });
    hasLocalStore =
        llvm::any_of(completionSources, [](const PipeCompletionSource &s) {
          return s.kind == PipeTransportKind::LocalStore;
        });
    tmeSourceCount =
        llvm::count_if(completionSources, [](const PipeCompletionSource &s) {
          return s.kind == PipeTransportKind::TME;
        });
    localStoreSourceCount =
        llvm::count_if(completionSources, [](const PipeCompletionSource &s) {
          return s.kind == PipeTransportKind::LocalStore;
        });

    if (!state->commitGroups.empty()) {
      const PipeCommitGroup &reference = *state->commitGroups.front();
      auto normalized = [](ArrayRef<PipeCompletionSource> sources) {
        SmallVector<const PipeCompletionSource *> ordered;
        for (const PipeCompletionSource &source : sources)
          ordered.push_back(&source);
        llvm::sort(ordered, [](const PipeCompletionSource *lhs,
                               const PipeCompletionSource *rhs) {
          if (lhs->destinationField != rhs->destinationField)
            return lhs->destinationField < rhs->destinationField;
          int64_t lhsOffset = lhs->coveredRegion.byteOffset.value_or(-1);
          int64_t rhsOffset = rhs->coveredRegion.byteOffset.value_or(-1);
          if (lhsOffset != rhsOffset)
            return lhsOffset < rhsOffset;
          int64_t lhsSize = lhs->coveredRegion.byteSize.value_or(-1);
          int64_t rhsSize = rhs->coveredRegion.byteSize.value_or(-1);
          if (lhsSize != rhsSize)
            return lhsSize < rhsSize;
          return static_cast<int>(lhs->kind) < static_cast<int>(rhs->kind);
        });
        return ordered;
      };
      SmallVector<const PipeCompletionSource *> lhs =
          normalized(reference.completionSources);
      SmallVector<const PipeCompletionSource *> rhs =
          normalized(completionSources);
      if (lhs.size() != rhs.size()) {
        if (hasStructuredPipeAfter(commit.getOperation()) &&
            areIfAlternatives(
                const_cast<PipeCommitGroup &>(reference).commit.getOperation(),
                commit.getOperation()))
          return commit.emitOpError(
              "MUSA TLE pipe lifecycle generation alternatives must have "
              "identical completion regions");
        return commit.emitOpError(
            "all commits on one MUSA TLE pipe must use identical per-field "
            "completion regions");
      }
      for (auto [previous, source] : llvm::zip_equal(lhs, rhs)) {
        if (previous->kind != source->kind ||
            previous->destinationField != source->destinationField ||
            previous->coveredRegion.memdescRoot !=
                source->coveredRegion.memdescRoot ||
            previous->coveredRegion.byteOffset !=
                source->coveredRegion.byteOffset ||
            previous->coveredRegion.byteSize !=
                source->coveredRegion.byteSize ||
            previous->transactionBytes != source->transactionBytes ||
            previous->barrierStorageOwner != source->barrierStorageOwner) {
          if (hasStructuredPipeAfter(commit.getOperation()) &&
              areIfAlternatives(const_cast<PipeCommitGroup &>(reference)
                                    .commit.getOperation(),
                                commit.getOperation()))
            return commit.emitOpError(
                "MUSA TLE pipe lifecycle generation alternatives must have "
                "identical completion regions");
          return commit.emitOpError(
              "all commits on one MUSA TLE pipe must use identical per-field "
              "completion regions");
        }
      }
    }

    if (state->barrierPlan.full.transactionBytes &&
        *state->barrierPlan.full.transactionBytes != totalBytes)
      return commit.emitOpError(
          "all commits on one pipe must use identical transaction bytes");
    state->barrierPlan.full.transactionBytes = static_cast<int32_t>(totalBytes);
    int32_t tmeGroupArrivalCount = hasTME ? 1 : 0;
    int32_t localStoreArrivalCount = 0;
    if (hasLocalStore || hasAsyncCopy) {
      if (generation.writerEndpoint >= state->endpoints.size())
        return commit.emitOpError(
            "internal MUSA TLE pipe analysis lost writer endpoint mapping");
      localStoreArrivalCount =
          state->endpoints[generation.writerEndpoint].warpCount;
      if (localStoreArrivalCount <= 0)
        return commit.emitOpError(
            "MUSA TLE pipe local-store transport requires a positive "
            "producer warp count");
    }
    int32_t fullArrivalCount = tmeGroupArrivalCount + localStoreArrivalCount;
    if (!state->commitGroups.empty() &&
        state->barrierPlan.full.arrivalCount != fullArrivalCount)
      return commit.emitOpError(
          "all commits on one MUSA TLE pipe must use identical completion "
          "arrival counts");
    state->barrierPlan.full.arrivalCount = fullArrivalCount;

    if (externalBarrierRoot) {
      if (llvm::any_of(state->commitGroups, [](const auto &ownedGroup) {
            return llvm::any_of(ownedGroup->completionSources,
                                [](const PipeCompletionSource &source) {
                                  return source.kind ==
                                             PipeTransportKind::TME &&
                                         source.barrierStorageOwner ==
                                             PipeBarrierStorageOwner::Pipe;
                                });
          }))
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier must be bound to every "
            "TME source");
      auto allocation = externalBarrierRoot.getDefiningOp<BarrierAllocOp>();
      if (!allocation)
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier must be a "
            "stage-indexed barrier array");
      if (allocation.getNumBarriers() != state->capacity)
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier capacity must match "
            "pipe capacity");
      if (allocation.getInitPolarity() != 0)
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier must start in PENDING "
            "state");
      if (allocation.getArriveCount() != fullArrivalCount)
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier arrival count must "
            "match transport");
      auto expectBytes = allocation->getAttrOfType<IntegerAttr>("expect_bytes");
      if (!expectBytes || expectBytes.getInt() <= 0 ||
          expectBytes.getInt() != totalBytes)
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier expect_bytes must "
            "match aggregate TME bytes");
      if (state->barrierPlan.externalFull &&
          (state->barrierPlan.externalFull->base != externalBarrierRoot ||
           state->barrierPlan.externalFull->capacity != state->capacity))
        return commit.emitOpError(
            "MUSA TLE pipe external completion barrier must be used "
            "consistently across all commits");
      state->barrierPlan.full.storageOwner = PipeBarrierStorageOwner::External;
      state->barrierPlan.fullBarrierStorageOwner =
          PipeBarrierStorageOwner::External;
      state->barrierPlan.full.externalStorage = externalBarrierRoot;
      state->barrierPlan.externalFull = PipeExternalBarrierBinding{
          allocation,
          externalBarrierRoot,
          state->capacity,
          fullArrivalCount,
          PipeBarrierInitialState::Pending,
          static_cast<int32_t>(expectBytes.getInt())};
    } else if (hasTME && state->barrierPlan.full.storageOwner ==
                             PipeBarrierStorageOwner::External) {
      return commit.emitOpError(
          "MUSA TLE pipe external completion barrier must be bound to every "
          "TME source");
    }

    for (const PipeCompletionSource &source : completionSources) {
      PipeFieldState &field = state->fields[source.destinationField];
      if (field.transportKind != PipeTransportKind::Unknown &&
          field.transportKind != source.kind)
        return commit.emitOpError(
            "all commits on one MUSA TLE pipe require stable field "
            "transports");
      field.transportKind = source.kind;
    }

    auto group = std::make_unique<PipeCommitGroup>();
    group->stage = commit.getStage();
    group->phase = generation.phase;
    group->acquire = generation.acquire;
    group->completionSources = std::move(completionSources);
    group->totalTransactionBytes = static_cast<int32_t>(totalBytes);
    group->tmeGroupArrivalCount = tmeGroupArrivalCount;
    group->localStoreArrivalCount = localStoreArrivalCount;
    group->fullArrivalCount = fullArrivalCount;
    group->commit = commit;
    group->externalBarrierRoot = externalBarrierRoot;
    PipeCommitGroup *groupPtr = group.get();
    state->commitGroups.push_back(std::move(group));
    result->commitGroupByOperation[commit] = groupPtr;
    for (const PipeCompletionSource &source : groupPtr->completionSources) {
      if (!source.operation)
        continue;
      auto owner =
          result->completionSourceGroupByOperation.find(source.operation);
      if (owner != result->completionSourceGroupByOperation.end() &&
          owner->second != groupPtr)
        return source.operation->emitOpError(
            "MUSA TLE pipe completion source cannot belong to multiple "
            "writer generations");
      result->completionSourceGroupByOperation[source.operation] = groupPtr;
    }
    return success();
  }

  LogicalResult
  openReaderGeneration(PipeReaderWaitOp wait,
                       SmallVectorImpl<OpenReaderGeneration> &openReaders) {
    PipeState *state = result->lookupPipe(wait);
    PipeEndpointState *readerEndpoint = result->lookupEndpoint(wait);
    if (!readerEndpoint || readerEndpoint->role != PipeEndpointRole::Reader)
      return wait.emitOpError(
          "internal MUSA TLE pipe analysis lost reader endpoint");
    bool oneShot = state->lifecycle.mode == PipeLifecycleMode::OneShot;
    if (oneShot && (failed(getOneShotStage(*state, wait.getStage(), wait)) ||
                    failed(verifyOneShotPhase(wait.getPhase(), wait))))
      return failure();
    for (unsigned index = 0; index < openReaders.size(); ++index) {
      const OpenReaderGeneration &generation = openReaders[index];
      if (generation.pipe == state &&
          generation.readerEndpoint == readerEndpoint->index &&
          (oneShot || sameIndex(generation.stage, wait.getStage()))) {
        if (oneShot) {
          OpenReaderGeneration previous = std::move(openReaders[index]);
          openReaders.erase(openReaders.begin() + index);
          if (failed(finalizeReaderGeneration(std::move(previous), {})))
            return failure();
          break;
        }
        return wait.emitOpError(
            "duplicates an open reader generation for the same pipe stage");
      }
    }
    openReaders.push_back(OpenReaderGeneration{state,
                                               readerEndpoint->index,
                                               wait.getStage(),
                                               wait.getPhase(),
                                               wait,
                                               {}});
    return success();
  }

  LogicalResult
  closeReaderGeneration(PipeReaderReleaseOp release,
                        SmallVectorImpl<OpenReaderGeneration> &openReaders) {
    PipeState *state = result->lookupPipe(release);
    PipeEndpointState *readerEndpoint = result->lookupEndpoint(release);
    if (!readerEndpoint || readerEndpoint->role != PipeEndpointRole::Reader)
      return release.emitOpError(
          "internal MUSA TLE pipe analysis lost reader endpoint");
    if (state->lifecycle.mode == PipeLifecycleMode::OneShot) {
      std::optional<unsigned> match;
      for (auto [index, generation] : llvm::enumerate(openReaders)) {
        if (generation.pipe != state ||
            generation.readerEndpoint != readerEndpoint->index ||
            !sameIndex(generation.stage, release.getStage()))
          continue;
        if (match)
          return release.emitOpError(
              "matches multiple open reader generations for one pipe stage");
        match = index;
      }
      // One-shot release is a hardware no-op and is idempotent.  When it
      // follows an active wait, however, it delimits that endpoint's
      // analysis window so a later reader drain cannot be associated with
      // the already released wait.
      if (!match)
        return success();
      OpenReaderGeneration generation = std::move(openReaders[*match]);
      openReaders.erase(openReaders.begin() + *match);
      return finalizeReaderGeneration(std::move(generation), release);
    }
    bool hasLocalMatch =
        llvm::any_of(openReaders, [&](const OpenReaderGeneration &generation) {
          return generation.pipe == state &&
                 generation.readerEndpoint == readerEndpoint->index &&
                 sameIndex(generation.stage, release.getStage());
        });
    if (!hasLocalMatch && hasStructuredPipeAfter(release.getOperation()))
      return release.emitOpError(
          "MUSA TLE pipe reader release must post-dominate the wait on all "
          "normal paths");
    FailureOr<unsigned> match = findOpenReader(
        openReaders, state, readerEndpoint->index, release.getStage(), release,
        "MUSA TLE pipe reader.release requires a same-endpoint, same-block, "
        "same-stage reader.wait");
    if (failed(match))
      return failure();
    OpenReaderGeneration generation = std::move(openReaders[*match]);
    openReaders.erase(openReaders.begin() + *match);

    return finalizeReaderGeneration(std::move(generation), release);
  }

  LogicalResult finalizeReaderGeneration(OpenReaderGeneration generation,
                                         PipeReaderReleaseOp release) {
    PipeState *state = generation.pipe;
    if (!state)
      return generation.wait.emitOpError(
          "internal MUSA TLE pipe analysis lost reader generation owner");

    auto group = std::make_unique<PipeReaderDrainGroup>();
    group->readerEndpoint = generation.readerEndpoint;
    group->stage = generation.stage;
    group->phase = generation.phase;
    group->wait = generation.wait;
    group->drainSources = std::move(generation.drainSources);
    group->issuePublicationPolicy =
        state->executionMode == PipeExecutionMode::StaticWarpSpecialized
            ? PipeReaderIssuePublicationPolicy::PipeFullWait
            : PipeReaderIssuePublicationPolicy::NonWarpSpecializedCTA;
    group->sourceModifiedAfterWait = false;
    group->release = release;
    group->modifiedRegions = generation.modifiedRegions;

    for (const PipeReaderDrainSource &source : group->drainSources) {
      for (const PipeCoveredRegion &modified : generation.modifiedRegions) {
        if (modified.fieldIndex != source.sourceField ||
            modified.memdescRoot != source.coveredRegion.memdescRoot)
          continue;
        bool overlaps =
            !modified.exact || !source.coveredRegion.exact ||
            !modified.byteOffset || !modified.byteSize ||
            !source.coveredRegion.byteOffset ||
            !source.coveredRegion.byteSize ||
            intervalsOverlap({*modified.byteOffset, *modified.byteSize},
                             {*source.coveredRegion.byteOffset,
                              *source.coveredRegion.byteSize});
        if (overlaps) {
          group->sourceModifiedAfterWait = true;
          break;
        }
      }
      if (*group->sourceModifiedAfterWait)
        break;
    }

    if (state->lifecycle.mode == PipeLifecycleMode::OneShot &&
        !generation.modifiedFields.empty() && generation.firstModification)
      return generation.firstModification->emitOpError(
          "MUSA TLE one-shot pipe payload is immutable after publication");
    if (*group->sourceModifiedAfterWait && generation.firstModification)
      return generation.firstModification->emitOpError(
          "MUSA TLE pipe reader TME store source must not be modified "
          "after reader.wait");

    if (!group->drainSources.empty()) {
      for (const PipeReaderDrainSource &source : group->drainSources) {
        if (source.operation)
          source.operation->setAttr(
              triton::musa::kTLEPipeReaderTMEStoreAttr,
              UnitAttr::get(source.operation->getContext()));
      }
    }

    PipeReaderDrainGroup *groupPtr = group.get();
    state->readerDrainGroups.push_back(std::move(group));
    result->readerDrainGroupByWait[generation.wait] = groupPtr;
    for (const PipeReaderDrainSource &source : groupPtr->drainSources) {
      if (!source.operation)
        continue;
      auto owner = result->readerDrainGroupBySource.find(source.operation);
      if (owner != result->readerDrainGroupBySource.end() &&
          owner->second != groupPtr)
        return source.operation->emitOpError(
            "MUSA TLE pipe reader lifecycle generation is not path "
            "complete");
      result->readerDrainGroupBySource[source.operation] = groupPtr;
    }
    if (release)
      result->readerDrainGroupByOperation[release] = groupPtr;
    return success();
  }

  LogicalResult recordTerminalCloseWait(OpenReaderGeneration &generation) {
    PipeState *state = generation.pipe;
    if (state && state->create->getAttrOfType<ArrayAttr>("readers") &&
        state->closeGenerations.empty())
      return generation.wait.emitOpError(
          "MUSA TLE cyclic named reader must wait and release each payload "
          "generation exactly once");
    if (!state || state->closeGenerations.empty()) {
      if (hasStructuredPipeAfter(generation.wait.getOperation()))
        return generation.wait.emitOpError(
            "MUSA TLE pipe reader lifecycle generation is not path "
            "complete");
      return generation.wait.getOperation()->emitOpError(
          "requires a same-block matching reader.release");
    }
    if (!generation.drainSources.empty() || !generation.modifiedFields.empty())
      return generation.wait.getOperation()->emitOpError(
          "MUSA TLE pipe close generation does not carry payload");
    if (state->closeGenerations.size() != 1)
      return generation.wait.getOperation()->emitOpError(
          "MUSA TLE pipe terminal reader.wait must match writer.close stage "
          "and phase");

    const PipeCloseGeneration &close = *state->closeGenerations.front();
    Operation *closeOp =
        const_cast<PipeWriterCloseOp &>(close.close).getOperation();
    Operation *waitOp = generation.wait.getOperation();
    if (closeOp->getBlock() == waitOp->getBlock() &&
        !closeOp->isBeforeInBlock(waitOp))
      return waitOp->emitOpError(
          "MUSA TLE pipe terminal reader.wait must match writer.close stage "
          "and phase");
    if (!sameIndex(generation.stage, close.stage) ||
        !sameIndex(generation.phase, close.phase))
      return waitOp->emitOpError(
          "MUSA TLE pipe terminal reader.wait must match writer.close stage "
          "and phase");
    if (llvm::any_of(state->closeWaits, [&](const auto &wait) {
          return wait->readerEndpoint == generation.readerEndpoint;
        }))
      return waitOp->emitOpError(
          "MUSA TLE pipe supports at most one terminal close wait per "
          "reader");

    auto closeWait = std::make_unique<PipeReaderCloseWait>();
    closeWait->wait = generation.wait;
    closeWait->release = {};
    closeWait->readerEndpoint = generation.readerEndpoint;
    closeWait->stage = generation.stage;
    closeWait->phase = generation.phase;
    PipeReaderCloseWait *closeWaitPtr = closeWait.get();
    state->closeWaits.push_back(std::move(closeWait));
    result->closeWaitByOperation[generation.wait] = closeWaitPtr;
    return success();
  }

  LogicalResult classifyTerminalCloseWaits() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      if (state.lifecycle.mode == PipeLifecycleMode::OneShot ||
          state.closeGenerations.empty())
        continue;
      if (state.closeGenerations.size() != 1 || state.waits.empty())
        return state.create.emitOpError(
            "MUSA TLE pipe terminal reader.wait must match writer.close "
            "stage and phase");
      const PipeCloseGeneration &close = *state.closeGenerations.front();

      for (unsigned endpointIndex = 1; endpointIndex < state.endpoints.size();
           ++endpointIndex) {
        PipeReaderWaitOp terminalWait;
        for (PipeReaderWaitOp wait : state.waits) {
          PipeEndpointState *endpoint = result->lookupEndpoint(wait);
          if (endpoint && endpoint->index == endpointIndex)
            terminalWait = wait;
        }
        if (!terminalWait)
          return state.create.emitOpError(
              "MUSA TLE cyclic named reader must observe writer.close "
              "exactly once");
        if (!sameIndex(terminalWait.getStage(), close.stage) ||
            !sameIndex(terminalWait.getPhase(), close.phase)) {
          PipeReaderDrainGroup *terminalGroup =
              result->lookupReaderDrainGroup(terminalWait);
          bool matchesPayload =
              terminalGroup &&
              llvm::any_of(state.commitGroups, [&](const auto &commit) {
                return sameIndex(terminalGroup->stage, commit->stage) &&
                       sameIndex(terminalGroup->phase, commit->phase);
              });
          if (matchesPayload)
            return terminalWait.emitOpError(
                "MUSA TLE cyclic named reader must observe writer.close "
                "exactly once");
          return terminalWait.emitOpError(
              "MUSA TLE pipe terminal reader.wait must match writer.close "
              "stage and phase");
        }

        SmallVector<PipeReaderCloseWait *> endpointCloseWaits;
        for (const auto &ownedWait : state.closeWaits) {
          if (ownedWait->readerEndpoint == endpointIndex)
            endpointCloseWaits.push_back(ownedWait.get());
        }
        if (endpointCloseWaits.size() > 1)
          return terminalWait.emitOpError(
              "MUSA TLE pipe supports at most one terminal close wait per "
              "reader");

        if (PipeReaderCloseWait *existing =
                result->lookupCloseWait(terminalWait)) {
          if (endpointCloseWaits.size() != 1 ||
              endpointCloseWaits.front() != existing || existing->release)
            return terminalWait.emitOpError(
                "MUSA TLE pipe supports at most one terminal close wait per "
                "reader");
          continue;
        }

        if (!endpointCloseWaits.empty())
          return terminalWait.emitOpError(
              "MUSA TLE cyclic named reader must observe writer.close "
              "exactly once");

        PipeReaderDrainGroup *group =
            result->lookupReaderDrainGroup(terminalWait);
        if (!group || !group->release || group->readerEndpoint != endpointIndex)
          return terminalWait.emitOpError(
              "MUSA TLE pipe terminal reader.wait must match writer.close "
              "stage and phase");
        if (!group->drainSources.empty())
          return terminalWait.emitOpError(
              "MUSA TLE pipe close generation does not carry payload");

        auto closeWait = std::make_unique<PipeReaderCloseWait>();
        closeWait->wait = terminalWait;
        closeWait->release = group->release;
        closeWait->readerEndpoint = group->readerEndpoint;
        closeWait->stage = group->stage;
        closeWait->phase = group->phase;
        PipeReaderCloseWait *closeWaitPtr = closeWait.get();
        state.closeWaits.push_back(std::move(closeWait));
        result->closeWaitByOperation[terminalWait] = closeWaitPtr;
      }
    }
    return success();
  }

  LogicalResult analyzeBlock(Block &block) {
    SmallVector<OpenWriterGeneration> openWriters;
    SmallVector<OpenReaderGeneration, 4> openReaders;
    for (Operation &operation : block) {
      Operation *op = &operation;
      if (auto acquire = dyn_cast<PipeWriterAcquireOp>(op)) {
        if (failed(openWriterGeneration(acquire, openWriters)))
          return failure();
        continue;
      }
      if (auto commit = dyn_cast<PipeWriterCommitOp>(op)) {
        PipeState *state = result->lookupPipe(commit);
        if (state && !state->closeGenerations.empty())
          return commit.emitOpError(
              "MUSA TLE pipe writer operations are not allowed after "
              "writer.close");
        if (failed(closeWriterGeneration(commit, openWriters)))
          return failure();
        continue;
      }
      if (auto close = dyn_cast<PipeWriterCloseOp>(op)) {
        if (failed(recordCloseGeneration(close, openWriters)))
          return failure();
        continue;
      }
      if (auto wait = dyn_cast<PipeReaderWaitOp>(op)) {
        if (failed(openReaderGeneration(wait, openReaders)))
          return failure();
        continue;
      }
      if (auto release = dyn_cast<PipeReaderReleaseOp>(op)) {
        if (failed(closeReaderGeneration(release, openReaders)))
          return failure();
        continue;
      }
      if (failed(recordReaderMutation(op, openReaders)))
        return failure();
      if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        if (failed(recordAsyncCopy(asyncCopy, openWriters, openReaders)))
          return failure();
        continue;
      }
      if (auto copy = dyn_cast<ttg::TMACopyOp>(op)) {
        if (failed(recordTMECopy(copy, openWriters, openReaders)))
          return failure();
        continue;
      }
      if (std::optional<LocalStoreTarget> target = getLocalStoreTarget(op);
          target &&
          failed(recordLocalStore(op, *target, openWriters, openReaders)))
        return failure();
    }

    for (OpenWriterGeneration &generation : openWriters) {
      if (generation.pipe &&
          generation.pipe->lifecycle.mode == PipeLifecycleMode::OneShot &&
          generation.completionSources.empty())
        continue;
      // A generation can remain open at the end of an outer block while its
      // commit is in a dominated structured-region alternative.  Defer the
      // decision until all blocks have been visited; this is the writer-side
      // counterpart of unmatchedReaderGenerations.
      unmatchedWriterGenerations.push_back(std::move(generation));
    }
    for (OpenReaderGeneration &generation : openReaders) {
      if (generation.pipe &&
          generation.pipe->lifecycle.mode == PipeLifecycleMode::OneShot) {
        if (failed(finalizeReaderGeneration(std::move(generation), {})))
          return failure();
        continue;
      }
      unmatchedReaderGenerations.push_back(std::move(generation));
    }
    return success();
  }

  LogicalResult analyzeLifecycleBlocks() {
    if (failed(buildFieldOwnerIndex()))
      return failure();
    SmallVector<Block *> blocks;
    llvm::DenseSet<Block *> seenBlocks;
    for (Operation *op : result->lifecycleOps) {
      if (isa<PipeCreateOp>(op))
        continue;
      if (seenBlocks.insert(op->getBlock()).second)
        blocks.push_back(op->getBlock());
    }
    for (Block *block : blocks) {
      if (failed(analyzeBlock(*block)))
        return failure();
    }
    return success();
  }

  LogicalResult finalizeUnmatchedReaderGenerations() {
    for (OpenWriterGeneration &generation : unmatchedWriterGenerations) {
      if (!generation.pipe)
        continue;
      if (generation.pipe->lifecycle.mode == PipeLifecycleMode::OneShot &&
          generation.completionSources.empty())
        continue;
      if (generation.pipe->lifecycle.mode == PipeLifecycleMode::OneShot) {
        FailureOr<int32_t> stage = getOneShotStage(
            *generation.pipe, generation.stage, generation.acquire);
        if (failed(stage))
          return failure();
        if (llvm::is_contained(generation.pipe->oneShotPublishedStages, *stage))
          return generation.acquire.emitOpError(
              "MUSA TLE one-shot pipe payload is immutable after publication");
      }
      if (hasStructuredPipeAfter(generation.acquire.getOperation()))
        return generation.acquire.emitOpError(
            "MUSA TLE pipe writer generation must commit on every reachable "
            "path");
      return generation.acquire.emitOpError(
          "requires a same-block matching writer.commit");
    }
    // Close after a structured region may be encountered before the nested
    // commit blocks in the module walk.  Materialize it only after all
    // deferred writer generations have been matched, so a branch commit is
    // never mistaken for an operation after close.
    for (PipeWriterCloseOp close : deferredCloseGenerations) {
      PipeState *state = result->lookupPipe(close);
      if (!state)
        return close.emitOpError(
            "internal MUSA TLE pipe analysis lost close ownership");
      deferredClosePipes.erase(state);
      SmallVector<OpenWriterGeneration> noOpenGenerations;
      if (failed(recordCloseGeneration(close, noOpenGenerations)))
        return failure();
    }
    deferredCloseGenerations.clear();
    unmatchedWriterGenerations.clear();
    for (OpenReaderGeneration &generation : unmatchedReaderGenerations) {
      if (failed(recordTerminalCloseWait(generation)))
        return failure();
    }
    unmatchedReaderGenerations.clear();
    return success();
  }

  static bool sameCompletionRegions(ArrayRef<PipeCompletionSource> lhs,
                                    ArrayRef<PipeCompletionSource> rhs) {
    if (lhs.size() != rhs.size())
      return false;
    auto normalize = [](ArrayRef<PipeCompletionSource> sources) {
      SmallVector<const PipeCompletionSource *> ordered;
      ordered.reserve(sources.size());
      for (const PipeCompletionSource &source : sources)
        ordered.push_back(&source);
      llvm::sort(ordered, [](const PipeCompletionSource *left,
                             const PipeCompletionSource *right) {
        if (left->destinationField != right->destinationField)
          return left->destinationField < right->destinationField;
        if (left->coveredRegion.byteOffset != right->coveredRegion.byteOffset)
          return left->coveredRegion.byteOffset.value_or(-1) <
                 right->coveredRegion.byteOffset.value_or(-1);
        if (left->coveredRegion.byteSize != right->coveredRegion.byteSize)
          return left->coveredRegion.byteSize.value_or(-1) <
                 right->coveredRegion.byteSize.value_or(-1);
        return static_cast<int>(left->kind) < static_cast<int>(right->kind);
      });
      return ordered;
    };
    SmallVector<const PipeCompletionSource *> left = normalize(lhs);
    SmallVector<const PipeCompletionSource *> right = normalize(rhs);
    for (auto [leftSource, rightSource] : llvm::zip_equal(left, right)) {
      if (leftSource->kind != rightSource->kind ||
          leftSource->destinationField != rightSource->destinationField ||
          leftSource->coveredRegion.memdescRoot !=
              rightSource->coveredRegion.memdescRoot ||
          leftSource->coveredRegion.byteOffset !=
              rightSource->coveredRegion.byteOffset ||
          leftSource->coveredRegion.byteSize !=
              rightSource->coveredRegion.byteSize ||
          leftSource->transactionBytes != rightSource->transactionBytes ||
          leftSource->barrierStorageOwner != rightSource->barrierStorageOwner ||
          leftSource->externalBarrierRoot != rightSource->externalBarrierRoot)
        return false;
    }
    return true;
  }

  static bool isStaticallyBoundedStage(Value stage, int32_t capacity) {
    Value value = canonicalizePipeIndex(stage);
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) {
      int64_t signedValue = constant.getSExtValue();
      return signedValue >= 0 && signedValue < capacity;
    }
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getNumOperands() != 2)
      return false;
    StringRef name = definition->getName().getStringRef();
    if (name != "arith.remui" && name != "arith.remsi")
      return false;
    APInt modulus;
    if (!matchPattern(canonicalizePipeIndex(definition->getOperand(1)),
                      m_ConstantInt(&modulus)))
      return false;
    int64_t modulusValue = modulus.getSExtValue();
    return modulusValue > 0 && modulusValue <= capacity;
  }

  static bool isStaticallyBoundedPhase(Value phase) {
    Value value = canonicalizePipeIndex(phase);
    if (value.getType().isInteger(1))
      return true;
    APInt constant;
    return matchPattern(value, m_ConstantInt(&constant)) &&
           (constant.isZero() || constant.isOne());
  }

  LogicalResult validateStructuredControlFlow() {
    bool valid = true;
    module.walk([&](Operation *operation) {
      if (!valid)
        return;

      if (isa<scf::WhileOp, scf::ExecuteRegionOp>(operation) &&
          hasPipeLifecycleDescendant(operation)) {
        operation->emitOpError(
            "MUSA TLE pipe lifecycle control flow is not structurally "
            "supported");
        valid = false;
        return;
      }
      if (isa<cf::BranchOp, cf::CondBranchOp>(operation) &&
          hasPipeLifecycleDescendant(operation->getParentOp())) {
        operation->emitOpError(
            "MUSA TLE pipe lifecycle control flow is not structurally "
            "supported");
        valid = false;
        return;
      }

      auto forOp = dyn_cast<scf::ForOp>(operation);
      if (forOp && hasPipeLifecycleDescendant(forOp)) {
        SmallVector<PipeWriterAcquireOp> writerAcquires;
        SmallVector<PipeWriterCommitOp> writerCommits;
        SmallVector<PipeReaderWaitOp> readerWaits;
        SmallVector<PipeReaderReleaseOp> readerReleases;
        for (Operation *lifecycle : result->lifecycleOps) {
          if (isa<PipeCreateOp>(lifecycle) ||
              !forOp.getRegion().isAncestor(lifecycle->getParentRegion()))
            continue;
          if (auto acquire = dyn_cast<PipeWriterAcquireOp>(lifecycle))
            writerAcquires.push_back(acquire);
          else if (auto commit = dyn_cast<PipeWriterCommitOp>(lifecycle))
            writerCommits.push_back(commit);
          else if (auto wait = dyn_cast<PipeReaderWaitOp>(lifecycle))
            readerWaits.push_back(wait);
          else if (auto release = dyn_cast<PipeReaderReleaseOp>(lifecycle))
            readerReleases.push_back(release);
          PipeState *state = result->lookupPipe(lifecycle);
          Value stage;
          Value phase;
          if (auto acquire = dyn_cast<PipeWriterAcquireOp>(lifecycle)) {
            stage = acquire.getStage();
            phase = acquire.getPhase();
          } else if (auto commit = dyn_cast<PipeWriterCommitOp>(lifecycle)) {
            stage = commit.getStage();
          } else if (auto close = dyn_cast<PipeWriterCloseOp>(lifecycle)) {
            stage = close.getStage();
            phase = close.getPhase();
          } else if (auto wait = dyn_cast<PipeReaderWaitOp>(lifecycle)) {
            stage = wait.getStage();
            phase = wait.getPhase();
          } else if (auto release = dyn_cast<PipeReaderReleaseOp>(lifecycle)) {
            stage = release.getStage();
          }
          if (!state || !stage ||
              !isStaticallyBoundedStage(stage, state->capacity) ||
              (phase && !isStaticallyBoundedPhase(phase))) {
            lifecycle->emitOpError(
                "MUSA TLE pipe loop-carried stage and phase must be "
                "statically bounded");
            valid = false;
            return;
          }
        }
        // A loop body with an acquire (or normal cyclic wait) but no matching
        // completion operation leaves an open generation on its backedge.  We
        // intentionally only diagnose the unambiguous zero-completion case;
        // alternatives within the body are validated by the enclosing
        // structured-if merge below without counting mutually-exclusive ops
        // twice.
        if (!writerAcquires.empty() && writerCommits.empty()) {
          writerAcquires.front().emitOpError(
              "MUSA TLE pipe writer generation must commit on every "
              "reachable path");
          valid = false;
          return;
        }
        if (!readerWaits.empty() && readerReleases.empty()) {
          for (PipeReaderWaitOp wait : readerWaits) {
            PipeState *state = result->lookupPipe(wait);
            if (state && state->lifecycle.mode == PipeLifecycleMode::Cyclic) {
              wait.emitOpError(
                  "MUSA TLE pipe reader lifecycle generation is not path "
                  "complete");
              valid = false;
              return;
            }
          }
        }
        return;
      }

      auto ifOp = dyn_cast<scf::IfOp>(operation);
      if (!ifOp || !hasPipeLifecycleDescendant(ifOp))
        return;
      if (!isUniformPipeCondition(ifOp)) {
        ifOp.emitOpError(
            "MUSA TLE pipe lifecycle branch condition must be uniform");
        valid = false;
        return;
      }
      if (!ifOp.elseBlock()) {
        bool hasWriterLifecycle = false;
        bool hasLifecycleOutsideIf = false;
        ifOp.walk([&](Operation *nested) {
          hasWriterLifecycle =
              hasWriterLifecycle ||
              isa<PipeWriterAcquireOp, PipeWriterCommitOp, PipeWriterCloseOp>(
                  nested);
        });
        for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
          PipeState &state = *ownedState;
          for (Operation *lifecycle : result->lifecycleOps) {
            if (result->lookupPipe(lifecycle) != &state)
              continue;
            if (isa<PipeCreateOp>(lifecycle))
              continue;
            if (!isPipeOperationInRegion(lifecycle, ifOp.getThenRegion())) {
              hasLifecycleOutsideIf = true;
            }
          }
        }
        if (!hasLifecycleOutsideIf)
          return;
        if (hasWriterLifecycle)
          ifOp.emitOpError(
              "MUSA TLE pipe writer generation must commit on every "
              "reachable path");
        else
          ifOp.emitOpError(
              "MUSA TLE pipe reader lifecycle generation is not path "
              "complete");
        valid = false;
        return;
      }

      for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
        PipeState &state = *ownedState;
        SmallVector<PipeCommitGroup *> thenCommits;
        SmallVector<PipeCommitGroup *> elseCommits;
        for (const std::unique_ptr<PipeCommitGroup> &ownedGroup :
             state.commitGroups) {
          PipeCommitGroup *group = ownedGroup.get();
          if (isPipeOperationInRegion(group->commit, ifOp.getThenRegion()))
            thenCommits.push_back(group);
          else if (isPipeOperationInRegion(group->commit, ifOp.getElseRegion()))
            elseCommits.push_back(group);
        }
        if (thenCommits.size() != elseCommits.size()) {
          if ((!thenCommits.empty() || !elseCommits.empty())) {
            ifOp.emitOpError(
                "MUSA TLE pipe writer generation must commit on every "
                "reachable path");
            valid = false;
            return;
          }
        }
        for (auto [thenGroup, elseGroup] :
             llvm::zip_equal(thenCommits, elseCommits)) {
          if (!equivalentPipeValue(thenGroup->stage, elseGroup->stage) ||
              !equivalentPipeValue(thenGroup->phase, elseGroup->phase)) {
            ifOp.emitOpError(
                "MUSA TLE pipe lifecycle stage and phase are not equivalent "
                "at control-flow merge");
            valid = false;
            return;
          }
          if (!sameCompletionRegions(thenGroup->completionSources,
                                     elseGroup->completionSources)) {
            ifOp.emitOpError(
                "MUSA TLE pipe lifecycle generation alternatives must have "
                "identical completion regions");
            valid = false;
            return;
          }
        }

        for (unsigned endpoint = 1; endpoint < state.endpoints.size();
             ++endpoint) {
          // One-shot release is an analysis-only no-op.  It may be omitted,
          // repeated, or present only on one uniform branch, so the cyclic
          // wait/release path-cardinality checks below do not apply.
          if (state.lifecycle.mode != PipeLifecycleMode::Cyclic)
            continue;
          SmallVector<PipeReaderWaitOp> thenWaitOps;
          SmallVector<PipeReaderWaitOp> elseWaitOps;
          for (PipeReaderWaitOp wait : state.waits) {
            PipeEndpointState *mapped = result->lookupEndpoint(wait);
            if (!mapped || mapped->index != endpoint)
              continue;
            if (isPipeOperationInRegion(wait, ifOp.getThenRegion()))
              thenWaitOps.push_back(wait);
            else if (isPipeOperationInRegion(wait, ifOp.getElseRegion()))
              elseWaitOps.push_back(wait);
          }
          auto countInRegion = [&](Region &region, bool waits) {
            size_t count = 0;
            if (waits) {
              for (PipeReaderWaitOp wait : state.waits) {
                PipeEndpointState *mapped = result->lookupEndpoint(wait);
                if (mapped && mapped->index == endpoint &&
                    isPipeOperationInRegion(wait, region))
                  ++count;
              }
            } else {
              for (PipeReaderReleaseOp release : state.releases) {
                PipeEndpointState *mapped = result->lookupEndpoint(release);
                if (mapped && mapped->index == endpoint &&
                    isPipeOperationInRegion(release, region))
                  ++count;
              }
            }
            return count;
          };
          size_t thenWaits = countInRegion(ifOp.getThenRegion(), true);
          size_t elseWaits = countInRegion(ifOp.getElseRegion(), true);
          if (thenWaits != elseWaits) {
            ifOp.emitOpError(
                "MUSA TLE pipe reader lifecycle generation is not path "
                "complete");
            valid = false;
            return;
          }
          for (auto [thenWait, elseWait] :
               llvm::zip_equal(thenWaitOps, elseWaitOps)) {
            if (!equivalentPipeValue(thenWait.getStage(),
                                     elseWait.getStage()) ||
                !equivalentPipeValue(thenWait.getPhase(),
                                     elseWait.getPhase())) {
              ifOp.emitOpError(
                  "MUSA TLE pipe lifecycle stage and phase are not "
                  "equivalent at control-flow merge");
              valid = false;
              return;
            }
          }
          size_t thenReleases = countInRegion(ifOp.getThenRegion(), false);
          size_t elseReleases = countInRegion(ifOp.getElseRegion(), false);
          if (thenReleases != elseReleases) {
            ifOp.emitOpError(
                "MUSA TLE pipe reader release must post-dominate the wait on "
                "all normal paths");
            valid = false;
            return;
          }
        }
      }
    });
    return success(valid);
  }

  LogicalResult validatePathCompleteness() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      for (const std::unique_ptr<PipeCommitGroup> &ownedGroup :
           state.commitGroups) {
        PipeCommitGroup &group = *ownedGroup;
        Operation *acquire = group.acquire.getOperation();
        Operation *commit = group.commit.getOperation();
        if (!dominance.dominates(acquire, commit))
          return group.commit.emitOpError(
              "MUSA TLE pipe writer acquire must dominate all payload "
              "sources and commit");
        for (const PipeCompletionSource &source : group.completionSources) {
          if (!source.operation ||
              !dominance.dominates(acquire, source.operation))
            return group.commit.emitOpError(
                "MUSA TLE pipe writer acquire must dominate all payload "
                "sources and commit");
        }
      }

      for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
           state.readerDrainGroups) {
        PipeReaderDrainGroup &group = *ownedGroup;
        Operation *wait = group.wait.getOperation();
        for (const PipeReaderDrainSource &source : group.drainSources) {
          if (!source.operation || !dominance.dominates(wait, source.operation))
            return group.wait.emitOpError(
                "MUSA TLE pipe reader wait must dominate all drain sources "
                "and release");
          if (group.release &&
              state.lifecycle.mode == PipeLifecycleMode::Cyclic &&
              !postDominance.postDominates(group.release.getOperation(),
                                           source.operation))
            return group.release.emitOpError(
                "MUSA TLE pipe reader TME store must complete before every "
                "release or lifecycle exit");
        }
        if (group.release &&
            state.lifecycle.mode == PipeLifecycleMode::Cyclic &&
            (!dominance.dominates(wait, group.release.getOperation()) ||
             !postDominance.postDominates(group.release.getOperation(), wait)))
          return group.release.emitOpError(
              "MUSA TLE pipe reader release must post-dominate the wait on "
              "all normal paths");
      }
    }
    return success();
  }

  LogicalResult buildLogicalGenerations() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      state.logicalGenerations.clear();
      unsigned nextId = 0;
      SmallVector<PipeLogicalGeneration *> writerGenerations;

      for (const std::unique_ptr<PipeCommitGroup> &ownedGroup :
           state.commitGroups) {
        PipeCommitGroup &group = *ownedGroup;
        PipeLogicalGeneration *logical = nullptr;
        for (PipeLogicalGeneration *candidate : writerGenerations) {
          if (!equivalentPipeValue(candidate->stage, group.stage) ||
              !equivalentPipeValue(candidate->phase, group.phase) ||
              candidate->alternatives.empty() ||
              candidate->alternatives.front().operations.empty() ||
              !areIfAlternatives(
                  candidate->alternatives.front().operations.back(),
                  group.commit.getOperation()))
            continue;
          if (!sameCompletionRegions(
                  candidate->alternatives.front().completionSources,
                  group.completionSources))
            return group.commit.emitOpError(
                "MUSA TLE pipe lifecycle generation alternatives must have "
                "identical completion regions");
          logical = candidate;
          break;
        }
        if (!logical) {
          auto generation = std::make_unique<PipeLogicalGeneration>();
          generation->id = nextId++;
          generation->pipe = &state;
          generation->endpoint = 0;
          generation->stage = group.stage;
          generation->phase = group.phase;
          logical = generation.get();
          state.logicalGenerations.push_back(std::move(generation));
          writerGenerations.push_back(logical);
        }
        PipeGenerationAlternative alternative;
        alternative.operations.push_back(group.acquire);
        for (const PipeCompletionSource &source : group.completionSources) {
          alternative.operations.push_back(source.operation);
          alternative.completionSources.push_back(source);
        }
        alternative.operations.push_back(group.commit);
        alternative.path.pathAnchor = group.acquire.getOperation();
        alternative.path.commits.push_back(group.commit.getOperation());
        alternative.path.writerOpenAtExit = false;
        alternative.path.loopBackedge =
            static_cast<bool>(group.commit->getParentOfType<scf::ForOp>());
        collectPipePathExits(alternative.path.pathAnchor,
                             alternative.path.normalExits);
        alternative.loopCarried =
            static_cast<bool>(group.commit->getParentOfType<scf::ForOp>());
        logical->alternatives.push_back(std::move(alternative));
        group.logicalGeneration = logical;
        result->logicalGenerationByOperation[group.acquire] = logical;
        result->logicalGenerationByOperation[group.commit] = logical;
        for (const PipeCompletionSource &source : group.completionSources)
          result->logicalGenerationByOperation[source.operation] = logical;
      }

      for (unsigned endpoint = 1; endpoint < state.endpoints.size();
           ++endpoint) {
        llvm::DenseMap<unsigned, PipeLogicalGeneration *> logicalById;
        unsigned writerCursor = 0;
        for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
             state.readerDrainGroups) {
          PipeReaderDrainGroup &group = *ownedGroup;
          if (group.readerEndpoint != endpoint ||
              result->lookupCloseWait(group.wait))
            continue;

          PipeLogicalGeneration *writerLogical = nullptr;
          for (unsigned index = writerCursor; index < writerGenerations.size();
               ++index) {
            PipeLogicalGeneration *candidate = writerGenerations[index];
            if (!equivalentPipeValue(candidate->stage, group.stage) ||
                !equivalentPipeValue(candidate->phase, group.phase))
              continue;
            writerLogical = candidate;
            writerCursor = index + 1;
            break;
          }
          if (!writerLogical) {
            // One-shot permits repeated non-consuming waits.  Associate them
            // with the already-published physical stage without advancing a
            // cyclic generation cursor.
            for (PipeLogicalGeneration *candidate : writerGenerations) {
              if (equivalentPipeValue(candidate->stage, group.stage) &&
                  equivalentPipeValue(candidate->phase, group.phase)) {
                writerLogical = candidate;
                break;
              }
            }
          }
          unsigned id = writerLogical ? writerLogical->id : nextId++;
          PipeLogicalGeneration *logical = logicalById.lookup(id);
          if (!logical) {
            auto generation = std::make_unique<PipeLogicalGeneration>();
            generation->id = id;
            generation->pipe = &state;
            generation->endpoint = endpoint;
            generation->stage = group.stage;
            generation->phase = group.phase;
            logical = generation.get();
            state.logicalGenerations.push_back(std::move(generation));
            logicalById[id] = logical;
          }
          PipeGenerationAlternative alternative;
          alternative.operations.push_back(group.wait);
          for (const PipeReaderDrainSource &source : group.drainSources) {
            alternative.operations.push_back(source.operation);
            alternative.drainSources.push_back(source);
          }
          if (group.release)
            alternative.operations.push_back(group.release);
          alternative.path.pathAnchor = group.wait.getOperation();
          if (group.release)
            alternative.path.releases.push_back(group.release.getOperation());
          alternative.path.readerOpenAtExit = !group.release;
          alternative.path.loopBackedge =
              static_cast<bool>(group.wait->getParentOfType<scf::ForOp>());
          collectPipePathExits(alternative.path.pathAnchor,
                               alternative.path.normalExits);
          alternative.loopCarried =
              static_cast<bool>(group.wait->getParentOfType<scf::ForOp>());
          logical->alternatives.push_back(std::move(alternative));
          group.logicalGeneration = logical;
          result->logicalGenerationByOperation[group.wait] = logical;
          if (group.release)
            result->logicalGenerationByOperation[group.release] = logical;
          for (const PipeReaderDrainSource &source : group.drainSources)
            result->logicalGenerationByOperation[source.operation] = logical;
        }

        for (const std::unique_ptr<PipeReaderCloseWait> &ownedCloseWait :
             state.closeWaits) {
          PipeReaderCloseWait &closeWait = *ownedCloseWait;
          if (closeWait.readerEndpoint != endpoint)
            continue;
          auto generation = std::make_unique<PipeLogicalGeneration>();
          generation->id = nextId++;
          generation->pipe = &state;
          generation->endpoint = endpoint;
          generation->stage = closeWait.stage;
          generation->phase = closeWait.phase;
          PipeGenerationAlternative alternative;
          alternative.operations.push_back(closeWait.wait);
          if (closeWait.release)
            alternative.operations.push_back(closeWait.release);
          alternative.path.pathAnchor = closeWait.wait.getOperation();
          if (closeWait.release)
            alternative.path.releases.push_back(
                closeWait.release.getOperation());
          alternative.path.terminalClose = true;
          alternative.path.readerOpenAtExit = false;
          collectPipePathExits(alternative.path.pathAnchor,
                               alternative.path.normalExits);
          generation->alternatives.push_back(std::move(alternative));
          PipeLogicalGeneration *logical = generation.get();
          state.logicalGenerations.push_back(std::move(generation));
          closeWait.logicalGeneration = logical;
          result->logicalGenerationByOperation[closeWait.wait] = logical;
          if (closeWait.release)
            result->logicalGenerationByOperation[closeWait.release] = logical;
          if (closeWait.release) {
            if (PipeReaderDrainGroup *group =
                    result->lookupReaderDrainGroup(closeWait.release)) {
              group->logicalGeneration = logical;
              result->logicalGenerationByOperation[group->wait] = logical;
              result->logicalGenerationByOperation[group->release] = logical;
            }
          }
        }
      }
    }
    return success();
  }

  LogicalResult validateComplexLifecyclePaths() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      llvm::DenseMap<Operation *, PipeCommitGroup *> completionOwners;
      llvm::DenseMap<Operation *, PipeLogicalGeneration *> logicalOwners;

      // A concrete source is emitted exactly once.  If it is reachable from
      // two concrete commits, accepting the operation would double-count
      // transaction bytes and could publish the same payload twice.
      for (const std::unique_ptr<PipeCommitGroup> &ownedGroup :
           state.commitGroups) {
        PipeCommitGroup *group = ownedGroup.get();
        for (const PipeCompletionSource &source : group->completionSources) {
          if (!source.operation)
            continue;
          auto [it, inserted] =
              completionOwners.try_emplace(source.operation, group);
          if (!inserted && it->second != group)
            return source.operation->emitOpError(
                "MUSA TLE pipe completion source cannot belong to multiple "
                "writer generations");
        }
      }

      // BuildLogicalGenerations intentionally permits the same operation in
      // mutually-exclusive alternatives of one logical generation, but never
      // permits it to be owned by two logical generations.
      for (const std::unique_ptr<PipeLogicalGeneration> &ownedGeneration :
           state.logicalGenerations) {
        PipeLogicalGeneration *generation = ownedGeneration.get();
        for (const PipeGenerationAlternative &alternative :
             generation->alternatives) {
          for (Operation *operation : alternative.operations) {
            if (!operation)
              continue;
            auto [it, inserted] =
                logicalOwners.try_emplace(operation, generation);
            if (!inserted && it->second != generation) {
              if (isa<ttg::TMACopyOp>(operation))
                return operation->emitOpError(
                    "MUSA TLE pipe completion source cannot belong to "
                    "multiple writer generations");
              return operation->emitOpError(
                  "internal MUSA TLE pipe operation belongs to multiple "
                  "logical generations");
            }
          }

          const PipeLifecyclePathSummary &path = alternative.path;
          Operation *anchor = path.pathAnchor;
          if (generation->endpoint == 0) {
            if (path.commits.size() != 1 || path.writerOpenAtExit)
              return (anchor ? anchor : state.create.getOperation())
                  ->emitOpError(
                      "MUSA TLE pipe writer generation must commit on every "
                      "reachable path");
          } else if (state.lifecycle.mode == PipeLifecycleMode::Cyclic &&
                     !path.terminalClose &&
                     (path.releases.size() != 1 || path.readerOpenAtExit)) {
            return (anchor ? anchor : state.create.getOperation())
                ->emitOpError(
                    "MUSA TLE pipe reader lifecycle generation is not path "
                    "complete");
          }
        }
      }

      for (const std::unique_ptr<PipeReaderDrainGroup> &ownedGroup :
           state.readerDrainGroups) {
        PipeReaderDrainGroup &group = *ownedGroup;
        Operation *wait = group.wait.getOperation();
        if (!wait)
          continue;
        if (!group.release) {
          if (state.lifecycle.mode == PipeLifecycleMode::Cyclic &&
              !result->lookupCloseWait(group.wait))
            return wait->emitOpError(
                "MUSA TLE pipe reader lifecycle generation is not path "
                "complete");
          // A one-shot release is a hardware no-op and may be omitted.  The
          // generated store/read-wait sequence remains guaranteed by the
          // high-level TME-store lowering contract.
          continue;
        }

        Operation *release = group.release.getOperation();
        if (state.lifecycle.mode == PipeLifecycleMode::Cyclic &&
            (!dominance.dominates(wait, release) ||
             !postDominance.postDominates(release, wait)))
          return release->emitOpError(
              "MUSA TLE pipe reader release must post-dominate the wait on "
              "all normal paths");

        for (const PipeReaderDrainSource &source : group.drainSources) {
          if (!source.operation)
            continue;
          // A release that dominates a drain source, or fails to post-dominate
          // it, leaves a path where the reader can recycle the slot before the
          // TME store's generated commit/read-wait completes.
          if (dominance.dominates(release, source.operation) ||
              !postDominance.postDominates(release, source.operation))
            return source.operation->emitOpError(
                "MUSA TLE pipe reader TME store must complete before every "
                "release or lifecycle exit");
        }
      }

      // Close waits are terminal by construction.  They are exempt from the
      // normal release requirement, but may never carry a drain source.
      for (const std::unique_ptr<PipeReaderCloseWait> &ownedWait :
           state.closeWaits) {
        PipeReaderCloseWait &closeWait = *ownedWait;
        if (!closeWait.wait)
          continue;
        if (closeWait.release &&
            !dominance.dominates(closeWait.wait.getOperation(),
                                 closeWait.release.getOperation()))
          return closeWait.release.emitOpError(
              "MUSA TLE pipe reader release must post-dominate the wait on "
              "all normal paths");
      }
    }
    return success();
  }

  LogicalResult finalizePipeStates() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      bool oneShot = state.lifecycle.mode == PipeLifecycleMode::OneShot;
      bool hasClose = !state.closeGenerations.empty();
      bool closeOnly = hasClose && state.commitGroups.empty();
      if (oneShot) {
        if (hasClose || !state.closes.empty())
          return state.create.emitOpError(
              "MUSA TLE one-shot pipe does not support writer.close");
        if (state.acquires.empty() || state.commits.empty() ||
            state.commitGroups.size() != state.commits.size())
          return state.create.emitOpError(
              "requires at least one writer acquire/commit pair");
        if (state.waits.empty() ||
            state.readerDrainGroups.size() != state.waits.size())
          return state.create.emitOpError("requires at least one reader wait");
      }
      if (!oneShot &&
          ((!hasClose && (state.acquires.empty() || state.commits.empty())) ||
           (state.acquires.empty() != state.commits.empty())))
        return state.create.emitOpError(
            "requires at least one writer acquire/commit pair");
      if (!oneShot && (state.waits.empty() ||
                       (state.releases.empty() && state.closeWaits.empty())))
        return state.create.emitOpError(
            "requires at least one reader wait/release pair");
      if (!oneShot) {
        size_t unreleasedCloseWaits = llvm::count_if(
            state.closeWaits, [](const auto &wait) { return !wait->release; });
        if (state.waits.size() != state.releases.size() + unreleasedCloseWaits)
          return state.create.emitOpError(
              "internal MUSA TLE pipe analysis produced an unmatched reader "
              "generation");
      }
      if (!hasClose && !state.barrierPlan.full.transactionBytes)
        return state.create.emitOpError(
            "could not infer transaction bytes or consumer warp count");
      if (hasClose && state.commitGroups.empty()) {
        state.barrierPlan.full.transactionBytes = 0;
        state.barrierPlan.full.arrivalCount =
            state.closeGenerations.front()->fullArrivalCount;
      }
      if ((!state.barrierPlan.full.transactionBytes && !closeOnly) ||
          (state.barrierPlan.full.transactionBytes &&
           *state.barrierPlan.full.transactionBytes < 0) ||
          (oneShot ? state.barrierPlan.empty.has_value()
                   : !state.barrierPlan.empty.has_value()) ||
          state.barrierPlan.full.arrivalCount <= 0 ||
          (!oneShot && state.barrierPlan.empty->arrivalCount <= 0))
        return state.create.emitOpError(
            "could not infer transaction bytes or consumer warp count");

      // Freeze the endpoint-to-barrier contribution once analysis has
      // established all endpoint placements.  Lowering must consume this
      // ledger instead of deriving warp counts from whichever region happens
      // to be visited first.
      state.barrierPlan.writerParticipant.reset();
      state.barrierPlan.readerParticipants.clear();
      llvm::DenseSet<unsigned> staticPartitions;
      int64_t totalReaderWarps = 0;
      for (const PipeEndpointState &endpoint : state.endpoints) {
        if (endpoint.warpCount <= 0 || endpoint.warpBegin < 0)
          return state.create.emitOpError(
              "internal MUSA TLE pipe analysis produced an invalid barrier "
              "participant");
        PipeBarrierParticipant participant{
            endpoint.index, endpoint.partitionIndex, endpoint.partition,
            endpoint.warpBegin, endpoint.warpCount};
        if (endpoint.partition != PipePartitionKind::CTA &&
            !staticPartitions.insert(endpoint.partitionIndex).second)
          return state.create.emitOpError(
              "MUSA TLE static warp-specialized pipe partitions must host at "
              "most one pipe endpoint");
        if (endpoint.role == PipeEndpointRole::Writer) {
          if (state.barrierPlan.writerParticipant.has_value())
            return state.create.emitOpError(
                "internal MUSA TLE pipe analysis produced multiple writer "
                "participants");
          state.barrierPlan.writerParticipant = participant;
        } else {
          state.barrierPlan.readerParticipants.push_back(participant);
          totalReaderWarps += endpoint.warpCount;
          if (totalReaderWarps > std::numeric_limits<int32_t>::max())
            return state.create.emitOpError(
                "MUSA TLE pipe reader arrival count exceeds the positive "
                "i32 range");
        }
      }
      if (!state.barrierPlan.writerParticipant.has_value() ||
          state.barrierPlan.writerParticipant->endpointIndex != 0 ||
          totalReaderWarps <= 0 ||
          (!oneShot &&
           state.barrierPlan.empty->arrivalCount != totalReaderWarps))
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis produced inconsistent barrier "
            "participant accounting");

      for (unsigned endpointIndex = 0; endpointIndex < state.endpoints.size();
           ++endpointIndex) {
        PipeEndpointState &endpoint = state.endpoints[endpointIndex];
        if (endpoint.role != PipeEndpointRole::Reader)
          continue;
        for (unsigned fieldIndex : endpoint.subscribedFields)
          state.fields[fieldIndex].subscribedReaders.push_back(endpointIndex);
      }
    }
    return success();
  }

  LogicalResult validateNamedReaderGenerations() {
    for (const std::unique_ptr<PipeState> &ownedState : result->pipes) {
      PipeState &state = *ownedState;
      auto declaredReaders = state.create->getAttrOfType<ArrayAttr>("readers");
      if (!declaredReaders)
        continue;
      if (state.lifecycle.mode != PipeLifecycleMode::Cyclic)
        continue;
      bool hasClose = !state.closeGenerations.empty();

      // Structured alternatives are represented by one writer logical
      // generation with multiple concrete commit groups.  The old
      // basic-block implementation compared concrete operation counts and
      // consequently rejected a single reader wait after an scf.if.
      SmallVector<PipeLogicalGeneration *> writerGenerations;
      for (const std::unique_ptr<PipeLogicalGeneration> &ownedGeneration :
           state.logicalGenerations) {
        if (ownedGeneration->endpoint == 0)
          writerGenerations.push_back(ownedGeneration.get());
      }
      if (writerGenerations.empty() && !state.commitGroups.empty())
        return state.create.emitOpError(
            "internal MUSA TLE pipe analysis lost logical writer generations");

      for (unsigned endpointIndex = 1; endpointIndex < state.endpoints.size();
           ++endpointIndex) {
        if (hasClose) {
          size_t closeWaitCount =
              llvm::count_if(state.closeWaits, [&](const auto &wait) {
                return wait->readerEndpoint == endpointIndex;
              });
          if (closeWaitCount != 1)
            return state.create.emitOpError(
                "MUSA TLE cyclic named reader must observe writer.close "
                "exactly once");
        }

        SmallVector<PipeReaderDrainGroup *> readerGroups;
        for (const std::unique_ptr<PipeReaderDrainGroup> &group :
             state.readerDrainGroups) {
          if (group->readerEndpoint != endpointIndex ||
              result->lookupCloseWait(group->wait))
            continue;
          readerGroups.push_back(group.get());
        }

        SmallVector<PipeLogicalGeneration *> readerGenerations;
        llvm::DenseSet<unsigned> seenIds;
        for (PipeReaderDrainGroup *group : readerGroups) {
          if (!group->logicalGeneration)
            return group->wait.emitOpError(
                "internal MUSA TLE pipe analysis lost logical reader "
                "generation");
          if (seenIds.insert(group->logicalGeneration->id).second)
            readerGenerations.push_back(group->logicalGeneration);
          if (!group->release)
            return group->wait.emitOpError(
                "MUSA TLE cyclic named reader must wait and release each "
                "payload generation exactly once");
        }

        if (readerGenerations.size() != writerGenerations.size())
          return state.create.emitOpError(
              "MUSA TLE cyclic named reader must wait and release each "
              "payload generation exactly once");

        for (PipeLogicalGeneration *reader : readerGenerations) {
          auto writer = llvm::find_if(writerGenerations,
                                      [&](PipeLogicalGeneration *candidate) {
                                        return candidate->id == reader->id;
                                      });
          if (writer == writerGenerations.end())
            return state.create.emitOpError(
                "MUSA TLE cyclic named reader generation must match writer "
                "stage and phase");
          if (!equivalentPipeValue(reader->stage, (*writer)->stage) ||
              !equivalentPipeValue(reader->phase, (*writer)->phase))
            return state.create.emitOpError(
                "MUSA TLE cyclic named reader generation must match writer "
                "stage and phase");

          for (PipeReaderDrainGroup *group : readerGroups) {
            if (!group->logicalGeneration ||
                group->logicalGeneration->id != reader->id)
              continue;
            if (!group->release || group->readerEndpoint != endpointIndex ||
                !result->lookupEndpoint(group->wait) ||
                result->lookupEndpoint(group->wait)->index != endpointIndex ||
                !result->lookupEndpoint(group->release) ||
                result->lookupEndpoint(group->release)->index != endpointIndex)
              return group->wait.emitOpError(
                  "MUSA TLE cyclic named reader must wait and release each "
                  "payload generation exactly once");
          }
        }
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
  PostDominanceInfo postDominance;
  std::unique_ptr<PipeAnalysisResult> result =
      std::make_unique<PipeAnalysisResult>();
  SmallVector<PipeDefinition> definitions;
  llvm::DenseMap<Value, SmallVector<unsigned>> definitionsByFirstRoot;
  llvm::DenseMap<Value, SmallVector<PipeFieldOwner>> fieldOwnersByRoot;
  llvm::DenseMap<Operation *, PipeState *> externalBarrierOwners;
  llvm::DenseSet<Operation *> pipeExternalTMECopies;
  SmallVector<OpenWriterGeneration, 4> unmatchedWriterGenerations;
  SmallVector<PipeWriterCloseOp, 4> deferredCloseGenerations;
  llvm::SmallPtrSet<PipeState *, 4> deferredClosePipes;
  SmallVector<OpenReaderGeneration, 4> unmatchedReaderGenerations;
};

FailureOr<std::unique_ptr<PipeAnalysisResult>>
analyzeMUSAPipes(ModuleOp module) {
  return PipeAnalysisBuilder(module).run();
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
