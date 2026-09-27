#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERPIPE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace tle = triton::tle;
namespace musa_tle = triton::musa_tle;
namespace ttmg = triton::musa;

// MTT's async transaction barrier accepts at most 64 KiB for one logical
// transaction.  A pipe may own more payload than that, but it must publish the
// payload through several full-barrier rings and make every reader wait for
// every ring before exposing the slot.
static constexpr int32_t kMaxAsyncTransactionBytes = 64 * 1024;

static int64_t getCapacity(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("capacity").getInt();
}

static bool isOneShotPipe(Operation *op) {
  if (auto oneShot = op->getAttrOfType<BoolAttr>("one_shot"))
    return oneShot.getValue();
  return false;
}

static OperandRange getFields(Operation *op) {
  if (auto pipe = dyn_cast<tle::PipeCreateOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterAcquireOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterCommitOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeWriterCloseOp>(op))
    return pipe.getFields();
  if (auto pipe = dyn_cast<tle::PipeReaderWaitOp>(op))
    return pipe.getFields();
  return cast<tle::PipeReaderReleaseOp>(op).getFields();
}

static bool isPipeOp(Operation *op) {
  return isa<tle::PipeCreateOp, tle::PipeWriterAcquireOp,
             tle::PipeWriterCommitOp, tle::PipeWriterCloseOp,
             tle::PipeReaderWaitOp, tle::PipeReaderReleaseOp>(op);
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
    return current;
  }
}

static std::string getPipeKey(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << getCapacity(op) << "|";
  op->getAttr("scope").print(os);
  os << "|";
  if (Attribute name = op->getAttr("pipe_name"))
    name.print(os);
  os << "|";
  op->getAttr("field_names").print(os);
  os << "|";
  for (Value field : getFields(op))
    os << getMemDescRoot(field).getAsOpaquePointer() << ",";
  return key;
}

static bool sameIndex(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  APInt lhsValue;
  APInt rhsValue;
  return matchPattern(lhs, m_ConstantInt(&lhsValue)) &&
         matchPattern(rhs, m_ConstantInt(&rhsValue)) && lhsValue == rhsValue;
}

// Structured control flow may rematerialize an index expression inside each
// branch, producing distinct SSA values for identical arithmetic.  Recognize
// only side-effect-free, same-op integer expressions and constants.
static bool sameIndexExpr(Value lhs, Value rhs) {
  if (sameIndex(lhs, rhs))
    return true;
  if (!lhs || !rhs || lhs.getType() != rhs.getType())
    return false;
  auto lhsDef = lhs.getDefiningOp();
  auto rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || lhsDef->getName() != rhsDef->getName() ||
      lhsDef->getNumOperands() != rhsDef->getNumOperands() ||
      lhsDef->getNumResults() != 1 || rhsDef->getNumResults() != 1)
    return false;
  // The expression matcher is used as a control-flow proof, not as a
  // general CSE heuristic.  Restrict it to operations without observable
  // effects so two same-shaped custom operations cannot accidentally make a
  // wait/release pair appear path-equivalent.
  if (!isPure(lhsDef) || !isPure(rhsDef))
    return false;
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhsDef->getOperands(), rhsDef->getOperands()))
    if (!sameIndexExpr(lhsOperand, rhsOperand))
      return false;
  return true;
}

static bool isConstantTrue(Value value) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.getBoolValue();
}

enum class PipeRole { None, Consumer, Producer };

static bool hasDefaultProducer(ttg::WarpSpecializeOp ws) {
  return ws && ws->hasAttr("musa_tle.default_producer");
}

static PipeRole getPipeRole(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &ws.getDefaultRegion())
        return hasDefaultProducer(ws) ? PipeRole::Producer
                                      : PipeRole::Consumer;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      auto ws = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
      for (auto [index, partition] : llvm::enumerate(partitions.getRegions())) {
        if (region == partition)
          return hasDefaultProducer(ws)
                     ? PipeRole::Consumer
                     : (index + 1 == partitions.getNumRegions()
                            ? PipeRole::Producer
                            : PipeRole::Consumer);
      }
    }
    region = parent->getParentRegion();
  }
  return PipeRole::None;
}

static std::optional<std::pair<ttg::WarpSpecializeOp, Region *>>
getEnclosingPartition(Operation *op) {
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

static LogicalResult verifyCommonContract(Operation *op) {
  if (getCapacity(op) <= 0)
    return op->emitOpError("requires positive capacity");
  auto scope = op->getAttrOfType<StringAttr>("scope");
  if (!scope || scope.getValue() != "cta")
    return op->emitOpError(
        "initial mthreads tle.pipe supports only scope='cta'");
  if (getFields(op).empty())
    return op->emitOpError("requires at least one payload field");
  if (getFields(op).size() > 4)
    return op->emitOpError(
        "mthreads tle.pipe supports at most four payload fields");
  auto fieldNames = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNames || fieldNames.size() != getFields(op).size())
    return op->emitOpError("requires one field name per payload field");
  return success();
}

static FailureOr<int32_t> getConsumerWarps(Operation *op) {
  int warps = ttg::lookupNumWarps(op);
  if (warps <= 0 || warps > std::numeric_limits<int32_t>::max()) {
    op->emitOpError("requires a positive consumer warp count");
    return failure();
  }
  return static_cast<int32_t>(warps);
}

static Region *getConsumerExecutionRegion(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto ws = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &ws.getDefaultRegion())
        return region;
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      for (Region &partition : partitions.getPartitionRegions()) {
        if (region == &partition)
          return region;
      }
    }
    region = parent->getParentRegion();
  }
  if (auto func = op->getParentOfType<tt::FuncOp>())
    return &func.getBody();
  return nullptr;
}

static FailureOr<int32_t> getTransactionBytes(ttg::TMACopyOp copy) {
  auto descTy = dyn_cast<tt::TensorDescType>(copy.getSrc().getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(copy.getDst().getType());
  if (!descTy || !memDescTy) {
    copy.emitOpError("initial mthreads tle.pipe requires a tensor-descriptor "
                     "to shared-memory TME copy");
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

static bool isExactSlot(Value destination, Value fieldRoot, Value stage) {
  auto index = destination.getDefiningOp<ttg::MemDescIndexOp>();
  return index && getMemDescRoot(index.getSrc()) == fieldRoot &&
         sameIndex(index.getIndex(), stage);
}

// A reader wait/release pair is normally emitted in one block.  Structured
// control-flow lowering can, however, split a loop body into multiple CFG
// blocks while preserving one execution path.  Only relax the historical
// same-block requirement for this narrow case: both operations must belong to
// the same scf.for body region and the wait/release must form a proper
// dominance/post-dominance pair.  In particular, this rejects a wait outside
// a loop paired with a release in the loop, and rejects sibling if branches
// where only one branch executes a release.  Those cases require a path-aware
// phase protocol rather than moving a hardware arrival to an arbitrary block.
static bool canMatchReaderWait(tle::PipeReaderWaitOp wait,
                               tle::PipeReaderReleaseOp release,
                               StringRef pipeKey,
                               DominanceInfo &dominance,
                               PostDominanceInfo &postDominance) {
  if (getPipeKey(wait.getOperation()) != pipeKey ||
      !sameIndexExpr(wait.getStage(), release.getStage()))
    return false;

  auto waitFor = wait->getParentOfType<scf::ForOp>();
  auto releaseFor = release->getParentOfType<scf::ForOp>();
  if (!waitFor || waitFor != releaseFor)
    return false;

  // The usual form keeps both operations in one block.  Dominance across an
  // scf.if region must not be inferred from nested operations alone, because
  // an else path could skip the wait or release.
  if (wait->getParentRegion() == release->getParentRegion())
    return dominance.properlyDominates(wait.getOperation(),
                                       release.getOperation()) &&
           postDominance.properlyPostDominates(release.getOperation(),
                                               wait.getOperation());

  // Sibling scf.if operations have independent hardware scheduling and stage
  // lifetimes even when their conditions happen to be equal.  Do not infer a
  // pipe phase pairing across those regions; a previous experiment compiled
  // but hung a real MM kernel.  Supporting this form requires an explicit
  // loop-carried protocol, not a local matcher.
  return false;
}

struct PipeState {
  tle::PipeCreateOp create;
  SmallVector<Value> fieldRoots;
  SmallVector<int32_t> fieldTransactionBytes;
  int32_t capacity = 0;
  int32_t transactionBytes = -1;
  int32_t consumerWarps = 0;
  bool oneShot = false;
  DenseMap<Region *, int32_t> consumerPartitionWarps;
  std::optional<bool> warpSpecialized;
  SmallVector<Value> fullBases;
  SmallVector<Value> emptyBases;
  SmallVector<SmallVector<unsigned>> fullBarrierFieldGroups;
  SmallVector<int32_t> fullBarrierTransactionBytes;
  SmallVector<tle::PipeWriterAcquireOp> acquires;
  SmallVector<tle::PipeWriterCommitOp> commits;
  SmallVector<tle::PipeReaderWaitOp> waits;
  SmallVector<tle::PipeReaderReleaseOp> releases;
  SmallVector<std::string> readerNames;
  SmallVector<std::pair<std::string, Region *>> readerRegions;
};

static StringRef getReaderName(Operation *op) {
  if (auto name = op->getAttrOfType<StringAttr>("reader_name"))
    return name.getValue();
  return {};
}

static bool sameReader(Operation *lhs, Operation *rhs) {
  return getReaderName(lhs) == getReaderName(rhs);
}

static LogicalResult recordReaderEndpoint(PipeState &state, Operation *op) {
  StringRef name = getReaderName(op);
  if (state.readerNames.empty()) {
    if (!name.empty())
      return op->emitOpError(
          "named reader endpoint requires pipe.create readers");
    return success();
  }
  if (name.empty())
    return op->emitOpError(
        "named-reader pipe requires reader_name on every endpoint");
  if (!llvm::any_of(state.readerNames,
                    [&](const std::string &declared) {
                      return StringRef(declared) == name;
                    }))
    return op->emitOpError("reader_name is not declared by pipe.create");

  Region *region = getConsumerExecutionRegion(op);
  if (!region)
    return op->emitOpError("cannot identify the reader execution region");
  for (auto &[recordedName, recordedRegion] : state.readerRegions) {
    if (recordedName == name && recordedRegion != region)
      return op->emitOpError(
          "one named reader cannot span multiple execution partitions");
    if (recordedRegion == region && recordedName != name)
      return op->emitOpError(
          "one execution partition cannot host multiple named readers");
    if (recordedName == name && recordedRegion == region)
      return success();
  }
  state.readerRegions.emplace_back(name.str(), region);
  return success();
}

static LogicalResult recordConsumerWarps(PipeState &state, Operation *op) {
  FailureOr<int32_t> warps = getConsumerWarps(op);
  if (failed(warps))
    return failure();
  Region *executionRegion = getConsumerExecutionRegion(op);
  if (!executionRegion)
    return op->emitOpError("cannot identify the reader execution region");
  auto [it, inserted] =
      state.consumerPartitionWarps.try_emplace(executionRegion, *warps);
  if (!inserted) {
    if (it->second != *warps)
      return op->emitOpError(
          "reader warp count changed within one execution partition");
    return success();
  }
  if (state.consumerWarps > std::numeric_limits<int32_t>::max() - *warps)
    return op->emitOpError("total consumer warp count exceeds int32 range");
  state.consumerWarps += *warps;
  return success();
}

static LogicalResult recordExecutionMode(PipeState &state, Operation *op,
                                         PipeRole actualRole,
                                         PipeRole warpSpecializedRole,
                                         StringRef endpoint,
                                         bool allowProducerConsumer = false) {
  bool fusedProducerConsumer =
      allowProducerConsumer && warpSpecializedRole == PipeRole::Consumer &&
      actualRole == PipeRole::Producer;
  if (actualRole != PipeRole::None && actualRole != warpSpecializedRole &&
      !fusedProducerConsumer)
    return op->emitOpError()
           << "requires " << endpoint
           << " operations either outside warp_specialize or in the "
           << (warpSpecializedRole == PipeRole::Producer
                   ? "final worker partition"
                   : "default or consumer worker partition");

  bool usesWarpSpecialize = actualRole != PipeRole::None;
  if (state.warpSpecialized && *state.warpSpecialized != usesWarpSpecialize)
    return op->emitOpError(
        "cannot mix warp-specialized and non-warp-specialized endpoints on "
        "one pipe");
  state.warpSpecialized = usesWarpSpecialize;
  return success();
}

class LowerPipePass
    : public impl::TritonMUSAGPUTLELowerPipeBase<LowerPipePass> {
  LogicalResult analyze(ModuleOp module,
                        std::map<std::string, PipeState> &pipes,
                        DenseMap<Operation *, SmallVector<ttg::TMACopyOp>>
                            &commitCopies) {
    DominanceInfo dominance(module);
    PostDominanceInfo postDominance(module);
    std::map<std::string, DenseSet<Operation *>> matchedReaderWaits;
    SmallVector<Operation *> pipeOps;
    module.walk([&](Operation *op) {
      if (isPipeOp(op))
        pipeOps.push_back(op);
    });

    for (Operation *op : pipeOps) {
      if (failed(verifyCommonContract(op)))
        return failure();
      std::string key = getPipeKey(op);

      if (auto create = dyn_cast<tle::PipeCreateOp>(op)) {
        if (!create->getParentOfType<tt::FuncOp>() ||
            create->getParentOfType<ttg::WarpSpecializeOp>())
          return create.emitOpError(
              "requires pipe.create outside warp_specialize");
        if (pipes.find(key) != pipes.end())
          return create.emitOpError("duplicates an existing pipe identity");
        PipeState state;
        state.create = create;
        for (Value field : create.getFields()) {
          Value root = getMemDescRoot(field);
          if (llvm::is_contained(state.fieldRoots, root))
            return create.emitOpError(
                "requires payload fields with distinct shared-memory roots");
          state.fieldRoots.push_back(root);
        }
        state.capacity = static_cast<int32_t>(getCapacity(op));
        state.oneShot = isOneShotPipe(create);
        if (auto readers = create->getAttrOfType<ArrayAttr>("readers")) {
          if (state.oneShot)
            return create.emitOpError(
                "mthreads SPMC pipe version 1 supports only cyclic pipes");
          for (Attribute attr : readers)
            state.readerNames.push_back(cast<StringAttr>(attr).getValue().str());
        }
        pipes.emplace(key, std::move(state));
        continue;
      }

      auto it = pipes.find(key);
      if (it == pipes.end())
        return op->emitOpError("requires a preceding matching pipe.create");
      PipeState &state = it->second;

      if (auto close = dyn_cast<tle::PipeWriterCloseOp>(op))
        return close.emitOpError(
            "initial mthreads tle.pipe does not support writer.close");
      if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op)) {
        if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                       PipeRole::Producer, "writer")))
          return failure();
        if (state.oneShot && isConstantTrue(acquire.getPhase()))
          return acquire.emitOpError(
              "one_shot pipe requires the initial (phase=false) edge");
        state.acquires.push_back(acquire);
        continue;
      }
      if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
        if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                       PipeRole::Producer, "writer")))
          return failure();
        if (state.oneShot && !state.commits.empty())
          return commit.emitOpError(
              "one_shot pipe supports only one writer.commit");

        tle::PipeWriterAcquireOp matchingAcquire;
        SmallVector<SmallVector<ttg::TMACopyOp>> matchingFieldCopies(
            state.fieldRoots.size());
        for (Operation *previous = commit->getPrevNode(); previous;
             previous = previous->getPrevNode()) {
          if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(previous)) {
            if (getPipeKey(previous) == key &&
                sameIndex(acquire.getStage(), commit.getStage())) {
              matchingAcquire = acquire;
              break;
            }
            continue;
          }
          if (auto copy = dyn_cast<ttg::TMACopyOp>(previous)) {
            for (auto [fieldIndex, fieldRoot] :
                 llvm::enumerate(state.fieldRoots)) {
              if (isExactSlot(copy.getDst(), fieldRoot, commit.getStage()))
                matchingFieldCopies[fieldIndex].push_back(copy);
            }
          }
        }
        // A cyclic pipe needs acquire to delimit the payload window and to
        // prove which slot is being reused.  one_shot pipes have no empty
        // barrier and may commit their single ready edge directly after the
        // pipe definition; retain the stricter acquire requirement for all
        // other protocols.
        if (!matchingAcquire && !state.oneShot)
          return commit.emitOpError(
              "requires a same-block, same-stage matching writer.acquire");
        SmallVector<ttg::TMACopyOp> matchingCopies;
        SmallVector<int32_t> fieldBytes;
        int64_t totalBytes = 0;
        for (auto [fieldIndex, fieldCopies] :
             llvm::enumerate(matchingFieldCopies)) {
          if (fieldCopies.size() != 1)
            // Preserve the stable diagnostic prefix consumed by existing
            // frontend tests while retaining the field index for debugging
            // multi-payload producer protocols.
            return commit.emitOpError()
                   << "requires exactly one TME copy between acquire and commit; found "
                   << fieldCopies.size() << " (payload field " << fieldIndex
                   << ")";
          ttg::TMACopyOp copy = fieldCopies.front();
          if (copy.getCompletionBarrier())
            return copy.emitOpError(
                "pipe-managed TME copy must not provide an explicit barrier");
          FailureOr<int32_t> bytes = getTransactionBytes(copy);
          if (failed(bytes))
            return failure();
          totalBytes += *bytes;
          if (totalBytes > std::numeric_limits<int32_t>::max())
            return commit.emitOpError(
                "total TME transaction bytes exceed positive i32 range");
          matchingCopies.push_back(copy);
          fieldBytes.push_back(*bytes);
        }
        if (state.transactionBytes >= 0 &&
            state.transactionBytes != totalBytes)
          return commit.emitOpError(
              "all commits on one pipe must use identical transaction bytes");
        state.transactionBytes = static_cast<int32_t>(totalBytes);
        if (!state.fieldTransactionBytes.empty() &&
            state.fieldTransactionBytes != fieldBytes)
          return commit.emitOpError(
              "all commits on one pipe must use identical per-field transaction bytes");
        state.fieldTransactionBytes = std::move(fieldBytes);
        state.commits.push_back(commit);
        commitCopies[commit.getOperation()] = std::move(matchingCopies);
        continue;
      }
      if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                       PipeRole::Consumer, "reader",
                                       /*allowProducerConsumer=*/true)))
          return failure();
        if (state.oneShot && !state.waits.empty())
          return wait.emitOpError(
              "one_shot pipe supports only one reader.wait");
        if (state.oneShot && isConstantTrue(wait.getPhase()))
          return wait.emitOpError(
              "one_shot pipe requires the initial (phase=false) edge");
        if (failed(recordReaderEndpoint(state, op)) ||
            failed(recordConsumerWarps(state, op)))
          return failure();
        state.waits.push_back(wait);
        continue;
      }

      auto release = cast<tle::PipeReaderReleaseOp>(op);
      if (failed(recordExecutionMode(state, op, getPipeRole(op),
                                     PipeRole::Consumer, "reader",
                                     /*allowProducerConsumer=*/true)))
        return failure();
      if (failed(recordReaderEndpoint(state, op)))
        return failure();
      bool matchingWait = false;
      auto &usedWaits = matchedReaderWaits[key];
      for (Operation *previous = release->getPrevNode(); previous;
           previous = previous->getPrevNode()) {
        if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(previous)) {
          if (getPipeKey(previous) == key &&
              sameIndex(wait.getStage(), release.getStage()) &&
              sameReader(wait.getOperation(), release.getOperation()) &&
              !usedWaits.contains(wait.getOperation())) {
            matchingWait = true;
            usedWaits.insert(wait.getOperation());
            break;
          }
        }
      }
      // The fast path above preserves the existing nearest-sibling matching
      // behavior.  If no sibling exists, use the guarded CFG matcher for a
      // wait in another block of the same scf.for body.  Never reuse one wait
      // for multiple releases: a branch fan-out needs an explicit phase
      // protocol and cannot be represented by one hardware arrival.
      if (!matchingWait) {
        for (auto waitIt = state.waits.rbegin(); waitIt != state.waits.rend();
             ++waitIt) {
          tle::PipeReaderWaitOp wait = *waitIt;
          if (usedWaits.contains(wait.getOperation()))
            continue;
          if (!sameReader(wait.getOperation(), release.getOperation()))
            continue;
          if (!canMatchReaderWait(wait, release, key, dominance,
                                  postDominance))
            continue;
          matchingWait = true;
          usedWaits.insert(wait.getOperation());
          break;
        }
      }
      if (!matchingWait)
        {
          // Keep the primary diagnostic stable for frontend callers, but make
          // the reason for rejecting a cross-region candidate explicit.  A
          // wait in a sibling scf.if cannot be represented by the current
          // phase protocol: lowering it as an unconditional barrier would
          // deadlock the path on which the branch is not taken.  This note is
          // intentionally emitted only when a same-stage candidate exists;
          // unrelated missing waits retain the concise legacy error.
          bool hasCrossRegionCandidate = false;
          for (tle::PipeReaderWaitOp wait : state.waits) {
            if (usedWaits.contains(wait.getOperation()) ||
                getPipeKey(wait.getOperation()) != key ||
                !sameIndexExpr(wait.getStage(), release.getStage()) ||
                !sameReader(wait.getOperation(), release.getOperation()))
              continue;
            if (wait->getParentRegion() != release->getParentRegion()) {
              hasCrossRegionCandidate = true;
              break;
            }
          }
          auto diag = release.emitOpError(
              "requires a same-block or same-scoped-CFG, same-stage matching "
              "reader.wait");
          if (hasCrossRegionCandidate)
            diag.attachNote()
                << "same-stage reader.wait is in a different control-flow "
                   "region; sibling scf.if paths require an explicit "
                   "loop-carried phase protocol";
          return failure();
        }
      if (failed(recordConsumerWarps(state, op)))
        return failure();
      state.releases.push_back(release);
    }

    for (auto &[key, state] : pipes) {
      if ((!state.oneShot && state.acquires.empty()) || state.commits.empty())
        return state.create.emitOpError(
            "requires at least one writer acquire/commit pair");
      if (state.waits.empty() || (!state.oneShot && state.releases.empty()))
        return state.create.emitOpError(
            "requires at least one reader wait/release pair");
      if (state.transactionBytes <= 0 || state.consumerWarps <= 0)
        return state.create.emitOpError(
            "could not infer transaction bytes or consumer warp count");

      // Greedily partition payload fields into hardware-sized full
      // transactions.  Field order is stable, so the same grouping applies
      // to every commit already proven to have identical per-field sizes.
      // The empty ring stays shared: a slot is reusable only after all named
      // readers release the complete logical payload.
      SmallVector<unsigned> currentGroup;
      int32_t currentBytes = 0;
      for (auto [fieldIndex, fieldBytes] :
           llvm::enumerate(state.fieldTransactionBytes)) {
        if (fieldBytes <= 0 || fieldBytes > kMaxAsyncTransactionBytes)
          return state.create.emitOpError()
                 << "payload field " << fieldIndex << " transaction size "
                 << fieldBytes << " exceeds the 64-KiB hardware barrier limit";
        if (!currentGroup.empty() &&
            currentBytes > kMaxAsyncTransactionBytes - fieldBytes) {
          state.fullBarrierFieldGroups.push_back(currentGroup);
          state.fullBarrierTransactionBytes.push_back(currentBytes);
          currentGroup.clear();
          currentBytes = 0;
        }
        currentGroup.push_back(static_cast<unsigned>(fieldIndex));
        currentBytes += fieldBytes;
      }
      if (!currentGroup.empty()) {
        state.fullBarrierFieldGroups.push_back(currentGroup);
        state.fullBarrierTransactionBytes.push_back(currentBytes);
      }
      if (state.fullBarrierFieldGroups.empty())
        return state.create.emitOpError(
            "could not form a hardware-sized full-barrier payload group");

      for (const std::string &readerName : state.readerNames) {
        bool hasWait = llvm::any_of(state.waits, [&](auto wait) {
          return getReaderName(wait.getOperation()) == readerName;
        });
        bool hasRelease = llvm::any_of(state.releases, [&](auto release) {
          return getReaderName(release.getOperation()) == readerName;
        });
        if (!hasWait || !hasRelease)
          return state.create.emitOpError()
                 << "named reader '" << readerName
                 << "' requires at least one wait/release pair";
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

  LogicalResult rewrite(ModuleOp module,
                        std::map<std::string, PipeState> &pipes,
                        DenseMap<Operation *, SmallVector<ttg::TMACopyOp>>
                            &commitCopies) {
    SmallVector<Operation *> pipeOps;
    module.walk([&](Operation *op) {
      if (isPipeOp(op))
        pipeOps.push_back(op);
    });

    for (Operation *op : pipeOps) {
      PipeState &state = pipes.at(getPipeKey(op));
      OpBuilder builder(op);
      Location loc = op->getLoc();

      if (auto create = dyn_cast<tle::PipeCreateOp>(op)) {
        auto capacity = builder.getI32IntegerAttr(state.capacity);
        auto one = builder.getI32IntegerAttr(1);
        auto pending = builder.getI32IntegerAttr(0);
        auto ready = builder.getI32IntegerAttr(1);
        for (int32_t transactionBytes : state.fullBarrierTransactionBytes)
          state.fullBases.push_back(musa_tle::BarrierAllocOp::create(
              builder, loc, capacity, one, pending,
              builder.getI32IntegerAttr(transactionBytes)));
        // A one-shot edge transitions only once from empty to full.  There is
        // no producer-side acquire or consumer-side release, so allocating an
        // empty barrier ring would add initialization and synchronization that
        // can never be observed.  Cyclic pipes retain the empty ring because
        // it is the slot-reuse handshake for the next iteration.
        if (!state.oneShot)
          state.emptyBases.push_back(musa_tle::BarrierAllocOp::create(
              builder, loc, capacity,
              builder.getI32IntegerAttr(state.consumerWarps), ready,
              IntegerAttr()));
        create.erase();
        continue;
      }

      if (auto acquire = dyn_cast<tle::PipeWriterAcquireOp>(op)) {
        if (state.oneShot) {
          acquire.erase();
          continue;
        }
        Value phase = toI32Phase(builder, loc, acquire.getPhase(), true);
        Value barrier = createIndex(builder, loc, op, state.emptyBases.front(),
                                    acquire.getStage());
        musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        acquire.erase();
        continue;
      }

      if (auto commit = dyn_cast<tle::PipeWriterCommitOp>(op)) {
        auto copies = commitCopies.lookup(op);
        if (copies.size() != state.fieldRoots.size())
          return commit.emitOpError("lost the analyzed pipe TME copies");
        // A logical pipe commit may exceed one hardware transaction barrier.
        // Publish each <=64-KiB field group on its own full-barrier ring.  A
        // reader wait below joins all rings before exposing any payload field.
        for (auto [groupIndex, fieldGroup] :
             llvm::enumerate(state.fullBarrierFieldGroups)) {
          unsigned firstField = fieldGroup.front();
          OpBuilder barrierBuilder(copies[firstField]);
          Value barrier = createIndex(
              barrierBuilder, copies[firstField].getLoc(), copies[firstField],
              state.fullBases[groupIndex], commit.getStage());
          SmallVector<int32_t> groupParts;
          for (unsigned fieldIndex : fieldGroup)
            groupParts.push_back(state.fieldTransactionBytes[fieldIndex]);

          for (auto [indexInGroup, fieldIndex] :
               llvm::enumerate(fieldGroup)) {
            ttg::TMACopyOp copy = copies[fieldIndex];
            OpBuilder copyBuilder(copy);
            auto replacement = ttg::TMACopyOp::create(
                copyBuilder, copy.getLoc(), copy.getSrc(), copy.getDst(),
                copy.getIndices(), barrier);
            replacement->setDiscardableAttrs(
                copy->getDiscardableAttrDictionary());
            if (fieldGroup.size() > 1)
              replacement->setAttr(ttmg::kTLEGroupedCompletionAttr,
                                   copyBuilder.getUnitAttr());
            if (indexInGroup == 0) {
              replacement->setAttr(
                  "expect_bytes",
                  copyBuilder.getI32IntegerAttr(
                      state.fullBarrierTransactionBytes[groupIndex]));
              if (fieldGroup.size() > 1)
                replacement->setAttr(
                    ttmg::kTLEExpectBytesPartsAttr,
                    copyBuilder.getDenseI32ArrayAttr(groupParts));
            }
            if (fieldGroup.size() > 1 &&
                indexInGroup + 1 == fieldGroup.size())
              replacement->setAttr(ttmg::kTLEGroupedCompletionFinalAttr,
                                   copyBuilder.getUnitAttr());
            copy.erase();
          }
        }
        commit.erase();
        continue;
      }

      if (auto wait = dyn_cast<tle::PipeReaderWaitOp>(op)) {
        Value phase = toI32Phase(builder, loc, wait.getPhase(), false);
        for (Value fullBase : state.fullBases) {
          Value barrier =
              createIndex(builder, loc, op, fullBase, wait.getStage());
          musa_tle::BarrierWaitOp::create(builder, loc, barrier, phase);
        }
        if (!wait.getIsClosed().use_empty()) {
          Value notClosed = arith::ConstantIntOp::create(builder, loc, 0, 1);
          wait.getIsClosed().replaceAllUsesWith(notClosed);
        }
        wait.erase();
        continue;
      }

      if (isa<tle::PipeWriterCloseOp>(op))
        return op->emitOpError("writer.close must have failed analysis");

      auto release = cast<tle::PipeReaderReleaseOp>(op);
      if (state.oneShot) {
        release.erase();
        continue;
      }
      Value phase = arith::ConstantIntOp::create(builder, loc, 0, 32);
      Value barrier = createIndex(builder, loc, op, state.emptyBases.front(),
                                  release.getStage());
      musa_tle::BarrierArriveOp::create(builder, loc, barrier, phase,
                                        builder.getI32IntegerAttr(1));
      release.erase();
    }

    bool hasPipeOps = false;
    module.walk([&](Operation *op) { hasPipeOps |= isPipeOp(op); });
    if (hasPipeOps)
      return module.emitError("mthreads TLE pipe lowering left lifecycle ops");
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::map<std::string, PipeState> pipes;
    DenseMap<Operation *, SmallVector<ttg::TMACopyOp>> commitCopies;
    if (failed(analyze(module, pipes, commitCopies)) ||
        failed(rewrite(module, pipes, commitCopies)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
