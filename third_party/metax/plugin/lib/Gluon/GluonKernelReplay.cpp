#include "Gluon/GluonKernelReplay.h"
#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "metax-gluon-kernel-replay-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace {

static bool isPointerLike(Type type) {
  if (isa<tt::PointerType, tt::TensorDescType>(type) ||
      tt::isTensorPointerType(type))
    return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isa<tt::PointerType>(tensor.getElementType());
}

static bool isFunctionEntryArgument(BlockArgument argument, tt::FuncOp func) {
  Block *block = argument.getOwner();
  return block->isEntryBlock() && block->getParentOp() == func;
}

static bool isReplayTransparentCompilerOp(Operation *op);

/// Trace one external effect address to every possible pointer-like function
/// argument. Pointer-producing memory reads and opaque pointer creation are
/// rejected because their pointee cannot be attributed to a kernel argument.
static LogicalResult collectPointerArguments(
    Value root, tt::FuncOp func, const GluonRegionBranchAnalysis &carriers,
    llvm::SetVector<unsigned> &arguments) {
  SmallVector<Value, 8> worklist{root};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !isPointerLike(value.getType()) ||
        !visited.insert(value).second)
      continue;

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (isFunctionEntryArgument(argument, func)) {
        arguments.insert(argument.getArgNumber());
        continue;
      }
    }

    // RegionBranchOpInterface exposes both entry block arguments and parent
    // results as carrier inputs. Query the shared carrier graph before
    // inspecting a defining operation so pointer-valued scf.if/scf.for results
    // are traced to their incoming kernel arguments.
    ArrayRef<Value> predecessors = carriers.getPredecessors(value);
    if (!predecessors.empty()) {
      llvm::append_range(worklist, predecessors);
      continue;
    }
    if (isa<BlockArgument>(value)) {
      LDBG("[replay-reject] value="
           << value << " reason=unresolved-block-argument-carrier");
      return failure();
    }

    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || (!isMemoryEffectFree(definingOp) &&
                        !isReplayTransparentCompilerOp(definingOp))) {
      LDBG("[replay-reject] value="
           << value << " reason=effectful-or-missing-pointer-producer");
      return failure();
    }

    llvm::SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.omitUsesFromAbove = false;
    if (failed(getBackwardSlice(value, &backwardSlice, options))) {
      LDBG("[replay-reject] op="
           << definingOp->getName() << " reason=backward-slice-failed");
      return failure();
    }

    bool foundPredecessor = false;
    for (Value operand : definingOp->getOperands())
      if (isPointerLike(operand.getType())) {
        worklist.push_back(operand);
        foundPredecessor = true;
      }
    if (!foundPredecessor) {
      LDBG("[replay-reject] op="
           << definingOp->getName()
           << " reason=pointer-producer-has-no-pointer-predecessor"
           << " backward-slice-size=" << backwardSlice.size());
      return failure();
    }
  }
  return success(!arguments.empty());
}

template <typename Container>
static SmallVector<unsigned, 4> sortedArguments(const Container &arguments) {
  SmallVector<unsigned, 4> result(arguments.begin(), arguments.end());
  llvm::sort(result);
  return result;
}

/// These operations have no user-visible runtime memory effect. Some are
/// scheduling/token operations; SetAutoLayout is intentionally not `Pure`
/// because an unused result still anchors the compiler's layout analysis.
/// Keeping this list op-based (rather than kernel-name based) makes the
/// exceptional replay proof explicit and auditable without making the seed
/// removable by generic DCE.
static bool isReplayTransparentCompilerOp(Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<ttg::AsyncWaitOp, ttg::AsyncCommitGroupOp, ttg::LocalBarrierOp,
            ttg::GVMArriveOp, ttg::BarrierOp, ttg::SchedBoundOp, ttg::IGLPOp,
            ttg::BarrierSharedOp>([](Operation *) { return true; })
      .Case<gluon_dialect::SetAutoLayoutOp>(
          [](gluon_dialect::SetAutoLayoutOp) { return true; })
      .Default([](Operation *) { return false; });
}

static std::string serializeJSON(llvm::json::Object object) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << llvm::json::Value(std::move(object));
  return storage;
}

} // namespace

KernelReplayEffectContract analyzeKernelReplayEffectContract(tt::FuncOp func) {
  KernelReplayEffectContract contract;
  if (func.getBody().empty()) {
    LDBG("[replay-reject] function=" << func.getName()
                                      << " reason=external-function");
    return contract;
  }

  GluonRegionBranchAnalysis carriers(func);
  if (failed(carriers.initialize())) {
    LDBG("[replay-reject] function=" << func.getName()
                                      << " reason=region-analysis-failed");
    return contract;
  }

  llvm::SetVector<unsigned> tensorArgs;
  for (BlockArgument argument : func.getBody().front().getArguments())
    if (isPointerLike(argument.getType()))
      tensorArgs.insert(argument.getArgNumber());

  llvm::SetVector<unsigned> readArgs;
  llvm::SetVector<unsigned> writtenArgs;
  llvm::SetVector<unsigned> atomicArgs;
  bool rejected = false;
  WalkResult walk = func.walk([&](Operation *op) -> WalkResult {
    // Operation::walk includes its root. The function owns nested effects but
    // is not itself an external memory access.
    if (op == func.getOperation())
      return WalkResult::advance();

    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) {
      if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
          isMemoryEffectFree(op) || isReplayTransparentCompilerOp(op))
        return WalkResult::advance();
      LDBG("[replay-reject] op="
           << op->getName() << " reason=missing-memory-effect-interface");
      rejected = true;
      return WalkResult::interrupt();
    }

    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const MemoryEffects::EffectInstance &effect : instances) {
      if (effect.getResource() == ttg::SharedMemory::get())
        continue;

      bool reads = isa<MemoryEffects::Read>(effect.getEffect());
      bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (effect.getResource() != tt::GlobalMemory::get() ||
          (!reads && !writes)) {
        LDBG("[replay-reject] op="
             << op->getName() << " reason=unmodeled-external-effect resource="
             << effect.getResource()->getName());
        rejected = true;
        return WalkResult::interrupt();
      }

      Value pointer = effect.getValue();
      if (!pointer || !isPointerLike(pointer.getType())) {
        LDBG("[replay-reject] op="
             << op->getName() << " reason=unscoped-global-effect");
        rejected = true;
        return WalkResult::interrupt();
      }

      llvm::SetVector<unsigned> affectedArgs;
      if (failed(
              collectPointerArguments(pointer, func, carriers, affectedArgs))) {
        LDBG("[replay-reject] op="
             << op->getName() << " reason=untraceable-pointer-effect");
        rejected = true;
        return WalkResult::interrupt();
      }
      for (unsigned argument : affectedArgs) {
        if (reads)
          readArgs.insert(argument);
        if (writes)
          writtenArgs.insert(argument);
        if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op))
          atomicArgs.insert(argument);
      }
    }
    return WalkResult::advance();
  });
  if (rejected || walk.wasInterrupted())
    return contract;

  if (writtenArgs.empty()) {
    LDBG("[replay-reject] function="
         << func.getName() << " reason=no-observable-written-output");
    return contract;
  }
  contract.tensorArgs = sortedArguments(tensorArgs);
  contract.readArgs = sortedArguments(readArgs);
  contract.writtenArgs = sortedArguments(writtenArgs);
  contract.atomicArgs = sortedArguments(atomicArgs);
  for (unsigned written : contract.writtenArgs)
    if (!readArgs.contains(written))
      contract.writeOnlyArgs.push_back(written);
  llvm::SetVector<unsigned> effectArgs;
  effectArgs.insert(contract.readArgs.begin(), contract.readArgs.end());
  effectArgs.insert(contract.writtenArgs.begin(), contract.writtenArgs.end());
  llvm::SetVector<std::pair<unsigned, unsigned>> noAlias;
  for (unsigned written : contract.writtenArgs)
    for (unsigned other : effectArgs)
      if (written != other)
        noAlias.insert(std::minmax(written, other));
  llvm::append_range(contract.requiredNoAlias, noAlias);
  llvm::sort(contract.requiredNoAlias);
  contract.replayable = true;

  LDBG("[replay-contract] function="
       << func.getName() << " replayable=true tensor-args="
       << contract.tensorArgs.size() << " read=" << contract.readArgs.size()
       << " written=" << contract.writtenArgs.size()
       << " write-only=" << contract.writeOnlyArgs.size()
       << " atomic=" << contract.atomicArgs.size()
       << " required-noalias=" << contract.requiredNoAlias.size());
  return contract;
}

std::string serializeKernelReplayEffectContract(
    const KernelReplayEffectContract &contract) {
  auto array = [](ArrayRef<unsigned> values) {
    llvm::json::Array result;
    for (unsigned value : values)
      result.push_back(static_cast<int64_t>(value));
    return result;
  };
  llvm::json::Array requiredNoAlias;
  for (auto [lhs, rhs] : contract.requiredNoAlias) {
    llvm::json::Array pair;
    pair.push_back(static_cast<int64_t>(lhs));
    pair.push_back(static_cast<int64_t>(rhs));
    requiredNoAlias.push_back(std::move(pair));
  }
  return serializeJSON(llvm::json::Object{
      {"version", 2},
      {"argument_index_space", "lowered-tt-func-runtime-abi"},
      {"replayable", contract.replayable},
      // Floating-point atomics need not produce bit-identical scratch contents
      // across replays. This does not make timing unsafe: the runtime restores
      // their input state outside the measured interval and never observes the
      // scratch result as a user output.
      {"deterministic", contract.replayable && contract.atomicArgs.empty()},
      {"tensor_args", array(contract.tensorArgs)},
      {"read_args", array(contract.readArgs)},
      {"written_args", array(contract.writtenArgs)},
      {"write_only_args", array(contract.writeOnlyArgs)},
      {"atomic_args", array(contract.atomicArgs)},
      {"required_noalias", std::move(requiredNoAlias)},
  });
}

} // namespace mlir::triton::gpu::metax::gluon
