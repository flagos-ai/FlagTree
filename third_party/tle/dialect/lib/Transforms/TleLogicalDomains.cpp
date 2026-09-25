/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */

#include "tle/dialect/include/Transforms/LogicalDomain.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dialect/include/Analysis/AxisInfoExt.h"
#include "tle/dialect/include/IR/ExactSMEM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#ifndef __HCU__
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#endif
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <array>
#include <limits>
#include <numeric>
#include <tuple>

namespace mlir::triton::tle {
namespace ttg = mlir::triton::gpu;

namespace {
constexpr llvm::StringLiteral kLogicalAllocShape("tle.logical_alloc_shape");
constexpr llvm::StringLiteral
    kLogicalNonPowerAxis("tle.logical_non_power_axis");
constexpr llvm::StringLiteral kStoragePlan("tle.storage_plan");
constexpr llvm::StringLiteral
    kLogicalDescriptorShape("tle.logical_descriptor_shape");

static RankedTensorType getTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

static SmallVector<int32_t> identityAxisMap(unsigned rank) {
  SmallVector<int32_t> result;
  for (unsigned i = 0; i < rank; ++i)
    result.push_back(i);
  return result;
}

static LogicalShape intersectShapes(ArrayRef<int64_t> lhs,
                                    ArrayRef<int64_t> rhs) {
  assert(lhs.size() == rhs.size());
  LogicalShape result;
  for (auto [a, b] : llvm::zip(lhs, rhs))
    result.push_back(std::min(a, b));
  return result;
}

static int64_t nextPowerOfTwo(int64_t value) {
  int64_t result = 1;
  while (result < value)
    result <<= 1;
  return result;
}

static int64_t largestPowerOfTwoDivisorNoGreaterThan(int64_t value,
                                                     int64_t upperBound) {
  assert(value > 0 && upperBound > 0);
  int64_t divisor = 1;
  while (divisor <= upperBound / 2 && value % (divisor * 2) == 0)
    divisor *= 2;
  return divisor;
}

static bool isGlobalPointerTensor(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(tensorType.getElementType());
  return pointerType && pointerType.getAddressSpace() == 1;
}

static LogicalRootRewriteAction *findRootAction(LogicalDomainPlan &plan,
                                                Operation *root) {
  for (LogicalRootRewriteAction &action : plan.roots)
    if (action.alloc.getOperation() == root)
      return &action;
  return nullptr;
}

static LogicalDomainProvenance mergeProvenance(LogicalDomainProvenance lhs,
                                               LogicalDomainProvenance rhs) {
  for (Operation *root : rhs.roots)
    if (!llvm::is_contained(lhs.roots, root))
      lhs.roots.push_back(root);
  for (Operation *seed : rhs.seeds)
    if (!llvm::is_contained(lhs.seeds, seed))
      lhs.seeds.push_back(seed);
  return lhs;
}

static bool mergeProvenanceInto(LogicalDomainProvenance &target,
                                const LogicalDomainProvenance &source) {
  size_t oldRootCount = target.roots.size();
  size_t oldSeedCount = target.seeds.size();
  target = mergeProvenance(std::move(target), source);
  return target.roots.size() != oldRootCount ||
         target.seeds.size() != oldSeedCount;
}

static FailureOr<LogicalReductionIdentity>
getReductionIdentity(triton::ReduceOp op) {
  Operation *combiner = op.getSingleCombiner();
  if (!combiner)
    return failure();
  return llvm::StringSwitch<FailureOr<LogicalReductionIdentity>>(
             combiner->getName().getStringRef())
      .Cases("arith.maxnumf", "arith.maximumf",
             LogicalReductionIdentity::NegativeInfinity)
      .Cases("arith.minnumf", "arith.minimumf",
             LogicalReductionIdentity::PositiveInfinity)
      .Cases("arith.addf", "arith.addi", LogicalReductionIdentity::Zero)
      .Cases("arith.mulf", "arith.muli", LogicalReductionIdentity::One)
      .Case("arith.andi", LogicalReductionIdentity::True)
      .Case("arith.ori", LogicalReductionIdentity::False)
      .Default(failure());
}

static bool validatePrefixShape(ArrayRef<int64_t> physical,
                                ArrayRef<int64_t> logical) {
  if (physical.size() != logical.size())
    return false;
  for (auto [p, l] : llvm::zip(physical, logical))
    if (l <= 0 || l > p)
      return false;
  return true;
}

static std::optional<int64_t>
getContiguousLinearPrefixSize(ArrayRef<int64_t> physical,
                              ArrayRef<int64_t> logical) {
  if (!validatePrefixShape(physical, logical))
    return std::nullopt;
  int restrictedAxis = -1;
  for (int i = 0, e = physical.size(); i < e; ++i) {
    if (logical[i] == physical[i])
      continue;
    if (restrictedAxis >= 0)
      return std::nullopt;
    restrictedAxis = i;
  }
  if (restrictedAxis >= 0) {
    for (int i = 0; i < restrictedAxis; ++i)
      if (logical[i] != 1)
        return std::nullopt;
    for (int i = restrictedAxis + 1, e = physical.size(); i < e; ++i)
      if (logical[i] != physical[i])
        return std::nullopt;
  }
  return std::accumulate(logical.begin(), logical.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

static std::optional<LogicalShape>
representLinearPrefixAsRectangle(ArrayRef<int64_t> target,
                                 int64_t validElements) {
  int64_t suffix = 1;
  for (int axis = target.size() - 1; axis >= 0; --axis) {
    if (validElements % suffix == 0) {
      int64_t extent = validElements / suffix;
      if (extent > 0 && extent <= target[axis]) {
        LogicalShape result(target.size(), 1);
        result[axis] = extent;
        for (int i = axis + 1, e = target.size(); i < e; ++i)
          result[i] = target[i];
        return result;
      }
    }
    suffix *= target[axis];
  }
  return std::nullopt;
}

static std::optional<LogicalShape>
reshapeOnlyUnitDimensions(ArrayRef<int64_t> sourcePhysical,
                          ArrayRef<int64_t> sourceLogical,
                          ArrayRef<int64_t> targetPhysical) {
  SmallVector<int64_t> sourceNonUnitPhysical;
  SmallVector<int64_t> sourceNonUnitLogical;
  for (auto [physical, logical] : llvm::zip(sourcePhysical, sourceLogical)) {
    if (physical == 1) {
      if (logical != 1)
        return std::nullopt;
      continue;
    }
    sourceNonUnitPhysical.push_back(physical);
    sourceNonUnitLogical.push_back(logical);
  }
  SmallVector<int64_t> targetNonUnitPhysical;
  for (int64_t physical : targetPhysical)
    if (physical != 1)
      targetNonUnitPhysical.push_back(physical);
  if (sourceNonUnitPhysical != targetNonUnitPhysical)
    return std::nullopt;
  LogicalShape result;
  unsigned nextLogical = 0;
  for (int64_t physical : targetPhysical)
    result.push_back(physical == 1 ? 1 : sourceNonUnitLogical[nextLogical++]);
  return result;
}

} // namespace

SmallVector<int64_t, 2>
selectLogicalPointerCopyMicroTile(Operation *copy,
                                  ArrayRef<int64_t> storageTileShape,
                                  ttg::NVMMASharedEncodingAttr storageEncoding,
                                  Type elementType, unsigned vectorBytes) {
  assert(storageTileShape.size() == 2 &&
         "logical pointer-copy storage tile must be rank two");
  assert(elementType.isIntOrFloat() &&
         "exact-SMEM element type must have a bit width");

  // Size one copy wave for the participating producer threads.  The later
  // async-copy legality pass still selects 4/8/16-byte transactions from
  // AxisInfo; this is only the rectangular work partition presented to it.
  int64_t numWarps = ttg::maybeLookupNumWarps(copy).value_or(4);
  int64_t elementBits = elementType.getIntOrFloatBitWidth();
  int64_t targetElements = numWarps * 32 * vectorBytes * 8 / elementBits;

  unsigned contiguousAxis = storageEncoding.getTransposed() ? 0u : 1u;
  unsigned outerAxis = 1u - contiguousAxis;
  int64_t swizzleBytes =
      std::max<int64_t>(storageEncoding.getSwizzlingByteWidth(), 16);
  int64_t swizzleElements = swizzleBytes * 8 / elementBits;
  int64_t minContiguous =
      std::min(storageTileShape[contiguousAxis], swizzleElements);
  int64_t minOuter = std::min(storageTileShape[outerAxis], int64_t{8});

  // Both storage extents are powers of two, so enumerating their divisors is
  // cheap.  Maximize useful work within one producer wave while keeping at
  // least one complete NVMMA core.  For equal areas prefer one swizzle span;
  // this avoids the larger register layout and code size seen when 16x128 or
  // 16x256 fp16 copies exceed the useful work of one four-warp wave.
  SmallVector<int64_t, 2> best{minOuter, minContiguous};
  if (contiguousAxis == 0)
    std::swap(best[0], best[1]);
  int64_t bestElements = best[0] * best[1];
  bool bestFits = bestElements <= targetElements;
  for (int64_t rows = 1; rows <= storageTileShape[0]; rows *= 2) {
    for (int64_t cols = 1; cols <= storageTileShape[1]; cols *= 2) {
      SmallVector<int64_t, 2> candidate{rows, cols};
      if (candidate[contiguousAxis] < minContiguous ||
          candidate[outerAxis] < minOuter)
        continue;
      int64_t elements = rows * cols;
      bool fits = elements <= targetElements;
      if (fits != bestFits) {
        if (!fits)
          continue;
      } else if (fits && elements < bestElements) {
        continue;
      } else if (!fits && elements > bestElements) {
        continue;
      }
      if (fits != bestFits || elements != bestElements ||
          candidate[contiguousAxis] < best[contiguousAxis]) {
        best = candidate;
        bestElements = elements;
        bestFits = fits;
      }
    }
  }
  return best;
}

FailureOr<SmallVector<int64_t, 2>>
selectLogicalSMEMStorageTileShape(ttg::MemDescType stageType,
                                  ArrayRef<int64_t> logicalShape) {
  if (stageType.getRank() != 2 || logicalShape.size() != 2)
    return failure();
  if (!isa<ttg::NVMMASharedEncodingAttr>(stageType.getEncoding()))
    return failure();

  bool fragmentedRows = !llvm::isPowerOf2_64(logicalShape[0]);
  bool fragmentedCols = !llvm::isPowerOf2_64(logicalShape[1]);
  if (fragmentedRows == fragmentedCols)
    return failure();
  unsigned fragmentAxis = fragmentedRows ? 0u : 1u;
  SmallVector<int64_t, 2> storageTile(logicalShape.begin(), logicalShape.end());
  storageTile[fragmentAxis] = largestPowerOfTwoDivisorNoGreaterThan(
      logicalShape[fragmentAxis], logicalShape[fragmentAxis]);
  if (storageTile[fragmentAxis] < kExactSMEMFragmentQuantum ||
      logicalShape[fragmentAxis] % storageTile[fragmentAxis] != 0)
    return failure();

  // Cap a contiguous panel at one upstream swizzle span. The fragment
  // grouping stays independent: N96 may use 32 rows while N80 uses 16.
  // Instruction K and copy-wave sizing are consumers of this layout.
  auto encoding = cast<ttg::NVMMASharedEncodingAttr>(stageType.getEncoding());
  unsigned contiguous = encoding.getTransposed() ? 0u : 1u;
  if (fragmentAxis != contiguous) {
    int64_t panelElements =
        1024 / stageType.getElementType().getIntOrFloatBitWidth();
    storageTile[contiguous] = std::min(storageTile[contiguous], panelElements);
  }
  return storageTile;
}

namespace {

enum class LogicalDomainPhase : uint8_t { Propagate, Plan };

/// Internal compatibility facade for the existing transfer and planning
/// helpers. Tensor propagation itself is driven by LogicalDomainAnalysis
/// below; this facade owns the finalized facts and the rewrite plan only.
class LogicalDomainContext {
public:
  explicit LogicalDomainContext(LogicalDomainPlan &plan);

  const MemDescLogicalState *lookupMemDesc(Value value) const;
  const TensorDescriptorLogicalState *lookupDescriptor(Value value) const;
  const TensorFragmentState *lookupTensor(Value value) const;
  bool mergeMemDesc(Value value, const MemDescLogicalState &state);
  bool hasRestrictedOperand(Operation *op) const;
  LogicalResult validateTensorResult(Operation *op) const;

  LogicalResult processCandidateAlloc(Operation *op, LogicalDomainPhase phase);
  LogicalResult processMemDescIndex(Operation *op, LogicalDomainPhase phase);
  LogicalResult processLogicalTMACopy(Operation *op, LogicalDomainPhase phase);
  LogicalResult processLogicalPointerCopy(Operation *op,
                                          LogicalDomainPhase phase);
  LogicalResult processLocalStore(Operation *op, LogicalDomainPhase phase);
  LogicalResult processExtractStage(Operation *op, LogicalDomainPhase phase);
  LogicalResult processWarpSpecialize(Operation *op, LogicalDomainPhase phase);
  LogicalResult processPipe(Operation *op, LogicalDomainPhase phase);
  LogicalResult processMemDescUse(Operation *op, LogicalDomainPhase phase);
  LogicalResult processMemDescTranspose(Operation *op,
                                        LogicalDomainPhase phase);
  LogicalResult processWGMMA(Operation *op, LogicalDomainPhase phase);
  LogicalResult processWGMMAWait(Operation *op, LogicalDomainPhase phase);
  LogicalResult processSameShape(Operation *op, LogicalDomainPhase phase);
  LogicalResult processExpandDims(Operation *op, LogicalDomainPhase phase);
  LogicalResult processBroadcast(Operation *op, LogicalDomainPhase phase);
  LogicalResult processTranspose(Operation *op, LogicalDomainPhase phase);
  LogicalResult processReshape(Operation *op, LogicalDomainPhase phase);
  LogicalResult processCat(Operation *op, LogicalDomainPhase phase);
  LogicalResult processJoin(Operation *op, LogicalDomainPhase phase);
  LogicalResult processSplit(Operation *op, LogicalDomainPhase phase);
  LogicalResult processReduce(Operation *op, LogicalDomainPhase phase);
  LogicalResult processDot(Operation *op, LogicalDomainPhase phase);
  LogicalResult processStore(Operation *op, LogicalDomainPhase phase);
  LogicalResult processAtomicRMW(Operation *op, LogicalDomainPhase phase);
  LogicalResult processRejected(Operation *op, LogicalDomainPhase phase,
                                StringRef reason);

  LogicalResult emitError(Operation *op, unsigned operandIndex,
                          const Twine &reason) const;

private:
  LogicalDomainPlan &plan;
};

enum class LogicalBehavior : uint8_t {
  CandidateAlloc,
  CandidateDescriptor,
  MemDescIndex,
  LogicalTMACopy,
  LogicalPointerCopy,
  LocalStore,
  ExtractStage,
  WarpSpecialize,
  Pipe,
  MemDescUse,
  MemDescTranspose,
  WGMMA,
  WGMMAWait,
  ExpandDims,
  Broadcast,
  Transpose,
  Reshape,
  Cat,
  Join,
  Split,
  Reduce,
  Dot,
  Store,
  AtomicRMW,
  SameShape,
  RejectGather,
  RejectHistogram,
  RejectAtomicCAS,
  RejectLoad,
  RejectEscape,
  RejectScan,
};

static LogicalResult dispatchLogicalBehavior(LogicalDomainContext &context,
                                             Operation *op,
                                             LogicalDomainPhase phase,
                                             LogicalBehavior behavior) {
  switch (behavior) {
  case LogicalBehavior::CandidateDescriptor:
    return success();
  case LogicalBehavior::CandidateAlloc:
    return context.processCandidateAlloc(op, phase);
  case LogicalBehavior::MemDescIndex:
    return context.processMemDescIndex(op, phase);
  case LogicalBehavior::LogicalTMACopy:
    return context.processLogicalTMACopy(op, phase);
  case LogicalBehavior::LogicalPointerCopy:
    return context.processLogicalPointerCopy(op, phase);
  case LogicalBehavior::LocalStore:
    return context.processLocalStore(op, phase);
  case LogicalBehavior::ExtractStage:
    return context.processExtractStage(op, phase);
  case LogicalBehavior::WarpSpecialize:
    return context.processWarpSpecialize(op, phase);
  case LogicalBehavior::Pipe:
    return context.processPipe(op, phase);
  case LogicalBehavior::MemDescUse:
    return context.processMemDescUse(op, phase);
  case LogicalBehavior::MemDescTranspose:
    return context.processMemDescTranspose(op, phase);
  case LogicalBehavior::WGMMA:
    return context.processWGMMA(op, phase);
  case LogicalBehavior::WGMMAWait:
    return context.processWGMMAWait(op, phase);
  case LogicalBehavior::ExpandDims:
    return context.processExpandDims(op, phase);
  case LogicalBehavior::Broadcast:
    return context.processBroadcast(op, phase);
  case LogicalBehavior::Transpose:
    return context.processTranspose(op, phase);
  case LogicalBehavior::Reshape:
    return context.processReshape(op, phase);
  case LogicalBehavior::Cat:
    return context.processCat(op, phase);
  case LogicalBehavior::Join:
    return context.processJoin(op, phase);
  case LogicalBehavior::Split:
    return context.processSplit(op, phase);
  case LogicalBehavior::Reduce:
    return context.processReduce(op, phase);
  case LogicalBehavior::Dot:
    return context.processDot(op, phase);
  case LogicalBehavior::Store:
    return context.processStore(op, phase);
  case LogicalBehavior::AtomicRMW:
    return context.processAtomicRMW(op, phase);
  case LogicalBehavior::SameShape:
    return context.processSameShape(op, phase);
  case LogicalBehavior::RejectGather:
    return context.processRejected(
        op, phase, "gather does not preserve a rectangular logical prefix");
  case LogicalBehavior::RejectHistogram:
    return context.processRejected(
        op, phase, "histogram observes values outside the logical prefix");
  case LogicalBehavior::RejectAtomicCAS:
    return context.processRejected(
        op, phase, "atomic CAS is not a supported logical-domain terminal");
  case LogicalBehavior::RejectLoad:
    return context.processRejected(
        op, phase,
        "restricted values cannot participate in a load address or mask");
  case LogicalBehavior::RejectEscape:
    return context.processRejected(
        op, phase, "logical-domain value escapes the early TTIR function");
  case LogicalBehavior::RejectScan:
    return context.processRejected(
        op, phase,
        "attention fragment v1 does not support scan or exclusive_cumsum");
  }
  llvm_unreachable("unknown logical-domain behavior");
}

/// A shared SSA lattice for tensors, SMEM views and tensor descriptors. Full
/// denotes the ordinary carrier contract; the other facts retain explicit
/// logical domains. Memory initialization/alias effects are validated
/// separately. Unsupported transfers and incompatible control-flow joins never
/// become Full.
class LogicalDomainFact {
public:
  enum class Kind : uint8_t {
    Uninitialized,
    Full,
    Fragment,
    MemDesc,
    Descriptor,
    Conflict
  };

  LogicalDomainFact() = default;

  static LogicalDomainFact getPessimisticValueState(Value) { return getFull(); }

  static LogicalDomainFact getFull() {
    LogicalDomainFact fact;
    fact.kind = Kind::Full;
    return fact;
  }

  static LogicalDomainFact getFragment(Value value, int32_t axis,
                                       int64_t logicalExtent,
                                       LogicalDomainProvenance provenance) {
    auto type = getTensorType(value);
    if (!type || axis < 0 || axis >= type.getRank() || logicalExtent <= 0 ||
        logicalExtent > type.getShape()[axis])
      return getConflict();
    if (logicalExtent == type.getShape()[axis])
      return getFull();
    LogicalDomainFact fact;
    fact.kind = Kind::Fragment;
    fact.state.axis = axis;
    fact.state.logicalExtent = logicalExtent;
    fact.state.provenance = std::move(provenance);
    return fact;
  }

  static LogicalDomainFact getPrefix(Value value,
                                     ArrayRef<int64_t> logicalShape,
                                     LogicalDomainProvenance provenance) {
    auto type = getTensorType(value);
    if (!type || !validatePrefixShape(type.getShape(), logicalShape))
      return getConflict();
    std::optional<int32_t> fragmentAxis;
    for (auto [axis, extents] :
         llvm::enumerate(llvm::zip(type.getShape(), logicalShape))) {
      auto [physical, logical] = extents;
      if (physical == logical)
        continue;
      if (fragmentAxis)
        return getConflict();
      fragmentAxis = static_cast<int32_t>(axis);
    }
    if (!fragmentAxis)
      return getFull();
    return getFragment(value, *fragmentAxis, logicalShape[*fragmentAxis],
                       std::move(provenance));
  }

  static LogicalDomainFact getConflict() {
    LogicalDomainFact fact;
    fact.kind = Kind::Conflict;
    return fact;
  }

  static LogicalDomainFact getMemDesc(const MemDescLogicalState &state) {
    LogicalDomainFact fact;
    fact.kind = Kind::MemDesc;
    fact.memdesc = state;
    return fact;
  }

  bool isMemDesc() const { return kind == Kind::MemDesc; }
  const MemDescLogicalState &getMemDesc() const {
    assert(isMemDesc());
    return memdesc;
  }

  static LogicalDomainFact
  getDescriptor(const TensorDescriptorLogicalState &state) {
    LogicalDomainFact fact;
    fact.kind = Kind::Descriptor;
    fact.descriptor = state;
    return fact;
  }
  bool isDescriptor() const { return kind == Kind::Descriptor; }
  const TensorDescriptorLogicalState &getDescriptor() const {
    assert(isDescriptor());
    return descriptor;
  }

  Kind getKind() const { return kind; }
  bool isFragment() const { return kind == Kind::Fragment; }
  const TensorFragmentState &getState() const {
    assert(isFragment());
    return state;
  }

  // This join combines alternative definitions (CFG edges and loop carries).
  // Intersecting pointwise operands belongs to their operation's model, not
  // to this join: taking min at a CFG join could silently discard live lanes.
  static LogicalDomainFact join(const LogicalDomainFact &lhs,
                                const LogicalDomainFact &rhs) {
    if (lhs.kind == Kind::Uninitialized)
      return rhs;
    if (rhs.kind == Kind::Uninitialized)
      return lhs;
    if (lhs.kind != rhs.kind || lhs.kind == Kind::Conflict)
      return getConflict();
    if (lhs.kind == Kind::Full)
      return lhs;
    LogicalDomainFact result = lhs;
    if (lhs.kind == Kind::Descriptor) {
      if (lhs.descriptor.logicalShape != rhs.descriptor.logicalShape)
        return getConflict();
      result.descriptor.provenance =
          mergeProvenance(lhs.descriptor.provenance, rhs.descriptor.provenance);
      return result;
    }
    if (lhs.kind == Kind::MemDesc) {
      const auto &a = lhs.memdesc;
      const auto &b = rhs.memdesc;
      if (a.physicalShape != b.physicalShape ||
          a.logicalShape != b.logicalShape || a.axisMap != b.axisMap ||
          a.isStage != b.isStage || a.viewTransposed != b.viewTransposed ||
          a.provenance.primaryRoot() != b.provenance.primaryRoot())
        return getConflict();
      result.memdesc.provenance = mergeProvenance(a.provenance, b.provenance);
      return result;
    }
    if (lhs.state.axis != rhs.state.axis ||
        lhs.state.logicalExtent != rhs.state.logicalExtent)
      return getConflict();
    result.state.provenance =
        mergeProvenance(lhs.state.provenance, rhs.state.provenance);
    return result;
  }

  bool operator==(const LogicalDomainFact &other) const {
    if (kind != other.kind)
      return false;
    if (kind == Kind::Descriptor)
      return descriptor.logicalShape == other.descriptor.logicalShape &&
             descriptor.provenance.roots == other.descriptor.provenance.roots &&
             descriptor.provenance.seeds == other.descriptor.provenance.seeds;
    if (kind == Kind::MemDesc)
      return memdesc.physicalShape == other.memdesc.physicalShape &&
             memdesc.logicalShape == other.memdesc.logicalShape &&
             memdesc.axisMap == other.memdesc.axisMap &&
             memdesc.isStage == other.memdesc.isStage &&
             memdesc.viewTransposed == other.memdesc.viewTransposed &&
             memdesc.provenance.roots == other.memdesc.provenance.roots &&
             memdesc.provenance.seeds == other.memdesc.provenance.seeds;
    if (kind != Kind::Fragment)
      return true;
    return state.axis == other.state.axis &&
           state.logicalExtent == other.state.logicalExtent &&
           state.provenance.roots == other.state.provenance.roots &&
           state.provenance.seeds == other.state.provenance.seeds;
  }

  void print(raw_ostream &os) const {
    switch (kind) {
    case Kind::Uninitialized:
      os << "uninitialized";
      return;
    case Kind::Full:
      os << "full";
      return;
    case Kind::MemDesc:
      os << "memdesc<";
      llvm::interleaveComma(memdesc.logicalShape, os);
      os << ">";
      return;
    case Kind::Descriptor:
      os << "tensor_descriptor<";
      llvm::interleaveComma(descriptor.logicalShape, os);
      os << ">";
      return;
    case Kind::Conflict:
      os << "conflict";
      return;
    case Kind::Fragment:
      os << "fragment<axis=" << state.axis << ", extent=" << state.logicalExtent
         << '>';
      return;
    }
    llvm_unreachable("unknown tensor fact");
  }

private:
  Kind kind = Kind::Uninitialized;
  TensorFragmentState state;
  MemDescLogicalState memdesc;
  TensorDescriptorLogicalState descriptor;
};

using LogicalDomainLattice = dataflow::Lattice<LogicalDomainFact>;

static const TensorFragmentState *
getFragmentState(ArrayRef<const LogicalDomainLattice *> operands,
                 unsigned index) {
  if (index >= operands.size() || !operands[index]->getValue().isFragment())
    return nullptr;
  return &operands[index]->getValue().getState();
}

static LogicalShape materializeLogicalShape(Value value,
                                            const TensorFragmentState &state) {
  auto type = getTensorType(value);
  assert(type && state.axis >= 0 && state.axis < type.getRank());
  LogicalShape logicalShape(type.getShape());
  logicalShape[state.axis] = state.logicalExtent;
  return logicalShape;
}

static LogicalDomainFact
inferSameShapeFact(Operation *op, Value result,
                   ArrayRef<const LogicalDomainLattice *> operands) {
  auto resultType = getTensorType(result);
  if (!resultType)
    return LogicalDomainFact::getFull();
  std::optional<int32_t> fragmentAxis;
  int64_t logicalExtent = 0;
  LogicalDomainProvenance provenance;
  for (auto [operandValue, operand] : llvm::zip(op->getOperands(), operands)) {
    const LogicalDomainFact &fact = operand->getValue();
    if (fact.getKind() == LogicalDomainFact::Kind::Conflict)
      return LogicalDomainFact::getConflict();
    if (!fact.isFragment())
      continue;
    auto operandType = getTensorType(operandValue);
    if (!operandType || operandType.getShape() != resultType.getShape())
      continue;
    const TensorFragmentState &state = fact.getState();
    if (fragmentAxis && *fragmentAxis != state.axis)
      return LogicalDomainFact::getConflict();
    if (!fragmentAxis) {
      fragmentAxis = state.axis;
      logicalExtent = state.logicalExtent;
    } else {
      logicalExtent = std::min(logicalExtent, state.logicalExtent);
    }
    provenance = mergeProvenance(std::move(provenance), state.provenance);
  }
  if (!fragmentAxis)
    return LogicalDomainFact::getFull();
  return LogicalDomainFact::getFragment(result, *fragmentAxis, logicalExtent,
                                        std::move(provenance));
}

static bool isGenericElementwiseTransfer(Operation *op) {
  return op->getNumRegions() == 0 && op->hasTrait<OpTrait::Elementwise>() &&
         isMemoryEffectFree(op);
}

static bool isMapElementwiseTransfer(Operation *op) {
  return isa<triton::MapElementwiseOp>(op) && isMemoryEffectFree(op);
}

static SmallVector<LogicalDomainFact, 2>
inferTensorDomains(Operation *op, LogicalBehavior behavior,
                   ArrayRef<const LogicalDomainLattice *> operands) {
  SmallVector<LogicalDomainFact, 2> inferred(op->getNumResults(),
                                             LogicalDomainFact::getFull());
  auto setResult = [&](unsigned index, LogicalDomainFact fact) {
    if (index < inferred.size())
      inferred[index] = std::move(fact);
  };
  auto inferSameShape = [&] {
    for (auto [index, result] : llvm::enumerate(op->getResults()))
      setResult(index, inferSameShapeFact(op, result, operands));
  };

  if (behavior == LogicalBehavior::WGMMA) {
    auto dot = cast<WGMMAOp>(op);
    auto resultType = getTensorType(dot.getD());
    if (!resultType || resultType.getRank() != 2)
      return inferred;
    std::optional<int32_t> fragmentAxis;
    int64_t logicalExtent = 0;
    LogicalDomainProvenance provenance;
    bool conflict = false;
    auto mergeOutputFragment = [&](int32_t axis, int64_t extent,
                                   const LogicalDomainProvenance &source) {
      if (extent == resultType.getShape()[axis])
        return;
      if (fragmentAxis && *fragmentAxis != axis) {
        conflict = true;
        return;
      }
      if (!fragmentAxis) {
        fragmentAxis = axis;
        logicalExtent = extent;
      } else {
        logicalExtent = std::min(logicalExtent, extent);
      }
      provenance = mergeProvenance(std::move(provenance), source);
    };
    if (operands[1]->getValue().isMemDesc()) {
      const auto &b = operands[1]->getValue().getMemDesc();
      if (b.logicalShape.size() == 2 && b.logicalShape[1] != b.physicalShape[1])
        mergeOutputFragment(1, b.logicalShape[1], b.provenance);
    }
    if (const TensorFragmentState *aState = getFragmentState(operands, 0)) {
      if (aState->axis == 0)
        mergeOutputFragment(0, aState->logicalExtent, aState->provenance);
      else if (aState->axis != 1)
        conflict = true;
    }
    if (const TensorFragmentState *cState = getFragmentState(operands, 2)) {
      if (cState->axis < 0 || cState->axis >= 2)
        conflict = true;
      else
        mergeOutputFragment(cState->axis, cState->logicalExtent,
                            cState->provenance);
    }
    if (conflict) {
      setResult(0, LogicalDomainFact::getConflict());
    } else if (fragmentAxis) {
      mergeProvenanceInto(provenance, {nullptr, dot});
      setResult(0, LogicalDomainFact::getFragment(dot.getD(), *fragmentAxis,
                                                  logicalExtent,
                                                  std::move(provenance)));
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::WGMMAWait) {
    inferSameShape();
    return inferred;
  }

  if (behavior == LogicalBehavior::ExpandDims) {
    auto expand = cast<triton::ExpandDimsOp>(op);
    if (const TensorFragmentState *state = getFragmentState(operands, 0)) {
      int32_t resultAxis =
          state->axis >= expand.getAxis() ? state->axis + 1 : state->axis;
      setResult(0, LogicalDomainFact::getFragment(
                       expand.getResult(), resultAxis, state->logicalExtent,
                       state->provenance));
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Broadcast) {
    auto broadcast = cast<triton::BroadcastOp>(op);
    if (const TensorFragmentState *state = getFragmentState(operands, 0)) {
      auto srcType = getTensorType(broadcast.getSrc());
      auto resultType = getTensorType(broadcast.getResult());
      if (srcType && resultType && srcType.getRank() == resultType.getRank() &&
          srcType.getShape()[state->axis] == resultType.getShape()[state->axis])
        setResult(0, LogicalDomainFact::getFragment(
                         broadcast.getResult(), state->axis,
                         state->logicalExtent, state->provenance));
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Transpose) {
    auto transpose = cast<triton::TransOp>(op);
    if (const TensorFragmentState *state = getFragmentState(operands, 0)) {
      auto resultAxis = llvm::find(transpose.getOrder(), state->axis);
      if (resultAxis != transpose.getOrder().end())
        setResult(0, LogicalDomainFact::getFragment(
                         transpose.getResult(),
                         static_cast<int32_t>(std::distance(
                             transpose.getOrder().begin(), resultAxis)),
                         state->logicalExtent, state->provenance));
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Reshape) {
    auto reshape = cast<triton::ReshapeOp>(op);
    if (const TensorFragmentState *state = getFragmentState(operands, 0)) {
      auto srcType = getTensorType(reshape.getSrc());
      auto resultType = getTensorType(reshape.getResult());
      std::optional<LogicalShape> shape;
      LogicalShape sourceLogical =
          materializeLogicalShape(reshape.getSrc(), *state);
      if (srcType && resultType && !reshape.getAllowReorder()) {
        if (srcType.getShape() == resultType.getShape()) {
          shape = sourceLogical;
        } else {
          shape = reshapeOnlyUnitDimensions(srcType.getShape(), sourceLogical,
                                            resultType.getShape());
          if (!shape)
            if (auto prefix = getContiguousLinearPrefixSize(srcType.getShape(),
                                                            sourceLogical))
              shape = representLinearPrefixAsRectangle(resultType.getShape(),
                                                       *prefix);
        }
      }
      setResult(0, shape ? LogicalDomainFact::getPrefix(
                               reshape.getResult(), *shape, state->provenance)
                         : LogicalDomainFact::getConflict());
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Cat) {
    auto cat = cast<triton::CatOp>(op);
    const TensorFragmentState *lhs = getFragmentState(operands, 0);
    const TensorFragmentState *rhs = getFragmentState(operands, 1);
    auto lhsType = getTensorType(cat.getLhs());
    auto rhsType = getTensorType(cat.getRhs());
    auto resultType = getTensorType(cat.getResult());
    if ((lhs || rhs) && lhsType && rhsType && resultType &&
        resultType.getRank() > 0 && lhsType.getRank() == resultType.getRank() &&
        rhsType.getRank() == resultType.getRank()) {
      LogicalShape left = lhs ? materializeLogicalShape(cat.getLhs(), *lhs)
                              : LogicalShape(lhsType.getShape());
      LogicalShape right = rhs ? materializeLogicalShape(cat.getRhs(), *rhs)
                               : LogicalShape(rhsType.getShape());
      LogicalShape shape(resultType.getShape());
      for (unsigned i = 0; i + 1 < shape.size(); ++i)
        shape[i] = std::min(left[i], right[i]);
      unsigned axis = shape.size() - 1;
      shape[axis] = left[axis] == lhsType.getShape()[axis]
                        ? lhsType.getShape()[axis] + right[axis]
                        : left[axis];
      setResult(0, LogicalDomainFact::getPrefix(
                       cat.getResult(), shape,
                       mergeProvenance(
                           lhs ? lhs->provenance : LogicalDomainProvenance{},
                           rhs ? rhs->provenance : LogicalDomainProvenance{})));
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Join) {
    auto join = cast<triton::JoinOp>(op);
    const TensorFragmentState *lhs = getFragmentState(operands, 0);
    const TensorFragmentState *rhs = getFragmentState(operands, 1);
    auto lhsType = getTensorType(join.getLhs());
    auto rhsType = getTensorType(join.getRhs());
    if ((lhs || rhs) && lhsType && rhsType &&
        lhsType.getShape() == rhsType.getShape()) {
      LogicalShape left = lhs ? materializeLogicalShape(join.getLhs(), *lhs)
                              : LogicalShape(lhsType.getShape());
      LogicalShape right = rhs ? materializeLogicalShape(join.getRhs(), *rhs)
                               : LogicalShape(rhsType.getShape());
      LogicalShape shape = intersectShapes(left, right);
      shape.push_back(2);
      setResult(0, LogicalDomainFact::getPrefix(
                       join.getResult(), shape,
                       mergeProvenance(
                           lhs ? lhs->provenance : LogicalDomainProvenance{},
                           rhs ? rhs->provenance : LogicalDomainProvenance{})));
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Split) {
    auto split = cast<triton::SplitOp>(op);
    if (const TensorFragmentState *state = getFragmentState(operands, 0)) {
      LogicalShape sourceLogical =
          materializeLogicalShape(split.getSrc(), *state);
      if (!sourceLogical.empty() && sourceLogical.back() == 2) {
        LogicalShape shape = std::move(sourceLogical);
        shape.pop_back();
        for (auto [index, result] : llvm::enumerate(split.getResults()))
          setResult(index, LogicalDomainFact::getPrefix(result, shape,
                                                        state->provenance));
      }
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Reduce) {
    auto reduce = cast<triton::ReduceOp>(op);
    int32_t axis = reduce.getAxis();
    std::optional<LogicalShape> reducedShape;
    LogicalDomainProvenance provenance;
    for (unsigned index = 0; index < reduce.getSrcs().size(); ++index) {
      const TensorFragmentState *state = getFragmentState(operands, index);
      if (!state)
        continue;
      LogicalShape shape =
          materializeLogicalShape(reduce.getSrcs()[index], *state);
      if (axis < 0 || axis >= static_cast<int32_t>(shape.size()))
        continue;
      shape.erase(shape.begin() + axis);
      reducedShape = reducedShape
                         ? std::optional<LogicalShape>(
                               intersectShapes(*reducedShape, shape))
                         : std::optional<LogicalShape>(std::move(shape));
      provenance = mergeProvenance(std::move(provenance), state->provenance);
    }
    if (reducedShape) {
      for (auto [index, result] : llvm::enumerate(reduce.getResults())) {
        if (!getTensorType(result))
          continue;
        setResult(index, LogicalDomainFact::getPrefix(result, *reducedShape,
                                                      provenance));
      }
    }
    return inferred;
  }

  if (behavior == LogicalBehavior::Dot) {
    if (op->getNumOperands() < 3 || op->getNumResults() != 1)
      return inferred;
    const TensorFragmentState *aState = getFragmentState(operands, 0);
    const TensorFragmentState *bState = getFragmentState(operands, 1);
    const TensorFragmentState *cState = getFragmentState(operands, 2);
    auto aType = getTensorType(op->getOperand(0));
    auto bType = getTensorType(op->getOperand(1));
    auto resultType = getTensorType(op->getResult(0));
    if ((!aState && !bState && !cState) || !aType || !bType || !resultType ||
        aType.getRank() != 2 || bType.getRank() != 2 ||
        resultType.getRank() != 2)
      return inferred;
    std::optional<int32_t> resultAxis;
    int64_t resultExtent = 0;
    LogicalDomainProvenance provenance;
    auto mergeResultFragment = [&](int32_t axis, int64_t extent,
                                   const LogicalDomainProvenance &source) {
      if (resultAxis && *resultAxis != axis)
        return false;
      if (!resultAxis) {
        resultAxis = axis;
        resultExtent = extent;
      } else {
        resultExtent = std::min(resultExtent, extent);
      }
      provenance = mergeProvenance(std::move(provenance), source);
      return true;
    };
    bool valid = true;
    if (aState && aState->axis == 0)
      valid &=
          mergeResultFragment(0, aState->logicalExtent, aState->provenance);
    if (bState && bState->axis == 1)
      valid &=
          mergeResultFragment(1, bState->logicalExtent, bState->provenance);
    if (cState)
      valid &= mergeResultFragment(cState->axis, cState->logicalExtent,
                                   cState->provenance);
    if (!valid)
      setResult(0, LogicalDomainFact::getConflict());
    else if (resultAxis)
      setResult(0, LogicalDomainFact::getFragment(op->getResult(0), *resultAxis,
                                                  resultExtent,
                                                  std::move(provenance)));
    return inferred;
  }

  if (behavior == LogicalBehavior::ExtractStage) {
    if (const auto *state = getFragmentState(operands, 0))
      setResult(0, state->axis > 0
                       ? LogicalDomainFact::getFragment(
                             op->getResult(0), state->axis,
                             state->logicalExtent, state->provenance)
                       : LogicalDomainFact::getConflict());
    return inferred;
  }

  if (behavior == LogicalBehavior::SameShape)
    inferSameShape();
  return inferred;
}

static std::optional<LogicalBehavior> getExplicitLogicalBehavior(Operation *op);

struct LogicalDomainModel {
  LogicalBehavior behavior;

  FailureOr<SmallVector<LogicalDomainFact, 2>>
  infer(Operation *op, ArrayRef<const LogicalDomainLattice *> operands) const {
    switch (behavior) {
    case LogicalBehavior::CandidateDescriptor: {
      auto shape =
          op->getAttrOfType<DenseI64ArrayAttr>(kLogicalDescriptorShape);
      if (!shape)
        return SmallVector<LogicalDomainFact, 2>{LogicalDomainFact::getFull()};
      auto block = cast<triton::MakeTensorDescOp>(op).getType().getBlockType();
      ArrayRef<int64_t> logical = shape.asArrayRef();
      if (logical.size() < 2 ||
          !validatePrefixShape(block.getShape(), logical) ||
          llvm::any_of(logical.drop_back(2),
                       [](int64_t dim) { return dim != 1; }) ||
          llvm::count_if(
              logical, [](int64_t dim) { return !llvm::isPowerOf2_64(dim); }) !=
              1 ||
          llvm::any_of(logical.take_back(2),
                       [](int64_t dim) { return dim % 16; })) {
        op->emitOpError("invalid TLE logical descriptor block shape");
        return failure();
      }
      return SmallVector<LogicalDomainFact, 2>{LogicalDomainFact::getDescriptor(
          {LogicalShape(logical), LogicalDomainProvenance(op, op)})};
    }
    case LogicalBehavior::CandidateAlloc:
    case LogicalBehavior::MemDescIndex:
    case LogicalBehavior::MemDescTranspose: {
      // A local evaluation snapshot, not the final plan or a shared side map.
      // Every read comes from operand lattices and is tracked by the solver.
      LogicalDomainPlan facts;
      for (auto [operand, lattice] : llvm::zip(op->getOperands(), operands)) {
        const auto &fact = lattice->getValue();
        if (fact.isMemDesc())
          facts.memdescs.try_emplace(operand, fact.getMemDesc());
        else if (fact.isFragment())
          facts.tensors.try_emplace(operand, fact.getState());
      }
      LogicalDomainContext context(facts);
      if (failed(dispatchLogicalBehavior(
              context, op, LogicalDomainPhase::Propagate, behavior)))
        return failure();
      SmallVector<LogicalDomainFact, 2> outputs;
      for (Value result : op->getResults()) {
        auto it = facts.memdescs.find(result);
        outputs.push_back(it == facts.memdescs.end()
                              ? LogicalDomainFact::getFull()
                              : LogicalDomainFact::getMemDesc(it->second));
      }
      return outputs;
    }
    case LogicalBehavior::RejectGather:
    case LogicalBehavior::RejectHistogram:
    case LogicalBehavior::RejectAtomicCAS:
    case LogicalBehavior::RejectLoad:
    case LogicalBehavior::RejectEscape:
    case LogicalBehavior::RejectScan: {
      bool restricted = llvm::any_of(operands, [](const auto *lattice) {
        const auto &fact = lattice->getValue();
        return fact.isFragment() || fact.isMemDesc() || fact.isDescriptor();
      });
      return SmallVector<LogicalDomainFact, 2>(
          op->getNumResults(), restricted ? LogicalDomainFact::getConflict()
                                          : LogicalDomainFact::getFull());
    }
    default:
      return inferTensorDomains(op, behavior, operands);
    }
  }

  LogicalResult collectRequirements(LogicalDomainContext &context,
                                    Operation *op) const {
    return dispatchLogicalBehavior(context, op, LogicalDomainPhase::Plan,
                                   behavior);
  }
};

static std::optional<LogicalDomainModel> getLogicalDomainModel(Operation *op) {
  if (auto behavior = getExplicitLogicalBehavior(op))
    return LogicalDomainModel{*behavior};
  if (isMapElementwiseTransfer(op) || isGenericElementwiseTransfer(op))
    return LogicalDomainModel{LogicalBehavior::SameShape};
  return std::nullopt;
}

class LogicalDomainAnalysis final
    : public dataflow::SparseForwardDataFlowAnalysis<LogicalDomainLattice> {
  using Base = dataflow::SparseForwardDataFlowAnalysis<LogicalDomainLattice>;

public:
  using Base::Base;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<const LogicalDomainLattice *> operands,
                 ArrayRef<LogicalDomainLattice *> results) override {
    // Never publish Full merely because a descriptor/tensor input is pending.
    if (llvm::any_of(operands, [](const auto *lattice) {
          return lattice->getValue().getKind() ==
                 LogicalDomainFact::Kind::Uninitialized;
        }))
      return success();
    bool conflict = llvm::any_of(operands, [](const auto *lattice) {
      return lattice->getValue().getKind() == LogicalDomainFact::Kind::Conflict;
    });
    auto model = getLogicalDomainModel(op);
    bool restricted = llvm::any_of(operands, [](const auto *lattice) {
      const auto &fact = lattice->getValue();
      return fact.isFragment() || fact.isMemDesc() || fact.isDescriptor();
    });
    if (conflict || (!model && restricted)) {
      for (auto *result : results)
        propagateIfChanged(result,
                           result->join(LogicalDomainFact::getConflict()));
      return success();
    }
    FailureOr<SmallVector<LogicalDomainFact, 2>> inferred =
        model ? model->infer(op, operands)
              : SmallVector<LogicalDomainFact, 2>(results.size(),
                                                  LogicalDomainFact::getFull());
    if (failed(inferred))
      return failure();
    for (auto [result, fact] : llvm::zip(results, *inferred))
      propagateIfChanged(result, result->join(fact));
    return success();
  }

private:
  void setToEntryState(LogicalDomainLattice *lattice) override {
    // Upstream RegionBranch handles the default region. Partition captures
    // are hidden inside an isolated holder and need an explicit dependency.
    if (auto argument = dyn_cast<BlockArgument>(lattice->getAnchor())) {
      if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
              argument.getOwner()->getParentOp())) {
        auto warp = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
        Value capture = warp.getExplicitCaptures()[argument.getArgNumber()];
        const auto *source = getLatticeElementFor(
            getProgramPointBefore(argument.getOwner()), capture);
        propagateIfChanged(lattice, lattice->join(source->getValue()));
        return;
      }
    }
    propagateIfChanged(lattice, lattice->join(LogicalDomainFact::getFull()));
  }
};

} // namespace

LogicalDomainContext::LogicalDomainContext(LogicalDomainPlan &plan)
    : plan(plan) {}

const MemDescLogicalState *
LogicalDomainContext::lookupMemDesc(Value value) const {
  auto it = plan.memdescs.find(value);
  return it == plan.memdescs.end() ? nullptr : &it->second;
}
const TensorDescriptorLogicalState *
LogicalDomainContext::lookupDescriptor(Value value) const {
  auto it = plan.descriptors.find(value);
  return it == plan.descriptors.end() ? nullptr : &it->second;
}
const TensorFragmentState *
LogicalDomainContext::lookupTensor(Value value) const {
  auto it = plan.tensors.find(value);
  return it == plan.tensors.end() ? nullptr : &it->second;
}

bool LogicalDomainContext::mergeMemDesc(Value value,
                                        const MemDescLogicalState &state) {
  assert(validatePrefixShape(state.physicalShape, state.logicalShape));
  return plan.memdescs.try_emplace(value, state).second;
}

bool LogicalDomainContext::hasRestrictedOperand(Operation *op) const {
  return llvm::any_of(op->getOperands(), [&](Value value) {
    return lookupTensor(value) || lookupMemDesc(value) ||
           lookupDescriptor(value);
  });
}

LogicalResult
LogicalDomainContext::validateTensorResult(Operation *operation) const {
  std::optional<unsigned> restrictedOperand;
  for (auto [index, operand] : llvm::enumerate(operation->getOperands()))
    if (lookupTensor(operand)) {
      restrictedOperand = index;
      break;
    }
  if (!restrictedOperand)
    return success();
  if (llvm::any_of(operation->getResults(),
                   [&](Value result) { return lookupTensor(result); }))
    return success();
  return emitError(operation, *restrictedOperand,
                   "logical-domain transfer produced no restricted tensor "
                   "result");
}
LogicalResult LogicalDomainContext::emitError(Operation *op,
                                              unsigned operandIndex,
                                              const Twine &reason) const {
  InFlightDiagnostic diag = op->emitOpError();
  diag << "logical-domain operand " << operandIndex << " rejected: " << reason;
  if (operandIndex >= op->getNumOperands())
    return failure();
  Value value = op->getOperand(operandIndex);
  if (const TensorFragmentState *state = lookupTensor(value)) {
    auto type = getTensorType(value);
    LogicalShape logicalShape = materializeLogicalShape(value, *state);
    diag << "; physical=" << type.getShape() << ", logical=" << logicalShape;
    for (Operation *root : state->provenance.roots)
      diag.attachNote(root->getLoc()) << "logical domain root is here";
    for (Operation *seed : state->provenance.seeds) {
      if (llvm::is_contained(state->provenance.roots, seed))
        continue;
      diag.attachNote(seed->getLoc()) << "logical tensor seed is here";
    }
  } else if (const MemDescLogicalState *state = lookupMemDesc(value)) {
    diag << "; physical=" << state->physicalShape
         << ", logical=" << state->logicalShape;
    for (Operation *root : state->provenance.roots)
      diag.attachNote(root->getLoc()) << "logical domain root is here";
  }
  return failure();
}

LogicalResult
LogicalDomainContext::processCandidateAlloc(Operation *operation,
                                            LogicalDomainPhase phase) {
  auto alloc = cast<ttg::LocalAllocOp>(operation);
  auto storageAttr = alloc->getAttrOfType<StringAttr>(kStoragePlan);
  if (!storageAttr) {
    if (hasRestrictedOperand(alloc))
      return processRejected(
          alloc, phase,
          "ordinary local allocation cannot consume a restricted logical "
          "tensor");
    return success();
  }
  if (storageAttr.getValue() != "candidate")
    return alloc.emitOpError("unknown TLE logical storage plan");
  auto logicalAttr =
      alloc->getAttrOfType<DenseI64ArrayAttr>(kLogicalAllocShape);
  auto axisAttr = alloc->getAttrOfType<IntegerAttr>(kLogicalNonPowerAxis);
  if (!logicalAttr || !axisAttr)
    return alloc.emitOpError("has incomplete logical candidate metadata");
  auto type = alloc.getType();
  ArrayRef<int64_t> logical = logicalAttr.asArrayRef();
  int64_t fragmentAxis = axisAttr.getInt();
  if (logical.size() != 3 || type.getRank() != 3 || fragmentAxis < 1 ||
      fragmentAxis >= 3)
    return alloc.emitOpError(
        "logical candidate requires an explicit capacity and a rank-2 payload");
  int64_t capacity = logical[0];
  if (capacity <= 0 || capacity > std::numeric_limits<int32_t>::max())
    return alloc.emitOpError(
        "logical capacity must fit a positive i32 stage index");
  if (llvm::isPowerOf2_64(logical[fragmentAxis]) ||
      llvm::any_of(llvm::enumerate(logical.drop_front()),
                   [&](auto indexedExtent) {
                     int64_t rootAxis = indexedExtent.index() + 1;
                     return rootAxis != fragmentAxis &&
                            !llvm::isPowerOf2_64(indexedExtent.value());
                   }))
    return alloc.emitOpError(
        "logical candidate metadata must identify its only non-power-of-two "
        "payload axis");
  if (llvm::any_of(logical.drop_front(),
                   [](int64_t extent) { return extent <= 0; }))
    return alloc.emitOpError("logical payload extents must be positive");
  if (logical[fragmentAxis] % 16 != 0)
    return alloc.emitOpError(
        "logical fragment extent must be a multiple of 16");
  if (!isSupportedExactSMEMElementType(type.getElementType()))
    return alloc.emitOpError(
        "logical candidate requires a Hopper WGMMA-compatible element type");
  if (!type.getMutableMemory() ||
      !isa<ttg::SharedMemorySpaceAttr>(type.getMemorySpace()))
    return alloc.emitOpError(
        "logical candidate requires mutable shared-memory storage");
  LogicalShape expected(logical.begin(), logical.end());
  expected[fragmentAxis] = nextPowerOfTwo(expected[fragmentAxis]);
  if (type.getShape() != ArrayRef<int64_t>(expected) ||
      type.getAllocShape() != ArrayRef<int64_t>(expected))
    return alloc.emitOpError(
        "logical candidate carrier type does not match its padded shape");
  if (Value initializer = alloc.getSrc()) {
    auto tensorType = getTensorType(initializer);
    if (!tensorType || tensorType.getEncoding() ||
        tensorType.getShape() != ArrayRef<int64_t>(expected) ||
        tensorType.getElementType() != type.getElementType())
      return alloc.emitOpError(
          "logical initializer must be an unencoded tensor matching the "
          "complete padded allocation carrier");
    if (const auto *fragment = lookupTensor(initializer)) {
      if (ArrayRef<int64_t>(materializeLogicalShape(initializer, *fragment)) !=
          logical)
        return emitError(alloc, 0,
                         "initializer logical domain does not match the "
                         "declared allocation domain");
    }
  }
  if (phase == LogicalDomainPhase::Propagate) {
    MemDescLogicalState state;
    state.physicalShape = expected;
    state.logicalShape.assign(logical.begin(), logical.end());
    state.axisMap = identityAxisMap(logical.size());
    state.provenance = {alloc, alloc};
    mergeMemDesc(alloc.getResult(), state);
  } else if (!findRootAction(plan, alloc)) {
    LogicalRootRewriteAction action;
    action.alloc = alloc;
    action.logicalShape.assign(logical.begin(), logical.end());
    action.initializer = alloc.getSrc();
    plan.roots.push_back(std::move(action));
  }
  return success();
}

LogicalResult
LogicalDomainContext::processMemDescIndex(Operation *operation,
                                          LogicalDomainPhase phase) {
  auto index = cast<ttg::MemDescIndexOp>(operation);
  const MemDescLogicalState *source = lookupMemDesc(index.getSrc());
  auto rootType = cast<ttg::MemDescType>(index.getSrc().getType());
  int64_t capacity = source ? source->logicalShape[0] : rootType.getShape()[0];
  if (auto constant = index.getIndex().getDefiningOp<arith::ConstantIntOp>()) {
    int64_t stage = constant.value();
    if (stage < 0 || stage >= capacity)
      return emitError(index, 1, "stage index exceeds capacity");
  }
  if (!source)
    return success();
  auto stageType = dyn_cast<ttg::MemDescType>(index.getType());
  if (!stageType || source->logicalShape.size() != 3)
    return emitError(index, 0,
                     "logical stage requires capacity plus a rank-2 payload");
  SmallVector<int64_t> expectedShape(source->physicalShape.begin() + 1,
                                     source->physicalShape.end());
  auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(stageType.getEncoding());
  if (stageType.getShape() != ArrayRef<int64_t>(expectedShape) ||
      stageType.getAllocShape() != ArrayRef<int64_t>(expectedShape) ||
      stageType.getElementType() != rootType.getElementType() ||
      stageType.getMemorySpace() != rootType.getMemorySpace() ||
      !stageType.getMutableMemory() || !nvmma ||
      nvmma.getElementBitWidth() !=
          stageType.getElementType().getIntOrFloatBitWidth())
    return emitError(
        index, 0,
        "planned tiled stage requires a mutable NVMMAShared carrier whose "
        "bitwidth matches the candidate storage");
  if (!index.getIndex().getType().isInteger(32))
    return emitError(index, 1, "planned tiled stage index must be i32");
  if (phase == LogicalDomainPhase::Propagate) {
    MemDescLogicalState state = *source;
    state.physicalShape.erase(state.physicalShape.begin());
    state.logicalShape.erase(state.logicalShape.begin());
    state.axisMap.erase(state.axisMap.begin());
    state.isStage = true;
    mergeMemDesc(index.getResult(), state);
  } else {
    auto *root = findRootAction(plan, source->provenance.primaryRoot());
    if (!root)
      return emitError(index, 0, "candidate root has no storage action");
    if (!llvm::is_contained(root->stages, index))
      root->stages.push_back(index);
  }
  return success();
}

LogicalResult
LogicalDomainContext::processLogicalTMACopy(Operation *operation,
                                            LogicalDomainPhase phase) {
  auto copy = cast<ttg::TMACopyOp>(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const MemDescLogicalState *state = lookupMemDesc(operand);
    if (!state)
      continue;
    if (index != 1)
      return emitError(copy, index,
                       "logical TMA supports only global-to-SMEM direction");
#ifndef __HCU__
    if (copy.getBarrier()) {
      auto expectBytes = copy.getExpectBytesAttr();
      if (!expectBytes || expectBytes.getInt() <= 0)
        return emitError(copy, index,
                         "explicit completion barrier requires positive "
                         "expect_bytes");
    }
#endif
    auto descType = dyn_cast<triton::TensorDescType>(copy.getSrc().getType());
    auto dstType = dyn_cast<ttg::MemDescType>(copy.getDst().getType());
    if (!descType || !dstType || state->logicalShape.size() != 2 ||
        dstType.getRank() != 2)
      return emitError(copy, index,
                       "logical TMA requires a rank-2 stage and tensor "
                       "descriptor source");
    if (!state->isStage || state->viewTransposed ||
        !copy.getDst().getDefiningOp<ttg::MemDescIndexOp>())
      return emitError(copy, index,
                       "logical TMA destination must be a direct stage view");
    RankedTensorType blockType = descType.getSignlessBlockType();
    const auto *descriptor = lookupDescriptor(copy.getSrc());
    if (!descriptor ||
        ArrayRef<int64_t>(descriptor->logicalShape).take_back(2) !=
            ArrayRef<int64_t>(state->logicalShape))
      return emitError(copy, 0,
                       "descriptor logical block_shape must match the stage; "
                       "copy.shape cannot enlarge a descriptor block");
    ArrayRef<int64_t> blockShape = descriptor->logicalShape;
    auto logical =
        copy->getAttrOfType<DenseI64ArrayAttr>(kLogicalCopyShapeAttr);
    if (!logical || logical.asArrayRef() != blockShape)
      return emitError(copy, index,
                       "copy shape must match descriptor logical block_shape");
    if (copy.getIndices().size() != blockShape.size())
      return emitError(copy, index,
                       "logical TMA coordinate count must match descriptor "
                       "rank");
    if (blockType.getElementType() != dstType.getElementType())
      return emitError(copy, index,
                       "logical TMA source and destination element types "
                       "must match");
#ifndef __HCU__
    if (copy.getBarrier()) {
      auto expectBytes = copy.getExpectBytesAttr();
      int64_t elementBytes =
          blockType.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t logicalBytes =
          state->logicalShape[0] * state->logicalShape[1] * elementBytes;
      // A barrier may cover multiple logical copies, but every complete
      // logical copy contributes exactly this many bytes.
      if (expectBytes.getInt() % logicalBytes != 0)
        return emitError(copy, index,
                         "expect_bytes must be a multiple of the logical TMA "
                         "byte count (" +
                             Twine(logicalBytes) + ")");
    }
#endif
    if (phase == LogicalDomainPhase::Plan) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(copy, index, "candidate root has no storage action");
      if (!llvm::is_contained(root->copies, copy))
        root->copies.push_back(copy);
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processLogicalPointerCopy(Operation *operation,
                                                LogicalDomainPhase phase) {
  auto pointers = cast<LocalPointersOp>(operation);
  const MemDescLogicalState *state = lookupMemDesc(pointers.getSrc());
  if (!state)
    return success();
  if (phase == LogicalDomainPhase::Propagate)
    return success();

  auto srcType = dyn_cast<ttg::MemDescType>(pointers.getSrc().getType());
  auto ptrType = dyn_cast<RankedTensorType>(pointers.getResult().getType());
  if (!state->isStage || state->viewTransposed ||
      !pointers.getSrc().getDefiningOp<ttg::MemDescIndexOp>() || !srcType ||
      !ptrType || state->logicalShape.size() != 2 || srcType.getRank() != 2)
    return emitError(
        pointers, 0,
        "logical pointer copy requires a direct rank-2 stage destination");

  auto logical =
      pointers->getAttrOfType<DenseI64ArrayAttr>(kLogicalCopyShapeAttr);
  if (!logical ||
      logical.asArrayRef() != ArrayRef<int64_t>(state->logicalShape))
    return emitError(
        pointers, 0,
        "logical pointer copy shape must match the logical stage shape");
  if (ptrType.getShape() != srcType.getShape() ||
      ptrType.getShape() != ArrayRef<int64_t>(state->physicalShape) ||
      pointers.getIndices().size() != 2)
    return emitError(
        pointers, 0,
        "logical pointer copy must address the complete padded stage carrier");
  for (Value index : pointers.getIndices()) {
    auto indexType = dyn_cast<RankedTensorType>(index.getType());
    if (!indexType || indexType.getShape() != ptrType.getShape())
      return emitError(
          pointers, 0,
          "logical pointer copy indices must cover the padded stage carrier");
  }

  if (!pointers.getResult().hasOneUse())
    return emitError(
        pointers, 0,
        "logical pointer copy local pointers must feed exactly one store");
  auto store =
      dyn_cast<triton::StoreOp>(*pointers.getResult().getUsers().begin());
  if (!store || store.getPtr() != pointers.getResult() || store.getMask() ||
      !store.getBoundaryCheck().empty())
    return emitError(
        pointers, 0,
        "logical pointer copy requires one unmasked full-carrier store");

  auto load = store.getValue().getDefiningOp<triton::LoadOp>();
  auto loadType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!load || !load->hasOneUse() || load.getIsVolatile() ||
      !load.getBoundaryCheck().empty() || load.getPadding() || !loadType ||
      !isGlobalPointerTensor(load.getPtr()) ||
      loadType.getShape() != ptrType.getShape() ||
      loadType.getElementType() != srcType.getElementType())
    return emitError(
        pointers, 0,
        "logical pointer copy requires a direct non-volatile global load "
        "whose carrier shape and element type match the stage");

  FailureOr<SmallVector<int64_t, 2>> selectedStorageTile =
      selectLogicalSMEMStorageTileShape(srcType, state->logicalShape);
  if (failed(selectedStorageTile))
    return emitError(
        pointers, 0,
        "logical stage has no exact power-of-two storage tile of at least 16 "
        "along its fragment axis");

  auto *root = findRootAction(plan, state->provenance.primaryRoot());
  if (!root)
    return emitError(pointers, 0, "candidate root has no storage action");
  if (root->storageTileShape.empty())
    root->storageTileShape = *selectedStorageTile;
  if (llvm::none_of(root->pointerCopies, [&](const auto &action) {
        return action.pointers == pointers;
      }))
    root->pointerCopies.push_back({pointers, store, load});
  return success();
}

LogicalResult
LogicalDomainContext::processExtractStage(Operation *operation,
                                          LogicalDomainPhase phase) {
  auto extract = cast<ExtractTileOp>(operation);
  if (const auto *state = lookupTensor(extract.getSrc()))
    if (state->axis == 0)
      return emitError(extract, 0,
                       "stage extraction cannot index a fragmented axis");
  return validateTensorResult(extract);
}

LogicalResult
LogicalDomainContext::processLocalStore(Operation *operation,
                                        LogicalDomainPhase phase) {
  auto store = cast<ttg::LocalStoreOp>(operation);
  const auto *state = lookupMemDesc(store.getDst());
  if (!state)
    return hasRestrictedOperand(store)
               ? processRejected(store, phase,
                                 "ordinary local store cannot consume a "
                                 "restricted logical tensor")
               : success();
  if (!state->isStage || state->viewTransposed ||
      !store.getDst().getDefiningOp<ttg::MemDescIndexOp>() ||
      state->logicalShape.size() != 2)
    return emitError(store, 1,
                     "logical local store requires a direct rank-2 stage");
  auto type = getTensorType(store.getSrc());
  if (!type || type.getEncoding() ||
      type.getShape() != ArrayRef<int64_t>(state->physicalShape))
    return emitError(store, 0,
                     "logical local store must cover the padded stage carrier");
  if (const auto *fragment = lookupTensor(store.getSrc()))
    if (ArrayRef<int64_t>(materializeLogicalShape(store.getSrc(), *fragment)) !=
        ArrayRef<int64_t>(state->logicalShape))
      return emitError(store, 0,
                       "initializer logical domain does not match the "
                       "declared allocation domain");
  if (phase == LogicalDomainPhase::Plan) {
    auto *root = findRootAction(plan, state->provenance.primaryRoot());
    if (!root)
      return emitError(store, 1, "candidate root has no storage action");
    if (!llvm::is_contained(root->localStores, store))
      root->localStores.push_back(store);
  }
  return success();
}

LogicalResult LogicalDomainContext::processPipe(Operation *operation,
                                                LogicalDomainPhase phase) {
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const MemDescLogicalState *state = lookupMemDesc(operand);
    if (!state)
      continue;
    auto names = operation->getAttrOfType<ArrayAttr>("field_names");
    if (!names || index >= names.size())
      return emitError(operation, index,
                       "candidate memdesc is not a pipe field operand");
    if (phase == LogicalDomainPhase::Plan && !state->isStage) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(operation, index,
                         "candidate root has no storage action");
      OpOperand *use = &operation->getOpOperand(index);
      if (llvm::none_of(root->memdescUses,
                        [&](const auto &action) { return action.use == use; }))
        root->memdescUses.push_back({use, true});
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processWarpSpecialize(Operation *operation,
                                            LogicalDomainPhase phase) {
  auto warpSpecialize = cast<ttg::WarpSpecializeOp>(operation);
  for (auto [index, capture] :
       llvm::enumerate(warpSpecialize.getExplicitCaptures())) {
    const MemDescLogicalState *state = lookupMemDesc(capture);
    if (!state)
      continue;
    for (Region *partition : warpSpecialize.getPartitionRegions()) {
      if (index >= partition->getNumArguments())
        return emitError(warpSpecialize, index,
                         "candidate capture has no partition argument");
      Value argument = partition->getArgument(index);
      if (argument.getType() != capture.getType())
        return emitError(warpSpecialize, index,
                         "candidate capture and partition argument types "
                         "must match");
      if (phase == LogicalDomainPhase::Propagate)
        mergeMemDesc(argument, *state);
    }
    if (phase == LogicalDomainPhase::Plan && !state->isStage) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(warpSpecialize, index,
                         "candidate root has no storage action");
      OpOperand *use = &operation->getOpOperand(index);
      if (llvm::none_of(root->memdescUses,
                        [&](const auto &action) { return action.use == use; }))
        root->memdescUses.push_back({use, false});
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processMemDescUse(Operation *operation,
                                        LogicalDomainPhase phase) {
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const MemDescLogicalState *state = lookupMemDesc(operand);
    if (!state)
      continue;
    if (phase == LogicalDomainPhase::Plan && !state->isStage) {
      auto *root = findRootAction(plan, state->provenance.primaryRoot());
      if (!root)
        return emitError(operation, index,
                         "candidate root has no storage action");
      OpOperand *use = &operation->getOpOperand(index);
      if (llvm::none_of(root->memdescUses,
                        [&](const auto &action) { return action.use == use; }))
        root->memdescUses.push_back({use, false});
    }
  }
  return success();
}

LogicalResult
LogicalDomainContext::processMemDescTranspose(Operation *operation,
                                              LogicalDomainPhase phase) {
  auto transpose = dyn_cast<ttg::MemDescTransOp>(operation);
  ArrayRef<int32_t> order =
      transpose ? transpose.getOrder()
                : cast<MemDescWGMMAViewOp>(operation).getOrder();
  const MemDescLogicalState *source = lookupMemDesc(operation->getOperand(0));
  if (!source)
    return success();
  if (source->viewTransposed)
    return emitError(operation, 0,
                     "nested descriptor transpose is unsupported");
  if (order != ArrayRef<int32_t>({1, 0}))
    return emitError(operation, 0,
                     "descriptor transpose must use order [1, 0]");
  if (phase == LogicalDomainPhase::Propagate) {
    if (source->logicalShape.size() != 2)
      return emitError(operation, 0, "WGMMA descriptor view requires rank two");
    MemDescLogicalState result = *source;
    std::swap(result.physicalShape[0], result.physicalShape[1]);
    std::swap(result.logicalShape[0], result.logicalShape[1]);
    std::swap(result.axisMap[0], result.axisMap[1]);
    result.viewTransposed = true;
    mergeMemDesc(operation->getResult(0), result);
  } else if (transpose) {
    auto *root = findRootAction(plan, source->provenance.primaryRoot());
    if (!root)
      return emitError(operation, 0, "candidate root has no storage action");
    if (!llvm::is_contained(root->transposes, transpose))
      root->transposes.push_back(transpose);
  }
  return success();
}

static LogicalResult
validatePlannedWGMMA(WGMMAOp dot, std::optional<int64_t> activeN,
                     std::optional<int64_t> activeK,
                     const MemDescLogicalState *candidateB) {
  auto aTensorType = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto aMemDescType = dyn_cast<ttg::MemDescType>(dot.getA().getType());
  auto bType = dyn_cast<ttg::MemDescType>(dot.getB().getType());
  auto cType = getTensorType(dot.getC());
  if ((!aTensorType && !aMemDescType) || !bType || !cType)
    return dot.emitOpError(
        "planned logical extent requires ranked WGMMA carriers");

  ArrayRef<int64_t> aShape =
      aTensorType ? aTensorType.getShape() : aMemDescType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  Type aElementType = aTensorType ? aTensorType.getElementType()
                                  : aMemDescType.getElementType();
  Type bElementType = bType.getElementType();
  std::optional<int64_t> instructionK = getWGMMAInstructionK(aElementType);
  if (!instructionK || !isSupportedWGMMATypeCombination(
                           aElementType, bElementType, cType.getElementType(),
                           dot.getInputPrecision()))
    return dot.emitOpError(
        "planned logical extent has an unsupported Hopper WGMMA operand, "
        "accumulator, or input precision combination");
  if (candidateB && (candidateB->logicalShape.size() != 2 ||
                     candidateB->physicalShape.size() != 2))
    return dot.emitOpError("planned logical SMEM operand must have rank two");

  auto isDirectCandidateB = [&] {
    return dot.getB().getDefiningOp<ttg::MemDescIndexOp>() != nullptr;
  };
  auto isTransposedCandidateB = [&] {
    Operation *view = dot.getB().getDefiningOp();
    return view && isa<ttg::MemDescTransOp, MemDescWGMMAViewOp>(view) &&
           view->getOperand(0).getDefiningOp<ttg::MemDescIndexOp>() != nullptr;
  };
  auto hasCandidateBTopology = [&] {
    return candidateB && (candidateB->viewTransposed ? isTransposedCandidateB()
                                                     : isDirectCandidateB());
  };

  if (!activeN && !activeK && candidateB) {
    if (!candidateB->viewTransposed &&
        !supportsWGMMAOperandTranspose(bElementType))
      return dot.emitOpError(
          "TF32, FP8, and int8 tiled B require a transposed logical view so "
          "WGMMA consumes a column-major descriptor without PTX transpose "
          "operands");
    if (!candidateB->viewTransposed && aTensorType && isDirectCandidateB())
      return success();
    if (candidateB->viewTransposed && aMemDescType &&
        candidateB->logicalShape[1] == candidateB->physicalShape[1] &&
        isTransposedCandidateB())
      return success();
    return dot.emitOpError(
        "planned tiled SMEM operand requires active_n, active_k, or a "
        "full-shape compatible WGMMA form");
  }
  if (!activeN && !activeK)
    return success();
  if (activeN && activeK)
    return dot.emitOpError("planned active_n and active_k cannot be combined");
  if (aShape.size() != 2 || bShape.size() != 2 || cType.getRank() != 2)
    return dot.emitOpError(
        "planned active extent requires rank-two WGMMA carriers");
  // The type combination was checked above; active extents additionally
  // require a 32-bit accumulator.
  if (cType.getElementTypeBitWidth() != 32)
    return dot.emitOpError(
        "planned active extent requires a supported Hopper WGMMA type "
        "combination with a 32-bit accumulator");
  if (candidateB && !candidateB->viewTransposed &&
      !supportsWGMMAOperandTranspose(bElementType))
    return dot.emitOpError(
        "TF32, FP8, and int8 tiled B require a transposed logical view so "
        "WGMMA consumes a column-major descriptor without PTX transpose "
        "operands");
  if (aShape[0] != 64)
    return dot.emitOpError(
        "planned active extent currently requires physical M=64");

  if (activeN) {
    if (*activeN <= 0 || *activeN % 8 != 0 || *activeN > bShape[1])
      return dot.emitOpError(
          "planned active_n must be a positive multiple of 8 within N");
    int64_t maxInstructionN = aElementType.isInteger(8) ? 224 : 256;
    if (bShape[1] > maxInstructionN)
      return dot.emitOpError(
          "planned active_n physical N carrier exceeds the WGMMA type limit");
    if (dot->hasAttr("tle.wgmma_accumulator_chain_c"))
      return dot.emitOpError(
          "planned active_n does not support an accumulator chain");
    if (candidateB) {
      if (!hasCandidateBTopology() || candidateB->logicalShape.size() != 2 ||
          *activeN != candidateB->logicalShape[1] ||
          bShape[1] != nextPowerOfTwo(candidateB->logicalShape[1]) ||
          candidateB->logicalShape[0] != candidateB->physicalShape[0] ||
          bShape[0] != candidateB->physicalShape[0] ||
          aShape[1] != candidateB->physicalShape[0])
        return dot.emitOpError(
            "planned active_n carrier does not match the logical SMEM stage");
    }
    return success();
  }

  if (*activeK <= 0 || *activeK % *instructionK != 0 || *activeK > aShape[1])
    return dot.emitOpError(
        "planned active_k must be a positive multiple of the operand type's "
        "WGMMA instruction K within the physical carrier");
  if (aShape[1] % *instructionK != 0) {
    return dot.emitOpError(
        "planned active_k requires physical K divisible by the WGMMA "
        "instruction K");
  }
  int64_t physicalSplits = aShape[1] / *instructionK;
  if ((physicalSplits & (physicalSplits - 1)) != 0)
    return dot.emitOpError(
        "planned active_k physical carrier must contain a power-of-two "
        "number of WGMMA K instructions");
  if (candidateB) {
    if (!hasCandidateBTopology() || candidateB->logicalShape.size() != 2 ||
        *activeK != candidateB->logicalShape[0] ||
        aShape[1] != nextPowerOfTwo(candidateB->logicalShape[0]) ||
        bShape[0] != aShape[1] ||
        candidateB->logicalShape[1] != candidateB->physicalShape[1] ||
        bShape[1] != candidateB->physicalShape[1])
      return dot.emitOpError(
          "planned active_k carrier does not match the logical SMEM stage");
  }
  return success();
}

LogicalResult LogicalDomainContext::processWGMMA(Operation *operation,
                                                 LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto dot = cast<WGMMAOp>(operation);
  const MemDescLogicalState *aMemDescState = lookupMemDesc(dot.getA());
  const MemDescLogicalState *bState = lookupMemDesc(dot.getB());
  const TensorFragmentState *aState = lookupTensor(dot.getA());
  if (aMemDescState)
    return emitError(dot, 0,
                     "candidate SMEM is not supported as WGMMA operand A");
  LogicalWGMMAAction action{dot, std::nullopt};
  std::optional<int64_t> activeK;
  unsigned activeKOperand = 0;
  int32_t activeKAxis = 1;
  if (bState) {
    auto *root = findRootAction(plan, bState->provenance.primaryRoot());
    if (!root)
      return emitError(dot, 1, "candidate root has no storage action");
    root->reachesWGMMA = true;
    bool restrictedK = bState->logicalShape[0] != bState->physicalShape[0];
    bool restrictedN = bState->logicalShape[1] != bState->physicalShape[1];
    if (restrictedK && restrictedN)
      return emitError(dot, 1, "candidate WGMMA B has multiple fragment axes");
    if (restrictedN) {
      int64_t logicalN = bState->logicalShape[1];
      action.activeN = logicalN;
    } else if (restrictedK) {
      int64_t logicalK = bState->logicalShape[0];
      activeK = logicalK;
      activeKOperand = 1;
      activeKAxis = 0;
    }
  }
  if (aState) {
    auto aType = getTensorType(dot.getA());
    if (!aType || aType.getRank() != 2)
      return emitError(dot, 0, "restricted WGMMA A must have rank two");
    if (aState->axis < 0 || aState->axis >= 2)
      return emitError(dot, 0, "restricted WGMMA A axis is out of range");
    if (aState->axis == 1) {
      int64_t physicalK = aType.getShape()[1];
      int64_t logicalK = aState->logicalExtent;
      auto bType = dyn_cast<ttg::MemDescType>(dot.getB().getType());
      if (!bType || bType.getRank() != 2 || bType.getShape()[0] != physicalK)
        return emitError(dot, 0,
                         "A and B physical contraction extents disagree");
      std::optional<int64_t> instructionK =
          getWGMMAInstructionK(aType.getElementType());
      if (!instructionK || logicalK <= 0 || logicalK % *instructionK != 0)
        return emitError(dot, 0,
                         "inferred active_k must be a positive multiple of "
                         "the operand type's WGMMA instruction K");
      if (action.activeN)
        return emitError(dot, 0, "active_n and active_k cannot be combined");
      if (bState && bState->logicalShape[0] != bState->physicalShape[0] &&
          bState->logicalShape[0] != logicalK)
        return emitError(dot, 1,
                         "K/V logical extents disagree at the PV WGMMA join");
      activeK = logicalK;
      activeKOperand = 0;
      activeKAxis = 1;
    }
  }
  if (failed(validatePlannedWGMMA(dot, action.activeN, activeK, bState)))
    return failure();
  if (action.activeN)
    plan.wgmmas.push_back(action);
  if (activeK)
    plan.folds.push_back({dot, activeKOperand, activeKAxis, *activeK,
                          LogicalFragmentFoldMechanism::WGMMAActiveK,
                          LogicalReductionIdentity::Zero});
  if (lookupTensor(dot.getD()))
    return validateTensorResult(operation);
  return success();
}

LogicalResult LogicalDomainContext::processWGMMAWait(Operation *operation,
                                                     LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processSameShape(Operation *operation,
                                                     LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  for (auto [operandIndex, operand] :
       llvm::enumerate(operation->getOperands())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    auto operandType = getTensorType(operand);
    bool transferred = llvm::any_of(operation->getResults(), [&](Value result) {
      auto resultType = getTensorType(result);
      return operandType && resultType &&
             resultType.getShape() == operandType.getShape() &&
             lookupTensor(result);
    });
    if (!transferred)
      return emitError(operation, operandIndex,
                       "same-shape model could not transfer the restricted "
                       "operand to a tensor result");
  }
  return success();
}

LogicalResult
LogicalDomainContext::processExpandDims(Operation *operation,
                                        LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processBroadcast(Operation *operation,
                                                     LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processTranspose(Operation *operation,
                                                     LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processReshape(Operation *operation,
                                                   LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processCat(Operation *operation,
                                               LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processJoin(Operation *operation,
                                                LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processSplit(Operation *operation,
                                                 LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto split = cast<triton::SplitOp>(operation);
  const TensorFragmentState *state = lookupTensor(split.getSrc());
  if (!state)
    return success();
  auto sourceType = getTensorType(split.getSrc());
  if (!sourceType || (state->axis == sourceType.getRank() - 1 &&
                      state->logicalExtent != sourceType.getShape().back()))
    return emitError(split, 0,
                     "split requires the joined minor dimension to be full");
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processReduce(Operation *operation,
                                                  LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  auto reduce = cast<triton::ReduceOp>(operation);
  for (auto [index, operand] : llvm::enumerate(reduce.getSrcs())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    int32_t axis = reduce.getAxis();
    auto type = getTensorType(operand);
    if (!type || axis < 0 || axis >= type.getRank())
      return emitError(reduce, index, "reduction axis is out of range");
    if (axis != state->axis)
      return emitError(
          reduce, index,
          "attention fragment reduction must consume the fragment axis");
    if (reduce.getNumOperands() != 1 || reduce.getNumResults() != 1)
      return emitError(reduce, index,
                       "restricted reduction supports one input and result");
    FailureOr<LogicalReductionIdentity> identity = getReductionIdentity(reduce);
    if (failed(identity))
      return emitError(reduce, index,
                       "reduction combiner has no supported tail identity");
    if (type.getEncoding())
      return emitError(reduce, index,
                       "strict early-TTIR analysis rejects encoded tensors");
    plan.folds.push_back(
        {reduce, static_cast<unsigned>(index), axis, state->logicalExtent,
         LogicalFragmentFoldMechanism::IdentityMask, *identity});
  }
  return success();
}

LogicalResult LogicalDomainContext::processDot(Operation *operation,
                                               LogicalDomainPhase phase) {
  assert(phase == LogicalDomainPhase::Plan);
  if (operation->getNumOperands() < 3 || operation->getNumResults() != 1)
    return processRejected(operation, phase, "malformed dot operation");
  const TensorFragmentState *aState = lookupTensor(operation->getOperand(0));
  const TensorFragmentState *bState = lookupTensor(operation->getOperand(1));
  const TensorFragmentState *cState = lookupTensor(operation->getOperand(2));
  if (!aState && !bState && !cState)
    return success();
  auto aType = getTensorType(operation->getOperand(0));
  auto bType = getTensorType(operation->getOperand(1));
  auto resultType = getTensorType(operation->getResult(0));
  if (!aType || !bType || !resultType || aType.getRank() != 2 ||
      bType.getRank() != 2 || resultType.getRank() != 2)
    return processRejected(operation, phase,
                           "logical dot currently requires rank-two tensors");
  if ((aState && aState->axis == 1) || (bState && bState->axis == 0))
    return emitError(operation, aState ? 0 : 1,
                     "ordinary tt.dot requires a full contraction K");
  if (isa<triton::DotScaledOp>(operation))
    for (unsigned i = 3; i < operation->getNumOperands(); ++i)
      if (lookupTensor(operation->getOperand(i)))
        return emitError(operation, i,
                         "dot_scaled scale operands must be full-shape");
  return validateTensorResult(operation);
}

LogicalResult LogicalDomainContext::processStore(Operation *operation,
                                                 LogicalDomainPhase phase) {
  auto store = cast<triton::StoreOp>(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    if (index != 1 || !isGlobalPointerTensor(store.getPtr()))
      return emitError(store, index,
                       "only a value stored through global pointers is a "
                       "logical-domain terminal");
    if (phase == LogicalDomainPhase::Plan) {
      auto valueType = getTensorType(store.getValue());
      if (!valueType || valueType.getEncoding())
        return emitError(
            store, index,
            "strict early-TTIR store requires an unencoded tensor");
      plan.guards.push_back(
          {store, 1,
           store.getMask() ? std::optional<unsigned>(2) : std::nullopt,
           state->axis, state->logicalExtent});
    }
  }
  return success();
}

LogicalResult LogicalDomainContext::processAtomicRMW(Operation *operation,
                                                     LogicalDomainPhase phase) {
  auto atomic = cast<triton::AtomicRMWOp>(operation);
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    const TensorFragmentState *state = lookupTensor(operand);
    if (!state)
      continue;
    if (index != 1 || !isGlobalPointerTensor(atomic.getPtr()))
      return emitError(atomic, index,
                       "only an atomic value through global pointers is a "
                       "logical-domain terminal");
    if (!atomic.getResult().use_empty())
      return emitError(
          atomic, index,
          "attention fragment atomic_rmw requires an unused result");
    if (phase == LogicalDomainPhase::Plan) {
      auto valueType = getTensorType(atomic.getVal());
      if (!valueType || valueType.getEncoding())
        return emitError(
            atomic, index,
            "strict early-TTIR atomic requires an unencoded tensor");
      plan.guards.push_back(
          {atomic, 1,
           atomic.getMask() ? std::optional<unsigned>(2) : std::nullopt,
           state->axis, state->logicalExtent});
    }
  }
  return success();
}

LogicalResult LogicalDomainContext::processRejected(Operation *operation,
                                                    LogicalDomainPhase phase,
                                                    StringRef reason) {
  if (phase == LogicalDomainPhase::Propagate)
    return success();
  for (auto [index, operand] : llvm::enumerate(operation->getOperands()))
    if (lookupTensor(operand) || lookupMemDesc(operand) ||
        lookupDescriptor(operand))
      return emitError(operation, index, reason);
  return success();
}

namespace {

static std::optional<LogicalBehavior>
getExplicitLogicalBehavior(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<LogicalBehavior>>(op)
      .Case<triton::MakeTensorDescOp>(
          [](auto) { return LogicalBehavior::CandidateDescriptor; })
      .Case<ttg::LocalAllocOp>(
          [](auto) { return LogicalBehavior::CandidateAlloc; })
      .Case<ttg::LocalStoreOp>([](auto) { return LogicalBehavior::LocalStore; })
      .Case<ttg::MemDescIndexOp>(
          [](auto) { return LogicalBehavior::MemDescIndex; })
      .Case<ttg::TMACopyOp>(
          [](auto) { return LogicalBehavior::LogicalTMACopy; })
      .Case<LocalPointersOp>([](LocalPointersOp op) {
        return op->hasAttr(kLogicalCopyShapeAttr)
                   ? LogicalBehavior::LogicalPointerCopy
                   : LogicalBehavior::RejectEscape;
      })
      .Case<ttg::WarpSpecializeOp>(
          [](auto) { return LogicalBehavior::WarpSpecialize; })
      .Case<ttg::MemDescTransOp, MemDescWGMMAViewOp>(
          [](auto) { return LogicalBehavior::MemDescTranspose; })
      .Case<WGMMAOp>([](auto) { return LogicalBehavior::WGMMA; })
      .Case<WGMMAWaitOp>([](auto) { return LogicalBehavior::WGMMAWait; })
      .Case<ExclusiveCumsumOp>([](auto) { return LogicalBehavior::RejectScan; })
      .Case<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
            PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp>(
          [](auto) { return LogicalBehavior::Pipe; })
      .Case<WGMMASharedOperandFenceOp>(
          [](auto) { return LogicalBehavior::MemDescUse; })
      .Case<triton::ExpandDimsOp>(
          [](auto) { return LogicalBehavior::ExpandDims; })
      .Case<triton::BroadcastOp>(
          [](auto) { return LogicalBehavior::Broadcast; })
      .Case<triton::TransOp>([](auto) { return LogicalBehavior::Transpose; })
      .Case<triton::ReshapeOp>([](auto) { return LogicalBehavior::Reshape; })
      .Case<triton::CatOp>([](auto) { return LogicalBehavior::Cat; })
      .Case<triton::JoinOp>([](auto) { return LogicalBehavior::Join; })
      .Case<triton::SplitOp>([](auto) { return LogicalBehavior::Split; })
      .Case<triton::ReduceOp>([](auto) { return LogicalBehavior::Reduce; })
      .Case<triton::ScanOp>([](auto) { return LogicalBehavior::RejectScan; })
      .Case<triton::DotOp, triton::DotScaledOp>(
          [](auto) { return LogicalBehavior::Dot; })
      .Case<triton::StoreOp>([](auto) { return LogicalBehavior::Store; })
      .Case<triton::AtomicRMWOp>(
          [](auto) { return LogicalBehavior::AtomicRMW; })
      .Case<triton::GatherOp>(
          [](auto) { return LogicalBehavior::RejectGather; })
      .Case<triton::HistogramOp>(
          [](auto) { return LogicalBehavior::RejectHistogram; })
      .Case<triton::AtomicCASOp>(
          [](auto) { return LogicalBehavior::RejectAtomicCAS; })
      .Case<triton::LoadOp>([](auto) { return LogicalBehavior::RejectLoad; })
      .Case<ExtractTileOp>([](ExtractTileOp op) {
        auto src = cast<RankedTensorType>(op.getSrc().getType());
        auto dst = cast<RankedTensorType>(op.getType());
        auto index = getStaticExtractTileIndex(op);
        return src.getRank() >= 2 && dst.getRank() == src.getRank() &&
                       dst.getShape()[0] == 1 &&
                       src.getShape().drop_front() ==
                           dst.getShape().drop_front() &&
                       index && *index >= 0 && *index < src.getShape()[0]
                   ? LogicalBehavior::ExtractStage
                   : LogicalBehavior::RejectEscape;
      })
      .Case<InsertTileOp, triton::CallOp, triton::ReturnOp>(
          [](auto) { return LogicalBehavior::RejectEscape; })
      .Default([](Operation *) -> std::optional<LogicalBehavior> {
        return std::nullopt;
      });
}

static bool hasRestrictedMemDescOperand(const LogicalDomainContext &context,
                                        Operation *op) {
  return llvm::any_of(op->getOperands(), [&](Value value) {
    return context.lookupMemDesc(value);
  });
}

static bool isTensorControlFlow(Operation *op) {
  return isa<RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
             BranchOpInterface>(op);
}

static bool containsOperand(OperandRange range, unsigned operandNumber) {
  if (range.empty())
    return false;
  unsigned begin = range.getBeginOperandIndex();
  return operandNumber >= begin && operandNumber < begin + range.size();
}

static bool isForwardedControlFlowOperand(Operation *op,
                                          unsigned operandNumber) {
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (RegionSuccessor successor : successors)
      if (containsOperand(
              branch.getEntrySuccessorOperands(RegionBranchPoint(successor)),
              operandNumber))
        return true;
  }
  if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
    SmallVector<Attribute> operandAttrs(op->getNumOperands());
    SmallVector<RegionSuccessor> successors;
    terminator.getSuccessorRegions(operandAttrs, successors);
    for (RegionSuccessor successor : successors)
      if (containsOperand(
              terminator.getSuccessorOperands(RegionBranchPoint(successor)),
              operandNumber))
        return true;
  }
  if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    for (unsigned index = 0; index < op->getNumSuccessors(); ++index)
      if (containsOperand(
              branch.getSuccessorOperands(index).getForwardedOperands(),
              operandNumber))
        return true;
  }
  return false;
}

static LogicalResult collectLogicalDomainFacts(ModuleOp module,
                                               DataFlowSolver &solver,
                                               LogicalDomainPlan &plan) {
  bool failedCollection = false;
  auto collect = [&](Value value) {
    if (failedCollection)
      return;
    const LogicalDomainLattice *lattice =
        solver.lookupState<LogicalDomainLattice>(value);
    if (!lattice)
      return;
    const LogicalDomainFact &fact = lattice->getValue();
    if (fact.getKind() == LogicalDomainFact::Kind::Conflict) {
      Operation *owner = value.getDefiningOp();
      if (!owner)
        owner = cast<BlockArgument>(value).getOwner()->getParentOp();
      owner->emitOpError(
          "logical-domain dataflow has an unsupported transfer or "
          "incompatible domains at a control-flow join");
      failedCollection = true;
      return;
    }
    if (fact.isFragment())
      plan.tensors.try_emplace(value, fact.getState());
    else if (fact.isMemDesc())
      plan.memdescs.try_emplace(value, fact.getMemDesc());
    else if (fact.isDescriptor())
      plan.descriptors.try_emplace(value, fact.getDescriptor());
  };

  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      collect(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collect(argument);
  });
  return failure(failedCollection);
}

static LogicalResult planOperation(LogicalDomainContext &context,
                                   Operation *op) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (!context.lookupDescriptor(operand))
      continue;
    if (isa<ttg::TMACopyOp>(op) && index == 0 &&
        context.lookupMemDesc(op->getOperand(1)))
      continue;
    if (isa<ttg::WarpSpecializeOp>(op) ||
        isForwardedControlFlowOperand(op, index))
      continue;
    return context.emitError(
        op, index,
        "logical tensor descriptors currently require a TLE "
        "global-to-SMEM copy into a logical stage");
  }
  if (!context.hasRestrictedOperand(op))
    return success();

  if (auto model = getLogicalDomainModel(op))
    return model->collectRequirements(context, op);

  // Shared-memory facts are intentionally not propagated through generic
  // operations or control flow: changing their physical representation
  // requires an explicit storage rewrite action.
  if (hasRestrictedMemDescOperand(context, op))
    return context.processRejected(
        op, LogicalDomainPhase::Plan,
        "operation has no logical-domain transfer semantics");

  // SparseForwardDataFlowAnalysis owns RegionBranch/CFG joins for tensor and
  // tensor-descriptor facts, including their alternative source definitions.
  if (isTensorControlFlow(op)) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!context.lookupTensor(operand) && !context.lookupDescriptor(operand))
        continue;
      if (!isForwardedControlFlowOperand(op, index))
        return context.emitError(
            op, index,
            "restricted tensor is not forwarded across this control-flow "
            "edge");
    }
    return success();
  }

  return context.processRejected(
      op, LogicalDomainPhase::Plan,
      "operation has no logical-domain transfer semantics");
}

} // namespace

// Select after the complete producer/consumer graph is known. In particular,
// a K/V allocation can feed both a transposed QK and a direct PV descriptor.
// Neither the first copy nor the first WGMMA may choose its layout alone.
static LogicalResult
selectRootSMEMPlan(ArrayRef<LogicalRootRewriteAction *> roots,
                   LogicalDomainPlan &plan, ModuleAxisInfoAnalysis &axisInfo) {
  auto &root = *roots.front();
  struct Consumer {
    bool viewTransposed;
    unsigned instructionK, maxN;
    int64_t logicalK, logicalN;
    bool integer;
  };
  SmallVector<Consumer> consumers;
  plan.module.walk([&](WGMMAOp dot) {
    auto it = plan.memdescs.find(dot.getB());
    if (it == plan.memdescs.end() || llvm::none_of(roots, [&](auto *member) {
          return llvm::is_contained(it->second.provenance.roots,
                                    member->alloc.getOperation());
        }))
      return;
    auto type = cast<ttg::MemDescType>(dot.getB().getType());
    const auto &state = it->second;
    auto instructionK = getWGMMAInstructionK(type.getElementType());
    assert(instructionK &&
           "WGMMA types were validated before storage planning");
    consumers.push_back(
        {state.viewTransposed, static_cast<unsigned>(*instructionK),
         static_cast<unsigned>(std::min<int64_t>(state.logicalShape[1], 256)),
         state.logicalShape[0], state.logicalShape[1],
         type.getElementType().isInteger(8)});
  });
  if (consumers.empty())
    return root.alloc.emitOpError(
        "logical root has no supported WGMMA B consumer");

  auto stageType = root.stages.front().getType();
  auto carrier = cast<ttg::NVMMASharedEncodingAttr>(stageType.getEncoding());
  auto elementType = stageType.getElementType();
  unsigned bits = elementType.getIntOrFloatBitWidth();
  ArrayRef<int64_t> shape = ArrayRef<int64_t>(root.logicalShape).drop_front();
  unsigned fragmentAxis = !llvm::isPowerOf2_64(shape[0]) ? 0 : 1;
  auto preferredTile = root.storageTileShape;
  bool hasTMA =
      llvm::any_of(roots, [](auto *member) { return !member->copies.empty(); });
  if ((root.initializer || !root.localStores.empty()) &&
      preferredTile.empty()) {
    auto preferred = selectLogicalSMEMStorageTileShape(stageType, shape);
    if (failed(preferred))
      return root.alloc.emitOpError("initializer has no legal storage tile");
    preferredTile = *preferred;
  }

  auto copyVectorBytes = [&](LogicalPointerCopyAction &copy, unsigned axis) {
    unsigned vec = 1;
    if (auto *info = axisInfo.getAxisInfo(copy.load.getPtr())) {
      if (info->getRank() == 2)
        vec = std::min<int64_t>(
            info->getContiguity(axis),
            std::max<int64_t>(1, info->getDivisibility(axis) / (bits / 8)));
    }
    if (auto mask = copy.load.getMask())
      if (auto *info = axisInfo.getAxisInfo(mask))
        if (info->getRank() == 2)
          vec = std::min<int64_t>(vec, info->getConstancy(axis));
    return std::max<unsigned>(1, std::min<unsigned>(vec, 128 / bits)) * bits /
           8;
  };

  // Lexicographic, deterministic cost: complete tensor instructions, copy
  // transactions, producer waves, bank-conflict avoidance, address work.
  // This is a bounded compile-time model, not a claim about measured latency.
  using Cost = std::array<int64_t, 6>;
  std::optional<Cost> bestCost;
  std::optional<SMEMLayoutPlan> best;
  for (int64_t rows = 8; rows <= shape[0]; rows *= 2) {
    for (int64_t cols = 8; cols <= shape[1]; cols *= 2) {
      SmallVector<int64_t, 2> tile{rows, cols};
      if (shape[0] % rows || shape[1] % cols ||
          tile[fragmentAxis] < kExactSMEMFragmentQuantum)
        continue;
      for (bool storageTransposed : {false, true}) {
        for (unsigned swizzle : {128u, 64u, 32u, 0u}) {
          auto encoding = ttg::NVMMASharedEncodingAttr::get(
              root.alloc.getContext(), swizzle, storageTransposed, bits, false,
              carrier.getCTALayout());
          if (!isLegalSMEMTile(shape, tile, encoding))
            continue;
          // Native TMA loads use a non-transposed shared layout. Its actual
          // hardware box participates in the same plan as the WGMMA layout.
          if (hasTMA && storageTransposed)
            continue;
          SMEMLayoutPlan storage(shape, tile, encoding);
          int64_t instructions = 0;
          bool legal = true;
          for (const Consumer &consumer : consumers) {
            // TF32, FP8, and int8 have no operand transpose immediates.
            if (!supportsWGMMAOperandTranspose(elementType) &&
                storageTransposed == consumer.viewTransposed) {
              legal = false;
              break;
            }
            auto operand = planWGMMAOperand(storage, consumer.viewTransposed,
                                            consumer.instructionK,
                                            consumer.maxN, consumer.integer);
            if (!operand) {
              legal = false;
              break;
            }
            instructions += consumer.logicalK / consumer.instructionK *
                            (consumer.logicalN / operand->instructionN);
          }
          if (!legal)
            continue;
          SmallVector<int64_t> box;
          if (hasTMA) {
#ifdef __HCU__
            continue;
#else
            box = triton::nvidia_gpu::getTMABlockShape(encoding, tile, false);
            if (rows % box[0] || cols % box[1])
              continue;
            // TMA writes each box in row-major order before swizzling. Check
            // its basis against the shared layout selected for WGMMA.
            for (int64_t row = 1; row < box[0]; row *= 2)
              legal &= storage.offsetBeforeSwizzle(row, 0) ==
                       row * box[1] * bits / 8;
            for (int64_t col = 1; col < box[1]; col *= 2)
              legal &= storage.offsetBeforeSwizzle(0, col) == col * bits / 8;
            if (!legal)
              continue;
#endif
          }
          int64_t transactions = 0, waves = 0;
          int64_t tiles = shape[0] * shape[1] / (rows * cols);
          for (auto *member : roots) {
            for (auto &copy : member->pointerCopies) {
              unsigned vectorBytes =
                  copyVectorBytes(copy, storageTransposed ? 0 : 1);
              auto micro = selectLogicalPointerCopyMicroTile(
                  copy.store, tile, encoding, elementType, vectorBytes);
              transactions += storage.stageBytes() / vectorBytes;
              waves += shape[0] * shape[1] / (micro[0] * micro[1]);
            }
            int64_t initializedStages = member->localStores.size();
            if (member->initializer)
              initializedStages += member->logicalShape[0];
            if (initializedStages) {
              auto micro = selectLogicalPointerCopyMicroTile(
                  root.alloc, tile, encoding, elementType, 16);
              transactions += initializedStages * storage.stageBytes() / 16;
              waves += initializedStages * shape[0] * shape[1] /
                       (micro[0] * micro[1]);
            }
            for (auto copy : member->copies) {
              int64_t requests = rows * cols / (box[0] * box[1]);
              int64_t warps = ttg::maybeLookupNumWarps(copy).value_or(4);
              transactions += tiles * requests;
              waves += tiles * llvm::divideCeil(requests, warps);
            }
          }
          Cost cost{instructions,
                    transactions,
                    waves,
                    -int64_t(swizzle),
                    tiles,
                    tile == preferredTile &&
                            storageTransposed == carrier.getTransposed()
                        ? 0
                        : 1};
          if (!bestCost || cost < *bestCost) {
            bestCost = cost;
            best = std::move(storage);
          }
        }
      }
    }
  }
  if (!best)
    return root.alloc.emitOpError(
        "no shared layout satisfies all exact-copy and WGMMA consumers");
  for (auto *member : roots) {
    member->storageTileShape = {best->tileRows, best->tileCols};
    member->storageEncoding = best->encoding;
    for (auto &copy : member->pointerCopies) {
      copy.vectorBytes =
          copyVectorBytes(copy, best->encoding.getTransposed() ? 0 : 1);
      copy.microTileShape = selectLogicalPointerCopyMicroTile(
          copy.store, member->storageTileShape, member->storageEncoding,
          elementType, copy.vectorBytes);
    }
    for (auto copy : member->copies) {
      const auto &descriptor = plan.descriptors.find(copy.getSrc())->second;
      LogicalShape block = descriptor.logicalShape;
      block[block.size() - 2] = best->tileRows;
      block.back() = best->tileCols;
#ifndef __HCU__
      auto box =
          triton::nvidia_gpu::getTMABlockShape(best->encoding, block, false);
      for (Operation *source : descriptor.provenance.roots)
        plan.descriptorRewrites.try_emplace(
            source, LogicalDescriptorRewriteAction{block, box});
#endif
    }
  }
  return success();
}

FailureOr<LogicalDomainPlan> analyzeLogicalDomains(ModuleOp module) {
  LogicalDomainPlan plan;
  plan.module = module;
  SmallVector<ttg::LocalAllocOp> candidates;
  SmallVector<triton::MakeTensorDescOp> descriptors;
  module.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (alloc->hasAttr(kStoragePlan) || alloc->hasAttr(kLogicalAllocShape))
        candidates.push_back(alloc);
    } else if (auto desc = dyn_cast<triton::MakeTensorDescOp>(op)) {
      if (desc->hasAttr(kLogicalDescriptorShape))
        descriptors.push_back(desc);
    }
  });
  if (candidates.empty()) {
    if (!descriptors.empty()) {
      descriptors.front().emitOpError(
          "logical descriptor has no planned TLE TMA consumer");
      return failure();
    }
    return plan;
  }
  LogicalDomainContext context(plan);

  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  solver->load<LogicalDomainAnalysis>();
  WalkResult analyzed = module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
        failed(solver->initializeAndRun(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (analyzed.wasInterrupted() ||
      failed(collectLogicalDomainFacts(module, *solver, plan)))
    return failure();

  // Materialize every storage root before planning consumers, independent of
  // their nesting or traversal order.
  for (ttg::LocalAllocOp alloc : candidates)
    if (failed(context.processCandidateAlloc(alloc, LogicalDomainPhase::Plan)))
      return failure();

  WalkResult planned = module.walk([&](Operation *op) {
    if (failed(planOperation(context, op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (planned.wasInterrupted())
    return failure();

  for (const LogicalRootRewriteAction &root : plan.roots) {
    if (root.stages.empty()) {
      root.alloc->emitOpError("logical domain has no stage views");
      return failure();
    }
    if ((root.copies.empty() && root.pointerCopies.empty() &&
         root.localStores.empty() && !root.initializer) ||
        !root.reachesWGMMA) {
      root.alloc->emitOpError(
          "logical domain must reach both an exact producer and WGMMA");
      return failure();
    }
  }
  ModuleAxisInfoAnalysis axisInfo(module);
  // A descriptor forwarded through CFG joins or captured by multiple producers
  // must retain one physical block and encoding. Plan all its roots together.
  SmallVector<unsigned> leaders(plan.roots.size());
  std::iota(leaders.begin(), leaders.end(), 0);
  auto leader = [&](unsigned index) {
    while (leaders[index] != index)
      index = leaders[index];
    return index;
  };
  DenseMap<Operation *, unsigned> descriptorOwners;
  for (auto [index, root] : llvm::enumerate(plan.roots))
    for (auto copy : root.copies)
      for (Operation *source :
           plan.descriptors.find(copy.getSrc())->second.provenance.roots) {
        auto [it, inserted] = descriptorOwners.try_emplace(source, index);
        if (!inserted)
          leaders[leader(index)] = leader(it->second);
      }
  SmallVector<SmallVector<LogicalRootRewriteAction *>> groups(
      plan.roots.size());
  for (auto [index, root] : llvm::enumerate(plan.roots))
    groups[leader(index)].push_back(&root);
  for (const auto &group : groups)
    if (!group.empty() && failed(selectRootSMEMPlan(group, plan, axisInfo)))
      return failure();
  for (const auto &[value, descriptor] : plan.descriptors)
    for (Operation *source : descriptor.provenance.roots)
      if (!plan.descriptorRewrites.count(source)) {
        source->emitOpError(
            "logical descriptor has no planned TLE TMA consumer");
        return failure();
      }
  return plan;
}

namespace {
struct PredicateCacheEntry {
  Block *block;
  LogicalShape physicalShape;
  int32_t axis;
  int64_t logicalExtent;
  Value predicate;
};

static Value
createLogicalPredicate(OpBuilder &builder, Operation *anchor,
                       RankedTensorType tensorType, int32_t axis,
                       int64_t logicalExtent,
                       SmallVectorImpl<PredicateCacheEntry> &cache) {
  assert(!tensorType.getEncoding() &&
         "strict TTIR predicate must be unencoded");
  for (const PredicateCacheEntry &entry : cache) {
    Operation *predicateOp = entry.predicate.getDefiningOp();
    if (entry.block == builder.getInsertionBlock() &&
        ArrayRef<int64_t>(entry.physicalShape) == tensorType.getShape() &&
        entry.axis == axis && entry.logicalExtent == logicalExtent &&
        predicateOp && predicateOp->isBeforeInBlock(anchor))
      return entry.predicate;
  }
  int32_t rank = tensorType.getRank();
  int64_t physicalExtent = tensorType.getShape()[axis];
  auto rangeType =
      RankedTensorType::get({physicalExtent}, builder.getI32Type());
  Value expanded = triton::MakeRangeOp::create(builder, anchor->getLoc(),
                                               rangeType, 0, physicalExtent);
  LogicalShape expandedShape{physicalExtent};
  for (int32_t dim = 0; dim < rank; ++dim) {
    if (dim == axis)
      continue;
    expandedShape.insert(expandedShape.begin() + dim, 1);
    expanded = triton::ExpandDimsOp::create(
        builder, anchor->getLoc(),
        RankedTensorType::get(expandedShape, builder.getI32Type()), expanded,
        dim);
  }
  Value extent = arith::ConstantIntOp::create(
      builder, anchor->getLoc(), builder.getI32Type(), logicalExtent);
  Value extentTensor = triton::SplatOp::create(builder, anchor->getLoc(),
                                               expanded.getType(), extent);
  Value predicate =
      arith::CmpIOp::create(builder, anchor->getLoc(),
                            arith::CmpIPredicate::slt, expanded, extentTensor);
  if (expandedShape != tensorType.getShape())
    predicate = triton::BroadcastOp::create(
        builder, anchor->getLoc(),
        RankedTensorType::get(tensorType.getShape(), builder.getI1Type()),
        predicate);
  cache.push_back(
      {builder.getInsertionBlock(),
       LogicalShape(tensorType.getShape().begin(), tensorType.getShape().end()),
       axis, logicalExtent, predicate});
  return predicate;
}

static Value createIdentityTensor(OpBuilder &builder, Location loc,
                                  RankedTensorType type,
                                  LogicalReductionIdentity identity) {
  Type elementType = type.getElementType();
  Attribute scalar;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    double value = 0.0;
    switch (identity) {
    case LogicalReductionIdentity::NegativeInfinity:
      value = -std::numeric_limits<double>::infinity();
      break;
    case LogicalReductionIdentity::PositiveInfinity:
      value = std::numeric_limits<double>::infinity();
      break;
    case LogicalReductionIdentity::One:
      value = 1.0;
      break;
    case LogicalReductionIdentity::Zero:
      break;
    default:
      llvm_unreachable("boolean identity used with float tensor");
    }
    scalar = FloatAttr::get(floatType, value);
  } else {
    auto integerType = cast<IntegerType>(elementType);
    APInt value(integerType.getWidth(), 0);
    if (identity == LogicalReductionIdentity::True)
      value = APInt::getAllOnes(integerType.getWidth());
    else if (identity == LogicalReductionIdentity::One)
      value = APInt(integerType.getWidth(), 1);
    scalar = IntegerAttr::get(integerType, value);
  }
  return arith::ConstantOp::create(builder, loc,
                                   DenseElementsAttr::get(type, scalar));
}

static void replaceFollowingUses(Value original, Value replacement,
                                 Operation *anchor) {
  auto replace = [&](Operation *op) {
    if (isa<triton::ReduceOp>(op))
      return;
    for (OpOperand &operand : op->getOpOperands())
      if (operand.get() == original)
        operand.set(replacement);
  };
  for (Operation *following = anchor->getNextNode(); following;
       following = following->getNextNode()) {
    replace(following);
    following->walk([&](Operation *nested) {
      if (nested != following)
        replace(nested);
    });
  }
}
} // namespace

void applyLogicalTensorActions(LogicalDomainPlan &plan) {
  for (LogicalWGMMAAction &action : plan.wgmmas) {
    if (action.activeN) {
      action.op->removeAttr("active_k");
      action.op->setAttr(
          "active_n",
          IntegerAttr::get(IntegerType::get(plan.module.getContext(), 32),
                           *action.activeN));
    }
  }
  SmallVector<PredicateCacheEntry> predicateCache;
  for (LogicalFragmentFoldAction &action : plan.folds) {
    if (action.mechanism == LogicalFragmentFoldMechanism::WGMMAActiveK) {
      auto dot = cast<WGMMAOp>(action.op);
      dot->removeAttr("active_n");
      dot->setAttr(
          "active_k",
          IntegerAttr::get(IntegerType::get(plan.module.getContext(), 32),
                           action.logicalExtent));
      continue;
    }
    Value operand = action.op->getOperand(action.operandIndex);
    auto type = cast<RankedTensorType>(operand.getType());
    OpBuilder builder(action.op);
    Value predicate =
        createLogicalPredicate(builder, action.op, type, action.axis,
                               action.logicalExtent, predicateCache);
    Value identity = createIdentityTensor(builder, action.op->getLoc(), type,
                                          action.identity);
    Value masked = arith::SelectOp::create(builder, action.op->getLoc(),
                                           predicate, operand, identity);
    action.op->setOperand(action.operandIndex, masked);
    if (action.identity == LogicalReductionIdentity::NegativeInfinity)
      replaceFollowingUses(operand, masked, action.op);
  }
  for (LogicalFragmentGuardAction &action : plan.guards) {
    Value value = action.op->getOperand(action.valueOperand);
    auto type = cast<RankedTensorType>(value.getType());
    OpBuilder builder(action.op);
    Value logicalMask =
        createLogicalPredicate(builder, action.op, type, action.axis,
                               action.logicalExtent, predicateCache);
    if (action.maskOperand) {
      Value userMask = action.op->getOperand(*action.maskOperand);
      logicalMask = arith::AndIOp::create(builder, action.op->getLoc(),
                                          userMask, logicalMask);
    }
    if (auto store = dyn_cast<triton::StoreOp>(action.op))
      store.getMaskMutable().assign(logicalMask);
    else
      cast<triton::AtomicRMWOp>(action.op).getMaskMutable().assign(logicalMask);
  }
}

} // namespace mlir::triton::tle
