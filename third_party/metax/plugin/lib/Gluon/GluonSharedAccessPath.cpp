#include "Gluon/GluonSharedAccessPath.h"
#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Targets/GluonC500Layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <limits>

#define DEBUG_TYPE "metax-gluon-shared-access-path"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace {

constexpr unsigned kMaxAccessPathAlternatives = 8;
constexpr uint64_t kMaxEnumeratedSharedElements = 1u << 18;

bool haveSameFact(const SharedAccessPathFact &lhs,
                  const SharedAccessPathFact &rhs) {
  return lhs.logicalRoot == rhs.logicalRoot && lhs.coverage == rhs.coverage &&
         lhs.slot == rhs.slot && lhs.indices == rhs.indices &&
         lhs.offsets == rhs.offsets && lhs.byteBase == rhs.byteBase &&
         lhs.logicalOffsets == rhs.logicalOffsets &&
         lhs.byteInterval == rhs.byteInterval;
}

void appendFact(SharedAccessPathState &state, SharedAccessPathFact fact) {
  if (llvm::any_of(state.alternatives, [&](const SharedAccessPathFact &other) {
        return haveSameFact(fact, other);
      }))
    return;
  if (state.alternatives.size() == kMaxAccessPathAlternatives) {
    state.alternatives.clear();
    state.unknown = true;
    return;
  }
  state.alternatives.push_back(std::move(fact));
}

void joinState(SharedAccessPathState &target,
               const SharedAccessPathState &source) {
  target.unknown |= source.unknown;
  if (target.unknown) {
    target.alternatives.clear();
    return;
  }
  for (const SharedAccessPathFact &fact : source.alternatives)
    appendFact(target, fact);
}

bool isMemDesc(Value value) { return isa<ttg::MemDescType>(value.getType()); }

std::optional<uint64_t> checkedProduct(ArrayRef<int64_t> values) {
  uint64_t result = 1;
  for (int64_t value : values) {
    if (ShapedType::isDynamic(value) || value < 0 ||
        (value != 0 && result > std::numeric_limits<uint64_t>::max() /
                                  static_cast<uint64_t>(value)))
      return std::nullopt;
    result *= static_cast<uint64_t>(value);
  }
  return result;
}

std::optional<uint64_t> getElementByteWidth(ttg::MemDescType type) {
  Type elementType = type.getElementType();
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

std::optional<uint64_t> getFootprintBytes(ttg::MemDescType type) {
  std::optional<uint64_t> elementBytes = getElementByteWidth(type);
  if (!elementBytes)
    return std::nullopt;

  SmallVector<int64_t> shape = ttg::getShapePerCTA(type);
  std::optional<uint64_t> elements;
  if (auto padded =
          dyn_cast<ttg::PaddedSharedEncodingAttr>(type.getEncoding())) {
    int64_t paddedElements = padded.getPaddedSize(shape);
    if (paddedElements < 0)
      return std::nullopt;
    elements = static_cast<uint64_t>(paddedElements);
  } else {
    elements = checkedProduct(ttg::getAllocationShapePerCTA(type));
  }
  if (!elements || *elements > std::numeric_limits<uint64_t>::max() /
                                    *elementBytes)
    return std::nullopt;
  return *elements * *elementBytes;
}

std::optional<uint64_t>
applyStaticPadding(ttg::PaddedSharedEncodingAttr padded, uint64_t offset) {
  uint64_t padding = 0;
  for (auto [interval, amount] :
       llvm::zip_equal(padded.getIntervals(), padded.getPaddings())) {
    if (!llvm::isPowerOf2_64(interval) || !llvm::isPowerOf2_64(amount))
      return std::nullopt;
    uint64_t groups = offset >> llvm::Log2_64(interval);
    if (groups > std::numeric_limits<uint64_t>::max() / amount ||
        padding > std::numeric_limits<uint64_t>::max() - groups * amount)
      return std::nullopt;
    padding += groups * amount;
  }
  if (offset > std::numeric_limits<uint64_t>::max() - padding)
    return std::nullopt;
  return offset + padding;
}

std::optional<uint64_t>
getPhysicalElementOffset(ttg::MemDescType type, ArrayRef<int64_t> coordinates) {
  if (coordinates.size() != static_cast<size_t>(type.getRank()) ||
      llvm::any_of(coordinates, [](int64_t value) {
        return value < 0 || value > std::numeric_limits<int32_t>::max();
      }))
    return std::nullopt;
  auto encoding = dyn_cast<ttg::LayoutEncodingTrait>(type.getEncoding());
  if (!encoding || encoding.getRank() != static_cast<unsigned>(type.getRank()))
    return std::nullopt;

  triton::LinearLayout layout;
  if (auto padded =
          dyn_cast<ttg::PaddedSharedEncodingAttr>(type.getEncoding()))
    layout = padded.getLinearComponent();
  else {
    FailureOr<triton::LinearLayout> linear = getC500LinearLayout(type);
    if (failed(linear))
      return std::nullopt;
    layout = std::move(*linear);
  }

  MLIRContext *context = type.getContext();
  StringAttr offsetDim = StringAttr::get(context, "offset");
  SmallVector<StringAttr> logicalDims =
      standardOutDimNames(context, type.getRank());
  triton::LinearLayout logicalToOffset =
      layout.sublayout({offsetDim}, logicalDims).invert();
  SmallVector<std::pair<StringAttr, int32_t>> namedCoordinates;
  for (auto [dimension, coordinate] : llvm::zip(logicalDims, coordinates))
    namedCoordinates.emplace_back(dimension, static_cast<int32_t>(coordinate));
  auto result = logicalToOffset.apply(namedCoordinates);
  auto offset = llvm::find_if(result, [&](const auto &entry) {
    return entry.first == offsetDim;
  });
  if (offset == result.end() || offset->second < 0)
    return std::nullopt;

  uint64_t physicalOffset = static_cast<uint64_t>(offset->second);
  if (auto padded =
          dyn_cast<ttg::PaddedSharedEncodingAttr>(type.getEncoding()))
    return applyStaticPadding(padded, physicalOffset);
  return physicalOffset;
}

void refreshByteInterval(Value value, SharedAccessPathFact &fact) {
  fact.byteInterval.reset();
  auto type = dyn_cast<ttg::MemDescType>(value.getType());
  if (!type || !fact.byteBase ||
      fact.logicalOffsets.size() != static_cast<size_t>(type.getRank()))
    return;

  std::optional<uint64_t> elementBytes = getElementByteWidth(type);
  std::optional<uint64_t> elementCount = checkedProduct(type.getShape());
  if (!elementBytes || !elementCount ||
      *elementCount > kMaxEnumeratedSharedElements)
    return;
  if (*elementCount == 0) {
    fact.byteInterval = std::make_pair(*fact.byteBase, *fact.byteBase);
    return;
  }

  uint64_t minByte = std::numeric_limits<uint64_t>::max();
  uint64_t maxByte = 0;
  SmallVector<int64_t> coordinates(type.getRank());
  for (uint64_t linear = 0; linear < *elementCount; ++linear) {
    uint64_t remainder = linear;
    for (int dimension = type.getRank() - 1; dimension >= 0; --dimension) {
      int64_t extent = type.getDimSize(dimension);
      coordinates[dimension] = fact.logicalOffsets[dimension] +
                               static_cast<int64_t>(remainder % extent);
      remainder /= extent;
    }
    std::optional<uint64_t> elementOffset =
        getPhysicalElementOffset(type, coordinates);
    if (!elementOffset ||
        *elementOffset > (std::numeric_limits<uint64_t>::max() -
                          *fact.byteBase) /
                             *elementBytes)
      return;
    uint64_t byte = *fact.byteBase + *elementOffset * *elementBytes;
    minByte = std::min(minByte, byte);
    maxByte = std::max(maxByte, byte);
  }
  if (maxByte > std::numeric_limits<uint64_t>::max() - *elementBytes)
    return;
  fact.byteInterval = std::make_pair(minByte, maxByte + *elementBytes);
}

SharedAccessPathFact makeRootFact(Value value) {
  auto type = cast<ttg::MemDescType>(value.getType());
  SharedAccessPathFact fact;
  fact.logicalRoot = value;
  fact.coverage = SharedAccessCoverage::Whole;
  fact.byteBase = 0;
  fact.logicalOffsets.assign(type.getRank(), 0);
  if (std::optional<uint64_t> bytes = getFootprintBytes(type))
    fact.byteInterval = std::make_pair(0, *bytes);
  return fact;
}

void transferIndexPhysicalPath(ttg::MemDescIndexOp index,
                               SharedAccessPathFact &fact) {
  fact.byteInterval.reset();
  auto destinationType = cast<ttg::MemDescType>(index.getType());
  APInt constantIndex;
  if (!fact.byteBase ||
      llvm::any_of(fact.logicalOffsets,
                   [](int64_t offset) { return offset != 0; }) ||
      !matchPattern(index.getIndex(), m_ConstantInt(&constantIndex))) {
    fact.byteBase.reset();
    fact.logicalOffsets.clear();
    return;
  }

  int64_t indexValue = constantIndex.getSExtValue();
  std::optional<uint64_t> strideElements =
      checkedProduct(ttg::getAllocationShapePerCTA(destinationType));
  std::optional<uint64_t> elementBytes = getElementByteWidth(destinationType);
  if (indexValue < 0 || !strideElements || !elementBytes ||
      static_cast<uint64_t>(indexValue) >
          std::numeric_limits<uint64_t>::max() / *strideElements) {
    fact.byteBase.reset();
    fact.logicalOffsets.clear();
    return;
  }
  uint64_t elementOffset = static_cast<uint64_t>(indexValue) * *strideElements;
  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(
          destinationType.getEncoding())) {
    std::optional<uint64_t> paddedOffset =
        applyStaticPadding(padded, elementOffset);
    if (!paddedOffset) {
      fact.byteBase.reset();
      fact.logicalOffsets.clear();
      return;
    }
    elementOffset = *paddedOffset;
  }
  if (elementOffset > (std::numeric_limits<uint64_t>::max() -
                       *fact.byteBase) /
                          *elementBytes) {
    fact.byteBase.reset();
    fact.logicalOffsets.clear();
    return;
  }
  fact.byteBase = *fact.byteBase + elementOffset * *elementBytes;
  fact.logicalOffsets.assign(destinationType.getRank(), 0);
  refreshByteInterval(index.getResult(), fact);
}

void transferReinterpretPhysicalPath(ttg::MemDescReinterpretOp reinterpret,
                                     SharedAccessPathFact &fact) {
  auto sourceType = cast<ttg::MemDescType>(reinterpret.getSrc().getType());
  auto destinationType = cast<ttg::MemDescType>(reinterpret.getType());
  std::optional<uint64_t> elementOffset =
      getPhysicalElementOffset(sourceType, fact.logicalOffsets);
  std::optional<uint64_t> sourceElementBytes =
      getElementByteWidth(sourceType);
  if (!fact.byteBase || !elementOffset || !sourceElementBytes ||
      *elementOffset > (std::numeric_limits<uint64_t>::max() -
                        *fact.byteBase) /
                           *sourceElementBytes) {
    fact.byteBase.reset();
    fact.logicalOffsets.clear();
    fact.byteInterval.reset();
    return;
  }
  fact.byteBase = *fact.byteBase + *elementOffset * *sourceElementBytes;
  fact.logicalOffsets.assign(destinationType.getRank(), 0);
  refreshByteInterval(reinterpret.getResult(), fact);
}

bool isWholeSubslice(ttg::MemDescSubsliceOp subslice) {
  auto sourceType = cast<ttg::MemDescType>(subslice.getSrc().getType());
  auto resultType = cast<ttg::MemDescType>(subslice.getType());
  return sourceType.getShape() == resultType.getShape() &&
         llvm::all_of(subslice.getOffsets(),
                      [](int32_t offset) { return offset == 0; });
}

enum class ReinterpretExtentRelation { Equal, Subset, Unknown };

std::optional<uint64_t> getStaticBitExtent(ttg::MemDescType type) {
  Type elementType = type.getElementType();
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  uint64_t bits = elementType.getIntOrFloatBitWidth();
  for (int64_t extent : type.getShape()) {
    if (ShapedType::isDynamic(extent) || extent < 0 ||
        (extent != 0 && bits > std::numeric_limits<uint64_t>::max() /
                                      static_cast<uint64_t>(extent)))
      return std::nullopt;
    bits *= static_cast<uint64_t>(extent);
  }
  return bits;
}

ReinterpretExtentRelation
classifyReinterpretExtent(ttg::MemDescReinterpretOp reinterpret) {
  auto sourceType = cast<ttg::MemDescType>(reinterpret.getSrc().getType());
  auto resultType = cast<ttg::MemDescType>(reinterpret.getType());
  std::optional<uint64_t> sourceBits = getStaticBitExtent(sourceType);
  std::optional<uint64_t> resultBits = getStaticBitExtent(resultType);
  if (!sourceBits || !resultBits || *resultBits > *sourceBits)
    return ReinterpretExtentRelation::Unknown;
  return *resultBits == *sourceBits ? ReinterpretExtentRelation::Equal
                                    : ReinterpretExtentRelation::Subset;
}

const SharedAccessPathFact *getSingletonKnownFact(
    const SharedAccessPathState &state) {
  return !state.unknown && state.alternatives.size() == 1
             ? &state.alternatives.front()
             : nullptr;
}

/// Return the source of a coverage-preserving view, or of a subslice whose
/// reads are covered by a complete initialization of the source. Indexing is
/// intentionally excluded: two child slots of the same carrier need distinct
/// access identities unless their index correlation is proven explicitly.
Value getCarrierIdentitySource(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {};
  return TypeSwitch<Operation *, Value>(result.getOwner())
      .Case<ttg::MemDescSubsliceOp>([](ttg::MemDescSubsliceOp op) -> Value {
        return isWholeSubslice(op) ? op.getSrc() : Value();
      })
      .Case<ttg::MemDescReinterpretOp>(
          [](ttg::MemDescReinterpretOp op) -> Value {
            return classifyReinterpretExtent(op) ==
                           ReinterpretExtentRelation::Equal
                       ? op.getSrc()
                       : Value();
          })
      .Case<ttg::MemDescTransOp, ttg::MemDescReshapeOp,
            gluon_dialect::RequireLayoutOp,
            gluon_dialect::ReleaseLayoutOp>(
          [](auto op) -> Value { return op.getSrc(); })
      .Default([](Operation *) -> Value { return {}; });
}

} // namespace

StringRef stringifySharedAccessCoverage(SharedAccessCoverage coverage) {
  switch (coverage) {
  case SharedAccessCoverage::Whole:
    return "whole";
  case SharedAccessCoverage::ExactSlot:
    return "exact-slot";
  case SharedAccessCoverage::Partial:
    return "partial";
  case SharedAccessCoverage::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown shared access coverage");
}

bool SharedAccessPathState::operator==(
    const SharedAccessPathState &other) const {
  if (unknown != other.unknown || alternatives.size() != other.alternatives.size())
    return false;
  return llvm::all_of(llvm::zip(alternatives, other.alternatives),
                      [](auto pair) {
                        return haveSameFact(std::get<0>(pair),
                                            std::get<1>(pair));
                      });
}

SharedAccessPathState SharedAccessPathAnalysis::transfer(Value value) const {
  SharedAccessPathState result;
  if (!isMemDesc(value))
    return result;

  if (carriers.isCarrier(value)) {
    for (Value predecessor : carriers.getPredecessors(value))
      joinState(result, lookup(predecessor));
    return result;
  }

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (isa_and_nonnull<tt::FuncOp>(argument.getOwner()->getParentOp()))
      appendFact(result, makeRootFact(value));
    else
      result.unknown = true;
    return result;
  }

  auto opResult = dyn_cast<OpResult>(value);
  if (!opResult) {
    result.unknown = true;
    return result;
  }
  Operation *def = opResult.getOwner();
  if (isa<ttg::LocalAllocOp>(def)) {
    appendFact(result, makeRootFact(value));
    return result;
  }

  Value source;
  bool handled = true;
  TypeSwitch<Operation *>(def)
      .Case<ttg::MemDescIndexOp>([&](ttg::MemDescIndexOp index) {
        source = index.getSrc();
        for (SharedAccessPathFact fact : lookup(source).alternatives) {
          APInt constantIndex;
          // A view may change the coordinate represented by a later index;
          // only a direct root index establishes a physical root slot.
          const bool selectsRootSlot =
              index.getSrc() == fact.logicalRoot &&
              fact.coverage == SharedAccessCoverage::Whole &&
              fact.indices.empty();
          if (fact.coverage != SharedAccessCoverage::Partial)
            fact.coverage = SharedAccessCoverage::ExactSlot;
          if (matchPattern(index.getIndex(), m_ConstantInt(&constantIndex))) {
            int64_t constant = constantIndex.getSExtValue();
            if (selectsRootSlot)
              fact.slot = constant;
            fact.indices.push_back({constant, {}});
          } else {
            if (selectsRootSlot)
              fact.slot.reset();
            fact.indices.push_back({std::nullopt, index.getIndex()});
          }
          transferIndexPhysicalPath(index, fact);
          appendFact(result, std::move(fact));
        }
        result.unknown |= lookup(source).unknown;
      })
      .Case<ttg::MemDescSubsliceOp>([&](ttg::MemDescSubsliceOp subslice) {
        source = subslice.getSrc();
        for (SharedAccessPathFact fact : lookup(source).alternatives) {
          if (!isWholeSubslice(subslice)) {
            fact.coverage = SharedAccessCoverage::Partial;
            llvm::append_range(fact.offsets, subslice.getOffsets());
          }
          if (fact.logicalOffsets.size() == subslice.getOffsets().size()) {
            for (auto [offset, delta] :
                 llvm::zip(fact.logicalOffsets, subslice.getOffsets()))
              offset += delta;
            refreshByteInterval(subslice.getResult(), fact);
          } else {
            fact.byteBase.reset();
            fact.logicalOffsets.clear();
            fact.byteInterval.reset();
          }
          appendFact(result, std::move(fact));
        }
        result.unknown |= lookup(source).unknown;
      })
      .Case<ttg::MemDescTransOp>([&](ttg::MemDescTransOp op) {
        source = op.getSrc();
        for (SharedAccessPathFact fact : lookup(source).alternatives) {
          if (fact.logicalOffsets.size() == op.getOrder().size()) {
            SmallVector<int64_t> transposedOffsets;
            transposedOffsets.reserve(op.getOrder().size());
            for (int32_t dimension : op.getOrder())
              transposedOffsets.push_back(fact.logicalOffsets[dimension]);
            fact.logicalOffsets = std::move(transposedOffsets);
            refreshByteInterval(op.getResult(), fact);
          } else {
            fact.byteBase.reset();
            fact.logicalOffsets.clear();
            fact.byteInterval.reset();
          }
          appendFact(result, std::move(fact));
        }
        result.unknown |= lookup(source).unknown;
      })
      .Case<ttg::MemDescReshapeOp>([&](ttg::MemDescReshapeOp op) {
        source = op.getSrc();
        auto sourceType = cast<ttg::MemDescType>(source.getType());
        auto destinationType = cast<ttg::MemDescType>(op.getType());
        for (SharedAccessPathFact fact : lookup(source).alternatives) {
          std::optional<uint64_t> sourceElements =
              checkedProduct(sourceType.getShape());
          if (!sourceElements ||
              fact.logicalOffsets.size() !=
                  static_cast<size_t>(sourceType.getRank())) {
            fact.byteBase.reset();
            fact.logicalOffsets.clear();
            fact.byteInterval.reset();
            appendFact(result, std::move(fact));
            continue;
          }
          uint64_t linearOffset = 0;
          bool valid = true;
          for (auto [offset, extent] :
               llvm::zip(fact.logicalOffsets, sourceType.getShape())) {
            if (offset < 0 || offset >= extent ||
                linearOffset >
                    (std::numeric_limits<uint64_t>::max() - offset) / extent) {
              valid = false;
              break;
            }
            linearOffset = linearOffset * extent + offset;
          }
          if (!valid) {
            fact.byteBase.reset();
            fact.logicalOffsets.clear();
            fact.byteInterval.reset();
            appendFact(result, std::move(fact));
            continue;
          }
          fact.logicalOffsets.assign(destinationType.getRank(), 0);
          for (int dimension = destinationType.getRank() - 1; dimension >= 0;
               --dimension) {
            int64_t extent = destinationType.getDimSize(dimension);
            fact.logicalOffsets[dimension] = linearOffset % extent;
            linearOffset /= extent;
          }
          refreshByteInterval(op.getResult(), fact);
          appendFact(result, std::move(fact));
        }
        result.unknown |= lookup(source).unknown;
      })
      .Case<ttg::MemDescReinterpretOp>([&](ttg::MemDescReinterpretOp op) {
        source = op.getSrc();
        ReinterpretExtentRelation relation = classifyReinterpretExtent(op);
        if (relation == ReinterpretExtentRelation::Unknown) {
          result.unknown = true;
          return;
        }
        for (SharedAccessPathFact fact : lookup(source).alternatives) {
          // Logical bit extent alone does not prove physical confinement:
          // lowering resets coordinates at the source affine base, while the
          // destination allocation shape or padding may escape a root slot.
          fact.slot.reset();
          if (relation != ReinterpretExtentRelation::Equal)
            fact.coverage = SharedAccessCoverage::Partial;
          transferReinterpretPhysicalPath(op, fact);
          appendFact(result, std::move(fact));
        }
        result.unknown |= lookup(source).unknown;
      })
      .Case<gluon_dialect::RequireLayoutOp>(
          [&](gluon_dialect::RequireLayoutOp op) { source = op.getSrc(); })
      .Case<gluon_dialect::ReleaseLayoutOp>(
          [&](gluon_dialect::ReleaseLayoutOp op) { source = op.getSrc(); })
      .Default([&](Operation *) { handled = false; });

  if (!handled) {
    result.unknown = true;
    return result;
  }
  if (source && result.alternatives.empty() && !result.unknown)
    result = lookup(source);
  return result;
}

LogicalResult SharedAccessPathAnalysis::initialize() {
  if (failed(carriers.initialize()))
    return failure();
  states.clear();
  carrierIdentities.clear();
  unknownState = SharedAccessPathState{{}, true};

  llvm::SetVector<Value> values;
  root->walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (isMemDesc(result))
        values.insert(result);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (isMemDesc(argument))
            values.insert(argument);
  });
  for (Value value : values)
    states.try_emplace(value, SharedAccessPathState{});

  const unsigned maxIterations =
      std::max<unsigned>(1, values.size() * (kMaxAccessPathAlternatives + 2));
  bool changed = true;
  unsigned iteration = 0;
  while (changed && iteration++ < maxIterations) {
    changed = false;
    for (Value value : values) {
      SharedAccessPathState next = states.lookup(value);
      joinState(next, transfer(value));
      if (next == states.lookup(value))
        continue;
      states[value] = std::move(next);
      changed = true;
    }
  }
  if (changed)
    return root->emitError()
           << "shared access-path analysis did not reach a bounded fixed point "
              "after "
           << maxIterations << " iterations";

  for (Value value : values) {
    SharedAccessPathState &state = states[value];
    if (state.alternatives.empty())
      state.unknown = true;
    LDBG("[access-path] value=" << value << " alternatives="
                                << state.alternatives.size()
                                << " unknown=" << state.unknown);
  }
  buildCarrierIdentities(values.getArrayRef());
  return success();
}

const SharedAccessPathState &
SharedAccessPathAnalysis::lookup(Value value) const {
  auto it = states.find(value);
  return it == states.end() ? unknownState : it->second;
}

void SharedAccessPathAnalysis::buildCarrierIdentities(ArrayRef<Value> values) {
  DenseMap<Value, Value> parent;
  auto find = [&](Value value) {
    Value current = value;
    while (parent.count(current) && parent[current] != current)
      current = parent[current];
    Value root = current;
    current = value;
    while (parent.count(current) && parent[current] != current) {
      Value next = parent[current];
      parent[current] = root;
      current = next;
    }
    return root;
  };
  auto unite = [&](Value lhs, Value rhs) {
    parent.try_emplace(lhs, lhs);
    parent.try_emplace(rhs, rhs);
    Value lhsRoot = find(lhs);
    Value rhsRoot = find(rhs);
    if (lhsRoot == rhsRoot)
      return false;
    parent[rhsRoot] = lhsRoot;
    return true;
  };

  // Every known access starts with its own exact SSA identity. Reversible
  // views union with their source; memdesc_index keeps a distinct self identity
  // because its selector is part of the access path.
  for (Value value : values)
    if (!lookup(value).unknown && !lookup(value).alternatives.empty())
      parent.try_emplace(value, value);
  for (Value value : values) {
    Value source = getCarrierIdentitySource(value);
    if (source && parent.count(value) && parent.count(source))
      (void)unite(source, value);
  }

  // RegionBranch carriers use consensus, not root equality. A single incoming
  // identity is an exact forwarding relation. Multiple incoming identities
  // may merge only when every path names the same singleton root-coordinate
  // access; a {slot0, slot1} join must retain a distinct identity.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Value successor : values) {
      ArrayRef<Value> predecessors = carriers.getPredecessors(successor);
      if (predecessors.empty() || !parent.count(successor) ||
          llvm::any_of(predecessors,
                       [&](Value value) { return !parent.count(value); }))
        continue;

      Value consensus = find(predecessors.front());
      bool sameIdentity = llvm::all_of(predecessors, [&](Value predecessor) {
        return find(predecessor) == consensus;
      });
      bool sameSingletonFact = false;
      if (!sameIdentity) {
        const SharedAccessPathFact *successorFact =
            getSingletonKnownFact(lookup(successor));
        sameSingletonFact = successorFact &&
                            llvm::all_of(predecessors, [&](Value predecessor) {
                              const SharedAccessPathFact *predecessorFact =
                                  getSingletonKnownFact(lookup(predecessor));
                              return predecessorFact &&
                                     haveSameFact(*predecessorFact,
                                                  *successorFact);
                            });
      }
      if (!sameIdentity && !sameSingletonFact)
        continue;
      for (Value predecessor : predecessors)
        changed |= unite(successor, predecessor);
    }
  }

  DenseMap<Value, Value> canonical;
  for (Value value : values) {
    if (!parent.count(value))
      continue;
    Value component = find(value);
    canonical.try_emplace(component, value);
  }
  for (Value value : values)
    if (parent.count(value))
      carrierIdentities[value] = canonical.lookup(find(value));
}

Value SharedAccessPathAnalysis::getCarrierIdentity(Value value) const {
  auto it = carrierIdentities.find(value);
  return it == carrierIdentities.end() ? Value() : it->second;
}

Value SharedAccessPathAnalysis::getCoveringIdentity(Value value) const {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {};
  Value source = TypeSwitch<Operation *, Value>(result.getOwner())
                     .Case<ttg::MemDescSubsliceOp>(
                         [](ttg::MemDescSubsliceOp op) -> Value {
                           return isWholeSubslice(op) ? Value() : op.getSrc();
                         })
                     .Case<ttg::MemDescReinterpretOp>(
                         [](ttg::MemDescReinterpretOp op) -> Value {
                           return classifyReinterpretExtent(op) ==
                                          ReinterpretExtentRelation::Subset
                                      ? op.getSrc()
                                      : Value();
                         })
                     .Default([](Operation *) -> Value { return {}; });
  return source ? getCarrierIdentity(source) : Value();
}

} // namespace mlir::triton::gpu::metax::gluon
