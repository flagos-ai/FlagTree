#ifndef TRITON_METAX_GLUON_C500_PHYSICAL_REGISTER_SLICE_H
#define TRITON_METAX_GLUON_C500_PHYSICAL_REGISTER_SLICE_H

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::triton::gpu::metax::gluon {

namespace detail {

inline FailureOr<uint64_t> checkedProduct(ArrayRef<uint64_t> factors) {
  uint64_t result = 1;
  for (uint64_t factor : factors) {
    std::optional<uint64_t> next =
        llvm::checkedMulUnsigned(result, factor);
    if (!next)
      return failure();
    result = *next;
  }
  return result;
}

/// Delinearize an index using a minor-to-major dimension order.
inline FailureOr<SmallVector<uint64_t>>
delinearize(uint64_t linear, ArrayRef<uint64_t> shape,
            ArrayRef<unsigned> order) {
  if (shape.empty() || shape.size() != order.size() ||
      llvm::is_contained(shape, uint64_t{0}))
    return failure();

  FailureOr<uint64_t> domain = checkedProduct(shape);
  if (failed(domain) || linear >= *domain)
    return failure();

  SmallVector<uint64_t> coordinates(shape.size());
  SmallVector<bool> seen(shape.size(), false);
  for (unsigned dimension : order) {
    if (dimension >= shape.size() || seen[dimension])
      return failure();
    seen[dimension] = true;
    coordinates[dimension] = linear % shape[dimension];
    linear /= shape[dimension];
  }
  if (linear != 0 || llvm::any_of(seen, [](bool value) { return !value; }))
    return failure();
  return coordinates;
}

/// Linearize coordinates in minor-to-major order without intermediate
/// overflow. Computing the complete domain also rejects layouts whose register
/// address space cannot be represented by the decoder.
inline FailureOr<uint64_t> linearize(ArrayRef<uint64_t> coordinates,
                                     ArrayRef<uint64_t> shape) {
  if (coordinates.empty() || coordinates.size() != shape.size())
    return failure();

  uint64_t linear = 0;
  uint64_t stride = 1;
  for (auto [coordinate, extent] : llvm::zip(coordinates, shape)) {
    if (extent == 0 || coordinate >= extent)
      return failure();
    std::optional<uint64_t> term =
        llvm::checkedMulUnsigned(coordinate, stride);
    if (!term)
      return failure();
    std::optional<uint64_t> next = llvm::checkedAddUnsigned(linear, *term);
    if (!next)
      return failure();
    linear = *next;
    std::optional<uint64_t> nextStride =
        llvm::checkedMulUnsigned(stride, extent);
    if (!nextStride)
      return failure();
    stride = *nextStride;
  }
  return linear;
}

inline bool hasSupportedMacaShape(RankedTensorType type,
                                  MACAMmaEncodingAttr mma) {
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  return type.getRank() == 2 && mma.getWarpsPerCTA().size() == 2 &&
         elements.size() == 3 && llvm::none_of(elements, [](unsigned value) {
           return value == 0;
         }) &&
         mma.getVersionMajor() == 2 && !mma.getIsATrans() &&
         !mma.getIsBTrans();
}

} // namespace detail

/// Decode the source-thread register indices selected by ttg.extract_tensor
/// and ttg.insert_tensor. This is the single physical contract shared by the
/// Gluon slice planner, the late verifier, and the MetaX LLVM lowering.
inline FailureOr<SmallVector<unsigned>> decodeRegisterSliceIndices(
    RankedTensorType sourceType, ArrayRef<int64_t> ctaIndices,
    ArrayRef<int64_t> elementIndices) {
  if (!sourceType || !sourceType.getEncoding())
    return failure();

  // LinearLayout and the target ABI use bounded integer dimensions. Reject
  // malformed or dynamic tensor shapes before any target utility can narrow
  // them or construct a physical layout from them.
  if (llvm::any_of(sourceType.getShape(), [](int64_t dimension) {
        return ShapedType::isDynamic(dimension) || dimension <= 0 ||
               static_cast<uint64_t>(dimension) >
                   std::numeric_limits<unsigned>::max();
      }) ||
      ctaIndices.empty() || elementIndices.empty())
    return failure();

  Attribute encoding = sourceType.getEncoding();
  if (auto dot = dyn_cast<DotOperandEncodingAttr>(encoding)) {
    auto mma = dyn_cast<MACAMmaEncodingAttr>(dot.getParent());
    if (!mma || (dot.getOpIdx() != 0 && dot.getOpIdx() != 1) ||
        !detail::hasSupportedMacaShape(sourceType, mma))
      return failure();
  } else if (auto mma = dyn_cast<MACAMmaEncodingAttr>(encoding)) {
    if (!detail::hasSupportedMacaShape(sourceType, mma))
      return failure();
  } else if (!isa<BlockedEncodingAttr>(encoding)) {
    return failure();
  }

  triton::LinearLayout sourceLayout = toLinearLayout(sourceType);
  StringAttr reg = StringAttr::get(sourceType.getContext(), "register");
  if (!sourceLayout.hasInDim(reg))
    return failure();
  int32_t signedRegisterCount = sourceLayout.getInDimSize(reg);
  if (signedRegisterCount <= 0)
    return failure();
  uint64_t registerCount = static_cast<uint64_t>(signedRegisterCount);

  std::optional<uint64_t> selectedCardinality = llvm::checkedMulUnsigned(
      static_cast<uint64_t>(ctaIndices.size()),
      static_cast<uint64_t>(elementIndices.size()));
  if (!selectedCardinality)
    return failure();

  SmallVector<unsigned> selected;

  if (isa<BlockedEncodingAttr, DotOperandEncodingAttr>(encoding)) {
    SmallVector<uint64_t> sizePerThread;
    if (auto blocked = dyn_cast<BlockedEncodingAttr>(encoding)) {
      llvm::append_range(sizePerThread, blocked.getSizePerThread());
    } else {
      auto dot = cast<DotOperandEncodingAttr>(encoding);
      auto mma = cast<MACAMmaEncodingAttr>(dot.getParent());
      ArrayRef<unsigned> elements = mma.getElementsMNK();
      sizePerThread = dot.getOpIdx() == 0
                          ? SmallVector<uint64_t>{elements[0], elements[2]}
                          : SmallVector<uint64_t>{elements[2], elements[1]};
    }

    if (sizePerThread.size() != static_cast<size_t>(sourceType.getRank()) ||
        llvm::is_contained(sizePerThread, uint64_t{0}))
      return failure();
    FailureOr<uint64_t> registersPerReplica =
        detail::checkedProduct(sizePerThread);
    if (failed(registersPerReplica) || *registersPerReplica == 0 ||
        registerCount % *registersPerReplica != 0)
      return failure();
    uint64_t replicaCount = registerCount / *registersPerReplica;

    for (int64_t ctaIndex : ctaIndices) {
      for (int64_t elementIndex : elementIndices) {
        if (ctaIndex < 0 || elementIndex < 0)
          return failure();
        uint64_t cta = static_cast<uint64_t>(ctaIndex);
        uint64_t element = static_cast<uint64_t>(elementIndex);
        if (cta >= replicaCount || element >= *registersPerReplica)
          return failure();
        std::optional<uint64_t> linear = llvm::checkedMulAddUnsigned(
            cta, *registersPerReplica, element);
        if (!linear || *linear >= registerCount ||
            *linear > std::numeric_limits<unsigned>::max())
          return failure();
        selected.push_back(static_cast<unsigned>(*linear));
      }
    }
  } else if (auto mma = dyn_cast<MACAMmaEncodingAttr>(encoding)) {
    SmallVector<unsigned> shapePerCTA = getShapePerCTATile(sourceType);
    ArrayRef<unsigned> elements = mma.getElementsMNK();
    if (shapePerCTA.size() != 2 || shapePerCTA[0] == 0 ||
        shapePerCTA[1] == 0)
      return failure();

    SmallVector<uint64_t> replicas = {
        static_cast<uint64_t>(sourceType.getShape()[0]) / shapePerCTA[0],
        static_cast<uint64_t>(sourceType.getShape()[1]) / shapePerCTA[1]};
    SmallVector<uint64_t> elementsMN = {elements[0], elements[1]};
    FailureOr<uint64_t> replicaDomain = detail::checkedProduct(replicas);
    FailureOr<uint64_t> elementDomain = detail::checkedProduct(elementsMN);
    if (failed(replicaDomain) || failed(elementDomain) ||
        *replicaDomain == 0 || *elementDomain == 0)
      return failure();
    SmallVector<unsigned> order = mma.getColMajor()
                                      ? SmallVector<unsigned>{0, 1}
                                      : SmallVector<unsigned>{1, 0};

    FailureOr<uint64_t> registerDomain = detail::checkedProduct(
        SmallVector<uint64_t>{*replicaDomain, *elementDomain, 4});
    if (failed(registerDomain) || *registerDomain != registerCount)
      return failure();

    std::optional<uint64_t> mmaCardinality =
        llvm::checkedMulUnsigned(*selectedCardinality, uint64_t{4});
    if (!mmaCardinality ||
        *mmaCardinality > std::numeric_limits<size_t>::max())
      return failure();
    selected.reserve(static_cast<size_t>(*mmaCardinality));

    for (int64_t ctaIndex : ctaIndices) {
      if (ctaIndex < 0 ||
          static_cast<uint64_t>(ctaIndex) >= *replicaDomain)
        return failure();
      FailureOr<SmallVector<uint64_t>> replica = detail::delinearize(
          static_cast<uint64_t>(ctaIndex), replicas, order);
      if (failed(replica))
        return failure();
      for (int64_t elementIndex : elementIndices) {
        if (elementIndex < 0 ||
            static_cast<uint64_t>(elementIndex) >= *elementDomain)
          return failure();
        FailureOr<SmallVector<uint64_t>> element = detail::delinearize(
            static_cast<uint64_t>(elementIndex), elementsMN, order);
        if (failed(element))
          return failure();
        for (uint64_t depth = 0; depth < 4; ++depth) {
          SmallVector<uint64_t> coordinates;
          SmallVector<uint64_t> physicalShape;
          if (mma.getColMajor()) {
            coordinates = {(*element)[1], depth, (*replica)[1],
                           (*element)[0], (*replica)[0]};
            physicalShape = {elementsMN[1], 4, replicas[1], elementsMN[0],
                             replicas[0]};
          } else {
            coordinates = {(*element)[1], (*element)[0], depth,
                           (*replica)[1], (*replica)[0]};
            physicalShape = {elementsMN[1], elementsMN[0], 4, replicas[1],
                             replicas[0]};
          }
          FailureOr<uint64_t> linear =
              detail::linearize(coordinates, physicalShape);
          if (failed(linear) || *linear >= registerCount ||
              *linear > std::numeric_limits<unsigned>::max())
            return failure();
          selected.push_back(static_cast<unsigned>(*linear));
        }
      }
    }
  }

  uint64_t expectedCardinality = *selectedCardinality;
  if (isa<MACAMmaEncodingAttr>(encoding)) {
    std::optional<uint64_t> expanded =
        llvm::checkedMulUnsigned(expectedCardinality, uint64_t{4});
    if (!expanded)
      return failure();
    expectedCardinality = *expanded;
  }
  if (expectedCardinality != selected.size())
    return failure();

  llvm::sort(selected);
  if (std::adjacent_find(selected.begin(), selected.end()) != selected.end())
    return failure();
  return selected;
}

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_C500_PHYSICAL_REGISTER_SLICE_H
