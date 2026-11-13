/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */

#ifndef TRITON_TO_GRAPH_PROGRAM_AXIS_DEPENDENCE_ANALYSIS_H
#define TRITON_TO_GRAPH_PROGRAM_AXIS_DEPENDENCE_ANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>

namespace mlir {
namespace triton {
namespace cfg {

enum class StoreAddressIndependence : uint8_t {
  NotProgramDependent,
  ProvenDisjoint,
  CanonicalDynamicStride,
  Unknown,
};

struct StoreAddressDependence {
  Operation *store = nullptr;
  bool pointerDependsOnAxis = false;
  StoreAddressIndependence independence =
      StoreAddressIndependence::NotProgramDependent;
};

struct ProgramAxisDependence {
  int32_t axis = 0;
  SmallVector<Value> programIds;
  SmallVector<Operation *> dependenceClosure;
  SmallVector<StoreAddressDependence> stores;
  SmallVector<int32_t> reductionAxes;

  bool escapes = false;
  bool hasSideEffects = false;
  bool hasUnsupportedSideEffects = false;
  bool readsNumPrograms = false;

  bool hasProgramId() const { return !programIds.empty(); }
  bool hasOnlyDisjointStoreAddresses() const;
  bool hasOnlyLiftableStoreAddressesForProgramMapping() const;
  bool isIndependentAxisTransformCandidate() const;
  bool isProgramMappingTransformCandidate() const;
};

class ProgramAxisDependenceAnalysis {
public:
  explicit ProgramAxisDependenceAnalysis(triton::FuncOp function);

  const ProgramAxisDependence &get(int32_t axis) const;

private:
  std::array<ProgramAxisDependence, 3> axes;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif
