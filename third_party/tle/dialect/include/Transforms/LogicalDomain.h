/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef TRITON_TLE_TRANSFORMS_LOGICAL_DOMAIN_H_
#define TRITON_TLE_TRANSFORMS_LOGICAL_DOMAIN_H_

#include "mlir/IR/BuiltinOps.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir::triton::tle {

using LogicalShape = SmallVector<int64_t>;

struct LogicalDomainProvenance {
  SmallVector<Operation *, 2> roots;
  SmallVector<Operation *, 2> seeds;

  LogicalDomainProvenance() = default;
  LogicalDomainProvenance(Operation *root, Operation *seed) {
    if (root)
      roots.push_back(root);
    if (seed)
      seeds.push_back(seed);
  }

  Operation *primaryRoot() const {
    return roots.empty() ? nullptr : roots.front();
  }
};

struct MemDescLogicalState {
  LogicalShape physicalShape;
  LogicalShape logicalShape;
  SmallVector<int32_t> axisMap;
  LogicalDomainProvenance provenance;
  bool isStage = false;
  // Logical view relative to the allocation, not storage major order.
  bool viewTransposed = false;
};

/// A tensor carries at most one invalid physical tail.  The physical shape is
/// read from the tensor type; this fact stores only the restricted axis and
/// its valid prefix extent.
struct TensorFragmentState {
  int32_t axis = 0;
  int64_t logicalExtent = 0;
  LogicalDomainProvenance provenance;
};

struct TensorDescriptorLogicalState {
  LogicalShape logicalShape;
  LogicalDomainProvenance provenance;
};

struct LogicalDescriptorRewriteAction {
  LogicalShape blockShape;
  LogicalShape boxShape;
};

enum class LogicalReductionIdentity : uint8_t {
  NegativeInfinity,
  PositiveInfinity,
  Zero,
  One,
  True,
  False,
};

struct LogicalMemDescUseAction {
  OpOperand *use = nullptr;
  bool markTiledPipeField = false;
};

struct LogicalPointerCopyAction {
  LocalPointersOp pointers;
  triton::StoreOp store;
  triton::LoadOp load;
  SmallVector<int64_t, 2> microTileShape;
  unsigned vectorBytes = 16;
};

struct LogicalRootRewriteAction {
  gpu::LocalAllocOp alloc;
  LogicalShape logicalShape;
  SmallVector<int64_t, 2> storageTileShape;
  gpu::NVMMASharedEncodingAttr storageEncoding;
  Value initializer;
  SmallVector<gpu::LocalStoreOp> localStores;
  SmallVector<gpu::MemDescIndexOp> stages;
  SmallVector<gpu::MemDescTransOp> transposes;
  SmallVector<gpu::TMACopyOp> copies;
  SmallVector<LogicalPointerCopyAction> pointerCopies;
  SmallVector<LogicalMemDescUseAction> memdescUses;
  bool reachesWGMMA = false;
};

struct LogicalWGMMAAction {
  WGMMAOp op;
  std::optional<int64_t> activeN;
};

enum class LogicalFragmentFoldMechanism : uint8_t {
  IdentityMask,
  WGMMAActiveK,
};

struct LogicalFragmentFoldAction {
  Operation *op = nullptr;
  unsigned operandIndex = 0;
  int32_t axis = 0;
  int64_t logicalExtent = 0;
  LogicalFragmentFoldMechanism mechanism =
      LogicalFragmentFoldMechanism::IdentityMask;
  LogicalReductionIdentity identity = LogicalReductionIdentity::Zero;
};

struct LogicalFragmentGuardAction {
  Operation *op = nullptr;
  unsigned valueOperand = 0;
  std::optional<unsigned> maskOperand;
  int32_t axis = 0;
  int64_t logicalExtent = 0;
};

/// An analysis product that owns every fact required by the infallible apply
/// phase.  It deliberately stores no temporary IR created during analysis.
struct LogicalDomainPlan {
  ModuleOp module;
  SmallVector<LogicalRootRewriteAction, 2> roots;
  SmallVector<LogicalWGMMAAction, 4> wgmmas;
  SmallVector<LogicalFragmentFoldAction, 4> folds;
  SmallVector<LogicalFragmentGuardAction, 4> guards;
  DenseMap<Value, MemDescLogicalState> memdescs;
  DenseMap<Value, TensorFragmentState> tensors;
  DenseMap<Value, TensorDescriptorLogicalState> descriptors;
  DenseMap<Operation *, LogicalDescriptorRewriteAction> descriptorRewrites;
};

FailureOr<SmallVector<int64_t, 2>>
selectLogicalSMEMStorageTileShape(gpu::MemDescType stageType,
                                  ArrayRef<int64_t> logicalShape);
SmallVector<int64_t, 2>
selectLogicalPointerCopyMicroTile(Operation *copy,
                                  ArrayRef<int64_t> storageTileShape,
                                  gpu::NVMMASharedEncodingAttr storageEncoding,
                                  Type elementType, unsigned vectorBytes = 16);
FailureOr<LogicalDomainPlan> analyzeLogicalDomains(ModuleOp module);
void applyLogicalDomainPlan(LogicalDomainPlan &&plan);

} // namespace mlir::triton::tle

#endif // TRITON_TLE_TRANSFORMS_LOGICAL_DOMAIN_H_
