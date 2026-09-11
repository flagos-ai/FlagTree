#ifndef TRITON_METAX_GLUON_C500_LAYOUT_H
#define TRITON_METAX_GLUON_C500_LAYOUT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu::metax::gluon {

bool supportsC500MacaAccumulatorOrder(Type lhsElementType,
                                     Type rhsElementType, unsigned colMajor);

enum class LayoutContractRole {
  DotOperand,
  RegisterToSharedSink,
  RegisterSubview,
};

struct RegisterToSharedContractInfo {
  bool lowerable = false;
  bool completeCoverage = false;
  unsigned freeWarpMask = 0;
};

/// Canonicalize MACA accumulator encodings whose transpose flags describe
/// operand-loading policy rather than accumulator ownership.
unsigned normalizeMacaAccumulatorEncodings(Operation *root);
LogicalResult verifyMacaEncodingContracts(ModuleOp module,
                                          StringRef boundary);

FailureOr<triton::LinearLayout>
getC500LinearLayout(RankedTensorType tensorType);
FailureOr<triton::LinearLayout>
getC500LinearLayout(triton::gpu::MemDescType memDescType);

bool isSimpleDotLocalLoad(triton::gpu::LocalLoadOp loadOp);

FailureOr<RegisterToSharedContractInfo>
checkRegisterToSharedContract(RankedTensorType registerType,
                              triton::gpu::MemDescType memDescType);
LogicalResult verifyRegisterToSharedContract(
    Operation *op, RankedTensorType registerType,
    triton::gpu::MemDescType memDescType, LayoutContractRole role);

LogicalResult checkBsmPermPhysicalContract(RankedTensorType sourceType,
                                           RankedTensorType resultType);
LogicalResult verifyBsmPermPhysicalContract(triton::gpu::BsmPermOp op);
LogicalResult checkAtomicRmwOwnership(RankedTensorType valueType);
LogicalResult verifyAtomicRmwOwnership(triton::AtomicRMWOp atomicOp);

LogicalResult
verifyExtractTensorContract(triton::gpu::ExtractTensorOp extractOp);
LogicalResult
verifyInsertTensorContract(triton::gpu::InsertTensorOp insertOp);

namespace c500 {

struct C500RegisterSlicePlan {
  SmallVector<int64_t> ctaIndices;
  SmallVector<int64_t> elementIndices;
  SmallVector<unsigned> sourceElementSlots;
};

struct RegisterSliceMaterializationPlan {
  RankedTensorType physicalFullType;
  RankedTensorType physicalSubType;
  C500RegisterSlicePlan physicalSlice;
};

FailureOr<C500RegisterSlicePlan>
planC500RegisterSlice(RankedTensorType fullType, RankedTensorType subType,
                      ArrayRef<int64_t> offsets);
FailureOr<RankedTensorType> inferC500RegisterSlicePhysicalType(
    RankedTensorType fullType, const C500RegisterSlicePlan &plan, Location loc);
FailureOr<RegisterSliceMaterializationPlan>
planC500RegisterSliceMaterialization(RankedTensorType fullType,
                                     RankedTensorType logicalSubType,
                                     ArrayRef<int64_t> offsets,
                                     Operation *anchor);

struct LayoutTransferRequirement {
  RankedTensorType sourceType;
  RankedTensorType resultType;
  bool isTerminalGlobalStore = false;
  bool isOutsideLoop = false;
};

enum class LayoutTransferImplementation { Generic, RepeatedShared };

struct LayoutTransferMetrics {
  unsigned scratchElements = 0;
  unsigned repetitions = 0;
};

struct LayoutTransferPlan {
  LayoutTransferImplementation implementation =
      LayoutTransferImplementation::Generic;
  LayoutTransferMetrics genericMetrics;
  LayoutTransferMetrics selectedMetrics;
};

FailureOr<LayoutTransferPlan>
planLayoutTransfer(const LayoutTransferRequirement &requirement,
                   Operation *anchor);

} // namespace c500
} // namespace mlir::triton::gpu::metax::gluon

#endif
