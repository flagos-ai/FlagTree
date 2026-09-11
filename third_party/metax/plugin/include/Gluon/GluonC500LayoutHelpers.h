#ifndef TRITON_METAX_GLUON_C500_LAYOUT_HELPERS_H
#define TRITON_METAX_GLUON_C500_LAYOUT_HELPERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir::triton::gpu::metax::gluon::c500 {

Value getMemDescViewSource(Operation *op);
Value getMemDescRoot(Value value);
void collectMemDescAliasValues(Value root, llvm::SetVector<Value> &aliases);

unsigned getTensorElementOrPointeeBitWidth(RankedTensorType tensorTy);

enum class Rank2OrderProvenance {
  ConcreteEncoding,
  AxisInfo,
  ContiguityHint,
  PointerExpression,
};

struct Rank2ContiguousOrderInfo {
  SmallVector<unsigned, 2> order;
  /// Maximum number of adjacent elements proven in `order.front()` from
  /// address analysis. Zero means that only register ownership order is known
  /// and therefore cannot justify a vector global-memory transaction.
  unsigned maxContiguousElements = 0;
  Rank2OrderProvenance provenance;
};

StringRef stringifyRank2OrderProvenance(Rank2OrderProvenance provenance);

FailureOr<Rank2ContiguousOrderInfo>
inferRank2ContiguousOrder(Value value,
                          triton::ModuleAxisInfoAnalysis *axisInfoAnalysis,
                          Operation *diagnosticOp = nullptr);

std::optional<triton::gpu::BlockedEncodingAttr>
getC500GlobalBlockedEncoding(Value value, Operation *anchor,
                             ArrayRef<unsigned> order,
                             unsigned transactionWidthCap = 0);

std::optional<triton::gpu::BlockedEncodingAttr>
getC500GlobalBlockedEncodingLike(Value value, Operation *anchor,
                                 triton::gpu::BlockedEncodingAttr model);

std::optional<unsigned> getContiguityHint(Value value, unsigned dim);

/// Validates that subview materialization selects contiguous CTA/element
/// ranges. This is intentionally conservative for C500 async_copy because a
/// non-contiguous ttg.extract_tensor index sequence would describe a scattered
/// global/shared transfer even if the tensor encoding itself looks legal.
bool hasContiguousSubviewIndices(Value value);

struct AsyncCopyLegalizationInfo {
  unsigned contiguousDim = 0;
  unsigned currentVec = 0;
  unsigned legalVec = 0;
  unsigned elemBits = 0;

  bool needsSourceClip() const { return currentVec > legalVec; }
};

FailureOr<AsyncCopyLegalizationInfo> getC500AsyncCopyLegalizationInfo(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

bool isC500AsyncCopyContiguousSharedWrite(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

bool isC500AsyncCopyContiguousSharedWriteLegalizable(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

LogicalResult verifyC500AsyncCopyContiguousSharedWrite(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

LogicalResult verifyC500AsyncCopyContiguousSharedWriteLegalizable(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    Attribute sharedEncoding = {});

bool isUniformTensorValue(Value value);

} // namespace mlir::triton::gpu::metax::gluon::c500

#endif // TRITON_METAX_GLUON_C500_LAYOUT_HELPERS_H
