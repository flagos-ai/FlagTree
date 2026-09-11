#ifndef TRITON_METAX_GLUON_C500_ASYNC_COPY_LAYOUT_H
#define TRITON_METAX_GLUON_C500_ASYNC_COPY_LAYOUT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::metax::gluon::c500 {

/// Address-space evidence for one rank-2 global transaction. This fact is
/// independent of register ownership: a concrete BlockedEncodingAttr may
/// determine which thread owns an element, but cannot prove that neighboring
/// pointer values address neighboring global elements.
struct AsyncCopyAddressContiguity {
  unsigned contiguousDim = 0;
  unsigned maxContiguousElements = 0;

  bool isProven() const { return maxContiguousElements != 0; }
};

LogicalResult materializeAsyncCopyFollowerLayouts(triton::FuncOp func);

LogicalResult verifyAsyncCopyLayouts(triton::FuncOp func);

} // namespace mlir::triton::gpu::metax::gluon::c500

#endif // TRITON_METAX_GLUON_C500_ASYNC_COPY_LAYOUT_H
