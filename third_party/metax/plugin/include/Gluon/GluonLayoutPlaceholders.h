#ifndef TRITON_METAX_GLUON_LAYOUT_PLACEHOLDERS_H
#define TRITON_METAX_GLUON_LAYOUT_PLACEHOLDERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton {
class FuncOp;
}

namespace mlir::triton::gpu::metax::gluon {

Attribute unwrapNoVerifyEncoding(Attribute attr);
Type cloneTypeWithEncoding(Type type, Attribute encoding);

/// Recursively unwrap deferred encodings from SSA types and type-bearing
/// attributes. Returns the number of distinct wrapper attributes replaced.
unsigned unwrapNoVerifyEncodings(Operation *root);

/// Synchronize FunctionOpInterface's stored ABI after entry block arguments
/// and return-value layouts have had their deferred wrappers removed.
void synchronizeFunctionTypeAfterNoVerifyUnwrap(triton::FuncOp func);

bool hasAutoEncoding(Value value);
bool containsAutoEncoding(Type type);
bool containsNoVerifyEncoding(Attribute attr);
bool containsNoVerifyEncoding(Type type);

/// Layout analysis is intraprocedural, so every call must be inlined before
/// placeholder resolution or target layout construction.
LogicalResult verifyNoResidualCalls(ModuleOp module, StringRef boundary);

LogicalResult verifyNoResidualNoVerifyEncodings(ModuleOp module,
                                                StringRef boundary);
LogicalResult verifyNoResidualAutoEncodings(ModuleOp module,
                                            StringRef boundary);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_LAYOUT_PLACEHOLDERS_H
