#ifndef TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
#define TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"

namespace mlir::triton::gluon {

/// Returns true when the operation declares that all ranked-tensor operands
/// and results share one encoding. SameOperandsAndResultEncoding is an ODS
/// contract; Elementwise is the existing Triton layout convention used by
/// dialect inference.
bool hasSameTensorEncodingRelation(Operation *op);

/// Returns true for the load/store ODS variants of the same-encoding contract.
/// These traits deliberately permit tensor-pointer operands.
bool hasSameLoadStoreTensorEncodingRelation(Operation *op);

LogicalResult
inferLayout(FuncOp func, llvm::function_ref<bool(Type)> typeCheck,
            const SmallVector<std::pair<Value, Attribute>> &seedEncodings);

LogicalResult doubleCheckEncodings(ModuleOp &mod,
                                   llvm::function_ref<bool(Type)> typeCheck);

} // namespace mlir::triton::gluon

#endif // TRITON_DIALECT_GLUON_TRANSFORMS_INFERLAYOUTUTILS_H_
