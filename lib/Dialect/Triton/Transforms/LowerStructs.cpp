//===- LowerStructs.cpp - Lower custom struct/array types -----------------===//
//
// Rewrites `tt.struct_gep` / `tt.array_gep` into byte-offset address arithmetic
// and rewrites pointer types whose pointee is a struct/array into opaque byte
// pointers, so that no custom aggregate type reaches a backend.
//
// See docs/custom-types/RFC.md.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONLOWERSTRUCTS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

// A pointer whose pointee is a struct or array (i.e. a custom aggregate type).
bool isAggregatePointer(Type type) {
  auto ptrTy = dyn_cast<PointerType>(type);
  return ptrTy && isa<StructType, ArrayType>(ptrTy.getPointeeType());
}

// Rewrites an aggregate pointer type to an opaque byte pointer, preserving the
// address space.
Type canonicalizeType(Type type, MLIRContext *ctx) {
  if (!isAggregatePointer(type))
    return type;
  auto ptrTy = cast<PointerType>(type);
  return PointerType::get(IntegerType::get(ctx, 8), ptrTy.getAddressSpace());
}

// struct_gep %p, "field" -> int_to_ptr(ptr_to_int(%p) + offsetof(field))
LogicalResult lowerStructGep(StructGepOp op, IRRewriter &rewriter) {
  auto ptrTy = dyn_cast<PointerType>(op.getPtr().getType());
  if (!ptrTy) {
    op.emitOpError("expected a pointer operand");
    return failure();
  }
  auto structTy = dyn_cast<StructType>(ptrTy.getPointeeType());
  if (!structTy) {
    op.emitOpError("expected a pointer to a struct");
    return failure();
  }
  int64_t index = structTy.getFieldIndex(op.getField());
  if (index < 0) {
    op.emitOpError() << "unknown struct field '" << op.getField() << "'";
    return failure();
  }
  int64_t offset = structTy.getFieldOffsets()[index];
  Type resultTy = op.getResult().getType();

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Type i64 = rewriter.getI64Type();
  Value ptrInt = rewriter.create<PtrToIntOp>(loc, i64, op.getPtr());
  Value offsetC = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(offset));
  Value sum = rewriter.create<arith::AddIOp>(loc, ptrInt, offsetC);
  Value res = rewriter.create<IntToPtrOp>(loc, resultTy, sum);
  rewriter.replaceOp(op, res);
  return success();
}

// array_gep %p, %i -> int_to_ptr(ptr_to_int(%p) + i * sizeof(element))
LogicalResult lowerArrayGep(ArrayGepOp op, IRRewriter &rewriter) {
  auto ptrTy = dyn_cast<PointerType>(op.getPtr().getType());
  if (!ptrTy) {
    op.emitOpError("expected a pointer operand");
    return failure();
  }
  auto arrTy = dyn_cast<ArrayType>(ptrTy.getPointeeType());
  if (!arrTy) {
    op.emitOpError("expected a pointer to an array");
    return failure();
  }
  auto sizeAlign = getTypeSizeAlign(arrTy.getElementType());
  if (!sizeAlign) {
    op.emitOpError("unsupported array element type");
    return failure();
  }
  int64_t elemSize = sizeAlign->first;
  Type resultTy = op.getResult().getType();

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Type i64 = rewriter.getI64Type();
  Value ptrInt = rewriter.create<PtrToIntOp>(loc, i64, op.getPtr());
  Value index = op.getIndex();
  if (index.getType() != i64)
    index = rewriter.create<arith::ExtSIOp>(loc, i64, index);
  Value sizeC = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(elemSize));
  Value byteOffset = rewriter.create<arith::MulIOp>(loc, index, sizeC);
  Value sum = rewriter.create<arith::AddIOp>(loc, ptrInt, byteOffset);
  Value res = rewriter.create<IntToPtrOp>(loc, resultTy, sum);
  rewriter.replaceOp(op, res);
  return success();
}

// After the address ops are lowered, the only remaining aggregate pointer types
// are function arguments and intermediate `int_to_ptr` results.  Rewrite them
// (and the function signatures) to opaque byte pointers.
void rewriteAggregatePointerTypes(ModuleOp mod, MLIRContext *ctx) {
  mod.walk([&](FunctionOpInterface func) {
    for (BlockArgument arg : func.getArguments())
      arg.setType(canonicalizeType(arg.getType(), ctx));

    auto fnTy = dyn_cast<FunctionType>(func.getFunctionType());
    if (!fnTy)
      return;
    SmallVector<Type> inputs, results;
    inputs.reserve(fnTy.getNumInputs());
    results.reserve(fnTy.getNumResults());
    for (Type t : fnTy.getInputs())
      inputs.push_back(canonicalizeType(t, ctx));
    for (Type t : fnTy.getResults())
      results.push_back(canonicalizeType(t, ctx));
    func.setFunctionType(FunctionType::get(ctx, inputs, results));
  });

  mod.walk([&](Operation *op) {
    for (Value res : op->getResults())
      if (isAggregatePointer(res.getType()))
        res.setType(canonicalizeType(res.getType(), ctx));
  });
}

struct LowerStructs : public triton::impl::TritonLowerStructsBase<LowerStructs> {
  using TritonLowerStructsBase::TritonLowerStructsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // 1. Lower the address ops to byte arithmetic.
    SmallVector<Operation *> geps;
    mod.walk([&](Operation *op) {
      if (isa<StructGepOp, ArrayGepOp>(op))
        geps.push_back(op);
    });
    for (Operation *op : geps) {
      LogicalResult result = success();
      if (auto sGep = dyn_cast<StructGepOp>(op))
        result = lowerStructGep(sGep, rewriter);
      else if (auto aGep = dyn_cast<ArrayGepOp>(op))
        result = lowerArrayGep(aGep, rewriter);
      if (failed(result))
        return signalPassFailure();
    }

    // 2. Eliminate the aggregate pointer types.
    rewriteAggregatePointerTypes(mod, ctx);

    // 3. Verify that no aggregate pointer type remains.
    WalkResult walkResult = mod.walk([&](Operation *op) -> WalkResult {
      auto check = [&](Value v) -> bool {
        return v && isAggregatePointer(v.getType());
      };
      for (Value v : op->getOperands())
        if (check(v))
          return op->emitOpError(
              "aggregate pointer type remained after lowering");
      for (Value v : op->getResults())
        if (check(v))
          return op->emitOpError(
              "aggregate pointer type remained after lowering");
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
