// NOTE: hcu and mthreads keep their own copy of the TTIR transforms, so this
// file mirrors lib/Dialect/Triton/Transforms/LowerStructs.cpp.  Keep the two in
// sync (the pass body is backend-agnostic).
//===- LowerStructs.cpp - Lower custom struct/array types -----------------===//
//
// Rewrites `tt.struct_gep` / `tt.array_gep` into byte-offset address arithmetic
// and rewrites pointer types whose pointee is a struct/array into opaque byte
// pointers, so that no custom aggregate type reaches a backend.
//
// Custom aggregates are accepted in **global memory only**: the host builds the
// bytes and the kernel reaches the fields through an absolute address, which
// only exists for global memory.  In any other address space that address is
// meaningless, so such pointers are rejected instead of silently addressing
// global memory.
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

// A scalar pointer whose pointee is a 64-bit integer (the byte-address carrier
// used for pointer-typed fields).
bool isI64Pointer(Type type) {
  auto ptrTy = dyn_cast<PointerType>(type);
  if (!ptrTy)
    return false;
  auto intTy = dyn_cast<IntegerType>(ptrTy.getPointeeType());
  return intTy && intTy.getWidth() == 64;
}

// A field whose value is itself a pointer (`ptr<ptr<X>>`) cannot be represented
// as pointer-typed memory on some backends (a memref element type may not be a
// pointer).  Represent its storage as `ptr<i64>` instead; lowerPointerMemory
// then restores the pointer with an int round-trip.
Type fieldStorageType(Type resultTy, MLIRContext *ctx) {
  auto ptrTy = dyn_cast<PointerType>(resultTy);
  if (ptrTy && isa<PointerType>(ptrTy.getPointeeType()))
    return PointerType::get(IntegerType::get(ctx, 64), ptrTy.getAddressSpace());
  return resultTy;
}

// `tt.ptr`'s default address space; the only one where a host-provided byte
// address is meaningful.
constexpr int kGlobalAddressSpace = 1;

// Diagnoses an aggregate pointer that does not live in global memory.  The
// lowering below reaches fields through `ptr_to_int`/`int_to_ptr`, i.e. through
// an absolute address; for a non-global address space that would quietly become
// a global access, so reject it up front instead of miscompiling.
LogicalResult checkGlobalAddressSpace(Operation *op, Value ptr) {
  auto ptrTy = dyn_cast<PointerType>(ptr.getType());
  if (!ptrTy)
    return success();
  int addressSpace = ptrTy.getAddressSpace();
  if (addressSpace == kGlobalAddressSpace)
    return success();
  return op->emitOpError() << "custom struct/array types are supported in "
                              "global memory only, but "
                              "this pointer is in address space "
                           << addressSpace;
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
  Type resultTy =
      fieldStorageType(op.getResult().getType(), rewriter.getContext());

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
  Type resultTy =
      fieldStorageType(op.getResult().getType(), rewriter.getContext());

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

// Rewrites loads/stores whose *value* is a pointer into integer loads/stores
// plus int_to_ptr/ptr_to_int.  The gep lowering represents a pointer field's
// storage as `ptr<i64>` (see fieldStorageType); this restores the pointer.
void lowerPointerMemory(ModuleOp mod, IRRewriter &rewriter) {
  SmallVector<Operation *> worklist;
  mod.walk([&](Operation *op) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (!load.getMask() && isa<PointerType>(load.getResult().getType()) &&
          isI64Pointer(load.getPtr().getType()))
        worklist.push_back(op);
    } else if (auto store = dyn_cast<StoreOp>(op)) {
      if (!store.getMask() && isa<PointerType>(store.getValue().getType()) &&
          isI64Pointer(store.getPtr().getType()))
        worklist.push_back(op);
    }
  });
  for (Operation *op : worklist) {
    rewriter.setInsertionPoint(op);
    Location loc = op->getLoc();
    if (auto load = dyn_cast<LoadOp>(op)) {
      Value raw =
          rewriter.create<LoadOp>(loc, load.getPtr(), load.getCache(),
                                  load.getEvict(), load.getIsVolatile());
      Value res =
          rewriter.create<IntToPtrOp>(loc, load.getResult().getType(), raw);
      rewriter.replaceOp(load, res);
    } else if (auto store = dyn_cast<StoreOp>(op)) {
      Value raw = rewriter.create<PtrToIntOp>(loc, rewriter.getI64Type(),
                                              store.getValue());
      rewriter.create<StoreOp>(loc, store.getPtr(), raw, store.getCache(),
                               store.getEvict());
      rewriter.eraseOp(store);
    }
  }
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
    function_interface_impl::setFunctionType(
        func, FunctionType::get(ctx, inputs, results));
  });

  mod.walk([&](Operation *op) {
    for (Value res : op->getResults())
      if (isAggregatePointer(res.getType()))
        res.setType(canonicalizeType(res.getType(), ctx));
  });
}

// Removes the redundant pointer<->int round-trips that chained field/element
// access can create during lowering:
//   ptr_to_int(int_to_ptr(x)) -> x
//   int_to_ptr(ptr_to_int(x)) -> x   (when the pointer types agree)
void foldPointerRoundTrips(ModuleOp mod) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    mod.walk([&](Operation *op) {
      if (auto p2i = dyn_cast<PtrToIntOp>(op)) {
        if (auto i2p = p2i.getSrc().getDefiningOp<IntToPtrOp>()) {
          p2i.getResult().replaceAllUsesWith(i2p.getSrc());
          dead.push_back(op);
          changed = true;
        }
      } else if (auto i2p = dyn_cast<IntToPtrOp>(op)) {
        if (auto p2i = i2p.getSrc().getDefiningOp<PtrToIntOp>()) {
          if (p2i.getSrc().getType() == i2p.getType()) {
            i2p.getResult().replaceAllUsesWith(p2i.getSrc());
            dead.push_back(op);
            changed = true;
          }
        }
      }
    });
    for (Operation *op : dead)
      op->erase();
  }
}

struct LowerStructs
    : public triton::impl::TritonLowerStructsBase<LowerStructs> {
  using TritonLowerStructsBase::TritonLowerStructsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // 0. Custom aggregates are global-memory only; reject anything else before
    //    rewriting, so the diagnostic points at the original operation.
    bool badAddressSpace = false;
    mod.walk([&](Operation *op) {
      Value ptr;
      if (auto sGep = dyn_cast<StructGepOp>(op))
        ptr = sGep.getPtr();
      else if (auto aGep = dyn_cast<ArrayGepOp>(op))
        ptr = aGep.getPtr();
      else
        return;
      if (failed(checkGlobalAddressSpace(op, ptr)))
        badAddressSpace = true;
    });
    if (badAddressSpace)
      return signalPassFailure();

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

    // 2. Loads/stores of pointer-typed values become int loads/stores.
    lowerPointerMemory(mod, rewriter);

    // 3. Eliminate the aggregate pointer types.
    rewriteAggregatePointerTypes(mod, ctx);

    // 4. Fold the redundant pointer<->int round-trips created by chained
    //    field/element access.
    foldPointerRoundTrips(mod);

    // 5. Verify that no aggregate pointer type remains.
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
