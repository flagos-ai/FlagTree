//===- ConvertTritonToAscend.cpp - Convert Triton IR to AscendModel -------===//
//
// This file implements the conversion from Triton IR to AscendModel dialect
// for Ascend 910B performance modeling.
//
// Key design: Triton's pointer arithmetic chain (tt.splat -> tt.make_range ->
// tt.addptr -> tt.load/store) is folded into a single ascend.vector_load or
// ascend.vector_store. The address computation ops have zero hardware cost
// on Ascend -- they are absorbed into the DMA descriptor.
//
// When built with Triton support (TRITONSIM_HAS_TRITON), this pass uses
// typed op matching. Without Triton, string-based matching is used.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/HardwareParams.h"
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Triton typed op headers (optional)
#ifdef TRITONSIM_HAS_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_CONVERTTRITONTOASCENDPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static int64_t getByteSize(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return -1;
      count *= dim;
    }
    Type elemType = tensorType.getElementType();
    int64_t elemSize = (elemType.getIntOrFloatBitWidth() + 7) / 8;
    return count * elemSize;
  }
  if (type.isIntOrFloat()) {
    return (type.getIntOrFloatBitWidth() + 7) / 8;
  }
  return 0;
}

/// Check if an op matches a given dialect.op name string.
static bool isOpNamed(Operation *op, StringRef name) {
  return op->getName().getStringRef() == name;
}

/// Walk back through the Triton pointer arithmetic chain:
///   tt.load <- tt.addptr <- tt.splat <- %blockArg
/// Returns the ultimate source Value (typically a function argument).
/// Used for provenance tracking; the exact address is irrelevant for
/// performance modeling.
static Value tracePointerSource(Value ptr) {
  Value current = ptr;
  while (Operation *def = current.getDefiningOp()) {
    StringRef name = def->getName().getStringRef();
    if (name == "tt.addptr" || name == "tt.splat" || name == "tt.bitcast" ||
        name == "tt.int_to_ptr") {
      current = def->getOperand(0);
    } else {
      break;
    }
  }
  return current;
}

/// Create a placeholder tensor source for ascend load ops.
/// Triton pointer types (tensor<N x !tt.ptr<f32>>) are not valid
/// AscendModel tensor inputs, so we create a tensor.empty with the
/// correct shape to represent "loading N bytes from memory".
static Value createPlaceholderSource(Type resultType, Location loc,
                                     PatternRewriter &rewriter) {
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    return rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                            tensorType.getElementType());
  }
  return Value();
}

//===----------------------------------------------------------------------===//
// Triton-Specific Conversion Patterns
//===----------------------------------------------------------------------===//

/// Helper: check if a value comes from a cube operation (dot/matmul or
/// cube_load)
static bool isFromCubeOp(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  StringRef name = def->getName().getStringRef();
  return name == "tt.dot" || name == "ascend.matmul" ||
         name == "ascend.cube_load";
}

/// Helper: check if a value comes from a load (tt.load or ascend.*_load)
static bool isFromLoad(Value v) {
  // Trace through tt.trans etc
  Value current = v;
  while (Operation *def = current.getDefiningOp()) {
    StringRef name = def->getName().getStringRef();
    if (name == "tt.load" || name == "ascend.cube_load" ||
        name == "ascend.vector_load") {
      return true;
    }
    if (name == "tt.trans" || name == "tt.bitcast" || name == "tt.reshape" ||
        name == "tt.expand_dims") {
      current = def->getOperand(0);
    } else {
      break;
    }
  }
  return false;
}

/// Convert tt.dot -> ascend.matmul with proper data transfer ops
///
/// Data flow rules:
/// - If input comes from tt.load: already converted to cube_load by
/// ConvertTritonLoad
/// - If input comes from vector ops: insert vector_store + cube_load
/// - Output always goes to cube_store + vector_load (for subsequent vector ops)
struct ConvertTritonDot : public RewritePattern {
  ConvertTritonDot(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
#ifdef TRITONSIM_HAS_TRITON
    auto dotOp = dyn_cast<triton::DotOp>(op);
    if (!dotOp)
      return failure();
    Value lhs = dotOp.getA();
    Value rhs = dotOp.getB();
    Value acc = dotOp.getC();
#else
    if (!isOpNamed(op, "tt.dot"))
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value acc = op->getNumOperands() > 2 ? op->getOperand(2) : Value();
#endif

    Location loc = op->getLoc();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2)
      return failure();

    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t N = rhsType.getShape()[1];

    auto resultType = op->getResult(0).getType();

    // Helper lambda: ensure operand is in Cube memory (cube_load output)
    // If it comes from vector ops, insert vector_store + cube_load
    auto ensureCubeInput = [&](Value operand) -> Value {
      // If from load (already handled by ConvertTritonLoad), use as-is
      if (isFromLoad(operand)) {
        return operand;
      }

      // If from vector operations, need to transfer to cube
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType)
        return operand;

      int64_t bytes = getByteSize(tensorType);

      // Vector store (write from vector core to HBM via MTE3)
      rewriter.create<VectorStoreOp>(
          loc, operand, bytes, rewriter.getStringAttr("ub"),
          rewriter.getStringAttr("hbm"), nullptr, nullptr);

      // Cube load (read into cube core L1 via MTE2)
      Value placeholder = createPlaceholderSource(tensorType, loc, rewriter);
      auto cubeLoad = rewriter.create<CubeLoadOp>(
          loc, tensorType, placeholder ? placeholder : operand, bytes,
          rewriter.getStringAttr("hbm"), rewriter.getStringAttr("l1"), nullptr,
          nullptr);

      return cubeLoad.getResult();
    };

    // Prepare inputs - ensure they're in cube memory
    Value lhsInput = ensureCubeInput(lhs);
    Value rhsInput = ensureCubeInput(rhs);

    // Handle accumulator if present and non-zero
    Value accInput = acc;
    if (acc) {
      // Check if acc is a non-zero constant
      bool isZeroAcc = false;
      if (auto defOp = acc.getDefiningOp()) {
        if (defOp->getName().getStringRef() == "arith.constant") {
          if (auto attr = defOp->getAttr("value")) {
            if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
              isZeroAcc = denseAttr.isSplat() &&
                          denseAttr.getSplatValue<APFloat>().isZero();
            }
          }
        }
      }

      // If non-zero accumulator from vector ops, transfer to cube
      if (!isZeroAcc && !isFromLoad(acc) && !isFromCubeOp(acc)) {
        accInput = ensureCubeInput(acc);
      }
    }

    // Create matmul
    auto matmul = rewriter.create<MatmulOp>(loc, resultType, lhsInput, rhsInput,
                                            M, N, K, nullptr, nullptr);

    // Output: cube_store (write from cube L0C to HBM via FixPipe)
    int64_t resultBytes = getByteSize(resultType);
    rewriter.create<CubeStoreOp>(
        loc, matmul.getResult(), resultBytes, rewriter.getStringAttr("l0c"),
        rewriter.getStringAttr("hbm"), nullptr, nullptr);

    // Check if result is used by vector operations (or loop yield)
    // If so, insert vector_load to bring data into vector core
    bool usedByVectorOps = false;
    for (Operation *user : op->getResult(0).getUsers()) {
      StringRef userName = user->getName().getStringRef();
      // Only skip if used exclusively by another dot/matmul
      if (userName != "tt.dot" && userName != "ascend.matmul") {
        usedByVectorOps = true;
        break;
      }
    }

    if (usedByVectorOps) {
      // Vector load (read from HBM into vector core UB via MTE2)
      auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
      Value placeholder = createPlaceholderSource(resultType, loc, rewriter);
      auto vecLoad = rewriter.create<VectorLoadOp>(
          loc, resultType, placeholder ? placeholder : matmul.getResult(),
          resultBytes, rewriter.getStringAttr("hbm"),
          rewriter.getStringAttr("ub"), nullptr, nullptr);
      rewriter.replaceOp(op, vecLoad.getResult());
    } else {
      rewriter.replaceOp(op, matmul.getResult());
    }

    return success();
  }
};

/// Convert tt.load -> ascend.cube_load or ascend.vector_load
///
/// If the load has "ascend.used_by_dot" attribute (set in analysis phase),
/// it becomes a cube_load. Otherwise, it becomes a vector_load.
struct ConvertTritonLoad : public RewritePattern {
  ConvertTritonLoad(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
#ifdef TRITONSIM_HAS_TRITON
    if (!isa<triton::LoadOp>(op))
      return failure();
#else
    if (!isOpNamed(op, "tt.load"))
      return failure();
#endif

    if (op->getNumResults() == 0)
      return failure();

    Location loc = op->getLoc();
    auto resultType = op->getResult(0).getType();
    int64_t bytes = getByteSize(resultType);

    // Create a placeholder tensor source
    Value source = createPlaceholderSource(resultType, loc, rewriter);
    if (!source) {
      source = op->getOperand(0);
    }

    // Check if this load is marked as used by dot (from analysis phase)
    bool usedByDot = op->hasAttr("ascend.used_by_dot");

    if (usedByDot) {
      auto cubeLoad = rewriter.create<CubeLoadOp>(
          loc, resultType, source, bytes, rewriter.getStringAttr("hbm"),
          rewriter.getStringAttr("l1"), nullptr, nullptr);
      rewriter.replaceOp(op, cubeLoad.getResult());
    } else {
      auto vecLoad = rewriter.create<VectorLoadOp>(
          loc, resultType, source, bytes, rewriter.getStringAttr("hbm"),
          rewriter.getStringAttr("ub"), nullptr, nullptr);
      rewriter.replaceOp(op, vecLoad.getResult());
    }
    return success();
  }
};

/// Convert tt.trans -> direct replacement (zero cost in perf modeling)
///
/// Transpose is a logical reshape that doesn't incur hardware cost
/// on Ascend (handled by address calculation in hardware).
/// We replace it directly with its input for modeling purposes.
struct ConvertTritonTrans : public RewritePattern {
  ConvertTritonTrans(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isOpNamed(op, "tt.trans"))
      return failure();

    if (op->getNumOperands() == 0 || op->getNumResults() == 0)
      return failure();

    // tt.trans just reorders the tensor - for perf modeling we pass through
    // The actual memory access pattern is handled by the load/store
    Value input = op->getOperand(0);
    rewriter.replaceOp(op, input);
    return success();
  }
};

/// Convert tt.store -> ascend.vector_store
///
/// The data operand (operand 1) is already a proper tensor -- used
/// directly. The pointer operand (operand 0, from tt.addptr) is discarded.
struct ConvertTritonStore : public RewritePattern {
  ConvertTritonStore(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
#ifdef TRITONSIM_HAS_TRITON
    if (!isa<triton::StoreOp>(op))
      return failure();
#else
    if (!isOpNamed(op, "tt.store"))
      return failure();
#endif

    if (op->getNumOperands() < 2)
      return failure();

    // operand(0) = pointer (from addptr chain, discarded)
    // operand(1) = data tensor (used directly)
    Value data = op->getOperand(1);
    int64_t bytes = getByteSize(data.getType());

    rewriter.create<VectorStoreOp>(
        op->getLoc(), data, bytes, rewriter.getStringAttr("ub"),
        rewriter.getStringAttr("hbm"), nullptr, nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Erase Triton address-computation ops that become dead after load/store
/// conversion. These ops have zero hardware cost:
///   tt.addptr, tt.splat, tt.make_range, tt.expand_dims,
///   tt.get_program_id, tt.int_to_ptr, tt.ptr_to_int, tt.bitcast, tt.trans
///
/// The pattern runs at low benefit so load/store conversions happen first.
/// Greedy rewriting iterates until convergence, so the cascade works:
///   tt.load erased -> tt.addptr dead -> tt.splat dead -> tt.make_range dead
struct EraseDeadTritonAddrOps : public RewritePattern {
  EraseDeadTritonAddrOps(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only match Triton dialect ops
    if (!op->getName().getStringRef().starts_with("tt."))
      return failure();

    StringRef name = op->getName().getStringRef();
    bool isAddrOrNoCost =
        (name == "tt.addptr" || name == "tt.splat" || name == "tt.make_range" ||
         name == "tt.expand_dims" || name == "tt.get_program_id" ||
         name == "tt.int_to_ptr" || name == "tt.ptr_to_int" ||
         name == "tt.bitcast" || name == "tt.trans" || name == "tt.reshape");

    if (!isAddrOrNoCost)
      return failure();

    // Only erase if all results are unused
    if (!op->use_empty())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

// NOTE: tt.func / tt.return are intentionally NOT converted.
// They are Triton's function shell. func.return requires func.func as parent,
// but the parent here is tt.func. For perf modeling we only care about the
// compute ops inside the function body; the shell is harmless and all
// downstream passes walk ModuleOp, not func.func specifically.

struct ConvertTritonReduce : public RewritePattern {
  ConvertTritonReduce(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
#ifdef TRITONSIM_HAS_TRITON
    auto reduceOp = dyn_cast<triton::ReduceOp>(op);
    if (!reduceOp)
      return failure();
    int64_t axis = reduceOp.getAxis();
#else
    if (!isOpNamed(op, "tt.reduce"))
      return failure();
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!axisAttr)
      return failure();
    int64_t axis = axisAttr.getInt();
#endif

    Location loc = op->getLoc();
    auto resultType = op->getResult(0).getType();

    // Determine reduction kind from the body
    bool isMax = false, isMin = false, isProd = false;
    if (!op->getRegions().empty()) {
      Region &region = op->getRegion(0);
      if (!region.empty()) {
        Block &block = region.front();
        for (Operation &innerOp : block) {
          StringRef innerName = innerOp.getName().getStringRef();
          if (innerName.contains("max"))
            isMax = true;
          else if (innerName.contains("min"))
            isMin = true;
          else if (innerName.contains("mul"))
            isProd = true;
        }
      }
    }

    Operation *newReduceOp;
    if (isMax) {
      newReduceOp = rewriter.create<ReduceMaxOp>(
          loc, resultType, op->getOperand(0), axis, nullptr, nullptr);
    } else if (isMin) {
      newReduceOp = rewriter.create<ReduceMinOp>(
          loc, resultType, op->getOperand(0), axis, nullptr, nullptr);
    } else if (isProd) {
      newReduceOp = rewriter.create<ReduceProdOp>(
          loc, resultType, op->getOperand(0), axis, nullptr, nullptr);
    } else {
      newReduceOp = rewriter.create<ReduceSumOp>(
          loc, resultType, op->getOperand(0), axis, nullptr, nullptr);
    }

    rewriter.replaceOp(op, newReduceOp->getResult(0));
    return success();
  }
};

struct ConvertTritonBroadcast : public RewritePattern {
  ConvertTritonBroadcast(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
#ifdef TRITONSIM_HAS_TRITON
    if (!isa<triton::BroadcastOp>(op))
      return failure();
#else
    if (!isOpNamed(op, "tt.broadcast"))
      return failure();
#endif

    auto resultType = op->getResult(0).getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType)
      return failure();

    SmallVector<int64_t> shape(tensorType.getShape());
    auto broadcastOp = rewriter.create<BroadcastOp>(
        op->getLoc(), resultType, op->getOperand(0),
        rewriter.getDenseI64ArrayAttr(shape), nullptr, nullptr);

    rewriter.replaceOp(op, broadcastOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arith / Math Conversion Patterns
//===----------------------------------------------------------------------===//

struct ConvertArithBinaryOp : public RewritePattern {
  ConvertArithBinaryOp(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();
    if (op->getNumResults() == 0)
      return failure();

    auto resultType = op->getResult(0).getType();
    if (!isa<RankedTensorType>(resultType))
      return failure();
    if (op->getNumOperands() < 2)
      return failure();

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();
    Operation *newOp = nullptr;

    if (name == "arith.addf" || name == "arith.addi") {
      newOp =
          rewriter.create<AddOp>(loc, resultType, lhs, rhs, nullptr, nullptr);
    } else if (name == "arith.subf" || name == "arith.subi") {
      newOp =
          rewriter.create<SubOp>(loc, resultType, lhs, rhs, nullptr, nullptr);
    } else if (name == "arith.mulf" || name == "arith.muli") {
      newOp =
          rewriter.create<MulOp>(loc, resultType, lhs, rhs, nullptr, nullptr);
    } else if (name == "arith.divf" || name == "arith.divsi" ||
               name == "arith.divui") {
      newOp =
          rewriter.create<DivOp>(loc, resultType, lhs, rhs, nullptr, nullptr);
    } else if (name == "arith.maxnumf" || name == "arith.maximumf" ||
               name == "arith.maxsi" || name == "arith.maxui") {
      newOp =
          rewriter.create<MaxOp>(loc, resultType, lhs, rhs, nullptr, nullptr);
    } else if (name == "arith.minnumf" || name == "arith.minimumf" ||
               name == "arith.minsi" || name == "arith.minui") {
      newOp =
          rewriter.create<MinOp>(loc, resultType, lhs, rhs, nullptr, nullptr);
    } else {
      return failure();
    }

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ConvertArithCmpOp : public RewritePattern {
  ConvertArithCmpOp(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();

    // Match arith.cmpf and arith.cmpi
    if (!name.starts_with("arith.cmpf") && !name.starts_with("arith.cmpi"))
      return failure();

    if (op->getNumResults() == 0 || op->getNumOperands() < 2)
      return failure();

    auto resultType = op->getResult(0).getType();
    if (!isa<RankedTensorType>(resultType))
      return failure();

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Location loc = op->getLoc();

    // Get the predicate attribute
    auto predicateAttr = op->getAttrOfType<mlir::IntegerAttr>("predicate");
    if (!predicateAttr)
      return failure();

    int64_t predicate = predicateAttr.getInt();
    Operation *newOp = nullptr;

    if (name == "arith.cmpf") {
      // Float comparison predicates (arith::CmpFPredicate)
      // 1=OEQ, 2=OGT, 3=OGE, 4=OLT, 5=OLE, 6=ONE, 8=UEQ, 9=UGT, 10=UGE, 11=ULT,
      // 12=ULE, 13=UNE
      switch (predicate) {
      case 1: // OEQ - ordered equal
      case 8: // UEQ - unordered equal
        newOp = rewriter.create<CmpEqOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 6:  // ONE - ordered not equal
      case 13: // UNE - unordered not equal
        newOp = rewriter.create<CmpNeOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 4:  // OLT - ordered less than
      case 11: // ULT - unordered less than
        newOp = rewriter.create<CmpLtOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 5:  // OLE - ordered less or equal
      case 12: // ULE - unordered less or equal
        newOp = rewriter.create<CmpLeOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 2: // OGT - ordered greater than
      case 9: // UGT - unordered greater than
        newOp = rewriter.create<CmpGtOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 3:  // OGE - ordered greater or equal
      case 10: // UGE - unordered greater or equal
        newOp = rewriter.create<CmpGeOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      default:
        return failure();
      }
    } else if (name == "arith.cmpi") {
      // Integer comparison predicates (arith::CmpIPredicate)
      // 0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
      switch (predicate) {
      case 0: // eq
        newOp = rewriter.create<CmpEqOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 1: // ne
        newOp = rewriter.create<CmpNeOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 2: // slt
      case 6: // ult
        newOp = rewriter.create<CmpLtOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 3: // sle
      case 7: // ule
        newOp = rewriter.create<CmpLeOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 4: // sgt
      case 8: // ugt
        newOp = rewriter.create<CmpGtOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      case 5: // sge
      case 9: // uge
        newOp = rewriter.create<CmpGeOp>(loc, resultType, lhs, rhs, nullptr,
                                         nullptr);
        break;
      default:
        return failure();
      }
    } else {
      return failure();
    }

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ConvertMathUnaryOp : public RewritePattern {
  ConvertMathUnaryOp(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();
    if (op->getNumResults() == 0 || op->getNumOperands() == 0)
      return failure();

    auto resultType = op->getResult(0).getType();
    if (!isa<RankedTensorType>(resultType))
      return failure();

    Value input = op->getOperand(0);
    Location loc = op->getLoc();
    Operation *newOp = nullptr;

    if (name == "math.exp" || name == "math.exp2") {
      newOp = rewriter.create<ExpOp>(loc, resultType, input, nullptr, nullptr);
    } else if (name == "math.log" || name == "math.log2") {
      newOp = rewriter.create<LogOp>(loc, resultType, input, nullptr, nullptr);
    } else if (name == "math.sqrt") {
      newOp = rewriter.create<SqrtOp>(loc, resultType, input, nullptr, nullptr);
    } else if (name == "math.rsqrt") {
      newOp =
          rewriter.create<RsqrtOp>(loc, resultType, input, nullptr, nullptr);
    } else if (name == "math.tanh") {
      newOp = rewriter.create<TanhOp>(loc, resultType, input, nullptr, nullptr);
    } else if (name == "arith.negf") {
      newOp = rewriter.create<NegOp>(loc, resultType, input, nullptr, nullptr);
    } else if (name == "math.absf" || name == "math.absi") {
      newOp = rewriter.create<AbsOp>(loc, resultType, input, nullptr, nullptr);
    } else {
      return failure();
    }

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ConvertArithSelect : public RewritePattern {
  ConvertArithSelect(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isOpNamed(op, "arith.select"))
      return failure();
    if (op->getNumOperands() != 3 || op->getNumResults() == 0)
      return failure();

    auto resultType = op->getResult(0).getType();
    if (!isa<RankedTensorType>(resultType))
      return failure();

    auto selectOp = rewriter.create<SelectOp>(
        op->getLoc(), resultType, op->getOperand(0), op->getOperand(1),
        op->getOperand(2), nullptr, nullptr);

    rewriter.replaceOp(op, selectOp.getResult());
    return success();
  }
};

struct ConvertArithCast : public RewritePattern {
  ConvertArithCast(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef name = op->getName().getStringRef();

    // Check if it's a type conversion operation
    // All cast operations (1 cycle each) are unified to ascend.cast
    if (!name.starts_with("arith.ext") && !name.starts_with("arith.trunc") &&
        !name.starts_with("arith.sitofp") &&
        !name.starts_with("arith.uitofp") &&
        !name.starts_with("arith.fptosi") &&
        !name.starts_with("arith.fptoui") &&
        !name.starts_with("arith.bitcast") &&
        !name.starts_with("arith.index_cast"))
      return failure();

    if (op->getNumResults() == 0 || op->getNumOperands() == 0)
      return failure();

    auto resultType = op->getResult(0).getType();
    if (!isa<RankedTensorType>(resultType))
      return failure();

    // All cast operations map to ascend.cast (1 cycle per vector op)
    auto castOp = rewriter.create<CastOp>(op->getLoc(), resultType,
                                          op->getOperand(0), nullptr, nullptr);

    rewriter.replaceOp(op, castOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertTritonToAscendPass
    : public impl::ConvertTritonToAscendPassBase<ConvertTritonToAscendPass> {
  using ConvertTritonToAscendPassBase::ConvertTritonToAscendPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // === Phase 1: Analysis ===
    // Mark all tt.load operations that are used by tt.dot (directly or through
    // tt.trans) This allows ConvertTritonLoad to correctly choose cube_load vs
    // vector_load
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "tt.dot")
        return;

      // For each operand of tt.dot, trace back to find source tt.load
      for (Value operand : op->getOperands()) {
        Value current = operand;
        // Trace through tt.trans, tt.bitcast, etc.
        while (Operation *def = current.getDefiningOp()) {
          StringRef defName = def->getName().getStringRef();
          if (defName == "tt.load") {
            // Mark this load as used by dot
            def->setAttr("ascend.used_by_dot", UnitAttr::get(ctx));
            break;
          }
          if (defName == "tt.trans" || defName == "tt.bitcast" ||
              defName == "tt.reshape" || defName == "tt.expand_dims") {
            current = def->getOperand(0);
          } else {
            break;
          }
        }
      }
    });

    // === Phase 2: Pattern Rewriting ===
    RewritePatternSet patterns(ctx);

    // === High benefit (10): Triton compute ops -> AscendModel ops ===
    // These run first and break pointer chain dependencies.
    patterns.add<ConvertTritonDot>(ctx); // tt.dot -> matmul + cube_store
    patterns.add<ConvertTritonLoad>(
        ctx); // tt.load -> cube_load (for dot) or vector_load
    patterns.add<ConvertTritonTrans>(
        ctx); // tt.trans -> pass-through (zero cost)
    patterns.add<ConvertTritonStore>(ctx);  // tt.store -> ascend.vector_store
    patterns.add<ConvertTritonReduce>(ctx); // tt.reduce -> ascend.reduce_*
    patterns.add<ConvertTritonBroadcast>(
        ctx); // tt.broadcast -> ascend.broadcast
    // NOTE: tt.func/tt.return are left as-is (see comment above)

    // === Low benefit (1): cleanup and arith/math conversion ===
    // Erase dead address ops (cascade: addptr -> splat -> make_range)
    patterns.add<EraseDeadTritonAddrOps>(ctx);

    // Arith/Math tensor ops -> AscendModel vector ops
    patterns.add<ConvertArithBinaryOp>(ctx);
    patterns.add<ConvertArithCmpOp>(ctx);
    patterns.add<ConvertMathUnaryOp>(ctx);
    patterns.add<ConvertArithSelect>(ctx);
    patterns.add<ConvertArithCast>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
