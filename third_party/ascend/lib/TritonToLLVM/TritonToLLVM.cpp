#include "TritonToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOLLVM
#include "ascend/include/TritonToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
struct TritonToLLVMPass
    : public mlir::triton::impl::TritonToLLVMBase<TritonToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

namespace {

static Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementType();
  return type;
}

static int64_t getTensorNumElements(Value tensor) {
  auto type = mlir::cast<RankedTensorType>(tensor.getType());
  return type.getNumElements();
}

static Value getInt32Value(RewriterBase &rewriter, Location loc, int val) {
  Type ty = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, ty,
                                           rewriter.getIntegerAttr(ty, val));
}

// If operand size is smaller than 32 bits, pack in groups of 32 bits.
SmallVector<Value> packOperands(mlir::triton::ElementwiseInlineAsmOp op,
                                const SmallVector<SmallVector<Value>> &operands,
                                RewriterBase &rewriter, Location loc) {
  SmallVector<Value> packedOperands;
  unsigned numPackedElements = op.getPackedElement();
  for (int i = 0, e = op.getNumOperands(); i < e; i++) {
    Type elemTy = getElementType(op.getOperand(i));
    unsigned bitWidth =
        elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
    unsigned numElementPerReg = std::max(32 / bitWidth, 1u);
    numElementPerReg = std::min(numElementPerReg, numPackedElements);
    for (int j = 0; j < numPackedElements; j += numElementPerReg) {
      if (numElementPerReg == 1) {
        packedOperands.push_back(operands[j][i]);
        continue;
      }
      Type t = VectorType::get(numElementPerReg, elemTy);
      Value packed = rewriter.create<LLVM::UndefOp>(loc, t);
      for (int k = 0; k < numElementPerReg; k++) {
        packed = rewriter.create<LLVM::InsertElementOp>(
            loc, packed, operands[j + k][i], getInt32Value(rewriter, loc, k));
      }
      packedOperands.push_back(packed);
    }
  }
  return packedOperands;
}

static SmallVector<Value> unpackElements(Location loc, Value packedValues,
                                         RewriterBase &rewriter) {
  auto type = mlir::cast<RankedTensorType>(packedValues.getType());
  auto elementType = type.getElementType();
  auto shape = type.getShape();

  int64_t numElements = type.getNumElements();

  SmallVector<Value> result;
  for (int64_t linearIdx = 0; linearIdx < numElements; linearIdx++) {
    SmallVector<Value> indexes(shape.size());
    int64_t remaining = linearIdx;
    for (int64_t dim = shape.size() - 1; dim >= 0; dim--) {
      indexes[dim] =
          rewriter.create<arith::ConstantIndexOp>(loc, remaining % shape[dim]);
      remaining /= shape[dim];
    }
    Value extracted = rewriter.create<tensor::ExtractOp>(loc, elementType,
                                                         packedValues, indexes);
    result.push_back(extracted);
  }

  return result;
}

static SmallVector<SmallVector<Value>>
createDestOps(triton::ElementwiseInlineAsmOp op, RewriterBase &rewriter,
              const SmallVector<SmallVector<Value>> operands, Location loc) {
  auto ctx = op->getContext();
  if (operands.size() % op.getPackedElement() != 0)
    llvm::report_fatal_error("Inline asm op has more packed elements than "
                             "number of elements per thread.");

  // Pack elems smaller than 32 bits into 32-bit registers.
  SmallVector<Value> packedOperands = packOperands(op, operands, rewriter, loc);

  // Types returned by the LLVM asm op.  If there's more than one, they'll be
  // wrapped in a type tuple.
  SmallVector<Type> asmRetTypes;
  for (auto result : op.getResult()) {
    auto ty = getElementType(result);

    // Pack return elements into 32-bits.
    unsigned bitWidth = ty.isIntOrFloat() ? ty.getIntOrFloatBitWidth() : 64;
    unsigned numElemsPerReg =
        std::min(std::max(32 / bitWidth, 1u), op.getPackedElement());
    assert(op.getPackedElement() % numElemsPerReg == 0);
    if (numElemsPerReg > 1) {
      ty = VectorType::get(numElemsPerReg, ty);
    }
    for (unsigned i = 0; i < op.getPackedElement() / numElemsPerReg; i++) {
      asmRetTypes.push_back(ty);
    }
  }
  Type asmRetType = asmRetTypes.size() > 1
                        ? LLVM::LLVMStructType::getLiteral(ctx, asmRetTypes)
                        : asmRetTypes[0];

  Value asmResults =
      rewriter
          .create<LLVM::InlineAsmOp>(
              loc, asmRetType,
              /*operands=*/packedOperands,
              /*asm_string=*/op.getAsmString(),
              /*constraints=*/op.getConstraints(),
              /*has_side_effects=*/!op.getPure(),
              /*is_align_stack=*/false,
              /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
              /*asm_dialect=*/
              LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                        LLVM::AsmDialect::AD_ATT),
              /*operand_attrs=*/ArrayAttr())
          ->getResult(0);

  // asmResults is a flat struct; pack its values into
  // [return_value][op.getPackedElement()].
  SmallVector<SmallVector<Value>> ret(op->getNumResults());
  int structIdx = 0;
  for (int i = 0; i < op->getNumResults(); i++) {
    for (int j = 0; j < op.getPackedElement(); j++) {
      Value val;
      if (asmRetTypes.size() > 1) {
        val =
            rewriter.create<LLVM::ExtractValueOp>(loc, asmResults, structIdx++);
      } else {
        val = asmResults;
      }
      if (auto vectorTy = dyn_cast<VectorType>(val.getType())) {
        for (int k = 0; k < vectorTy.getNumElements(); k++) {
          ret[i].push_back(rewriter.create<LLVM::ExtractElementOp>(
              loc, val, getInt32Value(rewriter, loc, k)));
        }
        j += vectorTy.getNumElements() - 1;
      } else {
        ret[i].push_back(val);
      }
    }
  }
  return ret;
}

static LogicalResult processScalarInlineAsm(triton::ElementwiseInlineAsmOp op,
                                            PatternRewriter &rewriter) {
  Location loc = op.getLoc();

  auto outsWrapped = createDestOps(op, rewriter, {}, loc);

  SmallVector<Value> outs;
  for (const auto &resWrapped : outsWrapped) {
    outs.push_back(resWrapped[0]);
  }
  rewriter.replaceOp(op, outs);

  return success();
}

static LogicalResult processVectorInlineAsm(triton::ElementwiseInlineAsmOp op,
                                            PatternRewriter &rewriter) {
  Location loc = op.getLoc();

  SmallVector<SmallVector<Value>> unpackedOperands;
  for (auto operand : op.getOperands()) {
    auto unpackedOperand = unpackElements(loc, operand, rewriter);
    unpackedOperands.push_back(unpackedOperand);
  }

  int64_t resultLength = getTensorNumElements(op->getResult(0));
  if (resultLength % op.getPackedElement()) {
    op.emitError("Result tensor should be diveded to pack");
    return failure();
  }

  SmallVector<SmallVector<Value>> unpackedResults(op->getNumResults());
  for (int64_t i = 0; i < resultLength; i += op.getPackedElement()) {
    // Block of elements to process with one call to the inline asm.  This is
    // ordered opposite `unpackedResults`: The outer dim is
    // op.getPackedElement(), and the inner dim is the operand.
    SmallVector<SmallVector<Value>> block(op.getPackedElement());
    for (auto &os : unpackedOperands) {
      for (int j = 0; j < op.getPackedElement(); j++) {
        block[j].push_back(os[i + j]);
      }
    }
    auto cur = createDestOps(op, rewriter, block, loc);
    assert(cur.size() == unpackedResults.size());
    for (unsigned j = 0; j < cur.size(); j++) {
      unpackedResults[j].insert(unpackedResults[j].end(), cur[j].begin(),
                                cur[j].end());
    }
  }
  // Reorder and pack the results.
  SmallVector<Value> outs;
  for (int i = 0; i < unpackedResults.size(); i++) {
    outs.push_back(rewriter.create<tensor::FromElementsOp>(
        loc, op->getResult(i).getType(), unpackedResults[i]));
  }
  rewriter.replaceOp(op, outs);

  return success();
}

} // namespace

struct ElementwiseInlineAsmOpConversion
    : OpRewritePattern<triton::ElementwiseInlineAsmOp> {
  using OpRewritePattern<triton::ElementwiseInlineAsmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ElementwiseInlineAsmOp op,
                                PatternRewriter &rewriter) const final {
    return op.getOperands().empty() ? processScalarInlineAsm(op, rewriter)
                                    : processVectorInlineAsm(op, rewriter);
  }
};

void TritonToLLVMPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<tensor::TensorDialect, LLVM::LLVMDialect,
                         arith::ArithDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ElementwiseInlineAsmOpConversion>(patterns.getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToLLVMPass() {
  return std::make_unique<TritonToLLVMPass>();
}
