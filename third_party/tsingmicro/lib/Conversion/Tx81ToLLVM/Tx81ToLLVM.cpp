//===--------------------- Tx81ToLLVM.cpp ---------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to convert operations from tx dialect to
// LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Tx81ToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "utils/utils.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "tx81-to-llvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81ToLLVM/Passes.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//
// Crt func name
const char rdmaFuncName[] = "__Rdma";
const char wdmaFuncName[] = "__Wdma";
const char memcpyFuncName[] = "__Memcpy";
const char addVVFuncName[] = "__AddVV";
const char subVVFuncName[] = "__SubVV";
const char mulVVFuncName[] = "__MulVV";
const char divVVFuncName[] = "__DivVV";
const char absVVFuncName[] = "__AbsVV";
const char rsqrtVVFuncName[] = "__RsqrtVV";
const char sqrtVVFuncName[] = "__SqrtVV";
const char recipVVFuncName[] = "__RecipVV";
const char lnFuncName[] = "__Ln";
const char log2FuncName[] = "__Log2";
const char expFuncName[] = "__Exp";
const char pow2FuncName[] = "__Pow2";
const char sinFuncName[] = "__Sin";
const char cosFuncName[] = "__Cos";
const char addVSFuncName[] = "__AddVS";
const char subVSFuncName[] = "__SubVS";
const char mulVSFuncName[] = "__MulVS";
const char divVSFuncName[] = "__DivVS";
const char argMinFuncName[] = "__ArgMin";
const char argMaxFuncName[] = "__ArgMax";
const char reduceSumFuncName[] = "__ReduceSum";
const char reduceMaxFuncName[] = "__ReduceMax";
const char reduceMinFuncName[] = "__ReduceMin";
// Int8
const char int8ToBf16FuncName[] = "__INT8_BF16";
const char int8ToFp16FuncName[] = "__INT8_FP16";
const char int8ToFp32FuncName[] = "__INT8_FP32";
const char int8ToTf32FuncName[] = "__INT8_TF32";
// Int16
const char int16ToFp16FuncName[] = "__INT16_FP16";
const char int16ToBf16FuncName[] = "__INT16_BF16";
const char int16ToFp32FuncName[] = "__INT16_FP32";
const char int16ToTf32FuncName[] = "__INT16_TF32";
// Int32
const char int32ToFp16FuncName[] = "__INT32_FP16";
const char int32ToBf16FuncName[] = "__INT32_BF16";
const char int32ToFp32FuncName[] = "__INT32_FP32";
const char int32ToTf32FuncName[] = "__INT32_TF32";
// BF16
const char bf16ToInt8FuncName[] = "__BF16_INT8";
const char bf16ToInt16FuncName[] = "__BF16_INT16";
const char bf16ToInt32FuncName[] = "__BF16_INT32";
const char bf16ToFp16FuncName[] = "__BF16_FP16";
const char bf16ToFp32FuncName[] = "__BF16_FP32";
const char bf16ToTf32FuncName[] = "__BF16_TF32";
// FP16
const char fp16ToBf16FuncName[] = "__FP16_BF16";
const char fp16ToFp32FuncName[] = "__FP16_FP32";
const char fp16ToTf32FuncName[] = "__FP16_TF32";
const char fp16ToInt8FuncName[] = "__FP16_INT8";
const char fp16ToInt16FuncName[] = "__FP16_INT16";
const char fp16ToInt32FuncName[] = "__FP16_INT32";
// FP32
const char fp32ToInt8FuncName[] = "__FP32_INT8";
const char fp32ToInt16FuncName[] = "__FP32_INT16";
const char fp32ToInt32FuncName[] = "__FP32_INT32";
const char fp32ToFp16FuncName[] = "__FP32_FP16";
const char fp32ToBf16FuncName[] = "__FP32_BF16";
const char fp32ToTf32FuncName[] = "__FP32_TF32";
// TF32
const char tf32ToInt8FuncName[] = "__TF32_INT8";
const char tf32ToInt16FuncName[] = "__TF32_INT16";
const char tf32ToInt32FuncName[] = "__TF32_INT32";
const char tf32ToFp16FuncName[] = "__TF32_FP16";
const char tf32ToBf16FuncName[] = "__TF32_BF16";
const char tf32ToFp32FuncName[] = "__TF32_FP32";
// MXFP
const char fp8E4M3ToBF16FuncName[] = "__FP8E4M3_BF16";
const char fp8E5M2ToBF16FuncName[] = "__FP8E5M2_BF16";
const char fp4E2M1ToBF16FuncName[] = "__FP4E2M1_BF16";

const char boolEqualVVFuncName[] = "__BoolEqualVV";
const char boolUnEqualVVFuncName[] = "__BoolUnEqualVV";
const char boolGreaterEqualVVFuncName[] = "__BoolGreaterEqualVV";
const char boolGreaterVVFuncName[] = "__BoolGreaterVV";
const char boolLessEqualVVFuncName[] = "__BoolLessEqualVV";
const char boolLessThenVVFuncName[] = "__BoolLessThenVV";
const char equalVVFuncName[] = "__EqualVV";
const char unEqualVVFuncName[] = "__UnEqualVV";
const char greaterEqualVVFuncName[] = "__GreaterEqualVV";
const char greaterVVFuncName[] = "__GreaterVV";
const char lessEqualVVFuncName[] = "__LessEqualVV";
const char lessThenVVFuncName[] = "__LessThenVV";
const char boolEqualVSFuncName[] = "__BoolEqualVS";
const char boolUnEqualVSFuncName[] = "__BoolUnEqualVS";
const char boolGreaterEqualVSFuncName[] = "__BoolGreaterEqualVS";
const char boolGreaterVSFuncName[] = "__BoolGreaterVS";
const char boolLessEqualVSFuncName[] = "__BoolLessEqualVS";
const char boolLessThenVSFuncName[] = "__BoolLessThenVS";
const char equalVSFuncName[] = "__EqualVS";
const char unEqualVSFuncName[] = "__UnEqualVS";
const char greaterEqualVSFuncName[] = "__GreaterEqualVS";
const char greaterVSFuncName[] = "__GreaterVS";
const char lessEqualVSFuncName[] = "__LessEqualVS";
const char lessThenVSFuncName[] = "__LessThenVS";
const char andVVFuncName[] = "__AndVV";
const char orVVFuncName[] = "__OrVV";
const char xorVVFuncName[] = "__XorVV";
const char boolNotVFuncName[] = "__BoolNotV";
const char boolAndVFuncName[] = "__BoolAndV";
const char boolOrVFuncName[] = "__BoolOrV";
const char boolXorVFuncName[] = "__BoolXorV";
const char MaxVVFuncName[] = "__MaxVV";
const char MinVVFuncName[] = "__MinVV";
const char transposeFuncName[] = "__Transpose";
const char nchw2nhwcFuncName[] = "__Nchw2nhwc";
const char nhwc2nchwFuncName[] = "__Nhwc2nchw";
const char tanhFuncName[] = "__Tanh";

static Value adjustElemCountType(ConversionPatternRewriter &rewriter,
                                 Location loc, Value elemCount) {
  Value newElemCount = elemCount;
  if (isa<IndexType>(elemCount.getType())) {
    newElemCount = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), elemCount);
  } else if (isa<IntegerType>(elemCount.getType())) {
    auto elemCountType = dyn_cast<IntegerType>(elemCount.getType());
    if (elemCountType.isInteger(64))
      newElemCount = rewriter.create<arith::TruncIOp>(
          loc, rewriter.getI32Type(), elemCount);
  }
  return newElemCount;
}

static Value castIndexToInt32(ConversionPatternRewriter &rewriter, Location loc,
                              Value indexOp) {
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                             indexOp);
}

static Value createInt32ValueArray(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> array,
                                   Operation *currentOp) {
  auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Ty = rewriter.getI32Type();
  auto i64Ty = rewriter.getI64Type();

  // Find the parent function of the current operation
  Operation *parentFunc = currentOp->getParentOfType<LLVM::LLVMFuncOp>();
  LLVM::LLVMFuncOp funcOp = dyn_cast<LLVM::LLVMFuncOp>(parentFunc);
  assert(funcOp &&
         "Expected to find a parent function for the current operation\n");

  // Save current insertion point
  auto savedInsertionPoint = rewriter.saveInsertionPoint();

  // Insert alloca at the beginning of the function entry block
  Block &entryBlock = funcOp.getBody().front();
  rewriter.setInsertionPointToStart(&entryBlock);

  // Allocate memory for array
  Value rank = rewriter.create<LLVM::ConstantOp>(
      loc, i64Ty, rewriter.getI64IntegerAttr(array.size()));

  auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, i32PtrTy, i32Ty, rank);

  rewriter.restoreInsertionPoint(savedInsertionPoint);

  //   assert(moduleOp && moduleOp->hasAttr("triton_tsm.spm_use") &&
  //          "ModuleOp should not be null when creating an array");
  //   auto spmPointer =
  //   cast<IntegerAttr>(moduleOp->getAttr("triton_tsm.spm_use"))
  //                         .getValue()
  //                         .getZExtValue();
  //   Value spmOffsetOp = rewriter.create<LLVM::ConstantOp>(
  //       loc, rewriter.getI64Type(), rewriter.getI32IntegerAttr(spmPointer));
  //   auto elementPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  //   Value spmAddr = rewriter.create<LLVM::ZeroOp>(loc, elementPtrType);
  //   spmAddr =
  //       rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(),
  //       spmAddr);
  //   spmAddr = rewriter.create<LLVM::AddOp>(loc, rewriter.getI64Type(),
  //   spmAddr,
  //                                          spmOffsetOp);

  //   // Types for function declaration
  //   SmallVector<Type, 5> argTypes = {
  //       rewriter.getI64Type() // offset
  //   };

  // Declare the function
  //   Value funcPtr = triton::utils::declareTx81Function(
  //       moduleOp, rewriter, loc, "get_spm_memory_mapping_wrapper",
  //       elementPtrType, argTypes);

  // Create the call to __Rdma
  //   auto spmMemoryAddrPtr = rewriter.create<LLVM::CallOp>(
  //       loc, TypeRange{elementPtrType},
  //       "get_spm_memory_mapping_wrapper", // funcPtr,
  //       ValueRange{spmAddr});

  // Restore insertion point
  rewriter.restoreInsertionPoint(savedInsertionPoint);

  // Store each dimension in the array
  for (size_t i = 0; i < array.size(); i++) {
    // Create the index
    Value idx = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI32IntegerAttr(i));

    // Create GEP to get pointer to array element
    Value elemPtr = rewriter.create<LLVM::GEPOp>(loc, i32PtrTy, i32Ty, allocaOp,
                                                 ArrayRef<Value>{idx});

    // Store the value
    rewriter.create<LLVM::StoreOp>(loc, array[i], elemPtr);
  }

  //   spmPointer += array.size() * sizeof(int32_t);
  //   // Record spm usage.
  //   moduleOp->setAttr(
  //       "triton_tsm.spm_use",
  //       mlir::IntegerAttr::get(mlir::IntegerType::get(moduleOp.getContext(),
  //       32),
  //                              spmPointer));

  return allocaOp;
}

static Value
indexValueArrayToInt32ValueArray(ConversionPatternRewriter &rewriter,
                                 Location loc, ValueRange array,
                                 Operation *currentOp) {

  SmallVector<Value> arrayValues;
  for (size_t i = 0; i < array.size(); i++) {
    // Create the dimension value
    arrayValues.push_back(castIndexToInt32(rewriter, loc, array[i]));
  }

  return createInt32ValueArray(rewriter, loc, arrayValues, currentOp);
}

static Value int32ArrayToInt32ValueArray(ConversionPatternRewriter &rewriter,
                                         Location loc, ArrayRef<int32_t> array,
                                         Operation *currentOp) {

  SmallVector<Value> arrayValues;
  auto i32Ty = rewriter.getI32Type();
  for (size_t i = 0; i < array.size(); i++) {
    // Create the dimension value
    arrayValues.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(array[i])));
  }
  return createInt32ValueArray(rewriter, loc, arrayValues, currentOp);
}

//===----------------------------------------------------------------------===//
// Arith Operation Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert constant operations to LLVM constants
struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the constant value
    auto constAttr = op.getValue();

    // Get the result type
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Handle different attribute types
    if (auto intAttr = dyn_cast<IntegerAttr>(constAttr)) {
      // Convert integer attribute
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, intAttr);
      return success();
    } else if (auto floatAttr = dyn_cast<FloatAttr>(constAttr)) {
      // Convert float attribute
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, resultType, floatAttr);
      return success();
    } else if (auto boolAttr = dyn_cast<BoolAttr>(constAttr)) {
      // Convert bool attribute to i1
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, resultType,
          rewriter.getIntegerAttr(resultType, boolAttr.getValue()));
      return success();
    }

    return failure();
  }
};

// Convert arith.index_cast to appropriate LLVM conversions
struct IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get source and result types
    auto srcType = adaptor.getIn().getType();
    auto dstType = getTypeConverter()->convertType(op.getResult().getType());

    // Convert from index to specific integer type
    if (isa<LLVM::LLVMPointerType>(srcType) && isa<IntegerType>(dstType)) {
      rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, dstType,
                                                    adaptor.getIn());
      return success();
    }

    // Convert from specific integer type to index
    if (isa<IntegerType>(srcType) && isa<LLVM::LLVMPointerType>(dstType)) {
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, dstType,
                                                    adaptor.getIn());
      return success();
    }

    // Handle integer to integer casts
    if (isa<IntegerType>(srcType) && isa<IntegerType>(dstType)) {
      unsigned srcWidth = cast<IntegerType>(srcType).getWidth();
      unsigned dstWidth = cast<IntegerType>(dstType).getWidth();

      if (srcWidth < dstWidth) {
        // Sign extend if source is signed, zero extend otherwise
        rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstType, adaptor.getIn());
      } else if (srcWidth > dstWidth) {
        // Truncate
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, dstType,
                                                   adaptor.getIn());
      } else {
        // Same width, just pass through
        rewriter.replaceOp(op, adaptor.getIn());
      }
      return success();
    }

    return failure();
  }
};

// Convert arith.addi to LLVM add
struct AddIOpConversion : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

// Convert arith.muli to LLVM mul
struct MulIOpConversion : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tx81 Operation Conversion Patterns
//===----------------------------------------------------------------------===//

struct BarrierConversion : public OpConversionPattern<tx::BarrierOp> {
  using OpConversionPattern<tx::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Barrier runtime function if not already declared
    /*
    void __Barrier()
    */

    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__Barrier", i8PtrTy, {});

    // Create the call to __Barrier
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{i8PtrTy},
                                              "__Barrier", // funcPtr,
                                              ValueRange{});

    // Replace the op with the call
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct RdmaWdmaOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the __Rdma runtime function if not already declared
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(ctx);

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy,  // src
        i8PtrTy,  // target
        i32PtrTy, // src_shape array
        i32PtrTy, // src_strides array
        i32PtrTy, // dst_shape array
        i32PtrTy, // dst_strides array
        i32Ty,    // rank
        i32Ty,    // elemBytes
        i32Ty     // fmt
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, loc, funcPrefix, i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();
    src = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, src);

    Value target = adaptor.getTarget();
    target = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, target);

    // Create arrays for shapes and strides

    // Create arrays for shapes and strides
    Value srcShapeArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getSrcShape(), op);
    Value srcStridesArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getSrcStrides(), op);
    Value dstShapeArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstShape(), op);
    Value dstStridesArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstStrides(), op);

    // Handle rank attribute
    Value rank = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getRank()));

    // Handle elem byte attribute
    Value elemBytes = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getElemBytes()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __Rdma
    auto call = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i8PtrTy}, funcPrefix,
        ValueRange{src, target, srcShapeArray, srcStridesArray, dstShapeArray,
                   dstStridesArray, rank, elemBytes, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.mask_move to LLVM call to __MaskMove function
struct MaskMoveOpConversion : public OpConversionPattern<tx::MaskMoveOp> {
  using OpConversionPattern<tx::MaskMoveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::MaskMoveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __MaskMove runtime function if not already declared
    // Signature: void* __MaskMove(void* source, void* target, uint32_t
    // elem_count, int32_t* masks, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy,  // source
        i8PtrTy,  // target
        i32Ty,    // elem_count
        i32PtrTy, // masks
        i32Ty     // fmt
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__MaskMove", i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();

    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    Value target = adaptor.getTarget();

    // Need to bitcast src to i8*
    target = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, target);
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op->getLoc(), elemCount);

    // Handle mask arrays
    Value mask = adaptor.getMask();

    // Need to bitcast src to i8*
    mask = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, mask);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call to __MaskMove
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__MaskMove", // funcPtr,
        ArrayRef<Value>{src, target, elemCount, mask, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct TransformOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __Transpose(uint64_t *src, uint64_t *dst, int32_t *src_shape, int32_t
    // *dst_shape, uint16_t fmt)

    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32PtrTy, i32PtrTy,
                                      i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSource();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value dst = adaptor.getTarget();
    // Need to bitcast src to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert shape attribute to Value
    ArrayRef<int32_t> srcShape = adaptor.getSrcShape();
    ArrayRef<int32_t> dstShape = adaptor.getDstShape();

    // Get shape llvm array
    auto srcArray =
        int32ArrayToInt32ValueArray(rewriter, op.getLoc(), srcShape, op);
    auto dstArray =
        int32ArrayToInt32ValueArray(rewriter, op.getLoc(), dstShape, op);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, dst, srcArray, dstArray, fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

struct GatherScatterOpConversion
    : public OpConversionPattern<tx::GatherScatter> {
  using OpConversionPattern<tx::GatherScatter>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::GatherScatter op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __GatherScatter runtime function if not already declared
    /*
    void __GatherScatter(uint64_t *src, uint64_t *dst, uint32_t bytes,
                     uint32_t src_strideN, uint32_t src_strideH,
                     uint32_t src_strideW, uint32_t src_iterN,
                     uint32_t src_iterH, uint32_t src_iterW,
                     uint32_t dst_strideN, uint32_t dst_strideH,
                     uint32_t dst_strideW, uint32_t dst_iterN,
                     uint32_t dst_iterH, uint32_t dst_ite_W)
    */
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy, // src
        i8PtrTy, // dst
        i32Ty,   // bytes
        i32Ty,   // src_StrideN
        i32Ty,   // src_StrideH
        i32Ty,   // src_StrideW
        i32Ty,   // dst_StrideN
        i32Ty,   // dst_StrideH
        i32Ty,   // dst_StrideW
        i32Ty,   // src_IterN
        i32Ty,   // src_IterH
        i32Ty,   // src_IterW
        i32Ty,   // dst_IterN
        i32Ty,   // dst_IterH
        i32Ty    // dst_IterW
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, loc, "__GatherScatter", i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSource();
    src = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, src);

    // Get the operands
    Value dst = adaptor.getTarget();
    dst = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, dst);

    // Get bytes
    auto bytes =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getBytes());

    // Get strides
    auto srcStrideN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcStrideN());
    auto srcStrideH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcStrideH());
    auto srcStrideW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcStrideW());
    auto dstStrideN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstStrideN());
    auto dstStrideH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstStrideH());
    auto dstStrideW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstStrideW());

    // Get iterator
    auto srcIterN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcIterN());
    auto srcIterH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcIterH());
    auto srcIterW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getSrcIterW());
    auto dstIterN =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstIterN());
    auto dstIterH =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstIterH());
    auto dstIterW =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, adaptor.getDstIterW());

    // Create the call to __GatherScatter
    auto call = rewriter.create<LLVM::CallOp>(
        loc, TypeRange{i8PtrTy}, "__GatherScatter", // funcPtr,
        ValueRange{src, dst, bytes, srcStrideN, srcStrideH, srcStrideW,
                   srcIterN, srcIterH, srcIterW, dstStrideN, dstStrideH,
                   dstStrideW, dstIterN, dstIterH, dstIterW});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct ArgMinMaxOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:

    // __ArgMinMax(uint64_t *src, uint64_t *dst0, uint64_t *dst1,
    //             uint32_t elem_count, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    // Convert results
    Value value = adaptor.getValue();
    Value index = adaptor.getIndex();
    // Need to bitcast `value` and `index` to i8*
    value = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, value);
    index = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, index);

    // Get elem_count operand, convert Index to I32
    Value elemCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getElemCount()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, value, index, elemCount, fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert tx81.binary op to LLVM call
template <typename Tx81Op, const char *funcPrefix>
struct ReduceOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __ReduceSum(uint64_t *src, uint64_t *dst, uint32_t dim, uint16_t src_n,
    // uint16_t src_h, uint16_t src_w, uint16_t src_c, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value srcB = adaptor.getSrc();
    Value dst = adaptor.getDst();
    // Need to bitcast src to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert dim attribute to Value
    Value dim = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getDim()));

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, dst, dim, shape_n, shape_h, shape_w, shape_c,
                        fmt});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert tx81.elementwise op to LLVM call
template <typename Tx81Op, const char *funcPrefix>
struct ElementWiseOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;
  // using OpConversionPattern<Tx81Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Add(void* a, void* b, void* out, uint32_t elem_count,
    // uint32_t rnd_mode, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy,

                                      i32Ty,   i32Ty,   i32Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle round attribute
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRndMode()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, rnd_mode, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct UnaryOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Abs(void* src, void* dst, uint32_t elem_count,
    // uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    // Need to bitcast src to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    Value out = adaptor.getOut();
    // Need to bitcast out to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// FIXME: Use trait to refactor the BinaryVSOpConversion and
// ElementWiseOpConversion
template <typename Tx81Op, const char *funcPrefix>
struct BinaryVSOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __Add(void* a, void* b, void* out, uint32_t elem_count,
    // uint32_t rnd_mode, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i32Ty, i8PtrTy,
                                      i32Ty,   i32Ty, i32Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);

    Value srcB = adaptor.getValue();

    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle round attribute
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRndMode()));

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, rnd_mode, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct BinaryLogicVVOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __XorVV(void* a, void* b, void* out, uint32_t
    // elem_count, uint32_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src0_addr
        i8PtrTy, // src1_addr
        i8PtrTy, // dst_addr
        i32Ty,   // elem_count
        i32Ty    // fmt
    };

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct UnaryBoolLogicVOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __BoolNotV(void* src, void* dst, uint32_t elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src_addr
        i8PtrTy, // dst_addr
        i32Ty    // elem_count
    };

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getInput();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);

    Value out = adaptor.getOut();
    // Need to bitcast dest to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{src, out, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename Tx81Op, const char *funcPrefix>
struct BinaryBoolLogicVOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void* __BoolAndV(void* a, void* b, void* out, uint32_t
    // elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy, // src0_addr
        i8PtrTy, // src1_addr
        i8PtrTy, // dst_addr
        i32Ty    // elem_count
    };

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

template <typename RelationVVOp, const char *funcPrefix>
struct RelationVVOpConversion : public OpConversionPattern<RelationVVOp> {
  using OpConversionPattern<RelationVVOp>::OpConversionPattern;
  using OpAdaptor = typename RelationVVOp::Adaptor;
  // using OpConversionPattern<Tx81Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RelationVVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __BoolLessEqualVV(uint64_t *src0, uint64_t *src1,
    // uint64_t *dst, uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    Value srcB = adaptor.getInput1();
    // Need to bitcast src to i8*
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// FIXME: Use trait to refactor the RelationVSOpConversion and
// ElementWiseOpConversion
template <typename Tx81Op, const char *funcPrefix>
struct RelationVSOpConversion : public OpConversionPattern<Tx81Op> {
  using OpConversionPattern<Tx81Op>::OpConversionPattern;
  using OpAdaptor = typename Tx81Op::Adaptor;

  LogicalResult
  matchAndRewrite(Tx81Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __BoolEqualVS(uint64_t *src0, uint32_t src1, uint64_t
    // *dst,uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i32Ty, i8PtrTy, i32Ty, i32Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getInput0();
    // Need to bitcast src to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);

    Value srcB = adaptor.getValue();

    Value out = adaptor.getOut();
    // Need to bitcast src to i8*
    out = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, out);

    // Get elem_count operand, convert Index to I32
    Value elemCount = op.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{srcA, srcB, out, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.ZeroPointConvertOp op to LLVM
template <typename ZeroPointConvertOp, const char *funcPrefix>
struct ZeroPointConvertOpConversion
    : public OpConversionPattern<ZeroPointConvertOp> {
  using OpConversionPattern<ZeroPointConvertOp>::OpConversionPattern;
  using OpAdaptor = typename ZeroPointConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(ZeroPointConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __INT8_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // zero_point, uint32_t elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i32Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getSrc();
    Value output = adaptor.getDst();
    Value zeroPoint = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, adaptor.getZeroPointAttr());
    Value elemCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, adaptor.getElemCountAttr());

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, zeroPoint, elemCount});

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert tx81.NormalConvertOp op to LLVM
template <typename NormalConvertOp, const char *funcPrefix>
struct NormalConvertOpConversion : public OpConversionPattern<NormalConvertOp> {
  using OpConversionPattern<NormalConvertOp>::OpConversionPattern;
  using OpAdaptor = typename NormalConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(NormalConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __FP16_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // elem_count);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, elemCount});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.RoundConvertOp op to LLVM
template <typename RoundConvertOp, const char *funcPrefix>
struct RoundConvertOpConversion : public OpConversionPattern<RoundConvertOp> {
  using OpConversionPattern<RoundConvertOp>::OpConversionPattern;
  using OpAdaptor = typename RoundConvertOp::Adaptor;

  LogicalResult
  matchAndRewrite(RoundConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __INT16_FP32(uint64_t *src, uint64_t *dst, uint32_t
    // elem_count, RND_MODE round);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), funcPrefix, i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOutput();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);
    Value rnd_mode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getRndMode()));

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, funcPrefix, // funcPtr,
        ArrayRef<Value>{input, output, elemCount, rnd_mode});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

struct MXFPScaleBF16OpConversion
    : public OpConversionPattern<tx::MXFPScaleBF16Op> {
  using OpConversionPattern<tx::MXFPScaleBF16Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::MXFPScaleBF16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 5> argTypes = {
        i8PtrTy, // src
        i8PtrTy, // scale
        i8PtrTy, // dst
        i32Ty,   // elem_count
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__mxfpScaleBf16", i8PtrTy, argTypes);

    // Get the operands
    Value src = adaptor.getSrc();
    Value scale = adaptor.getScale();
    Value dst = adaptor.getDst();

    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    scale = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, scale);
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    Value elemCount = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, adaptor.getElemCountAttr());

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__mxfpScaleBf16", // funcPtr,
        ArrayRef<Value>{src, scale, dst, elemCount});

    op->erase();
    return success();
  }
};

struct BitToFPOpConversion : public OpConversionPattern<tx::Bit2FpOp> {
  using OpConversionPattern<tx::Bit2FpOp>::OpConversionPattern;
  using OpAdaptor = tx::Bit2FpOp::Adaptor;

  LogicalResult
  matchAndRewrite(tx::Bit2FpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature: void __Bit2Fp(uint64_t *src, uint64_t *target, uint32_t
    // elem_count, uint16_t fmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__Bit2Fp", i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getSrc();
    Value output = adaptor.getTarget();
    Value elemCount = adaptor.getElemCount();
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Bit2Fp", // funcPtr,
        ArrayRef<Value>{input, output, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.channel_norm op
struct ChannelNormOpConversion : public OpConversionPattern<tx::ChannelNormOp> {
  using OpConversionPattern<tx::ChannelNormOp>::OpConversionPattern;
  using OpAdaptor = typename tx::ChannelNormOp::Adaptor;

  LogicalResult
  matchAndRewrite(tx::ChannelNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __ChannelNorm(uint64_t *src, uint64_t *dst, uint16_t n,
    // uint16_t h, uint16_t w, uint16_t c, uint16_t c0, uint16_t
    // dtype_size)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i16Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__ChannelNorm", i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value dst = adaptor.getDst();
    // Need to bitcast dst to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Convert c0_align attribute to Value
    Value c0Align = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getC0Align()));

    // Convert dtype_size attribute to Value
    Value dtypeSize = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI32IntegerAttr(op.getDtypeSize()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__ChannelNorm", // funcPtr,
        ArrayRef<Value>{src, dst, shape_n, shape_h, shape_w, shape_c, c0Align,
                        dtypeSize});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

struct DechannelNormOpConversion
    : public OpConversionPattern<tx::DechannelNormOp> {
  using OpConversionPattern<tx::DechannelNormOp>::OpConversionPattern;
  using OpAdaptor = typename tx::DechannelNormOp::Adaptor;

  LogicalResult
  matchAndRewrite(tx::DechannelNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->template getParentOfType<ModuleOp>();

    // Declare the runtime function if not already declared
    // Signature:
    // __DechannelNorm(uint64_t *src, uint64_t *dst, uint16_t n,
    // uint16_t h, uint16_t w, uint16_t c, uint16_t c0, uint16_t
    // dtype_size)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i16Ty, i16Ty,
                                      i16Ty,   i16Ty,   i16Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__DechannelNorm", i8PtrTy, argTypes);

    // Convert operands
    Value src = adaptor.getSrc();
    // Need to bitcast src to i8*
    src = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, src);
    Value dst = adaptor.getDst();
    // Need to bitcast dst to i8*
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    // Convert shape attribute to Value
    Value shape_n =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[0]);
    Value shape_h =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[1]);
    Value shape_w =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[2]);
    Value shape_c =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), i16Ty, op.getShape()[3]);

    // Convert c0_align attribute to Value
    Value c0Align = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getC0Align()));

    // Convert dtype_size attribute to Value
    Value dtypeSize = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI32IntegerAttr(op.getDtypeSize()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__DechannelNorm", // funcPtr,
        ArrayRef<Value>{src, dst, shape_n, shape_h, shape_w, shape_c, c0Align,
                        dtypeSize});

    // Erase the old op
    rewriter.eraseOp(op);

    return success();
  }
};

// Convert tx81.gemm to LLVM call to __Gemm function
struct GemmOpConversion : public OpConversionPattern<tx::GemmOp> {
  using OpConversionPattern<tx::GemmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Gemm runtime function if not already declared
    // Signature: void __Gemm(int64_t* srcA, int64_t *srcB, int64_t * srcBias,
    // int64_t *dst, int32_t *dims, bool enPsum, int64_t *psum, bool enTransA,
    // bool enTransB, int64_t batchSizeA, int64_t batchSizeB, bool enLeakyRelu,
    // bool enBias,bool enNegScale, int64_t *negScale, bool enPosScale, int64_t
    // *posScale, int64_t srcFmt, int64_t dstFmt)
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i64PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i1Ty = rewriter.getI1Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {
        i8PtrTy,  // srcA
        i8PtrTy,  // srcB
        i8PtrTy,  // srcBias
        i8PtrTy,  // dst
        i32PtrTy, // dims
        i1Ty,     // enPsum
        i8PtrTy,  // psum
        i1Ty,     // enTransA
        i1Ty,     // enTransB
        i32Ty,    // batchSizeA
        i32Ty,    // batchSizeB
        i32Ty,    // reluMode
        i1Ty,     // enBias
        i1Ty,     // enNegScale
        i8PtrTy,  // negScale
        i1Ty,     // enPosScale
        i8PtrTy,  // posScale
        i32Ty,    // srcFmt
        i32Ty     // dstFmt
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__Gemm", i8PtrTy, argTypes);

    // Convert operands
    Value srcA = adaptor.getSrcA();
    Value srcB = adaptor.getSrcB();
    Value srcBias = adaptor.getSrcBias();
    Value dst = adaptor.getDst();

    Value psumAddr = adaptor.getPsumAddr();
    Value srcNegScale = adaptor.getSrcNegScale();
    Value srcPosScale = adaptor.getSrcPosScale();

    // Bitcast all pointers to i8*
    srcA = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcA);
    srcB = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcB);
    srcBias = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcBias);
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);
    psumAddr =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, psumAddr);
    srcNegScale =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcNegScale);
    srcPosScale =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, srcPosScale);

    // Handle dims array - need to convert from attribute to runtime array
    auto dimsAttr = op.getDims();
    SmallVector<int32_t, 3> dimsValues;
    for (auto dimAttr : dimsAttr)
      dimsValues.push_back(cast<IntegerAttr>(dimAttr).getInt());

    // Allocate memory for the dims array
    Value rank = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(dimsValues.size()));

    // Use alloc to allocate memory for dims array
    auto dimsArrayI32Ptr = rewriter.create<LLVM::AllocaOp>(
        op.getLoc(), i32PtrTy, rewriter.getI32Type(), rank,
        /*alignment=*/0);

    // Store each dimension in the array
    for (size_t i = 0; i < dimsValues.size(); i++) {
      // Create the index
      Value idx = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), i64Ty, rewriter.getI32IntegerAttr(i));

      // Create GEP to get pointer to array element
      Value elemPtr = rewriter.create<LLVM::GEPOp>(
          op.getLoc(), i64PtrTy, i32Ty, dimsArrayI32Ptr, ArrayRef<Value>{idx});

      // Create the dimension value
      Value dimValue = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(dimsValues[i]));

      // Store the value
      rewriter.create<LLVM::StoreOp>(op.getLoc(), dimValue, elemPtr);
    }

    // Convert boolean attributes
    Value transA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getTransSrcA()));
    Value transB = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getTransSrcB()));
    Value enPSum = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnPsum()));
    Value reluMode = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getReluMode()));
    Value enBias = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnBias()));
    Value enNegScale = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnNegScale()));
    Value enPosScale = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i1Ty, rewriter.getBoolAttr(op.getEnPosScale()));

    // Convert integer attributes
    Value batchA = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getBatchSrcA()));
    Value batchB = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getBatchSrcB()));
    Value srcFmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getSrcFmt()));
    Value dstFmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getDstFmt()));

    // Create the call to __Gemm
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Gemm", // funcPtr,
        ArrayRef<Value>{srcA, srcB, srcBias, dst, dimsArrayI32Ptr, enPSum,
                        psumAddr, transA, transB, batchA, batchB, reluMode,
                        enBias, enNegScale, srcNegScale, enPosScale,
                        srcPosScale, srcFmt, dstFmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

struct SigmoidOpConversion : public OpConversionPattern<tx::Sigmoid> {
  using OpConversionPattern<tx::Sigmoid>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::Sigmoid op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Sigmoid runtime function if not already declared
    // Signature: void __Sigmoid(int64_t* src, int64_t *dst,
    // uint32_t elem_count, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 17> argTypes = {i8PtrTy, i8PtrTy, i32Ty, i16Ty};

    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__Sigmoid", i8PtrTy, argTypes);

    // Convert operands
    Value input = adaptor.getInput();
    Value output = adaptor.getOut();
    Value elemCount = adaptor.getElemCount();

    // Bitcast all pointers to i8*
    input = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, input);
    output = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, output);
    elemCount = castIndexToInt32(rewriter, op.getLoc(), elemCount);

    // Handle format attribute
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Sigmoid", // funcPtr,
        ArrayRef<Value>{input, output, elemCount, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tx81.memset to LLVM call to __Memset function
struct MemsetOpConversion : public OpConversionPattern<tx::MemsetOp> {
  using OpConversionPattern<tx::MemsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tx::MemsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Memset runtime function if not already declared
    // Signature: void* __Memset(void* dst, int64_t value, uint32_t rank,
    //                    int32_t* strides, int32_t* iterations, uint16_t fmt);
    auto i8PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i32PtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i16Ty = rewriter.getI16Type();

    // Types for function declaration
    SmallVector<Type, 6> argTypes = {
        i8PtrTy,  // Spm addr
        i32Ty,    // value
        i32PtrTy, // src_shape array
        i32PtrTy, // src_strides array
        i32Ty,    // rank
        i16Ty     // fmt
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__Memset", i8PtrTy, argTypes);

    // Get operands
    Value dst = adaptor.getTarget();
    dst = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), i8PtrTy, dst);

    Value value = adaptor.getValue();

    // Handle strides and iterations arrays
    // Create arrays for shapes and strides
    Value dstShapeArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstShape(), op);
    Value dstStridesArray = indexValueArrayToInt32ValueArray(
        rewriter, loc, adaptor.getDstStrides(), op);

    // Convert fmt attribute to Value
    Value fmt = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i16Ty, rewriter.getI16IntegerAttr(op.getFmt()));

    Value rank = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(op.getRank()));

    // Create the call to __Memset
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), i8PtrTy, "__Memset", // funcPtr,
        ArrayRef<Value>{dst, value, dstShapeArray, dstStridesArray, rank, fmt});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// Convert tt.get_program_id to LLVM call to __get_pid function
// Think this as Tx81 special action. May can separate to a single pass or use
// tx81.get_program_id op
struct GetProgramIDConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      mlir::triton::getMaxEnumValForProgramIDDim() + 1;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the module for function declarations
    auto module = op->getParentOfType<ModuleOp>();

    // Declare the __Memset runtime function if not already declared
    // Signature: uint32_t __get_pid(uint32_t);
    auto i32Ty = rewriter.getI32Type();

    // Types for function declaration
    SmallVector<Type, 6> argTypes = {
        i32Ty, // x: 0/y: 1/z: 2,
    };

    // Declare the function
    Value funcPtr = triton::utils::declareTx81Function(
        module, rewriter, op.getLoc(), "__get_pid", i32Ty, argTypes);

    // Get operands
    auto axis = (uint32_t)op.getAxis();

    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    // Convert fmt attribute to Value
    Value src = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(axis));

    // Create the call to __Memset
    auto call = rewriter.create<LLVM::CallOp>(op.getLoc(), i32Ty,
                                              "__get_pid", // funcPtr,
                                              ArrayRef<Value>{src});

    // Replace the op with the result of the call
    rewriter.replaceOp(op, call.getResult());

    return success();
  }
};

// The conversion pass
class Tx81ToLLVMPass : public Tx81ToLLVMBase<Tx81ToLLVMPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, tx::Tx81Dialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Setup LLVM lowering options object which should live across the call to
    // applyFull/PartialConversion.
    LowerToLLVMOptions options(context);
    options.useBarePtrCallConv = false;

    // Setup conversion target
    target.addLegalDialect<LLVM::LLVMDialect, memref::MemRefDialect,
                           arith::ArithDialect, scf::SCFDialect,
                           func::FuncDialect, math::MathDialect>();
    // Handle the tx81 op to llvm.call and support kcore load/store op's spm
    // offset
    target.addIllegalDialect<triton::TritonDialect, linalg::LinalgDialect,
                             tensor::TensorDialect, affine::AffineDialect,
                             tx::Tx81Dialect>();

    // Setup rewrite patterns
    RewritePatternSet patterns(context);

    // NOTE: LLVMTypeConverter should be enough for MLIR core dialects.
    LLVMTypeConverter llvmTypeConverter(context, options);

    // Add the Tx81 to LLVM conversion patterns
    // clang-format off
    patterns.add</* INT8 */
                ZeroPointConvertOpConversion<tx::INT8ToFP16Op, int8ToFp16FuncName>,
                ZeroPointConvertOpConversion<tx::INT8ToBF16Op, int8ToBf16FuncName>,
                ZeroPointConvertOpConversion<tx::INT8ToFP32Op, int8ToFp32FuncName>,
                ZeroPointConvertOpConversion<tx::INT8ToTF32Op, int8ToTf32FuncName>,
                /* INT16 */
                 NormalConvertOpConversion<tx::INT16ToFP16Op, int16ToFp16FuncName>,
                 RoundConvertOpConversion<tx::INT16ToBF16Op, int16ToBf16FuncName>,
                 RoundConvertOpConversion<tx::INT16ToFP32Op, int16ToFp32FuncName>,
                 RoundConvertOpConversion<tx::INT16ToTF32Op, int16ToTf32FuncName>,
                 /* INT32 */
                 RoundConvertOpConversion<tx::INT32ToFP16Op, int32ToFp16FuncName>,
                 RoundConvertOpConversion<tx::INT32ToBF16Op, int32ToBf16FuncName>,
                 RoundConvertOpConversion<tx::INT32ToFP32Op, int32ToFp32FuncName>,
                 RoundConvertOpConversion<tx::INT32ToTF32Op, int32ToTf32FuncName>,
                 /* BF16 */
                 NormalConvertOpConversion<tx::BF16ToINT8Op, bf16ToInt8FuncName>,
                 RoundConvertOpConversion<tx::BF16ToINT16Op, bf16ToInt16FuncName>,
                 RoundConvertOpConversion<tx::BF16ToINT32Op, bf16ToInt32FuncName>,
                 NormalConvertOpConversion<tx::BF16ToFP16Op, bf16ToFp16FuncName>,
                 NormalConvertOpConversion<tx::BF16ToFP32Op, bf16ToFp32FuncName>,
                 NormalConvertOpConversion<tx::BF16ToTF32Op, bf16ToTf32FuncName>,
                 /* FP16 */
                 RoundConvertOpConversion<tx::FP16ToINT8Op, fp16ToInt8FuncName>,
                 RoundConvertOpConversion<tx::FP16ToINT16Op,fp16ToInt16FuncName>,
                 RoundConvertOpConversion<tx::FP16ToINT32Op, fp16ToInt32FuncName>,
                 RoundConvertOpConversion<tx::FP16ToBF16Op, fp16ToBf16FuncName>,
                 NormalConvertOpConversion<tx::FP16ToFP32Op, fp16ToFp32FuncName>,
                 NormalConvertOpConversion<tx::FP16ToTF32Op, fp16ToTf32FuncName>,
                 /* FP32 */
                 RoundConvertOpConversion<tx::FP32ToINT8Op, fp32ToInt8FuncName>,
                 RoundConvertOpConversion<tx::FP32ToINT16Op, fp32ToInt16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToINT32Op, fp32ToInt32FuncName>,
                 RoundConvertOpConversion<tx::FP32ToFP16Op, fp32ToFp16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToBF16Op,fp32ToBf16FuncName>,
                 RoundConvertOpConversion<tx::FP32ToTF32Op, fp32ToTf32FuncName>,
                 /* TF32 */
                 RoundConvertOpConversion<tx::TF32ToINT8Op, tf32ToInt8FuncName>,
                 RoundConvertOpConversion<tx::TF32ToINT16Op, tf32ToInt16FuncName>,
                 RoundConvertOpConversion<tx::TF32ToINT32Op, tf32ToInt32FuncName>,
                 NormalConvertOpConversion<tx::TF32ToFP16Op, tf32ToFp16FuncName>,
                 RoundConvertOpConversion<tx::TF32ToBF16Op, tf32ToBf16FuncName>,
                 NormalConvertOpConversion<tx::TF32ToFP32Op, tf32ToFp32FuncName>,
                 /* MXFP */
                 NormalConvertOpConversion<tx::FP8E4M3ToBF16Op, fp8E4M3ToBF16FuncName>,
                 NormalConvertOpConversion<tx::FP8E5M2ToBF16Op, fp8E5M2ToBF16FuncName>,
                 NormalConvertOpConversion<tx::FP4E2M1ToBF16Op, fp4E2M1ToBF16FuncName>,
                 MXFPScaleBF16OpConversion,
                 ArgMinMaxOpConversion<tx::ArgMaxOp, argMaxFuncName>,
                 ArgMinMaxOpConversion<tx::ArgMinOp, argMinFuncName>,
                 ReduceOpConversion<tx::ReduceSumOp,reduceSumFuncName>,
                 ReduceOpConversion<tx::ReduceMaxOp,reduceMaxFuncName>,
                 ReduceOpConversion<tx::ReduceMinOp,reduceMinFuncName>,
                 ElementWiseOpConversion<tx::AddVVOp, addVVFuncName>,
                 ElementWiseOpConversion<tx::SubVVOp, subVVFuncName>,
                 ElementWiseOpConversion<tx::MulVVOp, mulVVFuncName>,
                 ElementWiseOpConversion<tx::DivVVOp, divVVFuncName>,
                 ElementWiseOpConversion<tx::MaxVVOp, MaxVVFuncName>,
                 ElementWiseOpConversion<tx::MinVVOp, MinVVFuncName>,
                 UnaryOpConversion<tx::AbsVVOp, absVVFuncName>,
                 UnaryOpConversion<tx::RsqrtVVOp, rsqrtVVFuncName>,
                 UnaryOpConversion<tx::SqrtVVOp, sqrtVVFuncName>,
                 UnaryOpConversion<tx::RecipVVOp, recipVVFuncName>,
                 UnaryOpConversion<tx::LnOp, lnFuncName>,
                 UnaryOpConversion<tx::Log2Op, log2FuncName>,
                 UnaryOpConversion<tx::ExpOp, expFuncName>,
                 UnaryOpConversion<tx::Pow2Op, pow2FuncName>,
                 UnaryOpConversion<tx::SinOp, sinFuncName>,
                 UnaryOpConversion<tx::CosOp, cosFuncName>,
                 UnaryOpConversion<tx::Tanh, tanhFuncName>,
                 BinaryVSOpConversion<tx::AddVSOp, addVSFuncName>,
                 BinaryVSOpConversion<tx::SubVSOp, subVSFuncName>,
                 BinaryVSOpConversion<tx::MulVSOp, mulVSFuncName>,
                 BinaryVSOpConversion<tx::DivVSOp, divVSFuncName>,
                 RelationVVOpConversion<tx::BoolEqualVV, boolEqualVVFuncName>,
                 RelationVVOpConversion<tx::BoolUnEqualVV, boolUnEqualVVFuncName>,
                 RelationVVOpConversion<tx::BoolGreaterEqualVV, boolGreaterEqualVVFuncName>,
                 RelationVVOpConversion<tx::BoolGreaterVV, boolGreaterVVFuncName>,
                 RelationVVOpConversion<tx::BoolLessEqualVV, boolLessEqualVVFuncName>,
                 RelationVVOpConversion<tx::BoolLessThenVV, boolLessThenVVFuncName>,
                 RelationVVOpConversion<tx::EqualVV, equalVVFuncName>,
                 RelationVVOpConversion<tx::UnEqualVV, unEqualVVFuncName>,
                 RelationVVOpConversion<tx::GreaterEqualVV, greaterEqualVVFuncName>,
                 RelationVVOpConversion<tx::GreaterVV, greaterVVFuncName>,
                 RelationVVOpConversion<tx::LessEqualVV, lessEqualVVFuncName>,
                 RelationVVOpConversion<tx::LessThenVV, lessThenVSFuncName>,
                 RelationVSOpConversion<tx::BoolEqualVS, boolEqualVSFuncName>,
                 RelationVSOpConversion<tx::BoolUnEqualVS, boolUnEqualVSFuncName>,
                 RelationVSOpConversion<tx::BoolGreaterEqualVS, boolGreaterEqualVSFuncName>,
                 RelationVSOpConversion<tx::BoolGreaterVS, boolGreaterVSFuncName>,
                 RelationVSOpConversion<tx::BoolLessEqualVS, boolLessEqualVSFuncName>,
                 RelationVSOpConversion<tx::BoolLessThenVS, boolLessThenVSFuncName>,
                 RelationVSOpConversion<tx::EqualVS,equalVSFuncName>,
                 RelationVSOpConversion<tx::UnEqualVS,unEqualVSFuncName>,
                 RelationVSOpConversion<tx::GreaterEqualVS,greaterEqualVSFuncName>,
                 RelationVSOpConversion<tx::GreaterVS,greaterVSFuncName>,
                 RelationVSOpConversion<tx::LessEqualVS,lessEqualVSFuncName>,
                 RelationVSOpConversion<tx::LessThenVS,lessThenVSFuncName>,
                 BinaryLogicVVOpConversion<tx::AndVV, andVVFuncName>,
                 BinaryLogicVVOpConversion<tx::OrVV, orVVFuncName>,
                 BinaryLogicVVOpConversion<tx::XorVV, xorVVFuncName>,
                 UnaryBoolLogicVOpConversion<tx::BoolNotV, boolNotVFuncName>,
                 BinaryBoolLogicVOpConversion<tx::BoolAndV, boolAndVFuncName>,
                 BinaryBoolLogicVOpConversion<tx::BoolXorV, boolXorVFuncName>,
                 BinaryBoolLogicVOpConversion<tx::BoolOrV, boolOrVFuncName>,
                 RdmaWdmaOpConversion<tx::RdmaOp,rdmaFuncName>,
                 RdmaWdmaOpConversion<tx::WdmaOp,wdmaFuncName>,
                 UnaryOpConversion<tx::MemCopyOp, memcpyFuncName>,
                 TransformOpConversion<tx::Transpose,transposeFuncName>,
                 TransformOpConversion<tx::Nchw2nhwc,nchw2nhwcFuncName>,
                 TransformOpConversion<tx::Nhwc2nchw,nhwc2nchwFuncName>,
                 MaskMoveOpConversion,
                 GatherScatterOpConversion,
                 BitToFPOpConversion,
                 ChannelNormOpConversion,
                 DechannelNormOpConversion,
                 GemmOpConversion,
                 SigmoidOpConversion,
                 MemsetOpConversion,
                 GetProgramIDConversion,
                 BarrierConversion>(
        context);
    // clang-format on

    // Add call op conversion
    populateCallOpTypeConversionPattern(patterns, llvmTypeConverter);

    // Add return op conversion
    populateReturnOpTypeConversionPattern(patterns, llvmTypeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTx81ToLLVMPass() {
  return std::make_unique<Tx81ToLLVMPass>();
}
