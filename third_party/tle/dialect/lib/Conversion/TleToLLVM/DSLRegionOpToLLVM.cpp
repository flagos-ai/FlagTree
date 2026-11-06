/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/DSLRegionOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/LogicalResult.h"

namespace {
using namespace mlir;
using namespace mlir::triton;

struct DSLRegionOpConversion : public ConvertOpToLLVMPattern<tle::DSLRegionOp> {
  DSLRegionOpConversion(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::DSLRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  LogicalResult lowerWrapper(tle::DSLRegionOp op, OpAdaptor adaptor,
                             FlatSymbolRefAttr symbol,
                             ConversionPatternRewriter &rewriter) const;
};

struct YieldOpConversion : public ConvertOpToLLVMPattern<tle::YieldOp> {
  YieldOpConversion(LLVMTypeConverter &typeConverter, PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(tle::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

DSLRegionOpConversion::DSLRegionOpConversion(LLVMTypeConverter &typeConverter,
                                             PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult DSLRegionOpConversion::matchAndRewrite(
    tle::DSLRegionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (auto symbol =
          op->getAttrOfType<FlatSymbolRefAttr>("tle_raw.wrapper_symbol"))
    return lowerWrapper(op, adaptor, symbol, rewriter);

  auto newOp = rewriter.cloneWithoutRegions<tle::DSLRegionOp>(op);
  Region &body = op.getBody();
  Region &newBody = newOp.getBody();
  rewriter.inlineRegionBefore(body, newBody, newBody.end());

  if (failed(rewriter.convertRegionTypes(&newBody, *getTypeConverter()))) {
    return rewriter.notifyMatchFailure(op, "could not convert body types");
  }
  newOp->setOperands(adaptor.getOperands());
  for (OpResult result : newOp.getResults()) {
    result.setType(getTypeConverter()->convertType(result.getType()));
  }
  rewriter.replaceOp(op, newOp->getResults());

  return success();
}

// Match wrapper.py's wire types. In particular, never sign-extend i1: true
// must arrive at the C++ bool bit_cast as the byte 1, not 255.
static Type getWrapperABIType(Type type, const LLVMTypeConverter *converter) {
  if (type.isInteger(1))
    return IntegerType::get(type.getContext(), 8);
  if (auto fp = dyn_cast<FloatType>(type)) {
    if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type,
             Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
             Float8E5M2FNUZType>(type))
      return {};
    return IntegerType::get(type.getContext(), fp.getWidth());
  }
  if (isa<IntegerType, triton::PointerType, LLVM::LLVMPointerType>(type))
    return converter->convertType(type);
  return {};
}

static Value convertWrapperValue(Location loc, Value value, Type target,
                                 ConversionPatternRewriter &rewriter) {
  Type source = value.getType();
  if (source == target)
    return value;
  if (source.isInteger(1) && target.isInteger(8))
    return rewriter.create<LLVM::ZExtOp>(loc, target, value);
  // The caller's contract guarantees that returned bool bytes are 0 or 1.
  if (source.isInteger(8) && target.isInteger(1))
    return rewriter.create<LLVM::TruncOp>(loc, target, value);
  return rewriter.create<LLVM::BitcastOp>(loc, target, value);
}

LogicalResult
DSLRegionOpConversion::lowerWrapper(tle::DSLRegionOp op, OpAdaptor adaptor,
                                    FlatSymbolRefAttr symbol,
                                    ConversionPatternRewriter &rewriter) const {
  auto callee =
      SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(op, symbol);
  if (!callee)
    return op.emitOpError("cannot find LLVM raw wrapper ") << symbol;
  if (!op.getBody().empty() || op.getRegionDialect() != "cuda")
    return op.emitOpError("raw wrapper requires an empty CUDA region");
  auto infos = op->getAttrOfType<ArrayAttr>("tle_raw.arg_info_list");
  if (!infos || infos.size() != op.getNumOperands())
    return op.emitOpError("raw wrapper argument metadata must match operands");
  auto functionType = callee.getFunctionType();
  if (functionType.isVarArg())
    return op.emitOpError("raw wrapper must not be variadic");

  // Validate the entire signature before emitting any conversions. Return
  // fields follow operand order, independently of output_operand_indices.
  SmallVector<Type> inputTypes, returnTypes;
  SmallVector<unsigned> counts(op.getNumOperands(), 0);
  SmallVector<StringRef> kinds;
  auto *converter = getTypeConverter();
  for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
    auto info = dyn_cast<StringAttr>(infos[i]);
    if (!info)
      return op.emitOpError("raw wrapper argument kind must be a string at ")
             << i;
    StringRef kind = info.getValue();
    kinds.push_back(kind);
    if (kind.starts_with("tensor_fragment_")) {
      auto tensor = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensor || !tensor.getEncoding())
        return op.emitOpError(
                   "raw wrapper fragment requires a tensor layout at operand ")
               << i;
      Type wire = getWrapperABIType(tensor.getElementType(), converter);
      if (!wire)
        return op.emitOpError(
                   "unsupported raw wrapper element type at operand ")
               << i;
      counts[i] = triton::gpu::getTotalElemsPerThread(tensor);
      inputTypes.append(counts[i], wire);
      if (kind == "tensor_fragment_inout")
        returnTypes.append(counts[i], wire);
    } else if (auto memdesc =
                   dyn_cast<triton::gpu::MemDescType>(operand.getType())) {
      if (!isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
              memdesc.getMemorySpace()))
        return op.emitOpError("raw wrapper only supports shared memory "
                              "descriptors at operand ")
               << i;
      Type pointer = LLVM::LLVMPointerType::get(op.getContext(), 3);
      inputTypes.append(2, pointer);
      inputTypes.append(1 + 2 * memdesc.getRank(), rewriter.getI64Type());
    } else {
      Type wire = getWrapperABIType(operand.getType(), converter);
      if (!wire)
        return op.emitOpError("unsupported raw wrapper scalar type at operand ")
               << i;
      inputTypes.push_back(wire);
    }
  }
  if (functionType.getNumParams() != inputTypes.size())
    return op.emitOpError("raw wrapper parameter count mismatch: expected ")
           << inputTypes.size() << ", got " << functionType.getNumParams();
  for (auto [i, expected] : llvm::enumerate(inputTypes)) {
    if (functionType.getParamType(i) != expected)
      return op.emitOpError(
                 "raw wrapper parameter type mismatch at expanded parameter ")
             << i << ": expected " << expected << ", got "
             << functionType.getParamType(i);
  }
  Type returnType = functionType.getReturnType();
  if (returnTypes.empty()) {
    if (!isa<LLVM::LLVMVoidType>(returnType))
      return op.emitOpError(
          "raw wrapper without fragment outputs must return void");
  } else {
    auto structure = dyn_cast<LLVM::LLVMStructType>(returnType);
    if (!structure || structure.isOpaque() ||
        structure.getBody().size() != returnTypes.size())
      return op.emitOpError(
                 "raw wrapper return struct field count mismatch: expected ")
             << returnTypes.size();
    for (auto [i, expected] : llvm::enumerate(returnTypes)) {
      if (structure.getBody()[i] != expected)
        return op.emitOpError("raw wrapper return type mismatch at field ")
               << i << ": expected " << expected << ", got "
               << structure.getBody()[i];
    }
  }

  Location loc = op.getLoc();
  SmallVector<Value> arguments;
  for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
    Value converted = adaptor.getOperands()[i];
    if (kinds[i].starts_with("tensor_fragment_")) {
      auto elements = unpackLLElements(loc, converted, rewriter);
      if (elements.size() != counts[i])
        return op.emitOpError(
                   "raw wrapper fragment register count mismatch at operand ")
               << i;
      for (Value element : elements) {
        Type wire = inputTypes[arguments.size()];
        arguments.push_back(convertWrapperValue(loc, element, wire, rewriter));
      }
    } else if (auto memdesc =
                   dyn_cast<triton::gpu::MemDescType>(operand.getType())) {
      Type pointer = inputTypes[arguments.size()];
      arguments.push_back(
          rewriter.create<tle::ExtractAllocatedPtrOp>(loc, pointer, operand));
      arguments.push_back(
          rewriter.create<tle::ExtractAlignedPtrOp>(loc, pointer, operand));
      arguments.push_back(rewriter.create<tle::ExtractOffsetOp>(loc, operand));
      auto sizes =
          rewriter.create<tle::ExtractSizesOp>(loc, memdesc.getRank(), operand);
      auto strides = rewriter.create<tle::ExtractStridesOp>(
          loc, memdesc.getRank(), operand);
      llvm::append_range(arguments, sizes.getResults());
      llvm::append_range(arguments, strides.getResults());
    } else {
      Type wire = inputTypes[arguments.size()];
      arguments.push_back(convertWrapperValue(loc, converted, wire, rewriter));
    }
  }
  auto call = rewriter.create<LLVM::CallOp>(loc, callee, arguments);
  call.setAlwaysInline(true);
  call.setCConv(callee.getCConv());
  if (auto attrs = callee.getArgAttrsAttr())
    call.setArgAttrsAttr(attrs);
  if (auto attrs = callee.getResAttrsAttr())
    call.setResAttrsAttr(attrs);

  SmallVector<Value> updated(adaptor.getOperands());
  unsigned field = 0;
  for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
    if (kinds[i] != "tensor_fragment_inout")
      continue;
    auto tensor = cast<RankedTensorType>(operand.getType());
    Type elementType = converter->convertType(tensor.getElementType());
    SmallVector<Value> elements;
    for (unsigned j = 0; j < counts[i]; ++j) {
      Value value = rewriter.create<LLVM::ExtractValueOp>(
          loc, call.getResult(),
          ArrayRef<int64_t>{static_cast<int64_t>(field++)});
      elements.push_back(
          convertWrapperValue(loc, value, elementType, rewriter));
    }
    updated[i] = packLLElements(loc, converter, elements, rewriter, tensor);
  }
  SmallVector<Value> results;
  for (int32_t idx : op.getOutputOperandIndices())
    results.push_back(updated[idx]);
  rewriter.replaceOp(op, results);
  return success();
}

YieldOpConversion::YieldOpConversion(LLVMTypeConverter &typeConverter,
                                     PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult
YieldOpConversion::matchAndRewrite(tle::YieldOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<tle::YieldOp>(op, adaptor.getOperands());
  return success();
}

void tle::populateDSLRegionOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DSLRegionOpConversion, YieldOpConversion>(typeConverter,
                                                         benefit);
}
