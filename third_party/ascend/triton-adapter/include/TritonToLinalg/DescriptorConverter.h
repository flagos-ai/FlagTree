#ifndef TRITON_ADAPTER_DESCRIPTORCONVERTER_H
#define TRITON_ADAPTER_DESCRIPTORCONVERTER_H

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace DescriptorConverter {
using namespace mlir;
using namespace triton;

struct Descriptor {
    Value base;
    SmallVector<Value> shape;
    SmallVector<Value> strides;
};

bool hasATensorDescriptorType(mlir::TypeRange types);

class DescriptorLoadConverter : public OpConversionPattern<triton::DescriptorLoadOp> {
public:
    using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

class DescriptorStoreConverter : public OpConversionPattern<triton::DescriptorStoreOp> {
public:
    using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;
};

} // end of namespace DescriptorConverter

#endif // TRITON_ADAPTER_DESCRIPTORCONVERTER_H
