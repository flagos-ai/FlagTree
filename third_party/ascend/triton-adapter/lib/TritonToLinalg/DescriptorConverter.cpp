#include "TritonToLinalg/DescriptorConverter.h"
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToLinalg/TritonOpConverter.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <utility>
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace DescriptorConverter {
using namespace mlir;
using namespace triton;

bool hasATensorDescriptorType(mlir::TypeRange types)
{
    return llvm::any_of(types, [](mlir::Type t) { return llvm::isa<mlir::triton::TensorDescType>(t); });
}

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute> filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs)
{
    mlir::SmallVector<NamedAttribute> ret;
    llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
        auto attrName = attr.getName().getValue();
        return attrName != "operandSegmentSizes";
    });
    return ret;
}

Descriptor unpackDescriptor(TensorDescType type, Value desc, ConversionPatternRewriter &rewriter)
{
    auto makeDescOp = desc.getDefiningOp<triton::MakeTensorDescOp>();
    assert(makeDescOp && "Descriptor must be defined by MakeTensorDescOp");

    Descriptor res;

    // 直接回溯处理的 tt.make_tensor_descriptor
    res.base = makeDescOp.getBase();
    for (auto s : makeDescOp.getShape()) {
        res.shape.push_back(rewriter.createOrFold<arith::ExtSIOp>(makeDescOp.getLoc(), rewriter.getI64Type(), s));
    }
    for (auto st : makeDescOp.getStrides()) {
        res.strides.push_back(rewriter.createOrFold<arith::ExtSIOp>(makeDescOp.getLoc(), rewriter.getI64Type(), st));
    }

    return res;
}

SmallVector<int32_t> computeOrder(ArrayRef<int64_t> shape)
{
    SmallVector<int32_t> order;
    int rank = shape.size();
    order.reserve(rank);
    // 默认采用逆序 [dims - 1, ..., 0]
    for (int i = rank - 1; i >= 0; --i) {
        order.push_back(i);
    }
    return order;
}

LogicalResult DescriptorLoadConverter::matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                                                       ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto indices = op.getIndices();

    // 1. 解包 descriptor
    auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

    // 2. 新增 make_tensor_ptr
    SmallVector<int32_t> tensorShapeValues;
    for (auto dim : blockShape) {
        tensorShapeValues.push_back(static_cast<int32_t>(dim));
    }
    Value tensorPtr = rewriter.create<triton::MakeTensorPtrOp>(loc,
                                                               desc.base,               // 基址
                                                               desc.shape,              // 形状
                                                               desc.strides,            // 步长
                                                               indices,                 // 偏移
                                                               tensorShapeValues,       // tensorShape
                                                               computeOrder(blockShape) // 使用动态计算的 order
    );
    // 3. 替换 tt.load 操作
    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(op, descTy.getSignlessBlockType(), tensorPtr);

    // 保留原始操作的其他属性
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    return success();
}

LogicalResult DescriptorStoreConverter::matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                                                        ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto indices = op.getIndices();

    // 1. 解包 descriptor
    auto desc = unpackDescriptor(descTy, adaptor.getDesc(), rewriter);

    // 2. 新增 make_tensor_ptr
    SmallVector<int32_t> tensorShapeValues;
    for (auto dim : blockShape) {
        tensorShapeValues.push_back(static_cast<int32_t>(dim));
    }
    Value tensorPtr = rewriter.create<triton::MakeTensorPtrOp>(loc,
                                                               desc.base,               // 基址
                                                               desc.shape,              // 形状
                                                               desc.strides,            // 步长
                                                               indices,                 // 偏移
                                                               tensorShapeValues,       // tensorShape
                                                               computeOrder(blockShape) // 使用动态计算的 order
    );

    // 3. 替换 tt.store 操作
    Value valueToStore = adaptor.getSrc();

    auto maskType = RankedTensorType::get(blockShape, rewriter.getI1Type());
    rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(maskType, true));

    // 创建属性
    auto boundaryCheck = rewriter.getDenseI32ArrayAttr({}); // 空的边界检查
    auto cacheModifier = triton::CacheModifierAttr::get(rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictionPolicy = triton::EvictionPolicyAttr::get(rewriter.getContext(), triton::EvictionPolicy::NORMAL);

    // 创建 store 操作并替换原始操作
    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(op,            // 要替换的操作
                                                                 tensorPtr,     // 指针
                                                                 valueToStore,  // 要存储的值
                                                                 nullptr,       // 掩码
                                                                 boundaryCheck, // 边界检查
                                                                 cacheModifier, // 缓存修饰符
                                                                 evictionPolicy // 驱逐策略
    );

    // 保留原始操作的其他属性
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));
    return success();
}

} // namespace DescriptorConverter
