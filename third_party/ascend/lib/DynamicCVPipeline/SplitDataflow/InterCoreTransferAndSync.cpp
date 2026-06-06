/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.h"

#include <memory>
#include <optional>

#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "Utils/Utils.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "inter-core-transfer-and-sync";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace hivm;

// Attribute name constants
static constexpr const char *kBlockIdAttr = "ssbuffer.block_id";
static constexpr const char *kCoreTypeAttr = "ssbuffer.core_type";
static constexpr const char *kTransferIdAttr = "ssbuffer.transfer_id";

static constexpr int kIntegerBitWidth = 32;
static constexpr int NzDimWidth = 16;

static uint64_t getElemBytesForAlign(Type t)
{
    static constexpr uint64_t kBitsPerByte = 8;
    if (auto ft = dyn_cast<FloatType>(t)) {
        return (uint64_t)((ft.getWidth() + kBitsPerByte - 1) / kBitsPerByte);
    }
    if (auto it = dyn_cast<IntegerType>(t)) {
        return (uint64_t)((it.getWidth() + kBitsPerByte - 1) / kBitsPerByte);
    }
    if (isa<IndexType>(t)) {
        return 8ULL;
    }
    if (auto ct = dyn_cast<ComplexType>(t)) {
        return 2ULL * getElemBytesForAlign(ct.getElementType());
    }
    return 0ULL;
}

static uint64_t getBlockElemsFor32BAlign(Type elemType)
{
    constexpr uint64_t kAlignBytes = 32;
    uint64_t elemBytes = getElemBytesForAlign(elemType);
    if (elemBytes == 0) {
        return -1;
    }
    if (elemBytes < 0 || kAlignBytes % elemBytes != 0) {
        return -1;
    }
    if (elemBytes >= kAlignBytes) {
        return 1;
    }
    return kAlignBytes / elemBytes;
}

static void attachCommonTags(Operation *op, int blockId, StringRef coreType)
{
    MLIRContext *ctx = op->getContext();
    op->setAttr(kBlockIdAttr, IntegerAttr::get(IntegerType::get(ctx, kIntegerBitWidth), blockId));
    op->setAttr(kCoreTypeAttr, StringAttr::get(ctx, coreType));
}

static void attachTransferTags(Operation *op, int blockId, StringRef coreType, int transferId)
{
    MLIRContext *ctx = op->getContext();
    op->setAttr(kBlockIdAttr, IntegerAttr::get(IntegerType::get(ctx, kIntegerBitWidth), blockId));
    op->setAttr(kCoreTypeAttr, StringAttr::get(ctx, coreType));
    op->setAttr(kTransferIdAttr, IntegerAttr::get(IntegerType::get(ctx, kIntegerBitWidth), transferId));
}

// Block Start/End Operation Retrieval
std::pair<mlir::Operation *, mlir::Operation *> InterCoreTransferAndSyncPass::getBlockStartEnd(int targetId,
    mlir::ModuleOp module)
{
    mlir::Operation *knownOpInBlock = nullptr;
    module.walk([&](mlir::Operation *op) {
        if (knownOpInBlock) {
            return;
        }
        if (getSsbufferBlockId(op) == targetId) {
            knownOpInBlock = op;
        }
    });

    if (!knownOpInBlock) {
        return { nullptr, nullptr };
    }

    mlir::Block *block = knownOpInBlock->getBlock();
    if (!block)
        return { nullptr, nullptr };

    mlir::Operation *start = nullptr;
    mlir::Operation *end = nullptr;

    // Iterate through all operations in the current block
    for (Operation &op : *block) {
        int blockId = getSsbufferBlockId(&op);
        if (blockId == -1) {
            continue;
        }

        if (!start) {
            if (targetId == blockId) {
                start = &op;
                end = &op;
            }
        } else {
            if (targetId == blockId) {
                end = &op;
            } else {
                break;
            }
        }
    }
    return { start, end };
}

bool InterCoreTransferAndSyncPass::isOuterLayerDependency(size_t depIndex, mlir::Operation *currProdEnd,
    mlir::Operation *currConsStart, llvm::SmallVector<DependencyInfo> &memDependencies)
{
    if (!currProdEnd || !currConsStart) {
        return false;
    }
    mlir::Block *currBlock = currProdEnd->getBlock();
    if (currBlock != currConsStart->getBlock()) {
        return false;
    }
    for (size_t i = 0; i < memDependencies.size(); ++i) {
        if (i == depIndex) {
            continue;
        }
        auto &otherDep = memDependencies[i];

        auto [otherProdStart, otherProdEnd] = getBlockStartEnd(otherDep.producerBlockId, module);
        auto [otherConsStart, otherConsEnd] = getBlockStartEnd(otherDep.consumerBlockId, module);

        if (!otherProdEnd || !otherConsStart) {
            continue;
        }

        if (otherProdEnd->getBlock() != currBlock || otherConsStart->getBlock() != currBlock) {
            continue;
        }

        // otherProdEnd is before currProdEnd
        // AND currConsStart is before otherConsStart
        bool isOtherInsideCurrent =
            !otherProdEnd->isBeforeInBlock(currProdEnd) && !currConsStart->isBeforeInBlock(otherConsStart);

        if (otherProdEnd == currProdEnd && otherConsStart == currConsStart) {
            if (i < depIndex) {
                // if otherDep has smaller index, current dep is outer layer and can be skipped
                return true;
            }
        } else if (isOtherInsideCurrent) {
            return true;
        }
    }

    return false;
}

// Nd2NzNormalizer
SmallVector<int64_t> InterCoreTransferAndSyncPass::computeExpectedShape(Value value)
{
    auto tensorTy = dyn_cast<TensorType>(value.getType());
    static constexpr int NdShapeLength = 2;
    assert(tensorTy && tensorTy.getRank() == NdShapeLength && "source shape is not 2-dim!");

    int64_t M = tensorTy.getDimSize(0);
    int64_t N = tensorTy.getDimSize(1);

    // Compute bit width & Nwidth
    int64_t nWidth = getBlockElemsFor32BAlign(tensorTy.getElementType());
    if (nWidth == 0) {
        LOG_DEBUG("Unsupported element type for 32B alignment.\n");
        return { M, N };
    }
    // Calculate newM / newN using the formula
    int64_t blM = (M + NzDimWidth - 1) / NzDimWidth;
    int64_t newM = blM * NzDimWidth;

    int64_t blN = (N + nWidth - 1) / nWidth;
    int64_t newN = blN * nWidth;

    return { newM, newN }; // Return 2D shape
}

bool InterCoreTransferAndSyncPass::isShapeExpected(Value value, SmallVector<int64_t> &expectedShape)
{
    auto tensorTy = dyn_cast<TensorType>(value.getType());
    ArrayRef<int64_t> currShape = tensorTy.getShape();
    return currShape.equals(expectedShape);
}

void InterCoreTransferAndSyncPass::rewriteMatmulWithNewShape(OpBuilder &builder, Operation *matmulOp, Location loc)
{
    int matmulOpBlockId = getSsbufferBlockId(matmulOp);

    Value lhs = matmulOp->getOperands()[0];
    Value rhs = matmulOp->getOperands()[1];
    Value acc = matmulOp->getOperands()[2];
    Value originalResult = matmulOp->getResult(0);
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto accType = dyn_cast<RankedTensorType>(acc.getType());
    auto resType = dyn_cast<RankedTensorType>(originalResult.getType());
    ArrayRef<int64_t> accshape = accType.getShape();
    ArrayRef<int64_t> resshape = resType.getShape();
    SmallVector<int64_t> expectedShape = { lhsType.getShape()[0], rhsType.getShape()[1] };
    auto expectedType = RankedTensorType::get(expectedShape, resType.getElementType());

    builder.setInsertionPoint(matmulOp);

    auto floatElemTy = cast<FloatType>(resType.getElementType());
    auto zeroConstOp =
        builder.create<arith::ConstantFloatOp>(loc, floatElemTy, APFloat::getZero(floatElemTy.getFloatSemantics()));
    auto tensorEmptyOp = builder.create<tensor::EmptyOp>(loc, expectedShape, resType.getElementType());
    auto linalgFillOp = builder.create<linalg::FillOp>(loc, zeroConstOp.getResult(), tensorEmptyOp.getResult());

    attachCommonTags(zeroConstOp, matmulOpBlockId, "CUBE");
    attachCommonTags(tensorEmptyOp, matmulOpBlockId, "CUBE");
    attachCommonTags(linalgFillOp, matmulOpBlockId, "CUBE");

    Value newAccResult = linalgFillOp->getResult(0);

    builder.setInsertionPointAfter(matmulOp);
    static constexpr int accIndex = 2;
    matmulOp->setOperand(accIndex, newAccResult);
    matmulOp->getResult(0).setType(expectedType);
    auto newMatmulOp = dyn_cast<linalg::MatmulOp>(matmulOp);
    Value newMatmulResult = newMatmulOp->getResult(0);
    LOG_DEBUG("newmatmulOp" << newMatmulOp << "\n");
    SmallVector<OpFoldResult> offsets = { builder.getIndexAttr(0), builder.getIndexAttr(0) };
    SmallVector<OpFoldResult> strides = { builder.getIndexAttr(1), builder.getIndexAttr(1) };
    SmallVector<OpFoldResult> sizes = { builder.getIndexAttr(accType.getShape()[0]),
        builder.getIndexAttr(accType.getShape()[1]) };
    auto extractSliceOp = builder.create<tensor::ExtractSliceOp>(loc, newMatmulResult, offsets, sizes, strides);
    attachCommonTags(extractSliceOp, matmulOpBlockId, "CUBE");

    originalResult.replaceUsesWithIf(extractSliceOp.getResult(),
        [&](OpOperand &use) { return use.getOwner() != extractSliceOp.getOperation(); });
    LOG_DEBUG("cubeValueMapping[originalResult]" << originalResult << "\n");
    LOG_DEBUG("cubeValueMapping[originalResult]extractSliceOp.getResult()   " << extractSliceOp.getResult() << "\n");
    cubeValueMapping[originalResult] = extractSliceOp.getResult();
}

void InterCoreTransferAndSyncPass::rewriteTransposeWithNewShape(OpBuilder &builder, Operation *transposeOp,
    Location loc)
{
    Value inputvalue = transposeOp->getOperands()[0];
    Value outputvalue = transposeOp->getOperands()[0];

    auto inputTy = dyn_cast<RankedTensorType>(inputvalue.getType());
    Type elemType = inputTy.getElementType();
    SmallVector<int64_t> newOutputShape = { inputTy.getShape()[1], inputTy.getShape()[0] };
    auto expectedType = RankedTensorType::get(newOutputShape, elemType);
    // Create new empty tensor with new shape
    auto tensorEmptyOp = builder.create<tensor::EmptyOp>(loc, newOutputShape, elemType);
    attachCommonTags(tensorEmptyOp, getSsbufferBlockId(transposeOp), "CUBE");
    Value transposeOpResult = transposeOp->getResult(0);
    transposeOp->setOperand(1, tensorEmptyOp.getResult());
    transposeOp->getResult(0).setType(expectedType);
}

// padding v->c tensor
mlir::Value InterCoreTransferAndSyncPass::normalizeIfNeeded(OpBuilder &builder, DependencyInfo &dep, Location loc,
    mlir::Value origValue, SmallVector<int64_t> expectedShape, int originBlockId)
{
    auto origTensorType = dyn_cast<RankedTensorType>(origValue.getType());

    int64_t iniM = origTensorType.getDimSize(0);
    int64_t iniN = origTensorType.getDimSize(1);
    Type elemType = origTensorType.getElementType();

    builder.setInsertionPointAfter(origValue.getDefiningOp());

    auto floatElemTy = cast<FloatType>(elemType);
    auto zeroConstOp =
        builder.create<arith::ConstantFloatOp>(loc, floatElemTy, APFloat::getZero(floatElemTy.getFloatSemantics()));
    auto tensorEmptyOp = builder.create<tensor::EmptyOp>(loc, expectedShape, elemType);
    auto linalgFillOp = builder.create<linalg::FillOp>(loc, zeroConstOp.getResult(), tensorEmptyOp.getResult());
    SmallVector<OpFoldResult> offsets = { builder.getIndexAttr(0), builder.getIndexAttr(0) };
    SmallVector<OpFoldResult> insertsizes = { builder.getIndexAttr(iniM), builder.getIndexAttr(iniN) };
    SmallVector<OpFoldResult> strides = { builder.getIndexAttr(1), builder.getIndexAttr(1) };
    auto tensorInsertSliceOp = builder.create<tensor::InsertSliceOp>(loc, origValue, linalgFillOp->getResult(0),
        offsets, insertsizes, strides);

    attachCommonTags(zeroConstOp, originBlockId, "VECTOR");
    attachCommonTags(tensorEmptyOp, originBlockId, "VECTOR");
    attachCommonTags(linalgFillOp, originBlockId, "VECTOR");
    attachCommonTags(tensorInsertSliceOp, originBlockId, "VECTOR");
    int cId = dep.iniConsumerBlockId;
    LOG_DEBUG("int cId = dep.iniConsumerBlockId;" << cId << "\n");
    for (Operation *user : origValue.getUsers()) {
        LOG_DEBUG(*user << "\n");
        int userBlockId = getSsbufferBlockId(user);
        LOG_DEBUG("int userBlockId = getSsbufferBlockId(user);" << userBlockId << "\n");
        if (userBlockId == -1 || userBlockId != cId) {
            continue;
        }
        user->replaceUsesOfWith(origValue, tensorInsertSliceOp.getResult());
        if (auto matmulOp = dyn_cast<linalg::MatmulOp>(user)) {
            rewriteMatmulWithNewShape(builder, matmulOp, loc);
            continue;
        }
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(user)) {
            LOG_DEBUG("before rewriteTransposeWithNewShape\n");
            rewriteTransposeWithNewShape(builder, transposeOp, loc);
            LOG_DEBUG("after rewriteTransposeWithNewShape\n");
            for (Operation *transposeuser : transposeOp->getUsers()) {
                auto matmulOp = dyn_cast<linalg::MatmulOp>(transposeuser);
                if (matmulOp && getSsbufferBlockId(matmulOp) == cId) {
                    rewriteMatmulWithNewShape(builder, matmulOp, loc);
                }
            }
        }
    }
    cubeValueMapping[origValue] = tensorInsertSliceOp.getResult();
    return tensorInsertSliceOp.getResult();
}

void InterCoreTransferAndSyncPass::Nd2NzNormalize(OpBuilder &builder, DependencyInfo &dep, Location loc)
{
    Value origValue = dep.value;
    Value newValue = origValue;
    // Step 0: Check if this Value has already been processed
    auto it = vecValueMapping.find(origValue);
    if (it != vecValueMapping.end()) {
        return;
    }
    // Step 1: Compute expected shape
    SmallVector<int64_t> expectedShape = computeExpectedShape(origValue);
    int originBlockId = getSsbufferBlockId(origValue.getDefiningOp());
    // Step 2: If shapes match, return original value
    if (!isShapeExpected(origValue, expectedShape)) {
        newValue = normalizeIfNeeded(builder, dep, loc, origValue, expectedShape, originBlockId);
    }
    // Step 3: insert nd2nz
    auto srcTensorType = cast<RankedTensorType>(newValue.getType());
    int64_t M = srcTensorType.getDimSize(0);
    int64_t N = srcTensorType.getDimSize(1);
    Type elemType = srcTensorType.getElementType();

    int64_t blk = getBlockElemsFor32BAlign(elemType);
    if (blk == 0) {
        LOG_DEBUG("Invalid block size.\n");
        return;
    }

    SmallVector<int64_t> shape3D = { M, N / blk, blk };
    SmallVector<int64_t> shapeTrans = { N / blk, M, blk };
    SmallVector<int64_t> shapeFinal = { N / blk, M / NzDimWidth, NzDimWidth, blk };

    auto type3D = RankedTensorType::get(shape3D, elemType);
    auto typeTrans = RankedTensorType::get(shapeTrans, elemType);
    auto typeFinal = RankedTensorType::get(shapeFinal, elemType);
    builder.setInsertionPointAfter(newValue.getDefiningOp());
    auto reshape3Dcst = builder.create<arith::ConstantOp>(loc, builder.getI64TensorAttr(shape3D));
    auto reshape3DOp = builder.create<tensor::ReshapeOp>(loc, type3D, newValue, reshape3Dcst);

    auto emptyTrans = builder.create<tensor::EmptyOp>(loc, shapeTrans, elemType);
    SmallVector<int64_t> transposeOrder = { 1, 0, 2 };
    auto transposeOp =
        builder.create<linalg::TransposeOp>(loc, reshape3DOp.getResult(), emptyTrans.getResult(), transposeOrder);
    auto reshape4Dcst = builder.create<arith::ConstantOp>(loc, builder.getI64TensorAttr(shapeFinal));
    auto reshape4DOp = builder.create<tensor::ReshapeOp>(loc, typeFinal, transposeOp->getResult(0), reshape4Dcst);

    attachCommonTags(reshape3Dcst, originBlockId, "VECTOR");
    attachCommonTags(reshape3DOp, originBlockId, "VECTOR");
    attachCommonTags(emptyTrans, originBlockId, "VECTOR");
    attachCommonTags(transposeOp, originBlockId, "VECTOR");
    attachCommonTags(reshape4Dcst, originBlockId, "VECTOR");
    attachCommonTags(reshape4DOp, originBlockId, "VECTOR");
    LOG_DEBUG("[reshape3DOp]: " << *reshape3DOp << "\n");
    LOG_DEBUG("[transposeOp]: " << *transposeOp << "\n");
    LOG_DEBUG("[reshape4DOp]: " << *reshape4DOp << "\n");
    vecValueMapping[origValue] = reshape4DOp.getResult();
}

// mark memref.alloc
mlir::Operation *InterCoreTransferAndSyncPass::annotateTightlyCoupledBuffer(OpBuilder &builder, Operation *allocOp,
    Location loc)
{
    builder.setInsertionPointAfter(allocOp);
    auto markAllocOp = builder.create<annotation::MarkOp>(loc, allocOp->getResult(0));
    auto writeAttr = builder.getStringAttr("write");
    auto readAttr = builder.getStringAttr("read");
    auto effectsAttr = builder.getArrayAttr({ writeAttr, readAttr });
    markAllocOp->setAttr("effects", effectsAttr);
    markAllocOp->setAttr(hivm::HIVMTightlyCoupledBufferAttr::name,
        HIVMTightlyCoupledBufferAttr::get(builder.getContext(), markAllocIndex));
    return markAllocOp;
}

// find the insert point for memref.alloc
Operation *InterCoreTransferAndSyncPass::findMainLoopforTransfer(Operation *endOp, Operation *startOp)
{
    Operation *lca = endOp->getParentOp();
    assert(lca == startOp->getParentOp() && "startOp and endOp are not in the same parent block, which is unexpected.");
    Operation *current = lca;
    while (current) {
        if (isa<scf::ForOp>(current)) {
            return current;
        }
        current = current->getParentOp();
    }
    return nullptr;
}

std::pair<Operation *, Operation *> InterCoreTransferAndSyncPass::createTransferAllocs(OpBuilder &builder, Location loc,
    ArrayRef<int64_t> shape, Type elemType, hivm::AddressSpace addrSpace, Operation *prodEndOp, Operation *consStartOp,
    int prodBlockId, int consBlockId, StringRef prodTag, StringRef consTag, int transferIndex)
{
    auto addressSpaceAttr = builder.getAttr<hivm::AddressSpaceAttr>(addrSpace);
    auto allocType = MemRefType::get(shape, elemType, nullptr, addressSpaceAttr);

    Operation *prodAllocOp = nullptr;
    Operation *consAllocOp = nullptr;

    Operation *mainLoopOp = findMainLoopforTransfer(prodEndOp, consStartOp);

    if (mainLoopOp) {
        builder.setInsertionPoint(mainLoopOp);
        prodAllocOp = builder.create<memref::AllocOp>(loc, allocType);
        auto markProdOp = annotateTightlyCoupledBuffer(builder, prodAllocOp, loc);
        consAllocOp = builder.create<memref::AllocOp>(loc, allocType);
        auto markConsOp = annotateTightlyCoupledBuffer(builder, consAllocOp, loc);

        int loopBlockId = getSsbufferBlockId(mainLoopOp);
        attachTransferTags(prodAllocOp, loopBlockId, prodTag, transferIndex);
        attachTransferTags(consAllocOp, loopBlockId, consTag, transferIndex);
        attachTransferTags(markProdOp, loopBlockId, prodTag, transferIndex);
        attachTransferTags(markConsOp, loopBlockId, consTag, transferIndex);

        builder.setInsertionPointAfter(prodEndOp);
    } else {
        builder.setInsertionPoint(consStartOp);
        consAllocOp = builder.create<memref::AllocOp>(loc, allocType);
        auto markConsOp = annotateTightlyCoupledBuffer(builder, consAllocOp, loc);

        builder.setInsertionPointAfter(prodEndOp);
        prodAllocOp = builder.create<memref::AllocOp>(loc, allocType);
        auto markProdOp = annotateTightlyCoupledBuffer(builder, prodAllocOp, loc);

        attachTransferTags(prodAllocOp, prodBlockId, prodTag, transferIndex);
        attachTransferTags(consAllocOp, consBlockId, consTag, transferIndex);
        attachTransferTags(markProdOp, prodBlockId, prodTag, transferIndex);
        attachTransferTags(markConsOp, consBlockId, consTag, transferIndex);
    }
    markAllocIndex++;

    return { prodAllocOp, consAllocOp };
}

Operation *InterCoreTransferAndSyncPass::insertVectorToCubeTransfer(OpBuilder &builder, Value srcValue,
    Value normalizedValue, Operation *vectorEndOp, Operation *cubeStartOp, Location loc, int transferIndex,
    int iniConsumerId)
{
    LOG_DEBUG("Inserting [Vector->Cube] transfer for value: " << srcValue << "\n");
    // Step 1: Get input information (2D tensor: MxN)
    auto srcTensorType = cast<RankedTensorType>(srcValue.getType());
    auto normalizedTensorType = cast<RankedTensorType>(normalizedValue.getType());
    Type elemType = srcTensorType.getElementType();

    int vecBlockId = getSsbufferBlockId(vectorEndOp);
    int cubeBlockId = getSsbufferBlockId(cubeStartOp);

    auto [vecAllocOp, cubeAllocOp] = createTransferAllocs(builder, loc, normalizedTensorType.getShape(), elemType,
        hivm::AddressSpace::L1, vectorEndOp, cubeStartOp, vecBlockId, cubeBlockId, "VECTOR", "CUBE", transferIndex);

    auto copyOp = builder.create<hivm::CopyOp>(loc, mlir::TypeRange{}, normalizedValue, vecAllocOp->getResult(0));

    attachTransferTags(copyOp, vecBlockId, "VECTOR", transferIndex);

    LOG_DEBUG("[copyOp]: " << *copyOp << "\n");

    builder.setInsertionPoint(cubeStartOp);

    auto nzLayout = hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::nZ);
    auto ndLayout = hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::ND);
    auto cbufaddressSpaceAttr = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
    auto newAllocType = MemRefType::get(srcTensorType.getShape(), elemType, nullptr, cbufaddressSpaceAttr);
    auto convertLayoutOp = builder.create<hivm::ConvertLayoutOp>(loc, newAllocType, cubeAllocOp->getResult(0),
        nzLayout, // srcLayout
        ndLayout  // dstLayout
    );
    auto plainMemrefType = MemRefType::get(srcTensorType.getShape(), elemType);
    auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(loc, plainMemrefType, convertLayoutOp.getResult());
    auto toTensorOp =
        builder.create<bufferization::ToTensorOp>(loc, srcTensorType, memspaceCastOp.getResult(), true, true);

    attachTransferTags(convertLayoutOp, cubeBlockId, "CUBE", transferIndex);
    attachTransferTags(memspaceCastOp, cubeBlockId, "CUBE", transferIndex);
    attachTransferTags(toTensorOp, cubeBlockId, "CUBE", transferIndex);
    LOG_DEBUG("[toTensorOp]: " << *toTensorOp << "\n");

    for (Operation *user : srcValue.getUsers()) {
        int userBlockId = getSsbufferBlockId(user);
        if (userBlockId == iniConsumerId) {
            user->replaceUsesOfWith(srcValue, toTensorOp.getResult());
        }
    }
    return copyOp;
}

Operation *InterCoreTransferAndSyncPass::insertCubeToVectorTransfer(OpBuilder &builder, Value srcValue,
    Operation *cubeEndOp, Operation *vectorStartOp, Location loc, int transferIndex, int iniConsumerId)
{
    LOG_DEBUG("Inserting [Cube->Vector] transfer for value: " << srcValue << "\n");
    auto srcTensorType = cast<RankedTensorType>(srcValue.getType());
    int64_t M = srcTensorType.getDimSize(0);
    int64_t N = srcTensorType.getDimSize(1);
    Type elemType = srcTensorType.getElementType();

    int cubeBlockId = getSsbufferBlockId(srcValue.getDefiningOp());
    int vecBlockId = getSsbufferBlockId(vectorStartOp);

    auto [cubeAllocOp, vecAllocOp] = createTransferAllocs(builder, loc, { M, N }, elemType, hivm::AddressSpace::UB,
        cubeEndOp, vectorStartOp, cubeBlockId, vecBlockId, "CUBE", "VECTOR", transferIndex);

    FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(builder.getContext(), FixpipeDMAMode::NZ2ND);
    auto fixpipeOp = builder.create<hivm::FixpipeOp>(loc, mlir::TypeRange{}, // No return value
        srcValue,                                                            // src
        cubeAllocOp->getResult(0),                                           // dst
        mlir::ValueRange{}, dmaModeAttr, nullptr, nullptr, nullptr, nullptr, mlir::ArrayAttr{});
    attachTransferTags(fixpipeOp, cubeBlockId, "CUBE", transferIndex);
    LOG_DEBUG("[fixpipeOp]: " << *fixpipeOp << "\n");

    // Vector side: memspace_cast + to_tensor
    builder.setInsertionPoint(vectorStartOp);

    auto plainMemrefType = MemRefType::get({ M, N }, elemType);
    auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(loc, plainMemrefType, vecAllocOp->getResult(0));

    auto toTensorOp =
        builder.create<bufferization::ToTensorOp>(loc, srcTensorType, memspaceCastOp.getResult(), true, true);

    attachTransferTags(memspaceCastOp, vecBlockId, "VECTOR", transferIndex);
    attachTransferTags(toTensorOp, vecBlockId, "VECTOR", transferIndex);
    LOG_DEBUG("[toTensorOp]: " << *toTensorOp << "\n");

    for (Operation *user : srcValue.getUsers()) {
        int userBlockId = getSsbufferBlockId(user);
        if (userBlockId == iniConsumerId) {
            user->replaceUsesOfWith(srcValue, toTensorOp.getResult());
        }
    }
    return fixpipeOp;
}

void InterCoreTransferAndSyncPass::insertInterCoreSync(
    OpBuilder &builder, Operation *transferOp,
    Operation *consumerStartOp, Operation *consumerEndOp,
    int flag, Location loc, int transferIndex)
{
    LOG_DEBUG("Inserting inter-core synchronization for transferOp: " << *transferOp << "\n");
    auto cubeCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
    auto vecCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
    auto pipeFixAttr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
    auto pipeVAttr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);
    auto pipeMte3Attr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE3);
    auto pipeMte1Attr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE1);
    auto pipeMAttr = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_M);
    auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

    int producerBlockId = getSsbufferBlockId(transferOp);
    int consumerBlockId = getSsbufferBlockId(consumerStartOp);

    Operation *mainLoopOp = findMainLoopforTransfer(transferOp, consumerStartOp);

    if (dyn_cast<hivm::FixpipeOp>(transferOp)) {
        builder.setInsertionPointAfter(transferOp);
        auto setOpForRead = builder.create<SyncBlockSetOp>(loc, cubeCoreAttr, pipeFixAttr, pipeVAttr, flagId);
        attachTransferTags(setOpForRead, producerBlockId, "CUBE", transferIndex);
        builder.setInsertionPoint(consumerStartOp);
        auto waitOpForRead = builder.create<SyncBlockWaitOp>(loc, vecCoreAttr, pipeFixAttr, pipeVAttr, flagId);
        attachTransferTags(waitOpForRead, consumerBlockId, "VECTOR", transferIndex);

        if (mainLoopOp) {
            builder.setInsertionPoint(transferOp);
            auto waitOpForWrite = builder.create<SyncBlockWaitOp>(loc, cubeCoreAttr, pipeVAttr, pipeFixAttr, flagId);
            attachTransferTags(waitOpForWrite, producerBlockId, "CUBE", transferIndex);
            builder.setInsertionPointAfter(consumerEndOp);
            auto setOpForWrite = builder.create<SyncBlockSetOp>(loc, vecCoreAttr, pipeVAttr, pipeFixAttr, flagId);
            attachTransferTags(setOpForWrite, consumerBlockId, "VECTOR", transferIndex);

            builder.setInsertionPoint(mainLoopOp);
            auto setOpForStart = builder.create<SyncBlockSetOp>(loc, vecCoreAttr, pipeVAttr, pipeFixAttr, flagId);
            builder.setInsertionPointAfter(mainLoopOp);
            auto waitOpForEnd = builder.create<SyncBlockWaitOp>(loc, cubeCoreAttr, pipeVAttr, pipeFixAttr, flagId);

            int startEndBlockId = getSsbufferBlockId(mainLoopOp);
            attachTransferTags(setOpForStart, startEndBlockId, "VECTOR", transferIndex);
            attachTransferTags(waitOpForEnd, startEndBlockId, "CUBE", transferIndex);
        }
        return;
    } else if (dyn_cast<hivm::CopyOp>(transferOp)) {
        builder.setInsertionPointAfter(transferOp);
        auto setOpForRead = builder.create<SyncBlockSetOp>(loc, vecCoreAttr, pipeMte3Attr, pipeMte1Attr, flagId);
        attachTransferTags(setOpForRead, producerBlockId, "VECTOR", transferIndex);
        builder.setInsertionPoint(consumerStartOp);
        auto waitOpForRead = builder.create<SyncBlockWaitOp>(loc, cubeCoreAttr, pipeMte3Attr, pipeMte1Attr, flagId);
        attachTransferTags(waitOpForRead, consumerBlockId, "CUBE", transferIndex);

        if (mainLoopOp) {
            builder.setInsertionPoint(transferOp);
            auto waitOpForWrite = builder.create<SyncBlockWaitOp>(loc, vecCoreAttr, pipeMAttr, pipeMte3Attr, flagId);
            attachTransferTags(waitOpForWrite, producerBlockId, "VECTOR", transferIndex);

            builder.setInsertionPointAfter(consumerEndOp);
            auto setOpForWrite = builder.create<SyncBlockSetOp>(loc, cubeCoreAttr, pipeMAttr, pipeMte3Attr, flagId);
            attachTransferTags(setOpForWrite, consumerBlockId, "CUBE", transferIndex);

            builder.setInsertionPoint(mainLoopOp);
            auto setOpForStart = builder.create<SyncBlockSetOp>(loc, cubeCoreAttr, pipeMAttr, pipeMte3Attr, flagId);
            builder.setInsertionPointAfter(mainLoopOp);
            auto waitOpForEnd = builder.create<SyncBlockWaitOp>(loc, vecCoreAttr, pipeMAttr, pipeMte3Attr, flagId);

            int startEndBlockId = getSsbufferBlockId(mainLoopOp);
            attachTransferTags(setOpForStart, startEndBlockId, "CUBE", transferIndex);
            attachTransferTags(waitOpForEnd, startEndBlockId, "VECTOR", transferIndex);
        }
        return;
    }
}

void InterCoreTransferAndSyncPass::insertPipeSSync(OpBuilder &builder, Operation *producerOp, Operation *consumerOp,
    int flag, Location loc, bool isCubeToVector)
{
    LOG_DEBUG("Inserting PIPE_S sync: " << (isCubeToVector ? "CUBE->VECTOR" : "VECTOR->CUBE") << ", flag = " << flag <<
        "\n");

    // CUBE -> VECTOR: srcPipe = PIPE_FIX, srcCoreType = CUBE, dstCoreType = VECTOR
    // VECTOR -> CUBE: srcPipe = PIPE_MTE2, srcCoreType = VECTOR, dstCoreType = CUBE
    hivm::PIPE srcPipe = isCubeToVector ? hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE2;
    hivm::TCoreType srcCoreType = isCubeToVector ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
    hivm::TCoreType dstCoreType = isCubeToVector ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
    hivm::PIPE dstPipe = hivm::PIPE::PIPE_S;

    auto srcCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), srcCoreType);
    auto dstCoreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), dstCoreType);
    auto srcPipeAttr = PipeAttr::get(builder.getContext(), srcPipe);
    auto dstPipeAttr = PipeAttr::get(builder.getContext(), dstPipe);
    auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

    builder.setInsertionPointAfter(producerOp);
    auto setOp = builder.create<SyncBlockSetOp>(loc, srcCoreAttr, srcPipeAttr, dstPipeAttr, flagId);

    builder.setInsertionPoint(consumerOp);
    auto waitOp = builder.create<SyncBlockWaitOp>(loc, dstCoreAttr, srcPipeAttr, dstPipeAttr, flagId);

    int prodBlockId = getSsbufferBlockId(producerOp);
    int consBlockId = getSsbufferBlockId(consumerOp);
    if (prodBlockId != -1) {
        StringRef prodCoreType = isCubeToVector ? "CUBE" : "VECTOR";
        attachCommonTags(setOp, prodBlockId, prodCoreType);
    }
    if (consBlockId != -1) {
        StringRef consCoreType = isCubeToVector ? "VECTOR" : "CUBE";
        attachCommonTags(waitOp, consBlockId, consCoreType);
    }

    LOG_DEBUG("[PIPE_S setOp]: " << *setOp << "\n");
    LOG_DEBUG("[PIPE_S waitOp]: " << *waitOp << "\n");
}

// V->C Transfer Logic
LogicalResult InterCoreTransferAndSyncPass::handleVectorToCube(OpBuilder &builder, DependencyInfo &dep,
    llvm::DenseMap<mlir::Value, mlir::Value> vecvalueMapping, llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping,
    FlagIdManager &flagManager)
{
    mlir::Value srcValue = dep.value;
    auto it = cubeValueMapping.find(srcValue);
    if (it != cubeValueMapping.end()) {
        srcValue = it->second;
    }
    Location loc = dep.value.getLoc();
    // Step 1: Shape normalization (automatically insert slice)
    Value normalizedVal = vecvalueMapping[dep.value];

    // Get start/end operations for V/C blocks
    auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module);
    auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module);

    Operation *transferOp = insertVectorToCubeTransfer(builder, srcValue, normalizedVal, prodEnd, consStart, loc,
        transferIndex, dep.iniConsumerBlockId);

    int flagId = flagManager.acquireId(prodStart);
    assert(flagId != -1 && "No available flag_id!!");
    auto [newProdStart, newProdEnd] = getBlockStartEnd(dep.producerBlockId, module);
    auto [newConsStart, newConsEnd] = getBlockStartEnd(dep.consumerBlockId, module);
    insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId, loc, transferIndex);

    transferIndex++;
    LOG_DEBUG("Inserted V->C transfer and sync: block " << dep.producerBlockId << " -> block " << dep.consumerBlockId <<
        "\n");
    return success();
}

// C->V Transfer Logic
LogicalResult InterCoreTransferAndSyncPass::handleCubeToVector(OpBuilder &builder, DependencyInfo &dep,
    llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping, FlagIdManager &flagManager)
{
    mlir::Value srcValue = dep.value;
    auto it = cubeValueMapping.find(srcValue);
    if (it != cubeValueMapping.end()) {
        srcValue = it->second;
    }
    Location loc = srcValue.getLoc();
    auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module); // C Block
    auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module); // V Block
    LOG_DEBUG("[newProdStart]" << *prodStart << "\n");
    LOG_DEBUG("[newProdEnd]" << *prodEnd << "\n");
    LOG_DEBUG("[newConsStart]" << *consStart << "\n");
    LOG_DEBUG("[newConsEnd]" << *consEnd << "\n");
    Operation *transferOp =
        insertCubeToVectorTransfer(builder, srcValue, prodEnd, consStart, loc, transferIndex, dep.iniConsumerBlockId);

    auto [newProdStart, newProdEnd] = getBlockStartEnd(dep.producerBlockId, module); // C Block
    auto [newConsStart, newConsEnd] = getBlockStartEnd(dep.consumerBlockId, module); // V Block
    int flagId = flagManager.acquireId(newProdStart);
    assert(flagId != -1 && "No available flag_id!!");
    insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId, loc, transferIndex);

    transferIndex++;
    LOG_DEBUG("Inserted C->V transfer and sync: block " << dep.producerBlockId << " -> block " << dep.consumerBlockId <<
        "\n");
    return success();
}

// PIPE_S Memory Dependency
LogicalResult InterCoreTransferAndSyncPass::handleMemoryDependency(OpBuilder &builder, DependencyInfo &dep,
    size_t depIndex, llvm::SmallVector<DependencyInfo> memDependencies, FlagIdManager &flagManager)
{
    LOG_DEBUG("Handling PIPE_S memory dependency...\n");

    // Get producer and consumer block start/end operations
    auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module);
    auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module);

    if (!prodStart || !prodEnd || !consStart || !consEnd) {
        LOG_DEBUG("[ERROR] Failed to get block start/end operations.\n");
        return failure();
    }

    if (isOuterLayerDependency(depIndex, prodEnd, consStart, memDependencies)) {
        LOG_DEBUG("[PIPE_S] Skipping outer layer dependency: block " << dep.producerBlockId << " -> block " <<
            dep.consumerBlockId << "\n");
        return success();
    }

    // Get flag ID
    int flagId = flagManager.acquireId(prodStart);
    assert(flagId != -1 && "No available flag_id!!");

    // Determine sync direction: CUBE->VECTOR or VECTOR->CUBE
    bool isCubeToVector = (dep.type == DependencyType::CubeToVector);

    // Get location info
    Location loc = prodEnd->getLoc();

    // Insert PIPE_S sync
    insertPipeSSync(builder, prodEnd, consStart, flagId, loc, isCubeToVector);

    transferIndex++;

    LOG_DEBUG("Inserted PIPE_S sync: block " << dep.producerBlockId << " -> block " << dep.consumerBlockId <<
        ", flagId = " << flagId << "\n");

    return success();
}

// Main Processing
LogicalResult InterCoreTransferAndSyncPass::processDependencies(FlagIdManager &flagManager)
{
    LOG_DEBUG("Starting InterCoreTransferAndSyncPass processDependencies...\n");
    OpBuilder builder(module.getContext());

    auto &info = getAnalysis<DataDependencyInfo>();
    if (!info.isValid()) {
        LOG_DEBUG("Error: Data dependency analysis failed.\n");
        return failure();
    }

    llvm::SmallVector<DependencyInfo> &V2CDependencies = info.getV2CDependencies();
    LOG_DEBUG("[DEBUG] V2CDependencies size: " << V2CDependencies.size() << "\n");
    for (size_t i = 0; i < V2CDependencies.size(); ++i) {
        auto &dep = V2CDependencies[i];
        LOG_DEBUG("[V2C-" << i << "] producerBlockId = " << dep.producerBlockId << ", consumerBlockId = " <<
            dep.consumerBlockId << ", iniProducerBlockId = " << dep.iniProducerBlockId << ", iniConsumerBlockId = " <<
            dep.iniConsumerBlockId << ", value = " << dep.value << "\n");
    }
    // Step 1: Handle V->C dependencies
    for (auto &dep : V2CDependencies) {
        Location loc = dep.value.getLoc();
        Nd2NzNormalize(builder, dep, loc);
    }
    llvm::DenseMap<mlir::Value, mlir::Value> vecvalueMapping = getVecValueMapping();
    llvm::DenseMap<mlir::Value, mlir::Value> cubevalueMapping = getCubeValueMapping();
    for (auto &dep : V2CDependencies) {
        LOG_DEBUG("[V->C] producerBlockId = " << dep.producerBlockId << ", consumerBlockId = " << dep.consumerBlockId <<
            "\n");
        if (failed(handleVectorToCube(builder, dep, vecvalueMapping, cubevalueMapping, flagManager))) {
            LOG_DEBUG("[ERROR] V->C failed! producerBlockId = " << dep.producerBlockId << ", consumerBlockId = " <<
                dep.consumerBlockId << "\n");
            return failure();
        }
    }
    LOG_DEBUG("Completed V->C transfers and syncs.\n");

    llvm::SmallVector<DependencyInfo> &C2VDependencies = info.getC2VDependencies();
    LOG_DEBUG("[DEBUG] C2VDependencies size: " << C2VDependencies.size() << "\n");
    // Step 2: Handle C->V dependencies
    for (auto &dep : C2VDependencies) {
        LOG_DEBUG("[C->V] producerBlockId = " << dep.producerBlockId << ", consumerBlockId = " << dep.consumerBlockId <<
            "\n");
        if (failed(handleCubeToVector(builder, dep, cubevalueMapping, flagManager))) {
            LOG_DEBUG("[ERROR] C->V failed!  producerBlockId = " << dep.producerBlockId << ", consumerBlockId = " <<
                dep.consumerBlockId << "\n");
            return failure();
        }
    }
    LOG_DEBUG("Completed C->V transfers and syncs.\n");
    LOG_DEBUG("=====================================================\n");
    LOG_DEBUG("InterCoreTransferAndSyncPass success!\n");

    return success();
}


// Declare dependent dialects
void InterCoreTransferAndSyncPass::getDependentDialects(DialectRegistry &registry) const
{
    registry
        .insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect, hivm::HIVMDialect, annotation::AnnotationDialect>();
}

// Pass Entry Point
void InterCoreTransferAndSyncPass::runOnOperation()
{
    LOG_DEBUG("\n--- enter InterCoreTransferAndSyncPass --->\n");
    module = getOperation();

    // Phase 1: Initialize FlagIdManager as local variable
    FlagIdManager flagManager(module);

    // Phase 2: Execute transfer and sync insertion
    if (failed(processDependencies(flagManager))) {
        signalPassFailure();
        LOG_DEBUG("Error: Inter-core transfer and sync failed.\n");
        return;
    }

    LOG_DEBUG("Module after InterCoreTransferAndSyncPass:\n" << module << "\n");

    LOG_DEBUG("--- exit InterCoreTransferAndSyncPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createInterCoreTransferAndSyncPass()
{
    return std::make_unique<InterCoreTransferAndSyncPass>();
}
} // namespace triton
} // namespace mlir
