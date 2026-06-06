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

#ifndef TRITON_ADAPTER_INTER_CORE_TRANSFER_AND_SYNC_H
#define TRITON_ADAPTER_INTER_CORE_TRANSFER_AND_SYNC_H

#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
// Define pass
class InterCoreTransferAndSyncPass : public PassWrapper<InterCoreTransferAndSyncPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterCoreTransferAndSyncPass)

    InterCoreTransferAndSyncPass() = default;

    // Declare dependent dialects
    void getDependentDialects(DialectRegistry &registry) const override;

    // Run the pass
    void runOnOperation() override;

private:
    mlir::ModuleOp module;
    int transferIndex = 0;
    int markAllocIndex = 0;

    llvm::DenseMap<mlir::Value, mlir::Value> vecValueMapping;
    llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping;

    mlir::LogicalResult processDependencies(FlagIdManager &flagManager);
    mlir::LogicalResult handleVectorToCube(mlir::OpBuilder &builder,
                                           DependencyInfo &dep,
                                           llvm::DenseMap<mlir::Value, mlir::Value> vecvalueMapping,
                                           llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping,
                                           FlagIdManager &flagManager);
    mlir::LogicalResult handleCubeToVector(mlir::OpBuilder &builder,
                                           DependencyInfo &dep,
                                           llvm::DenseMap<mlir::Value, mlir::Value> cubeValueMapping,
                                           FlagIdManager &flagManager);
    mlir::LogicalResult handleMemoryDependency(mlir::OpBuilder &builder,
                                               DependencyInfo &dep,
                                               size_t depIndex,
                                               llvm::SmallVector<DependencyInfo> memDependencies,
                                               FlagIdManager &flagManager);

    std::pair<mlir::Operation *, mlir::Operation *> getBlockStartEnd(int blockId, mlir::ModuleOp module);
    bool isOuterLayerDependency(size_t depIndex,
                                mlir::Operation *currProdEnd,
                                mlir::Operation *currConsStart,
                                llvm::SmallVector<DependencyInfo> &memDependencies);

    SmallVector<int64_t> computeExpectedShape(mlir::Value value);
    bool isShapeExpected(mlir::Value value, llvm::SmallVector<int64_t> &expectedShape);
    mlir::Value normalizeIfNeeded(mlir::OpBuilder &builder,
                                  DependencyInfo &dep,
                                  mlir::Location loc,
                                  mlir::Value origValue,
                                  llvm::SmallVector<int64_t> expectedShape,
                                  int originBlockId);
    void Nd2NzNormalize(mlir::OpBuilder &builder, DependencyInfo &dep, mlir::Location loc);
    void rewriteMatmulWithNewShape(mlir::OpBuilder &builder,
                                   mlir::Operation *matmulOp,
                                   mlir::Location loc);
    void rewriteTransposeWithNewShape(mlir::OpBuilder &builder,
                                      mlir::Operation *transposeOp,
                                      mlir::Location loc);
    llvm::DenseMap<mlir::Value, mlir::Value> getVecValueMapping()
    {
        return vecValueMapping;
    }
    llvm::DenseMap<mlir::Value, mlir::Value> getCubeValueMapping()
    {
        return cubeValueMapping;
    }

    mlir::Operation *annotateTightlyCoupledBuffer(mlir::OpBuilder &builder, mlir::Operation *allocOp,
        mlir::Location loc);
    mlir::Operation *findMainLoopforTransfer(mlir::Operation *endOp, mlir::Operation *startOp);
    std::pair<mlir::Operation *, mlir::Operation *> createTransferAllocs(mlir::OpBuilder &builder, mlir::Location loc,
        llvm::ArrayRef<int64_t> shape, mlir::Type elemType, hivm::AddressSpace addrSpace, mlir::Operation *prodEndOp,
        mlir::Operation *consStartOp, int prodBlockId, int consBlockId, llvm::StringRef prodTag,
        llvm::StringRef consTag, int transferIndex);
    mlir::Operation *insertVectorToCubeTransfer(mlir::OpBuilder &builder, mlir::Value srcValue,
        mlir::Value normalizedValue, mlir::Operation *vectorEndOp, mlir::Operation *cubeStartOp, mlir::Location loc,
        int transferIndex, int iniConsumerId);
    mlir::Operation *insertCubeToVectorTransfer(mlir::OpBuilder &builder, mlir::Value srcValue,
        mlir::Operation *cubeEndOp, mlir::Operation *vectorStartOp, mlir::Location loc, int transferIndex,
        int iniConsumerId);
    void insertInterCoreSync(mlir::OpBuilder &builder, mlir::Operation *transferOp, mlir::Operation *consumerStartOp,
        mlir::Operation *consumerEndOp, int flag, mlir::Location loc, int transferIndex);
    void insertPipeSSync(mlir::OpBuilder &builder, mlir::Operation *producerOp, mlir::Operation *consumerOp, int flag,
        mlir::Location loc, bool isCubeToVector);
};

std::unique_ptr<OperationPass<ModuleOp>> createInterCoreTransferAndSyncPass();
} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_INTER_CORE_TRANSFER_AND_SYNC_H
