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

#ifndef TRITON_ADAPTER_DATA_DEPENDENCY_ANALYSIS_H
#define TRITON_ADAPTER_DATA_DEPENDENCY_ANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {
enum class DependencyType { VectorToCube, CubeToVector };

struct BlockInfo {
    int blockId;
    bool isCube;
    bool isControl;
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Value> outputs;
    llvm::SmallVector<mlir::Operation *> Operations;
};

struct DependencyInfo {
    DependencyType type;
    mlir::Value value;

    int producerBlockId;
    int consumerBlockId;
    int iniProducerBlockId;
    int iniConsumerBlockId;
};

class DataDependencyInfo {
public:
    explicit DataDependencyInfo(mlir::Operation *op) {}

    bool isValid() const
    {
        return valid;
    }

    // for MLIR Analysis framework
    bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa)
    {
        return false;
    }

    static bool classof(const DataDependencyInfo *info)
    {
        return true;
    }

    // get analyze result
    llvm::DenseMap<int, BlockInfo> &getBlockInfoMap() { return blockInfoMap; }
    llvm::SmallVector<DependencyInfo> &getV2CDependencies() { return v2cDependencies; }
    llvm::SmallVector<DependencyInfo> &getC2VDependencies() { return c2vDependencies; }
    llvm::SmallVector<DependencyInfo> &getMemoryDependencies() { return memoryDependencies; }

    void setValid(bool v)
    {
        valid = v;
    }

private:
    bool valid = false;
    llvm::DenseMap<int, BlockInfo> blockInfoMap;
    llvm::SmallVector<DependencyInfo> v2cDependencies;
    llvm::SmallVector<DependencyInfo> c2vDependencies;
    llvm::SmallVector<DependencyInfo> c2cDependencies;
    llvm::SmallVector<DependencyInfo> memoryDependencies;
};

// Define pass
// Pass for analyzing data dependencies between Vector and Cube blocks
class DataDependencyAnalysisPass : public PassWrapper<DataDependencyAnalysisPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DataDependencyAnalysisPass)

    DataDependencyAnalysisPass() = default;

    // Run the pass
    void runOnOperation();

private:
    void createBlockInfoMap(DataDependencyInfo &info);
    void collectBlockInfo(DataDependencyInfo &info, int blockId,
                          llvm::SmallVector<mlir::Operation *> &ops);

    void collectDepInfo(mlir::Value depvalue, DependencyType dependencyType,
                        llvm::SmallVector<DependencyInfo> &dependencies,
                        int iniProdId, int iniConsId, DataDependencyInfo &info);
    void analyzeExternalInputs(DataDependencyInfo &info);
    void analyzeExternalOutputs(DataDependencyInfo &info);

    void analyzeMemoryEffect(DataDependencyInfo &info);
    std::pair<int, int> findCommonLevelBlockIds(DataDependencyInfo &info,
                                                int producerBlockId,
                                                int consumerBlockId);

    bool isControlFlowOp(mlir::Operation *op);

    mlir::ModuleOp module;
};

std::unique_ptr<OperationPass<ModuleOp>> createDataDependencyAnalysisPass();

// Helper: Get BlockId
int getSsbufferBlockId(Operation *op);
} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DATA_DEPENDENCY_ANALYSIS_H
