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

#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/FlagIdReuse.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

namespace mlir {
namespace triton {

struct TransferPipeConfig {
  hivm::PipeAttr forReadTPipe;
  hivm::PipeAttr forReadPipe;
  hivm::PipeAttr forWriteTPipe;
  hivm::PipeAttr forWritePipe;
  hivm::TCoreTypeAttr srcCoreAttr;
  hivm::TCoreTypeAttr dstCoreAttr;
  llvm::StringRef srcCoreType;
  llvm::StringRef dstCoreType;
};

// Information found while matching a VECTOR to_tensor yielded from an SCF op.
struct VectorToTensorInfo {
  bufferization::ToTensorOp toTensor;
  scf::YieldOp yield;
  int tightlyCoupledBufferId;
};

// Result of matching a CUBE -> VECTOR direct-store pattern: CUBE data is
// transferred through UB and directly stored by VECTOR, without VECTOR tensor
// computation.
struct CubeToVectorDirectStoreInfo {
  std::optional<int> tightlyCoupledBufferId;
  Operation *storeOp = nullptr;
};

// Define pass
class InterCoreTransferAndSyncPass
    : public PassWrapper<InterCoreTransferAndSyncPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterCoreTransferAndSyncPass)

  InterCoreTransferAndSyncPass() = default;

  // Declare dependent dialects
  void getDependentDialects(DialectRegistry &registry) const override;

  // Run the pass
  void runOnOperation() override;

  static constexpr ::llvm::StringRef getArgumentName() {
    return "inter-core-transfer-and-sync";
  }
  ::llvm::StringRef getArgument() const override {
    return "inter-core-transfer-and-sync";
  }
  ::llvm::StringRef getDescription() const override {
    return "Insert inter-core transfer and synchronization operations between "
           "Vector and Cube cores";
  }
  ::llvm::StringRef getName() const override {
    return "InterCoreTransferAndSyncPass";
  }

private:
  mlir::ModuleOp module;
  int transferIndex = 0;
  int markAllocIndex = 0;
  int intraDepsGroupId = 0;

  llvm::DenseMap<int, llvm::SmallVector<LoopLikeOpInterface>> loopInclusions;
  llvm::DenseSet<int> singleLoopSet;

  llvm::DenseMap<mlir::Value, mlir::Value> ndnzValueMapping;
  SSBufferManager ssbufferManager;

  void analyzeLoopInclusion();
  void moveStartEndSync(mlir::OpBuilder &builder);

  mlir::LogicalResult
  processDependencies(FlagIdManager &flagManager,
                      FlagIdReuseManager &flagIdReuseManager);
  mlir::LogicalResult
  handleVectorToCube(mlir::OpBuilder &builder, DependencyInfo &dep,
                     FlagIdManager &flagManager,
                     FlagIdReuseManager &flagIdReuseManager);
  mlir::LogicalResult
  handleCubeToVector(mlir::OpBuilder &builder, DependencyInfo &dep,
                     FlagIdManager &flagManager,
                     FlagIdReuseManager &flagIdReuseManager);
  mlir::LogicalResult handleCubeToCube(mlir::OpBuilder &builder,
                                       DependencyInfo &dep);
  mlir::LogicalResult handleMemoryDependency(
      mlir::OpBuilder &builder, DependencyInfo &dep, size_t depIndex,
      llvm::SmallVector<DependencyInfo> memDependencies,
      FlagIdManager &flagManager, FlagIdReuseManager &flagIdReuseManager);

  std::pair<mlir::Operation *, mlir::Operation *>
  getBlockStartEnd(int blockId, mlir::ModuleOp module);
  std::pair<mlir::Operation *, mlir::Operation *>
  getSubBlockStartEnd(mlir::Operation *defOp);
  bool
  isOuterLayerDependency(size_t depIndex, mlir::Operation *currProdEnd,
                         mlir::Operation *currConsStart,
                         llvm::SmallVector<DependencyInfo> &memDependencies);

  SmallVector<int64_t> computeExpectedShape(mlir::Value depValue,
                                            int NzDimWidth);
  bool isExpectedShape(Value value, SmallVector<int64_t> &expectedShape);
  mlir::Value alignShapeByInsertSlice(mlir::OpBuilder &builder,
                                      DependencyInfo &dep, mlir::Location loc,
                                      mlir::Value origValue,
                                      llvm::SmallVector<int64_t> expectedShape,
                                      int originBlockId);
  void Nd2NzNormalize(mlir::OpBuilder &builder, DependencyInfo &dep,
                      mlir::Location loc);

  mlir::Operation *annotateTightlyCoupledBuffer(mlir::OpBuilder &builder,
                                                mlir::Operation *allocOp,
                                                mlir::Location loc);
  mlir::Operation *findMainLoopforTransfer(mlir::Operation *endOp,
                                           mlir::Operation *startOp);
  mlir::Operation *
  createC2CSharedL1Buffer(mlir::OpBuilder &builder, mlir::Location loc,
                          llvm::ArrayRef<int64_t> shape, mlir::Type elemType,
                          int prodBlockId, mlir::Operation *prodEnd,
                          mlir::Operation *consStart, bool isIntraC2C);
  std::pair<mlir::Operation *, mlir::Operation *>
  createTransferAllocs(mlir::OpBuilder &builder, mlir::Location loc,
                       llvm::ArrayRef<int64_t> shape, mlir::Type elemType,
                       hivm::AddressSpace addrSpace, mlir::Operation *prodEndOp,
                       mlir::Operation *consStartOp, int prodBlockId,
                       int consBlockId, llvm::StringRef prodTag,
                       llvm::StringRef consTag, int transferIndex);
  mlir::Operation *analyzeConsumerReadInsertPoint(Value srcValue,
                                                  int iniConsumerId);
  mlir::Operation *getConsumerWaitPoint(int transferIndex);
  mlir::Operation *getCopyPointBeforeStore(Value depValue,
                                           Operation *vectorEndOp,
                                           int iniProducerBlockId);
  mlir::Operation *getFixpipePointAfterProducer(Value depValue,
                                                int iniProducerBlockId);
  linalg::MatmulOp findYieldMatmulInSplitIf(Value splittedIfResult);
  void AnalyzeSplittedIf(DependencyInfo &dep);
  mlir::Operation *insertVectorToCubeTransfer(
      mlir::OpBuilder &builder, mlir::Value srcValue,
      mlir::Value normalizedValue, mlir::Operation *vectorEndOp,
      mlir::Operation *cubeStartOp, mlir::Location loc, int transferIndex,
      DependencyInfo &dep, bool is1DTensor,
      mlir::Operation **consumedDataOp = nullptr);
  mlir::Operation *
  insertCubeToVectorTransfer(mlir::OpBuilder &builder, mlir::Value srcValue,
                             mlir::Operation *cubeEndOp,
                             mlir::Operation *vectorStartOp, mlir::Location loc,
                             int transferIndex, DependencyInfo &dep,
                             mlir::Operation **consumedDataOp = nullptr);
  TransferPipeConfig getTransferPipeConfig(Operation *transferOp,
                                           bool isStoreDirectly = false);
  void insertInterCoreSync(mlir::OpBuilder &builder,
                           mlir::Operation *transferOp,
                           mlir::Operation *consumerStartOp,
                           mlir::Operation *consumerEndOp, int flag,
                           mlir::Location loc, int transferIndex,
                           FlagIdReuseManager &flagIdReuseManager,
                           mlir::Operation *consumedDataOp = nullptr,
                           bool isStoreDirectly = false);
  void insertMemDepSync(mlir::OpBuilder &builder,
                        mlir::Operation *producerStartOp,
                        mlir::Operation *producerEndOp,
                        mlir::Operation *consumerStartOp,
                        mlir::Operation *consumerEndOp, int flag,
                        mlir::Location loc, bool isCubeToVector,
                        FlagIdReuseManager &flagIdReuseManager);
  // Match the CUBE -> VECTOR direct-store pattern inside the given SCF op:
  // CUBE data reaches VECTOR through UB and is stored without VECTOR tensor
  // computation (`scf.for` / `scf.while` / `scf.if`).
  static std::optional<CubeToVectorDirectStoreInfo>
  matchCubeToVectorDirectStorePattern(Operation *scfOp);
  // Detect SCF ops whose yielded data flows out via fixpipe and is then
  // stored by a post-loop materialize_in_destination. Insert a
  // CUBE.PIPE_FIX -> VECTOR.PIPE_MTE3 sync pair to guard the store.
  void
  processCubeToVectorDirectStoreSync(mlir::OpBuilder &builder,
                                     FlagIdManager &flagManager,
                                     FlagIdReuseManager &flagIdReuseManager);
  // Remove VECTOR add-from-matmul pseudo-ops so the SCF yield points at the
  // real data source.
  void removeVectorPseudoOps();
  void sortDependencies(llvm::SmallVector<DependencyInfo> &dependencies,
                        mlir::ModuleOp module);
  llvm::SmallVector<mlir::Operation *>
  insertAnalyzeFlagRelations(mlir::ModuleOp module,
                             FlagIdReuseManager &flagIdReuseManager);
  void remapInterCoreTransferFlagIds(llvm::DenseMap<int, int> &remapResult);
  bool isStoreDirectlyInUserChain(mlir::Value toTensorValue);
  std::pair<bool, bool> analyzeMatmulOperand(DependencyInfo &dep);
  int getNzDimWidth(DependencyInfo &dep, int64_t blk);
};

std::unique_ptr<OperationPass<ModuleOp>> createInterCoreTransferAndSyncPass();

void registerInterCoreTransferAndSyncPasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_INTER_CORE_TRANSFER_AND_SYNC_H
