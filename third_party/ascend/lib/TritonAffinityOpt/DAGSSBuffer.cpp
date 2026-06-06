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

#include "TritonAffinityOpt/Passes.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"

// #include "mlir/Transforms/Canonicalizer.h"
// #include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DAGSSBUFFER
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
struct DAGSSBufferPass
    : public mlir::triton::impl::DAGSSBufferBase<
          DAGSSBufferPass> {
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<linalg::LinalgDialect>();
  }
};
} // namespace

void ControlSsbufV2(ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    // 用于记录已经处理过的scope.scope操作
    llvm::DenseSet<mlir::Operation*> processedScopes;

    auto aiCAttr = hivm::TCoreTypeAttr::get(
            builder.getContext(),
            hivm::TCoreType::CUBE);
    int cubeControlIndex = 15;
    int vectorControlIndex = 14;

    llvm::DenseSet<mlir::Operation*> processedScopes2;
    module->walk([&](SyncBlockWaitOp op) {
        auto pipeS = hivm::PipeAttr::get(op->getContext(), hivm::PIPE::PIPE_S);

        // 向上查找父scope.scope操作
        mlir::Operation* parentOp = op->getParentOp();
        mlir::Operation* scopeOp = nullptr;
        mlir::Operation* forOp = nullptr;
        
        // 向上遍历查找scope.scope操作
        while (parentOp) {
            if (dyn_cast<scope::ScopeOp>(parentOp)) {
                scopeOp = parentOp;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
        parentOp = op->getParentOp();
        while (parentOp) {
            if (dyn_cast<scf::ForOp>(parentOp)) {
                forOp = parentOp;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
        // 如果没有找到scope.scope操作，则跳过
        if (!scopeOp) {
            return;
        }
        if (!forOp) {
            return;
        }

        // 如果该scope已经处理过，则跳过
        if (processedScopes2.count(forOp) > 0) return;
        
        // 标记该scope为已处理
        processedScopes2.insert(forOp);

    });
    bool firstSet = true;
    bool firstWait = true;
    for (auto forOp : processedScopes2) {
        mlir::Operation* parentOp = forOp->getParentOp();
        mlir::Operation* scopeOp = nullptr;
        
        // 向上遍历查找scope.scope操作
        while (parentOp) {
            if (dyn_cast<scope::ScopeOp>(parentOp)) {
                scopeOp = parentOp;
                break;
            }
            parentOp = parentOp->getParentOp();
        }
        bool isAIC = false;
        // 1. 先检查操作是否有这个属性
        
        if (scopeOp->hasAttr("hivm.tcore_type")) {
            auto attr = scopeOp->getAttr("hivm.tcore_type");
            if (attr == aiCAttr) {
            isAIC = true;
            }
        }

        if (isAIC) {
            // 在for循环的开头插入代码
            builder.setInsertionPoint(scopeOp);
            // %ssb_ready_addr = llvm.mlir.constant(0 : i64) : i64
            auto i64Type = builder.getIntegerType(64);
            auto i32Type = builder.getIntegerType(32);

            builder.setInsertionPointToStart(&forOp->getRegion(0).front());
            // %ssb_ready_addr = llvm.mlir.constant(0 : i64) : i64
            // add sync_block_wait
            auto coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
            auto setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
            builder.create<SyncBlockWaitOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);

            // 在循环末尾（yield之前）插入代码
            auto &loopBody = forOp->getRegion(0).front();
            // 找到循环体的terminator（应该是yield操作）
            auto *terminator = loopBody.getTerminator();
            builder.setInsertionPoint(terminator);
            
            // add sync_block_set
            coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
            setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            flagId = builder.getIntegerAttr(builder.getI64Type(), cubeControlIndex);
            builder.create<SyncBlockSetOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);

            if (firstWait) {
                auto &scopeBlock = scopeOp->getRegion(0).front();
                auto *scope_terminator = scopeBlock.getTerminator();
                builder.setInsertionPoint(scope_terminator);            
                // add sync_block_wait
                coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
                setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
                builder.create<SyncBlockWaitOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
                firstWait = false;
            }
        }
        else {
            // 1. 在scopeop的开头插入代码
            // 假设scopeOp是一个具有区域的操作，我们获取其第一个块
            if (firstSet) {
                auto &scopeBlock = scopeOp->getRegion(0).front();
                builder.setInsertionPointToStart(&scopeBlock);
                
                // add sync_block_wait
                auto coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
                auto setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                auto waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
                auto flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
                builder.create<SyncBlockSetOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
                firstSet = false;
            }

            auto i64Type = builder.getIntegerType(64);
            auto i32Type = builder.getIntegerType(32);
            
            // 创建需要的常量
            auto c32ConstAttr = mlir::IntegerAttr::get(i64Type, 32);
            auto c32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i64Type, c32ConstAttr);
            
            auto c0i64ConstAttr = mlir::IntegerAttr::get(i64Type, 0);
            auto c0i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i64Type, c0i64ConstAttr);
            
            auto c0i32ConstAttr = mlir::IntegerAttr::get(i32Type, 0);
            auto c0i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i32Type, c0i32ConstAttr);
            
            auto c1i32ConstAttr = mlir::IntegerAttr::get(i32Type, 1);
            auto c1i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), i32Type, c1i32ConstAttr);
            
            // %sub_id = hivm.hir.get_sub_block_idx -> i64
            // 这里假设有一个getSubBlockIdxOp操作
            auto subIdOp = builder.create<GetSubBlockIdxOp>(
                scopeOp->getLoc(), i64Type);
            
            // %ssb_addr_offset = arith.muli %sub_id, %c32_i64 : i64
            auto ssbAddrOffsetOp = builder.create<mlir::arith::MulIOp>(
                scopeOp->getLoc(),
                subIdOp.getResult(),
                c32ConstOp.getResult());
            
            // %ssb_addr = arith.addi %ssb_addr_offset, %c32_i64 : i64
            auto ssbAddrOp = builder.create<mlir::arith::AddIOp>(
                scopeOp->getLoc(),
                ssbAddrOffsetOp.getResult(),
                c32ConstOp.getResult());
            
            // %vec_id = arith.cmpi eq, %sub_id, %c0_i64 : i64
            auto vecIdOp = builder.create<mlir::arith::CmpIOp>(
                scopeOp->getLoc(),
                mlir::arith::CmpIPredicate::eq,
                subIdOp.getResult(),
                c0i64ConstOp.getResult());
            
            // 2. 在parentop的开头插入代码
            builder.setInsertionPointToStart(&forOp->getRegion(0).front());
            
            // add sync_block_wait
            auto coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
            auto setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            auto flagId = builder.getIntegerAttr(builder.getI64Type(), cubeControlIndex);
            builder.create<SyncBlockWaitOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
            
            // 在循环末尾（yield之前）插入代码
            auto &loopBody = forOp->getRegion(0).front();
            // 找到循环体的terminator（应该是yield操作）
            auto *terminator = loopBody.getTerminator();
            builder.setInsertionPoint(terminator);
            
            // add sync_block_wait
            coreAttr = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
            setPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            waitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
            flagId = builder.getIntegerAttr(builder.getI64Type(), vectorControlIndex);
            builder.create<SyncBlockSetOp>(forOp->getLoc(), coreAttr, setPipe, waitPipe, flagId);
        }
    }
    
    auto i64Type = builder.getIntegerType(64);
    auto i32Type = builder.getIntegerType(32);
    auto initPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext(), 11);
    SmallVector<scope::ScopeOp> scopeOps;
    module->walk([&](mlir::Operation* op) {
        // 检查是否为目标操作
        if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
            scopeOps.push_back(scopeOp);
        }
    });
    if (!scopeOps.empty()) {
      auto scopeOp = scopeOps[0];
      builder.setInsertionPoint(scopeOp);
      auto c0i64ConstAttr = mlir::IntegerAttr::get(i64Type, 0);
      auto c0i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
          scopeOp->getLoc(), i64Type, c0i64ConstAttr);
      auto c32i64ConstAttr = mlir::IntegerAttr::get(i64Type, 32);
      auto c32i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
          scopeOp->getLoc(), i64Type, c32i64ConstAttr);
      auto c64i64ConstAttr = mlir::IntegerAttr::get(i64Type, 64);
      auto c64i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
          scopeOp->getLoc(), i64Type, c64i64ConstAttr);
      auto c96i64ConstAttr = mlir::IntegerAttr::get(i64Type, 96);
      auto c96i64ConstOp = builder.create<mlir::LLVM::ConstantOp>(
          scopeOp->getLoc(), i64Type, c96i64ConstAttr);
      auto c0i32ConstAttr = mlir::IntegerAttr::get(i32Type, 0);
      auto c0i32ConstOp = builder.create<mlir::LLVM::ConstantOp>(
          scopeOp->getLoc(), i32Type, c0i32ConstAttr);
      
      auto c0initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
          scopeOp->getLoc(), initPtrType, c0i64ConstOp.getResult());
      auto c32initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
          scopeOp->getLoc(), initPtrType, c32i64ConstOp.getResult());
      auto c64initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
          scopeOp->getLoc(), initPtrType, c64i64ConstOp.getResult());
      auto c96initInttoptrOp = builder.create<mlir::LLVM::IntToPtrOp>(
          scopeOp->getLoc(), initPtrType, c96i64ConstOp.getResult());
      
      builder.create<LLVM::StoreOp>(
              scopeOp->getLoc(),
              c0i32ConstOp,
              c0initInttoptrOp
          );
      builder.create<LLVM::StoreOp>(
              scopeOp->getLoc(),
              c0i32ConstOp,
              c32initInttoptrOp
          );
      builder.create<LLVM::StoreOp>(
              scopeOp->getLoc(),
              c0i32ConstOp,
              c64initInttoptrOp
          );
      builder.create<LLVM::StoreOp>(
              scopeOp->getLoc(),
              c0i32ConstOp,
              c96initInttoptrOp
          );
    }
}

scf::ForOp transformLoop(scf::ForOp forOp, OpBuilder &builder) {
    
    // 1. 获取原始循环的信息
    Value originalLowerBound = forOp.getLowerBound();
    Value originalUpperBound = forOp.getUpperBound();
    Value originalStep = forOp.getStep();
    SmallVector<Value> iterArgs;
    for (auto arg : forOp.getInitArgs()) {
        iterArgs.push_back(arg);
    }
    auto yields = forOp.getBody()->getTerminator();
    
    // 2. 检查循环体中是否有特定操作
    int hasTargetOps = 0;
    forOp.walk([&](Operation* op) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            if (ifOp->hasAttr("ssbuffer")) {
                hasTargetOps++;
            }
        }
    });
    // 3. 如果存在目标操作，在迭代参数中添加计数器
    Value counterInit = nullptr;
    mlir::Operation* parentOp = forOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    // 向上遍历查找scope.scope操作
    while (parentOp) {
        if (dyn_cast<scope::ScopeOp>(parentOp)) {
            scopeOp = parentOp;
            break;
        }
        parentOp = parentOp->getParentOp();
    }

    builder.setInsertionPoint(scopeOp);
    for (int i = 0; i < hasTargetOps; i++) {
        Location loc = forOp.getLoc();
        auto argType = originalLowerBound.getType();
        
        // 添加到迭代参数列表
        iterArgs.push_back(originalLowerBound);
    }
    // 2. 创建新的上界：originalUpperBound * 2
    Location loc = forOp.getLoc();
    Type ubType = originalStep.getType();
    builder.setInsertionPoint(forOp);
    
    int count = 0;
    for (auto &op : forOp.getBody()->getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto parentOp = ifOp->getParentOp();
        if (parentOp == forOp && ifOp->hasAttr("ssbuffer")) {
          count++;
        }
      }
    }

    Value two;
    if (ubType.isIndex()) {
        two = builder.create<arith::ConstantIndexOp>(loc, count - 1);
    } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
        // 对于整数类型，创建相应类型的常数2
        two = builder.create<arith::ConstantIntOp>(loc, intType, count - 1);
    } else {
        // 其他类型可能需要特殊处理
        llvm::errs() << "Warning: Unexpected type for upper bound: " << ubType << "\n";
        // 尝试创建索引类型的2然后转换
        auto indexTwo = builder.create<arith::ConstantIndexOp>(loc, count - 1);
        two = builder.create<arith::IndexCastOp>(loc, ubType, indexTwo);
    }
    
    auto steps = builder.create<arith::MulIOp>(
        forOp.getLoc(),
        originalStep,
        two
      );

    auto nowUpperBound = builder.create<arith::AddIOp>(
        forOp.getLoc(),
        originalUpperBound,
        steps
      );

    // 3. Create a new for loop
    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(),
        originalLowerBound,
        nowUpperBound,
        originalStep,
        iterArgs);
    
    // 4. 设置IR映射表，将旧循环的变量映射到新循环
    IRMapping mapper;
    
    // 映射迭代变量
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
    
    // 映射迭代参数
    for (auto [oldArg, newArg] : 
         llvm::zip(forOp.getRegionIterArgs(), 
                  newForOp.getRegionIterArgs())) {
        mapper.map(oldArg, newArg);
    }
    
    SmallVector<Value> newCounterArgs;
    for (int i = forOp.getRegionIterArgs().size(); i < newForOp.getRegionIterArgs().size(); i++) {
        newCounterArgs.push_back(newForOp.getRegionIterArgs()[i]);
    }
    // 5. 克隆循环体内容到新循环
    auto &newLoopBody = *newForOp.getBody();
    builder.setInsertionPointToStart(&newLoopBody);
    
    for (auto &op : forOp.getBody()->without_terminator()) {
        builder.clone(op, mapper);
    }
    
    // 6. 克隆yield操作
    if (auto yieldOp = dyn_cast<scf::YieldOp>(yields)) {
        SmallVector<Value> newYieldOperands;
        for (auto operand : yieldOp.getOperands()) {
            newYieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        if (hasTargetOps != 0) {
            for (auto currentCounter : newCounterArgs) {                
                // 将更新后的计数器添加到yield操作数中
                newYieldOperands.push_back(currentCounter);
            }
        }
        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    }
    
    // 7. 替换原循环的结果
    if (hasTargetOps != 0) {
        // 新循环有额外的计数器结果，但原循环没有对应结果
        // 我们可以选择只替换原循环对应的结果，或者忽略计数器结果
        unsigned numOriginalResults = forOp.getNumResults();
        SmallVector<Value> originalResults;
        for (unsigned i = 0; i < numOriginalResults; i++) {
            originalResults.push_back(newForOp.getResult(i));
        }
        forOp.replaceAllUsesWith(originalResults);
    } else {
        forOp.replaceAllUsesWith(newForOp.getResults());
    }
    
    // 8. 删除原循环
    forOp.erase();
    return newForOp;
    
}

// Find the first occurrence of convert_layout or fixpipe operation after the specified operation
Operation* findFirstTargetOpAfterWait(SyncBlockWaitOp waitOp, SmallVector<Operation*>& excludedValues)
{
    bool startSearching = false;
    
    for (Operation &op : waitOp->getBlock()->getOperations()) {
        // meet waitop, start searching
        if (&op == waitOp) {
            startSearching = true;
            continue;
        }

        // have not meet waitOp, skip
        if (!startSearching) {
            continue;
        }

        Operation* res = nullptr;

        if (isa<hivm::ConvertLayoutOp>(op)) {
            res = op.getOperands()[0].getDefiningOp();
        } else if (isa<hivm::FixpipeOp>(op)) {
            res = op.getOperands()[1].getDefiningOp();
        } else if (isa<hivm::CopyOp>(op)) {
            res = op.getOperands()[1].getDefiningOp();
        } else if (isa<memref::MemorySpaceCastOp>(op)) {
            res = op.getOperands()[0].getDefiningOp();
        }

        // get res
        if (res) {
          // in excludedValues → skip，search next one
          if (llvm::is_contained(excludedValues, res)) {
              continue;
          }
          // not in excludedValues → note and return
          excludedValues.push_back(res);
          return res;
        }
    }
    
    return nullptr;
}

void getWaitType(std::string CoreType, scf::ForOp forOp, SmallVector<bool>& waitTypes, SmallVector<Operation*>& allocTypes)
{
    auto scalarWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_S);
    auto cubeWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_FIX);
    auto vectorWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_MTE3);
    SmallVector<Operation*> excludedValues;
    forOp.walk([&](Operation* op) {
        if (auto waitOp = dyn_cast<SyncBlockWaitOp>(op)) {
            auto parentOp = op->getParentOp();
            if (isa<scf::IfOp>(parentOp) && parentOp->hasAttr("ssbuffer")) {
                auto ifOp = dyn_cast<scf::IfOp>(parentOp);
                if (forOp == ifOp->getParentOp()) {
                  auto waitPipe = waitOp.getPipe();
                  auto setPipe = waitOp.getTpipe();
                  if (waitPipe == scalarWaitPipe || setPipe == scalarWaitPipe) {
                      bool hasBackward = waitOp->hasAttr("ssbuf.backward");
                      if (hasBackward) {
                          waitTypes.push_back(0);
                          allocTypes.push_back(waitOp);
                      } else {
                          waitTypes.push_back(1);
                          allocTypes.push_back(waitOp);
                      }
                  } else {
                      if ((waitPipe == cubeWaitPipe && CoreType == "cube") || (waitPipe == vectorWaitPipe && CoreType == "vector")) {
                          auto allocOp = findFirstTargetOpAfterWait(waitOp, excludedValues);
                          waitTypes.push_back(0);
                          allocTypes.push_back(allocOp);
                      } else {
                          auto allocOp = findFirstTargetOpAfterWait(waitOp, excludedValues);
                          waitTypes.push_back(1);
                          allocTypes.push_back(allocOp);
                      }
                   }
                }
            }
        }
    });
}

DenseMap<int, int> getCounterOffset(scf::ForOp forOp) {
    int i = 0;
    DenseMap<int, int> bufferMap;
    auto scalarWaitPipe = PipeAttr::get(forOp.getContext(), hivm::PIPE::PIPE_S);
    forOp.walk([&](Operation* op) {
        bufferMap[i] = 0;
        auto ifOp = dyn_cast<scf::IfOp>(op);
        if (ifOp && ifOp->hasAttr("ssbuffer") && ifOp->getParentOp() == forOp) {
            ifOp.walk([&](Operation* op) {
                if (auto waitOp = dyn_cast<SyncBlockWaitOp>(op)) {
                    if (auto waitIfOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
                        if (waitIfOp == ifOp) {
                          auto waitPipe = waitOp.getPipe();
                          auto setPipe = waitOp.getTpipe();
                          if ((waitPipe != scalarWaitPipe || setPipe != scalarWaitPipe)) {
                              bufferMap[i]++;
                          }
                        }
                    }
                }
            });
            i ++;
        }
    });
    return bufferMap;
}

SmallVector<Value> addBufValLoop(scf::ForOp forOp, DenseMap<Operation*, int> VecBitMap,
                                 DenseMap<Operation*, int> CubeBitMap, OpBuilder &builder)
{
    auto aiCAttr = hivm::TCoreTypeAttr::get(
            builder.getContext(),
            hivm::TCoreType::CUBE);
    bool isAIC = false;
    // 向上查找父scope.scope操作
    mlir::Operation* parentOp = forOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    // 向上遍历查找scope.scope操作
    while (parentOp) {
        if (dyn_cast<scope::ScopeOp>(parentOp)) {
            scopeOp = parentOp;
            break;
        }
        parentOp = parentOp->getParentOp();
    }
    if (scopeOp->hasAttr("hivm.tcore_type")) {
        auto attr = scopeOp->getAttr("hivm.tcore_type");
        if (attr == aiCAttr) {
            isAIC = true;
        }
    }
    auto bufferMap = getCounterOffset(forOp);
    SmallVector<Value> buf_vals;
    SmallVector<Value> if_conditions;
    builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());

    // 1. 提取并处理end值
    Value startValue = forOp.getLowerBound();
    Value endValue = forOp.getUpperBound();
    // 2. 提取并处理step值
    Value stepValue = forOp.getStep();
    builder.setInsertionPoint(forOp);
    Location loc = forOp.getLoc();
    int count = 0;
    for (auto &op : forOp.getBody()->getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto parentOp = ifOp->getParentOp();
        if (parentOp == forOp && ifOp->hasAttr("ssbuffer")) {
          count++;
        }
      }
    }

    Value two;
    Type ubType = stepValue.getType();
    if (ubType.isIndex()) {
        two = builder.create<arith::ConstantIndexOp>(loc, count - 1);
    } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
        // 对于整数类型，创建相应类型的常数2
        two = builder.create<arith::ConstantIntOp>(loc, intType, count - 1);
    } else {
        // 其他类型可能需要特殊处理
        llvm::errs() << "Warning: Unexpected type for upper bound: " << ubType << "\n";
        // 尝试创建索引类型的2然后转换
        auto indexTwo = builder.create<arith::ConstantIndexOp>(loc, count - 1);
        two = builder.create<arith::IndexCastOp>(loc, ubType, indexTwo);
    }

    auto steps = builder.create<arith::MulIOp>(
        forOp.getLoc(),
        endValue.getType(),
        stepValue,
        two
    );

    auto subLoopValue = builder.create<arith::SubIOp>(
        forOp.getLoc(),
        endValue.getType(),
        endValue,
        steps
    );

    SmallVector<bool> WaitType;
    SmallVector<Operation*> AllocType;
    SmallVector<Value> bufferPtrs;
    if (isAIC) {
        builder.setInsertionPointToStart(&forOp->getRegion(0).front());
        // 创建常量32和64
        Value c0 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 0, 32  // 值32，64位
        );
        Value c32 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 32, 64  // 值32，64位
        );
        Value c64 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 64, 64  // 值64，64位
        );
        // 创建inttoptr操作
        Value ssb_vec0_ptr = builder.create<LLVM::IntToPtrOp>(
            forOp.getLoc(),
            LLVM::LLVMPointerType::get(builder.getContext(), 11),  // 地址空间11
            c32
        );
        Value ssb_vec1_ptr = builder.create<LLVM::IntToPtrOp>(
            forOp.getLoc(),
            LLVM::LLVMPointerType::get(builder.getContext(), 11),  // 地址空间11
            c64
        );
        bufferPtrs.push_back(ssb_vec0_ptr);
        bufferPtrs.push_back(ssb_vec1_ptr);
        // 创建load操作
        Value status_vec0 = builder.create<LLVM::LoadOp>(
            forOp.getLoc(), builder.getI32Type(), ssb_vec0_ptr
        );
        
        Value status_vec1 = builder.create<LLVM::LoadOp>(
            forOp.getLoc(), builder.getI32Type(), ssb_vec1_ptr
        );

        getWaitType("cube", forOp, WaitType, AllocType);

        for (auto i = 0; i < WaitType.size(); i++) {
            auto correnspondAlloc = CubeBitMap[AllocType[i]];
            auto i32ConstAttr = mlir::IntegerAttr::get(builder.getI32Type(), 1 << correnspondAlloc);
            auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), builder.getI32Type(), i32ConstAttr);
            Value bufi_vec0_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), status_vec0, buf_constant_set
            );
            Value bufi_vec1_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), status_vec1, buf_constant_set
            );
            Value flag_bufi_vec0;
            Value flag_bufi_vec1;
            // 创建比较操作
            if (WaitType[i] == 0) {
                flag_bufi_vec0 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec0_val, c0
                );
                flag_bufi_vec1 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec1_val, c0
                );
            }
            else {
                flag_bufi_vec0 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec0_val, buf_constant_set
                );
                flag_bufi_vec1 = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_vec1_val, buf_constant_set
                );
            }
            // 创建最终的and操作
            Value bufi_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), flag_bufi_vec0, flag_bufi_vec1
            );
            buf_vals.push_back(bufi_val);
        }
        
    } else {
        builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
        Value c0 = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), 0, 32  // 值32，64位
        );
        auto i64Type = builder.getIntegerType(64);
        // %sub_id = hivm.hir.get_sub_block_idx -> i64
        // 这里假设有一个getSubBlockIdxOp操作
        Value subId = builder.create<GetSubBlockIdxOp>(
            scopeOp->getLoc(), i64Type);
        auto i64ConstAttr = mlir::IntegerAttr::get(i64Type, 32);
        Value cstOffset = builder.create<mlir::LLVM::ConstantOp>(
            scopeOp->getLoc(), i64Type, i64ConstAttr);
        Value ssbAddrOffset = builder.create<arith::MulIOp>(
          scopeOp->getLoc(), subId, cstOffset);
        Value ssbAddr = builder.create<arith::AddIOp>(
          scopeOp->getLoc(), ssbAddrOffset, cstOffset);
        builder.setInsertionPointToStart(&forOp->getRegion(0).front());
        // 创建inttoptr操作
        Value ssb_cube_ptr = builder.create<LLVM::IntToPtrOp>(
            forOp.getLoc(),
            LLVM::LLVMPointerType::get(builder.getContext(), 11),  // 地址空间11
          ssbAddr
        );
        bufferPtrs.push_back(ssb_cube_ptr);
        // 创建load操作
        Value status_cube = builder.create<LLVM::LoadOp>(
            forOp.getLoc(), builder.getI32Type(), ssb_cube_ptr
        );

        getWaitType("vector", forOp, WaitType, AllocType);
        for (auto i = 0; i < WaitType.size(); i++) {
            auto correnspondAlloc = VecBitMap[AllocType[i]];
            auto i32ConstAttr = mlir::IntegerAttr::get(builder.getI32Type(), 1 << correnspondAlloc);
            auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                scopeOp->getLoc(), builder.getI32Type(), i32ConstAttr);
            Value bufi_cube_val = builder.create<arith::AndIOp>(
                forOp.getLoc(), status_cube, buf_constant_set
            );

            Value flag_bufi_cube;
            // 创建比较操作
            if (WaitType[i] == 0) {
                flag_bufi_cube = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_cube_val, c0
                );
            }
            else {
                flag_bufi_cube = builder.create<arith::CmpIOp>(
                    forOp.getLoc(), arith::CmpIPredicate::eq, bufi_cube_val, buf_constant_set
                );
            }
            buf_vals.push_back(flag_bufi_cube);
        }
    }
    int bufIdx = 0;
    int groupIdx = 0;

    for (const auto &pair : bufferMap) {
        if (bufferMap[groupIdx] == 0) {
            continue;
        }
        
        // 获取对应的region迭代参数
        Value cnti = builder.create<arith::CmpIOp>(
            forOp.getLoc(), arith::CmpIPredicate::slt, 
            forOp.getRegionIterArgs()[forOp.getRegionIterArgs().size() - (bufferMap.size() - 1 - groupIdx)], 
            subLoopValue
        );
        
        // 计算该组中所有buffer值的AND
        Value finalBufVal = buf_vals[bufIdx];
        for (int count = 1; count < bufferMap[groupIdx]; count++) {
            finalBufVal = builder.create<arith::AndIOp>(
                forOp.getLoc(), finalBufVal, buf_vals[bufIdx + count]
            );
        }
        
        auto cond = builder.create<arith::AndIOp>(
            forOp.getLoc(), finalBufVal, cnti
        );
        if_conditions.push_back(cond);
        
        // 更新索引
        bufIdx += bufferMap[groupIdx];
        groupIdx++;
    }
    int ifIndex = 0;
    int acc = 0;
    int bufferBit = 0;
    for (int i = 0; i < CubeBitMap.size(); i++) {
        bufferBit += (1 << i);
    }
    forOp.getBody()->walk([&](Operation* op) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp && ifOp->hasAttr("ssbuffer")) {
        // 获取then区域
        Block* thenBlock = &ifOp.getThenRegion().front();
        
        // 找到then区域中的yield操作
        Operation* yieldOp = nullptr;
        for (auto& op : *thenBlock) {
            if (isa<scf::YieldOp>(op)) {
                yieldOp = &op;
                break;
            }
        }
        if (yieldOp) {
            builder.setInsertionPoint(yieldOp);
            
            if (isAIC) {
                // 创建插入的语句
                // %status_v2 = llvm.load %ssb_ptr : !llvm.ptr<11> -> i32
                Value status_v2_0 = builder.create<LLVM::LoadOp>(
                    yieldOp->getLoc(), 
                    builder.getIntegerType(32),  // i32类型
                    bufferPtrs[0]  // 假设ssb_ptr已在作用域中定义
                );
                Value status_v2_1 = builder.create<LLVM::LoadOp>(
                    yieldOp->getLoc(), 
                    builder.getIntegerType(32),  // i32类型
                    bufferPtrs[1]  // 假设ssb_ptr已在作用域中定义
                );
                Value buf_val_new_0 = status_v2_0;
                Value buf_val_new_1 = status_v2_1;
                auto bufferNum = bufferMap[ifIndex];
                for (int i = 0; i < bufferNum; i++) {
                    if (WaitType[acc + i] == 0) {
                        auto correnspondAlloc = CubeBitMap[AllocType[acc + i]];
                        auto i32ConstAttr = mlir::IntegerAttr::get(builder.getI32Type(), 1 << correnspondAlloc);
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), builder.getI32Type(), i32ConstAttr);
                        buf_val_new_0 = builder.create<arith::OrIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_0,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                        buf_val_new_1 = builder.create<arith::OrIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_1,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                    else {
                        auto correnspondAlloc = CubeBitMap[AllocType[acc + i]];
                        int bitPos = correnspondAlloc;
                        int basePattern = bufferBit;
                        int finalValue = basePattern ^ (1 << bitPos);
                        auto i32ConstAttr = mlir::IntegerAttr::get(builder.getI32Type(), finalValue);
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), builder.getI32Type(), i32ConstAttr);
                        buf_val_new_0 = builder.create<arith::AndIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_0,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                        buf_val_new_1 = builder.create<arith::AndIOp>(
                            yieldOp->getLoc(),
                            buf_val_new_1,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                }
                acc += bufferNum;
                builder.create<LLVM::StoreOp>(
                    yieldOp->getLoc(),
                    buf_val_new_0,
                    bufferPtrs[0]
                );
                builder.create<LLVM::StoreOp>(
                    yieldOp->getLoc(),
                    buf_val_new_1,
                    bufferPtrs[1]
                );
                
            }
            else {
                // 创建插入的语句
                // %status_v2 = llvm.load %ssb_ptr : !llvm.ptr<11> -> i32
                Value status_v2 = builder.create<LLVM::LoadOp>(
                    yieldOp->getLoc(), 
                    builder.getIntegerType(32),  // i32类型
                    bufferPtrs[0]  // 假设ssb_ptr已在作用域中定义
                );
                Value buf_val_new = status_v2;
                auto bufferNum = bufferMap[ifIndex];
                for (int i = 0; i < bufferNum; i++) {
                    if (WaitType[acc + i] == 0) {
                        auto correnspondAlloc = VecBitMap[AllocType[acc + i]];
                        auto i32ConstAttr = mlir::IntegerAttr::get(builder.getI32Type(), 1 << correnspondAlloc);
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), builder.getI32Type(), i32ConstAttr);
                        buf_val_new = builder.create<arith::OrIOp>(
                            yieldOp->getLoc(),
                            buf_val_new,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                    else {
                        auto correnspondAlloc = VecBitMap[AllocType[acc + i]];
                        int bitPos = correnspondAlloc;
                        int basePattern = bufferBit;
                        int finalValue = basePattern ^ (1 << bitPos);
                        auto i32ConstAttr = mlir::IntegerAttr::get(builder.getI32Type(), finalValue);
                        auto buf_constant_set = builder.create<mlir::LLVM::ConstantOp>(
                            scopeOp->getLoc(), builder.getI32Type(), i32ConstAttr);
                        buf_val_new = builder.create<arith::AndIOp>(
                            yieldOp->getLoc(),
                            buf_val_new,
                            buf_constant_set  // 假设buf3_clear已在作用域中定义
                        );
                    }
                }
                acc += bufferNum;
                builder.create<LLVM::StoreOp>(
                    yieldOp->getLoc(),
                    buf_val_new,
                    bufferPtrs[0]
                );
            }
            ifIndex ++;
        }
      }
    });

    return if_conditions;
}

void ReplaceIf(scf::ForOp forOp, SmallVector<Value> conditions, SmallVector<Operation*>& opsToErase, DenseMap<scf::IfOp, Value>& ifArgMap, OpBuilder &builder, ModuleOp moduleOp)
{
    SmallVector<scf::IfOp> ifToProcess;
    llvm::outs()<<"enter replaceif\n";
    Value step = forOp.getStep();
    auto aiCAttr = hivm::TCoreTypeAttr::get(
            builder.getContext(),
            hivm::TCoreType::CUBE);
    forOp.getBody()->walk([&](Operation* op) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp && ifOp->hasAttr("ssbuffer") && forOp == ifOp->getParentOp()) {
        ifToProcess.push_back(ifOp);
      }
    });

    IRMapping IRMap;
    for (int i = 0; i < ifToProcess.size(); i++) {
        auto ifOp = ifToProcess[i];
        auto parentOp = ifOp->getParentOp();
        auto loc = ifOp.getLoc();
        // 获取for循环的iterargs（迭代参数）
        auto iterArgs = forOp.getRegionIterArgs();
        if (iterArgs.size() < conditions.size()) {
            return;
        }
        auto thenYieldOp = dyn_cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
        SmallVector<Value> thenResults;
        if (thenYieldOp) {
            // 如果已有返回值，保留它们
            for (auto result : thenYieldOp.getResults()) {
                thenResults.push_back(result);
            }
        }
        // 创建新的else区域，返回两个迭代参数
        SmallVector<Value> elseResults;
        scf::YieldOp elseYieldOp = nullptr;
        bool hasElse = false;
        if (!ifOp.getElseRegion().empty()) {
          elseYieldOp = dyn_cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
          hasElse = true;
        }
        if (elseYieldOp) {
            for (auto result : elseYieldOp.getResults()) {
                elseResults.push_back(result);
            }
        }
        // 获取最后两个迭代参数
        Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
        // 创建新的then区域，返回两个迭代参数
        thenResults.push_back(iterArgMinus);
        elseResults.push_back(iterArgMinus);
        
        // 保存原有的操作，以便后续克隆
        SmallVector<Operation*> thenOps;
        for (auto &op : ifOp.getThenRegion().front()) {
            thenOps.push_back(&op);
        }
        
        SmallVector<Operation*> elseOps;
        if (!ifOp.getElseRegion().empty()) {
            for (auto &op : ifOp.getElseRegion().front()) {
                elseOps.push_back(&op);
            }
        }
        SmallVector<Type> resultTypes;
        for (auto val : thenResults) {
            resultTypes.push_back(val.getType());
        }
        // 创建新的scf.if操作
        builder.setInsertionPoint(ifOp);
        auto newIfOp = builder.create<scf::IfOp>(
            loc,
            resultTypes,
            conditions[i],
            /*withElseRegion=*/true);
        newIfOp->setAttr("ssbuffer", builder.getUnitAttr());
        // 处理then区域
        auto &newThenBlock = newIfOp.getThenRegion().front();
        builder.setInsertionPointToStart(&newThenBlock);
        
        // 克隆then区域的操作
        for (auto op : thenOps) {
            if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                // 处理yield的操作数映射
                SmallVector<Value> mappedOperands;
                for (auto operand : yieldOp->getOperands()) {
                    mappedOperands.push_back(IRMap.lookupOrDefault(operand));
                }
                // 获取最后两个迭代参数
                Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
        
                // %ssb_addr = arith.addi %ssb_addr_offset, %c32_i64 : i64
                auto AddIOp = builder.create<mlir::arith::AddIOp>(
                    forOp->getLoc(),
                    iterArgMinus,
                    step);
                // 这里加个add1
                mappedOperands.push_back(AddIOp);
                builder.create<scf::YieldOp>(loc, mappedOperands);
            } else {
                auto newOp = builder.clone(*op, IRMap);
                IRMap.map(op->getResults(), newOp->getResults());
            }
        }
        
        // 处理else区域
        auto &newElseBlock = newIfOp.getElseRegion().front();
        builder.setInsertionPointToStart(&newElseBlock);
        // 克隆else区域的操作
        if (hasElse) {
          for (auto op : elseOps) {
              if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                  // 处理yield的操作数映射
                  SmallVector<Value> mappedOperands;
                  for (auto operand : yieldOp->getOperands()) {
                      mappedOperands.push_back(IRMap.lookupOrDefault(operand));
                  }
                  Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
                  mappedOperands.push_back(iterArgMinus);
                  builder.create<scf::YieldOp>(loc, mappedOperands);
              } else {
                  auto newOp = builder.clone(*op, IRMap);
                  IRMap.map(op->getResults(), newOp->getResults());
              }
          }
        } else {
          SmallVector<Value> cntOperands;
          cntOperands.push_back(iterArgMinus);
          builder.create<scf::YieldOp>(loc, cntOperands);
        }
            
        // 替换原有if操作的使用
        // 首先，将原if操作的结果替换为新if操作的对应结果
        for (unsigned j = 0; j < ifOp.getNumResults(); ++j) {
            ifOp.getResult(j).replaceAllUsesWith(newIfOp.getResult(j));
        }
        // 获取新if操作所在的块
        Block* newIfBlock = ifOp->getBlock();
        // 在for循环体内替换迭代参数的使用
        forOp.getBody()->walk([&](Operation* op) {
            // 检查操作是否与新ifOp在同一个块中
            Block* opBlock = op->getBlock();
            if (opBlock != newIfBlock) {
                // 不在同一个块中，跳过
                return;
            }
            if (op->isBeforeInBlock(newIfOp)) {
                return; // 只处理if操作之后的use
            }
            for (unsigned j = 0; j < op->getNumOperands(); ++j) {
                for (auto argIndex = 0; argIndex < conditions.size(); argIndex ++) {
                    // 获取最后两个迭代参数
                    Value iterArgMinus = iterArgs[iterArgs.size() - (conditions.size() - i)];
                    if (op->getOperand(j) == iterArgMinus) {
                        op->setOperand(j, newIfOp.getResults()[newIfOp.getNumResults() - 1]);
                    }
                }
            }
        });
        
          // // 删除原有的if操作
          opsToErase.push_back(ifOp);
          if (ifArgMap.find(newIfOp) == ifArgMap.end()) {
                ifArgMap[newIfOp] = iterArgMinus;
            }
        }
}

int getNestingDepth(scf::ForOp forOp) {
    int depth = 0;
    Operation* op = forOp.getOperation();
    while (op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "scf") {
            ++depth;
        }
        op = op->getParentOp();
    }
    return depth;
}

void printDenseMap(const mlir::DenseMap<mlir::Operation*, int>& Map)
{
    for (const auto& pair : Map) {
        auto op = pair.first;
        auto bitValue = pair.second;

        // ✅ 正确打印 Operation 的方法：print(llvm::outs())
        llvm::outs() << "Operation: \n";
        llvm::outs().flush();
        llvm::outs() << "Operation Name: " << *op << "\n";
        
        // 打印对应的值
        llvm::outs() << "  |  Bit Value: " << bitValue << "  |  allocmap\n";
        llvm::outs().flush();
        llvm::outs() << "------------------------------------------------\n\n";
        llvm::outs().flush();
    }
    llvm::outs().flush();
}

void getAllocBit(ModuleOp module, DenseMap<Operation*, int>& VecBitMap,
                 DenseMap<Operation*, int> & CubeBitMap, OpBuilder builder)
{
    auto aiCAttr = hivm::TCoreTypeAttr::get(
        builder.getContext(),
        hivm::TCoreType::CUBE);
    auto scalarWaitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
    auto cubeWaitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_FIX);
    auto vectorWaitPipe = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_MTE3);

    SmallVector<scope::ScopeOp> scopeOpToEdit;
    module.walk([&](scope::ScopeOp scopeOp) {
      scopeOpToEdit.push_back(scopeOp);
    });

    int cubeAcc = 0;
    int vecAcc = 0;
    for (auto scopeOp : scopeOpToEdit) {
      SmallVector<Operation*> excludedValues;
      if (!scopeOp->hasAttr("hivm.tcore_type")) continue;

      auto attr = scopeOp -> getAttr("hivm.tcore_type");
      if (attr == aiCAttr) {
        // CUBE 类型：分配 cubeAcc 自增 ID
        scopeOp.walk([&](SyncBlockWaitOp waitOp) {
          auto parentOp = waitOp->getParentOp();
          if (isa<scf::IfOp>(parentOp) && parentOp->hasAttr("ssbuffer")) {
            auto waitPipe = waitOp.getPipe();
            if (waitPipe != scalarWaitPipe) {
              auto allocOp = findFirstTargetOpAfterWait(waitOp, excludedValues);
              if (VecBitMap.count(allocOp)) {
                CubeBitMap[allocOp] = VecBitMap[allocOp];
                cubeAcc++;
              } else {
                CubeBitMap[allocOp] = cubeAcc++;
              }
            }
          }
        });
      } else {
        // VEC 类型：分配 vecAcc 自增 ID
        scopeOp.walk([&](SyncBlockWaitOp waitOp) {
          auto parentOp = waitOp->getParentOp();
          if (isa<scf::IfOp>(parentOp) && parentOp->hasAttr("ssbuffer")) {
            auto waitPipe = waitOp.getPipe();
            if (waitPipe != scalarWaitPipe) {
              auto allocOp = findFirstTargetOpAfterWait(waitOp, excludedValues);
              if (!VecBitMap.count(allocOp)) {
                VecBitMap[allocOp] = vecAcc++;
              }
            }
          }
        });
      }
    }

    // 记录【自增ID的最大值】，用于后面偏移计算
    const int maxVecId = vecAcc - 2;
    const int maxCubeId = cubeAcc - 2;
    // ======================== 阶段 2：分配【偏移ID】给 waitOp（绝对不重复）========================
    for (auto scopeOp : scopeOpToEdit) {
      if (!scopeOp->hasAttr("hivm.tcore_type")) continue;

      auto attr = scopeOp->getAttr("hivm.tcore_type");
      if (attr == aiCAttr) {
        // CUBE：waitOp = maxCubeId + 1 + flagid
        scopeOp.walk([&](SyncBlockWaitOp waitOp) {
          auto parentOp = waitOp->getParentOp();
          if (isa<scf::IfOp>(parentOp) && parentOp->hasAttr("ssbuffer")) {
            auto setPipe = waitOp.getTpipe();
            auto waitPipe = waitOp.getPipe();
            if (setPipe != scalarWaitPipe && waitPipe == scalarWaitPipe) {
              uint32_t flagid = 0;
              if (auto attr = waitOp->getAttrOfType<mlir::IntegerAttr>("ssbuf.flagid")) {
                flagid = attr.getUInt();
              }
              // 核心修改：偏移分配，不再写死 16
              CubeBitMap[waitOp] = (maxCubeId >= 0 ? (maxCubeId + 1) : 0) + flagid;
            }
          }
        });
      } else {
        // VEC：waitOp = maxVecId + 1 + flagid
        scopeOp.walk([&](SyncBlockWaitOp waitOp) {
          auto parentOp = waitOp->getParentOp();
          if (isa<scf::IfOp>(parentOp) && parentOp->hasAttr("ssbuffer")) {
            auto setPipe = waitOp.getTpipe();
            auto waitPipe = waitOp.getPipe();
            if (setPipe != scalarWaitPipe && waitPipe == scalarWaitPipe) {
              uint32_t flagid = 0;
              if (auto attr = waitOp->getAttrOfType<mlir::IntegerAttr>("ssbuf.flagid")) {
                flagid = attr.getUInt();
              }
              // 核心修改：偏移分配
              VecBitMap[waitOp] = (maxVecId >= 0 ? (maxVecId + 1) : 0) + flagid;
            }
          }
        });
      }
    }
}

void modifyForIterargDeps(scf::ForOp forOp, DenseMap<scf::IfOp, Value> ifCounters)
{
  Value iterArg = forOp.getInductionVar();

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifCounters.find(ifOp) != ifCounters.end()) {
        Value counter = ifCounters[ifOp];

        ifOp.walk([&](Operation* opInIf) {
          for (auto [i, operand] : llvm::enumerate(opInIf->getOperands())) {
            if (operand == iterArg) {
                opInIf->setOperand(i, counter);
            }
          }
        });
      }
    }
  }
}

void FlowSssbuf(ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    // 收集所有需要转换的循环
    SmallVector<scf::ForOp> targetLoops;
    llvm::outs()<<"enter flowsssbuf\n\n";
    module.walk([&](Operation* op) {
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            // 检查循环是否包含特定的 sync_block_set 操作
            bool hasSyncBlockSet = false;
            forOp.walk([&](Operation *op) {
                if (isa<SyncBlockSetOp>(op)) {
                    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
                        if (forOp == ifOp->getParentOp() && ifOp->hasAttr("ssbuffer")) {
                            hasSyncBlockSet = true;
                        }
                    }
                }
            });
            
            if (hasSyncBlockSet) {
                if (llvm::find(targetLoops, forOp) == targetLoops.end()) {
                    targetLoops.push_back(forOp);
                }
            }
        }
    });
    llvm::outs()<<"enter flowsssbuf\n\n";

    SmallVector<scf::ForOp> transformLoops;
    // 转换每个目标循环
    for (scf::ForOp forOp : targetLoops) {
      auto newforOp = transformLoop(forOp, builder);
    }

    module.walk([&](Operation* op) {
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            // 检查循环是否包含特定的 sync_block_set 操作
            bool hasSyncBlockSet = false;
            forOp.walk([&](Operation *op) {
                if (isa<SyncBlockSetOp>(op)) {
                    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
                        if (forOp == ifOp->getParentOp() && ifOp->hasAttr("ssbuffer")) {
                            hasSyncBlockSet = true;
                        }
                    }
                }
            });
            
            if (hasSyncBlockSet) {
                if (llvm::find(transformLoops, forOp) == transformLoops.end()) {
                    transformLoops.push_back(forOp);
                }
            }
        }
      
    });

    llvm::sort(transformLoops, [](scf::ForOp a, scf::ForOp b) {
        return getNestingDepth(a) > getNestingDepth(b);
    });
    DenseMap<Operation*, int> VecBitMap;
    DenseMap<Operation*, int> CubeBitMap;
    getAllocBit(module, VecBitMap, CubeBitMap, builder);
    printDenseMap(CubeBitMap);
    printDenseMap(VecBitMap);
    SmallVector<Operation*> opsToErase;
    for (scf::ForOp forOp : transformLoops) {
        DenseMap<scf::IfOp, Value> ifArgMap;
        llvm::outs()<<"before replaceif\n";
        auto bufvals = addBufValLoop(forOp, VecBitMap, CubeBitMap, builder);
        ReplaceIf(forOp, bufvals, opsToErase, ifArgMap, builder, module);
        llvm::outs()<<"after replaceif\n";
        for (const auto& pair : ifArgMap) {
          auto val = pair.first;
          auto bitValue = pair.second;
          llvm::outs()<<val<<"  "<<bitValue<<"  ifargmrp\n\n\n";
          llvm::outs().flush();
        }

        modifyForIterargDeps(forOp, ifArgMap);
    }
    for (auto op : opsToErase) {
      op->erase();
    }

    
}

bool isTransOp(mlir::Operation *op) {
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op);
  if (fixpipeOp)
    return true;

  auto copyOp = dyn_cast<hivm::CopyOp>(op);
  if (!copyOp)
    return false;
  else {

      Value copySrc = copyOp.getODSOperands(0).front();
      MemRefType copySrcTy = dyn_cast<MemRefType>(copySrc.getType());
      auto SrcAddrSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(copySrcTy.getMemorySpace());
      bool isSrcUbSpace = SrcAddrSpace.getAddressSpace() == hivm::AddressSpace::UB;

      Value copyDst = copyOp.getODSOperands(1).front();
      MemRefType copyDstTy = dyn_cast<MemRefType>(copyDst.getType());
      auto DstAddrSpace = dyn_cast_or_null<hivm::AddressSpaceAttr>(copyDstTy.getMemorySpace());
      bool isDstCbufSpace = DstAddrSpace.getAddressSpace() == hivm::AddressSpace::L1;

      return isSrcUbSpace && isDstCbufSpace;
  }
}

void FindAndMarkBuffer(ModuleOp module) {
  OpBuilder builder(module.getContext());
  unsigned int BufferIdx = 0;
  Type idxType = builder.getI32Type();
  StringAttr setFlagAttr = builder.getStringAttr("Set flag");
  StringAttr waitFlagAttr = builder.getStringAttr("Wait flag");
  IntegerAttr idxAttr = builder.getI32IntegerAttr(BufferIdx);

  module.walk([&](mlir::Operation *op) {

    if (isTransOp(op)) {
      llvm::outs() << "Buffer idx" << BufferIdx << "\n";
      llvm::outs() << "Trans Op" << *op << "\n";
      Value SharedBuffer;
      if (auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op)) {
        SharedBuffer = fixpipeOp.getODSOperands(1).front();
      } else {
        auto copyOp = dyn_cast<hivm::CopyOp>(op);
        SharedBuffer = copyOp.getODSOperands(1).front();
      }
      llvm::outs() << "SharedBuffer" << SharedBuffer << "\n";

      if (!SharedBuffer) {
        op->emitWarning("fixpipe op has empty output operand!");
        return;
      }

      // 在Buffer的生产op后set flag标记，在Buffer消费op前增加wait flag标记
      op->setAttr("Buffer idx", builder.getI32IntegerAttr(BufferIdx));
      op->setAttr("Wait Flag", builder.getI32IntegerAttr(0));
      op->setAttr("Set Flag", builder.getI32IntegerAttr(1));

      for (Operation *consumerOp : SharedBuffer.getUsers()) {
        if (consumerOp == op) 
          continue;
        if (!consumerOp) continue;
        
        llvm::outs() << "consumerOp: " << *consumerOp << "\n";
        
        consumerOp->setAttr("Buffer idx", builder.getI32IntegerAttr(BufferIdx));
        consumerOp->setAttr("Wait Flag", builder.getI32IntegerAttr(0));
      }
      BufferIdx++;
    }
  });
}

// 结构体存 wait-set 区块信息
struct WaitSetRegion {
  Operation *waitOp;
  Operation *lastSetOp;
  SmallVector<Operation *> opsToMove;
  bool hasCopyOrFixpipe = false;
};

struct MergedRegion {
  SmallVector<WaitSetRegion *> regions;
  SmallVector<Operation *> opsToMove;
  SmallVector<Value> yieldValues;
  SmallVector<Type> resultTypes;
};

void MoveIterArgUsersIntoIf(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // iter_arg -> mergedRegion index
  DenseMap<BlockArgument, int> iterArgToRegion;

  for (int r = 0; r < mergedRegions.size(); ++r) {
    MergedRegion &mr = mergedRegions[r];

    for (Operation *op : mr.opsToMove) {
      for (Value v : op->getOperands()) {
        if (auto barg = mlir::dyn_cast<BlockArgument>(v)) {
          if (barg.getOwner() == &body) {
            iterArgToRegion.try_emplace(barg, r);
          }
        }
      }
    }
  }

  if (iterArgToRegion.empty())
    return;

  // 找最后一个 mergedRegion 的最后一个 op
  Operation *lastOp = nullptr;
  for (MergedRegion &mr : mergedRegions)
    lastOp = mr.opsToMove.back();

  if (!lastOp)
    return;

  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  int startIdx = opIndex[lastOp] + 1;

  // 扫描 for body 尾部 op
  for (Operation &op : body) {
    if (opIndex[&op] < startIdx)
      continue;

    llvm::SmallDenseSet<int, 2> usedRegions;
    for (Value v : op.getOperands()) {
      if (auto barg = mlir::dyn_cast<BlockArgument>(v)) {
        auto it = iterArgToRegion.find(barg);
        if (it != iterArgToRegion.end())
          usedRegions.insert(it->second);
      }
    }

    // 必须且只能依赖一个 mergedRegion
    if (usedRegions.size() != 1)
      continue;

    int target = *usedRegions.begin();

    mergedRegions[target].opsToMove.push_back(&op);
  }
}

void ComputeYieldForMergedRegion(
    MergedRegion &mr, Block &body) {

  mr.yieldValues.clear();
  mr.resultTypes.clear();

  SmallPtrSet<Operation *, 32> inRegion(
      mr.opsToMove.begin(), mr.opsToMove.end());

  for (Operation *op : mr.opsToMove) {
    for (Value res : op->getResults()) {
      bool usedOutside = false;

      for (OpOperand &use : res.getUses()) {
        Operation *user = use.getOwner();

        // 不在同一个 for body，交给外层处理（通常不会出现）
        if (user->getBlock() != &body)
          continue;

        // 只要有一个 use 在 region 外，就必须 yield
        if (!inRegion.contains(user)) {
          usedOutside = true;
          break;
        }
      }

      if (usedOutside) {
        mr.yieldValues.push_back(res);
        mr.resultTypes.push_back(res.getType());
      }
    }
  }
}

static void ComputeYieldForMergedRegionV2(
    MergedRegion &mr, Block &body) {

  mr.yieldValues.clear();
  mr.resultTypes.clear();

  // 当前 region 内的 ops
  SmallPtrSet<Operation *, 32> inRegion(
      mr.opsToMove.begin(), mr.opsToMove.end());

  for (Operation *op : mr.opsToMove) {
    for (Value res : op->getResults()) {

      bool usedOutside = false;

      for (OpOperand &use : res.getUses()) {
        Operation *user = use.getOwner();

        // 如果使用在 region 内部 op，跳过
        if (inRegion.contains(user))
          continue;

        // 使用在 region 外部，包括嵌套 region 内部的 block
        usedOutside = true;
        break;
      }

      if (usedOutside) {
        mr.yieldValues.push_back(res);
        mr.resultTypes.push_back(res.getType());
      }
    }
  }
}

static void ComputeYieldForMergedRegionV3(MergedRegion &mr) {
    mr.yieldValues.clear();
    mr.resultTypes.clear();

    // 用 DenseSet 暂存当前 region 的所有 ops
    DenseSet<Operation *> regionOps(mr.opsToMove.begin(), mr.opsToMove.end());

    for (Operation *op : mr.opsToMove) {
        for (Value res : op->getResults()) {

            bool needsYield = false;

            for (OpOperand &use : res.getUses()) {
                Operation *user = use.getOwner();

                // 如果 user 不在当前 region，则需要 yield
                if (!regionOps.contains(user)) {
                    needsYield = true;
                    break;
                }
            }

            if (needsYield) {
                mr.yieldValues.push_back(res);
                mr.resultTypes.push_back(res.getType());
            }
        }
    }
}

// 递归收集 op 和它所有 region 内的 ops
static void CollectAllNestedOps(Operation *op, DenseSet<Operation *> &regionOps) {
    if (!op)
        return;
    
    if (regionOps.contains(op))
        return; // 已经收集过

    regionOps.insert(op);

    // 遍历所有 region，递归收集
    for (Region &region : op->getRegions()) {
        for (Block &block : region) {
            for (Operation &nestedOp : block) {
                CollectAllNestedOps(&nestedOp, regionOps);
            }
        }
    }
}

static void ComputeYieldForMergedRegionV4(MergedRegion &mr) {
    mr.yieldValues.clear();
    mr.resultTypes.clear();

    // 用 DenseSet 暂存当前 region 的所有 ops
    // 初始 DenseSet: 顶层 opsToMove
    DenseSet<Operation *> regionOps;
    for (Operation *op : mr.opsToMove) {
        CollectAllNestedOps(op, regionOps); // 完整展开嵌套
    }

    for (Operation *op : mr.opsToMove) {
        for (Value res : op->getResults()) {

            bool needsYield = false;

            for (OpOperand &use : res.getUses()) {
                Operation *user = use.getOwner();

                // 如果 user 不在当前 region，则需要 yield
                if (!regionOps.contains(user)) {
                    needsYield = true;
                    break;
                }
            }

            if (needsYield) {
                mr.yieldValues.push_back(res);
                mr.resultTypes.push_back(res.getType());
            }
        }
    }
}

int findTargetRegion(
    Operation *startOp,
    Block &body,
    DenseMap<Operation *, int> &opToRegion) {

  SmallVector<Operation *> worklist{startOp};
  SmallPtrSet<Operation *, 16> visited;

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;

    auto it = opToRegion.find(op);
    if (it != opToRegion.end())
      return it->second;

    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      Operation *defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == &body)
        worklist.push_back(defOp);
    }
  }

  return -1;
}

void greedyAbsorbToRegion(
    Operation *startOp,
    int regionIdx,
    int lowerBound,
    Block &body,
    DenseMap<Operation *, int> &opIndex,
    DenseMap<Operation *, int> &opToRegion,
    SmallVector<MergedRegion> &mergedRegions) {

  auto &mr = mergedRegions[regionIdx];

  SmallVector<Operation *> worklist;
  SmallPtrSet<Operation *, 32> visited(
      mr.opsToMove.begin(), mr.opsToMove.end());

  // 先把 startOp 本身吸收（如果还没被吸收）
  if (!opToRegion.count(startOp)) {
    mr.opsToMove.push_back(startOp);
    opToRegion[startOp] = regionIdx;
    visited.insert(startOp);
  }

  worklist.push_back(startOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      Operation *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != &body)
        continue;

      int defIdx = opIndex[defOp];

      // 超过前一个 region 的末尾
      if (defIdx < lowerBound)
        continue;

      auto it = opToRegion.find(defOp);

      // 不能跨到其他 region
      if (it != opToRegion.end() &&
          it->second != regionIdx)
        continue;

      // 去重
      if (!visited.insert(defOp).second)
        continue;

      // 吸收 defOp
      mr.opsToMove.push_back(defOp);
      opToRegion[defOp] = regionIdx;
      worklist.push_back(defOp);
    }
  }
}

SmallVector<Value> getOperationInput(Operation *op, SmallVector<Value> dependValues,
    DenseMap<Value, std::pair<Value, SmallVector<Operation *>>> &collectDepValueMap)
{
  // Analyse each Op's input
  DenseSet<Value> opInput;
  if (isa<scf::IfOp>(op) || isa<scf::ForOp>(op)) {
    SmallVector<Block *> regionBlocks;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      regionBlocks.push_back(&(ifOp.getThenRegion().front()));
      regionBlocks.push_back(&(ifOp.getElseRegion().front()));
    } else {
      auto forOp = dyn_cast<scf::ForOp>(op);
      regionBlocks.push_back(forOp.getBody());
    }

    // recursively walk scf op
    for (Block *curBlock: regionBlocks) {
      for (auto &curOp : *curBlock) {
        for (auto operand : getOperationInput(&curOp, dependValues, collectDepValueMap)) {
          Operation *defOp;
          if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            Block* ownerBlock = blockArg.getOwner();
            defOp = ownerBlock->getParentOp();
          } else {
            defOp = operand.getDefiningOp();
          }
          Block *defBlock = defOp->getBlock();

          if (!(defOp == op || llvm::is_contained(regionBlocks, defBlock))) {
            opInput.insert(operand);
          }
        }
      }
    }
    SmallVector<Value> retVector(opInput.begin(), opInput.end());
    return retVector;
  } else {
    SmallVector<Value> operands = op->getOperands();
    // store ifresult value that will be replaced
    for (auto operand : operands) {
      if (llvm::is_contained(dependValues, operand)) {
        if (collectDepValueMap.find(operand) != collectDepValueMap.end()) {
          collectDepValueMap[operand].second.push_back(op);
        } else {
          SmallVector<Operation *> userOps;
          userOps.push_back(op);
          collectDepValueMap[operand] = {operand, userOps};
        }
      }
    }
    return operands;
  }
}

SmallVector<Operation *> collectDepValuesCalculation(DenseSet<Operation *> forRegionOps,
    DenseSet<Operation *> regionOps, Operation *op, SmallVector<Value> dependValues,
    DenseMap<Value, std::pair<Value, SmallVector<Operation*>>> &collectDepValueMap)
{
  DenseSet<Operation *> collectOps;
  std::deque<Operation *> opStack;
  bool flag = false;
  
  opStack.push_back(op);
  while (opStack.size()) {
    Operation *curOp = opStack.front();
    opStack.pop_front();

    for (auto operand : getOperationInput(curOp, dependValues, collectDepValueMap)) {
      if (llvm::is_contained(dependValues, operand)) {
        flag = true;
      }

      Operation *parentOp = operand.getDefiningOp();
      if (llvm::is_contained(regionOps, parentOp)) {
        opStack.push_back(parentOp);
        continue;
      } else if (llvm::is_contained(forRegionOps, parentOp)) {
        opStack.push_back(parentOp);
        collectOps.insert(parentOp);
      }
    }
  }

  if (flag) {
    SmallVector<Operation *> retVector(collectOps.begin(), collectOps.end());
    return retVector;
  } else {
    collectDepValueMap.clear();
    SmallVector<Operation *> emptyVector;
    emptyVector.clear();
    return emptyVector;
  }
}

void copyOpsToMergedRegion(scf::ForOp forOp, SmallVector<Operation *> collectOps, MergedRegion &mergedRegion,
    DenseMap<Value, std::pair<Value, SmallVector<Operation*>>> &collectDepValueMap)
{
  Block *forBodyBlock = forOp.getBody();
  OpBuilder builder(forOp);
  SmallVector<Operation *> clonedOps;
  IRMapping mapper;

  // copy calculation of ifreult value related to load/store op
  int cnt = 0;
  for (Operation &origOp : forBodyBlock->without_terminator()) {
    if (cnt >= collectOps.size())
      break;

    if (llvm::is_contained(collectOps, &origOp)) {
      builder.setInsertionPointAfter(&origOp);

      Operation *clonedOp = (&origOp)->clone(mapper);
      builder.insert(clonedOp);
      mapper.map(&origOp, clonedOp);

      clonedOps.push_back(clonedOp);
      cnt++;

      // replace the ifresult value by new cloned op's result
      SmallVector<Value> results = origOp.getResults();
      for (auto [idx, result] : llvm::enumerate(origOp.getResults())) {
        if (collectDepValueMap.find(result) != collectDepValueMap.end()) {
          collectDepValueMap[result].first = clonedOp->getResult(idx);
        }
      }
    }
  }

  DenseSet<Operation *> mergedRegionOps;
  for (Operation *op : mergedRegion.opsToMove) {
      CollectAllNestedOps(op, mergedRegionOps);
  }

  // replace the ifresult value by new cloned op's result
  for (Operation *op : mergedRegionOps) {
    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      if (collectDepValueMap.find(operand) != collectDepValueMap.end()) {
        op->setOperand(idx, collectDepValueMap[operand].first);
      }
    }
  }

  // update MergedRegion
  clonedOps.append(mergedRegion.opsToMove);
  mergedRegion.opsToMove = clonedOps;
}

void copyLoadCalculation(scf::ForOp forOp, SmallVector<Value> dependValues, SmallVector<MergedRegion> &mergedRegions)
{
  mlir::Operation* parentOp = forOp->getParentOp();
  mlir::Operation* scopeOp = nullptr;
  while (parentOp) {
      if (dyn_cast<scope::ScopeOp>(parentOp)) {
          scopeOp = parentOp;
          break;
      }
      parentOp = parentOp->getParentOp();
  }
  auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name);
  // only process the vector core
  if (coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE) {
    return;
  }

  // recursively collect all op in forOp
  DenseSet<Operation *> forRegionOps;
  for (Operation &op : forOp.getBody()->without_terminator()) {
      CollectAllNestedOps(&op, forRegionOps);
  }
  
  for (MergedRegion &mr : mergedRegions) {
    DenseSet<Operation *> regionOps;
    for (Operation *op : mr.opsToMove) {
        CollectAllNestedOps(op, regionOps);
    }

    for (Operation *op : regionOps) {
      if (isa<triton::StoreOp>(op) || isa<triton::LoadOp>(op)) {
        // recusively check that whether load/store op's operands originated from if results
        DenseMap<Value, std::pair<Value, SmallVector<Operation*>>> collectDepValueMap;
        SmallVector<Operation *> collectOps = \
            collectDepValuesCalculation(forRegionOps, regionOps, op, dependValues, collectDepValueMap);
        copyOpsToMergedRegion(forOp, collectOps, mr, collectDepValueMap);
      }
    }
  }
}

// 以 forOp 的 yield value 为中心
// 决定它应该归属哪个 mergedRegion, 然后再向前吸 operand
void ExpandMergedRegionOpsForAIV(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // 记录 block 中 op 顺序
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  // 建立 op -> region 映射
  DenseMap<Operation *, int> opToRegion;
  for (int r = 0; r < mergedRegions.size(); ++r)
    for (Operation *op : mergedRegions[r].opsToMove)
      opToRegion[op] = r;

  // 取 scf.yield
  auto yieldOp =
      cast<scf::YieldOp>(body.getTerminator());

  // 依次处理每个 yield value（按编号顺序）
  for (Value yv : yieldOp.getOperands()) {

    Operation *defOp = yv.getDefiningOp();
    if (!defOp || defOp->getBlock() != &body)
      continue;

    int targetRegion = -1;

    // 如果已经在 region 中
    auto it = opToRegion.find(defOp);
    if (it != opToRegion.end()) {
      targetRegion = it->second;
    } else {
      // 否则向前搜索确定归属
      targetRegion =
          findTargetRegion(defOp, body, opToRegion);
    }

    if (targetRegion == -1)
      continue;

    // 计算边界 lowerBound
    int lowerBound = 0;

    if (targetRegion > 0) {
      Operation *prevLast =
          mergedRegions[targetRegion - 1]
              .opsToMove.back();
      lowerBound = opIndex[prevLast] + 1;
    }

    // 真正贪心吸收
    greedyAbsorbToRegion(defOp,
                         targetRegion,
                         lowerBound,
                         body,
                         opIndex,
                         opToRegion,
                         mergedRegions);
  }

  // 每个 region 内按 block 顺序排序
  for (auto &mr : mergedRegions) {
    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

// 以 mergedRegion 为中心, 向前吸 operand
void ExpandMergedRegionOpsForAIC(scf::ForOp forOp,
                           SmallVector<MergedRegion> &mergedRegions) {
  Block &body = forOp.getRegion().front();

  // 记录每个 mergedRegion 的起始 op index
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body) {
    opIndex[&op] = idx++;
  }

  for (int r = 0; r < mergedRegions.size(); ++r) {
    MergedRegion &mr = const_cast<MergedRegion &>(mergedRegions[r]);

    // 本 mergedRegion 的最早 op
    Operation *firstOp = mr.opsToMove.front();
    int lowerBound = 0;

    // 边界: 前一个 mergedRegion 的最后一个 op
    if (r > 0) {
      Operation *prevLast =
          mergedRegions[r - 1].opsToMove.back();
      lowerBound = opIndex[prevLast] + 1;
    }

    SmallVector<Operation *> worklist(mr.opsToMove.begin(),
                                      mr.opsToMove.end());
    SmallPtrSet<Operation *, 32> visited(
        mr.opsToMove.begin(), mr.opsToMove.end());

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      
      // 往前吸收operand
      for (Value operand : op->getOperands()) {
        // BlockArgument
        if (mlir::isa<BlockArgument>(operand))
          continue;

        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;

        // 不在 for body
        if (defOp->getBlock() != &body)
          continue;

        int defIdx = opIndex[defOp];

        // 超出允许向前吸收的边界
        if (defIdx < lowerBound)
          continue;

        // 已经在 opsToMove
        if (!visited.insert(defOp).second)
          continue;

        // 吸收这个 defOp
        mr.opsToMove.push_back(defOp);
        worklist.push_back(defOp);
      }
    }

    // 最后按原 block 顺序排序
    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

static void pullInRegionDependencies(
    Operation *regionOp,
    int regionId,
    DenseMap<Operation *, int> &opToRegion,
    Block &body) {

  SmallVector<Operation *> worklist;

  // 先把 region 内的 op 放进去
  for (Region &region : regionOp->getRegions())
    for (Block &block : region)
      for (Operation &inner : block)
        worklist.push_back(&inner);

  SmallPtrSet<Operation *, 32> visited;

  while (!worklist.empty()) {
    Operation *innerOp = worklist.pop_back_val();

    if (!visited.insert(innerOp).second)
      continue;

    // operand 的 defining op
    for (Value operand : innerOp->getOperands()) {

      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;

      if (def->getBlock() != &body)
        continue;

      if (!opToRegion.count(def)) {

        opToRegion[def] = regionId;

        // 如果 def 也是 region-op，继续扩展
        if (def->getNumRegions() > 0)
          worklist.push_back(def);
      }
    }

    // 继续遍历 region
    for (Region &r : innerOp->getRegions())
      for (Block &b : r)
        for (Operation &child : b)
          worklist.push_back(&child);
  }
}

// BFS 查找某个 op 最早被哪个 region 使用
static int findEarliestRegion(
    Operation *startOp,
    const DenseMap<Operation *, int> &seedRegionMap,
    Block &body) {

  SmallVector<Operation *> worklist{startOp};
  SmallPtrSet<Operation *, 32> visited;
  int earliestRegion = -1;

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!visited.insert(op).second)
      continue;

    for (Value result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();

        if (user->getBlock() != &body)
          continue;

        auto it = seedRegionMap.find(user);
        if (it != seedRegionMap.end()) {
          int region = it->second;
          if (earliestRegion == -1 || region < earliestRegion)
            earliestRegion = region;
        } else {
          worklist.push_back(user);
        }
      }
    }
  }

  return earliestRegion;
}

void ExpandMergedRegionOpsForAll(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // block 内 op 顺序
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  // seed region map
  DenseMap<Operation *, int> seedRegionMap;
  for (int r = 0; r < mergedRegions.size(); r++) {
    for (Operation *op : mergedRegions[r].opsToMove) {
      seedRegionMap[op] = r;
    }
  }

  // 最终 op -> region
  DenseMap<Operation *, int> opToRegion = seedRegionMap;

  // ---------- Step1 顺序扫描 ----------
  for (Operation &op : body) {

    if (isa<scf::YieldOp>(&op))
      continue;

    if (opToRegion.count(&op))
      continue;

    int region = findEarliestRegion(&op, seedRegionMap, body);

    if (region != -1)
      opToRegion[&op] = region;
  }

  // ---------- Step2 region-op 依赖补全 ----------
  for (Operation &op : body) {

    auto it = opToRegion.find(&op);
    if (it == opToRegion.end())
      continue;

    if (op.getNumRegions() == 0)
      continue;

    pullInRegionDependencies(&op, it->second, opToRegion, body);
  }

  // ---------- Step3 append op ----------
  SmallPtrSet<Operation *, 32> seen;

  for (Operation &op : body) {

    auto it = opToRegion.find(&op);
    if (it == opToRegion.end())
      continue;

    if (!seen.insert(&op).second)
      continue;

    int region = it->second;
    mergedRegions[region].opsToMove.push_back(&op);
  }

  // ---------- Step4 排序 ----------
  for (auto &mr : mergedRegions) {

    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

void ExpandMergedRegionOpsByInput(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // block 内 op 顺序
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  // seed region map
  DenseMap<Operation *, int> seedRegionMap;
  for (int r = 0; r < mergedRegions.size(); r++) {
    for (Operation *op : mergedRegions[r].opsToMove) {
      seedRegionMap[op] = r;
    }
  }

  // 最终 op -> region
  DenseMap<Operation *, int> opToRegion = seedRegionMap;

  // ---------- Step1 顺序扫描 ----------
  for (Operation &op : body) {

    if (isa<scf::YieldOp>(&op))
      continue;

    if (opToRegion.count(&op))
      continue;

    int region = findEarliestRegion(&op, seedRegionMap, body);

    if (region != -1)
      opToRegion[&op] = region;
  }

  // ---------- Step2 region-op 依赖补全 ----------
  for (Operation &op : body) {

    auto it = opToRegion.find(&op);
    if (it == opToRegion.end())
      continue;

    if (op.getNumRegions() == 0)
      continue;

    pullInRegionDependencies(&op, it->second, opToRegion, body);
  }

  // ---------- Step3 append op ----------
  SmallPtrSet<Operation *, 32> seen;

  for (Operation &op : body) {

    auto it = opToRegion.find(&op);
    if (it == opToRegion.end())
      continue;

    if (!seen.insert(&op).second)
      continue;

    int region = it->second;
    mergedRegions[region].opsToMove.push_back(&op);
  }

  // ---------- Step4 排序 ----------
  for (auto &mr : mergedRegions) {

    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

static void ExpandMergedRegionOpsByOutput(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // block 顺序（保持 IR 顺序）
  DenseMap<Operation *, int> opOrder;
  int idx = 0;
  for (Operation &op : body)
    opOrder[&op] = idx++;

  for (auto &merged : mergedRegions) {

    // 收集 region 当前产生的 value
    SmallPtrSet<Value, 32> regionValues;

    for (Operation *op : merged.opsToMove)
      for (Value res : op->getResults())
        regionValues.insert(res);

    bool changed = true;

    while (changed) {
      changed = false;

      for (Operation &op : body) {

        if (isa<scf::IfOp>(op) || isa<scf::YieldOp>(op))
          continue;

        if (llvm::is_contained(merged.opsToMove, &op))
          continue;

        bool depends = false;

        for (Value operand : op.getOperands()) {
          if (regionValues.contains(operand)) {
            depends = true;
            break;
          }
        }

        if (!depends)
          continue;

        // 加入 region
        merged.opsToMove.push_back(&op);

        // 更新 regionValues
        for (Value res : op.getResults())
          regionValues.insert(res);

        changed = true;
      }
    }

    // 排序保持原 block 顺序
    llvm::sort(merged.opsToMove,
      [&](Operation *a, Operation *b) {
        return opOrder[a] < opOrder[b];
      });
  }
}

static void MoveIndependentOpsIntoIf(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // 记录哪些 op 已经在 region 里
  SmallPtrSet<Operation *, 32> alreadyAssigned;

  for (auto &mr : mergedRegions)
    for (Operation *op : mr.opsToMove)
      alreadyAssigned.insert(op);

  // 记录 iter_arg -> region
  DenseMap<Value, int> iterArgToRegion;

  for (int r = 0; r < mergedRegions.size(); r++) {
    for (Operation *op : mergedRegions[r].opsToMove) {

      for (Value operand : op->getOperands()) {

        if (auto barg = mlir::dyn_cast<BlockArgument>(operand)) {

          if (barg.getOwner() == &body)
            iterArgToRegion[barg] = r;
        }
      }
    }
  }

  // block 顺序
  DenseMap<Operation *, int> opIndex;
  int idx = 0;
  for (Operation &op : body)
    opIndex[&op] = idx++;

  // 扫描所有 op
  for (Operation &op : body) {

    if (isa<scf::IfOp>(op) || isa<scf::YieldOp>(op))
      continue;

    if (alreadyAssigned.contains(&op))
      continue;

    int targetRegion = -1;

    // 看 operand 是否来自 iter_arg
    for (Value operand : op.getOperands()) {

      if (auto barg = mlir::dyn_cast<BlockArgument>(operand)) {

        if (barg.getOwner() != &body)
          continue;

        auto it = iterArgToRegion.find(barg);
        if (it != iterArgToRegion.end()) {

          targetRegion = it->second;
          break;
        }
      }
    }

    if (targetRegion == -1)
      continue;

    mergedRegions[targetRegion].opsToMove.push_back(&op);
    alreadyAssigned.insert(&op);
  }

  // 排序保持 block 顺序
  for (auto &mr : mergedRegions) {

    llvm::sort(mr.opsToMove,
               [&](Operation *a, Operation *b) {
                 return opIndex[a] < opIndex[b];
               });
  }
}

// 暴力包裹
static void ExpandMergedRegionOpsGreedyMaximum(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // 记录哪些 op 已经属于 region
  DenseSet<Operation *> regionOps;

  for (auto &region : mergedRegions)
    for (Operation *op : region.opsToMove)
      regionOps.insert(op);

  // block op 列表
  SmallVector<Operation *> ops;
  for (Operation &op : body)
    ops.push_back(&op);

  DenseMap<Operation *, int> opIndex;
  for (int i = 0; i < ops.size(); i++)
    opIndex[ops[i]] = i;

  for (auto &region : mergedRegions) {

    if (region.opsToMove.empty())
      continue;

    // 找到 region 在 block 中的范围
    int start = ops.size();
    int end = -1;

    for (Operation *op : region.opsToMove) {
      int idx = opIndex[op];
      start = std::min(start, idx);
      end = std::max(end, idx);
    }

    SmallVector<Operation *> newOps;

    // ---------- backward 扩展 ----------
    for (int i = start - 1; i >= 0; i--) {
      Operation *op = ops[i];

      if (isa<scf::YieldOp>(op))
        break;

      if (regionOps.contains(op))
        break;

      newOps.push_back(op);
    }

    // ---------- forward 扩展 ----------
    for (int i = end + 1; i < ops.size(); i++) {
      Operation *op = ops[i];

      if (isa<scf::YieldOp>(op))
        break;

      if (regionOps.contains(op))
        break;

      newOps.push_back(op);
    }

    // 加入 region
    for (Operation *op : newOps) {
      region.opsToMove.push_back(op);
      regionOps.insert(op);
    }
  }

  // 最后保持 block 顺序
  for (auto &region : mergedRegions) {

    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(), region.opsToMove.end()),
        region.opsToMove.end());
  }
}

static void CollectForYieldRelatedOps(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions,
    DenseSet<Operation *> &yieldRelatedOps) {

  Block &body = forOp.getRegion().front();

  // 已经属于 region 的 op
  DenseSet<Operation *> regionOps;
  for (auto &region : mergedRegions)
    for (Operation *op : region.opsToMove)
      regionOps.insert(op);

  auto yield = cast<scf::YieldOp>(body.getTerminator());

  SmallVector<Value> worklist;
  DenseSet<Value> visited;

  // 初始化 worklist
  for (Value v : yield.getOperands())
    worklist.push_back(v);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();

    if (!visited.insert(v).second)
      continue;

    Operation *def = v.getDefiningOp();
    if (!def)
      continue;

    // 只处理 for body 内的 op
    if (def->getBlock() != &body)
      continue;

    // 已经在 region 内
    if (regionOps.contains(def))
      continue;

    // 记录
    if (yieldRelatedOps.insert(def).second) {

      // 继续向上找依赖
      for (Value operand : def->getOperands())
        worklist.push_back(operand);
    }
  }
}

// 贪心吸收region前后的op
static void ExpandMergedRegionOpsGreedy(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions,
    DenseSet<Operation *> &skipOps) {

  Block &body = forOp.getRegion().front();

  // 记录哪些 op 已经属于 region
  DenseSet<Operation *> regionOps;
  for (auto &region : mergedRegions)
    for (Operation *op : region.opsToMove)
      regionOps.insert(op);

  // block op 列表
  SmallVector<Operation *> ops;
  for (Operation &op : body)
    ops.push_back(&op);

  // op -> index
  DenseMap<Operation *, int> opIndex;
  for (int i = 0; i < ops.size(); i++)
    opIndex[ops[i]] = i;

  for (auto &region : mergedRegions) {

    if (region.opsToMove.empty())
      continue;

    // 找到 region 在 block 中的范围
    int start = ops.size();
    int end = -1;

    for (Operation *op : region.opsToMove) {
      int idx = opIndex[op];
      start = std::min(start, idx);
      end = std::max(end, idx);
    }

    SmallVector<Operation *> newOps;

    // ---------- backward 扩展 ----------
    for (int i = start - 1; i >= 0; i--) {
      Operation *op = ops[i];

      // block terminator
      if (isa<scf::YieldOp>(op))
        break;

      // 遇到其他 region 的 op
      if (regionOps.contains(op))
        break;

      // yield 关联 op，跳过但继续扫描
      if (skipOps.contains(op))
        continue;

      newOps.push_back(op);
    }

    // ---------- forward 扩展 ----------
    for (int i = end + 1; i < ops.size(); i++) {
      Operation *op = ops[i];

      // block terminator
      if (isa<scf::YieldOp>(op))
        break;

      // 遇到其他 region 的 op
      if (regionOps.contains(op))
        break;

      // yield 关联 op，跳过
      if (skipOps.contains(op))
        continue;

      newOps.push_back(op);
    }

    // 加入 region
    for (Operation *op : newOps) {
      region.opsToMove.push_back(op);
      regionOps.insert(op);
    }
  }

  // 最后保持 block 顺序
  for (auto &region : mergedRegions) {

    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(), region.opsToMove.end()),
        region.opsToMove.end());
  }
}

// 贪心吸收region前面的op
static void ExpandMergedRegionOpsGreedyV2(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions,
    DenseSet<Operation *> &skipOps) {

  Block &body = forOp.getRegion().front();

  // block op 列表
  SmallVector<Operation *> ops;
  for (Operation &op : body)
    ops.push_back(&op);

  // op -> index
  DenseMap<Operation *, int> opIndex;
  for (int i = 0; i < ops.size(); i++)
    opIndex[ops[i]] = i;

  // 记录哪些 op 已经属于 region
  DenseSet<Operation *> regionOps;
  for (auto &region : mergedRegions)
    for (Operation *op : region.opsToMove)
      regionOps.insert(op);

  for (int r = 0; r < mergedRegions.size(); r++) {

    auto &region = mergedRegions[r];
    if (region.opsToMove.empty())
      continue;

    // ---------- 当前 region block 范围 ----------
    int start = ops.size();
    int end = -1;

    for (Operation *op : region.opsToMove) {
      int idx = opIndex[op];
      start = std::min(start, idx);
      end = std::max(end, idx);
    }

    // ---------- 前一个 region 的末尾 ----------
    int prevEnd = -1;

    if (r > 0 && !mergedRegions[r - 1].opsToMove.empty()) {
      for (Operation *op : mergedRegions[r - 1].opsToMove) {
        prevEnd = std::max(prevEnd, opIndex[op]);
      }
    }

    SmallVector<Operation *> newOps;

    // ---------- backward expand ----------
    for (int i = start - 1; i > prevEnd; i--) {

      Operation *op = ops[i];

      // terminator
      if (isa<scf::YieldOp>(op))
        break;

      // 已属于 region
      if (regionOps.contains(op))
        break;

      // yield chain op
      if (skipOps.contains(op))
        continue;

      newOps.push_back(op);
    }

    // ---------- 加入 region ----------
    for (Operation *op : newOps) {
      region.opsToMove.push_back(op);
      regionOps.insert(op);
    }
  }

  // ---------- 保持 block 顺序 ----------
  for (auto &region : mergedRegions) {

    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(),
                    region.opsToMove.end()),
        region.opsToMove.end());
  }
}

// 贪心吸收region前面的op
static void ExpandMergedRegionOpsGreedyV2ForAIC(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();

  // block op 列表
  SmallVector<Operation *> ops;
  for (Operation &op : body)
    ops.push_back(&op);

  // op -> index
  DenseMap<Operation *, int> opIndex;
  for (int i = 0; i < ops.size(); i++)
    opIndex[ops[i]] = i;

  // 记录哪些 op 已经属于 region
  DenseSet<Operation *> regionOps;
  for (auto &region : mergedRegions)
    for (Operation *op : region.opsToMove)
      regionOps.insert(op);

  for (int r = 0; r < mergedRegions.size(); r++) {

    auto &region = mergedRegions[r];
    if (region.opsToMove.empty())
      continue;

    // ---------- 当前 region block 范围 ----------
    int start = ops.size();
    int end = -1;

    for (Operation *op : region.opsToMove) {
      int idx = opIndex[op];
      start = std::min(start, idx);
      end = std::max(end, idx);
    }

    // ---------- 前一个 region 的末尾 ----------
    int prevEnd = -1;

    if (r > 0 && !mergedRegions[r - 1].opsToMove.empty()) {
      for (Operation *op : mergedRegions[r - 1].opsToMove) {
        prevEnd = std::max(prevEnd, opIndex[op]);
      }
    }

    SmallVector<Operation *> newOps;

    // ---------- backward expand ----------
    for (int i = start - 1; i > prevEnd; i--) {

      Operation *op = ops[i];

      // terminator
      if (isa<scf::YieldOp>(op))
        break;

      // 已属于 region
      if (regionOps.contains(op))
        break;

      newOps.push_back(op);
    }

    // ---------- 加入 region ----------
    for (Operation *op : newOps) {
      region.opsToMove.push_back(op);
      regionOps.insert(op);
    }
  }

  // ---------- 保持 block 顺序 ----------
  for (auto &region : mergedRegions) {

    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(),
                    region.opsToMove.end()),
        region.opsToMove.end());
  }
}

static void MoveForYieldOpIntoRegion(
    scf::ForOp forOp,
    DenseSet<Operation *> &yieldRelatedOps,
    SmallVector<MergedRegion> &mergedRegions) {

  DenseMap<Operation *, int> opToRegion;

  for (int i = 0; i < mergedRegions.size(); i++)
    for (Operation *op : mergedRegions[i].opsToMove)
      opToRegion[op] = i;

  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  for (int i = 0; i < yield.getNumOperands(); i++) {

    Value iterArg = forOp.getRegionIterArgs()[i];
    Value yieldVal = yield.getOperand(i);

    Operation *def = yieldVal.getDefiningOp();
    if (!def)
      continue;

    if (!yieldRelatedOps.contains(def))
      continue;

    int targetRegion = -1;

    for (Operation *user : iterArg.getUsers()) {

      if (opToRegion.count(user)) {
        targetRegion = opToRegion[user];
        break;
      }
    }

    if (targetRegion == -1)
      continue;

    SmallVector<Operation *> stack;
    stack.push_back(def);

    while (!stack.empty()) {
      Operation *op = stack.pop_back_val();

      if (!yieldRelatedOps.contains(op))
        continue;

      mergedRegions[targetRegion].opsToMove.push_back(op);

      yieldRelatedOps.erase(op);

      for (Value operand : op->getOperands()) {
        if (Operation *dep = operand.getDefiningOp())
          stack.push_back(dep);
      }
    }
  }

  for (auto &region : mergedRegions) {

    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(), region.opsToMove.end()),
        region.opsToMove.end());
  }
}

static void MoveRemainingYieldOpsToPrevRegion(
    scf::ForOp forOp,
    DenseSet<Operation *> &yieldRelatedOps,
    SmallVector<MergedRegion> &mergedRegions) {

  if (yieldRelatedOps.empty())
    return;

  Block &body = forOp.getRegion().front();

  // op -> region index
  DenseMap<Operation *, int> opToRegion;
  for (int i = 0; i < mergedRegions.size(); i++)
    for (Operation *op : mergedRegions[i].opsToMove)
      opToRegion[op] = i;

  // block 顺序
  SmallVector<Operation *> ops;
  for (Operation &op : body)
    ops.push_back(&op);

  DenseMap<Operation *, int> opIndex;
  for (int i = 0; i < ops.size(); i++)
    opIndex[ops[i]] = i;

  for (Operation *op : yieldRelatedOps) {

    if (op->getBlock() != &body)
      continue;

    int idx = opIndex[op];

    int targetRegion = -1;

    // 向前找最近的 region
    for (int i = idx - 1; i >= 0; i--) {
      Operation *prev = ops[i];

      if (opToRegion.count(prev)) {
        targetRegion = opToRegion[prev];
        break;
      }
    }

    if (targetRegion == -1)
      continue;

    mergedRegions[targetRegion].opsToMove.push_back(op);
  }

  // 排序 + 去重
  for (auto &region : mergedRegions) {

    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(), region.opsToMove.end()),
        region.opsToMove.end());
  }
}

static void MoveIndependentOpsIntoRegionBackwardV2(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions) {

  Block &body = forOp.getRegion().front();
  SmallVector<Operation *> ops;
  for (Operation &op : body)
    ops.push_back(&op);

  DenseMap<Operation *, int> opToRegion;
  for (int i = 0; i < mergedRegions.size(); i++)
    for (Operation *op : mergedRegions[i].opsToMove)
      opToRegion[op] = i;

  // ----------- 收集移动计划 -----------
  DenseMap<Operation *, int> movePlan;

  for (int i = 0; i < mergedRegions.size(); i++) {
    MergedRegion &region = mergedRegions[i];
    if (region.opsToMove.empty()) continue;

    Operation *firstOp = region.opsToMove.front();
    Operation *lastOp  = region.opsToMove.back();
    auto itFirst = std::find(ops.begin(), ops.end(), firstOp);
    auto itLast  = std::find(ops.begin(), ops.end(), lastOp);
    if (itFirst == ops.end() || itLast == ops.end()) continue;

    int startIdx = std::distance(ops.begin(), itFirst);
    int endIdx   = std::distance(ops.begin(), itLast);

    // ----------- 收集 wait-set 区间 -----------
    SmallVector<std::pair<int,int>> waitIntervals;
    bool inWait = false;
    int begin = -1;
    for (int j = startIdx; j <= endIdx; j++) {
      Operation *op = ops[j];
      if (op->getName().getStringRef().contains("sync_block_wait")) {
        inWait = true; begin = j + 1; continue;
      }
      if (op->getName().getStringRef().contains("sync_block_set") && inWait) {
        inWait = false;
        waitIntervals.push_back({begin, j - 1});
      }
    }
    auto isInWaitSet = [&](int idx) {
      for (auto &p : waitIntervals)
        if (idx >= p.first && idx <= p.second) return true;
      return false;
    };

    // ----------- 从后往前扫描 region 内的 op -----------
    for (int j = endIdx; j >= startIdx; j--) {
      Operation *op = ops[j];
      if (isa<scf::YieldOp>(op) || isInWaitSet(j)) continue;

      // ---------- operand 是否依赖本 region ----------
      bool dependCurrentRegion = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def) continue;
        if (std::find(region.opsToMove.begin(),
                      region.opsToMove.end(),
                      def) != region.opsToMove.end()) {
          dependCurrentRegion = true; break;
        }
      }
      if (dependCurrentRegion) continue;

      // ---------- 当前 region 后续是否使用 ----------
      bool usedLaterInSameRegion = false;
      for (Value result : op->getResults())
        for (Operation *user : result.getUsers())
          if (std::find(region.opsToMove.begin(),
                        region.opsToMove.end(),
                        user) != region.opsToMove.end() &&
              std::find(region.opsToMove.begin(),
                        region.opsToMove.end(), op) <
              std::find(region.opsToMove.begin(),
                        region.opsToMove.end(), user)) {
            usedLaterInSameRegion = true; break;
          }
      if (usedLaterInSameRegion) continue;

      // ---------- 找使用该 op 的后续 region ----------
      int targetRegion = -1;
      for (int k = i + 1; k < mergedRegions.size(); ++k) {
        for (Operation *candidate : mergedRegions[k].opsToMove)
          for (Value operand : candidate->getOperands())
            if (operand.getDefiningOp() == op) {
              targetRegion = k; break;
            }
          if (targetRegion != -1) break;
        if (targetRegion != -1) break;
      }
      if (targetRegion == -1) continue;

      movePlan[op] = targetRegion;
      // llvm::outs() << "MJ: plan move " << *op
      //              << " -> region " << targetRegion << "\n";
    }
  }

  // ----------- 统一应用移动 ----------- 
  for (auto &it : movePlan) {
      Operation *op = it.first;
      int targetRegionIdx = it.second;
      MergedRegion &targetRegion = mergedRegions[targetRegionIdx];
      // 更新数据结构
      targetRegion.opsToMove.push_back(op);

      llvm::outs() << "MJ: move " << *op
                  << " -> region " << targetRegionIdx << "\n";
  }

  // ----------- 更新原 region 的 opsToMove -----------
  for (int i = 0; i < mergedRegions.size(); ++i) {
    MergedRegion &region = mergedRegions[i];
    SmallVector<Operation *> newOps;
    for (Operation *op : region.opsToMove) {
        auto it = movePlan.find(op);
        if (it == movePlan.end() || it->second == i) {
            // 没有移动计划，或者移动的目标就是自己，保留
            newOps.push_back(op);
        }
    }
    region.opsToMove.swap(newOps);
  }

  // ----------- 排序 + 去重 -----------
  for (auto &region : mergedRegions) {
    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });
    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(),
                    region.opsToMove.end()),
        region.opsToMove.end());
  }
}

// // debug: 如果一个forop的第一个region的最后3条op是%27 = tt.expand_dims %25#1 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
//           %28 = tt.broadcast %27 : tensor<64x1xf32> -> tensor<64x128xf32>
//           %29 = arith.mulf %arg10, %28 : tensor<64x128xf32>
// 直接放到第2个region里
static void TempChange(scf::ForOp forOp,
                       SmallVector<MergedRegion> &mergedRegions) {

  if (mergedRegions.size() < 2)
    return;

  auto &srcRegion = mergedRegions[0];
  auto &dstRegion = mergedRegions[1];

  if (srcRegion.opsToMove.size() < 3)
    return;

  Operation *op1 = srcRegion.opsToMove[srcRegion.opsToMove.size() - 3];
  Operation *op2 = srcRegion.opsToMove[srcRegion.opsToMove.size() - 2];
  Operation *op3 = srcRegion.opsToMove[srcRegion.opsToMove.size() - 1];

  // ---------- pattern 匹配 ----------
  if (!op1->getName().getStringRef().contains("tt.expand_dims"))
    return;

  if (!op2->getName().getStringRef().contains("tt.broadcast"))
    return;

  if (!op3->getName().getStringRef().contains("arith.mulf"))
    return;

  llvm::outs() << "TempChange triggered\n";

  SmallVector<Operation*> opsToMove = {op1, op2, op3};

  // ---------- 移动到 region2 末尾 ----------
  for (Operation *op : opsToMove) {
    dstRegion.opsToMove.push_back(op);
    llvm::outs() << "TempChange move: " << *op << "\n";
  }

  // ---------- 从 region1 删除 ----------
  srcRegion.opsToMove.resize(srcRegion.opsToMove.size() - 3);

  // ---------- 排序 ----------
  for (auto &region : mergedRegions) {
    llvm::sort(region.opsToMove,
               [](Operation *a, Operation *b) {
                 return a->isBeforeInBlock(b);
               });

    region.opsToMove.erase(
        std::unique(region.opsToMove.begin(),
                    region.opsToMove.end()),
        region.opsToMove.end());
  }
}

static void sortOperationsByDataFlow(llvm::SmallVector<Operation*> &ops) {
  llvm::DenseSet<Operation*> visited;
  llvm::SmallVector<Operation*> result;

  std::function<void(Operation*)> dfs = [&](Operation *op) {
    if (!visited.insert(op).second)
      return;

    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        if (llvm::is_contained(ops, def))
          dfs(def);
      }
    }

    result.push_back(op);
  };

  for (Operation *op : ops)
    dfs(op);

  ops.assign(result.begin(), result.end());
}

static void rewriteOperandsRecursively(Operation *op,
                                       DenseMap<Value, Value> &valueMap) {

  // 1 rewrite 当前 op 的 operands
  for (OpOperand &operand : op->getOpOperands()) {
    Value v = operand.get();
    auto it = valueMap.find(v);
    if (it != valueMap.end())
      operand.set(it->second);
  }

  // 2 递归进入 region
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        rewriteOperandsRecursively(&nestedOp, valueMap);
      }
    }
  }
}

static void CopyOpsToAfterwardRegions(
    SmallVector<MergedRegion> &mergedRegions,
    DenseMap<Value, Operation *> &yieldMap,
    DenseMap<Operation *, Operation *> &cloneAndOriYieldMap,
    SmallVector<scf::ForOp> &copiedForOps) {

  if (mergedRegions.size() <= 1)
    return;

  // 先整理一个 set，方便判断哪些 op 是 yield defining op
  DenseSet<Operation *> yieldDefOps;
  for (auto &it : yieldMap)
    yieldDefOps.insert(it.second);

  // 倒序遍历 region
  for (int i = mergedRegions.size() - 1; i >= 0; --i) {
    MergedRegion &curRegion = mergedRegions[i];

    DenseMap<Value, Value> valueMap;
    SmallVector<Operation *> clonedOps;

    // 遍历前面的 region
    for (int k = 0; k < i; ++k) {
      MergedRegion &prevRegion = mergedRegions[k];

      int waitSetLevel = 0;

      for (Operation *op : prevRegion.opsToMove) {

        if (isa<SyncBlockWaitOp>(op)) {
          waitSetLevel++;
          continue;
        }

        if (isa<SyncBlockSetOp>(op)) {
          waitSetLevel = std::max(waitSetLevel - 1, 0);
          continue;
        }

        if (waitSetLevel > 0)
          continue;

        if (isa<triton::DotOp>(op))
          continue;

        IRMapping mapper;

        for (auto result : op->getResults())
          if (valueMap.count(result))
            mapper.map(result, valueMap[result]);

        Operation *insertPoint =
            curRegion.opsToMove.empty() ? nullptr : curRegion.opsToMove.front();

        OpBuilder builder(insertPoint ? insertPoint : op);

        Operation *cloned = builder.clone(*op, mapper);

        // 记录 result mapping
        for (auto it : llvm::zip(op->getResults(), cloned->getResults()))
          valueMap[std::get<0>(it)] = std::get<1>(it);

        // 如果这个 op 是 yield defining op，记录 clone -> original
        if (yieldDefOps.contains(op)) {
          cloneAndOriYieldMap[cloned] = op;
        }
        
        // 记录copy的for op
        if (auto forOp = dyn_cast<scf::ForOp>(cloned)) {
          copiedForOps.push_back(forOp);
        }

        clonedOps.push_back(cloned);
      }
    }

    // 插入到当前 region 开头
    curRegion.opsToMove.insert(curRegion.opsToMove.begin(),
                               clonedOps.begin(), clonedOps.end());

    // rebuild SSA
    for (Operation *op : curRegion.opsToMove) {
      rewriteOperandsRecursively(op, valueMap);
    }

    // 排序保证拓扑顺序
    sortOperationsByDataFlow(curRegion.opsToMove);

  }
}

/// 记录 forOp 的 yield value 与其原始生成的 op 的映射
static void GetYieldMap(scf::ForOp forOp,
                        DenseMap<Value, Operation *> &yieldMap) {
  yieldMap.clear();

  // 取 forOp body 的 scf.yield
  auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp)
    return;

  for (Value yieldVal : yieldOp.getOperands()) {
    // 获取生成 yieldVal 的原始 op
    Operation *defOp = yieldVal.getDefiningOp();

    // 对 block arg（可能是 iter_arg）没有 definingOp 的情况，可以跳过或直接记录 nullptr
    if (!defOp)
      continue;

    yieldMap[yieldVal] = defOp;
  }
}

static Value findIterArgForAIC(Value v, scf::ForOp forOp) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() == forOp.getBody())
        return v;
      return Value();
    }

    Operation *def = v.getDefiningOp();
    if (!def)
      return Value();

    if (def->getNumOperands() == 0)
      return Value();

    v = def->getOperand(0);
  }
}

static Operation *findCloneOfYieldOp(
    Operation *oriYieldOp,
    DenseMap<Operation *, Operation *> &cloneAndOriYieldMap,
    MergedRegion &region) {

  for (Operation *op : region.opsToMove) {
    auto it = cloneAndOriYieldMap.find(op);
    if (it != cloneAndOriYieldMap.end() && it->second == oriYieldOp)
      return op;
  }
  return nullptr;
}

static void RebuildForYielValuesForAIC(
    scf::ForOp forOp,
    SmallVector<MergedRegion> &mergedRegions,
    DenseMap<Value, Operation *> &yieldMap,
    DenseMap<Operation *, Operation *> &cloneAndOriYieldMap) {

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  for (MergedRegion &region : mergedRegions) {

    triton::DotOp dotOp = nullptr;

    for (Operation *op : region.opsToMove) {
      if (auto d = dyn_cast<triton::DotOp>(op)) {
        dotOp = d;
        break;
      }
    }

    if (!dotOp)
      continue;

    // 处理 dot operand
    for (Value operand : dotOp->getOperands()) {

      Value iterArg = findIterArgForAIC(operand, forOp);
      if (!iterArg)
        continue;

      auto arg = cast<BlockArgument>(iterArg);
      int idx = arg.getArgNumber();

      if (idx >= yieldOp.getNumOperands())
        continue;

      Value oriYieldValue = yieldOp.getOperand(idx);

      auto it = yieldMap.find(oriYieldValue);
      if (it == yieldMap.end())
        continue;

      Operation *oriYieldOp = it->second;

      Operation *cloneOp =
          findCloneOfYieldOp(oriYieldOp, cloneAndOriYieldMap, region);

      if (!cloneOp)
        continue;

      yieldOp.setOperand(idx, cloneOp->getResult(0));
    }
  }
}

void ExpandMergedRegionOps(scf::ForOp forOp,
                           SmallVector<MergedRegion> &mergedRegions,
                           SmallVector<scf::ForOp> &copiedForOps) {
  bool isInAIV = false;
  auto scopeOp = forOp->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return;
  
  auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name);

  if (coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR) {
    isInAIV = true;
  }

  if (isInAIV) {
    DenseSet<Operation *> yieldRelatedOps;

    // 1 收集 yield 相关 op
    CollectForYieldRelatedOps(forOp, mergedRegions, yieldRelatedOps);

    // 2 greedy 扩展
    // ExpandMergedRegionOpsGreedy(forOp, mergedRegions, yieldRelatedOps);
    ExpandMergedRegionOpsGreedyV2(forOp, mergedRegions, yieldRelatedOps);

    // 3 与前面wait-set region独立的op应该被放入后面的关联的region
    MoveIndependentOpsIntoRegionBackwardV2(forOp, mergedRegions);

    // 4 根据 iter_arg 使用位置放入 region
    MoveForYieldOpIntoRegion(forOp, yieldRelatedOps, mergedRegions);

    // 5 剩余 yield chain 放入前一个 region
    MoveRemainingYieldOpsToPrevRegion(forOp, yieldRelatedOps, mergedRegions);
  }
  else { // AIC单独处理, 避免出现CUBE内的tensor变量依赖
    // 用Map记录原始的for yield op的<op, yield value>的映射
    DenseMap<Value, Operation *> yieldMap;
    GetYieldMap(forOp, yieldMap);
    
    llvm::outs()<<"YieldMap:\n";
    for(auto it: yieldMap) {
      llvm::outs()<<*(it.second)<<"\n";
    }

    // 2 greedy 扩展, yield value后续处理
    ExpandMergedRegionOpsGreedyV2ForAIC(forOp, mergedRegions);
    
    // 复制当前region的除tt.dot、以及[wait - set]之间的op到后续的所有MergedRegion
    // 倒序实现
    // 记录clone和original的yield对应op的map
    DenseMap<Operation *, Operation *> cloneAndOriYieldMap;
    CopyOpsToAfterwardRegions(mergedRegions, yieldMap, cloneAndOriYieldMap, copiedForOps);

    // 4 先确定每个MergedRegion的tt.dot的operand的来源是for的哪个iter_arg(递归查找), 假设为%arg0, 依据yieldMap可以得到oriYield
    // 遍历当前MergedRegion的所有op, 确定哪条op对应的cloneAndOriYieldMap的second是oriYield, 假设为%45
    // 最后替换for yield op对应位置的operand为%45
    RebuildForYielValuesForAIC(
      forOp,
      mergedRegions,
      yieldMap,
      cloneAndOriYieldMap);

  }
}

void MergeWaitSetRegions(SmallVector<WaitSetRegion> &regions,
                         SmallVector<MergedRegion> &merged) {
  for (int i = 0; i < regions.size();) {
    MergedRegion mr;
    mr.regions.push_back(&regions[i]);
    mr.opsToMove.append(regions[i].opsToMove);

    int j = i;
    while (!regions[j].hasCopyOrFixpipe &&
           j + 1 < regions.size()) {
      j++;
      mr.regions.push_back(&regions[j]);
      mr.opsToMove.append(regions[j].opsToMove);
    }

    merged.push_back(std::move(mr));
    i = j + 1;
  }

  for (MergedRegion &mr : merged) {
    SmallPtrSet<Value, 16> regionValues;
    SmallPtrSet<Operation *, 16> opSet;

    for (Operation *op : mr.opsToMove)
      opSet.insert(op);

    for (Operation *op : mr.opsToMove) {
      for (Value v : op->getResults()) {
        bool usedOutside = false;
        for (OpOperand &use : v.getUses()) {
          Operation *user = use.getOwner();
          if (!opSet.contains(user) &&
              user->getBlock() == op->getBlock()) {
            usedOutside = true;
            break;
          }
        }
        if (usedOutside) {
          mr.yieldValues.push_back(v);
          mr.resultTypes.push_back(v.getType());
        }
      }
    }
  }
}

void GetBlockInfos(SmallVector<WaitSetRegion> &regions, Block &body) {
  for (auto it = body.begin(); it != body.end();) {
    Operation *op = &*it;

    auto waitOp = dyn_cast<SyncBlockWaitOp>(op);
    if (!waitOp) {
        it++;
        continue;
    }

    auto pipeS = hivm::PipeAttr::get(op->getContext(), hivm::PIPE::PIPE_S);
    if (auto syncWait = dyn_cast<SyncBlockWaitOp>(op)) {
      if ((syncWait.getTpipe() == pipeS || syncWait.getPipe() == pipeS) && (std::next(it) != body.end())) {
        if (isa<scf::ForOp>(&*std::next(it)) || isa<triton::LoadOp>(&*std::next(it)))
          return;
      }
    }

    Operation *lastSetOp = nullptr;

    // 扫描到下一个 wait, 收集所有 set
    auto curIt = std::next(it);
    auto endIt = curIt;
    int setOpCount = 0;
    SmallVector<Operation *> opsInRegion;
    for (; curIt != body.end(); ++curIt) {
      Operation *curOp = &*curIt;
      if (isa<SyncBlockWaitOp>(curOp) && setOpCount >= 1) break;
      if (isa<SyncBlockSetOp>(curOp)) {
        setOpCount++;
        endIt = curIt; //setop的位置
        lastSetOp = curOp;  // 最后一个 set
      }
    }

    if (!lastSetOp) {
      it = curIt;
      continue;
    }// 没有 set, 不包

    // 收集 [wait, ..., lastSet] 之间的 ops
    bool hasCopyOrFixpipe = false;
    for (auto it2 = it; it2 != std::next(endIt); ++it2) {
      Operation *curOp = &*it2;
      opsInRegion.push_back(curOp);
      if (isa<CopyOp>(curOp) || isa<FixpipeOp>(curOp)) {
        hasCopyOrFixpipe = true;
      }
    }
    
    it = endIt++;
    regions.push_back({waitOp, lastSetOp, opsInRegion, hasCopyOrFixpipe});
  }
}

Value findIterArg(Value v, Type t) {
    SmallVector<Value> worklist = {v};
    SmallPtrSet<Value, 16> visited;

    while (!worklist.empty()) {
        Value cur = worklist.front();
        worklist.erase(worklist.begin());
        if (!visited.insert(cur).second)
            continue;

        // 匹配scf.for原始迭代参数, 直接返回
        if (auto b = mlir::dyn_cast<BlockArgument>(cur)) {
            auto forOp = mlir::dyn_cast<scf::ForOp>(b.getOwner()->getParentOp());
            if (forOp && b.getType() == t) {
                for (Value iterArg : forOp.getRegionIterArgs()) {
                    if (iterArg.getAsOpaquePointer() == b.getAsOpaquePointer()) {
                        return b;
                    }
                }
            }
        }

        Operation *defOp = cur.getDefiningOp();
        if (!defOp) continue;

        // 核心逻辑：如果当前值是scf.if的结果
        // 进入then块找源头
        if (auto ifOp = mlir::dyn_cast<scf::IfOp>(defOp)) {
            Block &thenBlock = ifOp.getThenRegion().front();
            // 找到then块最后一个op（scf.yield）
            // 取其operands（即ifOp结果的源头值）
            for (auto &innerOp : llvm::reverse(thenBlock)) {
                if (auto yieldOp = mlir::dyn_cast<scf::YieldOp>(&innerOp)) {
                    // 按索引匹配: cur是ifOp的第n个结果, 取yieldOp的第n个operand
                    for (auto [idx, res] : llvm::enumerate(ifOp.getResults())) {
                        if (res.getAsOpaquePointer() == cur.getAsOpaquePointer()) {
                            Value srcVal = yieldOp.getOperand(idx);
                            if (!visited.count(srcVal)) worklist.push_back(srcVal);
                            break;
                        }
                    }
                    break; // 找到yield即退出, 无需遍历其他op
                }
            }
        } else {
            // 非if结果值
            // 正常往前追溯operands
            for (Value operand : defOp->getOperands()) {
                if (!visited.count(operand)) worklist.push_back(operand);
            }
        }
    }

    llvm::outs() << "未找到迭代参数, 返回原值: "; v.print(llvm::outs()); llvm::outs() << "\n";
    return v;
}

// 如果 v 最终被 scf.for 的 yield 使用
// → 返回对应的 forOp 的 iter_arg
// 如果 v 只是流向后面的 wait-set region / 其他 op
// → 直接返回原值 v
Value findIterArgForAll(Value v, Type t) {
  for (Operation *user : v.getUsers()) {

    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {

      if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {

        for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {

          if (operand.getAsOpaquePointer() == v.getAsOpaquePointer()) {

            Value iterArg = forOp.getRegionIterArgs()[idx];

            if (iterArg.getType() == t)
              return iterArg;
          }
        }
      }
    }
  }

  return v;
}

void FindDependValues (SmallVector<Value> &dependValues, SmallVector<MergedRegion> mergedRegions) {
  dependValues.clear();
  for (auto &curMR : mergedRegions) {
    for (Value yieldValue : curMR.yieldValues) {
      // llvm::outs() << "yieldValue: "<< yieldValue << "\n";
      // 遍历当前区域的yieldValue的所有user OP，判断是否存在依赖关系
      for (OpOperand &use : yieldValue.getUses()) {
        Operation *userOp = use.getOwner();

        // llvm::outs() << "userOp: "<< *userOp << "\n";
        bool isUserInOtherRegion = false;
        for (auto &otherMR : mergedRegions) {
          // 跳过当前区域，只检查yieldValue是否被其他区域使用
          if (&otherMR == &curMR) continue;

          // 只要有一个 userOp在 otherMR 的 opsToMove 列表中，就认为是dependValue
          // llvm::outs() << "judge comtain\n";
          // for (size_t k = 0; k < otherMR.opsToMove.size(); k++) {
          //   llvm::outs() << "otherMR op: " << *(otherMR.opsToMove[k]) << "\n";
          // }
          // llvm::outs() << "otherMR end\n";

          // if (llvm::is_contained(otherMR.opsToMove, userOp)) {
          //   isUserInOtherRegion = true;
          //   llvm::outs() << "is_contained\n";
          //   break;
          // }

          // 用 DenseSet 暂存当前 region 的所有 ops
          // 初始 DenseSet: 顶层 opsToMove
          DenseSet<Operation *> otherOps;
          for (Operation *op : otherMR.opsToMove) {
              CollectAllNestedOps(op, otherOps); // 完整展开嵌套
          }
          if (otherOps.contains(userOp)) {
            isUserInOtherRegion = true;
            break;
          }

        }

        // 无重复的添加依赖变量
        if (isUserInOtherRegion) {
          if (!llvm::is_contained(dependValues, yieldValue)) {
            dependValues.push_back(yieldValue);
          }
          break;
        }
      }
    }
  }
}

void UpdateMergedRegionsWithNewForOp(SmallVector<MergedRegion> &mergedRegions, IRMapping &mapper) {
  for (auto &mr : mergedRegions) {
    // WaitSetRegion 后续已经不使用了，直接释放，否则会出现野指针
    SmallVector<WaitSetRegion *> newRegions;
    newRegions.clear();
    mr.regions = newRegions;
    // // 更新 opsToMove 列表
    // llvm::outs() << "before \n";
    // for (auto &op : mr.opsToMove) {
    //   llvm::outs() << "opsToMove: " << op << ", " << *op << '\n';
    // }
    SmallVector<Operation *> newOpsToMove;
    newOpsToMove.clear();
    for (Operation *op : mr.opsToMove) {
      if (op) {
        Operation *newOp = mapper.lookupOrNull(op);
        newOpsToMove.push_back(newOp);
      }
    }
    mr.opsToMove = newOpsToMove;
    // llvm::outs() << "after \n";
    // for (auto &op : mr.opsToMove) {
    //   llvm::outs() << "opsToMove: " << op << ", " << *op << '\n';
    // }
    // 更新 yieldValues 列表
    SmallVector<Value> newYieldValues;
    newYieldValues.clear();
    for (Value v : mr.yieldValues) {
      if (v) {
        newYieldValues.push_back(mapper.lookupOrNull(v));
      }
    }
    mr.yieldValues = newYieldValues;
    // resultTypes 是type 类型，无需更新
  }
}

void AddArgsForDependValues (scf::ForOp forOp, SmallVector<Value> &dependValues, SmallVector<MergedRegion> &mergedRegions, ModuleOp module) {
  OpBuilder moduleBuilder(module.getContext());
  SmallVector<Type> valueTypes;
  valueTypes.clear();

  if (dependValues.empty()) {
    return ;
  } else {
    for (Value v : dependValues) {
      Type valueType = v.getType();
      valueTypes.push_back(valueType);
    }
  }

  // 为每个 dependValue 创建一个初始值（可能不存在相同shape和type的常量tensor）
  SmallVector<Value> initTensors;
  initTensors.clear();
  module.walk([&](Operation *op) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      moduleBuilder.setInsertionPoint(constOp);
      for (Type valueType : valueTypes) {
        auto tensorType = dyn_cast<RankedTensorType>(valueType);
        triton::PointerType ptrType;
        ptrType = (tensorType) ? dyn_cast<triton::PointerType>(tensorType.getElementType()) : dyn_cast<triton::PointerType>(valueType);
        if (ptrType) {
            // 如果依赖变量是一个ptr类型
            // 1. 创建 i64 0
            // 2. cast 成 !tt.ptr<...>
            Value zero = moduleBuilder.create<arith::ConstantIntOp>(constOp.getLoc(), 0, 64);
            Value ptrValue = moduleBuilder.create<triton::IntToPtrOp>(constOp.getLoc(), ptrType, zero);
            if (tensorType) {
              // 3. splat 成 tensor<...x!tt.ptr<...>>
              Value ptrTensor = moduleBuilder.create<triton::SplatOp>(constOp.getLoc(), tensorType, ptrValue);
              initTensors.push_back(ptrTensor);
            } else {
              initTensors.push_back(ptrValue);
            }
        } else if (auto memrefType = dyn_cast<mlir::MemRefType>(valueType)) {
          // 如果中间变量是一个memref类型，为iterarg创建一个 alloc = memref
          // 仅支持#hivm.address_space<ub>，对于#hivm.address_space<cbuf>，不存在 copy cbuf to cbuf 行为
          auto spaceAttr = cast<hivm::AddressSpaceAttr>(memrefType.getMemorySpace());
          if (spaceAttr && spaceAttr.getAddressSpace() == hivm::AddressSpace::L1) {
            llvm::dbgs() << "AddArgsForDependValues: dependValue type is a memref hivm::AddressSpace::L1 type!!!\n";
            return mlir::WalkResult::interrupt();
          } else {
            mlir::Value alloc = moduleBuilder.create<mlir::memref::AllocOp>(constOp.getLoc(), memrefType);
            initTensors.push_back(alloc);
          }
        } else {
          // 非 ptr 类型创建零值常量
          auto zeroAttr = moduleBuilder.getZeroAttr(valueType);
          Value zeroTensor = moduleBuilder.create<arith::ConstantOp>(constOp.getLoc(), zeroAttr);
          initTensors.push_back(zeroTensor);
        }
      }
    return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  auto initArgs = forOp.getInitArgs();

  // 构建新的初始化参数列表
  SmallVector<Value> newInitArgs(initArgs.begin(), initArgs.end());
  // 添加 dependValue 的初始化参数
  for (Value initTensor : initTensors) {
    newInitArgs.push_back(initTensor);
  }

  // 获取原循环的边界和步长
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();

  // 创建新的 ForOp，插入点位于原操作之前
  OpBuilder builder(forOp);
  auto newForOp = builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInitArgs);

  // 获取新循环的 region 块（已自动包含循环索引和迭代参数）
  Block &newBlock = newForOp.getRegion().front();
  Block &oldBlock = forOp.getRegion().front();

  // 建立块参数的映射：原块参数 -> 新块参数
  IRMapping mapper;
  for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i) {
      mapper.map(oldBlock.getArgument(i), newBlock.getArgument(i));
  }
  // 将原循环体中的操作（不包括终结符）克隆到新块中
  // 同时按照顺序克隆新的 dependValues
  SmallVector<Value> newDependValues = dependValues;
  int cnt = 0;
  builder.setInsertionPointToStart(&newBlock);
  for (auto &op : oldBlock) {
      auto newOp = builder.clone(op, mapper);
      // dependValue 的定义OP 可能有多个 result
      for (size_t i = 0; i < dependValues.size(); i++) {
        Operation *defineOp = dependValues[i].getDefiningOp();
        if (defineOp == &op) {
          unsigned int index = cast<OpResult>(dependValues[i]).getResultNumber();
          newDependValues[i] = newOp->getResult(index);
          cnt++;
          break;
        }
      }
  }
  // 判断是否找到了所有的 dependValue
  if (newDependValues.size() != cnt) {
    llvm::outs() << "can not find the depend value! \n";
    return;
  }
  dependValues = newDependValues;

  // 更新 mergedRegions 中的 op 为新的for循环的 op
  UpdateMergedRegionsWithNewForOp(mergedRegions, mapper);
  
  // 创建新的循环 yield 操作：原操作数 + dependValues
  auto oldYield = cast<scf::YieldOp>(newBlock.getTerminator());
  SmallVector<Value> newYieldOps(oldYield.getOperands());
  // 按顺序增加找到的 dependvalue
  for (Value v : newDependValues) {
    newYieldOps.push_back(v);
  }
  builder.setInsertionPointToEnd(&newBlock);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOps);
  oldYield.erase();

  // 将原 forOp 的所有使用替换为新 forOp
  int oldResultNum = forOp->getResults().size();
  for (auto it : llvm::zip(forOp->getResults(), newForOp->getResults().take_front(oldResultNum))) {
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }
  forOp.erase();
}

void ComputeElseYieldValues (MergedRegion mergedRegion, SmallVector<Value> &elseYieldValues, SmallVector<Value> dependValues) {
  int idx = 0;
  for (Value v : mergedRegion.yieldValues) {
      Type yieldType = mergedRegion.resultTypes[idx];
      elseYieldValues.push_back(findIterArg(v, yieldType));
      idx++;
  }
}

void ComputeElseYieldValuesV2 (MergedRegion mergedRegion, SmallVector<Value> &elseYieldValues, SmallVector<Value> dependValues) {
  // 对于yieldValues，其中的 yield value 一定是被 for op yield 所引用，或者被其他 region 所使用
  auto forOp = dyn_cast<scf::ForOp>(mergedRegion.yieldValues[0].getDefiningOp()->getBlock()->getParentOp());
  if (!forOp) {
    llvm::outs() << "define op's parent is not ForOp \n";
    return;
  }
  auto iterArgs = forOp.getRegionIterArgs();
  auto forYieldValues = forOp.getYieldedValues();
  
  // 新增的与 dependvalue 相关的 initarg 是接在原本for循环args后面，数量与dependvalue数量相等
  int baseDependIdx = iterArgs.size() - dependValues.size();

  int idx = 0;
  for (Value v : mergedRegion.yieldValues) {
      Type yieldType = mergedRegion.resultTypes[idx];
      // yieldValue 中是dependvalue 的情况下
      // else yield value 使用对应的新增 iterargs
      if (llvm::is_contained(dependValues, v)) {
        int dependIdx = 0;
        for (; dependIdx < dependValues.size(); dependIdx++) {
          if (v == dependValues[dependIdx]) {
            break;
          }
        }
        // llvm::outs()<<"v2for:"<<forOp<<"\n";
        // llvm::outs() << "iterArgs.size():"<<iterArgs.size()<<"\n";
        // llvm::outs() << "baseDependIdx:"<<baseDependIdx<<"\n";
        // llvm::outs() << "dependIdx:"<<dependIdx<<"\n";
        elseYieldValues.push_back(iterArgs[baseDependIdx + dependIdx]);
      } else {
        elseYieldValues.push_back(findIterArgForAll(v, yieldType));
      }
      idx++;
  }
}

static void RemoveRedundantYieldValues(MergedRegion &region) {
  SmallVector<Value> newYieldValues;
  SmallVector<Type> newResultTypes;

  SmallPtrSet<Value, 16> seen;

  for (auto [idx, v] : llvm::enumerate(region.yieldValues)) {
    if (seen.insert(v).second) {
      newYieldValues.push_back(v);
      newResultTypes.push_back(region.resultTypes[idx]);
    }
  }

  region.yieldValues.swap(newYieldValues);
  region.resultTypes.swap(newResultTypes);
}

static void replaceExternalIfOpUses(scf::IfOp ifOp,
                                    ArrayRef<Value> oldYieldValues) {

  for (size_t i = 0; i < oldYieldValues.size(); ++i) {
    Value oldVal = oldYieldValues[i];
    Value newVal = ifOp.getResult(i);

    SmallVector<OpOperand *> usesToReplace;

    for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {

      Operation *user = use.getOwner();

      // 跳过 ifOp 内部的使用（then / else region）
      if (ifOp->isAncestor(user))
        continue;

      // 只替换 ifOp 之后的使用
      if (user->getBlock() == ifOp->getBlock()) {
        if (!ifOp->isBeforeInBlock(user))
          continue;
      }

      usesToReplace.push_back(&use);
    }

    for (OpOperand *use : usesToReplace)
      use->set(newVal);
  }
}

void CreateIfOps (SmallVector<MergedRegion> &mergedRegions, SmallVector<Value> dependValues) {
  for (auto &region : mergedRegions) {

    // 去重yieldvalues
    RemoveRedundantYieldValues(region);

    Operation *insertPt = region.opsToMove.front();
    OpBuilder builder(insertPt);
    Location loc = insertPt->getLoc();
    Value cond = builder.create<arith::ConstantOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(true));

    bool needsYield = !region.yieldValues.empty();
    scf::IfOp ifOp;
    if (needsYield)
      ifOp = builder.create<scf::IfOp>(loc, region.resultTypes, cond, true);
    else
      ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);

    // 加标记
    ifOp->setAttr("ssbuffer", builder.getUnitAttr());

    // 获取if yield value 在 else块 返回值
    SmallVector<Value> elseYieldValues;

    llvm::outs()<<"before ComputeElseYieldValuesV2"<<"\n";
    if (needsYield) {
    //   ComputeElseYieldValues(region, elseYieldValues, dependValues);
      ComputeElseYieldValuesV2(region, elseYieldValues, dependValues);
    }

    llvm::outs()<<"after ComputeElseYieldValuesV2"<<"\n";
    // 将op移进then块
    Block &thenBlock = ifOp.getThenRegion().front();
    for (Operation *m : llvm::reverse(region.opsToMove)) {
      m->moveBefore(&thenBlock, thenBlock.begin());
    }

    // 创建 then/else yield
    if (needsYield) {
      OpBuilder thenBuilder(builder.getContext());
      thenBuilder.setInsertionPointToEnd(&thenBlock);
      thenBuilder.create<scf::YieldOp>(loc, region.yieldValues);

    //   else block
      Block &elseBlock = ifOp.getElseRegion().front();
      OpBuilder elseBuilder(&elseBlock, elseBlock.end());
      elseBuilder.create<scf::YieldOp>(loc, elseYieldValues);

      // 替换外部使用

      replaceExternalIfOpUses(ifOp, region.yieldValues);

      // 旧的逻辑
      // Block *block = ifOp->getBlock();
      // auto ifIt = Block::iterator(ifOp);

      // for (size_t i = 0; i < region.yieldValues.size(); ++i) {
      //   Value oldVal = region.yieldValues[i];
      //   Value newVal = ifOp.getResult(i);

      //   SmallVector<OpOperand *> usesToReplace;

      //   for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
      //     Operation *user = use.getOwner();
      //     // 同一个 block, user 必须在 ifOp 之后, 不能在 ifOp 内部（then / else）
      //     if (user->getBlock() != ifOp->getBlock() || !ifOp->isBeforeInBlock(user) || user->getParentOp() == ifOp)
      //       continue;
      //     usesToReplace.push_back(&use);
      //   }

      //   for (OpOperand *use : usesToReplace)
      //     use->set(newVal);
      // }

    }

    llvm::outs() <<"Create ifOp: "<< *ifOp << "\n";
  }
}

void CreateIfOpsOrigin (SmallVector<MergedRegion> &mergedRegions) {
  for (auto &region : mergedRegions) {

    // 去重yieldvalues
    RemoveRedundantYieldValues(region);

    Operation *insertPt = region.opsToMove.front();
    OpBuilder builder(insertPt);
    Location loc = insertPt->getLoc();
    Value cond = builder.create<arith::ConstantOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(true));
 
    bool needsYield = !region.yieldValues.empty();
    scf::IfOp ifOp;
    if (needsYield)
      ifOp = builder.create<scf::IfOp>(loc, region.resultTypes, cond, true);
    else
      ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);

    // 加标记
    ifOp->setAttr("ssbuffer", builder.getUnitAttr());
    
    // 将op移进then块
    Block &thenBlock = ifOp.getThenRegion().front();
    for (Operation *m : llvm::reverse(region.opsToMove)) {
      m->moveBefore(&thenBlock, thenBlock.begin());
    }
 
    // 创建 then/else yield
    if (needsYield) {
      OpBuilder thenBuilder(builder.getContext());
      thenBuilder.setInsertionPointToEnd(&thenBlock);
      thenBuilder.create<scf::YieldOp>(loc, region.yieldValues);
 
      // else block
      SmallVector<Value> elseYieldValues;
      int idx = 0;
      for (Value v : region.yieldValues) {
          Type yieldType = region.resultTypes[idx];
          elseYieldValues.push_back(findIterArgForAll(v, yieldType));
          idx++;
      }
      Block &elseBlock = ifOp.getElseRegion().front();
      OpBuilder elseBuilder(&elseBlock, elseBlock.end());
      elseBuilder.create<scf::YieldOp>(loc, elseYieldValues);
 
      // 替换外部使用
      Block *block = ifOp->getBlock();
      auto ifIt = Block::iterator(ifOp);
 
      for (size_t i = 0; i < region.yieldValues.size(); ++i) {
        Value oldVal = region.yieldValues[i];
        Value newVal = ifOp.getResult(i);
 
        SmallVector<OpOperand *> usesToReplace;
 
        for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
          Operation *user = use.getOwner();
          // 同一个 block, user 必须在 ifOp 之后, 不能在 ifOp 内部（then / else）
          if (user->getBlock() != ifOp->getBlock() || !ifOp->isBeforeInBlock(user) || user->getParentOp() == ifOp)
            continue;
          usesToReplace.push_back(&use);
        }
 
        for (OpOperand *use : usesToReplace)
          use->set(newVal);
      }
 
    }
 
    llvm::outs() <<"Create ifOp: "<< *ifOp << "\n";
  }
}

void AddIfCondition(ModuleOp module) {
  SmallVector<scf::ForOp> copiedForOps;
  SmallVector<scf::ForOp> forOpList;
  SmallVector<SmallVector<MergedRegion>, 1> regionList;

  module.walk([&](scf::ForOp forOp) {
    Block &body = forOp.getRegion().front();
    SmallVector<WaitSetRegion> regions;

    // 获取基本的wait-set分块信息
    GetBlockInfos(regions, body);

    SmallVector<MergedRegion> mergedRegions;
    // 合并wait-set块, 依据copyop / fixpipeop合并
    MergeWaitSetRegions(regions, mergedRegions);

    // 扩展if包裹的op范围
    // AIV、AIC处理有区别
    ExpandMergedRegionOps(forOp, mergedRegions, copiedForOps);

    // 处理forop的末尾对于iter_arg的自增操作, 如tt.advance, 移进对应的if op
    MoveIterArgUsersIntoIf(forOp, mergedRegions);
    
    // 获取if yield的value, 并更新if内op的user为yield value
    for (MergedRegion &mr : mergedRegions) {
      // ComputeYieldForMergedRegion(mr, body);
      ComputeYieldForMergedRegionV4(mr);
    }

    //   // 创建最终的if op
    //   CreateIfOpsOrigin(mergedRegions);
    // });

    forOpList.push_back(forOp);
    regionList.push_back(mergedRegions);
  });

  llvm::outs()<<"CopyForOp:\n";
  for(auto op : copiedForOps){
    llvm::outs()<<*op<<"\n";
  }

  SmallVector<scf::ForOp> tmpOps;
  for (auto copiedOp : copiedForOps) {
    Block &body = copiedOp.getRegion().front();
    SmallVector<WaitSetRegion> regions;

    // 获取基本的wait-set分块信息
    GetBlockInfos(regions, body);

    SmallVector<MergedRegion> mergedRegions;
    // 合并wait-set块, 依据copyop / fixpipeop合并
    MergeWaitSetRegions(regions, mergedRegions);

    // 扩展if包裹的op范围
    // AIV、AIC处理有区别
    ExpandMergedRegionOps(copiedOp, mergedRegions, tmpOps);

    // 处理forop的末尾对于iter_arg的自增操作, 如tt.advance, 移进对应的if op
    MoveIterArgUsersIntoIf(copiedOp, mergedRegions);

    // 获取if yield的value, 并更新if内op的user为yield value
    for (MergedRegion &mr : mergedRegions) {
      // ComputeYieldForMergedRegion(mr, body);
      ComputeYieldForMergedRegionV4(mr);
    }

    //   // 创建最终的if op
    //   CreateIfOpsOrigin(mergedRegions);
    // });

    forOpList.push_back(copiedOp);
    regionList.push_back(mergedRegions);
  }

  for (size_t i = 0; i < forOpList.size(); ++i) {
    scf::ForOp oldForOp = forOpList[i];
    SmallVector<MergedRegion> newMergedRegions = regionList[i];

    // 找到所有的VV或CC依赖
    SmallVector<Value> dependValues;
    llvm::outs() << "FindDependValues! \n ";
    FindDependValues(dependValues, newMergedRegions);

    if (dependValues.size() != 0) {
      copyLoadCalculation(oldForOp, dependValues, newMergedRegions);
      
      // repeat previous operations
      for (MergedRegion &mr : newMergedRegions) {
        mr.yieldValues.clear();
        mr.resultTypes.clear();
        ComputeYieldForMergedRegionV4(mr);
      }
      FindDependValues(dependValues, newMergedRegions);
    }
    
    // 如果存在VV或CC依赖，更新ForOp添加新的对应args
    if (dependValues.size() != 0) {
      AddArgsForDependValues(oldForOp, dependValues, newMergedRegions, module);
    }
    
    // 创建最终的if op
    llvm::outs() << "before create if ops" << '\n';
    CreateIfOps(newMergedRegions, dependValues);
  }
}

void ChangeAdvanceOpForm(ModuleOp module) {
  module.walk([&](scf::ForOp forOp) {
    Block &body = forOp.getRegion().front();
    constexpr int num = 8;
    SmallVector<scf::IfOp, num> ifOps;
    for (Operation &op : body)
      if (auto ifOp = dyn_cast<scf::IfOp>(&op))
        ifOps.push_back(ifOp);

    for (scf::IfOp ifOp : ifOps) {
      // 找 then region 中的 advance
      triton::AdvanceOp advanceOp;
      for (Operation &thenOp : ifOp.getThenRegion().front()) {
        if (auto adv = dyn_cast<triton::AdvanceOp>(thenOp)) {
          advanceOp = adv;
          break;
        }
      }
      if (!advanceOp) continue;

      // base 必须是 for的iter_arg
      Value base = advanceOp.getPtr();
      auto barg = dyn_cast<BlockArgument>(base);
      if (!barg || barg.getOwner() != &body) continue;

      // yield 去掉 advance 的返回值
      auto thenYield = cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto elseYield = cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

      int advanceIdx = -1;
      for (auto it : llvm::enumerate(thenYield.getOperands())) {
        if (it.value() == advanceOp.getResult()) {
          advanceIdx = it.index();
          break;
        }
      }
      
      if (advanceIdx == -1) continue;

      // 删除 advance
      SmallVector<Value> thenOps(thenYield.getOperands().begin(), thenYield.getOperands().end());
      SmallVector<Value> elseOps(elseYield.getOperands().begin(), elseYield.getOperands().end());

      thenOps.erase(thenOps.begin() + advanceIdx);
      elseOps.erase(elseOps.begin() + advanceIdx);

      thenYield->setOperands(thenOps);
      elseYield->setOperands(elseOps);
      
      // 重建 ifOp（去掉 advance 对应的 result）
      OpBuilder ifBuilder(ifOp);
      ifBuilder.setInsertionPoint(ifOp);

      // 构造新的 result types
      SmallVector<Type> newResultTypes;
      for (int i = 0; i < ifOp.getNumResults(); ++i) {
        if (i != advanceIdx)
          newResultTypes.push_back(ifOp.getResult(i).getType());
      }

      // 创建新的 if
      auto newIf = ifBuilder.create<scf::IfOp>(
          ifOp.getLoc(),
          newResultTypes,
          ifOp.getCondition(),
          /*withElseRegion=*/true);
      newIf->setAttr("ssbuffer", ifBuilder.getUnitAttr());
      // 把已经修改过 yield 的 region 搬过去
      newIf.getThenRegion().takeBody(ifOp.getThenRegion());
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());

      // 替换if result的user
      int newIdx = 0;
      for (int oldIdx = 0; oldIdx < ifOp.getNumResults(); ++oldIdx) {
        if (oldIdx == advanceIdx)
          continue;
        ifOp.getResult(oldIdx).replaceAllUsesWith(newIf.getResult(newIdx++));
      }

      OpBuilder builder(newIf);
      builder.setInsertionPointAfter(newIf);

      Value flag = newIf.getCondition();

      SmallVector<Value, 4> newOffsets;
      for (Value off : advanceOp.getOffsets()) {
        auto intTy = cast<IntegerType>(off.getType());
        auto zero = builder.create<arith::ConstantIntOp>(
            newIf.getLoc(), 0, intTy.getWidth());
        auto sel = builder.create<arith::SelectOp>(
            newIf.getLoc(), flag, off, zero);
        newOffsets.push_back(sel);
      }

      auto newAdvance = builder.create<triton::AdvanceOp>(
        newIf.getLoc(), base.getType(), base, newOffsets);

      // 原 if 的 advance result 的 users，接到 newAdvance
      ifOp.getResult(advanceIdx).replaceAllUsesWith(newAdvance.getResult());

      // 删除旧的ifOp和advance
      advanceOp.erase();
      ifOp.erase();
    }
  });
}

void processRedudantIf(ModuleOp module) {
    SmallVector<scf::ForOp> forOps;
    llvm::outs()<<module<<" wwwww\n\n\n";
    module.walk([&](scf::ForOp forOp) {
        auto initArgs = forOp.getInitArgs();
        if (initArgs.size() == 5)
        {
            forOps.push_back(forOp);
        }  
    });

    for (auto forOp : forOps) {
        auto initArgs = forOp.getInitArgs();
        Value newInit = initArgs[2];

        // 构建新的初始化参数列表
        SmallVector<Value> newInitArgs(initArgs.begin(), initArgs.end());
        newInitArgs.push_back(newInit);

        // 获取原循环的边界和步长
        Value lb = forOp.getLowerBound();
        Value ub = forOp.getUpperBound();
        Value step = forOp.getStep();

        // 创建新的 ForOp，插入点位于原操作之前
        OpBuilder builder(forOp);
        auto newForOp = builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInitArgs);

        // 获取新循环的 region 块（已自动包含循环索引和迭代参数）
        Block &newBlock = newForOp.getRegion().front();
        Block &oldBlock = forOp.getRegion().front();

        // 建立块参数的映射：原块参数 -> 新块参数（前6个对应）
        IRMapping mapper;
        for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i) {
            mapper.map(oldBlock.getArgument(i), newBlock.getArgument(i));
        }
        // 将原循环体中的操作（不包括终结符）克隆到新块中
        builder.setInsertionPointToStart(&newBlock);
        for (auto &op : oldBlock) {
            auto newOp = builder.clone(op, mapper);
        }

        // 在新块中查找第一个 scf::IfOp（即原代码中的第一个 if）
        scf::IfOp firstIfOp = nullptr;
        for (auto &op : newBlock.getOperations()) {
            if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
                firstIfOp = ifOp;
                break;
            }
        }
        assert(firstIfOp && "Expected at least one if op in the loop body");

        // 修改第一个 if 的 else 分支的 yield 操作：
        // 将其第二个操作数（索引1）从原来的 %arg9 改为新迭代参数（新块参数索引6）
        Block &elseBlock = firstIfOp.getElseRegion().front();
        auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());
        SmallVector<Value> newElseYieldOps(elseYield.getOperands());
        newElseYieldOps[1] = newBlock.getArgument(6); // 新迭代参数
        builder.setInsertionPoint(elseYield);
        builder.create<scf::YieldOp>(elseYield.getLoc(), newElseYieldOps);
        elseYield->erase();

        // 创建新的循环 yield 操作：原5个操作数 + 第一个 if 的第二个结果
        auto oldYield = cast<scf::YieldOp>(newBlock.getTerminator());
        SmallVector<Value> newYieldOps(oldYield.getOperands());
        newYieldOps.push_back(firstIfOp.getResult(1)); // 第一个 if 的第二个结果
        builder.setInsertionPointToEnd(&newBlock);
        builder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOps);
        oldYield.erase();

        // 将原 forOp 的所有使用替换为新 forOp 的前5个结果
        for (auto it : llvm::zip(forOp->getResults(), newForOp->getResults().take_front(5))) {
            std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
        }
    }
    for (auto forOp : forOps) {
        forOp.erase();
    }
}
// 针对依赖变量，对原本的for op增加double buffer相关的迭代参数
scf::ForOp addDoubleBuffForArgs(ModuleOp module, SmallVector<Value> uniqueDeps, int bufferNum) {
    mlir::OpBuilder builder(module.getContext());
    SmallVector<int64_t> depValueForIdxs;

    // ========== 找到scf.if所在的scf::ForOp ==========
    if (!isa<scf::ForOp>(uniqueDeps[0].getDefiningOp()->getParentOp())) {
        llvm::errs() << "Error: parent op of scf.if is not scf.for";
    }
    scf::ForOp forOp = dyn_cast<scf::ForOp>(uniqueDeps[0].getDefiningOp()->getParentOp());

    for(Value dependencyValue : uniqueDeps){
        // ========== 步骤1：验证目标Value是scf.if的返回值，并找到对应的scf::IfOp ==========
        Operation *ifOp = dependencyValue.getDefiningOp();
        if (!ifOp || !isa<scf::IfOp>(ifOp)) {
            llvm::errs() << "Error: 目标Value不是scf.if的返回值\n";
            return nullptr;
        }
        scf::IfOp targetIfOp = dyn_cast<scf::IfOp>(ifOp);
        
        // 确认当前Value是scf.if的第几个返回值
        int64_t depValueIdx = -1;
        for (auto [idx, result] : llvm::enumerate(targetIfOp.getResults())) {
            if (result == dependencyValue) {
                depValueIdx = idx;
                break;
            }
        }

        // ========== 步骤2：找到%38#2关联的scf.for迭代参数以及索引 ==========
        // %38#2对应scf.if else分支yield的第2个操作数 → 即%arg10
        Operation *elseYield = targetIfOp.elseYield();
        Value dependencyArg = elseYield->getOperand(depValueIdx); // depValueIdx=2，对应else yield的第2个参数

        int64_t depValueForIdx = -1;
        for (auto [idx, result] : llvm::enumerate(forOp.getRegionIterArgs())) {
            if (result == dependencyArg) {
                depValueForIdx = idx;
                break;
            }
        }
        depValueForIdxs.push_back(depValueForIdx);
        llvm::outs() << "depValueForIdx: " << depValueForIdx << '\n';
    }

    llvm::outs() << "oldFor: " << forOp << '\n';
    
    // 获取原始循环的信息
    Value originalLowerBound = forOp.getLowerBound();
    Value originalUpperBound = forOp.getUpperBound();
    Value originalStep = forOp.getStep();
    SmallVector<Value> originalInitArgs = forOp.getInitArgs();
    SmallVector<Value> iterArgs;
    for (auto arg : originalInitArgs) {
        iterArgs.push_back(arg);
    }
    auto yields = forOp.getBody()->getTerminator();

    // 创建计数器初始零值
    Value counterInit = nullptr;
    mlir::Operation* parentOp = forOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    // 向上遍历查找scope.scope操作
    while (parentOp) {
        if (dyn_cast<scope::ScopeOp>(parentOp)) {
            scopeOp = parentOp;
            break;
        }
        parentOp = parentOp->getParentOp();
    }

    builder.setInsertionPoint(scopeOp);
    Location loc = forOp.getLoc();
    auto boundType = originalLowerBound.getType();
    counterInit = builder.create<arith::ConstantIntOp>(loc, boundType, 0);

    // 添加和depValueForIdxs相同的迭代参数和计数器
    for (int64_t idx : depValueForIdxs) {
        for (int i = 0; i < bufferNum - 1; i++) {
            iterArgs.push_back(originalInitArgs[idx]);
        }
        
        // 在迭代参数中添加计数器
        for (int i = 0; i < 2; i++) {
            iterArgs.push_back(counterInit);
        }
    }

    builder.setInsertionPoint(forOp);
    // 创建新的for循环
    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(),
        originalLowerBound,
        originalUpperBound,
        originalStep,
        iterArgs);
    
    // 设置IR映射表，将旧循环的变量映射到新循环
    IRMapping mapper;
    
    // 映射迭代变量
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
    
    // 映射迭代参数
    for (auto [oldArg, newArg] : 
         llvm::zip(forOp.getRegionIterArgs(), 
                  newForOp.getRegionIterArgs())) {
        mapper.map(oldArg, newArg);
    }
    
    SmallVector<Value> newArgs;
    for (int i = forOp.getRegionIterArgs().size(); i < newForOp.getRegionIterArgs().size(); i++) {
        newArgs.push_back(newForOp.getRegionIterArgs()[i]);
    }
    // 克隆循环体内容到新循环
    auto &newLoopBody = *newForOp.getBody();
    builder.setInsertionPointToStart(&newLoopBody);
    
    for (auto &op : forOp.getBody()->without_terminator()) {
        builder.clone(op, mapper);
    }
    
    // 克隆yield操作
    if (auto yieldOp = dyn_cast<scf::YieldOp>(yields)) {
        SmallVector<Value> newYieldOperands;
        for (auto operand : yieldOp.getOperands()) {
            newYieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        // 将新增的迭代参数添加到yield操作数中
        for (auto currentCounter : newArgs) {                
            newYieldOperands.push_back(currentCounter);
        }
        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    }
    
    // 替换原循环的结果
    unsigned numOriginalResults = forOp.getNumResults();
    SmallVector<Value> originalResults;
    for (unsigned i = 0; i < numOriginalResults; i++) {
        originalResults.push_back(newForOp.getResult(i));
    }
    forOp.replaceAllUsesWith(originalResults);
    
    // 8. 删除原循环
    forOp.erase();

    llvm::outs() << "for op erased!\n";
    return newForOp;
}

static bool isScalarValue(Value value)
{
  if (!value) return false;
  Type type = value.getType();

  return type.isIntOrIndex() || mlir::isa<FloatType>(type) ||
       mlir::isa<triton::PointerType>(type) ||
       mlir::isa<LLVM::LLVMPointerType>(type);
}

SmallVector<Value> buildNBufferProducer(OpBuilder &builder, Location loc,
                                        Value frontCnt, Value newDepVal,
                                        ArrayRef<Value> buffs,
                                        ArrayRef<Value> constants) {
  // N-buffer producer: determines which buffer is written to newDepVal based on frontCnt % N
  const int N = buffs.size();
  SmallVector<Value> results;

  // idx = frontCnt % N
  Value bufferIndex =
      builder.create<arith::RemSIOp>(loc, frontCnt, constants[N]);

  // 1. buffer0: handle the first buffer separately
  Value isBuffer0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  bufferIndex, constants[0]);

  auto dstShapedType = mlir::dyn_cast<ShapedType>(newDepVal.getType());
  Value mask;
  if (isScalarValue(isBuffer0) == 0) {
      auto maskType = RankedTensorType::get(dstShapedType.getShape(), isBuffer0.getType());
      mask = builder.create<tensor::SplatOp>(loc, maskType, isBuffer0);
      llvm::outs() << "build NBufferProducer debug15\n";
      llvm::outs().flush();
  } else {
    mask = isBuffer0;
  }
  Value newBuff0 = builder.create<arith::SelectOp>(loc, mask, newDepVal, buffs[0]);

  results.push_back(newBuff0);

  // 2. Double-buffer specialization (when N == 2, a direct select is sufficient)
  if (N == 2) {

    Value newBuff1 = builder.create<arith::SelectOp>(loc, mask, buffs[1], newDepVal);

    auto nextCnt = builder.create<arith::AddIOp>(loc, frontCnt, constants[1]);

    results.push_back(newBuff1);
    results.push_back(nextCnt.getResult());

    return results;
  }

  // 3. Build the root IF: when idx == 0,
 	// use the first buffer; otherwise enter the nestedIf chain to use other buffers
  SmallVector<Type> resultTypes;
  for (int i = 1; i < N; ++i)
    resultTypes.push_back(buffs[i].getType());

  auto rootIf = builder.create<scf::IfOp>(loc, resultTypes, isBuffer0, true);

  // ---- THEN: buffers are directly forwarded ----
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&rootIf.getThenRegion().front());

    SmallVector<Value> unchangedBuffers(buffs.begin() + 1, buffs.end());

    builder.create<scf::YieldOp>(loc, unchangedBuffers);
  }

  // 4. Construct the nested-if chain, updating one buffer at each level
  Block *currentElseBlock = &rootIf.getElseRegion().front();

  scf::IfOp parentIf = rootIf;

  for (int i = 1; i < N - 1; ++i) {

    builder.setInsertionPointToStart(currentElseBlock);

    // Check whether the current buffer is selected
    Value isCurrent = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, bufferIndex, constants[i]);

    // Update buffer[i]
    dstShapedType = mlir::dyn_cast<ShapedType>(newDepVal.getType());
    auto maskType = RankedTensorType::get(dstShapedType.getShape(), isCurrent.getType());
    mask = builder.create<tensor::SplatOp>(loc, maskType, isCurrent);
    Value updatedBuffer = builder.create<arith::SelectOp>(loc, mask, newDepVal, buffs[i]);

    // If this is the last level: directly yield both buffers
    if (i == N - 2) {

        dstShapedType = mlir::dyn_cast<ShapedType>(newDepVal.getType());
        maskType = RankedTensorType::get(dstShapedType.getShape(), isCurrent.getType());
        mask = builder.create<tensor::SplatOp>(loc, maskType, isCurrent);
        Value lastBuffer = builder.create<arith::SelectOp>(loc, mask, buffs[N - 1], newDepVal);

        builder.create<scf::YieldOp>(loc, ValueRange {updatedBuffer, lastBuffer});

        break;
    }

    // Create the next nested if
    SmallVector<Type> subResultTypes;
    for (int j = i + 1; j < N; ++j)
      subResultTypes.push_back(buffs[j].getType());

    auto nextIf =
        builder.create<scf::IfOp>(loc, subResultTypes, isCurrent, true);

    // THEN: forward the remaining buffers
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&nextIf.getThenRegion().front());

      SmallVector<Value> remainingBuffers(buffs.begin() + i + 1, buffs.end());

      builder.create<scf::YieldOp>(loc, remainingBuffers);
    }

    // Update the else yield
    builder.setInsertionPointToEnd(&parentIf.getElseRegion().front());

    SmallVector<Value> yields;
    yields.push_back(updatedBuffer);
    yields.append(nextIf.getResults().begin(), nextIf.getResults().end());

    builder.create<scf::YieldOp>(loc, yields);

    parentIf = nextIf;
    currentElseBlock = &nextIf.getElseRegion().front();
  }

  // 5. Update the frontCnt counter
  builder.setInsertionPointAfter(rootIf);

  auto nextCnt = builder.create<arith::AddIOp>(loc, frontCnt, constants[1]);

  // Collect results
  results.append(rootIf.getResults().begin(), rootIf.getResults().end());

  results.push_back(nextCnt.getResult());

  return results;
}

SmallVector<Value> buildNBufferConsumer(OpBuilder &builder, Location loc,
                                        Value postCnt, ArrayRef<Value> oldBuffs,
                                        ArrayRef<Value> constants) {
  // Consumer: selects which buffer to read based on postCnt % N
  const int bufferNum = oldBuffs.size();
  SmallVector<Value> results;

  // idx = postCnt % N
  Value bufferIndex =
      builder.create<arith::RemSIOp>(loc, postCnt, constants[bufferNum]);

  Value isBuffer0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                bufferIndex, constants[0]);
  auto dstShapedType = mlir::dyn_cast<ShapedType>(oldBuffs[0].getType());
  Value mask;
  if (isScalarValue(isBuffer0) == 0) {
    auto maskType = RankedTensorType::get(dstShapedType.getShape(), isBuffer0.getType());
    mask = builder.create<tensor::SplatOp>(loc, maskType, isBuffer0);
  } else {
    mask = isBuffer0;
  }

  // 1. Double-buffer specialization (avoid generating scf.if)
  if (bufferNum == 2) {
    Value selected = builder.create<arith::SelectOp>(loc, mask, oldBuffs[0], oldBuffs[1]);
    auto nextCnt = builder.create<arith::AddIOp>(loc, postCnt, constants[1]);

    results.push_back(selected);
    results.push_back(nextCnt);

    return results;
  }

  // 2. Build the root IF:
 	// when idx == 0, use the first buffer; otherwise enter the nestedIf chain to use other buffers
  SmallVector<Type> resultTypes{oldBuffs[0].getType()};

  auto rootIf = builder.create<scf::IfOp>(loc, resultTypes, isBuffer0, true);

  // ---- THEN: directly return buffer0 ----
  {
    builder.setInsertionPointToStart(&rootIf.getThenRegion().front());

    builder.create<scf::YieldOp>(loc, oldBuffs[0]);
  }

  // 3. Construct the nested-if chain
  Block *currentElse = &rootIf.getElseRegion().front();

  for (int i = 1; i < bufferNum - 2; ++i) {

    builder.setInsertionPointToStart(currentElse);

    Value isCurrent = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, bufferIndex, constants[i]);

    auto nestedIf = builder.create<scf::IfOp>(
        loc, TypeRange{oldBuffs[0].getType()}, isCurrent, true);

    // THEN → return the current buffer
    {
      builder.setInsertionPointToStart(&nestedIf.getThenRegion().front());

      builder.create<scf::YieldOp>(loc, oldBuffs[i]);
    }

    // ELSE → yield nested result
    builder.setInsertionPointToEnd(currentElse);
    builder.create<scf::YieldOp>(loc, nestedIf.getResult(0));

    // Enter the next else branch
    currentElse = &nestedIf.getElseRegion().front();
  }

  // 4. Final level (use select to finish)
  builder.setInsertionPointToStart(currentElse);

  int last = bufferNum - 2;

  Value isLast = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               bufferIndex, constants[last]);

  auto maskType = RankedTensorType::get({}, isLast.getType());
  dstShapedType = mlir::dyn_cast<ShapedType>(oldBuffs[last].getType());
  maskType = RankedTensorType::get(dstShapedType.getShape(), isLast.getType());
  mask = builder.create<tensor::SplatOp>(loc, maskType, isLast);

  Value finalSelect = builder.create<arith::SelectOp>(loc, mask, oldBuffs[last], oldBuffs[last + 1]);

  builder.create<scf::YieldOp>(loc, finalSelect);

  // rootIf result = selected buffer
  results.push_back(rootIf.getResult(0));

  // 5. Update the postCnt counter
  builder.setInsertionPointAfter(rootIf);

  auto nextCnt = builder.create<arith::AddIOp>(loc, postCnt, constants[1]);

  results.push_back(nextCnt);

  return results;
}

void replaceDepsMap(
    scf::IfOp oldIfOp,
    scf::IfOp newIfOp,
    SmallVector<Value> &newDeps,
    bool isFront,
    DenseMap<scf::IfOp, SmallVector<Value>> &newIfResultDeps)
{
  mlir::IRMapping valueMap;

  // old result -> new result
  for (unsigned i = 0; i < oldIfOp.getNumResults(); ++i) {
    valueMap.map(oldIfOp.getResult(i), newIfOp.getResult(i));
  }

  if (isFront) {
    for (int i = 0; i < newDeps.size(); i++) {
      Value v = newDeps[i];
      if (valueMap.contains(v))
        newDeps[i] = valueMap.lookup(v);
    }
  }

  // rewrite deps in-place
  for (auto &it : newIfResultDeps) {
    auto &deps = it.second;

    for (auto &value : deps) {
      if (auto mapped = valueMap.lookupOrNull(value))
        value = mapped;
    }
  }
}

scf::IfOp addResultsForFrontIfOp(scf::IfOp frontIfOp, OpBuilder builder,
                                 int bufferNum, Value depValue,
                                 SmallVector<Value> constants,
                                 SmallVector<Value> buffs, Value frontCnt,
                                 Value postCnt,
                                 SmallVector<int> &extraResultIndices,
                                 SmallVector<Value> &newDeps,
                                 DenseMap<scf::IfOp, SmallVector<Value>> &newIfResultDeps)
{
  OpBuilder::InsertionGuard guard(builder);

  Location loc = frontIfOp.getLoc();
  Value cond = frontIfOp.getCondition();

  auto &oldThenBlock = frontIfOp.getThenRegion().front();
  auto &oldElseBlock = frontIfOp.getElseRegion().front();

  // New result types = old results + extra buffers + counter
  SmallVector<Type> newResultTypes(frontIfOp.getResultTypes().begin(),
                                   frontIfOp.getResultTypes().end());

  for (int i = 1; i < bufferNum; ++i)
    newResultTypes.push_back(buffs[i].getType());

  newResultTypes.push_back(frontCnt.getType());

  unsigned oldNumResults = frontIfOp.getNumResults();

  // Create new IfOp
  builder.setInsertionPoint(frontIfOp);
  auto newIfOp =
      builder.create<scf::IfOp>(loc, newResultTypes, cond, /*hasElse=*/true);

  SmallVector<int> bufferIndices(bufferNum);
  SmallVector<Value> newBuffs;
  int frontCntIndex = -1;

  // THEN region
  {
    mlir::IRMapping mapping;
    Block &newThenBlock = newIfOp.getThenRegion().front();

    builder.setInsertionPointToStart(&newThenBlock);

    // Clone original then body
    for (auto &op : oldThenBlock.without_terminator())
      builder.clone(op, mapping);

    // Update dependency value position inf ifOp results
    auto result = dyn_cast<OpResult>(depValue);
    if (!result) {
      llvm::outs() << "depValue is not a result Value!\n";
      return nullptr;
    }

    int depIdx = result.getResultNumber();
    Value depYieldValue = frontIfOp.thenYield()->getOperand(depIdx);

    Value newDepVal = mapping.contains(depYieldValue)
                          ? mapping.lookup(depYieldValue)
                          : depYieldValue;

    builder.setInsertionPointAfter(newDepVal.getDefiningOp());

    // Create N buffer
    SmallVector<Value> produced = buildNBufferProducer(
        builder, loc, frontCnt, newDepVal, buffs, constants);

    // Last value in newBuffs is the counter
    newBuffs.append(produced.begin(), produced.end() - 1);

    // Rebuild new yield
    SmallVector<Value> thenOperands;

    for (Value v : oldThenBlock.getTerminator()->getOperands()) {
      Value mapped = mapping.lookupOrDefault(v);

      // Replace first buffer
      if (mapped == newDepVal) {
        thenOperands.push_back(newBuffs[0]);
        bufferIndices[0] = thenOperands.size() - 1;
      } else {
        thenOperands.push_back(mapped);
      }
    }

    // Replace other buffer
    for (int i = 1; i < bufferNum; ++i) {
      thenOperands.push_back(newBuffs[i]);
      bufferIndices[i] = thenOperands.size() - 1;
    }

    // Add counter
    thenOperands.push_back(produced.back());
    frontCntIndex = thenOperands.size() - 1;

    builder.setInsertionPointToEnd(&newThenBlock);
    builder.create<scf::YieldOp>(loc, thenOperands);

    // record new result indices
    for (int idx : bufferIndices)
      extraResultIndices.push_back(idx);

    extraResultIndices.push_back(frontCntIndex);
  }

  // ELSE region
  {
    mlir::IRMapping mapping;
    Block &newElseBlock = newIfOp.getElseRegion().front();

    builder.setInsertionPointToStart(&newElseBlock);

    // Clone original else body
    for (auto &op : oldElseBlock.without_terminator())
      builder.clone(op, mapping);

    builder.setInsertionPointToEnd(&newElseBlock);

    SmallVector<Value> elseOperands;

    for (Value v : oldElseBlock.getTerminator()->getOperands())
      elseOperands.push_back(mapping.lookupOrDefault(v));

    // Add buffer
    for (int i = 1; i < bufferNum; ++i)
      elseOperands.push_back(buffs[i]);

    // Add counter
    elseOperands.push_back(frontCnt);

    builder.create<scf::YieldOp>(loc, elseOperands);
  }

  // Update dependency value
  replaceDepsMap(frontIfOp, newIfOp, newDeps, true, newIfResultDeps);

  // Replace old ifOp
  frontIfOp.replaceAllUsesWith(newIfOp.getResults().take_front(oldNumResults));

  frontIfOp.erase();

  return newIfOp;
}

scf::IfOp addResultsForPostIfOp(scf::IfOp postIfOp, scf::IfOp newfrontIfOp,
                                OpBuilder builder, int bufferNum,
                                Value newDepValue, SmallVector<Value> constants,
                                SmallVector<Value> buffs, Value frontCnt,
                                Value postCnt,
                                SmallVector<int> &extraResultIndices,
                                SmallVector<Value> &newDeps,
                                DenseMap<scf::IfOp, SmallVector<Value>> &newIfResultDeps)
{
  // 1. Parse the extra result indices produced by frontIf (added buffers and counters)
  SmallVector<int> bufferIndices(extraResultIndices.begin(),
                                 extraResultIndices.end() - 1);
  int frontCntIndex = extraResultIndices[bufferNum];

  Location ifLoc = postIfOp.getLoc();
  Value cond = postIfOp.getCondition();

  auto &oldThenBlock = postIfOp.getThenRegion().front();
  auto &oldElseBlock = postIfOp.getElseRegion().front();

  // 2. Create a new IfOp (add a new postCnt result)
  SmallVector<Type> newResultTypes(postIfOp.getResultTypes().begin(),
                                   postIfOp.getResultTypes().end());
  newResultTypes.push_back(postCnt.getType());

  builder.setInsertionPoint(postIfOp);
  auto newIfOp = builder.create<scf::IfOp>(ifLoc, newResultTypes, cond,
                                           /*hasElse=*/true);

  mlir::IRMapping mapping;

  // 3. THEN region: clone the original logic, insert the multibuffer consumer and update dependency buffers
  auto &newThenBlock = newIfOp.getThenRegion().front();
  builder.setInsertionPointToStart(&newThenBlock);

  // clone then body
  for (auto &op : oldThenBlock.without_terminator())
    builder.clone(op, mapping);
  builder.setInsertionPointToStart(&newThenBlock);

   // Find dependency uses that need to be replaced (located inside the current IfOp)
  SmallVector<OpOperand*> replaceUses;
  for (auto &use : newDepValue.getUses()) {
    if (newIfOp == dyn_cast<scf::IfOp>(use.getOwner()->getParentOp())) {
      replaceUses.push_back(&use);
    }
  }

  // Collect buffers produced by frontIf
  SmallVector<Value> oldBuffers;
  for (int i = 0; i < bufferIndices.size(); ++i)
    oldBuffers.push_back(newfrontIfOp.getResult(bufferIndices[i]));

  // Multibuffer consumer caculation
  SmallVector<Value> consumerResults =
      buildNBufferConsumer(builder, ifLoc, postCnt, oldBuffers, constants);

  Value selectedBuffer = consumerResults[0];
  Value nextPostCnt = consumerResults[1];

  // Replace dependent buffer
  for (auto *usePtr : replaceUses) {
      usePtr->set(selectedBuffer);
  }

  // Create then yield
  SmallVector<Value> thenOperands;
  for (auto v : oldThenBlock.getTerminator()->getOperands())
    thenOperands.push_back(mapping.lookupOrDefault(v));

  int postCntIndex = thenOperands.size();
  thenOperands.push_back(nextPostCnt);

  builder.setInsertionPointToEnd(&newThenBlock);
  builder.create<scf::YieldOp>(ifLoc, thenOperands);
  extraResultIndices.push_back(postCntIndex);

  // 4. ELSE region：forward counter directly
  auto &newElseBlock = newIfOp.getElseRegion().front();

  for (auto &op : oldElseBlock.without_terminator())
    builder.clone(op, mapping);

  builder.setInsertionPointToEnd(&newElseBlock);

  SmallVector<Value> elseOperands;
  for (auto v : oldElseBlock.getTerminator()->getOperands())
    elseOperands.push_back(mapping.lookupOrDefault(v));

  elseOperands.push_back(postCnt);

  builder.create<scf::YieldOp>(ifLoc, elseOperands);

  // 5. Replace old ifOp with new one
  auto oldNumResults = postIfOp.getNumResults();

  // Update depency value
  replaceDepsMap(postIfOp, newIfOp, newDeps, false, newIfResultDeps);

  postIfOp.replaceAllUsesWith(newIfOp.getResults().take_front(oldNumResults));

  postIfOp.erase();

  return newIfOp;
}

void addMultiBuffCaculate(ModuleOp module, SmallVector<Value> newUniqueDeps,
                          DenseMap<scf::IfOp, SmallVector<Value>> &ifResultDeps, scf::ForOp &newForOp, int bufferNum)
{

  // ============================================================
  // Overall Idea
  //
  // For each dependency Value:
  // 1. Find the front IfOp that produces it
  // 2. Add multi-buffer results to the front IfOp
  // 3. Find the post IfOp that consumes the result and extend it accordingly
  // 4. Update the for-loop yield so that buffer states are correctly propagated
  // ============================================================

  OpBuilder builder(module.getContext());
  int processedDepCount = 0;

  SmallVector<scf::IfOp> postIfOps;
  newForOp.walk([&](scf::IfOp postIfOp) {
    postIfOps.push_back(postIfOp);
  });
  for (auto postIfOp:postIfOps) {
    if (!ifResultDeps.count(postIfOp)) {
      continue;
    }
    auto newDeps = ifResultDeps[postIfOp];
    for (int depValueIdx = 0; depValueIdx < newDeps.size(); depValueIdx++) {
      Value depValue = newDeps[depValueIdx];

      // Step 1. Locate the front IfOp that produces depValue
      Operation *defOp = depValue.getDefiningOp();
      if (!defOp || !isa<scf::IfOp>(defOp)) {
        llvm::outs() << "Error: depValue is not produced by scf.if\n";
        break;
      }

      scf::IfOp frontIfOp = cast<scf::IfOp>(defOp);

      // Position of depValue in the IfOp results
      auto result = dyn_cast<OpResult>(depValue);
      if (!result) {
        llvm::outs() << "depValue is not an OpResult!\n";
        return;
      }

      int64_t depResultIndex = result.getResultNumber();

      // Position of depValue in the IfOp results
      Value depYieldValue = frontIfOp.thenYield()->getOperand(depResultIndex);

      // Step 2. Find the multi-buffer position in the ForOp
      int64_t extraArgBaseIdx =
          newForOp.getRegionIterArgs().size() -
          (2 + bufferNum - 1) * (newUniqueDeps.size() - processedDepCount++);

      // Collect all buffers
      SmallVector<Value> buffers;

      // buffer0 来自 else yield
      buffers.push_back(frontIfOp.elseYield()->getOperand(depResultIndex));

      // Other buffers come from for iter args
      for (int i = 1; i < bufferNum; ++i) {
        buffers.push_back(newForOp.getRegionIterArgs()[extraArgBaseIdx + i - 1]);
      }

      // Two counters
      Value frontCnt =
          newForOp.getRegionIterArgs()[extraArgBaseIdx + bufferNum - 1];
      Value postCnt = newForOp.getRegionIterArgs()[extraArgBaseIdx + bufferNum];

      // Step 3. Create constants (0 ~ bufferNum) for rem / cmp buffer selection logic
      SmallVector<Value> constants;
      builder.setInsertionPoint(frontIfOp);

      auto dataType = frontCnt.getType();
      for (int i = 0; i <= bufferNum; ++i) {
        constants.push_back(builder.create<arith::ConstantOp>(
            frontIfOp.getLoc(), dataType,
            builder.getIntegerAttr(dataType, i)));
      }

      // Record the positions of newly added results in the IfOp
      SmallVector<int> extraResultIndices(bufferNum + 1);
      extraResultIndices.clear();

      // Step 4. Extend the front IfOp
      scf::IfOp newFrontIfOp = addResultsForFrontIfOp(
          frontIfOp, builder, bufferNum, depValue, constants, buffers, frontCnt,
          postCnt, extraResultIndices, newDeps, ifResultDeps);

      // buffer result indices
      SmallVector<int> bufferResultIndices(extraResultIndices.begin(),
                                          extraResultIndices.end() - 1);

      int frontCntResultIndex = extraResultIndices[bufferNum];

      Value newDepValue = newFrontIfOp.getResult(depResultIndex);

      // Step 5. Find the post IfOp that consumes the dependency value
      scf::IfOp postIfOp = nullptr;

      for (auto &use : newDepValue.getUses()) {
        if (auto candidate = dyn_cast<scf::IfOp>(use.getOwner()->getParentOp())) {
          postIfOp = candidate;
          break;
        }
      }

      if (!postIfOp) {
        llvm::outs() << "Error: no consuming IfOp found.\n";
        return;
      }

      // Step 6. Extend the post IfOp

      scf::IfOp newPostIfOp = addResultsForPostIfOp(
          postIfOp, newFrontIfOp, builder, bufferNum, newDepValue, constants,
          buffers, frontCnt, postCnt, extraResultIndices, newDeps, ifResultDeps);

      llvm::outs() << "after addResultsForPostIfOp.\n";

      int postCntResultIndex = extraResultIndices.back();

      // Step 7. Update the ForOp yield (buffer propagation)
      auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());

      // Update buffer1 ~ bufferN
      for (int i = 1; i < bufferNum; ++i) {

        int yieldIdx = extraArgBaseIdx + (i - 1);

        if (yieldIdx < forYield->getNumOperands() &&
            bufferResultIndices[i] < newFrontIfOp.getNumResults()) {

          forYield->setOperand(yieldIdx, newFrontIfOp.getResult(bufferResultIndices[i]));

          llvm::outs() << "Replaced yield operand " << yieldIdx << "\n";
        } else {
          llvm::errs() << "Warning: index out of range\n";
        }
      }

      // Step 8. Update frontCnt
      OpOperand *frontCntYieldUse = nullptr;

      for (auto &use : frontCnt.getUses()) {
        if (isa<scf::YieldOp>(use.getOwner()) &&
            newForOp == use.getOwner()->getParentOp()) {
          frontCntYieldUse = &use;
          break;
        }
      }

      frontCntYieldUse->set(newFrontIfOp.getResult(frontCntResultIndex));

      // Step 9. Update postCnt
      OpOperand *postCntYieldUse = nullptr;

      for (auto &use : postCnt.getUses()) {
        if (isa<scf::YieldOp>(use.getOwner()) &&
            newForOp == use.getOwner()->getParentOp()) {
          postCntYieldUse = &use;
          break;
        }
      }

      postCntYieldUse->set(newPostIfOp.getResult(postCntResultIndex));
    }
  }

  
  llvm::outs() << "multibuffer end!\n";
}

// Compute the nesting level of an ifOp within the specified forOp
static int computeIfLevel(scf::IfOp ifOp, scf::ForOp rootForOp)
{
  int level = 1;

  Operation *parent = ifOp->getParentOp();

  while (parent && parent != rootForOp.getOperation()) {
    if (isa<scf::IfOp>(parent))
      level++;

    parent = parent->getParentOp();
  }

  return level;
}

int assignIfOpLevels(scf::ForOp forOp)
{
  SmallVector<scf::IfOp> targetIfOps;
  int maxLevel = 0;
  // Collect all ifOp assigned with ssbuffer tag
  forOp.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr("ssbuffer")) {
      targetIfOps.push_back(ifOp);
    }
  });

  // Caculate buffer levels
  for (auto ifOp : targetIfOps) {
    int level = computeIfLevel(ifOp, forOp);
    maxLevel = std::max(level, maxLevel);
    Builder builder(ifOp.getContext());
    ifOp->setAttr("ssbuffer.level",
                  builder.getI32IntegerAttr(level));
  }
  return maxLevel;
}

static bool hasSSBufferIf(scf::ForOp forOp)
{
  bool found = false;

  forOp.walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr("ssbuffer")) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return found;
}

static bool hasAncestorSSBufferFor(scf::ForOp forOp)
{
  Operation *parent = forOp->getParentOp();

  while (parent) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasSSBufferIf(parentFor))
        return true;
    }
    parent = parent->getParentOp();
  }

  return false;
}

static bool hasAncestorRootFor(scf::ForOp forOp)
{
  Operation *parent = forOp->getParentOp();

  while (parent) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasSSBufferIf(parentFor))
        return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

SmallVector<Value> collectIfInfo(
    scf::ForOp &curForOp,
    DenseMap<scf::IfOp, SmallVector<Value>> &ifDeps,
    int level)
{
  // Find all dependency variables based on the inputs and outputs of ifOp
  SmallVector<Value> allDeps;
  DenseSet<Value> producedValues;
  scf::ForOp newForOp = nullptr;
  curForOp.walk([&](scf::IfOp ifOp) {
    auto attr = ifOp->getAttrOfType<IntegerAttr>("ssbuffer.level");
    // No level or level mismatch → continue searching
    if (!attr || attr.getInt() != level)
      return WalkResult::advance();

    // Levels match → check the direct parent
    if (auto parentFor = dyn_cast<scf::ForOp>(ifOp->getParentOp())) {
      newForOp = parentFor;   // 更新
    }

    // Stop walking regardless of whether the parent is a for-loop
    return WalkResult::interrupt();
  });

  if (newForOp)
    curForOp = newForOp;

  // Step 1: Collect first to preserve order
  SmallVector<scf::IfOp> ifOps;
  curForOp.walk([&](scf::IfOp ifOp) {
    auto curLevel = ifOp->getAttrOfType<IntegerAttr>("ssbuffer.level");
    if (!curLevel || curLevel.getInt() != level) {
      return WalkResult::advance();
    }
    ifOps.push_back(ifOp);
    return WalkResult::advance();
  });
  llvm::outs()<<"ifOps:"<<ifOps.size()<<"\n";
  
  int miniDepNum = 2;
  if (ifOps.size() < miniDepNum) {
    return allDeps;
  }
  // Step 2: Process in order
  for (auto ifOp : ifOps) {
    llvm::outs()<<"ifOp->getOperands():"<<ifOp->getOperands().size()<<"\n";
    SmallVector<Value> deps;
    if (producedValues.empty()) {
      llvm::outs()<<"producedValues为空!"<<"\n";
    }
    
    // inputs
    Region &thenRegion = ifOp.getThenRegion();
    for (Operation &op : thenRegion.front()) {
      for (Value operand : op.getOperands()) {
        for (Value v : producedValues) {
          if (operand == v && !llvm::is_contained(deps, operand)) {
            deps.push_back(operand);
          }
        }
      }
    }

    // outputs
    for (Value result : ifOp.getResults()) {
      producedValues.insert(result);
    }

    if (!deps.empty()) {
      ifDeps[ifOp] = deps;
      allDeps.append(deps.begin(), deps.end());
    }
  }
  llvm::outs().flush();
  return allDeps;
}

bool isCube(scope::ScopeOp scope) {
  bool ret = false;
  scope.walk([&](Operation *op) {
    if (isa<triton::DotOp>(op)) {
      ret = true;
    }
  });
  return ret;
}

// Traverse each Vector scope, find the outer ForOp, and process internal IfOps
void WalkAIVNestedForAndProcess(
    ModuleOp module, DenseMap<scf::IfOp, SmallVector<Value>> &ifResultDeps,
    int bufferNum) {
  if (bufferNum < 2) {
    return;
  }

  module.walk([&](scope::ScopeOp scope) {
    if (isCube(scope)) {
      return;
    }

    // Traverse ForOps inside the Cube scope (outer loops)
    SmallVector<scf::ForOp> targetFors;

    scope.walk([&](scf::ForOp forOp) {

      // Must contain an ssbuffer if
      if (!hasSSBufferIf(forOp))
        return WalkResult::advance();

      // Skip if an ancestor is already the root
      if (hasAncestorRootFor(forOp))
        return WalkResult::advance();

      // Find rootForOp
      targetFors.push_back(forOp);

      return WalkResult::advance();
    });
    llvm::outs() << "targetFors: " << targetFors.size();
    int maxLevels;
    for (auto outerFor : targetFors) {
      ifResultDeps.clear();
      scf::ForOp currentFor = outerFor;
      maxLevels = assignIfOpLevels(currentFor);
      for (int level = 1; level <= maxLevels; level++) {
        auto uniqueDeps = collectIfInfo(currentFor, ifResultDeps, level);
        llvm::outs()<<"maxLevels:"<<maxLevels<<"\n";
        if (uniqueDeps.empty()) {
          continue;
        }
        llvm::outs()<<"uniqueDeps:"<<uniqueDeps.size()<<"\n";
        auto newForOp = addDoubleBuffForArgs(module, uniqueDeps, bufferNum);
        DenseMap<scf::IfOp, SmallVector<Value>> newIfResultDeps;
        auto uniqueList = collectIfInfo(newForOp, newIfResultDeps, level);
        addMultiBuffCaculate(module, uniqueList, newIfResultDeps, newForOp, bufferNum);
      }
    }
  });
}

void DAGSSBufferPass::runOnOperation() {
  auto module = getOperation();

  AddIfCondition(module);

  FlowSssbuf(module);
  ControlSsbufV2(module);

  // advance不能出现在if里, 规避处理
  ChangeAdvanceOpForm(module);

  DenseMap<scf::IfOp, SmallVector<Value>> ifResultDeps;
  WalkAIVNestedForAndProcess(module, ifResultDeps, 2);

  return;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDAGSSBufferPass() {
  return std::make_unique<DAGSSBufferPass>();
}

