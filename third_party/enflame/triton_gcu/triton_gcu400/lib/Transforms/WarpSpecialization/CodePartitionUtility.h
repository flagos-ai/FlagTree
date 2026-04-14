/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRITON_GCU_DIALECT_TRANSFORMS_WS_CODEPARTITIONUTILITY_H_
#define TRITON_GCU_DIALECT_TRANSFORMS_WS_CODEPARTITIONUTILITY_H_

#include <algorithm>
#include <numeric>
#include <utility>

#include "Utility.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::gcu {

//===----------------------------------------------------------------------===//
// Channel and CommChannel struct
//===----------------------------------------------------------------------===//

/*
 * Channel is a pair of producer and consumers.
 * producer is the task id of the producer.
 * consumers is a list of task ids of the consumers.
 * op is the operation that produces the data.
 * operandIdx is the index of the operand that produces the data.
 * numBuffers is the number of buffers.
 * numWarps is the number of warps.
 */
struct Channel {
public:
  using Relation = std::pair<int, SmallVector<int>>;

  Channel(int producer, SmallVector<int> &consumers, Operation *op,
          unsigned operandIdx, unsigned numBuffers, unsigned numWarps)
      : relation(producer, consumers), op(op), operandIdx(operandIdx),
        numBuffers(numBuffers), numWarps(numWarps) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && operandIdx == c.operandIdx && op == c.op;
  }
  virtual ~Channel() = default;

  Operation *getDstOp() { return op; }
  unsigned getDstOperandIdx() { return operandIdx; }
  virtual Value getSrcOperand() { return op->getOperand(operandIdx); }
  virtual Operation *getSrcOp() { return getSrcOperand().getDefiningOp(); }

  Relation relation; // producer task Id, a list of consumer task Ids
  Operation *op;
  unsigned operandIdx;
  unsigned numBuffers;
  unsigned numWarps;
};

/*
 * CommChannel is a pair of producer and consumers.
 * producer is the task id of the producer.
 * consumers is a list of task ids of the consumers.
 * tokens is a map of task id to token.
 * producerBarrier is the barrier for the producer.
 * consumerBarriers is a map of task id to barrier.
 */
struct CommChannel {
  DenseMap<int, Value> tokens;
  // Producer barrier is only needed when the producer op itself can update the
  // barrier inline, such as the TMA load.
  std::optional<Value> producerBarrier;
  // Consumer barrier is only needed when the consumer op itself can update the
  // barrier inline, such as the TCGen5MMAOp.
  DenseMap<int, Value> consumerBarriers;
};

//===----------------------------------------------------------------------===//
// Channel and CommChannel struct
//===----------------------------------------------------------------------===//

RankedTensorType getTensorType(Type loadType);

bool enclosing(scf::IfOp ifOp, Operation *op);

bool enclosing(scf::ForOp forOp, Operation *op);

//===----------------------------------------------------------------------===//
// accumulation count related functions
//===----------------------------------------------------------------------===//

// Return number of AccumCnts for the given ctrlOp. Add a single
// AccumCnt for all channels under opsWithBufferReuse and it will be the
// last AccumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &regionsWithChannels);

unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &regionsWithChannels);

void appendAccumCntsForOps(SmallVector<Operation *> &taskTopOps,
                           const SmallVector<Channel *> &channels,
                           DenseSet<Operation *> &regionsWithChannels);

std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers);

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase);

//===----------------------------------------------------------------------===//
// Insert Async pipeline ops
//===----------------------------------------------------------------------===//

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx);

Operation *optimizeLoadOps(OpBuilderWithAsyncTaskIds &builder,
                           SmallVector<tt::LoadOp> &loadOps,
                           SmallVector<Value> &buffers, Value bufferIdx,
                           Value bufferIdxExtract, Operation *headProducer,
                           Operation *headConsumer);

//===----------------------------------------------------------------------===//
// Warp specialize
//===----------------------------------------------------------------------===//

void specializeRegion(triton::FuncOp funcOp, int32_t numWarps,
                      unsigned requestedRegisters);

} // namespace mlir::triton::gcu

#endif // TRITON_GCU_DIALECT_TRANSFORMS_WS_CODEPARTITIONUTILITY_H_
