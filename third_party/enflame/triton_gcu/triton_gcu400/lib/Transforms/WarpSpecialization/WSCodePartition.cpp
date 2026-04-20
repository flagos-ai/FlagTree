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
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "CodePartitionUtility.h"
#include "Dialect/GCUWS/IR/Dialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-ws-code-partition"
#define NUM_STAGE
#include "AttrName.h"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgcu = mlir::triton::gcu;
namespace ttgcuws = mlir::triton::gcuws;

namespace mlir::triton::gcu {

static unsigned getNumBuffersOrDefault(scf::ForOp forOp, unsigned numBuffers) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(tt::kNumStagesAttrName))
    return numBuffers;
  return mlir::cast<IntegerAttr>(forOp->getAttr(tt::kNumStagesAttrName))
      .getInt();
}

// Find transitive users of the root op. Track through control flow ops (such as
// yield) to get to the real users.
void getTransitiveUsers(Value root,
                        SetVector<std::pair<Operation *, unsigned>> &users) {
  for (Operation *userOp : root.getUsers()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(userOp)) {
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == root) {
          auto result =
              yieldOp->getParentOp()->getResult(operand.getOperandNumber());
          getTransitiveUsers(result, users);
        }
      }
    } else {
      // find operand index of root
      unsigned operandIndex = 0;
      for (OpOperand &operand : userOp->getOpOperands()) {
        if (operand.get() == root) {
          break;
        }
        operandIndex++;
      }
      assert(operandIndex < userOp->getNumOperands() &&
             "root is not an operand of userOp");
      users.insert({userOp, operandIndex});
    }
  }
}

// ProducerOp can be either the defining op of operand A or the accumulator.
static void createChannel(Operation *producerOp, Operation *op,
                          mlir::DominanceInfo &dom,
                          SmallVector<std::unique_ptr<Channel>> &channels,
                          unsigned producerNumBuffers, unsigned numWarps) {
  // For TMEM channels, op is Gen5 op, producerOp can be either A operand
  // or accumulator.
  auto producerTaskIds = getAsyncTaskIds(op);
  auto producerTaskId = producerTaskIds.front();
  for (auto result : producerOp->getResults()) {
    if (result.use_empty()) {
      continue;
    }

    SetVector<std::pair<Operation *, unsigned>> users;
    getTransitiveUsers(result, users);
    for (auto user : users) {
      auto userOp = user.first;
      if (op == userOp)
        continue;
      // rule out users that are not dominated by op
      if (op->getBlock() != userOp->getBlock()) {
        if (!dom.properlyDominates(op->getParentOp(), userOp)) {
          continue;
        }
      } else {
        if (!dom.properlyDominates(op, userOp) && op != userOp)
          continue;
      }

      auto consumerTaskIds = getAsyncTaskIds(userOp);
      if (consumerTaskIds.empty())
        continue;
      // Remove producer task id from consumerTaskIds.
      auto iter = std::remove(consumerTaskIds.begin(), consumerTaskIds.end(),
                              producerTaskId);
      consumerTaskIds.erase(iter, consumerTaskIds.end());

      // Add a channel from the single producer task to consumerTaskIds.
      if (consumerTaskIds.size() > 0) {
        channels.push_back(std::make_unique<Channel>(
            producerTaskId, consumerTaskIds, userOp, user.second,
            producerNumBuffers, numWarps));
      }
    }
  }
}

// Loads will be in producer warp groups. For now, we only allow a single
// warp group/task for a producer. For each LoadOp, create a channel from it
// to any direct user which belongs to a different taskId.
void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp, unsigned numBuffers) {
  mlir::DominanceInfo dom(funcOp);
  funcOp.walk([&](Operation *op) {
    // FIXME: It is possible that a local_alloc can start a channel, when a
    // gemm's operand is in smem and comes from local_alloc.
    if (isa<tt::LoadOp, tt::DescriptorLoadOp>(op) ||
        isa<tt::DotOpInterface>(op)) {
      auto producerTaskIds = getAsyncTaskIds(op);
      if (producerTaskIds.empty() || producerTaskIds.size() > 1) {
        LLVM_DEBUG({
          LDBG() << " ignoring load ops without async task id or with multiple"
                    " task ids: ";
          op->dump();
        });
        return;
      }
      unsigned producerNumBuffers = numBuffers;
      if (auto forOp = op->getParentOfType<scf::ForOp>()) {
        producerNumBuffers = getNumBuffersOrDefault(forOp, numBuffers);
      }
      // producerOp maybe from accumulator, currently not support
      unsigned numWarps = ttg::lookupNumWarps(funcOp);
      createChannel(op, op, dom, channels, producerNumBuffers, numWarps);
    }
  });

  LLVM_DEBUG({
    LDBG() << "Async channels:";
    for (auto &channel : channels) {
      LDBG() << "producer op: " << channel->relation.first;
      channel->getSrcOp()->dump();
      for (auto &asyncTaskId : channel->relation.second)
        LDBG() << "consumer: " << asyncTaskId;
      channel->getDstOp()->dump();
      LDBG() << "numBuffers: " << channel->numBuffers;
      LDBG() << "numWarps: " << channel->numWarps;
    }
  });
}

// When the consumer is a local_alloc loading from shared memory to registers,
// look ahead for the actual consumers, usually dot ops, that can directly use
// shared memory. The local_alloc will be removed later.
static SmallVector<Operation *> getActualConsumers(Operation *consumerOp) {
  if (isa<ttg::LocalAllocOp>(consumerOp)) {
    DenseSet<Operation *> users;
    for (auto user : consumerOp->getUsers()) {
      if (isa<tt::TransOp, ttg::MemDescTransOp>(user)) {
        // TransOp is not a real consumer. It calculates the shared memory
        // address for the real consumer. Continue to find its transitive users
        // recursively.
        DenseSet<Operation *> visited;
        SmallVector<Operation *> transUsers;
        transUsers.push_back(user);
        while (!transUsers.empty()) {
          auto transUser = transUsers.pop_back_val();
          visited.insert(transUser);
          if (isa<tt::TransOp, ttg::MemDescTransOp>(transUser)) {
            for (auto transitiveUser : transUser->getUsers()) {
              if (!visited.count(transitiveUser))
                transUsers.push_back(transitiveUser);
            }
          } else {
            users.insert(transUser);
          }
        }
      } else {
        users.insert(user);
      }
    }

    return SmallVector<Operation *>(users.begin(), users.end());
  }
  return {consumerOp};
}

static Operation *getUniqueActualConsumer(Operation *consumerOp) {
  auto consumers = getActualConsumers(consumerOp);
  return consumers.size() == 1 ? consumers[0] : consumerOp;
}

// Group channels in two ways:
//  - by producer ops. One producer corresponds to multiple channels. This
//    grouping will be used to create buffers per shared producer.
//  - by consumer ops. One consumer corresponds to multiple channels. This
//  grouping will be used to create barriers per shared consumer.
// Also compute orderedChannels, which will be keyed by getDstOp() of
// channels, to enforce deterministic order for map.
void groupChannels(
    SmallVector<Channel *> &channels,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByConsumers,
    SmallVector<Channel *> &orderedChannels) {
  // Group channels by producer op.
  DenseMap<Operation *, SmallVector<Channel *>> producerChannels;
  for (auto channel : channels) {
    producerChannels[channel->getSrcOp()].push_back(channel);
  }

#ifndef NDEBUG
  // Some sanity checks.
  for (auto &item : producerChannels) {
    auto &channels = item.second;
    unsigned numBuffers = channels.front()->numBuffers;
    for (auto c : channels) {
      assert(c->numBuffers == numBuffers && "Unmatched number of buffers");
    }
  }
#endif

  // Group channels by consumer op.
  DenseMap<Operation *, SmallVector<Channel *>> consumerChannels;

  // Two channels can be combined if
  //   src1 and src2 are in the same block and
  //   (dst1 == dst2 or
  //    (dst1 and dst2 are in the same block, both have a single user, and
  //     dst1User == dst2User and dst1User is in the same block as dst1))
  auto channelCanBeMerged = [](Channel *c1, Channel *c2) -> bool {
    if (c1->getSrcOp()->getBlock() != c2->getSrcOp()->getBlock())
      return false;
    Operation *dst1 = c1->getDstOp(), *dst2 = c2->getDstOp();
    if (dst1 == dst2)
      return true;
    // We only have one CommChannel for channels in channelsGroupedByConsumers.
    // A CommChannel can have multiple tokens, one for each consumer taskId.
    // Consider the case where channel v is between producer task 0 and
    // consumer task 1, while channel p is between producer task 2 and
    // consumer task 1, but in AllocBarrier, we only consider the first
    // channel in the group.
    if (getAsyncTaskIds(c1->getSrcOp()) != getAsyncTaskIds(c2->getSrcOp()))
      return false;
    // Check taskIds on dstOps.
    if (getAsyncTaskIds(dst1) != getAsyncTaskIds(dst2))
      return false;
    auto dst1User = getUniqueActualConsumer(dst1);
    auto dst2User = getUniqueActualConsumer(dst2);
    if (!dst1User || !dst2User)
      return false;
    return dst1User == dst2User && dst1User->getBlock() == dst1->getBlock();
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in the consumerChannels to see if
  // it can be combined.
  for (auto *channel : channels) {
    bool merged = false;
    for (auto &kv : consumerChannels) {
      if (kv.second.size() > 0 &&
          channelCanBeMerged(channel, kv.second.front())) {
        kv.second.push_back(channel);
        merged = true;
        break;
      }
    }
    if (!merged) { // Create a new entry.
      auto *keyOp = channel->getDstOp();
      if (!consumerChannels.count(keyOp))
        orderedChannels.push_back(channel);
      consumerChannels[keyOp].push_back(channel);
    }
  }

  // Reorder channels associated with one entry based on program order of the
  // producers.
  for (auto &group : make_second_range(consumerChannels)) {
    auto &allOps = group.front()->getSrcOp()->getBlock()->getOperations();
    DenseMap<Operation *, size_t> opIdx;
    opIdx.reserve(allOps.size());
    for (auto [idx, op] : enumerate(allOps)) {
      opIdx[&op] = idx;
    }
    sort(group, [&](Channel *a, Channel *b) {
      return opIdx[a->getSrcOp()] < opIdx[b->getSrcOp()];
    });
  }

  // Switch to using channel as the key instead of ops as ops can be
  for (auto &kv : producerChannels) {
    channelsGroupedByProducers[kv.second.front()] = kv.second;
  }
  for (auto &kv : consumerChannels) {
    channelsGroupedByConsumers[kv.second.front()] = kv.second;
  }

  LLVM_DEBUG({
    LDBG() << "Grouped channels by producer:";
    unsigned i = 0;
    for (auto &kv : channelsGroupedByProducers) {
      LDBG() << "Channel " << i++;
      LDBG() << "producer:  " << *kv.getFirst()->getSrcOp();
      for (auto &channel : kv.second) {
        LDBG() << "consumer: " << *channel->getDstOp();
        LDBG() << "numBuffers: " << channel->numBuffers;
      }
    }

    LDBG() << "Grouped channels by consumer:";
    i = 0;
    for (auto &kv : channelsGroupedByConsumers) {
      LDBG() << "Channel " << i++;
      LDBG() << "consumer: " << *kv.getFirst()->getDstOp();
      for (auto &channel : kv.second) {
        LDBG() << "producer: " << *channel->getSrcOp();
        for (auto &asyncTaskId : channel->relation.second)
          LDBG() << "consumer asyncTaskId: " << asyncTaskId << ", ";
        LDBG() << "numBuffers: " << channel->numBuffers;
      }
    }
  });
}

// Find top-level ops which contain at least one channel. If a channel's
// getSrcOp() and getDstOp() belong to the inner loop, the outer loop will be
// part of asyncTaskOps.
SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp,
                 const SmallVector<Channel *> &channels) {
  SmallVector<Operation *> asyncTaskOps;
  auto isAsyncTaskTopOp = [&](Operation *taskTopOp) -> bool {
    for (auto c : channels) {
      Operation *producer = c->getSrcOp();
      Operation *consumer = c->getDstOp();
      while (producer && !isa<triton::FuncOp>(producer->getParentOp())) {
        producer = producer->getParentOp();
      }
      while (consumer && !isa<triton::FuncOp>(consumer->getParentOp())) {
        consumer = consumer->getParentOp();
      }
      if (producer == taskTopOp && consumer == taskTopOp)
        return true;
    }
    return false;
  };
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (op->getNumRegions() <= 0)
        continue;
      // If this op does not contain both a producer taskId and a consumer
      // taskId, continue.
      if (getAsyncTaskIds(op).size() == 1)
        continue;
      if (isAsyncTaskTopOp(op))
        asyncTaskOps.push_back(op);
    }
  }

  LLVM_DEBUG({
    LDBG() << "=============== Step3.1 getTaskTopRegion ===============";
    LDBG() << "Top Task Bodies";
    for (auto op : asyncTaskOps) {
      LDBG() << "Task Body:";
      // op->dump();
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    }
  });
  return asyncTaskOps;
}

// Here we assume the source and destination ops are in the same region op.
// Go through channels, and get a set of region ops containing channels.
void collectRegionsWithChannels(const SmallVector<Channel *> &channels,
                                DenseSet<Operation *> &regionsWithChannels) {
  for (auto *channel : channels) {
    if (auto *pOp = channel->getDstOp()->getParentOp()) {
      if (isa<scf::ForOp, scf::IfOp>(pOp))
        regionsWithChannels.insert(pOp);
    }
  }
}

// Create an allocation to hold the mbarriers.
static Value createBarrierInit(triton::FuncOp funcOp, unsigned numWarps) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  return builder.create<ttgcu::InitBarrierOp>(funcOp->getLoc(), numWarps);
}

// channelsGroupedByConsumers: channels are grouped together.
// Go through each group, check the first channel in the group, create a token
// for each consumer taskId. Return a map that maps each channel + consumer
// taskId to a token. Also update barrierAllocMap that maps each channel +
// consumer taskId to a BarrierAlloc.
void AllocBarrier(const DenseMap<Channel *, SmallVector<Channel *>>
                      &channelsGroupedByConsumers,
                  const SmallVector<Channel *> &orderedChannels,
                  triton::FuncOp funcOp,
                  DenseMap<Channel *, CommChannel> &barrierMap) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (auto *key : orderedChannels) {
    auto it = channelsGroupedByConsumers.find(key);
    LLVM_DEBUG({
      LDBG() << "AllocBarrier key: consumer: ";
      key->getDstOp()->dump();

      LDBG() << "AllocBarrier channelsGroupedByConsumers:";
      for (auto map_key : make_first_range(channelsGroupedByConsumers)) {
        LDBG() << "representative consumer: ";
        map_key->getDstOp()->dump();
      }
    });
    assert(it != channelsGroupedByConsumers.end());
    Channel *channel = it->second.front();

    CommChannel commChannel;
    // auto producerOp = it->second.front()->getSrcOp();
    // auto dstOp = it->second.front()->getDstOp();

    // if (isa<tt::DescriptorLoadOp>(producerOp)) {
    //   commChannel.producerBarrier =
    //       createBarrierInit(funcOp, channel->numWarps);
    // }

    for (auto consumerAsyncTaskId : channel->relation.second) {
      commChannel.consumerBarriers[consumerAsyncTaskId] =
          createBarrierInit(funcOp, channel->numWarps);
    }

    // Channels in the group share the same set of tokens.
    for (auto &c : it->second) {
      barrierMap[c] = commChannel;
    }
  }

  LLVM_DEBUG({
    LDBG() << "Communication Channels:";
    for (auto &item : barrierMap) {
      llvm::dbgs() << "\ndata channel: \n";
      llvm::dbgs() << *item.first->getSrcOp() << "\n";
      llvm::dbgs() << *item.first->getDstOp() << "\n";
      llvm::dbgs() << "communication channel: \n";
      for (auto &kv : item.second.tokens) {
        llvm::dbgs() << "token: " << kv.first << " " << kv.second << "\n";
      }
      if (item.second.producerBarrier)
        llvm::dbgs() << "producer barrier: " << *item.second.producerBarrier
                     << "\n";
      for (auto &kv : item.second.consumerBarriers)
        llvm::dbgs() << "consumer barrier: " << kv.first << " " << kv.second
                     << "\n";
    }
  });
}

// Create a buffer array for each producer op, if the producer is in a ForOp,
// the buffer array will contain numBuffers.
DenseMap<Channel *, Value> createBuffer(
    DenseMap<Channel *, SmallVector<Channel *>> &channelsGroupedByProducers,
    const SmallVector<Channel *> &orderedChannels, triton::FuncOp funcOp) {
  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  DenseSet<Channel *> visited;
  for (auto &item : channelsGroupedByProducers) {
    auto &channels = item.second;
    for (auto c : channels) {
      assert(!visited.count(c));
      visited.insert(c);
    }
  }
  for (auto *channelInOrder : orderedChannels) {
    if (channelsGroupedByProducers.find(channelInOrder) ==
        channelsGroupedByProducers.end())
      continue;
    auto &channels = channelsGroupedByProducers[channelInOrder];
    auto srcValue = channelInOrder->getSrcOperand();
    auto srcOp = channelInOrder->getSrcOp();
    auto dstOp = channelInOrder->getDstOp();
    auto *channel = channels.front();
    unsigned numBuffers = channel->numBuffers;
    Value buffer;

    LLVM_DEBUG({
      LDBG() << "Creating buffers for channel:";
      LDBG() << "Producer:" << *srcOp;
      LDBG() << "Consumer:" << *dstOp;
    });

    auto srcType = srcValue.getType();
    auto tensorType = dyn_cast<RankedTensorType>(srcType);
    if (auto ptrType = dyn_cast<tt::PointerType>(srcType)) {
      tensorType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    }
    if (!tensorType) {
      llvm_unreachable("Unexpected result type");
    }

    // Get basic information from tensorType
    auto order = ttg::getOrderForMemory(tensorType);
    auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
    auto elemType = tensorType.getElementType();

    // Get shape, layout and type of a slice
    auto sliceShape = tensorType.getShape();

    // Check the consumer type
    auto actualConsumers = getActualConsumers(dstOp);
    LLVM_DEBUG({
      LDBG() << "actual consumers:";
      for (auto consumerOp : actualConsumers) {
        LDBG() << *consumerOp;
      }
    });

    bool requireMMASharedEncoding =
        llvm::any_of(actualConsumers,
                     [](Operation *op) { return isa<tt::DotOpInterface>(op); });
    assert(requireMMASharedEncoding && "must be DotOpInterface");

    Attribute sharedLayout;
    if (requireMMASharedEncoding) {
      sharedLayout = ttg::NVMMASharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType,
          /*fp4Padded*/ false);
    } else {
      // Create an unswizzled layout for now.
      sharedLayout = ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1,
                                                          order, CTALayout);
    }

    // Get shape, layout and type of the complete buffer
    SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
    if (srcOp->getParentOfType<scf::ForOp>())
      bufferShape.insert(bufferShape.begin(), numBuffers);
    else
      bufferShape.insert(bufferShape.begin(), 1);
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    Type memdescType =
        ttg::MemDescType::get(bufferShape, elemType, sharedLayout,
                              sharedMemorySpace, /*mutableMemory*/ true);
    buffer = builder.create<ttg::LocalAllocOp>(funcOp.getLoc(), memdescType);

    // Channels in the group share the same buffer.
    for (auto c : channels)
      bufferMap[c] = buffer;
  }
  return bufferMap;
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByProducers"
void collectPipelineChannels(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByProducers,
    DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap) {
  // For each producer op, create a async_copy or local_store from the producer
  // to the buffer. Create a local_load from the buffer at the dominating
  // consumer.
  mlir::DominanceInfo dom(funcOp);

  for (auto kv : channelsGroupedByProducers) {
    // Finding the dominating channel if possible.
    std::unordered_set<Channel *> mutuallyNonDominatingChannels;
    for (auto &c : kv.second) {
      // check if c is dominating all other previous channels.
      auto it = mutuallyNonDominatingChannels.begin();
      while (it != mutuallyNonDominatingChannels.end()) {
        auto channel = *it;
        if (dom.properlyDominates(c->getDstOp(), channel->getDstOp())) {
          it = mutuallyNonDominatingChannels.erase(it);
        } else if (dom.properlyDominates(channel->getDstOp(), c->getDstOp())) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingChannels.end())
        mutuallyNonDominatingChannels.insert(c);
    }

    assert(mutuallyNonDominatingChannels.size() == 1 &&
           "conditional consumers not supported");
    auto domininatingChannel = *mutuallyNonDominatingChannels.begin();
    auto srcOp = kv.getFirst()->getSrcOp();
    LLVM_DEBUG({
      LDBG() << "collectPipelineChannels handle channel ";
      srcOp->dump();
      domininatingChannel->getDstOp()->dump();
    });

    for (auto &channel : kv.second) {
      copyOpMap[channel] = std::pair<Operation *, Operation *>{
          srcOp, domininatingChannel->getDstOp()};
    }
  }
}

// Lower producers for channels. Here channels are grouped in
// "channelsGroupedByConsumers". barrierMap tracks the set of tokens for each
// channel.
void insertAsyncComm(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByConsumers,
    const DenseMap<Channel *, CommChannel> &barrierMap,
    const DenseMap<Channel *, Value> &bufferMap,
    const DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &regionsWithChannels, bool innerBarrier) {
  // Find the operation that is along producer's parent chain, and its parent
  // is the same op as producer's parent. Here p is producer, and c is consumer.
  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    Operation *op = c;
    while (!isa<triton::FuncOp>(op)) {
      if (op->getParentOp() == p->getParentOp()) {
        return op;
      }
      op = op->getParentOp();
    }
    op = p;
    while (!isa<triton::FuncOp>(op)) {
      if (c->getParentOp() == op->getParentOp()) {
        return c;
      }
      op = op->getParentOp();
    }
    llvm_unreachable("Failed to find consumer's same level Op with producer");
  };

  mlir::PostDominanceInfo pdom(funcOp);
  auto consumerReleaseHeuristic = [&](Operation *p, Operation *c,
                                      int consumerAsyncTaskId) -> Operation * {
    if (c->getBlock() != p->getBlock())
      return getSameLevelOp(p, c);

    // Find a common place for all users of the consumer, which would be the
    // common post dominator.
    auto actualConsumers = getActualConsumers(c);
    std::unordered_set<Operation *> mutuallyNonDominatingUsers;
    for (auto user : actualConsumers) {
      auto it = mutuallyNonDominatingUsers.begin();
      while (it != mutuallyNonDominatingUsers.end()) {
        if (pdom.properlyPostDominates(user, *it)) {
          it = mutuallyNonDominatingUsers.erase(it);
        } else if (pdom.properlyPostDominates(*it, user)) {
          break;
        } else {
          ++it;
        }
      }
      if (it == mutuallyNonDominatingUsers.end())
        mutuallyNonDominatingUsers.insert(user);
    }

    if (mutuallyNonDominatingUsers.size() == 1) {
      // Find the common parent of this user and c
      auto user = *mutuallyNonDominatingUsers.begin();
      while (user && user->getParentOp() != c->getParentOp())
        user = user->getParentOp();
      assert(user && "Failed to find common parent of this user and c");
      return user;
    }

    for (auto &op : reverse(c->getBlock()->getOperations())) {
      auto asyncTasks = getAsyncTaskIds(&op);
      if (asyncTasks.size() == 1 && asyncTasks[0] == consumerAsyncTaskId)
        return &op;
    }

    return nullptr;
  };

  // Go through each channel group.
  for (auto kv : channelsGroupedByConsumers) {
    // Find head and tail ops.
    DenseSet<Operation *> producerOps;
    DenseSet<Operation *> consumerOps;
    for (auto &c : kv.second) {
      auto pcOp = copyOpMap.find(c)->second;
      producerOps.insert(pcOp.first);
      consumerOps.insert(pcOp.second);
      consumerOps.insert(c->getDstOp());
      consumerOps.insert(getUniqueActualConsumer(c->getDstOp()));
    }

    // Find head producer
    auto producerBlock = kv.second.front()->getSrcOp()->getBlock();
    Operation *headProducer = nullptr;
    for (auto &op : producerBlock->getOperations()) {
      if (producerOps.count(&op)) {
        headProducer = &op;
        break;
      }
    }

    // Find tail producer
    Operation *tailProducer = nullptr;
    for (auto &op : reverse(producerBlock->getOperations())) {
      if (producerOps.count(&op)) {
        tailProducer = &op;
        break;
      }
    }

    // Find head consumer
    auto consumerBlock = kv.second.front()->getDstOp()->getBlock();
    Operation *headConsumer = nullptr;
    for (auto &op : consumerBlock->getOperations()) {
      if (consumerOps.count(&op)) {
        headConsumer = &op;
        break;
      }
    }

    // Find tail consumer
    Operation *tailConsumer = nullptr;
    for (auto &op : reverse(consumerBlock->getOperations())) {
      if (consumerOps.count(&op)) {
        tailConsumer = &op;
        break;
      }
    }

    // We have one set of tokens for each channel group.
    auto masterChannel = kv.first;

    SmallVector<AsyncTaskId> asyncTaskP;
    asyncTaskP.push_back(masterChannel->relation.first);
    SmallVector<AsyncTaskId> &asyncTaskC = masterChannel->relation.second;
    SmallVector<AsyncTaskId> asyncTasksPC = asyncTaskP;
    asyncTasksPC.insert(asyncTasksPC.end(), asyncTaskC.begin(),
                        asyncTaskC.end());

    OpBuilderWithAsyncTaskIds builder(headProducer->getContext());
    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAsynTaskIdsFromArray(asyncTasksPC);

    Value bufferIdx;
    Value phase = Value();
    if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      builder.setInsertionPoint(headProducer);
      LLVM_DEBUG({
        LDBG() << "call getBufferIdxAndPhase2 ";
        headProducer->dump();
      });
      getBufferIdxAndPhase(builder, headProducer, kv.second.front()->numBuffers,
                           regionsWithChannels, bufferIdx, phase);
    } else {
      // Producer is not in a ForOp, create phase and bufferIdx here.
      bufferIdx = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
      phase = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 1);
    }

    LLVM_DEBUG({
      LDBG() << "SrcOp of master Channel ";
      masterChannel->getSrcOp()->dump();
      LDBG() << "DstOp of master Channel ";
      masterChannel->getDstOp()->dump();
      LDBG() << "headProducer ";
      headProducer->dump();
      LDBG() << "tailProducer ";
      tailProducer->dump();
      LDBG() << "headConsumer ";
      headConsumer->dump();
      LDBG() << "tailConsumer ";
      tailConsumer->dump();
    });

    // Init producer consumer pipeline
    builder.clearAsynTaskIds();
    auto numStages = masterChannel->numBuffers;
    auto producerCount = 1; // currently only support single producer
    auto consumerCount =
        masterChannel->numWarps * masterChannel->relation.second.size();
    builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    auto pipelineType = ttgcuws::PipelineType::get(
        headProducer->getContext(), numStages, producerCount, consumerCount,
        /*inner_barrier=*/innerBarrier);
    auto pipeline = builder.createWithAsyncTaskIds<ttgcuws::InitPipelineOp>(
        funcOp->getLoc(), pipelineType);

    // Insert ProducerAcquireOp before the producer.
    builder.setAsynTaskIdsFromArray(masterChannel->relation.first);
    auto producerAcquirePoint = getSameLevelOp(headConsumer, headProducer);
    builder.setInsertionPoint(producerAcquirePoint);
    builder.createWithAsyncTaskIds<ttgcuws::ProducerAcquireOp>(
        headProducer->getLoc(), pipeline);

    // Insert ProducerCommitOp
    Operation *producerCommitPoint = getSameLevelOp(headConsumer, tailProducer);
    builder.setInsertionPointAfter(producerCommitPoint);
    builder.createWithAsyncTaskIds<ttgcuws::ProducerCommitOp>(
        tailProducer->getLoc(), pipeline);

    // Insert ConsumerWaitOp/ConsumerReleaseOp
    for (auto consumerTaskId : masterChannel->relation.second) {
      // Insert ConsumerWaitOp
      builder.setAsynTaskIdsFromArray(consumerTaskId);
      auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
      builder.setInsertionPoint(consumerWaitPoint);
      builder.createWithAsyncTaskIds<ttgcuws::ConsumerWaitOp>(
          headConsumer->getLoc(), pipeline);

      // Insert ConsumerReleaseOp
      auto consumerReleasePoint =
          consumerReleaseHeuristic(tailProducer, tailConsumer, consumerTaskId);
      builder.setInsertionPointAfter(consumerReleasePoint);
      auto consumerReleaseOp =
          builder.createWithAsyncTaskIds<ttgcuws::ConsumerReleaseOp>(
              consumerReleasePoint->getLoc(), pipeline);

      // Insert ArriveBarrierOp/WaitBarrierOp
      if (!innerBarrier) {
        auto &commChannel = barrierMap.find(kv.second.front())->second;
        auto iter = commChannel.consumerBarriers.find(consumerTaskId);
        if (iter != commChannel.consumerBarriers.end()) {
          builder.setInsertionPoint(consumerWaitPoint);
          auto barrier = iter->second;
          builder.createWithAsyncTaskIds<ttgcu::ArriveBarrierOp>(
              headConsumer->getLoc(), barrier);

          builder.setInsertionPointAfter(consumerReleaseOp);
          builder.createWithAsyncTaskIds<ttgcu::WaitBarrierOp>(
              consumerReleaseOp->getLoc(), barrier);
        }
      }
    }

    // Optimize load ops.
    SmallVector<tt::LoadOp> loadOps;
    SmallVector<Value> buffers;
    for (auto &c : kv.second) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(c->getSrcOp())) {
        loadOps.push_back(loadOp);
        buffers.push_back(bufferMap.find(c)->second);
      }
    }

    optimizeLoadOps(builder, loadOps, buffers, bufferIdx, bufferIdx,
                    headProducer, headConsumer);
  }
}

void foldLocalLoads(triton::FuncOp funcOp) {
  // If loadResult has a single use which is LocalAlloc, we can get rid of
  // sharedLoad and replace all uses of LocalAlloc with viewLoad.
  DenseMap<Operation *, Value> opsToReplace;
  funcOp.walk([&](ttg::LocalAllocOp localAlloc) {
    if (auto src = localAlloc.getSrc()) {
      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(src.getDefiningOp())) {
        // Only fold within the same task
        if (getAsyncTaskIds(localLoad) == getAsyncTaskIds(localAlloc)) {
          opsToReplace[localAlloc] = localLoad.getSrc();
        }
      }
    }
  });
  OpBuilderWithAsyncTaskIds builder(funcOp.getContext());
  for (auto kv : opsToReplace)
    tt::replaceUsesAndPropagateType(builder, kv.getFirst(), kv.getSecond());
}

void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers,
                     unsigned numWarps, bool innerBarrier) {
  // Step 1: collect all communications between producers and consumers.
  LDBG() << "=============== Step1. collectAsyncChannels ===============";
  SmallVector<std::unique_ptr<Channel>> channelsOrigin;
  collectAsyncChannels(channelsOrigin, funcOp, numBuffers);
  SmallVector<Channel *> channels;
  for (const auto &c : channelsOrigin) {
    channels.push_back(c.get());
  }
  if (channels.empty()) {
    return;
  }

  // Step 2: group channels
  // -  each entry of the channelsGroupedByProducers is keyed by the srcOp.
  // -  each entry of the channelsGroupedByConsumers is keyed by the dstOp.
  LDBG() << "=============== Step2. groupChannels ===============";
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByProducers;
  DenseMap<Channel *, SmallVector<Channel *>> channelsGroupedByConsumers;
  SmallVector<Channel *> orderedChannels;
  groupChannels(channels, channelsGroupedByProducers,
                channelsGroupedByConsumers, orderedChannels);

  // Step 3: find top-level ops that contain a channel, also create new ForOps
  // by adding phase and bufferIdx to the original ForOps, erase the original
  // ForOps.
  LDBG() << "=============== Step3. appendAccumCntsForOps ===============";
  SmallVector<Operation *> asyncTaskTopOps = getTaskTopRegion(funcOp, channels);
  DenseSet<Operation *> regionsWithChannels;
  collectRegionsWithChannels(channels, regionsWithChannels);
  appendAccumCntsForOps(asyncTaskTopOps, channels, regionsWithChannels);
  LLVM_DEBUG({
    LDBG() << "\nafter appendAccumCntsForOps";
    funcOp.dump();
  });

  // Step 4: Create buffers. An array of buffers for each channel.
  LDBG() << "=============== Step4. createBuffer ===============";
  DenseMap<Channel *, Value> bufferMap =
      createBuffer(channelsGroupedByProducers, channels, funcOp);
  LLVM_DEBUG({
    LDBG() << "\nafter createBuffer";
    funcOp.dump();
  });

  // Step 5: Collect producer and consumer pipeline ops for each channel in
  // producer groups.
  LDBG() << "=============== Step5. collectPipelineChannels ===============";
  DenseMap<Channel *, std::pair<Operation *, Operation *>> copyOpMap;
  collectPipelineChannels(funcOp, channelsGroupedByProducers, copyOpMap);
  LLVM_DEBUG({
    LDBG() << "\nafter collectPipelineChannels";
    funcOp.dump();
  });

  // Step 6: Alloc barrier for each consumer group.
  DenseMap<Channel *, CommChannel> barrierMap;
  if (!innerBarrier) {
    LDBG() << "=============== Step6. AllocBarrier ===============";
    AllocBarrier(channelsGroupedByConsumers, orderedChannels, funcOp,
                 barrierMap);
    LLVM_DEBUG({
      LDBG() << "\nafter AllocBarrier";
      funcOp.dump();
    });
  }

  // Step 7: add async communication ops (ProducerAcquire etc).
  LDBG() << "=============== Step7. insertAsyncComm ===============";
  insertAsyncComm(funcOp, channelsGroupedByConsumers, barrierMap, bufferMap,
                  copyOpMap, regionsWithChannels, innerBarrier);
  LLVM_DEBUG({
    LDBG() << "\nwith SyncOps";
    funcOp.dump();
  });

  // Step 8: If loadResult has a single use which is LocalAlloc, we can get rid
  // of sharedLoad and replace all uses of LocalAlloc with viewLoad.
  LDBG() << "=============== Step8. foldLocalLoads ===============";
  foldLocalLoads(funcOp);
  LLVM_DEBUG({
    LDBG() << "\nsimplify localLoad + localAlloc";
    funcOp.dump();
  });

  // Step 9: specializeRegion
  LDBG() << "=============== Step9. specializeRegion ===============";
  specializeRegion(funcOp, numWarps, /*requestedRegisters=*/0);
  LLVM_DEBUG({
    LDBG() << "\nwith specializeRegion";
    funcOp.dump();
  });
}

} // namespace mlir::triton::gcu

namespace mlir {
#define GEN_PASS_DEF_GCUTESTWSCODEPARTITION
#include "Transforms/Passes.h.inc"
} // namespace mlir

namespace {

class GCUTestWSCodePartitionPass
    : public impl::GCUTestWSCodePartitionBase<GCUTestWSCodePartitionPass> {
public:
  using impl::GCUTestWSCodePartitionBase<
      GCUTestWSCodePartitionPass>::GCUTestWSCodePartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    // Disable code partitioning when numBuffers is 0.
    if (numBuffers > 0 && numWarps > 0)
      ttgcu::doCodePartition(funcOp, numBuffers, numWarps, innerBarrier);
  }
  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG() << "post pass";
      getOperation()->dump();
    });
    return;
  }
};

} // namespace
