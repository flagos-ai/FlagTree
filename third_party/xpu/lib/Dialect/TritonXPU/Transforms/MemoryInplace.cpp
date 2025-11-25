//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include <climits>

#define DEBUG_TYPE "tritonxpu-lm-inplace"
#include <iostream>
#include <map>
#include <queue>
#include <set>

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUMEMORYINPLACE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct MemoryBlock {
  unsigned start;
  unsigned end;
  Operation *op;
  OffsetState offsetState;
  bool cache;
  MemoryBlock(Operation *_op, unsigned _start, unsigned _end,
              OffsetState _offsetState, bool _cache)
      : op(_op), start(_start), end(_end), offsetState(_offsetState),
        cache(_cache) {}
  bool operator<(const MemoryBlock &other) const {
    if (start == other.start)
      return end < other.end;
    return start < other.start;
  }
};

bool compare(const MemoryBlock &a, const MemoryBlock &b) {
  if (a.start == b.start)
    return a.end < b.end;
  return a.start < b.start;
}

struct TritonXPUMemoryInplace
    : public impl::TritonXPUMemoryInplaceBase<TritonXPUMemoryInplace> {

public:
  using impl::TritonXPUMemoryInplaceBase<
      TritonXPUMemoryInplace>::TritonXPUMemoryInplaceBase;

  TritonXPUMemoryInplace() = default;
  TritonXPUMemoryInplace(unsigned bufferSize, unsigned coreNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
  }

  void lmInplace(ModuleOp &m) {
    // 1. Get Alloca Life Time
    DenseMap<Operation *, unsigned> op2Line;
    getOpLine(m, op2Line);
    // Get Alloca Life Time
    std::vector<MemoryBlock> memoryBlocks;
    DenseMap<Operation *, Operation *> allocaMaps;
    m.walk([&](triton::xpu::AllocaOp allocaOp) {
      allocaMaps[allocaOp] = allocaOp;
      unsigned start = op2Line.size();
      unsigned end = 0;
      OffsetState offsetState = OffsetState::Unknown;
      bool cache = false;
      for (auto user : allocaOp->getUsers()) {
        if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
          offsetState = static_cast<OffsetState>(gm2lmOp.getOffsetState());
          cache = gm2lmOp.getCache();
          start = std::min(start, op2Line[gm2lmOp]);
          for (auto gm2lmUser : gm2lmOp->getUsers()) {
            if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(gm2lmUser)) {
              end = std::max(end, op2Line[loadOp]);
            } else {
              if (auto user = findUserOp<triton::xpu::LoadOp>(gm2lmUser)) {
                if (auto userLoadOp = dyn_cast<triton::xpu::LoadOp>(user)) {
                  offsetState = userLoadOp.getIsDiscrete()
                                    ? OffsetState::Discrete
                                    : offsetState;
                  end = std::max(end, op2Line[userLoadOp]);
                }
              }
            }
          }
        } else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(user)) {
          offsetState = static_cast<OffsetState>(gm2lmOp.getOffsetState());
          cache = gm2lmOp.getCache();
          start = std::min(start, op2Line[gm2lmOp]);
          for (auto gm2lmUser : gm2lmOp->getUsers()) {
            if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(gm2lmUser)) {
              end = std::max(end, op2Line[loadOp]);
            } else {
              if (auto user = findUserOp<triton::xpu::LoadOp>(gm2lmUser)) {
                if (auto userLoadOp = dyn_cast<triton::xpu::LoadOp>(user)) {
                  offsetState = userLoadOp.getIsDiscrete()
                                    ? OffsetState::Discrete
                                    : offsetState;
                  end = std::max(end, op2Line[userLoadOp]);
                }
              }
            }
          }
        } else if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(user)) {
          offsetState =
              loadOp.getIsDiscrete() ? OffsetState::Discrete : offsetState;
          end = std::max(end, op2Line[loadOp]);
        } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(user)) {
          start = std::min(start, op2Line[storeOp]);
        } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
          offsetState = static_cast<OffsetState>(lm2gmOp.getOffsetState());
          end = std::max(end, op2Line[lm2gmOp]);
        } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(user)) {
          offsetState = static_cast<OffsetState>(lm2gmOp.getOffsetState());
          end = std::max(end, op2Line[lm2gmOp]);
        } else {
          llvm_unreachable("The User of AllocaOp can only be "
                           "GM2LMOp/LoadOp/StoreOp/LM2GMOp");
        }
      }
      if (start >= 0 && end > start) {
        MemoryBlock memoryBlock(allocaOp, start, end, offsetState, cache);
        memoryBlocks.emplace_back(memoryBlock);
      }
    });
    // Map Blocks to Sizes
    std::map<unsigned, std::vector<MemoryBlock>> sizeBlockMaps;
    DenseMap<unsigned, Operation *> endAllocakMaps;
    for (int i = 0; i < memoryBlocks.size(); ++i) {
      auto memoryBlock = memoryBlocks[i];
      auto allocaOp = cast<triton::xpu::AllocaOp>(memoryBlock.op);
      unsigned tensorSize = getTensorSize(allocaOp.getResult().getType());
      unsigned pointeeBitWidth =
          triton::getPointeeBitWidth(allocaOp.getResult().getType());
      unsigned bitWidthWeight = 100000;
      unsigned cacheWeight = bitWidthWeight * 10;
      unsigned sameWeight = cacheWeight * 10;
      unsigned discreteWeight = sameWeight * 10;
      unsigned memoryBitWidth = pointeeBitWidth * bitWidthWeight + tensorSize;
      // Magic number is only used to distinguish different pointeeBitWidth
      memoryBitWidth +=
          i * cacheWeight *
          memoryBlock
              .cache; // Magic number is only used to distinguish cache or not
      // DiscreteSame/Discrete LM can not be in-place
      if (memoryBlock.offsetState == OffsetState::DiscreteSame) {
        memoryBitWidth +=
            i * sameWeight; // Magic number is only used to distinguish between
                            // different offsetStates
      } else if (memoryBlock.offsetState == OffsetState::Discrete) {
        memoryBitWidth += i * discreteWeight;
      }
      sizeBlockMaps[memoryBitWidth].emplace_back(memoryBlock);
      endAllocakMaps[memoryBlock.end] = memoryBlock.op;
    }

    // 2. Get Min Heap
    for (auto sizeBlockMap : sizeBlockMaps) {
      auto memoryBlocks = sizeBlockMap.second;
      sort(memoryBlocks.begin(), memoryBlocks.end(), compare);
      std::priority_queue<unsigned, std::vector<unsigned>,
                          std::greater<unsigned>>
          minHeap;
      for (auto memoryBlock : memoryBlocks) {
        while (!minHeap.empty() && minHeap.top() <= memoryBlock.start) {
          allocaMaps[endAllocakMaps[memoryBlock.end]] =
              endAllocakMaps[minHeap.top()];
          minHeap.pop();
        }
        minHeap.push(memoryBlock.end);
      }
    }

    // 3. Reuse Alloca
    // Remap to Get Root Alloca
    for (auto allocaMap : allocaMaps) {
      Operation *rootAlloca = allocaMap.second;
      while (allocaMaps[rootAlloca] != rootAlloca) {
        allocaMaps[allocaMap.first] = allocaMaps[rootAlloca];
        rootAlloca = allocaMaps[rootAlloca];
      }
    }
    // Replace All Uses of oriAlloca with reusedAlloca
    for (auto allocaMap : allocaMaps) {
      auto oriAlloca = cast<triton::xpu::AllocaOp>(allocaMap.first);
      auto reusedAlloca = cast<triton::xpu::AllocaOp>(allocaMap.second);
      if (oriAlloca != reusedAlloca) {
        oriAlloca.getResult().replaceAllUsesWith(reusedAlloca.getResult());
      }
    }
    // Erase Use Empty Alloca
    for (auto allocaMap : allocaMaps) {
      auto oriAlloca = allocaMap.first;
      if (oriAlloca->use_empty()) {
        oriAlloca->erase();
      }
    }
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    lmInplace(m);
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
