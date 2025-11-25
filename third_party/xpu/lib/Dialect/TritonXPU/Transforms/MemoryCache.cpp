//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-lm-cache"
#include <iostream>

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUMEMORYCACHE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUMemoryCache
    : public impl::TritonXPUMemoryCacheBase<TritonXPUMemoryCache> {

public:
  using impl::TritonXPUMemoryCacheBase<
      TritonXPUMemoryCache>::TritonXPUMemoryCacheBase;

  TritonXPUMemoryCache() = default;
  TritonXPUMemoryCache(unsigned bufferSize, unsigned coreNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
  }

  void memoryCse(ModuleOp &m) {
    // 1. Get Op Line
    DenseMap<Operation *, unsigned> op2Line;
    getOpLine(m, op2Line);

    // 2. Get Common LoadOps
    DenseMap<Operation *, SetVector<Operation *>> memoryMap;
    m.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto parentOp = gm2lmOp->getParentOp();
      memoryMap[parentOp].insert(gm2lmOp);
    });
    m.walk([&](triton::xpu::GM2LMMaskOp gm2lmOp) {
      auto parentOp = gm2lmOp->getParentOp();
      memoryMap[parentOp].insert(gm2lmOp);
    });

    // 3. Eliminate Common LoadOps, But Excluding Inplace Case
    SetVector<Operation *> erasedOps;
    for (const auto &pair : memoryMap) {
      auto op = pair.first;
      auto loadOps = pair.second;
      if (loadOps.size() > 1) {
        for (int i = 0; i < loadOps.size() - 1; ++i) {
          for (int j = i + 1; j < loadOps.size(); ++j) {
            if (!erasedOps.count(loadOps[i]) && !erasedOps.count(loadOps[j])) {
              if (auto load0 = dyn_cast<triton::xpu::GM2LMOp>(loadOps[i])) {
                auto ptr1 = load0.getPtr();
                auto ptr2 = cast<triton::xpu::GM2LMOp>(loadOps[j]).getPtr();
                bool noInplace = true;
                for (auto user : ptr1.getUsers()) {
                  if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
                    if (op2Line[loadOps[i]] < op2Line[storeOp] &&
                        op2Line[storeOp] < op2Line[loadOps[j]]) {
                      noInplace &= false;
                    }
                  }
                }
                if (noInplace && ptr1 == ptr2) {
                  loadOps[j]->getResults().replaceAllUsesWith(
                      loadOps[i]->getResults());
                  erasedOps.insert(loadOps[j]);
                }
              } else if (auto load0 =
                             dyn_cast<triton::xpu::GM2LMMaskOp>(loadOps[i])) {
                auto ptr1 = load0.getPtr();
                auto ptr2 = cast<triton::xpu::GM2LMMaskOp>(loadOps[j]).getPtr();
                bool noInplace = true;
                for (auto user : ptr1.getUsers()) {
                  if (auto storeOp = dyn_cast<triton::xpu::LM2GMMaskOp>(user)) {
                    if (op2Line[loadOps[i]] < op2Line[storeOp] &&
                        op2Line[storeOp] < op2Line[loadOps[j]]) {
                      noInplace &= false;
                    }
                  }
                }
                if (noInplace && ptr1 == ptr2) {
                  loadOps[j]->getResults().replaceAllUsesWith(
                      loadOps[i]->getResults());
                  erasedOps.insert(loadOps[j]);
                }
              }
            }
          }
        }
      }
      for (auto op : erasedOps) {
        if (op->use_empty()) {
          op->erase();
        }
      }
    }
  }

  void lmCache(ModuleOp &m) {
    SetVector<Operation *> broadCastGM2LMOps;
    m.walk([&](triton::xpu::BroadcastOp broadOp) {
      auto src = broadOp.getSrc();
      auto res = broadOp.getResult();
      auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
      auto resShape = cast<RankedTensorType>(res.getType()).getShape();
      if ((srcShape.front() == 1) && (srcShape.front() != resShape.front())) {
        if (auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(src)) {
          broadCastGM2LMOps.insert(gm2lmOp);
        } else if (auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMMaskOp>(src)) {
          broadCastGM2LMOps.insert(gm2lmOp);
        }
      }
    });

    for (auto op : broadCastGM2LMOps) {
      if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(op)) {
        OpBuilder builder(gm2lmOp);
        auto loc = gm2lmOp.getLoc();
        auto resTy = gm2lmOp.getResult().getType();
        int64_t resElemNum = 1;
        if (auto resTensorTy = dyn_cast<RankedTensorType>(resTy)) {
          resElemNum = product(resTensorTy.getShape());
        }
        if (resElemNum <= this->bufferSize) {
          // Set Cache Flag for GM2LM
          gm2lmOp->setAttr("cache", builder.getBoolAttr(true));
          LLVM_DEBUG(llvm::dbgs() << "[LM Cache]: Hit LM Cache"
                                  << "\n");
        }
      } else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(op)) {
        OpBuilder builder(gm2lmOp);
        auto loc = gm2lmOp.getLoc();
        auto resTy = gm2lmOp.getResult().getType();
        int64_t resElemNum = 1;
        if (auto resTensorTy = dyn_cast<RankedTensorType>(resTy)) {
          resElemNum = product(resTensorTy.getShape());
        }
        if (resElemNum <= this->bufferSize) {
          // Set Cache Flag for GM2LM
          gm2lmOp->setAttr("cache", builder.getBoolAttr(true));
          LLVM_DEBUG(llvm::dbgs() << "[LM Cache]: Hit LM Cache"
                                  << "\n");
        }
      }
    }
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    memoryCse(m);
    lmCache(m);
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
