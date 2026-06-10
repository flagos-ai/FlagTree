#include "epu/memory.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "evas/Transform/Linalg/MemoryAlloc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#define GEN_PASS_DECL_MEMORYALLOC
#define GEN_PASS_DEF_MEMORYALLOC
#include "evas/Transform/Linalg/Passes.h.inc"

#define DEBUG_TYPE "ev-memory-alloc"

namespace mlir::triton::ev {

namespace {
using namespace mlir;

// TODO(wyann): remove this in the future
void setLiveL2ForTransposeCluster(func::CallOp callOp) {
  OpBuilder b(callOp);
  auto liveL2Str = mlir::ev::LiveBufString(mlir::ev::L2);
  auto liveL2Attr = b.getArrayAttr(
      {b.getI64ArrayAttr({0, mlir::ev::memCapacity(mlir::ev::L2)})});
  auto liveL1Str  =  mlir::ev::LiveBufString(mlir::ev::MM);
  auto liveL1Attr = b.getArrayAttr(
      {b.getI64ArrayAttr({0, mlir::ev::memCapacity(mlir::ev::MM)})});
  callOp->setAttr(liveL2Str, liveL2Attr);
  callOp->setAttr(liveL1Str, liveL1Attr);
  auto funcOp = mlir::ev::getCalledFunction(callOp);
  funcOp->setAttr(liveL2Str, liveL2Attr);
  funcOp->setAttr(liveL1Str, liveL1Attr);
  return;
}
struct MemoryAllocPass : public ::impl::MemoryAllocBase<MemoryAllocPass> {
  // MemoryAllocPass(size_t memScope, size_t alignment, bool preview,
  //                       bool bankopt)
  //     : :MemoryAllocBase() {
  //   this->memScope = memScope;
  //   this->alignment = alignment;
  //   this->preview = preview;
  //   this->bankopt = bankopt;
  // }
  using MemoryAllocBase::MemoryAllocBase;

  void runOnOperation() override {
    const CallGraph &CG = getAnalysis<CallGraph>();
    const Liveness &LN = getAnalysis<Liveness>();

    LLVM_DEBUG(CG.dump());

    std::set<CallGraphNode *> visitedNodes;
    std::queue<CallGraphNode *> toVisitNodes;

    const CallGraphNode *extCallerNode = CG.getExternalCallerNode();
    for (const CallGraphNode::Edge &edge : *extCallerNode) {
      assert(edge.isAbstract() && "Unexpected edge from externel node");
      CallGraphNode *calleeNode = edge.getTarget();
      if (calleeNode->isExternal())
        continue;
      Operation *callee = calleeNode->getCallableRegion()->getParentOp();
      assert(isa<FunctionOpInterface>(callee) && "Unexpected operation");
      // FIXME: Why private function can be called from externel node?
      //        CallGraph Analysis should be improved.
      if (cast<FunctionOpInterface>(callee).isPublic())
        toVisitNodes.push(calleeNode);
    }

    while (!toVisitNodes.empty()) {
      CallGraphNode *toVisitNode = toVisitNodes.front();
      toVisitNodes.pop();
      assert(!visitedNodes.count(toVisitNode) && "Node can't be visited twice");
      visitedNodes.insert(toVisitNode);
      for (const CallGraphNode::Edge &edge : *toVisitNode) {
        assert(edge.isCall() && "TODO: Support children node");
        CallGraphNode *calleeNode = edge.getTarget();
        if (calleeNode->isExternal())
          continue;
        toVisitNodes.push(calleeNode);
      }

      AllocPolicy policy = {alignment.getValue(), SIZE_PRIOR};
      MemoryAllocImpl MAI(policy, preview.getValue(), /* update callee */ true,
                          bankopt.getValue());
      Operation *op = toVisitNode->getCallableRegion()->getParentOp();
      assert(isa<FunctionOpInterface>(op) && "Unexpected operation");
      auto func = cast<FunctionOpInterface>(op);
      if (func.isPublic()) {
        if (memScope == 0) {
          // Walk over all functions and set memory allocation result.
          MAI.runOnFunction(func, LN);
        } else {
          MAI.runOnFuncAtScope(MemScope(memScope.getValue()), func, LN);
        }
      }
    }

    // TODO(wyann): very tricky code here, delete in the future
    // Iterate through all call ops in the module to find transpose subkernel
    getOperation()->walk([&](func::CallOp callOp) {
      auto calledFunc = mlir::ev::getCalledFunction(callOp);
      if (!calledFunc)
        return;
      bool hasTranspose = false;
      calledFunc->walk(
          [&](linalg::TransposeOp transposeOp) { hasTranspose = true; });
      if (!hasTranspose)
        return;
      // Check if the function has only MM scope attribute
      if (auto memScopeAttr =
              callOp->getAttrOfType<ArrayAttr>(mlir::ev::MEMSCOPE)) {
        for (auto memScope : memScopeAttr) {
          if (cast<IntegerAttr>(memScope).getInt() != mlir::ev::MM)
            return;
        }
      } else {
        return;
      }
      setLiveL2ForTransposeCluster(callOp);
    });
  }
};
} // namespace

std::unique_ptr<Pass> createMemoryAllocPass() {
  return std::make_unique<MemoryAllocPass>();
}

std::unique_ptr<Pass> createMemoryAllocPass(size_t memScope, size_t alignment,
                                            bool preview, bool bankopt) {
  MemoryAllocOptions opts{memScope, alignment, preview, bankopt};
  return std::make_unique<MemoryAllocPass>(opts);
}
} // namespace mlir::triton::ev
