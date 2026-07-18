// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITONXPU_ANALYSIS_UTILITY_H
#define TRITONXPU_ANALYSIS_UTILITY_H

#include "third_party/xpu/include/triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "llvm/Support/Casting.h"
#include <unordered_set>
#include <vector>

namespace mlir {

#define XPU_MEMORY_OP                                                          \
  triton::xpu::GM2LMOp, triton::xpu::LM2GMOp, triton::xpu::SM2GMOp

template <class T> struct is_xpu_memory_op {
  static const bool value = false;
};
template <> struct is_xpu_memory_op<triton::xpu::GM2LMOp> {
  static const bool value = true;
};
template <> struct is_xpu_memory_op<triton::xpu::LM2GMOp> {
  static const bool value = true;
};
template <> struct is_xpu_memory_op<triton::xpu::SM2GMOp> {
  static const bool value = true;
};

#define XPU_MEMORY_MASK_OP                                                     \
  triton::xpu::GM2LMMaskOp, triton::xpu::LM2GMMaskOp, triton::xpu::SM2GMMaskOp

template <class T> struct is_xpu_memory_mask_op {
  static const bool value = false;
};
template <> struct is_xpu_memory_op<triton::xpu::GM2LMMaskOp> {
  static const bool value = true;
};
template <> struct is_xpu_memory_op<triton::xpu::LM2GMMaskOp> {
  static const bool value = true;
};
template <> struct is_xpu_memory_op<triton::xpu::SM2GMMaskOp> {
  static const bool value = true;
};

#define ARITH_PTR_UNARY_OP arith::ExtSIOp

#define ARITH_PTR_BINARY_OP                                                    \
  arith::DivSIOp, arith::RemSIOp, arith::MulIOp, arith::AddIOp, arith::SubIOp

#define XPU_VVECTORIZED_BINARY_OP                                              \
  triton::xpu::VvaddFOp, triton::xpu::VvmulFOp, triton::xpu::VvsubFOp,         \
      triton::xpu::VvmaxFOp, triton::xpu::VvxorIOp

#define XPU_SVECTORIZED_BINARY_OP                                              \
  triton::xpu::SvaddFOp, triton::xpu::SvmulFOp, triton::xpu::SvsubFOp,         \
      triton::xpu::SvmaxFOp

#define COMBINE_BINARY_OP                                                      \
  arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp,            \
      arith::OrIOp, arith::XOrIOp, arith::AndIOp

#define COMBINE_OP COMBINE_BINARY_OP, arith::CmpFOp

enum class OffsetState {
  Unknown = -1,
  DiscreteSame = 0,
  Continuous = 1,
  Discrete = 2,
  LocallyContinuous = 3
};

enum class ElemState {
  SS = 0, /*00*/
  SV = 1, /*01*/
  VS = 2, /*10*/
  VV = 3  /*11*/
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const OffsetState &state);

enum class AtomicMaskCond {
  PostiveCond = 1,
  NegativeCond = -1,
  NonActivate = 0,
};

enum class AtomicMaskType {
  NaiveMask = 1,
  OptimizationMask = 2,
};

enum class XPUArch { XPU2 = 2, XPU3 = 3 };

enum class MemCpyType { GM2LM = 0, LM2GM = 1, GM2SM = 2, SM2GM = 3 };

class SMHelper {
public:
  explicit SMHelper(Operation *op) : op(op) {}

  void setOffset(int64_t offset) { smOffsetMap[op] = offset; }

  int64_t getOffset() {
    int64_t offset = 0;
    if (hasOffset()) {
      offset = smOffsetMap[op];
    }
    return offset;
  }

  bool hasOffset() { return smOffsetMap.find(op) != smOffsetMap.end(); }

private:
  Operation *op;
  static std::map<Operation *, int64_t> smOffsetMap;
};

size_t previousPowerOf2(size_t n);

Type addrspaceCast(Type type, int addressSpace);

bool inOpChain(llvm::SetVector<Operation *> &opChain, Operation *op);

void getOpChainBwd(llvm::SetVector<Operation *> &opChain, Operation *op);
void getOpChainFwd(llvm::SetVector<Operation *> &opChain, Operation *op);
void getOpTreeBwd(llvm::SetVector<Operation *> &opTree,
                  llvm::SetVector<Operation *> &visitedOps, Operation *op);
void getOpTreeBwd(llvm::SetVector<Operation *> &opTree,
                  llvm::SetVector<Operation *> &visitedOps, Operation *op,
                  Block *block);
void checkDefUseShapeMatch(ModuleOp &m, MLIRContext *context);

llvm::SmallVector<Operation *>
sortOpTreeBwd(llvm::SmallVector<Operation *> &opTree);
llvm::SetVector<Operation *>
sortOpTreeBwd(llvm::SetVector<Operation *> &opTree);
llvm::SetVector<Operation *> sortOpTree(llvm::SetVector<Operation *> &opTree);

bool inSameSCFIfBlock(llvm::SetVector<Operation *> &storeOps,
                      Operation *storeOp);

void getOpLine(ModuleOp &m, DenseMap<mlir::Operation *, unsigned> &op2Line);

int64_t getTensorSize(Type type);

template <typename opType>
Operation *findUserOpImpl(Operation *op,
                          llvm::SetVector<Operation *> &visitedOps) {
  if (!op || op->use_empty() || visitedOps.contains(op))
    return nullptr;

  visitedOps.insert(op);

  if (isa<opType>(op)) {
    return op;
  }

  for (Operation *user : op->getUsers()) {
    Operation *userOp = findUserOpImpl<opType>(user, visitedOps);
    if (userOp) {
      return userOp;
    }
  }

  return nullptr;
}

template <typename opType> Operation *findUserOp(Operation *op) {
  llvm::SetVector<Operation *> visitedOps;
  return findUserOpImpl<opType>(op, visitedOps);
}

template <typename opType>
std::vector<Operation *> findAllTypeUserOps(Operation *startOp) {
  std::vector<Operation *> found;
  if (!startOp)
    return found;

  std::unordered_set<Operation *> visited;
  std::deque<Operation *> q;

  visited.insert(startOp);
  q.push_back(startOp);

  while (!q.empty()) {
    Operation *cur = q.front();
    q.pop_front();

    for (Value res : cur->getResults()) {
      for (Operation *userOp : res.getUsers()) {
        if (!userOp)
          continue;
        if (visited.insert(userOp).second) {
          if (llvm::dyn_cast<opType>(userOp)) {
            found.emplace_back(userOp);
          }
          q.push_back(userOp);
        }
      }
    }
  }

  return found;
}

template <typename opType> opType findDefOpBwd(const Value &val) {
  if (!val || !val.getDefiningOp()) {
    return nullptr;
  }
  auto op = val.getDefiningOp();
  if (op && isa<opType>(op)) {
    return cast<opType>(op);
  }
  for (auto operand : op->getOperands()) {
    op = findDefOpBwd<opType>(operand);
    if (op) {
      return cast<opType>(op);
    }
  }
  return nullptr;
}

} // namespace mlir

#endif // TRITONXPU_ANALYSIS_UTILITY_H
