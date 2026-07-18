// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// clang-format off
#include "TargetInfo.h"  // TargetInfo

#include "triton/Analysis/NewAnalysis/Utility.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
// clang-format on

using namespace mlir;

namespace mlir {
namespace triton {
namespace xpu {

bool TargetInfo::supportMaximumMinimum() const {
  llvm_unreachable("not impl");
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm_unreachable("not impl");
  return Value();
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  llvm_unreachable("not impl");
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  llvm_unreachable("not impl");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::XPU::llGetPid(loc, rewriter, moduleOp, static_cast<int>(axis));
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  llvm_unreachable("not impl");
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "_ZN3xpu6umulhiEjj" : "Unsupported";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  llvm_unreachable("not impl");
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  llvm_unreachable("not impl");
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  llvm_unreachable("not impl");
}

int TargetInfo::getSharedAddressSpace() const { return 2; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  // XPU uses address space 2 for shared/local memory (LM).
  return 2;
}

bool TargetInfo::supportVectorizedAtomics() const { return false; }

uint32_t TargetInfo::getXPUArch() const { return this->xpu_arch; }
uint32_t TargetInfo::getXPUBufferSize() const { return this->buffer_size; }
bool TargetInfo::getXPUIsUseMaskZero() const { return this->isUseMaskZero; }

} // namespace xpu
} // namespace triton
} // namespace mlir
