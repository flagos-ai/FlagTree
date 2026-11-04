#pragma once

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::tle {

// Device calls forward a frame base, without a CTA index. Grid collectives use
// it directly; ordinary scratch accesses add the physical CTA stride locally.
inline Value getGlobalScratchBase(Location loc, RewriterBase &rewriter,
                                  FunctionOpInterface func, Value frameOffset) {
  Value base = func.getArgument(func.getNumArguments() + kGlobalScratchBufferOffset);
  if (!frameOffset)
    return base;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 1);
  return b.gep(ptrTy, i8_ty, base, frameOffset);
}

} // namespace mlir::triton::tle
