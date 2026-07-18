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

#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h" // TODO(fywkevin): move Utility.h to include/
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::gpu::NVIDIA {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {

  auto getClockReg = [&](const std::string &clkName) {
    PTXBuilder builder;
    auto &movLow = builder.create("mov")->o("u32");
    auto *destLowOpr = builder.newOperand("=r");
    auto *sRegLowOpr = builder.newConstantOperand(clkName);
    movLow(destLowOpr, sRegLowOpr);
    Value clkLow32 =
        builder.launch(rewriter, loc, rewriter.getIntegerType(32), true);
    return clkLow32;
  };

  Value clkLow32 = getClockReg("%clock");

  if (!isClock64)
    return clkLow32;

  Value clkHigh32 = getClockReg("%clock_hi");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value clkLow64 = b.zext(i64_ty, clkLow32);
  Value clkHigh64 = b.zext(i64_ty, clkHigh32);
  Value clock64 = b.or_(b.shl(clkHigh64, b.i64_val(32)), clkLow64);
  return clock64;
}

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  // globaltimer is a 64-bit global clock counter in nanoseconds.
  // Reference:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringRef globalTimeIntrinsicName = "llvm.nvvm.read.ptx.sreg.globaltimer";
  Value globalTimeVal = LLVM::createLLVMIntrinsicCallOp(
                            rewriter, loc, globalTimeIntrinsicName, i64_ty, {})
                            .getResult(0);
  return globalTimeVal;
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return NVVM::SmIdOp::create(rewriter, loc, i32_ty);
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else if (mlir::isa<proton::gpu::GlobalMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "and GlobalMemorySpace for now");
  }
  return spaceId;
}

int TargetInfo::getIndexPtrAddrSpace() const {
  // Internal buffer index is private to each thread, we use generic address
  // space for NV GPUs. See detail discussion:
  // https://llvm.org/docs/NVPTXUsage.html#address-spaces
  // The reason we don't use address space 5 is due to the downstream compiler
  // generates incorrect `cvta` instruction for %SP/%SPL register that causes
  // IMA when we perform thread-private memory access like `ld.local`.
  return 0;
}

} // namespace mlir::triton::proton::gpu::NVIDIA
