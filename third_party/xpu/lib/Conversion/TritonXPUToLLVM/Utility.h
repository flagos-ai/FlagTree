#ifndef TRITON_CONVERSION_TRITONXPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONXPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

// NOTE: dragon-style free macros (i32_val, add, mul, bitcast, barrier, ...)
// have moved into "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h".
// That header MUST be included from every XPU `.cpp` translation unit AFTER
// all class headers (e.g. TargetInfo.h), and MUST NOT be pulled into other
// headers, otherwise macros like `barrier`, `select`, `add`, `addrspace_cast`
// collide with class methods and STL templates.

namespace mlir::triton {
inline size_t align(size_t elemNum, Type elemTy, size_t target) {
  size_t elemBit = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                       ? 64u
                       : elemTy.getIntOrFloatBitWidth();
  size_t elemBytes = (elemBit / 8u) ? (elemBit / 8u) : 1;
  size_t aligned = (elemNum * elemBytes + target - 1) / target * target;
  return aligned / elemBytes;
}
} // namespace mlir::triton

namespace mlir::LLVM::XPU {

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

Value createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                       Operation *op, Type &elemTy, ValueRange &operands,
                       Location &loc);

void createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                      Operation *op, ValueRange &operands, Location &loc);

SmallVector<SmallVector<unsigned>>
emitOffsetForClusterLayout(const triton::xpu::ClusterLayoutAttr &clusterLayout,
                           RankedTensorType type);

inline Value getGridDim(RewriterBase &rewriter, Location loc) {
  Value gridDim =
      rewriter.create<::mlir::gpu::GridDimOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, gridDim);
}

inline Value getBlockDim(RewriterBase &rewriter, Location loc) {
  Value blockDim =
      rewriter.create<::mlir::gpu::BlockDimOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockDim);
}

inline Value getBlockId(RewriterBase &rewriter, Location loc) {
  Value blockId =
      rewriter.create<::mlir::gpu::BlockIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

// XPU-local getThreadId, mirrors Triton 3.0's getThreadId in
// `triton/Conversion/TritonGPUToLLVM/Utility.h`. Triton 3.6's upstream
// `mlir::getThreadId` AND-masks the thread id by
// `numWarps * threadsPerWarp - 1` (=31 with the default num-warps=1,
// threads-per-warp=32 module attrs). On XPU the real range of `core_id` is
// [0, 64), so that mask aliases cores 32..63 onto 0..31 and produces the
// `(core_id & 31) * 16` lowering observed in MakeRangeOp. This XPU-namespaced
// helper recovers 3.0 semantics (raw core_id, no mask).
inline Value getThreadId(RewriterBase &rewriter, Location loc) {
  Value tid =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
}

} // namespace mlir::LLVM::XPU

#endif // TRITON_CONVERSION_TRITONXPU_TO_LLVM_UTILITY_H
