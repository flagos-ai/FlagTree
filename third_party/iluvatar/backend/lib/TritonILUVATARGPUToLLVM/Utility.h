#ifndef TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_UTILITY_H_

#include "TargetInfo.h"

#include "Dialect/TritonILUVATARGPU/Utility/CommonUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::ILUVATAR {

enum class MemoryOp { Load, Store };

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

// Emit the cta multicast mask for a given cta id based on the src layout
Value emitCtaMulticastMask(RewriterBase &rewriter, Location loc, Value blockId,
                           const LinearLayout &cvt);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
// forceNoAliasAsyncLoads=true adds alias information to the llvm.load to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, Value multicastMask,
             triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false, bool isVolatile = false);

// Stores to shared or global memory with predication.
// forceNoAliasAsyncLoads=true adds alias information to the llvm.store to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

// Get cache modifier information for creating load or store instruction
// Get flags <volatile, nontemporal> for a predicated Load or Store
std::pair<bool, bool> getCacheModifierFlagsForLoadStore(LLVM::CallOp);

Value cvtFp32ToFp16RTNE_oneValue(Location loc, RewriterBase &rewriter,
                                 const Value &v);

// Return a tensor of pointers with the same type of `basePtr` and the same
// shape of `offset`
Type getPointerTypeWithShape(Value basePtr, Value offset);

// Get contiguity for a tensor pointer `ptr`
unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);

// Get contiguity for a scalar pointer `ptr` and a tensor `offset`
unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

// Determine the vector size of a tensor of pointers
unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);

// Given a scalar pointer and a tensor of offsets, determine the vector size
unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

Type scaleDotElemTypeToMLIRType(MLIRContext *ctx, triton::ScaleDotElemType t);

// Returns true if we can perform coalesced write from the source encoding to
// the destination encoding for a given vec size.
bool canCoalesceWriteIntoSharedMemory(RewriterBase &rewriter,
                                      const LinearLayout &srcToSharedLayout,
                                      unsigned threadsPerWarp,
                                      unsigned vecSize);

// Returns true if the swizzling pattern does only swizzle the shared memory
// offsets of a warp and does not exchange destination elements across warps
bool doesSwizzleInsideWarp(RewriterBase &rewriter,
                           const LinearLayout &srcToSharedLayout,
                           unsigned threadsPerWarp);

// Return true if op is used by DotScaledOp or UpcastMXFPOp ops.
bool isUsedByDotScaledOp(Operation *op);

template <typename ConvertOp>
SmallVector<Value, 4>
upcast8xMxfp4_HW(RewriterBase &rewriter, Location loc, ArrayRef<Value> xVals,
                 int idx, Value scale, bool useShiftedScale = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp4Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  // In the DotScaledOp decomposition, the scale has already been left-shifted
  // by 7 to fit the exponent of bf16. So now we only need to further left-shift
  // it by 16
  Value scaleF32;
  if (useShiftedScale) {
    scaleF32 = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  } else {
    scaleF32 = b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  }
  SmallVector<Value, 4> results;
  for (int srcSelIndex : llvm::seq(4))
    results.push_back(ConvertOp::create(rewriter, loc, resType, packedVec,
                                        scaleF32, srcSelIndex));
  return results;
}

template <typename ConvertOp>
SmallVector<Value, 2>
upcast4xMxfp8_HW(RewriterBase &rewriter, Location loc, ArrayRef<Value> xVals,
                 int idx, Value scale, bool useShiftedScale = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packedVec = b.undef(vec_ty(i8_ty, 4));
  for (int i : llvm::seq(4))
    packedVec = b.insert_element(packedVec, xVals[idx + i], b.i32_val(i));
  packedVec = b.bitcast(packedVec, i32_ty);
  Type retElemType = bf16_ty;
  if constexpr (std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Fp8Op> ||
                std::is_same_v<ConvertOp, ROCDL::CvtScaleF32PkF16Bf8Op>)
    retElemType = f16_ty;
  Type resType = vec_ty(retElemType, 2);
  // In the DotScaledOp decomposition, the scale has already been left-shifted
  // by 7 to fit the exponent of bf16. So now we only need to further left-shift
  // it by 16
  Value scaleF32;
  if (useShiftedScale) {
    scaleF32 = b.bitcast(
        b.shl(b.zext(i32_ty, b.bitcast(scale, i16_ty)), b.i32_val(16)), f32_ty);
  } else {
    scaleF32 = b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  }
  SmallVector<Value, 2> results;
  results.push_back(ConvertOp::create(rewriter, loc, resType, packedVec,
                                      scaleF32,
                                      /*srcLoHiSel=*/false));
  results.push_back(ConvertOp::create(rewriter, loc, resType, packedVec,
                                      scaleF32,
                                      /*srcLoHiSel=*/true));
  return results;
}
} // namespace mlir::LLVM::ILUVATAR

#endif // TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_UTILITY_H_
