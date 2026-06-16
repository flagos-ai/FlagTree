#include "Utility.h"
#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
namespace tt = mlir::triton;
using mlir::triton::ModuleAxisInfoAnalysis;
using mlir::triton::gpu::appendOrGetExternFuncOp;

namespace {
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};
} // namespace

namespace mlir::LLVM::ILUVATAR {
static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, int strideInt, ShflKind mode,
                               Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.sext(i32_ty, val);

    val = shuffleCommonImpl(loc, rewriter, val, i, strideInt, mode, clamp);

    if (bits < 32)
      val = b.trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, val0, i, strideInt, mode, clamp);
    val1 = shuffleCommonImpl(loc, rewriter, val1, i, strideInt, mode, clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId = getThreadId(rewriter, loc);

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = b.i32_val(iWarpSize);
  Value laneId = b.urem(threadId, warpSize);

  auto index = b.i32_val(0);
  switch (mode) {
  case ShflKind::up:
    index = b.sub(laneId, i);
    break;
  case ShflKind::idx:
    index = i;
    break;
  case ShflKind::bfly:
    index = b.xor_(laneId, i);
    break;
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }

  /**
   *  Implemented `shflSync` with reference to cuda api `__shfl_down`, in
   * ixcc/clang/lib/Headers/__clang_cuda_intrinsics.h line 118. Notice: When
   * adding the boundary condition of `index`, the result is incorrect.
   *  Condition:
   *       index = (int)((self & (width - 1)) + lane_delta) >= width ? self :
   * index; Implementation is as follows: auto index_delta = add(and_(laneId,
   * sub(warpSize, i32_val(1))), i32_val(i)); auto index_is_illegal =
   * icmp_uge(index_delta, warpSize); auto index_dst = select(index_is_illegal,
   * laneId, index);
   */

  StringRef func_name = "llvm.bi.slb.shfl.idx.b32";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, func_name, {i32_ty},
                                         {val, index})
      .getResult(0);
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result =
      shuffleCommonImpl(loc, rewriter, val, i, strideInt, mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), i, ShflKind::bfly,
                       b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), i, ShflKind::up,
                       b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, i, 0, ShflKind::idx,
                       b.i32_val(0x1f));
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector) {
  Value args[] = {a, b, selector};
  auto op =
      createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.prmt", i32_ty, args);
  return op.getResult(0);
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  if (numCTAs == 1) {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::BlockIdXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::BlockIdYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::BlockIdZOp::create(rewriter, loc, i32_ty);
    }
  } else {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::ClusterIdXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::ClusterIdYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::ClusterIdZOp::create(rewriter, loc, i32_ty);
    }
  }
  llvm_unreachable("invalid axis");
}

// For multicast memory operations (e.g., cluster.load.async.to.lds), we need a
// bitmask indicating which CTAs in the CGA/cluster will access the same memory
// addresses. This allows the hardware to efficiently broadcast data to multiple
// CTAs. The linear layout's free variables in the block dimension tell us which
// CTAs form a "communication group" (i.e., access the same data):
//   - Free bit at position k: CTAs whose IDs differ only in bit k access
//     the same data and should be in the same multicast group.
//   - Fixed bits (non-free): Distinguish between different groups that
//     access different data.
// The multicast mask has bit i set if CTA i is in the same communication
// group as the current CTA. The free bits determine a groupMask whereas the
// non-free bits determine the group offset:
//   ctaMask = groupMask << groupOffset
// where:
//   - groupMask: Covers all 2^k CTAs in the group (k = number of free bits)
//   - groupOffset: Starting position of this group, determined by fixed bits
// As an example suppose we have 8 CTAs and freeVarMask = 0b101 (bits 0,2 free).
// This creates 2 groups of 4 CTAs each:
//   - Group 0: CTAs {0,1,4,5} (fixed bits = 0b000)
//   - Group 1: CTAs {2,3,6,7} (fixed bits = 0b010)
// For CTA 5 (0b101): groupOffset = 0b101 & 0b010 = 0 => ctaMask = 0b00110011
// For CTA 7 (0b111): groupOffset = 0b111 & 0b010 = 2 => ctaMask = 0b11001100
Value emitCtaMulticastMask(RewriterBase &rewriter, Location loc, Value groupId,
                           const LinearLayout &regLayout) {
  TritonLLVMOpBuilder b(loc, rewriter);

  auto kBlock = StringAttr::get(rewriter.getContext(), "block");
  auto freeVarMask = regLayout.getFreeVariableMasks()[kBlock];

  // If there are no free bits we do not share any data with other CTAs
  if (freeVarMask == 0) {
    return Value();
  }

  // Construct the groupMask with 1s at all positions representing CTAs in the
  // communication group. We start with 0b1 and iterate over free bits. For
  // every free bit at position k, we copy the current pattern 2^k positions
  // higher.
  // Example for freeVarMask = 0b101, x = non determined yet:
  //   Initial:          groupMask = 0bxxxxxxx1 (positions {0})
  //   Bit 0 (free):     groupMask = 0bxxxxxx11 (positions {0,1})
  //   Bit 1 (non-free): groupMask = 0bxxxx0011 (positions {0,1})
  //   Bit 2 (free):     groupMask = 0b00110011 (positions {0,1,4,5})
  int groupMask = 1;
  for (int log2 = 0; log2 < regLayout.getInDimSizeLog2(kBlock); log2++) {
    if (!(freeVarMask & (1 << log2)))
      continue;
    groupMask = groupMask | (groupMask << (1 << log2));
  }
  // If all bits are set we broadcast to all CTAs so return the group mask.
  if (freeVarMask == regLayout.getInDimSize(kBlock) - 1) {
    return b.i32_val(groupMask);
  }
  // The non-free bits set in the ctaId determine the group offset. For every
  // non-free bit set at position k, we shift the groupMask by 2^k positions.
  // This can be conviniently computed by masking the ctaId with the inverse
  // of the freeVarMask.
  // Example1: freeVarMask = 0b101
  //   ~freeVarMask  = 0b010
  //   shiftAmount   = 0b101 & 0b010 = 0b000 (no shift needed)
  //   blockMask     = 0b110011 << 0 = 0b00110011
  // Example2: freeVarMask = 0b101, ctaId = 0b111 (cta 7)
  //   ~freeVarMask  = 0b010
  //   shiftAmount   = 0b111 & 0b010 = 0b010 (shift by 2)
  //   blockMask     = 0b110011 << 2 = 0b11001100
  Value shiftAmount = b.and_(groupId, b.i32_val(~freeVarMask));
  Value ctaMask = b.shl(b.i32_val(groupMask), shiftAmount);
  return ctaMask;
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, Value multicastMask,
             triton::CacheModifier cm, bool forceNoAliasAsyncLoads,
             bool isVolatile) {
  return triton::iluvatargpu::MaskedLoadOp::create(
             rewriter, loc, elemTy, ptr, pred, falseVal, multicastMask, cm,
             forceNoAliasAsyncLoads, isVolatile)
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm,
             bool forceNoAliasAsyncLoads) {
  triton::iluvatargpu::MaskedStoreOp::create(rewriter, loc, ptr, val, pred, cm,
                                             forceNoAliasAsyncLoads);
}

Value cvtFp32ToFp16RTNE_oneValue(Location loc, RewriterBase &rewriter,
                                 const Value &v) {
  LLVM::RoundingMode rm = LLVM::RoundingMode::NearestTiesToEven;
  return LLVM::FPTruncOp::create(rewriter, loc, f16_ty, v);
}

Type getPointerTypeWithShape(Value basePtr, Value offset) {
  Type basePtrType = basePtr.getType();
  auto offsetType = cast<RankedTensorType>(offset.getType());
  return offsetType.cloneWith(std::nullopt, basePtrType);
}

unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  return axisAnalysisPass.getContiguity(ptr);
}

unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {

  Type type = getPointerTypeWithShape(ptr, offset);
  RankedTensorType tensorTy = cast<RankedTensorType>(type);

  // To compute the contiguity of the scalar/warp-uniform ptr and offset pair we
  // need to look at the contiguity of the offsets and the alignment of the ptr
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto contiguity = axisAnalysisPass.getContiguity(offset, elemNumBits);

  // To get the alignment of the scalar ptr we need to look at the divisibility
  auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
  auto maxMultipleBytes = axisInfo->getDivisibility(0);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto align = std::max<unsigned>(maxMultipleBytes / elemNumBytes, 1);

  // FIXME (Alex): this should not be needed anymore because it's done inside
  // getContiguity, but we have an order issues with LL, so we keep this
  // until the LL order issue is fixed
  auto linearLayout = triton::gpu::toLinearLayout(tensorTy);
  auto llAttr =
      triton::gpu::LinearEncodingAttr::get(tensorTy.getContext(), linearLayout);
  auto order = triton::gpu::getOrder(tensorTy);
  auto contigPerThread = llAttr.getContigPerThread();
  assert(order[0] < contigPerThread.size() &&
         "Unexpected contigPerThread size");
  contiguity = std::min(contiguity, contigPerThread[order[0]]);

  // Final contiguity is a min of the offset contiguity and pointer alignment
  return std::min(align, contiguity);
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto contiguity = getContiguity(ptr, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto contiguity = getContiguity(ptr, offset, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

Type scaleDotElemTypeToMLIRType(MLIRContext *ctx, triton::ScaleDotElemType t) {
  switch (t) {
  case triton::ScaleDotElemType::FP16:
    return Float16Type::get(ctx);
  case triton::ScaleDotElemType::BF16:
    return BFloat16Type::get(ctx);
  case triton::ScaleDotElemType::E4M3:
    return Float8E4M3FNType::get(ctx);
  case triton::ScaleDotElemType::E5M2:
    return Float8E5M2Type::get(ctx);
  case triton::ScaleDotElemType::E3M2:
    return Float6E3M2FNType::get(ctx);
  case triton::ScaleDotElemType::E2M3:
    return Float6E2M3FNType::get(ctx);
  case triton::ScaleDotElemType::E2M1:
    return Float4E2M1FNType::get(ctx);
  default:
    llvm_unreachable("unsupported ScaleDotElemType!");
  }
}

bool canCoalesceWriteIntoSharedMemory(RewriterBase &rewriter,
                                      const LinearLayout &srcToSharedLayout,
                                      unsigned threadsPerWarp,
                                      unsigned vecSize) {
  auto contig = srcToSharedLayout.getNumConsecutiveInOut();
  if (vecSize != srcToSharedLayout.getNumConsecutiveInOut()) {
    LDBG("Load vectorization ("
         << vecSize << ") and contiguity (" << contig
         << ") do not match resulting in strided writes");
    return false;
  }

  StringAttr kLane = rewriter.getStringAttr("lane");
  for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
    auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
    unsigned expected = contig * (1 << inLane);
    if (basis != expected) {
      LDBG("detected uncoalesced layout from blocked to shared in async copy "
           "for lane "
           << 1 + inLane << "; given " << basis << " but expected "
           << expected);
      return false;
    }
  }
  // Additionally we could swizzle based on the warp dimension so we need to
  // check that when all bases are divided by contig, none of the first
  // (log2(warpSize) + 1) bits are set to 1
  assert(llvm::isPowerOf2_32(threadsPerWarp));
  assert(llvm::isPowerOf2_32(contig));
  unsigned mask = (threadsPerWarp * contig) - 1;
  StringAttr kWarp = rewriter.getStringAttr("warp");
  for (int inWarp : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kWarp))) {
    auto basis = srcToSharedLayout.getBasis(kWarp, inWarp)[0];
    if ((basis & mask) != 0) {
      LDBG("detected uncoalesced layout from blocked to shared in async copy "
           "for warp "
           << inWarp);
      return false;
    }
  }

  return true;
}

bool doesSwizzleInsideWarp(RewriterBase &rewriter,
                           const LinearLayout &srcToSharedLayout,
                           unsigned threadsPerWarp) {
  auto contig = srcToSharedLayout.getNumConsecutiveInOut();
  // If all bases in lane dimension are below threadsPerWarp multiplied with the
  // contiguity we do not swizzle across warp boundaries.
  assert(llvm::isPowerOf2_32(threadsPerWarp));
  unsigned upperLimit = threadsPerWarp * contig;

  StringAttr kLane = rewriter.getStringAttr("lane");
  for (int inLane : llvm::seq(srcToSharedLayout.getInDimSizeLog2(kLane))) {
    auto basis = srcToSharedLayout.getBasis(kLane, inLane)[0];
    if (basis >= upperLimit) {
      return false;
    }
  }
  return true;
}

} // namespace mlir::LLVM::ILUVATAR
