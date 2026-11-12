#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir::triton::Intellifusion {
namespace {
bool hasAddrSpace(triton::gpu::AddrSpace mask, triton::gpu::AddrSpace space) {
  return (mask & space) != triton::gpu::AddrSpace::None;
}

// True iff `pred` is a statically-known non-zero i1 / iN constant.
bool isConstantTruePred(Value pred) {
  if (!pred)
    return true;
  APInt c;
  if (matchPattern(pred, m_ConstantInt(&c)))
    return !c.isZero();
  return false;
}
} // namespace

// ---------------------------------------------------------------------------
// Capability queries
// ---------------------------------------------------------------------------

bool TargetInfo::supportMaximumMinimum() const {
  return target::kSupportsMaximumMinimum;
}
bool TargetInfo::supportVectorizedAtomics() const {
  return target::kSupportsVectorizedAtomics;
}

// ---------------------------------------------------------------------------
// Program / Thread identification
// ---------------------------------------------------------------------------

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  int numCTAs =
      triton::gpu::lookupNumCTAs(&rewriter.getInsertionBlock()->front());
  if (numCTAs == 1)
    return arith::ConstantIntOp::create(rewriter, loc, 0, 32);

  // Derive the cluster rank from the physical CTA id so the CModel can also
  // execute non-cluster group-1 layouts.
  auto builder = TritonLLVMOpBuilder(loc, rewriter);
  Value physicalCTAId = LLVM::createLLVMIntrinsicCallOp(
                            rewriter, loc, "llvm.nvvm.read.ptx.sreg.ctaid.x",
                            TypeRange(rewriter.getI32Type()), ValueRange())
                            .getResult(0);
  return builder.urem(physicalCTAId, builder.i32_val(numCTAs));
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  StringRef intrinsicName;
  switch (axis) {
  case ProgramIDDim::X:
    intrinsicName = "llvm.nvvm.read.ptx.sreg.ctaid.x";
    break;
  case ProgramIDDim::Y:
    intrinsicName = "llvm.nvvm.read.ptx.sreg.ctaid.y";
    break;
  case ProgramIDDim::Z:
    intrinsicName = "llvm.nvvm.read.ptx.sreg.ctaid.z";
    break;
  }
  auto i32Ty = rewriter.getIntegerType(32);
  Value physicalProgramId =
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName,
                                      TypeRange(i32Ty), ValueRange())
          .getResult(0);
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);
  if (numCTAs == 1 || axis != ProgramIDDim::X)
    return physicalProgramId;

  // IFWA clusters are launched only along X. Dividing the physical CTA id by
  // the cluster width gives the logical Triton program id.
  auto builder = TritonLLVMOpBuilder(loc, rewriter);
  return builder.udiv(physicalProgramId, builder.i32_val(numCTAs));
}

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  // CTA-wide barrier. Lower directly to the NVVM dialect's bar.sync 0 op,
  // whose LLVM-IR translation interface is registered with the standard
  // NVVM target backend used by IFWA.
  NVVM::BarrierOp::create(rewriter, loc);
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter,
                                Operation * /*sourceOp*/) const {
  NVVM::ClusterArriveOp::create(rewriter, loc,
                                UnitAttr::get(rewriter.getContext()));
  NVVM::ClusterWaitOp::create(rewriter, loc,
                              UnitAttr::get(rewriter.getContext()));
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  // IFWA has only a single warp width and CTA-wide barrier; warpSync collapses
  // to the same barrier.
  barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

// ---------------------------------------------------------------------------
// Shared-memory load/store with predicate (used by ReduceOpConversion's
// cross-warp phase). Use an LLVM cond_br + phi pattern matching IFWA's
// existing LoadStoreOpToLLVM.cpp style.
// ---------------------------------------------------------------------------

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Value val, Value pred) const {
  if (ctaId) {
    // Map the pointer to addrspace(7); IFWA lowers accesses in that address
    // space as cluster-shared loads and stores.
    auto clusterPtrTy = ptr_ty(rewriter.getContext(), /*addrspace=*/7);
    ptr = NVVM::MapaOp::create(rewriter, loc, clusterPtrTy, ptr, ctaId);
  }
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *afterBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *trueBlock = rewriter.createBlock(afterBlock);

  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, trueBlock, afterBlock);

  rewriter.setInsertionPointToStart(trueBlock);
  LLVM::StoreOp::create(rewriter, loc, val, ptr);
  LLVM::BrOp::create(rewriter, loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Type elemTy, Value pred,
                              Operation *localLoadOp) const {
  if (ctaId) {
    // Map the pointer to addrspace(7); IFWA lowers accesses in that address
    // space as cluster-shared loads and stores.
    auto clusterPtrTy = ptr_ty(rewriter.getContext(), /*addrspace=*/7);
    ptr = NVVM::MapaOp::create(rewriter, loc, clusterPtrTy, ptr, ctaId);
  }
  // Build: if (pred) v = load ptr; else v = poison; (via cond_br + phi)
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *afterBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *trueBlock = rewriter.createBlock(afterBlock);

  // afterBlock takes a phi-style block argument with the loaded value.
  afterBlock->addArgument(elemTy, loc);

  rewriter.setInsertionPointToEnd(currentBlock);
  Value poison = LLVM::PoisonOp::create(rewriter, loc, elemTy);
  LLVM::CondBrOp::create(rewriter, loc, pred, trueBlock, ValueRange(),
                         afterBlock, ValueRange(poison));

  rewriter.setInsertionPointToStart(trueBlock);
  Value loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);
  LLVM::BrOp::create(rewriter, loc, ValueRange(loaded), afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
  return afterBlock->getArgument(0);
}

// ---------------------------------------------------------------------------
// Warp shuffle implementations using NVVM's `shfl.sync.bfly` intrinsic.
//
// IFWA vcc/llc natively supports `llvm.nvvm.shfl.sync.bfly.i32`, so lower
// `shuffleXor` directly to the hardware warp shuffle (no shared memory).
// ---------------------------------------------------------------------------

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // 64-bit values: split into 2x f32, shuffle each half, then reassemble.
  if (bits == 64) {
    Type f32Ty = rewriter.getF32Type();
    Type vecTy = LLVM::getVectorType(f32Ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32Ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32Ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommonImpl(loc, rewriter, val1, i, mode, clamp);
    vec = LLVM::UndefOp::create(rewriter, loc, vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  Type type = val.getType();
  Type i32Ty = rewriter.getIntegerType(32);
  if (type != i32Ty) {
    val = b.bitcast(val, rewriter.getIntegerType(bits));
    if (bits < 32)
      val = b.zext(i32Ty, val);
  }
  Value mask = b.i32_val(0xFFFFFFFF);
  Value result = NVVM::ShflOp::create(rewriter, loc, i32Ty, mask, val, i, clamp,
                                      mode, UnitAttr());
  if (type != i32Ty) {
    if (bits < 32)
      result = b.trunc(rewriter.getIntegerType(bits), result);
    result = b.bitcast(result, type);
  }
  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, route through i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(rewriter.getIntegerType(64), val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadMask = b.int_val(type.getIntOrFloatBitWidth(), -1);
  return NVVM::VoteSyncOp::create(rewriter, loc, type, threadMask, cmp,
                                  NVVM::VoteSyncKind::ballot);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value clamp = b.i32_val(0x1f);
  Value mask = b.i32_val(i);
  return shuffleCommon(loc, rewriter, val, mask, NVVM::ShflKind::bfly, clamp);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::up,
                       b.i32_val(0x0));
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(rewriter, loc, val, b.i32_val(i));
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx,
                       b.i32_val(0x1f));
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.prmt",
                                         rewriter.getI32Type(),
                                         ValueRange{a, b, selector})
      .getResult(0);
}

// ---------------------------------------------------------------------------
// warpReduce: full-warp fast path that iterates shfl.bfly with masks
// {16, 8, 4, 2, 1} and clones the user-supplied combine op each round.
// Returning true skips the framework's generic shuffleXor loop. For
// partitioned masks (cross-warp post-layout-conversion) we return false so
// the framework iterates only the bits set in `reduceLaneIdMask`.
//
// `op.getSingleCombiner()` covers the common single-op combines
// (arith.maxnumf/minnumf, arith.addf, arith.maxsi/maxui, arith.andi/ori, ...).
// We clone via IRMapping to remain combine-op agnostic, mirroring how the
// generic ReduceOpConversion::accumulate path handles non-redux targets.
// ---------------------------------------------------------------------------

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask) const {
  if (reduceLaneIdMask != static_cast<unsigned>(threadsPerWarp - 1))
    return false;

  Operation *combineOp = op.getSingleCombiner();
  if (!combineOp)
    return false;

  for (size_t i = 0; i < acc.size(); ++i) {
    for (int mask = threadsPerWarp / 2; mask >= 1; mask /= 2) {
      Value partner = shuffleXor(rewriter, loc, acc[i], mask);
      IRMapping mapping;
      mapping.map(combineOp->getOperand(0), acc[i]);
      mapping.map(combineOp->getOperand(1), partner);
      Operation *cloned = rewriter.clone(*combineOp, mapping);
      acc[i] = cloned->getResult(0);
    }
  }
  return true;
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<triton::gpu::SharedMemorySpaceAttr,
          triton::nvidia_gpu::TensorMemorySpaceAttr>(addressSpace))
    return 3;
  mlir::emitError(UnknownLoc::get(addressSpace.getContext()),
                  "GPNPU: unsupported address space attribute");
  return 0;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  return resultElementTy.isInteger(32) ? "__nv_umulhi" : "__nv_umul64hi";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>("printf");
  if (!func) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMFunctionType::get(rewriter.getI32Type(),
                                            {LLVM::LLVMPointerType::get(ctx)},
                                            /*isVarArg=*/true);
    func = LLVM::LLVMFuncOp::create(rewriter, loc, "printf", type);
  }

  SmallVector<Value> operands{formatStrStart};
  // IFWA printf takes C varargs and accepts generic-address-space literals.
  for (auto [i, arg] : llvm::enumerate(args)) {
    auto type = arg.getType();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (isSigned.empty() || isSigned[i])
        arg = b.sext(rewriter.getI32Type(), arg);
      else
        arg = b.zext(rewriter.getI32Type(), arg);
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      arg = b.fpext(rewriter.getF64Type(), arg);
    }
    operands.push_back(arg);
  }
  b.call(func, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  // The value overload emits the variadic printf call.
  llvm::SmallString<64> format(msg);
  format.push_back('\n');
  format.push_back('\0');
  Value formatPtr =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", format);
  printf(rewriter, formatPtr, format.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  std::string diagnostic =
      (llvm::Twine("device assertion failed: '") + message + "', in " + func +
       " at " + file + ":" + llvm::Twine(line))
          .str();
  // The diagnostic is a literal, not a user-supplied printf format string.
  // Escape percent signs before passing it to IFWA's printf lowering.
  llvm::SmallString<256> format;
  for (char c : diagnostic) {
    format.push_back(c);
    if (c == '%')
      format.push_back('%');
  }
  printf(rewriter, format, {});
  // Preserve the existing trap. Host error propagation depends on IFWA's
  // runtime and is independent of emitting the diagnostic above.
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.trap", {}, {});
}

} // namespace mlir::triton::Intellifusion
