// Compatibility header for the XPU backend.
//
// In Triton 3.0/dragon, `i32_val(...)`, `add(...)`, `mul(...)`, etc. were
// provided as free function-style macros that captured the surrounding
// `rewriter` and `loc` variables. In Triton 3.6 (LLVM22), the upstream
// `triton/Conversion/TritonGPUToLLVM/Utility.h` moved those shortcuts onto a
// `mlir::triton::TritonLLVMOpBuilder` struct that must be constructed
// explicitly per call site.
//
// To keep the XPU source files identical to their dragon counterparts (only
// LLVM-API interface changes were requested -- no logic changes), this header
// re-introduces the dragon-style macros. They expand to plain
// `rewriter.create<LLVM::*Op>(loc, ...)` calls, which still work under
// LLVM22's `OpBuilder::create<>` template.
//
// Include this header AFTER the upstream
// `triton/Conversion/TritonGPUToLLVM/Utility.h` so that the dragon-style names
// shadow the new struct method names within XPU translation units only.
//
// NOTE: These macros assume that `rewriter` (an `OpBuilder` / `RewriterBase`)
// and `loc` (a `Location`) are in scope at every use site, exactly as in the
// dragon codebase.

#ifndef TRITON_XPU_CONVERSION_TRITONXPU_TO_LLVM_LEGACY_LLVM_HELPERS_H
#define TRITON_XPU_CONVERSION_TRITONXPU_TO_LLVM_LEGACY_LLVM_HELPERS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

// Operators ------------------------------------------------------------------
#undef inttofloat
#undef inttoptr
#undef ptrtoint
#undef zext
#undef sext
#undef fpext
#undef trunc
#undef udiv
#undef urem
#undef add
#undef sub
#undef fadd
#undef mul
#undef fmul
#undef smax
#undef umax
#undef fmax
#undef smin
#undef umin
#undef fmin
#undef shl
#undef lshr
#undef and_
#undef xor_
#undef or_
#undef bitcast
#undef addrspacecast
#undef gep
#undef insert_val
#undef extract_val
#undef insert_element
#undef extract_element
#undef load
#undef store
#undef fcmp_ogt
#undef fcmp_olt
#undef fcmp_eq
#undef icmp_eq
#undef icmp_ne
#undef icmp_slt
#undef icmp_sle
#undef icmp_sgt
#undef icmp_sge
#undef icmp_ult
#undef icmp_ule
#undef icmp_ugt
#undef icmp_uge
#undef select
#undef address_of
#undef barrier
#undef undef
#undef null
#undef call

#define inttofloat(...) rewriter.create<LLVM::SIToFPOp>(loc, __VA_ARGS__)
#define inttoptr(...) rewriter.create<LLVM::IntToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<LLVM::PtrToIntOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<LLVM::ZExtOp>(loc, __VA_ARGS__)
#define sext(...) rewriter.create<LLVM::SExtOp>(loc, __VA_ARGS__)
#define fpext(...) rewriter.create<LLVM::FPExtOp>(loc, __VA_ARGS__)
#define trunc(...) rewriter.create<LLVM::TruncOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<LLVM::UDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<LLVM::URemOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<LLVM::AddOp>(loc, __VA_ARGS__)
#define sub(...) rewriter.create<LLVM::SubOp>(loc, __VA_ARGS__)
#define fadd(...) rewriter.create<LLVM::FAddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<LLVM::MulOp>(loc, __VA_ARGS__)
#define fmul(...) rewriter.create<LLVM::FMulOp>(loc, __VA_ARGS__)
#define smax(...) rewriter.create<LLVM::SMaxOp>(loc, __VA_ARGS__)
#define umax(...) rewriter.create<LLVM::UMaxOp>(loc, __VA_ARGS__)
#define fmax(...) rewriter.create<LLVM::MaxNumOp>(loc, __VA_ARGS__)
#define smin(...) rewriter.create<LLVM::SMinOp>(loc, __VA_ARGS__)
#define umin(...) rewriter.create<LLVM::UMinOp>(loc, __VA_ARGS__)
#define fmin(...) rewriter.create<LLVM::MinNumOp>(loc, __VA_ARGS__)
#define shl(...) rewriter.create<LLVM::ShlOp>(loc, __VA_ARGS__)
#define lshr(...) rewriter.create<LLVM::LShrOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<LLVM::AndOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<LLVM::XOrOp>(loc, __VA_ARGS__)
#define or_(...) rewriter.create<LLVM::OrOp>(loc, __VA_ARGS__)
#define bitcast(val__, type__)                                                 \
  rewriter.create<LLVM::BitcastOp>(loc, type__, val__)
#define addrspacecast(...)                                                     \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)
#define gep(...) rewriter.create<LLVM::GEPOp>(loc, __VA_ARGS__)
#define insert_val(...) rewriter.create<LLVM::InsertValueOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
#define store(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#define fcmp_ogt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::ogt, lhs, rhs)
#define fcmp_olt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::olt, lhs, rhs)
#define fcmp_eq(lhs, rhs)                                                      \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::oeq, lhs, rhs)
#define icmp_eq(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, __VA_ARGS__)
#define icmp_ne(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, __VA_ARGS__)
#define icmp_slt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, __VA_ARGS__)
#define icmp_sle(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, __VA_ARGS__)
#define icmp_sgt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, __VA_ARGS__)
#define icmp_sge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, __VA_ARGS__)
#define icmp_ult(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, __VA_ARGS__)
#define icmp_ule(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule, __VA_ARGS__)
#define icmp_ugt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, __VA_ARGS__)
#define icmp_uge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, __VA_ARGS__)
#define select(...) rewriter.create<LLVM::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<LLVM::AddressOfOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<LLVM::UndefOp>(loc, __VA_ARGS__)
#define null(...) rewriter.create<LLVM::ZeroOp>(loc, __VA_ARGS__)
#define call(...) rewriter.create<LLVM::CallOp>(loc, __VA_ARGS__)

// Constants ------------------------------------------------------------------
#undef f16_val
#undef f32_val
#undef f64_val
#undef i32_val
#undef i64_val
#undef int_val
#undef tid_val
#undef i16_val

#define f16_val(...) LLVM::createConstantF16(loc, rewriter, __VA_ARGS__)
#define f32_val(...) LLVM::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) LLVM::createConstantF64(loc, rewriter, __VA_ARGS__)
#define i32_val(...) LLVM::createConstantI32(loc, rewriter, __VA_ARGS__)
#define i64_val(...) LLVM::createConstantI64(loc, rewriter, __VA_ARGS__)
#define int_val(width, val)                                                    \
  LLVM::createLLVMIntegerConstant(rewriter, loc, width, val)
#define tid_val() ::mlir::LLVM::XPU::getThreadId(rewriter, loc)
#define i16_val(...)                                                           \
  LLVM::createLLVMIntegerConstant(rewriter, loc, 16, __VA_ARGS__)

// XPU-specific shortcuts (previously in xpu/.../Utility.h) -------------------
#undef addrspace_cast
#undef allocate
#undef idx_val
#undef sdiv
#undef srem
#undef load_sm
#undef store_sm
#undef xpu_barrier

#define addrspace_cast(...)                                                    \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)
#define allocate(...) rewriter.create<LLVM::AllocaOp>(loc, __VA_ARGS__)
#define idx_val(...)                                                           \
  LLVM::createIndexConstant(rewriter, loc, this->getTypeConverter(),           \
                            __VA_ARGS__)
#define sdiv(...) rewriter.create<LLVM::SDivOp>(loc, __VA_ARGS__)
#define srem(...) rewriter.create<LLVM::SRemOp>(loc, __VA_ARGS__)
#define load_sm(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
#define store_sm(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#define xpu_barrier() rewriter.create<mlir::LLVM::XPU::BarrierOp>(loc)

#endif // TRITON_XPU_CONVERSION_TRITONXPU_TO_LLVM_LEGACY_LLVM_HELPERS_H
