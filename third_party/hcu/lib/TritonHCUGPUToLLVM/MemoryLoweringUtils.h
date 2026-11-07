#ifndef TRITON_HCU_MEMORY_LOWERING_UTILS_H
#define TRITON_HCU_MEMORY_LOWERING_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

// Intrinsic declarations and scalar constants shared by ordinary memory lowering.
// Coordinate proofs here do not authorize memory motion or barrier removal.
namespace mlir::triton::HCU::memory_lowering {
inline LLVM::LLVMFuncOp getOrInsertHCUIntrinsic(ModuleOp module, Location loc,
    ConversionPatternRewriter &rewriter, StringRef name, Type result,
    ArrayRef<Type> args) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(rewriter, loc, name,
      LLVM::LLVMFunctionType::get(result, args, false));
  fn.setConvergent(true);
  fn.setPassthroughAttr(rewriter.getArrayAttr({rewriter.getStringAttr("nounwind")}));
  return fn;
}

inline std::optional<int64_t> getSignedI64Constant(Value v) {
  auto c = v.getDefiningOp<arith::ConstantOp>();
  auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
  if (!a || !a.getValue().isSignedIntN(64))
    return std::nullopt;
  return a.getInt();
}

} // namespace mlir::triton::HCU::memory_lowering
#endif
