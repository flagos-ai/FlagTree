// Copyright 2026- Xcoresigma Technology Co., Ltd

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dsa/dialect/include/IR/Dialect.h"

namespace mlir::triton::tle {

// -- SymmAtOp --
void SymmAtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

// -- ExternCallOp --
void ExternCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

Speculation::Speculatability ExternCallOp::getSpeculatability() {
  if (getPure())
    return Speculation::Speculatable;
  return Speculation::NotSpeculatable;
}

} // namespace mlir::triton::tle
