#ifndef LLVMXPU_DIALECT_LLVMIR_TRANSLATION_INTERFACE_H
#define LLVMXPU_DIALECT_LLVMIR_TRANSLATION_INTERFACE_H

// clang-format off
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"
#include "XPUToLLVMTranslationForSDNN.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Target/LLVMXPU/LLVMXPUToLLVMIRTranslation.h"
// clang-format on

using namespace mlir;
using namespace mlir::LLVM;

// Name-based intrinsic emission (public-LLVM22 frontend compatibility):
// the lowering tables call createIntrinsicCallByName(moduleTranslation,
// builder, "llvm.xpu.*", args[, retTy]) instead of referencing XTDK-private
// enum IDs.  The declaration comes from the gate
// (`intrinsic_attr_table::getOrCreateLLVMFunction`, the LLVM IR half of the one
// that creates MLIR declarations) and the signature is derived from the
// arguments; the LLVM 19 backend resolves the name from its own tables.
// The module is taken from moduleTranslation (not
// builder.GetInsertBlock()): in the SDNN translation context the builder may
// have no insertion point set, and dereferencing a null block crashes the
// simulator-embedded compiler (silent _Exit / hang).
namespace mlir {
namespace LLVM {
namespace detail {
inline llvm::CallInst *
createIntrinsicCallByName(LLVM::ModuleTranslation &moduleTranslation,
                          llvm::IRBuilderBase &builder, llvm::StringRef name,
                          llvm::ArrayRef<llvm::Value *> args = {},
                          llvm::Type *retTy = nullptr) {
  // NOTE(convergent-audit, 2026-09-08): deliberately does NOT set
  // `convergent` on the declared function.  t36 emits these declares without
  // convergent (verified on-hardware, see KB 3.3-bis), XTDK19's own
  // IntrinsicsXPU.td has no IntrConvergent on the sync primitives, and llc19
  // produces identical final asm with or without the attribute.  Adding it
  // here would diverge from the t36 baseline for no correctness gain.
  llvm::Module &module = *moduleTranslation.getLLVMModule();
  llvm::SmallVector<llvm::Type *, 4> argTys;
  for (auto *a : args)
    argTys.push_back(a->getType());
  llvm::Type *resultTy =
      retTy ? retTy : llvm::Type::getVoidTy(module.getContext());
  // The declaration comes from the gate, not from a `Function::Create` here.
  // Same rule as the MLIR half (`getOrCreateDeclaration`): reuse the
  // declaration the module already has, otherwise create it -- and read this
  // process's table *at creation*, so the in-process optimizer never sees a
  // private intrinsic without its facts.  (The public LLVM 22 this leg links
  // has no table of its own for these names; the fallback leg's LLVM 22 has,
  // and `applyTo` leaves it alone.)
  llvm::Function *fn = mlir::intrinsic_attr_table::getOrCreateLLVMFunction(
      module, name,
      llvm::FunctionType::get(resultTy, argTys,
                              /*isVarArg=*/false));
  return builder.CreateCall(fn, args);
}
} // namespace detail
} // namespace LLVM
} // namespace mlir

using mlir::LLVM::detail::createIntrinsicCall;
using mlir::LLVM::detail::createIntrinsicCallByName;

/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMXPU dialect to LLVM IR.
class LLVMXPUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
    if (SDNNConvertOperation(opInst, builder, moduleTranslation).succeeded())
      return success();

#include "triton/Dialect/LLVMXPU/IR/LLVMXPUConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    auto generateMetadata = [&](int dim, StringRef name) {
      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmContext, name),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmContext), dim))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmContext, llvmMetadata);
      moduleTranslation.getOrInsertNamedModuleMetadata("xpu.annotations")
          ->addOperand(llvmMetadataNode);
    };

    return success();
  }
};

#endif
