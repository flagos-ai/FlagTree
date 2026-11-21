// Copyright 2025- FlagOS Contributors
// SPDX-License-Identifier: MIT

#include "CopyCacheTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <stdexcept>

using namespace mlir;

namespace {
class CopyCacheTranslation : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;
  LogicalResult amendOperation(Operation *op,
                               ArrayRef<llvm::Instruction *> instructions,
                               NamedAttribute attribute,
                               LLVM::ModuleTranslation &) const override {
    if (attribute.getName() != "mccache.policy")
      return op->emitError("unknown MetaX copy cache translation attribute");
    auto mode = dyn_cast<StringAttr>(attribute.getValue());
    if (!mode || (mode != "streaming" && mode != "device"))
      return op->emitError("unknown MetaX copy cache metadata");
    if (instructions.size() != 1 ||
        (!llvm::isa<llvm::LoadInst>(instructions[0]) &&
         !llvm::isa<llvm::StoreInst>(instructions[0])))
      return op->emitError("MetaX cache metadata requires one load or store");
    auto *inst = instructions[0];
    if (mode == "device" && !llvm::isa<llvm::StoreInst>(inst))
      return op->emitError("MetaX device cache metadata requires a store");
    auto &context = inst->getContext();
    auto *one = llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1));
    auto *metadata = llvm::MDNode::get(context, one);
    if (mode == "streaming") {
      inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
      inst->setMetadata("metaxgpu.l2rp", metadata);
    } else {
      inst->setMetadata("metaxgpu.devc", metadata);
    }
    return success();
  }
};

class CopyCacheDialect : public Dialect {
public:
  explicit CopyCacheDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<CopyCacheDialect>()) {
    addInterfaces<CopyCacheTranslation>();
  }
  static StringRef getDialectNamespace() { return "mccache"; }
};
} // namespace

namespace mlir::triton::METAX {
void registerCopyCacheTranslation(DialectRegistry &registry) {
  registry.insert<CopyCacheDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
}

std::pair<std::string, std::string>
splitCopyCacheHelpers(const std::string &source) {
  // The common case retains exactly the pre-existing SDK translation pipeline.
  if (source.find("mccache.helper") == std::string::npos)
    return {source, ""};
  DialectRegistry registry;
  registry.insert<LLVM::LLVMDialect, MACA::MACADialect, scf::SCFDialect>();
  registerCopyCacheTranslation(registry);
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  auto main = parseSourceString<ModuleOp>(source, &context);
  if (!main)
    throw std::runtime_error("Failed to parse MetaX copy cache helper module");
  OwningOpRef<ModuleOp> helpers = ModuleOp::create(main->getLoc());
  for (auto function : main->getOps<LLVM::LLVMFuncOp>()) {
    if (!function->hasAttr("mccache.helper"))
      continue;
    function->removeAttr("mccache.helper");
    helpers->getBody()->push_back(function.clone());
    function.getBody().dropAllReferences();
    function.getBody().getBlocks().clear();
  }
  llvm::LLVMContext llvmContext;
  auto llvmHelpers = translateModuleToLLVMIR(*helpers, llvmContext);
  if (!llvmHelpers)
    throw std::runtime_error("Failed to export MetaX copy cache metadata");
  std::string mainSource, helperSource;
  llvm::raw_string_ostream mainStream(mainSource), helperStream(helperSource);
  main->print(mainStream);
  llvmHelpers->print(helperStream, nullptr);
  return {mainStream.str(), helperStream.str()};
}
} // namespace mlir::triton::METAX
