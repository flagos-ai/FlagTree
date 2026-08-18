//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton-shared/Transform/AddLLVMDebugInfo/AddLLVMDebugInfo.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "add-llvm-debug-info"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_ADDLLVMDEBUGINFO
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h.inc"
} // namespace mlir::triton

namespace {

class AddLLVMDebugInfoPass
    : public triton::impl::AddLLVMDebugInfoBase<AddLLVMDebugInfoPass> {

  static LLVM::DISubprogramFlags setSubprogramFlags(FuncOp funcOp) {
    LLVM::DISubprogramFlags subprogramFlags = LLVM::DISubprogramFlags{};

    if (!funcOp.isDeclaration()) {
      subprogramFlags = LLVM::DISubprogramFlags::Definition;
    }
    if (funcOp->hasAttr("llvm.optimized")) {
      subprogramFlags = subprogramFlags | LLVM::DISubprogramFlags::Optimized;
    }
    if (funcOp.getSymNameAttr() == "main") {
      subprogramFlags =
          subprogramFlags | LLVM::DISubprogramFlags::MainSubprogram;
    }

    return subprogramFlags;
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    Builder builder(context);
    SymbolTable symbolTable(moduleOp);

    // iterate through all functions
    moduleOp.walk([&](FuncOp funcOp) {
      StringRef fileName;
      StringRef filePath;
      unsigned line;
      unsigned col;
      bool isOptimized = funcOp->hasAttr("llvm.optimized");
      LLVM::DIEmissionKind emissionKind = LLVM::DIEmissionKind::LineTablesOnly;

      // get loc for function and pull line and file information
      Location loc = funcOp.getLoc();
      if (auto funcLoc = dyn_cast<FileLineColLoc>(loc)) {
        fileName = llvm::sys::path::filename(funcLoc.getFilename().getValue());
        filePath =
            llvm::sys::path::parent_path(funcLoc.getFilename().getValue());
        line = funcLoc.getLine();
        col = funcLoc.getColumn();
      } else {
        // the triton frontend should always provide a FileLineColLoc for the
        // kernel if this loc is a different type, error out
        moduleOp->emitError("invalid #loc attributes for pass ")
            << this->getName().str();
        return signalPassFailure();
      }

      // initialize useful attributes
      LLVM::DIFileAttr fileAttr =
          LLVM::DIFileAttr::get(context, fileName, filePath);
      StringAttr producer = StringAttr::get(context, "MLIR");
      LLVM::DICompileUnitAttr cuAttr = LLVM::DICompileUnitAttr::get(
          DistinctAttr::create(UnitAttr::get(context)),
          llvm::dwarf::getLanguage("DW_LANG_Python"), fileAttr, producer,
          isOptimized, emissionKind);

      // get subroutine types
      llvm::SmallVector<LLVM::DITypeAttr> types;

      // create subroutine type attribute from return and argument types
      unsigned callingConvention = llvm::dwarf::DW_CC_normal;
      LLVM::DISubroutineTypeAttr type =
          LLVM::DISubroutineTypeAttr::get(context, callingConvention, types);

      // set flags
      LLVM::DISubprogramFlags subprogramFlags = setSubprogramFlags(funcOp);

      // retained nodes
      llvm::ArrayRef<LLVM::DINodeAttr> importedModules;

      // annotations
      llvm::ArrayRef<LLVM::DINodeAttr> annotations;

      // initialize DI attribute for function
      LLVM::DISubprogramAttr spAttr = LLVM::DISubprogramAttr::get(
          context, DistinctAttr::create(UnitAttr::get(context)), cuAttr,
          fileAttr, // scope
          funcOp.getSymNameAttr(),
          funcOp.getSymNameAttr(), // linkage name
          fileAttr, line,
          col, // scope line
          subprogramFlags, type, importedModules, annotations);

      // annotate function
      funcOp->setLoc(builder.getFusedLoc({loc}, spAttr));
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createAddLLVMDebugInfoPass() {
  return std::make_unique<AddLLVMDebugInfoPass>();
}
