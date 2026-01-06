//===--------------------- ExportKernelSymbolsPass.cpp
//-----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Ludt) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/ExportKernelSymbols/ExportKernelSymbols.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "export-kernel-symbols"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/ExportKernelSymbols/Passes.h.inc"

namespace {

class ExportKernelSymbolsPass : public ExportKernelSymbolsBase<ExportKernelSymbolsPass> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, tx::Tx81Dialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder builder(&getContext());
        MLIRContext *ctx = &getContext();

        Type ptrType = LLVM::LLVMPointerType::get(ctx);
        Type symtabType = LLVM::LLVMStructType::getLiteral(ctx, {ptrType, ptrType});
        Type i8Type = IntegerType::get(ctx, 8);

        bool changed = false;
        LLVM_DEBUG(llvm::dbgs() << "ExportKernelSymbols: Processing module\n");

        LLVM::GlobalOp lastGlobal = nullptr;
        for (auto global : module.getOps<LLVM::GlobalOp>()) {
            lastGlobal = global;
        }

        auto addFunctionToSections = [&](LLVM::LLVMFuncOp funcOp, StringRef funcName) {
            std::string symbolName = funcName.str();
            Type arrayType = LLVM::LLVMArrayType::get(ctx, i8Type, symbolName.size() + 1);

            SmallVector<char> bytes;
            bytes.append(symbolName.begin(), symbolName.end());
            bytes.push_back(0);

            RankedTensorType tensorType = RankedTensorType::get({static_cast<int64_t>(bytes.size())}, i8Type);
            DenseElementsAttr nameAttr = DenseElementsAttr::get(tensorType, ArrayRef<char>(bytes));

            std::string nameVarName = ("_dynsym_" + funcName + "_name").str();
            if (lastGlobal) {
                builder.setInsertionPointAfter(lastGlobal);
            } else {
                builder.setInsertionPointToStart(module.getBody());
            }

            LLVM::GlobalOp nameVar = builder.create<LLVM::GlobalOp>(module.getLoc(), arrayType, /*isConstant=*/true,
                                                                    LLVM::Linkage::Weak, nameVarName, nameAttr);
            nameVar.setSection(".rodata.name");
            nameVar.setAlignment(1);
            lastGlobal = nameVar;

            std::string symtabVarName = ("_dynsym_" + funcName).str();
            builder.setInsertionPointAfter(nameVar);

            LLVM::GlobalOp symtabVar = builder.create<LLVM::GlobalOp>(
                module.getLoc(), symtabType, /*isConstant=*/true, LLVM::Linkage::External, symtabVarName, Attribute());
            symtabVar.setSection("ExportedDYNSYMTab");
            symtabVar.setAlignment(8);
            lastGlobal = symtabVar;

            Region &initRegion = symtabVar.getInitializerRegion();
            Block *initBlock = builder.createBlock(&initRegion);
            builder.setInsertionPointToStart(initBlock);

            Value funcAddr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, funcOp.getSymNameAttr());
            Value nameAddr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, nameVar.getSymNameAttr());

            Value structVal = builder.create<LLVM::UndefOp>(module.getLoc(), symtabType);
            structVal = builder.create<LLVM::InsertValueOp>(module.getLoc(), symtabType, structVal, funcAddr,
                                                            builder.getDenseI64ArrayAttr({0}));
            structVal = builder.create<LLVM::InsertValueOp>(module.getLoc(), symtabType, structVal, nameAddr,
                                                            builder.getDenseI64ArrayAttr({1}));

            builder.create<LLVM::ReturnOp>(module.getLoc(), structVal);
        };

        for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
            if (funcOp.isDeclaration()) continue;

            StringRef funcName = funcOp.getSymName();
            changed = true;
            addFunctionToSections(funcOp, funcName);
        }

        Type voidType = LLVM::LLVMVoidType::get(ctx);
        Type voidPtrType = LLVM::LLVMPointerType::get(ctx);
        LLVM::LLVMFunctionType funcType = LLVM::LLVMFunctionType::get(voidType, {voidPtrType}, /*isVarArg=*/false);

        auto getOrCreateFunc = [&](StringRef name) -> LLVM::LLVMFuncOp {
            if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) return existing;

            if (lastGlobal) {
                builder.setInsertionPointAfter(lastGlobal);
            } else {
                builder.setInsertionPointToStart(module.getBody());
            }

            auto func = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType, LLVM::Linkage::External,
                                                         /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C);
            changed = true;
            return func;
        };

        LLVM::LLVMFuncOp moduleInitFunc = getOrCreateFunc("module_init");
        builder.setInsertionPointAfter(moduleInitFunc);
        LLVM::LLVMFuncOp moduleCleanupFunc = getOrCreateFunc("module_cleanup");

        addFunctionToSections(moduleInitFunc, "module_init");
        addFunctionToSections(moduleCleanupFunc, "module_cleanup");

        LLVM_DEBUG(llvm::dbgs() << "ExportKernelSymbols: " << (changed ? "Modified" : "No changes") << " module\n");

        if (!changed) markAllAnalysesPreserved();
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createExportKernelSymbolsPass() {
    return std::make_unique<ExportKernelSymbolsPass>();
}
