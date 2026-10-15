#ifndef ILUVATAR_TLE_UTILS_RAW_MATERIALIZE_H_
#define ILUVATAR_TLE_UTILS_RAW_MATERIALIZE_H_

#include "IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

class TritonOpBuilder;

namespace mlir::triton::iluvatar_tle::raw {

OwningOpRef<ModuleOp> parseLLVMModule(MLIRContext *context,
                                      llvm::StringRef text);

LLVM::LLVMFuncOp findExternalLLVMFunc(ModuleOp module,
                                      std::optional<llvm::StringRef> name);

FailureOr<LLVM::LLVMFuncOp>
cloneLLVMSymbolsAndLookupFunc(ModuleOp curModule, ModuleOp parsedModule,
                              std::optional<llvm::StringRef> funcName);

LogicalResult buildDSLRegionBodyFromLLVMFunc(TritonOpBuilder &builder,
                                             DSLRegionOp dslRegionOp,
                                             LLVM::LLVMFuncOp funcOp);

LogicalResult materializeDeferredDSLRegion(ModuleOp module, DSLRegionOp op,
                                           llvm::StringRef llvmIr,
                                           llvm::StringRef externFuncName);

} // namespace mlir::triton::iluvatar_tle::raw

#endif
