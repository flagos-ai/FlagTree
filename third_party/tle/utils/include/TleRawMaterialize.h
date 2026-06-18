#ifndef TLE_UTILS_RAW_MATERIALIZE_H_
#define TLE_UTILS_RAW_MATERIALIZE_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

class TritonOpBuilder;

namespace mlir::triton::tle::raw {

OwningOpRef<ModuleOp> parseLLVMModule(MLIRContext *context,
                                      llvm::StringRef text);

LLVM::LLVMFuncOp findExternalLLVMFunc(ModuleOp module,
                                      std::optional<llvm::StringRef> name);

FailureOr<LLVM::LLVMFuncOp>
cloneLLVMSymbolsAndLookupFunc(ModuleOp curModule, ModuleOp parsedModule,
                              std::optional<llvm::StringRef> funcName);

LogicalResult buildDSLRegionBodyFromLLVMFunc(TritonOpBuilder &builder,
                                             tle::DSLRegionOp dslRegionOp,
                                             LLVM::LLVMFuncOp funcOp);

LogicalResult materializeDeferredDSLRegion(ModuleOp module, tle::DSLRegionOp op,
                                           llvm::StringRef llvmIr,
                                           llvm::StringRef externFuncName);

} // namespace mlir::triton::tle::raw

#endif
