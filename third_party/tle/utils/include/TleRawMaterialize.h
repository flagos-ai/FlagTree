/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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
