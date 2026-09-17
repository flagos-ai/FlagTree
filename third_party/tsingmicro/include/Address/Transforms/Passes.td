//===- StandalonePsss.td - Standalone dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ADDRESS_PASSES
#define MLIR_DIALECT_ADDRESS_PASSES

include "mlir/Pass/PassBase.td"

def AddrBarePtrConvention : Pass<"addr-apply-bare-ptr"> {
  let summary = "Applies the bare pointer convention";
  let description = [{
    This pass applies the bare pointer convention, transforming `memref`s with
    static shape in function parameters, results, and in call arguments to
    `address` values, reconstructing the `memref` appropriately.
    ```mlir
    func.func @bar(%arg : memref<i32>) -> memref<i32> {
      %memref = func.call @foo(%arg) : (memref<i32>) -> memref<i32>
      return %memref : memref<i32>
    }
    // Gets transformed to:
    func.func @bar(%arg0: !addr.address) -> !addr.address {
      %0 = addr.to_memref %arg0  : memref<i32>
      %1 = addr.from_memref [%0  : memref<i32>]
      %2 = call @foo(%1) : (!addr.address) -> !addr.address
      %3 = addr.to_memref %2  : memref<i32>
      %4 = addr.from_memref [%3  : memref<i32>]
      return %4 : !addr.address
    }

    ```
  }];
}

def AddrToLLVM : Pass<"addr-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Convert from the Address dialect to the LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // MLIR_DIALECT_ADDRESS_PASSES
