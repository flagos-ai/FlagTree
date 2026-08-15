// ===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: (c) 2024 SpacemiT
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMTASYNC_DIALECT
#define XSMTASYNC_DIALECT

include "mlir/IR/OpBase.td"

def XSMTAsync_Dialect : Dialect {
    let name = "xsmt_async";
    let summary = "A dialect for async.";
    let description = [{
        This dialect contains primitives for async.
    }];
    let cppNamespace = "::mlir::xsmt_async";
}

#endif // XSMTASYNC_DIALECT
