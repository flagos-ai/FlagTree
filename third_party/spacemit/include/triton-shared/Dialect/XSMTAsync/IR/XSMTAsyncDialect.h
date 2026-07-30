// ===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: (c) 2024 SpacemiT
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#define GET_OP_CLASSES
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h.inc"
