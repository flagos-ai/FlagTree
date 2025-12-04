//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file declares the implementation of the BufferizableOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef _MK_DIALECT_BUFFERIZABLEOPINTERFACEIMPL_H
#define _MK_DIALECT_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace mk {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace mk
} // namespace mlir

#endif // _MK_DIALECT_BUFFERIZABLEOPINTERFACEIMPL_H
