//===- XSMTAttrs.h - XSMT dialect attrs -------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef XSMT_ATTRS_H
#define XSMT_ATTRS_H

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTAttrs.h.inc"

#endif // XSMT_ATTRS_H
