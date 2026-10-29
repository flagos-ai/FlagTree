//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton-shared/Utils/MemorySpaceUtils.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::xsmt;

//===----------------------------------------------------------------------===//
// XSMTDialect
//===----------------------------------------------------------------------===//

void mlir::xsmt::XSMTDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/XSMT/IR/XSMTAttrs.cpp.inc"
      >();
}

void mlir::xsmt::XSMTDialect::initialize() {
  registerTypes();
  registerAttrs();

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.cpp.inc"
      >();
  addInterfaces<mlir::triton::TritonInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr verification
//===----------------------------------------------------------------------===//

LogicalResult
MemorySpaceAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        llvm::StringRef value) {
  if (mlir::triton::isSupportedMemorySpace(value))
    return success();

  return emitError() << "unsupported xsmt memory space \"" << value
                     << "\"; expected one of: \"global\", \"thread_local\", "
                        "\"cluster_shared_l2\", \"fragment\"";
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttrInterface implementations for MemorySpaceAttr
//===----------------------------------------------------------------------===//

bool MemorySpaceAttr::isValidLoad(
    Type pointeeType, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const DataLayout *layout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool MemorySpaceAttr::isValidStore(
    Type pointeeType, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const DataLayout *layout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool MemorySpaceAttr::isValidAtomicOp(
    ptr::AtomicBinOp op, Type pointeeType, ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const DataLayout *layout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool MemorySpaceAttr::isValidAtomicXchg(
    Type pointeeType, ptr::AtomicOrdering successOrdering,
    ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const DataLayout *layout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool MemorySpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src, function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool MemorySpaceAttr::isValidPtrIntCast(
    Type intType, Type ptrType,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

//===----------------------------------------------------------------------===//
// TableGen'd definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTAttrs.cpp.inc"

#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.cpp.inc"

#include "triton-shared/Dialect/XSMT/IR/XSMTOps.cpp.inc"
