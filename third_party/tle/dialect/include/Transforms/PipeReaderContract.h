/*
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

#ifndef TLE_PIPE_READER_CONTRACT_H
#define TLE_PIPE_READER_CONTRACT_H

// Single source of truth for the tle.pipe reader-declaration contract shared by
// every pipe-lowering backend (NVWS, RDNA, ...). Keeping the reader-name
// validation here guarantees the backends emit the SAME diagnostics and
// identify readers the SAME way (by logical reader name, not by physical
// region), so the empty barrier is sized by the sum of the DISTINCT declared
// readers on every target.

#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::triton::tle {

// The names a pipe declares via its optional `readers` array attribute. Empty
// when the pipe was created without an explicit reader list.
inline SmallVector<std::string> getDeclaredReaderNames(PipeCreateOp op) {
  SmallVector<std::string> readerNames;
  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    readerNames.reserve(readersAttr.size());
    for (Attribute attr : readersAttr)
      readerNames.push_back(cast<StringAttr>(attr).getValue().str());
  }
  return readerNames;
}

inline bool hasDeclaredReader(ArrayRef<std::string> readerNames,
                              StringRef readerName) {
  return llvm::any_of(readerNames, [&](const std::string &declared) {
    return StringRef(declared) == readerName;
  });
}

// Resolve and validate a reader op's logical reader name against the pipe's
// declared readers, emitting the reader-declaration diagnostics on violation.
// Returns the (possibly empty, for an unnamed pipe) reader name on success.
inline FailureOr<std::string>
getPipeReaderName(ArrayRef<std::string> readerNames, Operation *op) {
  std::string readerName;
  if (auto attr = op->getAttrOfType<StringAttr>("reader_name"))
    readerName = attr.getValue().str();

  if (readerNames.empty()) {
    if (!readerName.empty()) {
      op->emitOpError("uses named reader ")
          << readerName << " but pipe was created without readers";
      return failure();
    }
    return readerName;
  }

  if (readerName.empty()) {
    op->emitOpError("requires reader_name because pipe was created "
                    "with explicit readers");
    return failure();
  }
  if (!hasDeclaredReader(readerNames, readerName)) {
    op->emitOpError("uses undeclared pipe reader ") << readerName;
    return failure();
  }
  return readerName;
}

} // namespace mlir::triton::tle

#endif // TLE_PIPE_READER_CONTRACT_H
