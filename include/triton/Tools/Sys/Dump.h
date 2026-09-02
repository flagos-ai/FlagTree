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

#ifndef TRITON_TOOLS_SYS_DUMP_H
#define TRITON_TOOLS_SYS_DUMP_H

#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::tools {

inline llvm::raw_fd_ostream &mlirDumps() {
  std::error_code EC;
  static llvm::raw_fd_ostream S(getStrEnv("MLIR_DUMP_PATH"), EC,
                                llvm::sys::fs::CD_CreateAlways);
  assert(!EC && "failed to open MLIR_DUMP_PATH");
  return S;
}

inline llvm::raw_ostream &mlirDumpsOrDbgs() {
  if (!getStrEnv("MLIR_DUMP_PATH").empty())
    return mlirDumps();
  return llvm::dbgs();
}

} // namespace mlir::triton::tools

#endif
