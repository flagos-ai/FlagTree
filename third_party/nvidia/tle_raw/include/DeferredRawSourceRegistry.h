// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NV_TLE_RAW_DEFERRED_SOURCE_REGISTRY_H
#define NV_TLE_RAW_DEFERRED_SOURCE_REGISTRY_H

#include "llvm/ADT/StringMap.h"
#include <optional>
#include <string>

namespace mlir::triton::nvidia::tle_raw {

struct DeferredRawSourceEntry {
  std::string sourceId;
  std::string regionDialect;
  std::optional<std::string> externFuncName;
  std::string source;
  std::string llvmIr;
  std::string hint;
};

llvm::StringMap<DeferredRawSourceEntry> &getDeferredRawSourceRegistry();
void clearDeferredRawSourceRegistry();

} // namespace mlir::triton::nvidia::tle_raw

#endif
