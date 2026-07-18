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

#include "Driver/GPU/NvtxApi.h"
#include "Driver/GPU/CuptiApi.h"

#include <cstdint>
#include <cstdlib>

namespace proton {

namespace {

// Declare nvtx function params without including the nvtx header
struct RangePushAParams {
  const char *message;
};

} // namespace

namespace nvtx {

void enable() {
  // Get cupti lib path and append it to NVTX_INJECTION64_PATH
  const std::string cuptiLibPath =
      Dispatch<cupti::ExternLibCupti>::getLibPath();
  if (!cuptiLibPath.empty()) {
    setenv("NVTX_INJECTION64_PATH", cuptiLibPath.c_str(), 1);
  }
}

void disable() { unsetenv("NVTX_INJECTION64_PATH"); }

std::string getMessageFromRangePushA(const void *params) {
  if (const auto *p = static_cast<const RangePushAParams *>(params))
    return std::string(p->message ? p->message : "");
  return "";
}

} // namespace nvtx

} // namespace proton
