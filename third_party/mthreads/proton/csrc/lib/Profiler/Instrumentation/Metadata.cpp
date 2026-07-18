// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>

#include "Profiler/Instrumentation/Metadata.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace proton {

void InstrumentationMetadata::parse() {
  std::ifstream metadataFile(metadataPath);
  if (!metadataFile.is_open()) {
    throw std::runtime_error("Failed to open metadata file: " + metadataPath);
  }

  json metadataJson;
  metadataFile >> metadataJson;

  if (metadataJson.contains("profile_scratch_size")) {
    scratchMemorySize = metadataJson["profile_scratch_size"].get<size_t>();
  }

  if (metadataJson.contains("num_warps")) {
    numWarps = metadataJson["num_warps"].get<size_t>();
  }
}

} // namespace proton
