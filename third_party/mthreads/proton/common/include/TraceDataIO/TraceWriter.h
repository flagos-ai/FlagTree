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

#ifndef PROTON_COMMON_TRACE_WRITER_H_
#define PROTON_COMMON_TRACE_WRITER_H_

#include "CircularLayoutParser.h"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace proton {

struct KernelMetadata {
  std::map<int, std::string> scopeName;
  std::string kernelName;
  std::vector<std::string> callStack;
};

using KernelTrace = std::pair<std::shared_ptr<CircularLayoutParserResult>,
                              std::shared_ptr<KernelMetadata>>;

// StreamTraceWriter handles trace dumping for a single cuda stream.
// If we have multiple stream, simply having a for loop to write to multiple
// files (one for each stream). Other types of per-stream trace writers could
// subclass the StreamTraceWriter such as StreamPerfettoTraceWriter that
// produces a protobuf format trace.
class StreamTraceWriter {
public:
  explicit StreamTraceWriter(const std::vector<KernelTrace> &streamTrace,
                             const std::string &path);

  virtual ~StreamTraceWriter() = default;

  void dump();

  virtual void write(std::ostream &outfile) = 0;

protected:
  const std::string path;
  const std::vector<KernelTrace> &streamTrace;
};

class StreamChromeTraceWriter : public StreamTraceWriter {
public:
  explicit StreamChromeTraceWriter(const std::vector<KernelTrace> &streamTrace,
                                   const std::string &path);

  void write(std::ostream &outfile) override final;

private:
  void writeKernel(nlohmann::json &object, const KernelTrace &kernelTrace,
                   const uint64_t minInitTime);

  const std::vector<std::string> kChromeColor = {"cq_build_passed",
                                                 "cq_build_failed",
                                                 "thread_state_iowait",
                                                 "thread_state_running",
                                                 "thread_state_runnable",
                                                 "thread_state_unknown",
                                                 "rail_response",
                                                 "rail_idle",
                                                 "rail_load",
                                                 "cq_build_attempt_passed",
                                                 "cq_build_attempt_failed"};
};

} // namespace proton

#endif // PROTON_COMMON_TRACE_WRITER_H_
