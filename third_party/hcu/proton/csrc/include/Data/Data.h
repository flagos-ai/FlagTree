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

#ifndef PROTON_DATA_DATA_H_
#define PROTON_DATA_DATA_H_

#include "Context/Context.h"
#include "Metric.h"
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>

namespace proton {

enum class OutputFormat { Hatchet, ChromeTrace, Count };

class Data : public ScopeInterface {
public:
  Data(const std::string &path, ContextSource *contextSource = nullptr)
      : path(path), contextSource(contextSource) {}
  virtual ~Data() = default;

  /// Add an op to the data.
  /// If scopeId is already present, add an op under/inside it.
  /// Otherwise obtain the current context and append opName to it if opName is
  /// not empty.
  virtual size_t addOp(size_t scopeId, const std::string &opName = {}) = 0;

  /// Add an op with custom contexts to the data.
  /// This is often used when context source is not available or when
  /// the profiler itself needs to supply the contexts, such as
  /// instruction samples in GPUs whose contexts are
  /// synthesized from the instruction address (no unwinder).
  virtual size_t addOp(size_t scopeId,
                       const std::vector<Context> &contexts) = 0;

  /// Add a single metric to the data.
  virtual void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) = 0;

  /// Add multiple metrics to the data.
  virtual void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) = 0;

  /// Clear all caching data.
  virtual void clear() = 0;

  /// Dump the data to the given output format.
  void dump(const std::string &outputFormat);

  /// Get the contexts associated with the data.
  std::vector<Context> getContexts() const {
    return contextSource->getContexts();
  }

protected:
  /// The actual implementation of the dump operation.
  virtual void doDump(std::ostream &os, OutputFormat outputFormat) const = 0;

  virtual OutputFormat getDefaultOutputFormat() const = 0;

  mutable std::shared_mutex mutex;
  const std::string path{};
  ContextSource *contextSource{};
};

OutputFormat parseOutputFormat(const std::string &outputFormat);

const std::string outputFormatToString(OutputFormat outputFormat);

} // namespace proton

#endif // PROTON_DATA_DATA_H_
