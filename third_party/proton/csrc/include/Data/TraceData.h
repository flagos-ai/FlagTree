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

#ifndef PROTON_DATA_TRACE_DATA_H_
#define PROTON_DATA_TRACE_DATA_H_

#include "Data.h"
#include <memory>
#include <unordered_map>

namespace proton {

class TraceData : public Data {
public:
  TraceData(const std::string &path, ContextSource *contextSource = nullptr);
  virtual ~TraceData();

  size_t addOp(size_t scopeId, const std::string &name) override;

  size_t addOp(size_t scopeId, const std::vector<Context> &contexts) override;

  void addMetric(size_t scopeId, std::shared_ptr<Metric> metric) override;

  void
  addMetrics(size_t scopeId,
             const std::map<std::string, MetricValueType> &metrics) override;

  void clear() override;

  class Trace;

protected:
  // ScopeInterface
  void enterScope(const Scope &scope) override final;

  void exitScope(const Scope &scope) override final;

private:
  void doDump(std::ostream &os, OutputFormat outputFormat) const override;
  void dumpChromeTrace(std::ostream &os) const;

  OutputFormat getDefaultOutputFormat() const override {
    return OutputFormat::ChromeTrace;
  }

  std::unique_ptr<Trace> trace;
  // ScopeId -> ContextId
  std::unordered_map<size_t, size_t> scopeIdToContextId;
};

} // namespace proton

#endif // PROTON_DATA_TRACE_DATA_H_
