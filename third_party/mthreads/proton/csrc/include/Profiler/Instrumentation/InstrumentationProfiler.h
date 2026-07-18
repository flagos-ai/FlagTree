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

#ifndef PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
#define PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_

#include "Context/Context.h"
#include "Device.h"
#include "Metadata.h"
#include "Profiler/Profiler.h"
#include "Runtime/Runtime.h"
#include "TraceDataIO/Parser.h"
#include "Utility/Singleton.h"

namespace proton {

class InstrumentationProfiler : public Profiler,
                                public InstrumentationInterface,
                                public OpInterface,
                                public Singleton<InstrumentationProfiler> {
public:
  InstrumentationProfiler() = default;
  virtual ~InstrumentationProfiler();

protected:
  // Profiler
  virtual void doStart() override;
  virtual void doFlush() override;
  virtual void doStop() override;
  virtual void
  doSetMode(const std::vector<std::string> &modeAndOptions) override;
  virtual void doAddMetrics(
      size_t scopeId,
      const std::map<std::string, MetricValueType> &scalarMetrics,
      const std::map<std::string, TensorMetric> &tensorMetrics) override;

  // InstrumentationInterface
  void initFunctionMetadata(
      uint64_t functionId, const std::string &functionName,
      const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
      const std::vector<std::pair<size_t, size_t>> &scopeIdParentIds,
      const std::string &metadataPath) override;
  void enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                           uint8_t *buffer, size_t size) override;
  void exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                          uint8_t *buffer, size_t size) override;

  // OpInterface
  void startOp(const Scope &scope) override {
    for (auto data : dataSet) {
      dataToEntryMap.insert_or_assign(data, data->addOp(scope.name));
    }
  }
  void stopOp(const Scope &scope) override { dataToEntryMap.clear(); }

private:
  std::shared_ptr<ParserConfig> getParserConfig(uint64_t functionId,
                                                size_t bufferSize) const;

  Runtime *runtime;
  // device -> deviceStream
  std::map<void *, void *> deviceStreams;
  std::map<std::string, std::string> modeOptions;
  uint8_t *hostBuffer{nullptr};
  // functionId -> scopeId -> scopeName
  std::map<uint64_t, std::map<size_t, std::string>> functionScopeIdNames;
  // functionId -> scopeId -> contexts
  std::map<uint64_t, std::map<size_t, std::vector<Context>>>
      functionScopeIdContexts;
  ;
  // functionId -> functionName
  std::map<uint64_t, std::string> functionNames;
  // functionId -> metadata
  std::map<uint64_t, InstrumentationMetadata> functionMetadata;
  // data -> scopeId
  DataToEntryMap dataToEntryMap;
};

} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_PROFILER_H_
