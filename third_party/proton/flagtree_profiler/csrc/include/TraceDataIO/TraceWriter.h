#ifndef PROTON_TRACEDATAIO_TRACEWRITER_H_
#define PROTON_TRACEDATAIO_TRACEWRITER_H_

#include "nlohmann/json.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace proton {

struct CycleEntry {
  uint64_t cycle = 0;
  bool isStart = false;
  int scopeId = 0;
};

struct CircularLayoutParserResult {
  struct Trace {
    uint32_t uid = 0;
    std::vector<
        std::pair<std::shared_ptr<CycleEntry>, std::shared_ptr<CycleEntry>>>
        profileEvents;
  };

  struct BlockTrace {
    uint32_t blockId = 0;
    uint32_t procId = 0;
    uint64_t initTime = 0;
    uint64_t preFinalTime = 0;
    uint64_t postFinalTime = 0;
    std::vector<Trace> traces;
  };

  std::vector<BlockTrace> blockTraces;
};

struct KernelMetadata {
  std::string kernelName;
  std::map<int, std::string> scopeName;
  std::vector<std::string> callStack;
};

struct KernelTrace {
  KernelTrace(std::shared_ptr<CircularLayoutParserResult> parserResult,
              std::shared_ptr<KernelMetadata> metadata)
      : parserResult(std::move(parserResult)), metadata(std::move(metadata)) {}

  std::shared_ptr<CircularLayoutParserResult> parserResult;
  std::shared_ptr<KernelMetadata> metadata;
};

inline void
timeShift(int64_t shift,
          const std::shared_ptr<CircularLayoutParserResult> &trace) {
  if (!trace || shift == 0) {
    return;
  }
  for (auto &block : trace->blockTraces) {
    for (auto &unit : block.traces) {
      for (auto &event : unit.profileEvents) {
        if (event.first) {
          event.first->cycle =
              event.first->cycle > static_cast<uint64_t>(shift)
                  ? event.first->cycle - static_cast<uint64_t>(shift)
                  : 0;
        }
        if (event.second) {
          event.second->cycle =
              event.second->cycle > static_cast<uint64_t>(shift)
                  ? event.second->cycle - static_cast<uint64_t>(shift)
                  : 0;
        }
      }
    }
  }
}

class StreamChromeTraceWriter {
public:
  StreamChromeTraceWriter(std::vector<KernelTrace> timeline,
                          std::string processName)
      : timeline(std::move(timeline)), processName(std::move(processName)) {}

  void write(std::ostream &os) const {
    nlohmann::json object = {
        {"displayTimeUnit", "ns"},
        {"traceEvents", nlohmann::json::array()},
    };
    if (!processName.empty()) {
      object["metadata"]["process_name"] = processName;
    }

    for (const auto &kernel : timeline) {
      if (!kernel.parserResult || !kernel.metadata) {
        continue;
      }
      for (const auto &block : kernel.parserResult->blockTraces) {
        for (const auto &unit : block.traces) {
          for (const auto &range : unit.profileEvents) {
            if (!range.first || !range.second) {
              continue;
            }
            const auto start = range.first->cycle;
            const auto end = range.second->cycle;
            nlohmann::json event;
            auto nameIt = kernel.metadata->scopeName.find(range.first->scopeId);
            event["name"] = nameIt == kernel.metadata->scopeName.end()
                                ? kernel.metadata->kernelName
                                : nameIt->second;
            event["cat"] = "cycle";
            event["ph"] = "X";
            event["ts"] = start;
            event["dur"] = end >= start ? end - start : 0;
            event["pid"] = block.procId;
            event["tid"] = unit.uid;
            event["args"]["kernel"] = kernel.metadata->kernelName;
            event["args"]["block_id"] = block.blockId;
            event["args"]["call_stack"] = kernel.metadata->callStack;
            object["traceEvents"].push_back(std::move(event));
          }
        }
      }
    }

    os << object.dump() << "\n";
  }

private:
  std::vector<KernelTrace> timeline;
  std::string processName;
};

} // namespace proton

#endif // PROTON_TRACEDATAIO_TRACEWRITER_H_
