#ifndef PROTON_SESSION_SESSION_H_
#define PROTON_SESSION_SESSION_H_

#include "Context/Context.h"
#include "Data/Artifacts.h"
#include "Data/Metric.h"
#include "Profiler/Vendor/Mode.h"
#include "Utility/Singleton.h"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

namespace proton {

class Profiler;
class Data;
class VendorAdapter;

/// A session is a collection of profiler, context source, and data objects.
/// There could be multiple sessions in the system, each can correspond to a
/// different duration, or the same duration but with different configurations.
class Session {
public:
  ~Session() = default;

  void activate();

  void deactivate();

  void finalize(const std::string &outputFormat);

  size_t getContextDepth();

  Profiler *getProfiler() const { return profiler; }

private:
  Session(size_t id, const std::string &path, Profiler *profiler,
          std::unique_ptr<ContextSource> contextSource,
          std::unique_ptr<Data> data, const std::string &profilerName,
          const std::string &contextSourceName, const std::string &dataName,
          const std::string &mode, const std::string &hookName = {},
          const VendorAdapter *vendorAdapter = nullptr,
          VendorProfilePlan vendorPlan = {},
          std::unique_ptr<Data> timelineData = nullptr)
      : id(id), path(path), profiler(profiler),
        contextSource(std::move(contextSource)), data(std::move(data)),
        profilerName(profilerName), contextSourceName(contextSourceName),
        dataName(dataName), mode(mode), hookName(hookName),
        vendorAdapter(vendorAdapter), vendorPlan(std::move(vendorPlan)),
        timelineData(std::move(timelineData)) {}

  template <typename T> std::vector<T *> getInterfaces() {
    std::vector<T *> interfaces;
    if (auto interface = dynamic_cast<T *>(contextSource.get())) {
      interfaces.push_back(interface);
    }
    if (auto interface = dynamic_cast<T *>(profiler)) {
      interfaces.push_back(interface);
    }
    if (auto interface = dynamic_cast<T *>(data.get())) {
      interfaces.push_back(interface);
    }
    if (timelineData) {
      if (auto interface = dynamic_cast<T *>(timelineData.get())) {
        interfaces.push_back(interface);
      }
    }
    return interfaces;
  }

  const std::string path{};
  size_t id{};
  Profiler *profiler{};
  std::unique_ptr<ContextSource> contextSource{};
  std::unique_ptr<Data> data{};
  std::string profilerName{};
  std::string contextSourceName{};
  std::string dataName{};
  std::string mode{};
  std::string hookName{};
  const VendorAdapter *vendorAdapter{};
  VendorProfilePlan vendorPlan{};
  std::unique_ptr<Data> timelineData{};

  friend class SessionManager;
};

/// A session manager is responsible for managing the lifecycle of sessions.
/// There's a single and unique session manager in the system.
class SessionManager : public Singleton<SessionManager> {
public:
  SessionManager() = default;
  ~SessionManager() = default;

  size_t addSession(const std::string &path, const std::string &profilerName,
                    const std::string &profilerPath,
                    const std::string &contextSourceName,
                    const std::string &dataName, const std::string &mode = {},
                    const std::string &hookName = {});

  void finalizeSession(size_t sessionId, const std::string &outputFormat);

  void finalizeAllSessions(const std::string &outputFormat);

  void activateSession(size_t sesssionId);

  void activateAllSessions();

  void deactivateSession(size_t sessionId);

  void deactivateAllSessions();

  size_t getContextDepth(size_t sessionId);

  void enterScope(const Scope &scope);

  void exitScope(const Scope &scope);

  void enterOp(const Scope &scope);

  void exitOp(const Scope &scope);

  void initFunctionMetadata(
      uint64_t functionId, const std::string &functionName,
      const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
      const std::vector<std::pair<size_t, size_t>> &scopeIdParents,
      const std::string &metadataPath);

  void enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                           uint8_t *buffer, size_t size);

  void exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                          uint8_t *buffer, size_t size);

  void addMetrics(size_t scopeId,
                  const std::map<std::string, MetricValueType> &metrics);

  void setState(std::optional<Context> context);

private:
  std::unique_ptr<Session>
  makeSession(size_t id, const std::string &path,
              const std::string &profilerName, const std::string &profilerPath,
              const std::string &contextSourceName, const std::string &dataName,
              const std::string &mode, const std::string &hookName);

  void activateSessionImpl(size_t sesssionId);

  void deActivateSessionImpl(size_t sessionId);

  size_t getSessionId(const std::string &path) { return sessionPaths[path]; }

  bool hasSession(const std::string &path) {
    return sessionPaths.find(path) != sessionPaths.end();
  }

  bool hasSession(size_t sessionId) {
    return sessions.find(sessionId) != sessions.end();
  }

  void removeSession(size_t sessionId);

  template <typename Interface, typename Counter>
  void registerInterface(size_t sessionId, Counter &interfaceCounts) {
    auto interfaces = sessions[sessionId]->getInterfaces<Interface>();
    for (auto *interface : interfaces) {
      interfaceCounts[interface] += 1;
    }
  }

  template <typename Interface, typename Counter>
  void unregisterInterface(size_t sessionId, Counter &interfaceCounts) {
    auto interfaces = sessions[sessionId]->getInterfaces<Interface>();
    for (auto *interface : interfaces) {
      interfaceCounts[interface] -= 1;
    }
  }

  mutable std::shared_mutex mutex;

  size_t nextSessionId{};
  // path -> session id
  std::map<std::string, size_t> sessionPaths;
  // session id -> active
  std::map<size_t, bool> activeSessions;
  // session id -> session
  std::map<size_t, std::unique_ptr<Session>> sessions;
  // scope -> active count
  std::map<ScopeInterface *, size_t> scopeInterfaceCounts;
  // op -> active count
  std::map<OpInterface *, size_t> opInterfaceCounts;
  std::map<InstrumentationInterface *, size_t> instrumentationInterfaceCounts;
  std::map<ContextSource *, size_t> contextSourceCounts;
};

} // namespace proton

#endif // PROTON_SESSION_H_
