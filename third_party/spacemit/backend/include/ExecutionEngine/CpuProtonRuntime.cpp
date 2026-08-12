// CPU Proton Runtime for spine-triton
// Provides proton_record, proton_dump, proton_reset functions for CPU profiling
// Output formats: Console (default), Hatchet (.hatchet), Chrome Trace (.json)
//
// NOTE: Triton kernels run in parallel across multiple threads. Each thread
// (program instance) executes the same kernel code, so we need to track
// profiling records per-thread to correctly pair start/end events.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// RISC-V rdtime frequency (24 MHz for SpacemiT)
constexpr double RDTIME_FREQ_HZ = 24000000.0;

struct ProfileRecord {
  std::string name;
  int64_t start_cycle;
  int64_t end_cycle;
  std::thread::id thread_id;
  bool completed;
};

// Key for active scopes: (thread_id, scope_name)
struct ScopeKey {
  std::thread::id thread_id;
  std::string name;

  bool operator<(const ScopeKey &other) const {
    if (thread_id != other.thread_id) {
      return thread_id < other.thread_id;
    }
    return name < other.name;
  }
};

class CpuProtonProfiler {
public:
  static CpuProtonProfiler &getInstance() {
    static CpuProtonProfiler instance;
    return instance;
  }

  void record(const char *name, int64_t cycle, int32_t is_start) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string scope_name(name);
    std::thread::id tid = std::this_thread::get_id();
    ScopeKey key{tid, scope_name};

    if (is_start) {
      // Start a new scope for this thread
      ProfileRecord rec;
      rec.name = scope_name;
      rec.start_cycle = cycle;
      rec.end_cycle = 0;
      rec.thread_id = tid;
      rec.completed = false;
      active_scopes_[key] = records_.size();
      records_.push_back(rec);
    } else {
      // End an existing scope for this thread
      auto it = active_scopes_.find(key);
      if (it != active_scopes_.end()) {
        size_t idx = it->second;
        records_[idx].end_cycle = cycle;
        records_[idx].completed = true;
        active_scopes_.erase(it);
      }
    }
  }

  void dump() {
    std::lock_guard<std::mutex> lock(mutex_);

    const char *output_path = std::getenv("PROTON_OUTPUT");

    if (output_path == nullptr) {
      dumpConsole();
    } else {
      std::string path(output_path);
      if (path.find(".hatchet") != std::string::npos) {
        dumpHatchet(path);
      } else if (path.find(".json") != std::string::npos) {
        dumpChromeTrace(path);
      } else {
        dumpConsole();
      }
    }
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
    active_scopes_.clear();
  }

private:
  CpuProtonProfiler() = default;
  ~CpuProtonProfiler() {
    // Auto-dump on program exit if there are records
    if (!records_.empty()) {
      dump();
    }
  }
  CpuProtonProfiler(const CpuProtonProfiler &) = delete;
  CpuProtonProfiler &operator=(const CpuProtonProfiler &) = delete;

  // Helper to escape JSON strings
  static std::string escapeJson(const std::string &s) {
    std::ostringstream o;
    for (char c : s) {
      switch (c) {
      case '"':
        o << "\\\"";
        break;
      case '\\':
        o << "\\\\";
        break;
      case '\b':
        o << "\\b";
        break;
      case '\f':
        o << "\\f";
        break;
      case '\n':
        o << "\\n";
        break;
      case '\r':
        o << "\\r";
        break;
      case '\t':
        o << "\\t";
        break;
      default:
        if ('\x00' <= c && c <= '\x1f') {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(c);
        } else {
          o << c;
        }
      }
    }
    return o.str();
  }

  void dumpConsole() {
    // Aggregate statistics per scope name
    std::map<std::string, std::vector<int64_t>> scope_durations;
    int incomplete_count = 0;

    for (const auto &rec : records_) {
      if (rec.completed) {
        int64_t duration = rec.end_cycle - rec.start_cycle;
        scope_durations[rec.name].push_back(duration);
      } else {
        incomplete_count++;
      }
    }

    printf("\n=== CPU Proton Profiling Results (Aggregated) ===\n");
    printf("%-20s %10s %15s %15s %15s %15s %15s\n", "Scope", "Count",
           "Total(cyc)", "Min(cyc)", "Max(cyc)", "Avg(cyc)", "Avg(us)");
    printf(
        "----------------------------------------------------------------------"
        "--------------------------------------\n");

    for (const auto &kv : scope_durations) {
      const std::string &name = kv.first;
      const std::vector<int64_t> &durations = kv.second;

      int64_t total = 0;
      int64_t min_val = durations[0];
      int64_t max_val = durations[0];

      for (int64_t d : durations) {
        total += d;
        if (d < min_val)
          min_val = d;
        if (d > max_val)
          max_val = d;
      }

      int64_t avg = total / static_cast<int64_t>(durations.size());
      double avg_us = static_cast<double>(avg) / RDTIME_FREQ_HZ * 1e6;

      printf("%-20s %10zu %15ld %15ld %15ld %15ld %15.2f\n", name.c_str(),
             durations.size(), total, min_val, max_val, avg, avg_us);
    }

    if (incomplete_count > 0) {
      printf("\nWarning: %d incomplete records (start without end)\n",
             incomplete_count);
    }

    printf("==================================================================="
           "===="
           "=====================================\n");

    // Also print detailed per-thread view if requested via environment variable
    const char *verbose = std::getenv("PROTON_VERBOSE");
    if (verbose && std::string(verbose) == "1") {
      printf("\n=== Detailed Per-Thread Records ===\n");
      printf("%-20s %15s %15s %15s %15s %10s\n", "Scope", "Start", "End",
             "Duration(cyc)", "Duration(us)", "Thread");
      printf("-----------------------------------------------------------------"
             "-----"
             "--------------------------------------\n");

      for (const auto &rec : records_) {
        // Convert thread::id to a printable number
        std::hash<std::thread::id> hasher;
        size_t tid_hash = hasher(rec.thread_id) % 10000; // Last 4 digits

        if (rec.completed) {
          int64_t duration = rec.end_cycle - rec.start_cycle;
          double duration_us =
              static_cast<double>(duration) / RDTIME_FREQ_HZ * 1e6;
          printf("%-20s %15ld %15ld %15ld %15.2f %10zu\n", rec.name.c_str(),
                 rec.start_cycle, rec.end_cycle, duration, duration_us,
                 tid_hash);
        } else {
          printf("%-20s %15ld %15s %15s %15s %10zu\n", rec.name.c_str(),
                 rec.start_cycle, "N/A", "incomplete", "N/A", tid_hash);
        }
      }
      printf("================================================================="
             "======"
             "=====================================\n\n");
    }
  }

  void dumpHatchet(const std::string &path) {
    // Aggregate statistics per scope name
    struct ScopeStats {
      int64_t total_duration = 0;
      int64_t count = 0;
    };
    std::map<std::string, ScopeStats> scope_stats;

    for (const auto &rec : records_) {
      if (rec.completed) {
        int64_t duration = rec.end_cycle - rec.start_cycle;
        scope_stats[rec.name].total_duration += duration;
        scope_stats[rec.name].count++;
      }
    }

    // Build hatchet-compatible JSON
    std::ostringstream json;
    json << "[\n";

    // Root node with children
    json << "  {\n";
    json << "    \"frame\": {\"name\": \"root\", \"type\": \"function\"},\n";
    json << "    \"metrics\": {\n";

    // Calculate total duration for root
    int64_t total_all = 0;
    int64_t count_all = 0;
    for (const auto &kv : scope_stats) {
      total_all += kv.second.total_duration;
      count_all += kv.second.count;
    }
    json << "      \"time (cycles)\": " << total_all << ",\n";
    json << "      \"count\": " << count_all << "\n";
    json << "    },\n";

    // Children nodes (one per scope)
    json << "    \"children\": [\n";
    bool first = true;
    for (const auto &kv : scope_stats) {
      const std::string &name = kv.first;
      const ScopeStats &stats = kv.second;

      if (!first)
        json << ",\n";
      first = false;

      json << "      {\n";
      json << "        \"frame\": {\"name\": \"" << escapeJson(name)
           << "\", \"type\": \"function\"},\n";
      json << "        \"metrics\": {\n";
      json << "          \"time (cycles)\": " << stats.total_duration << ",\n";
      json << "          \"count\": " << stats.count << ",\n";
      // Convert cycles to nanoseconds (24MHz timebase: 1 cycle = 41.67ns)
      double time_ns =
          static_cast<double>(stats.total_duration) * 1000.0 / 24.0;
      json << "          \"time (ns)\": " << std::fixed << std::setprecision(2)
           << time_ns << ",\n";
      double avg_ns = stats.count > 0 ? time_ns / stats.count : 0;
      json << "          \"avg_time (ns)\": " << std::fixed
           << std::setprecision(2) << avg_ns << "\n";
      json << "        },\n";
      json << "        \"children\": []\n";
      json << "      }";
    }
    json << "\n    ]\n";
    json << "  },\n";

    // Device info
    json << "  {\n";
    json << "    \"CPU\": {\n";
    json << "      \"0\": {\n";
    json << "        \"clock_rate\": " << static_cast<int64_t>(RDTIME_FREQ_HZ)
         << ",\n";
    json << "        \"arch\": \"riscv64\",\n";
    json << "        \"num_sms\": 1\n";
    json << "      }\n";
    json << "    }\n";
    json << "  }\n";
    json << "]\n";

    // Write to file
    std::ofstream file(path);
    if (file.is_open()) {
      file << json.str();
      file.close();
      printf("Proton profile saved to: %s\n", path.c_str());
    } else {
      fprintf(stderr, "Error: Cannot open file %s for writing\n", path.c_str());
      dumpConsole();
    }
  }

  void dumpChromeTrace(const std::string &path) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"traceEvents\": [\n";

    // Find the minimum start time to normalize timestamps
    int64_t min_start = INT64_MAX;
    for (const auto &rec : records_) {
      if (rec.completed && rec.start_cycle < min_start) {
        min_start = rec.start_cycle;
      }
    }

    // Assign stable thread IDs (map thread::id to sequential integers)
    std::map<std::thread::id, int> thread_id_map;
    int next_tid = 1;
    for (const auto &rec : records_) {
      if (thread_id_map.find(rec.thread_id) == thread_id_map.end()) {
        thread_id_map[rec.thread_id] = next_tid++;
      }
    }

    // Write trace events
    bool first = true;
    for (const auto &rec : records_) {
      if (!rec.completed)
        continue;

      if (!first)
        json << ",\n";
      first = false;

      int tid = thread_id_map[rec.thread_id];
      // Convert cycles to microseconds (24MHz timebase: 1 cycle = 1/24 us)
      double start_us = static_cast<double>(rec.start_cycle - min_start) / 24.0;
      double duration_us =
          static_cast<double>(rec.end_cycle - rec.start_cycle) / 24.0;

      // Chrome Trace "X" (complete) event format
      json << "    {" << "\"name\": \"" << escapeJson(rec.name) << "\", "
           << "\"cat\": \"kernel\", " << "\"ph\": \"X\", "
           << "\"ts\": " << std::fixed << std::setprecision(3) << start_us
           << ", "
           << "\"dur\": " << std::fixed << std::setprecision(3) << duration_us
           << ", " << "\"pid\": 1, " << "\"tid\": " << tid << "}";
    }

    json << "\n  ],\n";

    // Add metadata
    json << "  \"displayTimeUnit\": \"us\",\n";
    json << "  \"metadata\": {\n";
    json << "    \"clock_rate_mhz\": 24,\n";
    json << "    \"arch\": \"riscv64\",\n";
    json << "    \"num_threads\": " << thread_id_map.size() << "\n";
    json << "  }\n";
    json << "}\n";

    // Write to file
    std::ofstream file(path);
    if (file.is_open()) {
      file << json.str();
      file.close();
      printf("Chrome Trace saved to: %s\n", path.c_str());
      printf("Open chrome://tracing or https://ui.perfetto.dev and load the "
             "file to visualize.\n");
    } else {
      fprintf(stderr, "Error: Cannot open file %s for writing\n", path.c_str());
      dumpConsole();
    }
  }

  std::mutex mutex_;
  std::vector<ProfileRecord> records_;
  std::map<ScopeKey, size_t> active_scopes_;
};

} // anonymous namespace

extern "C" {

// Main profiling function called from generated LLVM IR
// Parameters:
//   name: scope name (null-terminated string)
//   cycle: current cycle counter value from rdtime
//   is_start: 1 for scope start, 0 for scope end
__attribute__((visibility("default"))) void
proton_record(const char *name, int64_t cycle, int32_t is_start) noexcept {
  CpuProtonProfiler::getInstance().record(name, cycle, is_start);
}

// Dump all profiling results
// Set PROTON_OUTPUT=<filename>.json for Chrome Trace format
// Set PROTON_OUTPUT=<filename>.hatchet for Hatchet format
// Set PROTON_VERBOSE=1 for detailed per-thread output (console only)
__attribute__((visibility("default"))) void proton_dump() noexcept {
  CpuProtonProfiler::getInstance().dump();
}

// Reset all profiling data
__attribute__((visibility("default"))) void proton_reset() noexcept {
  CpuProtonProfiler::getInstance().reset();
}

// ============================================================================
// Kernel-level profiling APIs for automatic kernel capture
// These are called from the launcher's _launch function to automatically
// record kernel execution without modifying user code.
//
// NOTE: These functions are only called when PROTON_KERNEL_CAPTURE=1 is set
// at compile time. The decision is made in driver.py when generating the
// launcher code, so there's no runtime overhead when profiling is disabled.
// ============================================================================

// Get current cycle count (platform-specific)
static inline int64_t get_current_cycle() {
#if defined(__riscv)
  int64_t cycle;
  asm volatile("rdtime %0" : "=r"(cycle));
  return cycle;
#elif defined(__x86_64__) || defined(_M_X64)
  unsigned int lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((int64_t)hi << 32) | lo;
#elif defined(__aarch64__)
  int64_t cycle;
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycle));
  return cycle;
#else
  // Fallback: use clock_gettime
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

// Enter kernel scope - called at the beginning of _launch
// Parameters:
//   kernel_name: name of the kernel function
//   gridX, gridY, gridZ: grid dimensions
__attribute__((visibility("default"))) void
proton_enter_kernel(const char *kernel_name, int gridX, int gridY,
                    int gridZ) noexcept {
  // Create a scope name that includes grid info
  char scope_name[256];
  snprintf(scope_name, sizeof(scope_name), "%s[%d,%d,%d]", kernel_name, gridX,
           gridY, gridZ);

  int64_t cycle = get_current_cycle();
  CpuProtonProfiler::getInstance().record(scope_name, cycle, 1);
}

// Exit kernel scope - called at the end of _launch
// Parameters:
//   kernel_name: name of the kernel function
//   gridX, gridY, gridZ: grid dimensions (must match enter call)
__attribute__((visibility("default"))) void
proton_exit_kernel(const char *kernel_name, int gridX, int gridY,
                   int gridZ) noexcept {
  // Create the same scope name as enter
  char scope_name[256];
  snprintf(scope_name, sizeof(scope_name), "%s[%d,%d,%d]", kernel_name, gridX,
           gridY, gridZ);

  int64_t cycle = get_current_cycle();
  CpuProtonProfiler::getInstance().record(scope_name, cycle, 0);
}

} // extern "C"
