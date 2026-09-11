// Resolve tsm_prof_begin / tsm_prof_get_timing at runtime via dlopen/dlsym.
// Supports two loading paths:
//   1) LD_PRELOAD          → dlsym(RTLD_DEFAULT)
//   2) ROCP_TOOL_LIBRARIES → dlopen(RTLD_NOLOAD) + dlsym(handle)
//
// Usage:
//   #include "tsm_profiler.h"
//   if (resolve_prof_query_api()) {
//       uint64_t token = resolve_tsm_prof_begin();
//       ...
//       tsm_prof_timing_t t = {};
//       int rc = resolve_tsm_prof_get_timing(token, &t, 5000);
//   }
#pragma once

#include "tsm_prof_query.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>

static void *g_prof_tool_handle = nullptr;
static uint64_t (*resolve_tsm_prof_begin)(void) = nullptr;
static int (*resolve_tsm_prof_get_timing)(uint64_t, tsm_prof_timing_t *,
                                          uint32_t) = nullptr;

inline void *open_prof_tool_from_env() {
  if (g_prof_tool_handle)
    return g_prof_tool_handle;

  const char *env = std::getenv("ROCP_TOOL_LIBRARIES");
  if (!env || env[0] == '\0')
    return nullptr;

  std::string paths{env};
  size_t start = 0;
  while (start < paths.size()) {
    const size_t end = paths.find(':', start);
    const auto path = (end == std::string::npos)
                          ? paths.substr(start)
                          : paths.substr(start, end - start);
    start = (end == std::string::npos) ? paths.size() : end + 1;

    if (path.empty())
      continue;

    void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (!handle)
      handle = dlopen(path.c_str(), RTLD_LAZY);
    if (handle) {
      void *sym = dlsym(handle, "tsm_prof_begin");
      if (sym != nullptr) {
        g_prof_tool_handle = handle;
        return handle;
      }
    }
  }
  return nullptr;
}

inline bool resolve_prof_query_api() {
  void *handle = open_prof_tool_from_env();
  if (!handle)
    return false;

  if (!resolve_tsm_prof_begin)
    resolve_tsm_prof_begin =
        reinterpret_cast<uint64_t (*)(void)>(dlsym(handle, "tsm_prof_begin"));
  if (!resolve_tsm_prof_get_timing)
    resolve_tsm_prof_get_timing =
        reinterpret_cast<int (*)(uint64_t, tsm_prof_timing_t *, uint32_t)>(
            dlsym(handle, "tsm_prof_get_timing"));
  return resolve_tsm_prof_begin && resolve_tsm_prof_get_timing;
}
