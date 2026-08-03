// Copyright (c) 2026 Tsingmicro CORPORATION
//
// Query device kernel timing (AP / kcore) after txLaunch* + txStreamSynchronize.
// Requires libtsm-api-log-tracing.so listed in ROCP_TOOL_LIBRARIES.
// Resolve symbols via dlopen(ROCP_TOOL_LIBRARIES) + dlsym — not dlsym(RTLD_DEFAULT).
//
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TSM_PROF_API_OP_MAX   64
#define TSM_PROF_API_ARGS_MAX 512

typedef struct tsm_prof_timing_t {
  uint64_t token;
  uint64_t corid;
  uint64_t host_duration_nsec;
  uint64_t ap_duration_nsec;
  uint64_t kcore_duration_nsec;
  uint64_t ap_start;
  uint64_t ap_end;
  uint64_t kcore_start;
  uint64_t kcore_end;
  char     api_op[TSM_PROF_API_OP_MAX];
  char     api_args[TSM_PROF_API_ARGS_MAX];
  char     api_result[64];
  int      dev_id;
  int      ready;
} tsm_prof_timing_t;

/* Start a profiling scope; call immediately before the target traced API. Returns 0 on failure. */
uint64_t tsm_prof_begin(void);

/*
 * Query device timing for token. Call after txStreamSynchronize (same thread as tsm_prof_begin).
 * The next traced TSM_RUNTIME_API EXIT after begin binds token to that API's corid.
 * Returns 0 on success, -1 invalid args/unknown token, -2 corid not bound yet, -3 timeout.
 * timeout_ms: 0 = poll once, >0 = wait up to N ms for kernel notify.
 */
int tsm_prof_get_timing(uint64_t token, tsm_prof_timing_t* out, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif
