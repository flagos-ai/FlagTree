#ifndef TRITON_TOOLS_SYS_GETENV_HPP
#define TRITON_TOOLS_SYS_GETENV_HPP

#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>

namespace mlir::triton {

inline const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    // clang-format off
    "AMDGCN_ENABLE_DUMP",
    "AMDGCN_USE_BUFFER_ATOMICS",
    "AMDGCN_USE_BUFFER_OPS",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_MMA_V5",
    "DISABLE_PTXAS_OPT",
    "LLVM_IR_ENABLE_DUMP",
    "LLVM_ENABLE_TIMING",
    "LLVM_PASS_PLUGIN_PATH",
    "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
    "MLIR_ENABLE_DIAGNOSTICS",
    "MLIR_ENABLE_DUMP",
    "MLIR_DUMP_PATH",
    "MLIR_ENABLE_TIMING",
    "MLIR_DISABLE_MULTITHREADING",
    "TRITON_DEFAULT_FP_FUSION",
    "TRITON_DISABLE_ASYNC_COPY",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_DISABLE_LOOP_PEELING",
    "TRITON_DISABLE_WARP_REDUNDANT_REDUCE",
    "TRITON_DISABLE_HOIST_LOADS",
    "TRITON_DUMP_MIR",
    "TRITON_ENABLE_LLVM_DEBUG",
    "TRITON_HIP_USE_ASYNC_COPY",
    "TRITON_HIP_USE_BLOCK_PINGPONG",
    "TRITON_HIP_USE_IN_THREAD_TRANSPOSE",
    "TRITON_LLVM_DEBUG_ONLY",
    "TRITON_ENABLE_ASAN",
    "TRITON_OVERRIDE_ARCH",
    "USE_IR_LOC",
    "NVPTX_ENABLE_DUMP",
    "ALLOW_LHS_TMEM_LAYOUT_CONVERSION",
    "TRITON_F32_DEFAULT",
    "TRITON_PREFER_TMEM_16x256_LAYOUT",
    "TRITON_ENABLE_EXPERIMENTAL_CONSAN",
    "TRITONXPU_HP_MODE",
    "TRITONXPU_BF16_ROUND_MID",
    "TRITONXPU_BF16_FAST",
    "TRITONXPU_FP16_FAST",
    "TRITONXPU_PROBE_BOUNDARY",
    "TRITONXPU_LM_WAR_FENCE",
    "TRITONXPU_LIVE_RANGE",
    "TRITONXPU_PROBE_SIDE",
    "TRITON_PRINT_VERBOSE",
    "TRITON_TUNE_BUFFER_LM_SIZE",
    "TRITON_TLE_STACK_MARGIN",
    "LLVM_ERROR_LM_SIZE",
    "USE_FAST_MATH",
    // clang-format on
};

inline const std::set<std::string> CACHE_NEUTRAL_ENV_VARS = {
    // clang-format off
    "TRITON_REPRODUCER_PATH",
    "TRITON_DISABLE_CRASH_RECOVERY",
    "TRITON_ENABLE_PYTHON_STACKTRACE",
    "TRITONXPU_UNROLL_DRYRUN",
    "TRITONXPU_REDUCE_BOUNDARY",
    "TRITONXPU_REDUCE_REGION",
    "TRITONXPU_REDUCE_VEC",
    "TRITONXPU_SCAN_VEC",
    "TRITONXPU_VEC_COST",
    "TRITONXPU_LM_REPORT",
    "TRITONXPU_TILE_REPORT",
    "TRITONXPU_VEC_REPORT",
    "TRITONXPU_TILE_PLAN",
    "TRITONXPU_TILE_DECIDE",
    "TRITONXPU_VFLOW_REPORT",
    "TRITONXPU_VEC_HYPO_COVER",
    // clang-format on
    "TRITON_DISABLE_RESHAPE_ENCODING_INFERENCE",
    "TRITON_PRINT_HW_ID",
    "TRITON_XCN_DUMP_PREFIX",
    "DISABLE_FAST_REDUCTION",
    "USE_TTGIR_LOC",
};

namespace tools {

inline void assertIsRecognized(const std::string &env) {
  bool is_invalidating = CACHE_INVALIDATING_ENV_VARS.find(env.c_str()) !=
                         CACHE_INVALIDATING_ENV_VARS.end();
  bool is_neutral =
      CACHE_NEUTRAL_ENV_VARS.find(env.c_str()) != CACHE_NEUTRAL_ENV_VARS.end();
  bool is_xpu_allowed_prefix =
      env.rfind("TRITON_", 0) == 0 || env.rfind("TRITONXPU_", 0) == 0 ||
      env.rfind("MLIR_", 0) == 0 || env.rfind("LLVM_", 0) == 0;
  std::string errmsg = env + "is not recognized. "
                             "Please add it to triton/tools/sys/getenv.hpp";
  assert((is_invalidating || is_neutral || is_xpu_allowed_prefix) &&
         errmsg.c_str());
}

static std::mutex getenv_mutex;

inline std::string getStrEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  assertIsRecognized(env);
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}

// return value of a cache-invalidating boolean environment variable
inline bool getBoolEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  assertIsRecognized(env);
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

inline std::optional<bool> isEnvValueBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (str == "on" || str == "true" || str == "1")
    return true;
  if (str == "off" || str == "false" || str == "0")
    return false;
  return std::nullopt;
}

// XPU-private env helpers.
//
// `getStrEnv`/`getBoolEnv` are header-only `inline` functions, so they emit
// weak symbols. At link time the upstream (main-tree) definition wins via weak
// deduplication and it does NOT whitelist the `TRITONXPU_*` vars, which makes
// its `assertIsRecognized` abort at runtime. To keep reading env vars without
// touching the main tree, XPU code calls these distinctly-named helpers which
// allow the `TRITONXPU_` prefix and never assert.
inline std::string getStrEnvXPU(const std::string &env) {
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return "";
  return std::string(cstr);
}

inline bool getBoolEnvXPU(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}
} // namespace tools
} // namespace mlir::triton

#endif
