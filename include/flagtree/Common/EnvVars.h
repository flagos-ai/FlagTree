#ifndef FLAGTREE_COMMON_ENVVARS_H
#define FLAGTREE_COMMON_ENVVARS_H

#include <set>
#include <string>

namespace mlir::triton::flagtree {

// Environment variables owned by FlagTree rather than by upstream Triton.
//
// They are declared here instead of in include/triton/Tools/Sys/GetEnv.hpp so
// that upstream rebases of that file do not conflict. python/src/ir.cc merges
// this set into the upstream cache-invalidating set, so changing any of them
// still invalidates the JIT cache, exactly as for an upstream variable.
//
// Keep the Python-side readers in sync (grep for the variable name); the
// backends read these directly with os.getenv.
inline const std::string kDisableLaneVectorize =
    "TRITON_DISABLE_LANE_VECTORIZE";

inline const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    // clang-format off
    kDisableLaneVectorize,
    // clang-format on
};

} // namespace mlir::triton::flagtree

#endif
