#ifndef TRITON_AFFINITY_UTILS_HPP
#define TRITON_AFFINITY_UTILS_HPP

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::AffinityDAG {

template <typename T, typename F>
constexpr inline T enumOp(F &&func, T lhs, T rhs) {
  static_assert(std::is_enum_v<T>, "T must be an enum type");

  using U = std::underlying_type_t<T>;

  return static_cast<T>(std::invoke(std::forward<F>(func), static_cast<U>(lhs),
                                    static_cast<U>(rhs)));
}

// since we do not have llvm::set_intersects in this version...
template <class S1Ty, class S2Ty> bool intersects(S1Ty &s1, S2Ty &s2) {
  if (s1.size() > s2.size()) {
    return intersects(s2, s1);
  }

  return llvm::any_of(s1, [&](auto e) { return s2.count(e); });
}

/**
 * @returns the pointer to the value wrapped by the pointer if the key is in the
 * map, otherwise nullptr
 *
 * @safety The user is responsible for checking the nullity of the retured
 * pointer; the lifespan of the pointer is valid as long as the value belongs to
 * the map
 */
template <typename MapTy>
inline auto getFromSmartPtr(MapTy &map, const typename MapTy::key_type &key)
    -> decltype(map.find(key)->second.get()) {
  auto it = map.find(key);
  return it != map.end() ? it->second.get() : nullptr;
}

/**
 * @returns the pointer to the value if the key is in the map, otherwise nullptr
 *
 * @safety The user is responsible for checking the nullity of the retured
 * pointer; the lifespan of the pointer is valid as long as the the map is not
 * modified
 */
template <typename MapTy>
inline auto getPtr(MapTy &map, const typename MapTy::key_type &key)
    -> decltype(map.find(key)->second) * {
  auto it = map.find(key);
  return it != map.end() ? &it->second : nullptr;
}

} // namespace mlir::AffinityDAG

#endif
