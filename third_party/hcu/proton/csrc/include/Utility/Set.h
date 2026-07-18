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

#ifndef PROTON_UTILITY_SET_H_
#define PROTON_UTILITY_SET_H_

#include <set>
#include <shared_mutex>

namespace proton {

/// A simple thread safe set with read/write lock.
template <typename Key, typename Container = std::set<Key>>
class ThreadSafeSet {
public:
  ThreadSafeSet() = default;

  void insert(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    set.insert(key);
  }

  bool contain(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = set.find(key);
    if (it == set.end())
      return false;
    return true;
  }

  bool erase(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return set.erase(key) > 0;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    set.clear();
  }

private:
  Container set;
  std::shared_mutex mutex;
};

} // namespace proton

#endif // PROTON_UTILITY_MAP_H_
