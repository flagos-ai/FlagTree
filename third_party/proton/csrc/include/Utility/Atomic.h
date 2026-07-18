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

#ifndef PROTON_UTILITY_ATOMIC_H_
#define PROTON_UTILITY_ATOMIC_H_

#include <atomic>
#include <mutex>

namespace proton {

template <typename T> T atomicMax(std::atomic<T> &target, T value) {
  T current = target.load();
  while (current < value && !target.compare_exchange_weak(current, value))
    ;
  return current;
}

template <typename T> T atomicMin(std::atomic<T> &target, T value) {
  T current = target.load();
  while (current > value && !target.compare_exchange_weak(current, value))
    ;
  return current;
}

template <typename Condition, typename Function>
void doubleCheckedLock(Condition enterCondition, std::mutex &lock,
                       Function function) {
  if (!enterCondition())
    return;

  std::unique_lock<std::mutex> guard(lock);

  if (!enterCondition())
    return;

  function();
}

} // namespace proton

#endif // PROTON_UTILITY_ATOMIC_H_
