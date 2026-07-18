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

#ifndef PROTON_UTILITY_ENV_H_
#define PROTON_UTILITY_ENV_H_

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>

namespace proton {

static std::mutex getenv_mutex;

inline int64_t getIntEnv(const std::string &env, int64_t defaultValue) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *s = std::getenv(env.c_str());
  if (s == nullptr)
    return defaultValue;
  return std::stoll(s);
}

inline bool getBoolEnv(const std::string &env, bool defaultValue) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *s = std::getenv(env.c_str());
  if (s == nullptr)
    return defaultValue;
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}

inline std::string getStrEnv(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex);
  const char *s = std::getenv(env.c_str());
  return std::string(s ? s : "");
}

} // namespace proton

#endif // PROTON_UTILITY_ENV_H_
