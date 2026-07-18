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

#ifndef PROTON_UTILITY_STRING_H_
#define PROTON_UTILITY_STRING_H_

#include <string>
#include <vector>

namespace proton {

inline std::string toLower(const std::string &str) {
  std::string lower;
  for (auto c : str) {
    lower += tolower(c);
  }
  return lower;
}

inline std::string replace(const std::string &str, const std::string &src,
                           const std::string &dst) {
  std::string replaced = str;
  size_t pos = replaced.find(src);
  while (pos != std::string::npos) {
    replaced.replace(pos, src.length(), dst);
    pos += dst.length();
    pos = replaced.find(src, pos);
  }
  return replaced;
}

inline bool endWith(const std::string &str, const std::string &sub) {
  if (str.length() < sub.length()) {
    return false;
  }
  return str.compare(str.length() - sub.length(), sub.length(), sub) == 0;
}

inline std::string trim(const std::string &str) {
  size_t start = 0;
  size_t end = str.length();
  while (start < end && isspace(str[start])) {
    start++;
  }
  while (end > start && isspace(str[end - 1])) {
    end--;
  }
  return str.substr(start, end - start);
}

inline std::vector<std::string> split(const std::string &str,
                                      const std::string &delim) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = str.find(delim);
  while (end != std::string::npos) {
    result.push_back(str.substr(start, end - start));
    start = end + delim.length();
    end = str.find(delim, start);
  }
  result.push_back(str.substr(start, end));
  return result;
}

inline std::string formatFileLineFunction(const std::string &file, int line,
                                          const std::string &function) {
  return file + ":" + std::to_string(line) + "@" + function;
}

} // namespace proton

#endif // PROTON_UTILITY_STRING_H_
