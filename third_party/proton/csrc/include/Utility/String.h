#ifndef PROTON_UTILITY_STRING_H_
#define PROTON_UTILITY_STRING_H_

#include <cctype>
#include <string>
#include <vector>

namespace proton {

inline std::string toLower(const std::string &str) {
  std::string lower;
  for (auto c : str) {
    lower += std::tolower(static_cast<unsigned char>(c));
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
  while (start < end && std::isspace(static_cast<unsigned char>(str[start]))) {
    start++;
  }
  while (end > start &&
         std::isspace(static_cast<unsigned char>(str[end - 1]))) {
    end--;
  }
  return str.substr(start, end - start);
}

inline std::vector<std::string> split(const std::string &str,
                                      const std::string &delim) {
  std::vector<std::string> result;
  if (delim.empty()) {
    result.push_back(str);
    return result;
  }
  size_t start = 0;
  while (start <= str.size()) {
    size_t end = str.find(delim, start);
    result.push_back(str.substr(
        start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos)
      break;
    start = end + delim.size();
  }
  return result;
}

} // namespace proton

#endif // PROTON_UTILITY_STRING_H_
