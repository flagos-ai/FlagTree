#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

namespace simple_logger {

enum LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
public:
  explicit Logger(LogLevel level = INFO) : current_level_(level) {}

  void setLogLevel(LogLevel level) { current_level_ = level; }

  void log(LogLevel level, const char *format, ...) {
    if (level < current_level_) {
      return;
    }

    va_list args;
    va_start(args, format);
    int msg_len = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);

    if (msg_len <= 0) {
      const char *err = "<<format error>>";
      msg_len = static_cast<int>(std::strlen(err));
    }

    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto ms = now_ms.time_since_epoch() % 1000;

    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now_ms);
    std::tm tm{};
    localtime_r(&now_time_t, &tm);

    char time_buf[64];
    std::strftime(time_buf, sizeof(time_buf), "%Y%m%d %H:%M:%S", &tm);
    size_t time_len = std::strlen(time_buf);
    std::snprintf(time_buf + time_len, sizeof(time_buf) - time_len, ".%03d",
                  static_cast<int>(ms.count()));
    time_len += 4;

    const char *level_prefix;
    switch (level) {
    case DEBUG:
      level_prefix = "[DEBUG] ";
      break;
    case INFO:
      level_prefix = "[INFO ] ";
      break;
    case WARN:
      level_prefix = "[WARN ] ";
      break;
    case ERROR:
      level_prefix = "[ERROR] ";
      break;
    default:
      level_prefix = "[?????] ";
    }
    size_t level_len = std::strlen(level_prefix);

    size_t total_add_len =
        1 + time_len + 2 + level_len +
        static_cast<size_t>(
            msg_len > 0 ? msg_len
                        : static_cast<int>(std::strlen("<<format error>>"))) +
        1;

    std::string buffer;
    buffer.resize(total_add_len);
    char *out = &buffer[0];

    char *p = out;
    *p++ = '[';
    std::memcpy(p, time_buf, time_len);
    p += time_len;
    *p++ = ']';

    std::memcpy(p, level_prefix, level_len);
    p += level_len;

    if (msg_len > 0) {
      va_start(args, format);
      std::vsnprintf(p, static_cast<size_t>(msg_len) + 1, format, args);
      va_end(args);
      p += msg_len;
    } else {
      const char *err = "<<format error>>";
      std::memcpy(p, err, std::strlen(err));
      p += std::strlen(err);
    }

    *p++ = '\n';
    printf("%s", buffer.c_str());
  }

private:
  LogLevel current_level_;
};

} // namespace simple_logger
