// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_COMMON_PARSER_H_
#define PROTON_COMMON_PARSER_H_

#include "ByteSpan.h"
#include "Device.h"
#include "EntryDecoder.h"
#include <cstdint>
#include <stdexcept>

namespace proton {

struct ParserConfig {
  enum class PrintMode {
    SILENT, // Don't print anything
    ALL     // Print all messages
  };

  // Configure exception message visibility
  PrintMode printLevel = PrintMode::SILENT;

  // Device type that generated the trace
  Device device;

  virtual ~ParserConfig() = default;
};

// Define exception severity levels
enum class ExceptionSeverity {
  WARNING, // Continue parsing
  ERROR    // Stop parsing
};

struct ParserException : public std::runtime_error {
  ExceptionSeverity severity;

  ParserException(const std::string &msg, ExceptionSeverity sev);
};

class ParserBase {
public:
  explicit ParserBase(ByteSpan &buffer, const ParserConfig &config);

  virtual ~ParserBase() = default;

  virtual void parse() = 0;

  virtual const ParserConfig &getConfig() const;

protected:
  void reportException(const ParserException &e, size_t pos);

  const ParserConfig &config;
  ByteSpan &buffer;
};

} // namespace proton

#endif // PROTON_COMMON_PARSER_H_
