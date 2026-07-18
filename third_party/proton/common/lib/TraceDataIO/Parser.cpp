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

#include "TraceDataIO/Parser.h"

using namespace proton;

ParserException::ParserException(const std::string &msg, ExceptionSeverity sev)
    : std::runtime_error(msg), severity(sev) {}

ParserBase::ParserBase(ByteSpan &buffer, const ParserConfig &config)
    : buffer(buffer), config(config) {}

void ParserBase::reportException(const ParserException &e, size_t pos) {

  if (e.severity == ExceptionSeverity::ERROR ||
      config.printLevel == ParserConfig::PrintMode::ALL) {
    std::cerr << "ParserException [offset=" << pos << "]: " << e.what()
              << std::endl;
  }

  if (e.severity == ExceptionSeverity::WARNING)
    return;

  throw e;
}

const ParserConfig &ParserBase::getConfig() const { return config; }
