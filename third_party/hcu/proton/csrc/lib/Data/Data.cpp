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

#include "Data/Data.h"
#include "Utility/String.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <shared_mutex>

namespace proton {

void Data::dump(const std::string &outputFormat) {
  std::shared_lock<std::shared_mutex> lock(mutex);

  OutputFormat outputFormatEnum = outputFormat.empty()
                                      ? getDefaultOutputFormat()
                                      : parseOutputFormat(outputFormat);

  std::unique_ptr<std::ostream> out;
  if (path.empty() || path == "-") {
    out.reset(new std::ostream(std::cout.rdbuf())); // Redirecting to cout
  } else {
    out.reset(new std::ofstream(
        path + "." +
        outputFormatToString(outputFormatEnum))); // Opening a file for output
  }

  doDump(*out, outputFormatEnum);
}

OutputFormat parseOutputFormat(const std::string &outputFormat) {
  if (toLower(outputFormat) == "hatchet") {
    return OutputFormat::Hatchet;
  } else if (toLower(outputFormat) == "chrome_trace") {
    return OutputFormat::ChromeTrace;
  } else {
    throw std::runtime_error("Unknown output format: " + outputFormat);
  }
}

const std::string outputFormatToString(OutputFormat outputFormat) {
  if (outputFormat == OutputFormat::Hatchet) {
    return "hatchet";
  } else if (outputFormat == OutputFormat::ChromeTrace) {
    return "chrome_trace";
  }
  throw std::runtime_error("Unknown output format: " +
                           std::to_string(static_cast<int>(outputFormat)));
}

} // namespace proton
