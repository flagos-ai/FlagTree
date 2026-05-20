/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "lib/triton_gcu400_core.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string readFile(const char *filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cerr << "Error: cannot open input file: " << filename << "\n";
    return "";
  }
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

static std::string readStdin() {
  std::ostringstream ss;
  ss << std::cin.rdbuf();
  return ss.str();
}

int main(int argc, char **argv) {
  std::vector<const char *> passArgs;
  const char *inputFile = nullptr;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: triton-gcu400-opt [options] <input-file>\n"
                << "  Run MLIR optimization passes on GCU400 IR\n"
                << "  Options are pass names and flags (e.g., --pass-name)\n"
                << "  Use '-' for stdin input\n";
      return 0;
    }
    if (arg[0] != '-') {
      inputFile = argv[i];
    } else {
      passArgs.push_back(argv[i]);
    }
  }

  std::string input;
  if (inputFile && std::string(inputFile) != "-") {
    input = readFile(inputFile);
    if (input.empty()) {
      return 1;
    }
  } else {
    input = readStdin();
  }

  Gcu400String result =
      gcu400_run_opt(input.data(), input.size(), passArgs.data(),
                     static_cast<int>(passArgs.size()));

  if (!result.data) {
    std::cerr << "Error: optimization failed\n";
    return 1;
  }

  std::cout.write(result.data, result.len);
  gcu400_string_free(result);

  return 0;
}
