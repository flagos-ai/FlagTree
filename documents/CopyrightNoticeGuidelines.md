## Copyright Notice Guidelines

### Non-Code Files

#### Dockerfile

Files whose names contain `Dockerfile` generally belong to this category, though some may be missed and require manual annotation.

Rule: use Notice Scheme A at the beginning of the file.

#### GitHub Workflows & Actions

A file belongs to this category if it meets the following condition:

- `.github/**/*.yml`, except `.github/dependabot.yml`

Rule: use Notice Scheme A at the beginning of the file.

#### Other Non-Code Files

A file belongs to this category (documentation and script files) if it does not belong to any subsection above and meets any of the following conditions:

- `*.txt` (including `CMakeLists.txt`), `*.md`, `*.rst`, `*.png`, `*.jpg`, `*.yml`, `*.json`, `*.cmake`, `*.html`, `*.sh`, `*.bash`, `CODEOWNERS`

- `docs/**`, `documents/**`, `skills/**`, `packaging/**`, `reports/**`, `scripts/**`

- `.clang-format`, `.dockerignore`, `.editorconfig`, `.git-blame-ignore-revs`, `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml`

- `LICENSE` (handled separately)

- Files without an extension

Rule: do not modify the file or add any notice.

### Test or Tool Files Maintained Separately by OpenAI

A file belongs to this category (modified and maintained exclusively by OpenAI) if it meets either of the following conditions:

- `*.in`

- `test/**`, `unittest/**`, `utils/**`

Rule: do not modify the file or add any notice.

### Python Files

#### Tutorial and Test Files

A file belongs to this category if it meets any of the following conditions:

- `python/examples/**/*.py`

- `python/test/**/*.py`

- `python/tutorials/**/*.py`

- Files in the corresponding directories under `third_party`

Rule: do not modify the file or add any notice.

#### Files Added by FlagTree

A file belongs to this category if it meets either of the following conditions:

- Installation files: `python/setup_tools/**/*.py`
- Multi-backend management base files: `python/triton/_flagtree_backend.py`, `python/triton/_flagtree_spec.py`

Rule: use Notice Scheme A at the beginning of the file.

#### Core Code Files

A file belongs to this category if it does not belong to the tutorial or test file category above and meets the following condition:

- `*.py`

Main repository rule (except `python/triton/experimental/tle`): use Notice Scheme B at the beginning of the file.

Rule for the `python/triton/experimental/tle` directory: use Notice Scheme A at the beginning of the file.

Rule for the `third_party` directory: use Notice Scheme A only for files under `third_party/tle`.

### C/C++ and MLIR Files

#### Test Files

A file belongs to this category if it meets the following condition:

- `python/test/**/*.c`

#### Core Code Files

A file belongs to this category if it does not belong to the test file category above and meets the following condition:

- `*.mlir`, `*.td`, `*.h`, `*.hpp`, `*.cpp`, `*.cc`, `*.c`

Main repository rule: use Notice Scheme D at the beginning of the file.

Rule for the `third_party` directory: use Notice Scheme C only for files under `third_party/tle`.

### Files with Separate License Notices

- Files with copyright notices already added by OpenAI at the beginning

`include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`
`include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td`
`include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`
`include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h`
`include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td`
`lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
`lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
`lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp`
`python/triton/tools/disasm.py`
`include/triton/Dialect/TritonGPU/Transforms/PipelineExpander.h`
`lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp`

These files already have copyright notices added by OpenAI at the beginning. Leave them unchanged.

- `python/tutorials/tle/01-fft.py`

`fft_kernel_cutile` is a baseline implementation copied from NVIDIA for comparison, so the following notice has been added to that code block:

```Python
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
```

Rule: do not modify this file.

- `utils/generate-test-checks.py`

OpenAI added a separate notice to this file:

```Python
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

Rule: do not modify this file.

### Notice Schemes

The notice schemes referenced above are provided below.

#### Notice Scheme A

```Python
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

```

#### Notice Scheme B

```Python
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

```

#### Notice Scheme C

```C++
/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
```

#### Notice Scheme D

```C++
/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
```
