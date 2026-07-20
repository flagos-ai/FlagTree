## Copyright Notice Guidelines

### Non-Code Files

#### Dockerfile

Files whose names contain `Dockerfile` are usually covered by this category, but some may be missed and require manual annotation.

Policy: add a header notice at the top of the file

```YAML
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

#### GitHub Workflow & Actions

A file belongs to this category if it matches any of the following:

- `.github/**/*.yml`, except `.github/dependabot.yml`

Policy: same as Dockerfile

#### Other Non-Code Files

A file belongs to this category if it does not fall under any subsection above and matches any of the following (documentation and script-like files):

- `*.txt` (including `CMakeLists.txt`), `*.md`, `*.rst`, `*.png`, `*.jpg`, `*.yml`, `*.json`, `*.cmake`, `*.html`, `*.sh`, `*.bash`, `CODEOWNERS`

- `docs/**`, `documents/**`, `skills/**`, `packaging/**`, `reports/**`, `scripts/**`

- `.clang-format`, `.dockerignore`, `.editorconfig`, `.git-blame-ignore-revs`, `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml`

- `LICENSE` (annotated separately)

- files with no file extension

Policy: do not modify files; do not add any notice

### Test or Tool Files Maintained Separately by OpenAI

A file belongs to this category if it matches any of the following (maintained and modified by OpenAI only):

- `*.in`

- `test/**`, `unittest/**`, `utils/**`

Policy: do not modify files; do not add any notice

### Python Files

#### Tutorial and Test Files

A file belongs to this category if it matches any of the following:

- `python/examples/**/*.py`

- `python/test/**/*.py`

- `python/tutorials/**/*.py`

- the same directory patterns under `third_party`

Policy: do not modify files; do not add any notice

#### Installation Files

A file belongs to this category if it matches any of the following:

- `python/setup_tools/**/*.py`

Policy: add a header notice at the top of the file

```YAML
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

#### Core Code Files

A file belongs to this category if it is not covered by the tutorial or test categories above and matches any of the following:

- `*.py`

Main repository policy (excluding `python/triton/experimental/tle`): add a header notice at the top of the file

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

`python/triton/experimental/tle` directory policy: add a header notice at the top of the file

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

`third_party` directory policy: annotate only files under `third_party/tle`

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

### C/C++, MLIR Files

#### Test Files

A file belongs to this category if it matches any of the following:

- `python/test/**/*.c`

#### Core Code Files

A file belongs to this category if it is not covered by the test category above and matches any of the following:

- `*.mlir`, `*.td`, `*.h`, `*.hpp`, `*.cpp`, `*.cc`, `*.c`

Main repository policy: add a header notice at the top of the file

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

`third_party` directory policy: annotate only files under `third_party/tle`

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

### Files with Separate License Annotations

- Files whose headers already contain copyright notices annotated by OpenAI

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

These files already contain OpenAI copyright notices in their headers. Leave them unchanged.

- `python/tutorials/tle/01-fft.py`

The `fft_kernel_cutile` block was copied from NVIDIA as a baseline implementation for comparison, so that code block is annotated with:

```Python
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
```

Policy: do not modify this file

- `utils/generate-test-checks.py`

OpenAI annotated this file separately with:

```Python
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

Policy: do not modify this file
