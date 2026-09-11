## 版权声明标注规则

### 1. 非代码文件

#### 1.1 Dockerfile

通常在文件名中含 Dockerfile，但可能有遗漏需要人工标注。

规则：文件头部使用标注方案 A

#### 1.2 Github workflow & actions

满足以下任一条件属于本分类：

- `.github/**/*.yml`，但 `.github/dependabot.yml` 除外

规则：文件头部使用标注方案 A

#### 1.3 其他非代码文件

不属于上述任意一小节，且满足以下任一条件属于本分类（文档、脚本类文件）：

- `*.txt`（包括 CMakeLists\.txt）、`*.md`、`*.rst`、`*.png`、`*.jpg`、`*.yml`、`*.json`、`*.cmake`、`*.html`、`*.sh`、`*.bash`、`CODEOWNERS`

- `docs/**`、`documents/**`、`skills/**`、`packaging/**`、`reports/**`、`scripts/**`

- `.clang-format`、`.dockerignore`、`.editorconfig`、`.git-blame-ignore-revs`、`.gitattributes`、`.gitignore`、`.pre-commit-config.yaml`

- LICENSE（单独标注）

- 无扩展名后缀文件

规则：不修改文件，不做任何标注

### OpenAI 单独维护的测试或工具文件

满足以下任一条件属于本分类（仅 OpenAI 修改和维护）：

- `*.in`

- `test/**`、`unittest/**`、`utils/**`

规则：不修改文件，不做任何标注

### 2. python 文件

#### 2.1 教程、测试类文件

满足以下任一条件属于本分类：

- `python/examples/**/*.py`

- `python/test/**/*.py`

- `python/tutorials/**/*.py`

- `third_party` 中的上述目录文件

规则：不修改文件，不做任何标注

#### 2.2 FlagTree 新增的文件

满足以下任一条件属于本分类：

- 安装类文件 `python/setup_tools/**/*.py`
- 多后端管理基础文件 `python/triton/_flagtree_backend.py`、`python/triton/_flagtree_spec.py`

规则：文件头部使用标注方案 A

#### 2.3 核心代码文件

不属于上述教程、测试类文件，且满足以下任一条件属于本分类：

- `*.py`

主目录规则（`python/triton/experimental/tle` 除外）：文件头部使用标注方案 B

`python/triton/experimental/tle` 目录规则：文件头部使用标注方案 A

`third_party` 目录规则：仅对 `third_party/tle` 目录中的文件使用标注方案 A

### 3. C/C++、MLIR 文件

#### 3.1 测试类文件

满足以下任一条件属于本分类：

- `python/test/**/*.c`

规则：不修改文件，不做任何标注

#### 3.2 .mlir 文件

- `*.mlir`

规则：不修改文件，不做任何标注

`third_party` 目录规则：仅对 `third_party/tle` 目录中的文件使用标注方案 E

#### 3.3 核心代码文件

不属于上述测试类文件，且满足以下任一条件属于本分类：

- `*.td、*.h、*.hpp、*.cpp、*.cc、*.c`

主目录规则：文件头部使用标注方案 D

`third_party` 目录规则：仅对 `third_party/tle` 目录中的文件使用标注方案 C

### 4. 单独标注其他协议的文件

- OpenAI 已在头部标注版权声明的文件

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

头部已有 OpenAI 标注的版权声明，在原版权声明后添加 `Copyright 2025-     FlagOS Contributors`。

- `python/tutorials/tle/01-fft.py`

`fft_kernel_cutile` 是从 NVIDIA 拷贝过来用于对比的 baseline 实现，因此对该代码块标注了

```Python
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
```

规则：不修改该文件

- `utils/generate-test-checks.py`

OpenAI 对该文件单独标注

```Python
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

规则：不修改该文件

### 5. 标注方案

这里统一展示上文提到的各类标注方案

#### 5.1 标注方案 A

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

#### 5.2 标注方案 B

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

#### 5.3 标注方案 C

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

#### 5.4 标注方案 D

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

#### 5.5 标注方案 E

```C++
// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
