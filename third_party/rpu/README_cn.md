# Flagtree 第三方后端 - RPU 加速器支持

## 概述

Flagtree 第三方后端针对 **Rhino RPU** 加速器，提供 Triton 编译器后端以及测试套件，用于在 RPU 硬件平台上开发和部署 Triton kernel。

Rhino RPU 是**辉羲智能**（Huixi Intelligence，[rhino.auto](https://www.rhino.auto/)）的 AI 加速器；辉羲智能是一家面向具身智能和自动驾驶、研发高性能低功耗智能计算平台与芯片的公司（如已公开发布的光至 R1 SoC）。

> English: [README.md](./README.md)

## 前提条件

- aarch64 Linux 板上具备 RPU 设备节点 (`/dev/rpu`) 与内核驱动
- RPU LLVM 工具链（定制 LLVM，提供 `bin/clang`）
- RPU 启动运行时**库** `rhino-launch-kernel`（提供 `librhino_launch.so`），以及基于它构建的 `launch_kernel_runner` CLI（见「板上 launch_kernel 验证」）
- 最小 24 GB 内存、60 GB 可用磁盘（24 核板上首次干净构建约 18-25 分钟）
- Python 3.10+，`cmake >= 3.20`，`ninja`，较新的 `pip`

## 环境准备

### 1. 拉取源代码

```bash
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
```

### 2. 安装 RPU 驱动与运行时

按厂商说明安装 RPU 内核驱动（提供 `/dev/rpu`）与 `rhino-launch-kernel` 运行时**库**（`librhino_launch.so`，板上 launch_kernel 测试要用）。验证设备节点：

```bash
ls /dev/rpu
```

### 3. 准备 RPU LLVM 工具链

RPU 后端使用一个定制 LLVM 作为 `.rpubin` emitter。将 `RPU_LLVM_ROOT` 指向工具链安装前缀（即包含 `bin/clang` 的目录）：

```bash
# 工具链目录布局：
#   $RPU_LLVM_ROOT/bin/clang
#   $RPU_LLVM_ROOT/lib/...
export RPU_LLVM_ROOT=/opt/rpu/llvm
```

联系厂商获取工具链。

## 构建与安装

### 1. 配置构建环境

```bash
export FLAGTREE_BACKEND=rpu
export MAX_JOBS=8                # 按可用内存调整
```

### 2. 安装 Python 依赖

```bash
cd ~/FlagTree/python
pip3 install -r requirements.txt
```

### 3. 构建并安装

```bash
cd ~/FlagTree

# 首次构建
pip3 install . --no-build-isolation -v

# 源码修改后重新构建
pip3 install . --no-build-isolation --force-reinstall -v
```

Triton MLIR 主 LLVM 在首次 setup 时自动从 oaitriton 公网 blob 下载，无需手动操作。

## 测试验证

### 单元测试

编译测试会调用真实工具链，因此先设好 `RPU_LLVM_ROOT`（即包含 `bin/clang` 的目录）。未设或路径不存在时，测试会以明确信息中止并指出缺哪个变量。

```bash
cd ~/FlagTree
export RPU_LLVM_ROOT=/opt/rpu/llvm
pytest -s third_party/rpu/python/test/unit
```

测试套件覆盖编译流水线与各类算子。

### 板上 launch_kernel 验证

这需要 `launch_kernel_runner` CLI——它是对 `rhino-launch-kernel` 运行时库的薄前端。通过开启 `RPU_BUILD_LAUNCH_RUNNER`（默认 OFF，需要 `rhino-launch-kernel` 安装）让它随后端一起构建，然后在带 `/dev/rpu` 的板子上跑 smoke 测试：

```bash
TRITON_APPEND_CMAKE_ARGS="-DRPU_BUILD_LAUNCH_RUNNER=ON -DCMAKE_PREFIX_PATH=/path/to/rhino-launch-kernel/install" \
    pip3 install . --no-build-isolation --force-reinstall -v
export RPU_LK_RUNNER=$PWD/third_party/rpu/tools/launch_runner/launch_kernel_runner
python3 third_party/rpu/python/test/board/lk_board_smoke.py --require-board
```

它会编译一个小 kernel、在设备上派发、并与 numpy golden 比对。不带 `--require-board` 时，工具链/运行时/设备缺失会优雅跳过。（`tools/launch_runner` 也可独立构建——见[其 README](./tools/launch_runner)。）
