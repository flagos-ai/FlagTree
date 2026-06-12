[中文版|[English](./install_rpu.md)]

## 💫 Huixi Intelligence（辉羲智能）[rpu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.6.x/third_party/rpu/)

- 对应的 Triton 版本为 3.6，基于 aarch64 平台

**Rhino RPU** 是辉羲智能光至 R1 SoC 内的 AI 加速器
（[rhino.auto](https://www.rhino.auto/)）。与 CPU/GPU 后端不同，RPU 后端**没有预装镜像，也没有
CPU 模拟环境**——只能在物理 R1 SoC 板卡上编译和运行，其驱动、运行时与工具链均需从厂家获取。
请直接在板卡上构建与测试。

> RPU 的驱动、运行时与 LLVM 工具链不公开分发。请联系**辉羲智能**
> （[rhino.auto](https://www.rhino.auto/)）获取。

### 1. 构建及运行环境

#### 1.1 硬件与操作系统

- 一块运行 aarch64 Linux 的 R1 SoC 板卡，且暴露 RPU 设备节点 `/dev/rpu`（内核模块已加载）。
- 最低 24 GB 内存、60 GB 可用磁盘。
- Python 3.10+、`cmake >= 3.20`、`ninja`，以及较新版本的 `pip`。

RPU 后端没有 Docker 镜像，以下所有步骤均在 R1 SoC 板卡上原生执行。确认设备节点：

```shell
ls /dev/rpu
```

#### 1.2 RPU 驱动与运行时（厂家提供）

请按厂家说明安装：

- 提供 `/dev/rpu` 的 RPU 内核驱动；
- `rhino-launch-kernel` 运行时库（`librhino_launch.so`），供板上 launch_kernel 测试使用。

#### 1.3 RPU LLVM 工具链（厂家提供）

RPU 后端使用一套定制 LLVM 作为 `.rpubin` 发射器。将 `RPU_LLVM_ROOT` 指向工具链安装前缀
（即包含 `bin/clang` 的目录）：

```shell
# 工具链目录结构：
#   $RPU_LLVM_ROOT/bin/clang
#   $RPU_LLVM_ROOT/lib/...
export RPU_LLVM_ROOT=/opt/rpu/llvm
```

### 2. 安装命令

RPU 后端在板卡上从源码构建，没有免源码（pip wheel）安装方式。

#### 2.1 拉取源码

```shell
cd ~
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
git checkout -b triton_v3.6.x origin/triton_v3.6.x
```

#### 2.2 从源码构建

```shell
export FLAGTREE_BACKEND=rpu
export MAX_JOBS=8                        # 根据可用内存调整

cd ~/FlagTree/python
pip3 install -r requirements.txt         # 构建期依赖

cd ~/FlagTree
# 首次构建
pip3 install . --no-build-isolation -v
# 修改源码后重新构建
pip3 install . --no-build-isolation --force-reinstall -v
```

Triton 的 MLIR LLVM 会在首次执行 setup 时自动从公开的 oaitriton blob 下载，无需手动操作。

### 3. 测试验证

#### 3.1 单元测试

编译测试会驱动真实工具链，因此需先设置 `RPU_LLVM_ROOT`（即包含 `bin/clang` 的目录）。若未设置或
路径不存在，测试会停止并给出明确提示，指明缺失的变量名。

```shell
cd ~/FlagTree
export RPU_LLVM_ROOT=/opt/rpu/llvm
pytest -s third_party/rpu/python/test/unit
```

#### 3.2 板上 launch_kernel 验证

该步骤需要 `launch_kernel_runner` CLI，它是 `rhino-launch-kernel` 运行时库的一层薄封装。通过开启
`RPU_BUILD_LAUNCH_RUNNER`（默认关闭；需要已安装 `rhino-launch-kernel`）随后端一起构建，然后在带
`/dev/rpu` 的板卡上运行冒烟测试：

```shell
TRITON_APPEND_CMAKE_ARGS="-DRPU_BUILD_LAUNCH_RUNNER=ON -DCMAKE_PREFIX_PATH=/path/to/rhino-launch-kernel/install" \
    pip3 install . --no-build-isolation --force-reinstall -v
export RPU_LK_RUNNER=$PWD/third_party/rpu/tools/launch_runner/launch_kernel_runner
python3 third_party/rpu/python/test/board/lk_board_smoke.py --require-board
```

它会编译一个小 kernel，在设备上派发执行，并与 numpy golden 结果比对。

参考 [Tests of rpu backend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.6.x/.github/workflows/rpu3.6-build-and-test.yml)
