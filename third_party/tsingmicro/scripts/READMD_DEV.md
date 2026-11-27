# Tsingmicro Triton 开发与构建说明

## 依赖

- LLVM：版本必须与 Triton 匹配（3.6 → LLVM 22，3.7 → LLVM 23）
- tx8_deps
- Python 3.10（3.6）或 3.12（3.7）
- 3.7 配套的运行时为 `torch 2.13.0+cpu` + `torch_txda 0.1.0+20261109` 及以上
- 构建时需要 `ccache`、`lld`

路径解析统一在 `scripts/lib/env.sh`，查找顺序：

1. 环境变量 `LLVM_SYSPATH` / `TX8_DEPS_ROOT` / `KUIPER_ROOT` / `PYTHON`（`KUIPER_ROOT` 默认 `/usr/local/kuiper`）
2. 工作区下的 `llvm-<commit>`、`llvm-<major>`、`tx8_deps`
3. `~/.tsingmicro/`、`~/.triton/tsingmicro/`

脚本会自动校验 LLVM 主版本与 Triton 版本是否一致。

## 源码构建

```bash
# 安装到源码树（开发用）
third_party/tsingmicro/scripts/build_tsingmicro.sh -a install

# 打 wheel
third_party/tsingmicro/scripts/build_tsingmicro.sh -a wheel -t release
```

构建结束后会输出 wheel 的 glibc 需求：

```
[abi] wheel glibc_max=2.34 (triton-3.7.1+git7ea4b5be-cp312-...whl)
```

## 运行与测试

```bash
third_party/tsingmicro/scripts/run_tsingmicro.sh python your_script.py
third_party/tsingmicro/scripts/run_tsingmicro.sh pytest path/to/test.py
```

启动时会自动检查 LLVM 与 triton 扩展所需的 glibc 是否满足本机，不满足直接报错。

## LLVM 构建与打包

```bash
LLVM_PROJECT=/path/to/llvm-project third_party/tsingmicro/scripts/build_llvm.sh
```

产物：

```
$LLVM_PROJECT/build/                构建目录
$LLVM_PROJECT/install/              安装目录（可用 LLVM_INSTALL_PREFIX 覆盖）
$LLVM_PROJECT/llvm-<major>.tar.gz   可分发包
```

打包时会自动：

1. 把绝对软链实体化成真实文件；
2. 校验包内绝对软链数为 0；
3. 扫描所需的最高 glibc 并写入 `.tsm_abi`；
4. 生成 tar 后再抽查一次。

## 用户安装（公共目录）

```bash
pip install triton-*.whl

export LLVM_SYSPATH=/path/to/llvm-23
export TX8_DEPS_ROOT=/path/to/tx8_deps
export PYTHONPATH=$LLVM_SYSPATH/python_packages/mlir_core
export LD_LIBRARY_PATH=$TX8_DEPS_ROOT/lib:/usr/local/kuiper/lib:/usr/local/kuiper/tsm8-profiler/lib
```

`import triton` 会加载 torch_txda，因此还需要 `torch`、`torch_txda`、`txops` 已安装。

## ABI 说明

- LLVM 包内带 `.tsm_abi`，记录 `glibc_max`；
- triton 扩展在运行时扫描 `triton/_C/*.so`，结果缓存到 `~/.cache/tsm_abi/`；
- 两者在启动时与本机 glibc 比较，不满足会明确报错，而不是加载到一半失败。

## 离线依赖

```bash
# 生成离线包
third_party/tsingmicro/scripts/tools/offline_python_deps.sh \
    -p -r requirements_ts.txt -o offline_pkgs

# 从离线包安装
third_party/tsingmicro/scripts/tools/offline_python_deps.sh \
    -i -r requirements_ts.txt -d offline_pkgs
```

需要代理时设置 `PROXY` 环境变量，不设置则直连。
