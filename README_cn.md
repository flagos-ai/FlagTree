<div align="right"><a href="/README.md">English</a></div>

## <img width="30" height="30" alt="FlagTree-GitHub" src="https://github.com/user-attachments/assets/d8d24c81-6f46-4adc-94e2-b89b03afcb43" /> FlagTree

FlagTree 是面向多种 AI 芯片的开源、统一编译器。FlagTree 致力于打造多元 AI 芯片编译器及相关工具平台，发展和壮大 Triton 上下游生态。项目当前处于初期，目标是兼容现有适配方案，统一代码仓库，快速实现单仓库多后端支持。对于上游模型用户，提供多后端的统一编译能力；对于下游芯片厂商，提供 Triton 生态接入范例。<br>
各后端基于不同版本的 triton 适配，因此位于不同的主干分支，各主干分支均为保护分支且地位相等：<br>

|主干分支|厂商|后端|Triton 版本|
|-------|---|---|-----------|
|[main](https://github.com/flagos-ai/flagtree/tree/main)|NVIDIA<br>AMD<br>x86_64 cpu<br>ILUVATAR（天数智芯）<br>Moore Threads（摩尔线程）<br>KLX<br>MetaX（沐曦股份）<br>HYGON（海光信息）|[nvidia](/third_party/nvidia/)<br>[amd](/third_party/amd/)<br>[triton-shared](https://github.com/microsoft/triton-shared)<br>[iluvatar](/third_party/iluvatar/)<br>[mthreads](/third_party/mthreads/)<br>[xpu](/third_party/xpu/)<br>[metax](/third_party/metax/)<br>[hcu](third_party/hcu/)|3.1<br>3.1<br>3.1<br>3.1<br>3.1<br>3.0<br>3.1<br>3.0|
|[triton_v3.2.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x)|NVIDIA<br>AMD<br>Huawei Ascend（华为昇腾）<br>Cambricon（寒武纪）|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/amd/)<br>[ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/third_party/ascend)<br>[cambricon](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/cambricon/)|3.2|
|[triton_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x)|NVIDIA<br>AMD<br>x86_64 cpu<br>ARM China<br>Tsingmicro（清微智能）<br>Enflame（燧原）|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/amd/)<br>[triton-shared](https://github.com/microsoft/triton-shared)<br>[aipu](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/)<br>[tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)<br>[enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/)|3.3|
|[triton_v3.4.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.4.x)|NVIDIA<br>AMD|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.4.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.4.x/third_party/amd/)|3.4|
|[triton_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x)|NVIDIA<br>AMD|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.5.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.5.x/third_party/amd/)|3.5|

## 新特性

* 2025/12/24 支持拉取和安装 [Wheel](/README_cn.md#非源码安装)。
* 2025/12/08 新增接入 [enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/) 后端（对应 Triton 3.3），加入 CI/CD。
* 2025/11/26 添加 FlagTree 后端特化统一设计文档 [FlagTree_Backend_Specialization](/documents/decoupling/)。
* 2025/10/28 提供离线构建支持（预下载依赖包），改善网络环境受限时的构建体验，使用方法见后文。
* 2025/09/30 在 GPGPU 上支持编译指导 shared memory。
* 2025/09/29 SDK 存储迁移至金山云，大幅提升下载稳定性。
* 2025/09/25 支持编译指导 ascend 的后端编译能力。
* 2025/09/16 新增接入 [hcu](https://github.com/FlagTree/flagtree/tree/main/third_party/hcu/) 后端（对应 Triton 3.0），加入 CI/CD。
* 2025/09/09 Fork 并修改 [llvm-project](https://github.com/FlagTree/llvm-project)，承接 [FLIR](https://github.com/flagos-ai/flir) 的功能。
* 2025/09/01 新增适配 Paddle 框架，加入 CI/CD。
* 2025/08/16 新增适配北京超级云计算中心 AI 智算云。
* 2025/08/04 新增接入 T*** 后端（对应 Triton 3.1）。
* 2025/08/01 [FLIR](https://github.com/flagos-ai/flir) 支持编译指导 shared memory loading。
* 2025/07/30 更新 [cambricon](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/cambricon/) 后端（对应 Triton 3.2）。
* 2025/07/25 浪潮团队新增适配 OpenAnolis 龙蜥操作系统。
* 2025/07/09 [FLIR](https://github.com/flagos-ai/flir) 支持编译指导 Async DMA。
* 2025/07/08 新增多后端编译统一管理模块。
* 2025/07/02 [FlagGems](https://github.com/flagos-ai/FlagGems) LibTuner 适配 triton_v3.3.x 版本。
* 2025/07/02 新增接入 S*** 后端（对应 Triton 3.3）。
* 2025/06/20 [FLIR](https://github.com/flagos-ai/flir) 开始承接 MLIR 扩展功能。
* 2025/06/06 新增接入 [tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/) 后端（对应 Triton 3.3），加入 CI/CD。
* 2025/06/04 新增接入 [ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/third_party/ascend) 后端（对应 Triton 3.2），加入 CI/CD。
* 2025/06/03 新增接入 [metax](https://github.com/FlagTree/flagtree/tree/main/third_party/metax/) 后端（对应 Triton 3.1），加入 CI/CD。
* 2025/05/22 [FlagGems](https://github.com/flagos-ai/FlagGems) LibEntry 适配 triton_v3.3.x 版本。
* 2025/05/21 [FLIR](https://github.com/flagos-ai/flir) 开始承接到中间层的转换功能。
* 2025/04/09 新增接入 [aipu](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/) 后端（对应 Triton 3.3），提供 torch 标准扩展[范例](https://github.com/flagos-ai/flagtree/blob/triton_v3.3.x/third_party/aipu/backend/aipu_torch_dev.cpp)，加入 CI/CD。
* 2025/03/26 接入安全合规扫描。
* 2025/03/19 新增接入 [xpu](https://github.com/FlagTree/flagtree/tree/main/third_party/xpu/) 后端（对应 Triton 3.0），加入 CI/CD。
* 2025/03/19 新增接入 [mthreads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads/) 后端（对应 Triton 3.1），加入 CI/CD。
* 2025/03/12 新增接入 [iluvatar](https://github.com/FlagTree/flagtree/tree/main/third_party/iluvatar/) 后端（对应 Triton 3.1），加入 CI/CD。

## 从源代码安装

安装依赖（注意使用正确的 python3.x 执行）：
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python
python3 -m pip install -r requirements.txt
```

构建安装（网络畅通环境下推荐使用）：
```shell
# Set FLAGTREE_BACKEND using the backend name from the table above
export FLAGTREE_BACKEND=${backend_name}  # nvidia/amd/triton-shared do not set it
cd python  # For Triton 3.1, 3.2, 3.3, you need to enter the python directory to build
python3 -m pip install . --no-build-isolation -v  # 自动卸载 triton
python3 -m pip show flagtree
cd ${ANY_OTHER_PATH}; python3 -c 'import triton; print(triton.__path__)'
```

- [从源码构建技巧](/documents/build_cn.md#从源码构建技巧)
- [离线构建支持：预下载依赖包](/documents/build_cn.md#离线构建支持)

## 非源码安装

如果不希望从源码安装，可以直接拉取安装 whl（支持部分后端）。

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # TODO: automatically uninstall triton
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
```

|后端     |安装命令（版本号对应 git tag）|Triton 版本|支持的 Python 版本|
|--------|---------------------------|----------|----------------|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1 $RES            |3.1|3.10, 3.11, 3.12|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1+3.2 $RES        |3.2|3.10, 3.11, 3.12|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1+3.3 $RES        |3.3|3.10, 3.11, 3.12|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1+3.4 $RES        |3.4|3.12|
|nvidia  |python3 -m pip install flagtree==0.3.0rc1+3.5 $RES        |3.5|3.12|
|iluvatar|python3 -m pip install flagtree==0.3.0rc2+iluvatar3.1 $RES|3.1|3.10|
|mthreads|python3 -m pip install flagtree==0.3.0rc3+mthreads3.1 $RES|3.1|3.10|
|ascend  |python3 -m pip install flagtree==0.3.0rc1+ascend3.2 $RES  |3.2|3.11|
|hcu     |python3 -m pip install flagtree==0.3.0rc2+hcu3.0 $RES     |3.0|3.10|
|enflame |python3 -m pip install flagtree==0.3.0rc1+enflame3.3 $RES |3.3|3.10|

## 运行测试

安装完成后一般可以在设备支持的环境下运行测试，具体后端支持的测试可前往对应分支的 .github/workflow/${backend_name}-build-and-test.yml 查看。
```shell
# nvidia/amd
cd python/test/unit
python3 -m pytest -s
# other backends
cd third_party/${backend_name}/python/test/unit
python3 -m pytest -s
```

## 关于贡献

欢迎参与 FlagTree 的开发并贡献代码，详情请参考 [CONTRIBUTING.md](/CONTRIBUTING_cn.md)。

## 许可证

FlagTree 使用 [MIT license](/LICENSE)。
