<div align="right"><a href="/README_cn.md">中文版</a></div>

## <img width="30" height="30" alt="FlagTree-GitHub" src="https://github.com/user-attachments/assets/d8d24c81-6f46-4adc-94e2-b89b03afcb43" /> FlagTree

FlagTree is an open source, unified compiler for multiple AI chips project dedicated to developing a diverse ecosystem of AI chip compilers and related tooling platforms, thereby fostering and strengthening the upstream and downstream Triton ecosystem. Currently in its initial phase, the project aims to maintain compatibility with existing adaptation solutions while unifying the codebase to rapidly implement single-repository multi-backend support. For upstream model users, it provides unified compilation capabilities across multiple backends; for downstream chip manufacturers, it offers examples of Triton ecosystem integration. <br>
Each backend is based on different versions of triton, and therefore resides in different protected branches. All these protected branches have equal status.

|Branch|Vendor|Backend|Triton version|
|------|------|-------|--------------|
|[main](https://github.com/flagos-ai/flagtree/tree/main)|NVIDIA<br>AMD<br>x86_64 cpu<br>ILUVATAR（天数智芯）<br>Moore Threads（摩尔线程）<br>KLX<br>MetaX（沐曦股份）<br>HYGON（海光信息）|[nvidia](/third_party/nvidia/)<br>[amd](/third_party/amd/)<br>[triton-shared](https://github.com/microsoft/triton-shared)<br>[iluvatar](/third_party/iluvatar/)<br>[mthreads](/third_party/mthreads/)<br>[xpu](/third_party/xpu/)<br>[metax](/third_party/metax/)<br>[hcu](third_party/hcu/)|3.1<br>3.1<br>3.1<br>3.1<br>3.1<br>3.0<br>3.1<br>3.0|
|[triton_v3.2.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x)|NVIDIA<br>AMD<br>Huawei Ascend（华为昇腾）<br>Cambricon（寒武纪）|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/amd/)<br>[ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/third_party/ascend)<br>[cambricon](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/cambricon/)|3.2|
|[triton_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x)|NVIDIA<br>AMD<br>x86_64 cpu<br>ARM China<br>Tsingmicro（清微智能）<br>Enflame（燧原）|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/amd/)<br>[triton-shared](https://github.com/microsoft/triton-shared)<br>[aipu](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/)<br>[tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)<br>[enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/)|3.3|
|[triton_v3.4.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.4.x)|NVIDIA<br>AMD|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.4.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.4.x/third_party/amd/)|3.4|
|[triton_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x)|NVIDIA<br>AMD|[nvidia](https://github.com/FlagTree/flagtree/tree/triton_v3.5.x/third_party/nvidia/)<br>[amd](https://github.com/FlagTree/flagtree/tree/triton_v3.5.x/third_party/amd/)|3.5|

## Latest News

* 2025/12/24 Support pull and install [Wheel](/README.md#non-source-installation).
* 2025/12/08 Added [enflame](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/enflame/) backend integration (based on Triton 3.3), and added CI/CD.
* 2025/11/26 Add FlagTree_Backend_Specialization Unified Design Document [FlagTree_Backend_Specialization](/documents/decoupling/).
* 2025/10/28 Provides offline build support (pre-downloaded dependency packages), improving the build experience when network environment is limited. See usage instructions below.
* 2025/09/30 Support flagtree_hints for shared memory on GPGPU.
* 2025/09/29 SDK storage migrated to ksyuncs, improving download stability.
* 2025/09/25 Support flagtree_hints for ascend backend compilation capability.
* 2025/09/16 Added [hcu](https://github.com/FlagTree/flagtree/tree/main/third_party/hcu/) backend integration (based on Triton 3.0), and added CI/CD.
* 2025/09/09 Forked and modified [llvm-project](https://github.com/FlagTree/llvm-project) to support [FLIR](https://github.com/flagos-ai/flir).
* 2025/09/01 Added adaptation for Paddle framework, and added CI/CD.
* 2025/08/16 Added adaptation for Beijing Super Cloud Computing Center.
* 2025/08/04 Added T*** backend integration (based on Triton 3.1).
* 2025/08/01 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for shared memory loading.
* 2025/07/30 Updated [cambricon](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/cambricon/) backend (based on Triton 3.2).
* 2025/07/25 Inspur team added adaptation for OpenAnolis OS.
* 2025/07/09 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for Async DMA.
* 2025/07/08 Added UnifiedHardware manager for multi-backend compilation.
* 2025/07/02 [FlagGems](https://github.com/flagos-ai/FlagGems) LibTuner adapted to triton_v3.3.x version.
* 2025/07/02 Added S*** backend integration (based on Triton 3.3).
* 2025/06/20 [FLIR](https://github.com/flagos-ai/flir) began supporting MLIR extension functionality.
* 2025/06/06 Added [tsingmicro](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/) backend integration (based on Triton 3.3), and added CI/CD.
* 2025/06/04 Added [ascend](https://github.com/FlagTree/flagtree/blob/triton_v3.2.x/third_party/ascend) backend integration (based on Triton 3.2), and added CI/CD.
* 2025/06/03 Added [metax](https://github.com/FlagTree/flagtree/tree/main/third_party/metax/) backend integration (based on Triton 3.1), and added CI/CD.
* 2025/05/22 FlagGems LibEntry adapted to triton_v3.3.x version.
* 2025/05/21 [FLIR](https://github.com/flagos-ai/flir) began supporting conversion functionality to middle layer.
* 2025/04/09 Added [aipu](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/aipu/) backend integration (based on Triton 3.3), provided a torch standard extension [example](https://github.com/flagos-ai/flagtree/blob/triton_v3.3.x/third_party/aipu/backend/aipu_torch_dev.cpp), and added CI/CD.
* 2025/03/26 Integrated security compliance scanning.
* 2025/03/19 Added [xpu](https://github.com/FlagTree/flagtree/tree/main/third_party/xpu/) backend integration (based on Triton 3.0), and added CI/CD.
* 2025/03/19 Added [mthreads](https://github.com/FlagTree/flagtree/tree/main/third_party/mthreads/) backend integration (based on Triton 3.1), and added CI/CD.
* 2025/03/12 Added [iluvatar](https://github.com/FlagTree/flagtree/tree/main/third_party/iluvatar/) backend integration (based on Triton 3.1), and added CI/CD.

## Install from source

Installation dependencies (ensure you use the correct python3.x version):

```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

Building and Installation (Recommended for environments with good network connectivity):
```shell
# Set FLAGTREE_BACKEND using the backend name from the table above
export FLAGTREE_BACKEND=${backend_name}  # nvidia/amd/triton-shared do not set it
cd python  # For Triton 3.1, 3.2, 3.3, you need to enter the python directory to build
python3 -m pip install . --no-build-isolation -v  # Automatically uninstall triton
python3 -m pip show flagtree
cd ${ANY_OTHER_PATH}; python3 -c 'import triton; print(triton.__path__)'
```

- [Tips for building](/documents/build.md#tips-for-building)
- [Offline build support: pre-downloading dependency packages](/documents/build.md#offline-build-support)

## Non-Source Installation

If you do not wish to build from source, you can directly pull and install whl (supports some backends).

```shell
# Note: First install PyTorch, then execute the following commands
python3 -m pip uninstall -y triton  # TODO: automatically uninstall triton
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
```

|Backend |Install cmd (versions have corresponding git tags)|Triton version|Python version|
|--------|--------------------------------------------------|--------------|--------------|
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

## Running tests

After installation, you can generally run the following tests. For specific backend support tests, please refer to .github/workflow/${backend_name}-build-and-test.yml in the corresponding branch.
```shell
# nvidia/amd
cd python/test/unit
python3 -m pytest -s
# other backends
cd third_party/${backend_name}/python/test/unit
python3 -m pytest -s
```

## Contributing

Contributions to FlagTree development are welcome. Please refer to [CONTRIBUTING.md](/CONTRIBUTING_cn.md) for details.

## License

FlagTree is licensed under the [MIT license](/LICENSE).
