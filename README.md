[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)
[[中文版](./README_cn.md)|English]

<div align="right">
  <a href="https://www.linkedin.com/company/flagos-community" target="_blank">
    <img src=".github/assets/Linkedin.png" alt="LinkIn" width="32" height="32" />
  </a>

  <a href="https://www.youtube.com/@FlagOS_Official" target="_blank">
    <img src=".github/assets/youtube.png" alt="YouTube" width="32" height="32" />
  </a>

  <a href="https://x.com/FlagOS_Official" target="_blank">
    <img src=".github/assets/x.png" alt="X" width="32" height="32" />
  </a>

  <a href="https://www.facebook.com/flagosglobalcommunity" target="_blank">
    <img src=".github/assets/Facebook.png" alt="Facebook" width="32" height="32" />
  </a>

  <a href="https://discord.com/invite/ubqGuFMTNE" target="_blank">
    <img src=".github/assets/discord.png" alt="Discord" width="32" height="32" />
  </a>
</div>

<img width="90" height="514" alt="FlagTree" src=".github/assets/FlagTree.png" />

FlagTree is part of [FlagOS](https://flagos.io/), a fully open-source system software stack designed to unify the model–system–chip layers and foster an open and collaborative ecosystem.
It enables a "develop once, run anywhere" workflow across diverse AI accelerators,
unlocking hardware performance, eliminating fragmentation among AI chipset-specific software stacks,
and substantially lowering the cost of porting and maintaining AI workloads.

FlagTree is an open source, unified compiler for multiple AI chips project dedicated to developing a diverse ecosystem of AI chip compilers and related tooling platforms,
thereby fostering and strengthening the upstream and downstream Triton ecosystem.
Currently in its initial phase, the project aims to maintain compatibility with existing adaptation solutions while unifying the codebase to rapidly implement single-repository multi-backend support.
For upstream model users, it provides unified compilation capabilities across multiple backends;
for downstream chip manufacturers, it offers examples of Triton ecosystem integration.

## Multi-backend support

Each backend is based on different versions of Triton, and therefore resides in different protected branches.
All these protected branches have equal status. CI/CD runners are provisioned for every backend listed in the table.
Some backends support multiple Triton versions; only the latest version is shown in the table.

|Branch  |Vendor|Backend|Triton<br>version|Installation|
|:-------|:-----|:------|:----------------|:-----------|
|[main](https://github.com/flagos-ai/flagtree/tree/main)|NVIDIA<br>NVIDIA TileIR<br>AMD<br>Enflame（燧原）<br>ILUVATAR（天数智芯）<br>HYGON（海光信息）<br>Moore Threads（摩尔线程）<br>DAMO ACADEMY（阿里达摩院）<br>Huixi（辉羲智能）<br>MetaX（沐曦股份）<br>Sunrise（曦望芯科）<br>KLX<br>T-Head（平头哥）<br>SpacemiT（进迭时空）|[nvidia](/third_party/nvidia/)<br>[tileir](/third_party/tileir/)<br>[amd](/third_party/amd/)<br>[enflame](/third_party/enflame/)<br>[iluvatar](/third_party/iluvatar/)<br>[hcu](/third_party/hcu/)<br>[mthreads](/third_party/mthreads/)<br>[damoacademy](/third_party/thrive/)<br>[rpu](/third_party/rpu/)<br>[metax](/third_party/metax/)<br>[sunrise](/third_party/sunrise/)<br>[xpu](/third_party/xpu/)<br>[ppu](/third_party/ppu/)<br>[spacemit](/third_party/spacemit/)|3.6|[install nvidia](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-nvidia)<br>[install tileir](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-tileir)<br>-<br>[install enflame](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-enflame)<br>[install iluvatar](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-iluvatar)<br>[install hcu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-hcu)<br>[install mthreads](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-mthreads)<br>-<br>[install rpu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-rpu)<br>[install metax](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-metax)<br>[install sunrise](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-sunrise)<br>[install xpu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu)<br>[install ppu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ppu)<br>[install spacemit](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-spacemit)|
|[triton_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x)|Huawei Ascend（华为昇腾）|[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/third_party/ascend/)|3.5|[install ascend](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend)|
|[triton_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x)|ARM China（安谋科技）<br>Tsingmicro（清微智能）<br>ARM64 cpu<br>x86_64 cpu|[aipu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/aipu/)<br>[tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/)<br>[cpu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/cpu/)<br>[triton-shared](https://github.com/microsoft/triton-shared)|3.3|[install aipu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-aipu)<br>[install tsingmicro](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-tsingmicro)<br>[install cpu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-cpu)<br>-|
|[triton_v3.2.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x)|Cambricon（寒武纪）|[cambricon](https://github.com/flagos-ai/FlagTree/tree/triton_v3.2.x/third_party/cambricon/)|3.2|-|

FlagTree extension components are currently available on some backends:

|Branch  |Backend|Triton version|Extension components|
|:-------|:------|:-------------|:-------------------|
|[main](https://github.com/flagos-ai/flagtree/tree/main)|[nvidia](/third_party/nvidia/)<br>[enflame](/third_party/enflame/)|3.6|[TLE-Lite](https://github.com/flagos-ai/FlagTree/wiki/TLE#32-tle-lite)<br>[TLE-Struct GPU](https://github.com/flagos-ai/FlagTree/wiki/TLE#331-gpu)<br>[TLE-Raw](https://github.com/flagos-ai/FlagTree/wiki/TLE-Raw)<br>[HINTS](https://github.com/flagos-ai/FlagTree/wiki/HINTS)|
|[main](https://github.com/flagos-ai/flagtree/tree/main)|[amd](/third_party/amd/)<br>[mthreads](/third_party/mthreads/)<br>[sunrise](/third_party/sunrise/)<br>[hcu](/third_party/hcu/)<br>[iluvatar](/third_party/iluvatar/)<br>[ppu](/third_party/ppu/)|3.6|[TLE-Lite](https://github.com/flagos-ai/FlagTree/wiki/TLE#32-tle-lite)<br>[TLE-Struct GPU](https://github.com/flagos-ai/FlagTree/wiki/TLE#331-gpu)|
|[main](https://github.com/flagos-ai/flagtree/tree/main)|[metax](/third_party/metax/)|3.6|[TLE-Lite](https://github.com/flagos-ai/FlagTree/wiki/TLE#32-tle-lite)|
|[triton_v3.5.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.5.x)|[ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/third_party/ascend/)|3.5|[TLE-Struct DSA](https://github.com/flagos-ai/FlagTree/wiki/TLE#332-dsa)<br>[FLIR](https://github.com/flagos-ai/flir)<br>[HINTS](https://github.com/flagos-ai/FlagTree/wiki/HINTS)|
|[triton_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x)|[tsingmicro](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/third_party/tsingmicro/)|3.3|[TLE-Lite](https://github.com/flagos-ai/FlagTree/wiki/TLE#32-tle-lite)<br>[TLE-Struct DSA](https://github.com/flagos-ai/FlagTree/wiki/TLE#332-dsa)<br>[FLIR](https://github.com/flagos-ai/flir)|
|[triton_v3.3.x](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x)|[aipu](https://github.com/flagos-ai/FlagTree/blob/triton_v3.3.x/third_party/aipu/)|3.3|[FLIR](https://github.com/flagos-ai/flir)<br>[HINTS](https://github.com/flagos-ai/FlagTree/wiki/HINTS)|

## TLE (Triton Language Extensions)

If you want to use TLE on the NVIDIA backend, please use the main branch.
For other backends, please refer to the table above.

Triton provides strong productivity for kernel development, but heterogeneous AI chips and deeper performance tuning scenarios need more explicit control over distributed execution, memory access patterns, and hardware-specific primitives.
TLE extends Triton in a layered way to bridge this gap while keeping compatibility with existing Triton workflows.

<img alt="tle-speedup-20260626" src=".github/assets/tle-speedup-20260626.jpg" />

Key advantages of TLE:

* Progressive abstraction from portable usage to hardware-oriented tuning (`Lite` / `Struct` / `Raw`).
* Better coverage for multi-device, architecture-specific, and backend lowering scenarios.
* Lower migration cost from existing Triton kernels while preserving optimization headroom.

For detailed design, APIs, and examples, please refer to the [TLE Wiki](https://github.com/flagos-ai/FlagTree/wiki/TLE) and [TLE-Raw Wiki](https://github.com/flagos-ai/FlagTree/wiki/TLE-Raw).

## Performance Improvements

Without modifying any Triton operator code, FlagTree can achieve performance gains for certain shapes in real-world models.
The following uses the mm operator under some shapes called in the Qwen model as an example to demonstrate FlagTree's performance speedup ratio on various chips.

<img width="200" height="184" alt="nv_h100_bf16_mm_1a" src=".github/assets/nv_h100_bf16_mm_1a.png" />  <img width="200" height="184" alt="nv_h100_fp32_mm_1a" src=".github/assets/nv_h100_fp32_mm_1a.png" />
<img width="200" height="184" alt="hcu_bf16_mm_1a" src=".github/assets/hcu_bf16_mm_1a.png" />  <img width="200" height="184" alt="hcu_fp32_mm_1a" src=".github/assets/hcu_fp32_mm_1a.png" />
<img width="200" height="184" alt="hcu_bf16_mm_3d" src=".github/assets/hcu_bf16_mm_3d.png" />  <img width="200" height="184" alt="hcu_fp32_mm_3d" src=".github/assets/hcu_fp32_mm_3d.png" />
<img width="200" height="184" alt="mthreads_bf16_mm_1a_3c" src=".github/assets/mthreads_bf16_mm_1a_3c.png" />  <img width="200" height="184" alt="mthreads_fp32_mm_1a" src=".github/assets/mthreads_fp32_mm_1a.png" />
<img width="200" height="184" alt="mthreads_bf16_mm_1c" src=".github/assets/mthreads_bf16_mm_1c.png" />  <img width="200" height="184" alt="mthreads_fp32_mm_1c" src=".github/assets/mthreads_fp32_mm_1c.png" />
<img width="200" height="184" alt="enflame_bf16_mm_3d" src=".github/assets/enflame_bf16_mm_3d.png" />  <img width="200" height="184" alt="enflame_fp32_mm_3d" src=".github/assets/enflame_fp32_mm_3d.png" />

## Latest News

* 2026/08/18 Added the [spacemit](/third_party/spacemit/) backend integration (based on Triton 3.6) and added CI/CD.
* 2026/08/04 Added the [ppu](/third_party/ppu/) backend integration (based on Triton 3.6) and added CI/CD.
* 2026/07/07 Added the NVIDIA [tileir](/third_party/tileir/) backend integration (based on Triton 3.6) and added CI/CD.
* 2026/07/03 Upgraded the [iluvatar](/third_party/iluvatar/) backend to Triton 3.6 and added CI/CD.
* 2026/07/02 Upgraded the [xpu](/third_party/xpu/) backend to Triton 3.6 and added CI/CD.
* 2026/06/30 Upgraded the [sunrise](/third_party/sunrise/) backend to Triton 3.6 and added CI/CD.
* 2026/06/26 Upgraded the [metax](/third_party/metax/) backend to Triton 3.6 and added CI/CD.
* 2026/06/10 Added the [rpu](/third_party/rpu/) backend integration (based on Triton 3.6) and added CI/CD.
* 2026/06/08 Upgraded the [ascend](https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x/third_party/ascend/) backend to Triton 3.5 and added CI/CD.
* 2026/06/03 Added the ARM64 [cpu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/cpu/) backend integration (based on Triton 3.3).
* 2026/06/01 Added the [damoacademy](/third_party/thrive/) backend integration (based on Triton 3.6) and added CI/CD.
* 2026/05/12 Upgraded the [mthreads](/third_party/mthreads/) backend to Triton 3.6 and added CI/CD.
* 2026/05/07 Upgraded the [hcu](/third_party/hcu/) backend to Triton 3.6 and added CI/CD.
* 2026/04/24 Upgraded the [mthreads](https://github.com/flagos-ai/FlagTree/tree/triton_v3.2.x/third_party/mthreads/) backend to Triton 3.2 and added CI/CD.
* 2026/04/17 Upgraded the [enflame](/third_party/enflame/) backend to Triton 3.6 and added CI/CD.
* 2026/03/13 Upgraded the [enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.5.x/third_party/enflame/) backend to Triton 3.5 and added CI/CD.
* 2026/01/23 Added the [sunrise](https://github.com/flagos-ai/FlagTree/tree/triton_v3.4.x/third_party/sunrise/) backend integration (based on Triton 3.4) and added CI/CD.
* 2026/01/08 Added wiki pages for new features [HINTS](https://github.com/flagos-ai/FlagTree/wiki/HINTS), [TLE](https://github.com/flagos-ai/FlagTree/wiki/TLE), [TLE-Raw](https://github.com/flagos-ai/FlagTree/wiki/TLE-Raw).
* 2025/12/08 Added the [enflame](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/enflame/) backend integration (based on Triton 3.3) and added CI/CD.
* 2025/11/26 Added FlagTree_Backend_Specialization Unified Design Document [FlagTree-Backend-Specialization](https://github.com/flagos-ai/FlagTree/wiki/FlagTree-Backend-Specialization).
* 2025/10/28 Added support for the offline build with pre-downloaded dependency packages, improving the build experience in restricted environments. See the usage instructions below.
* 2025/09/30 Added support for shared memory flagtree_hints on GPGPU.
* 2025/09/29 Migrated the SDK storage to ksyuncs, improving download stability.
* 2025/09/25 Added support for flagtree_hints in the ascend backend compilation.
* 2025/09/16 Added the [hcu](https://github.com/flagos-ai/flagtree/tree/triton_v3.1.x/third_party/hcu/) backend integration (based on Triton 3.0) and added CI/CD.
* 2025/09/09 Forked and modified [llvm-project](https://github.com/FlagTree/llvm-project) to support [FLIR](https://github.com/flagos-ai/flir).
* 2025/09/01 Added support for Paddle framework and added CI/CD.
* 2025/08/16 Added support for Beijing Super Cloud Computing Center.
* 2025/08/04 Added the T*** backend integration (based on Triton 3.1).
* 2025/08/01 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for shared memory loading.
* 2025/07/30 Upgraded the [cambricon](https://github.com/flagos-ai/FlagTree/tree/triton_v3.2.x/third_party/cambricon/) backend to Triton 3.2.
* 2025/07/25 The Inspur team added support for OpenAnolis OS.
* 2025/07/09 [FLIR](https://github.com/flagos-ai/flir) added support for Async DMA flagtree_hints.
* 2025/07/08 Added UnifiedHardware manager for multi-backend compilation.
* 2025/07/02 Added the S*** backend integration (based on Triton 3.3).
* 2025/06/20 [FLIR](https://github.com/flagos-ai/flir) added support for MLIR extension functionality.
* 2025/06/06 Added the [tsingmicro](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/tsingmicro/) backend integration (based on Triton 3.3) and added CI/CD.
* 2025/06/04 Added the [ascend](https://github.com/flagos-ai/FlagTree/blob/triton_v3.2.x/third_party/ascend) backend integration (based on Triton 3.2) and added CI/CD.
* 2025/06/03 Added the [metax](https://github.com/flagos-ai/flagtree/tree/triton_v3.1.x/third_party/metax/) backend integration (based on Triton 3.0) and added CI/CD.
* 2025/05/21 [FLIR](https://github.com/flagos-ai/flir) added support for conversion functionality to middle layer.
* 2025/04/09 Added the [aipu](https://github.com/flagos-ai/FlagTree/tree/triton_v3.3.x/third_party/aipu/) backend integration (based on Triton 3.3), provided a torch standard extension [example](https://github.com/flagos-ai/flagtree/blob/triton_v3.3.x/third_party/aipu/backend/aipu_torch_dev.cpp) and added CI/CD.
* 2025/03/26 Integrated security compliance scanning.
* 2025/03/19 Added the [xpu](https://github.com/flagos-ai/flagtree/tree/triton_v3.1.x/third_party/xpu/) backend integration (based on Triton 3.0) and added CI/CD.
* 2025/03/19 Added the [mthreads](https://github.com/flagos-ai/flagtree/tree/triton_v3.1.x/third_party/mthreads/) backend integration (based on Triton 3.1) and added CI/CD.
* 2025/03/12 Added the [iluvatar](https://github.com/flagos-ai/flagtree/tree/triton_v3.1.x/third_party/iluvatar/) backend integration (based on Triton 3.1) and added CI/CD.

# Environment setup

The best practice to avoid environment compatibility issues is to use the image recommended in [User Manual](https://github.com/flagos-ai/FlagTree/wiki/User-Manual).

## Install from source

Installation dependencies (Confirm the correct python3.x version is being used):

```shell
apt update; apt install zlib1g zlib1g-dev libxml2 libxml2-dev nlohmann-json3-dev
python3 -m pip install -r python/requirements.txt
```

General building and installation procedure (Recommended for environments with good network connectivity):

```shell
# Set FLAGTREE_BACKEND using the backend name from the table above
export FLAGTREE_BACKEND=${backend_name}  # Do not set it on nvidia/amd/triton-shared

# For Triton 3.1/3.2/3.3 (branch: triton_v3.1.x, triton_v3.2.x, triton_v3.3.x)
cd python; python3 -m pip install . --no-build-isolation -v  # Install flagtree and uninstall triton

# For Triton 3.4/3.5/3.6 (branch: triton_v3.4.x, triton_v3.5.x, main)
python3 -m pip install . --no-build-isolation -v             # Install flagtree and uninstall triton
```

After installing `flagtree`, you can check it with:

```shell
python3 -m pip show flagtree
cd ${ANY_DIR_OTHER_THAN_FLAGTREE_PYTHON}; python3 -c 'import triton; print(triton.__path__)'
```

## Source-free Installation

Refer to [User Manual](https://github.com/flagos-ai/FlagTree/wiki/User-Manual).

## Contributing

Contributions to FlagTree development are welcome. Please refer to [CONTRIBUTING.md](/CONTRIBUTING.md) for details.

## License

FlagTree is licensed under the [MIT license](/LICENSE).
