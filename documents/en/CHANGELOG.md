[中文](../zh-cn/CHANGELOG.md)

## Change History

<!--TODO(Qiming): Tune the format-->
* 2025/12/24 Support pull and install [whl](/README.md#non-source-installation).
* 2025/12/08 Added [Enflame](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/enflame/)
  backend integration (based on Triton 3.3), and added CI/CD.
* 2025/11/26 Add FlagTree_Backend_Specialization unified design document [FlagTree_Backend_Specialization](reports/decoupling/).
* 2025/10/28 Provides offline build support (pre-downloaded dependency packages),
  improving the build experience when network environment is limited. See usage instructions below.
* 2025/09/30 Support flagtree_hints for shared memory on GPGPU.
* 2025/09/29 SDK storage migrated to ksyuncs, improving download stability.
* 2025/09/25 Support flagtree_hints for ascend backend compilation capability.
* 2025/09/16 Added [hcu](https://github.com/FlagTree/flagtree/tree/main/third_party/hcu/) backend integration
  (based on Triton 3.0), and added CI/CD.
* 2025/09/09 Forked and modified [llvm-project](https://github.com/FlagTree/llvm-project)
  to support [FLIR](https://github.com/flagos-ai/flir).
* 2025/09/01 Added adaptation for Paddle framework, and added CI/CD.
* 2025/08/16 Added adaptation for Beijing Super Cloud Computing Center.
* 2025/08/04 Added T\*\*\* backend integration (based on Triton 3.1).
* 2025/08/01 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for shared memory loading.
* 2025/07/30 Updated [cambricon](https://github.com/FlagTree/flagtree/tree/triton_v3.2.x/third_party/cambricon/)
  backend (based on Triton 3.2).
* 2025/07/25 Inspur team added adaptation for OpenAnolis OS.
* 2025/07/09 [FLIR](https://github.com/flagos-ai/flir) supports flagtree_hints for Async DMA.
* 2025/07/08 Added UnifiedHardware manager for multi-backend compilation.
* 2025/07/02 [FlagGems](https://github.com/flagos-ai/FlagGems) LibTuner adapted to triton_v3.3.x version.
* 2025/07/02 Added S\*\*\* backend integration (based on Triton 3.3).
* 2025/06/20 [FLIR](https://github.com/flagos-ai/flir) began supporting MLIR extension functionality.
* 2025/06/06 Added [TsingMicro](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)
  backend integration (based on Triton 3.3), and added CI/CD.
* 2025/06/04 Added [Ascend](https://github.com/flagos-ai/flagtree/blob/triton_v3.2.x/third_party/ascend)
  backend integration (based on Triton 3.2), and added CI/CD.
* 2025/06/03 Added [MetaX](https://github.com/flagos-ai/flagtree/tree/main/third_party/metax/)
  backend integration (based on Triton 3.1), and added CI/CD.
* 2025/05/22 FlagGems LibEntry adapted to triton_v3.3.x version.
* 2025/05/21 [FLIR](https://github.com/flagos-ai/flir) began supporting conversion functionality to middle layer.
* 2025/04/09 Added ARM [AIPU](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/aipu/)
  backend integration (based on Triton 3.3), provided a torch standard extension
  [example](https://github.com/flagos-ai/flagtree/blob/triton_v3.3.x/third_party/aipu/backend/aipu_torch_dev.cpp),
  and added CI/CD.
* 2025/03/26 Integrated security compliance scanning.
* 2025/03/19 Added Kunlunxin [XPU](https://github.com/flagos-ai/flagtree/tree/main/third_party/xpu/)
  backend integration (based on Triton 3.0), and added CI/CD.
* 2025/03/19 Added [Moore Threads](https://github.com/flagos-ai/flagtree/tree/main/third_party/mthreads/)
  backend integration (based on Triton 3.1), and added CI/CD.
* 2025/03/12 Added [Iluvatar](https://github.com/flagos-ai/flagtree/tree/main/third_party/iluvatar/)
  backend integration (based on Triton 3.1), and added CI/CD.
