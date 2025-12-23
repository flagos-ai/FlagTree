[English](../en/CHANGELOG.md)
## 变更历史

* 2025/12/24 支持拉取和安装 [Wheel](./install.md#no-source-install) 包。
* 2025/12/08 新增接入燧原 [Enflame](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/enflame/)
  后端（对应 Triton 3.3），支持 CI/CD。
* 2025/11/26 添加 FlagTree 后端特化统一设计文档 [FlagTree_Backend_Specialization](reports/decoupling/)。
* 2025/10/28 提供离线构建支持（预下载依赖包），改善网络环境受限时的构建体验，使用方法见后文。
* 2025/09/30 在 GPGPU 上支持编译指导共享内存。
* 2025/09/29 SDK 存储迁移至金山云（Ksyun），大幅提升下载稳定性。
* 2025/09/25 支持编译指导昇腾（Ascend）的后端编译能力。
* 2025/09/16 新增接入海光 [HCU](https://github.com/flagos-ai/flagtree/tree/main/third_party/hcu/)
  后端（对应 Triton 3.0），支持 CI/CD。
* 2025/09/09 Fork 并修改 [llvm-project](https://github.com/flagos-ai/llvm-project)，
  承接 [FLIR](https://github.com/flagos-ai/flir) 的功能。
* 2025/09/01 新增适配飞桨（PaddlePaddle）框架，包含 CI/CD。
* 2025/08/16 新增适配北京超级云计算中心 AI 智算云。
* 2025/08/04 新增接入 T\* 后端（对应 Triton 3.1）。
* 2025/08/01 [FLIR](https://github.com/flagos-ai/flir) 支持编译指导共享内存加载（Shared Memory Loading）。
* 2025/07/30 更新寒武纪 [Cambricon](https://github.com/flagos-ai/flagtree/tree/triton_v3.2.x/third_party/cambricon/)
  后端（对应 Triton 3.2）。
* 2025/07/25 浪潮团队新增适配龙蜥（OpenAnolis）操作系统。
* 2025/07/09 [FLIR](https://github.com/flagos-ai/flir) 支持编译指导异步 DMA（Async DMA）。
* 2025/07/08 新增多后端编译统一管理模块。
* 2025/07/02 [FlagGems](https://github.com/flagos-ai/FlagGems) LibTuner 适配 triton_v3.3.x 版本。
* 2025/07/02 新增接入 S\* 后端（对应 Triton 3.3）。
* 2025/06/20 [FLIR](https://github.com/flagos-ai/flir) 开始承接 MLIR 扩展功能。
* 2025/06/06 新增接入清微智能 [TsingMicro](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/tsingmicro/)
  后端（对应 Triton 3.3），支持 CI/CD。
* 2025/06/04 新增接入昇腾 [Ascend](https://github.com/flagos-ai/flagtree/blob/triton_v3.2.x/third_party/ascend)
  后端（对应 Triton 3.2），包含 CI/CD。
* 2025/06/03 新增接入沐曦 [MetaX](https://github.com/flagos-ai/flagtree/tree/main/third_party/metax/)
  后端（对应 Triton 3.1），包括 CI/CD。
* 2025/05/22 [FlagGems](https://github.com/flagos-ai/FlagGems) LibEntry 适配 triton_v3.3.x 版本。
* 2025/05/21 [FLIR](https://github.com/flagos-ai/flir) 开始承接到中间层的转换功能。
* 2025/04/09 新增接入 ARM [AIPU](https://github.com/flagos-ai/flagtree/tree/triton_v3.3.x/third_party/aipu/)
  后端（对应 Triton 3.3），提供 Torch 标准扩展[范例](https://github.com/flagos-ai/flagtree/blob/triton_v3.3.x/third_party/aipu/backend/aipu_torch_dev.cpp)，
  支持 CI/CD。
* 2025/03/26 接入安全合规扫描。
* 2025/03/19 新增接入昆仑芯 [XPU](https://github.com/flagos-ai/flagtree/tree/main/third_party/xpu/)
  后端（对应 Triton 3.0），支持 CI/CD。
* 2025/03/19 新增接入摩尔线程 [mthreads](https://github.com/flagos-ai/flagtree/tree/main/third_party/mthreads/)
  后端（对应 Triton 3.1），支持 CI/CD。
* 2025/03/12 新增接入天数智芯 [Iluvatar](https://github.com/flagos-ai/flagtree/tree/main/third_party/iluvatar/)
  后端（对应 Triton 3.1），支持 CI/CD。
