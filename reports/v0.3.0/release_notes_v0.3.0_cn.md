<div align="right"><a href="./release_notes_v0.3.0.md">English</a></div>

## FlagTree 0.3.0 Release

### Highlights

FlagTree 继承前一版本的能力，持续集成新的后端，壮大生态矩阵。项目当前处于初期，目标是兼容各芯片后端现有适配方案，统一代码仓库，打造代码共建平台，快速实现单仓库多后端支持。同时持续建设统一编程接口扩展，持续建设中间层表示及转换扩展（FLIR），提升硬件感知和编译指导支持能力与范围（flagtree_hints）。

### New features

* 新增多后端支持

目前支持的后端包括 triton_shared cpu、iluvatar、xpu (klx)、mthreads、metax、aipu(arm npu)、ascend npu & cpu、tsingmicro、cambricon、__hcu__，其中 __加粗__ 为本次新增。 <br>
各新增后端保持前一版本的能力：跨平台编译与快速验证、高差异度模块插件化、CI/CD、质量管理能力。 <br>

* 持续对接上游生态

感谢合作单位的技术支持，FlagTree 新增适配框架 Paddle、操作系统 OpenAnolis、北京超级云计算中心。

* 持续建设 FLIR

持续进行 DSL 扩展、TTIR 扩展、Linalg 中间层表示及转换扩展、MLIR 扩展，提供编程灵活性，丰富表达能力，完善转换能力。

* 确定编译指导规范，新增多后端编译统一管理模块

flagtree_hints 对硬件单元映射、编译转换优化选择进行指导，并通过统一模块管理多后端的指导差异。

* 与 FlagGems 算子库联合建设

在版本适配、后端接口、注册机制、测试修改等方面，与 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子库联合支持相关特性。

### Looking ahead

GPGPU 后端完善接入整合，后端特化与主代码实现解耦，为应用 FlagTree 通用扩展、通用优化打下工程基础。 <br>
以全面覆盖算子库中的多种写法为目标，完善 FLIR 编译的完备度，匹配多种后端需求，打通更多后端的编译。 <br>
flagtree_hints 在 TritonGPU、Linalg 两条路线的不同后端上继续挖掘算子的性能优化潜力。 <br>
