<div align="right"><a href="./release_notes_v0.2.0.md">English</a></div>

## FlagTree 0.2.0 Release

### Highlights

FlagTree 继承前一版本的能力，持续集成新的后端，拓展对 Triton 版本的支持，提供硬件感知优化能力。项目当前处于初期，目标是兼容各芯片后端现有适配方案，统一代码仓库，打造代码共建平台，快速实现单仓库多后端支持。

### New features

* 新增多后端支持

目前支持的后端包括 triton_shared cpu、iluvatar、xpu (klx)、mthreads、__metax__、__aipu__(arm npu)、__ascend__ npu & cpu、__tsingmicro__、cambricon，其中 __加粗__ 为本次新增。 <br>
各新增后端保持前一版本的能力：跨平台编译与快速验证、高差异度模块插件化、CI/CD、质量管理能力。 <br>
与后端厂商联合研发中间层通用扩展，并开源 pytorch 标准化扩展后端以支持 Triton / FlagTree 的实践。 <br>

* 两种编译路径支持

支持 TritonGPU、Linalg 两种编译路径。对非 GPGPU 后端提供多种接入范式，新增 FLIR 仓库支持基于 Linalg Dialect 扩展、、MLIR 扩展的后端编译。

* 新增 Triton 版本支持

目前支持的 Triton 版本包括 3.0.x、3.1.x、__3.2.x__、__3.3.x__，其中 __加粗__ 为本次新增。

* 硬件感知优化支持

支持为后端通用或特有的硬件特性提供指导编程接口。通过注解等兼容式地扩展，在语言层添加指导信息，提供算子编写与性能调优的灵活性。

* 与 FlagGems 算子库联合建设

在版本适配、后端接口、注册机制、测试修改等方面，与 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子库联合支持相关特性。

### Looking ahead

GPGPU 后端代码将进行整合，将后端差异化改动与 TritonGPU 解耦；非 GPGPU 后端将在 FLIR 基础上横向整合，对通用 Pass 进行统一设计。 <br>
为后端厂商提供 Triton 适配版本升级指南：3.0 -> 3.1 -> 3.2 -> 3.3。 <br>
CI/CD 将添加 FlagGems 算子库功能测试。 <br>
集成 C++ Runtime 功能，将 kernel 之外的运行时开销降低到与 CUDA 持平。 <br>
