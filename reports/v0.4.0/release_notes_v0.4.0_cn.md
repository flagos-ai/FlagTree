<div align="right"><a href="./release_notes_v0.3.0.md">English</a></div>

## FlagTree 0.4.0 Release

### Highlights

FlagTree 继承前一版本的能力，持续集成新的后端，壮大生态矩阵。项目在打造代码共建平台、实现单仓库多后端支持的基础上，持续建设后端特化统一设计，持续建设中间层表示及转换扩展（FLIR），提升硬件感知和编译指导支持能力与范围（flagtree_hints）。

### New features

* 新增多后端支持

目前支持的后端包括 nvidia、amd、triton_shared cpu、iluvatar、xpu、mthreads、metax、aipu、ascend、tsingmicro、cambricon、hcu、__enflame__，其中 __加粗__ 为本次新增。 <br>
各新增后端保持前一版本的能力：跨平台编译与快速验证、高差异度模块插件化、CI/CD、质量管理能力。 <br>

* 新增 Triton 版本支持

新增支持 Triton 3.4、3.5 两个版本，并建立对应的保护分支。

* 持续建设 FLIR

持续进行 Linalg 中间层表示及转换扩展、MLIR 扩展，提供编程灵活性，丰富表达能力，完善转换能力。确定多后端接入 FLIR 范式，集成 ascend 后端。

* FlagTree 后端特化统一设计

FlagTree 设计的后端统一特化，目的是整合后端接入范式，对后端的特化实现清晰化管理，为后端适配 Triton 版本升级迁移提供工程基础。详见 [FlagTree_Backend_Specialization](/documents/decoupling/)，已应用于 iluvatar、ascend 等后端。

* 编译指导 flagtree_hints

在 GPGPU 上支持编译指导 shared memory + async copy 并在 triton_v3.5.x 分支上验证。HINTS更多信息[wiki](https://github.com/flagos-ai/FlagTree/wiki/HINTS)。

* 低层次指令编写 Tle-Raw

在 Nvidia GPU 上支持直接使用MLIR/LLVM进行部分的关键代码编写，绕开Triton语法限制，并在 triton_v3.5.x 分支上验证。更多内容可参考[wiki](https://github.com/flagos-ai/FlagTree/wiki/EDSL)。

* 与 FlagGems 算子库联合建设

在版本适配、后端接口、注册机制、测试修改等方面，与 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子库联合支持相关特性。FlagGems 算子库当前已适配至 Triton 3.5。

### Looking ahead

FLIR 计划集成更多后端，2026 Q1 完成 tsingmicro 后端接入。 <br>
保护分支 triton_v3.4.x 计划接入新后端。 <br>
