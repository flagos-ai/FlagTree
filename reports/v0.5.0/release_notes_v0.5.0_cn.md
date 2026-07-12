[中文版|[English](./release_notes_v0.5.0.md)]

## FlagTree 0.5.0 Release

### Highlights

FlagTree 继承前一版本的能力，持续集成新的后端，壮大生态矩阵。项目在打造代码共建平台、实现单仓库多后端支持的基础上，持续建设后端特化统一设计，持续建设中间层表示及转换扩展（FLIR），提升硬件感知和编译指导支持能力与范围（HINTS），对 Triton 语言进行扩展（TLE）。

### New features

* 新增多后端支持

目前支持的后端包括 nvidia、amd、triton_shared cpu、iluvatar、xpu、mthreads、metax、aipu、ascend、tsingmicro、cambricon、hcu、enflame-gcu300、__enflame-gcu400__、__sunrise__，其中 __加粗__ 为本次新增。 <br>
各新增后端保持前一版本的能力：跨平台编译与快速验证、高差异度模块插件化、CI/CD、质量管理能力。 <br>

* 新增 Triton 版本支持

新增支持 Triton 3.6 版本，并建立对应的保护分支。

* 持续建设 FLIR

持续进行 Linalg 中间层表示及转换扩展、MLIR 扩展，提供编程灵活性，丰富表达能力，完善转换能力。确定多后端接入 FLIR 范式，新增集成 tsingmicro 后端，更新 ascend 后端，支持 TLE 在 DSA 芯片的编译降级。

* Triton 语言扩展：TLE

本版本持续演进 TLE（Triton Language Extentions），重点补充了三个层级的新增能力。详见 [wiki](https://github.com/flagos-ai/FlagTree/wiki/TLE)。

**TLE-Lite** 在本版本引入了分布式原语，并新增 `extract_slice` 与 `insert_slice`，进一步提升了在多设备/多卡场景和 Tensor 切片相关场景下的表达能力与可用性，继续保持对各类硬件后端的兼容。

**TLE-Struct** 在本版本新增了 `local_ptr`，增强了结构化层级下对本地存储与数据访问路径的表达能力，便于进一步性能优化。

**TLE-Raw** 在本版本支持了 inline CUDA，可直接嵌入 CUDA 代码以获得更细粒度的硬件控制与性能调优空间。更多内容可参考 [wiki](https://github.com/flagos-ai/FlagTree/wiki/TLE-Raw)。

* 与 FlagGems 算子库联合建设

在版本适配、后端接口、注册机制、测试修改等方面，与 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子库联合支持相关特性。FlagGems 已添加使用 TLE 编写的算子。

### Looking ahead

计划将 triton_v3.6.x 变更为 main 分支。<br>
TLE-Lite 计划扩展分布式原语在多卡多机场景下的支持，并推进编译器后端在 layout 上的优化。<br>
TLE-Struct 计划开放更多硬件相关原语，并提升性能接近硬件原生语言。<br>
TLE-Raw 计划在算子中验证性能提升机会，优化与 Triton 的衔接，同时寻找其他可用的接入语言。<br>
