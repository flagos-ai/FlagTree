# FlagMega 工作区交接（2026-10-15）

这是基于 `b60fea070` 的开发检查点，目标分支为 `feature/flagmega`。
本次保存当前源码、回归测试和验证状态；完整 GLM 四卡推理仍未验收。

## 源码范围

- 增加 `control.for`、循环携带值、parameter bank、类型与序列化契约，
  以及相应的 e-graph、分布、bufferization 和代码生成支持。
  当前完整 GLM importer 仍展开 45 层，尚未使用循环压缩重复层。
- 增加 GLM-5.3-Flash 的 importer、KDA、mHC、稀疏注意力与状态操作，
  补充 producer 输出 packed type、函数边界传播和 packed 表示成本契约。
- 支持跨 chip 的紧凑权重布局、descriptor table 和 Boxing，
  reusable function 通过 pool origin 与调用记录传递不同权重及 workspace。
- 恢复并整合真实的 rdata v1–v5 读写实现，包括按 chip 物理切分、
  分块摘要、源文件快照与地址映射，避免生成产物时反复重排、复制和 hash。
- 修复 checkpoint 重复 mmap、通信 context 指针顺序、销毁后的 context
  失效和部分初始化后的资源清理；增加对应回归。

修改集中在 FlagMega 和 TLE 的 Python 实现及测试，没有修改 native
compiler 的 C++ 后端。现有 QKV 私有统计融合和缓存 epoch 改动保留。
`skills/flagmega-developer` 的 IR SOP、验收规范、结构化循环与 BI-V150
硬件说明均保留，本次未修改 skill。

## 已定位的根因与回归

### Boxing 的 owner 维度混用

位置：`codegen/triton/kernel_call_renderers.py` 的
`_cross_chip_partial_source` / `_partial_boxing_leaf`。
触发条件：跨 chip partial reduction 同时改变输出 owner 的映射。
原实现把逐输出 owner 向量当作 reduction member 向量，导致期望的
`1 × 256` 读取扩展为 `256 × 256`，引入错误地址组合和寄存器压力。
修复把输出 base 与 reduction member 分开，base 并入逐输出 offset，
member 只枚举 reduction 维度；没有缩小模型或放宽资源检查。

`test_partial_reduce_owner_axes.py` 修复前四例失败，修复后通过。
真实四卡的六种 Boxing 场景通过，每种场景执行四次并核对完整输出。
完整模型 JIT 的 spill load/store 总量从 26,044 降到 828 字节，
剩余 spill 仍违反零 spill 契约，不能据此宣称完整模型可运行。

### Checkpoint 重复映射

位置：`importer/checkpoint.py::DirectoryCheckpoint.load_tensor`。
逐权重重新打开 shard，而返回的 tensor 持有原映射，完整模型因此耗尽
文件映射数。修复按文件身份和 device 复用 reader，限制缓存数量，并在
文件替换时更新 reader；活跃 tensor 继续持有其有效存储。
`test_shared_tensor_mapping.py` 覆盖共享映射、替换、并发和缓存驱逐；
共享映射断言修复前失败，修复后通过。未提高系统映射上限。

### 通信与测试 ABI

TLE device context 的 source view 使用 `(mem, comm)`，launcher 的命名参数
仍为 `(comm, mem)`。修复索引访问与 packed data 不一致的问题，并在
FlagCX teardown 后失效 singleton，清理时只释放实际创建的资源。
对应测试位于 `python/test/tle/unit/test_distributed_context_abi.py` 和
`test_communication_cleanup.py`。前者修复前四例失败；后者复现未创建
window 时的错误释放。相关修复后测试通过。

提交前还修正 pipeline 验收 helper，使其从当前 runtime 的
`tuple[PreparedKernel, ...]` 中取单个 kernel；原失败为读取 tuple 上不存在的
`compiled_kernel`，数值执行和零 spill 断言此前已经通过。
同时补齐 provider 清单中的 paired KV cache 与 QKV gather/reduce 操作，
将 descriptor 测试的默认选择更新为当前 catalog 已采用的 TN64；
仍精确检查 provider 集合和禁止复制权重的单 descriptor 候选。
这些集成检查及通信生命周期测试共 26 项通过（12.40 秒）。

## 完整模型验证边界

输入为原始 45 层 GLM-5.3-Flash FP8 checkpoint，revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`，未缩减层数、专家数或
4096 token cache。TP4 placement 为 `cyx=4x8x16/cbb`。

| 检查 | 已记录结果 | 边界 |
| --- | --- | --- |
| 完整 import | 147,695 个节点、4 个函数 | 仍为展开的层图 |
| compute body Pack 检查 | 已检查的编译边界为 0 | 不替代数值验证 |
| 最新 SM90 TMA trial | compile 203.92 秒，publish 51.06 秒，总计 286.46 秒 | 含 load 等开销，尚未达到端到端 4 分钟 |
| 另一正常 compile | pass 时间合计 169.27 秒 | 不是端到端时间 |
| 每 chip readonly data | 80,178,146,560 字节 | TP4 override 的计划值 |
| 每 chip 最小预算 | 83,457,247,328 字节 | 包括 external state 和 workspace，未证明实际可运行 |
| 完整模型 JIT | 最近已检查的 source 仍有 828 字节 spill | 零 spill 验收失败；最新 TMA trial 尚未做此检查 |
| CPU IR 参考 | 输入 token 11，首步输出 88，logits 有限 | 不是 GPU 推理或 native serving-runtime 验收 |
| 四卡 Boxing | 6 项通过，111.81 秒 | 不是完整模型推理 |

此前还通过 artifacts / checkpoint / NumPy recipe 的 154 项测试，
MHC / KDA 实际 device kernel 的 42 项测试，以及相关布局和 ABI 回归。
这些是各自范围的结果，不能相加为一次完整 suite 的通过数。

## 提交前检查与未通过项

本次选择 126 个变更测试文件，在单张 H800 上运行：
`1124 passed, 6 failed, 19 skipped, 1 xfailed`，耗时 422.04 秒。
该结果来自修正测试集成问题之前已启动的进程。provider 清单与
descriptor 默认选择这两项随后修正，相关定向复测为 `26 passed`。
四卡用例在本次单卡运行中跳过，前述独立四卡结果保留。

仍有以下四个失败用例，保留原断言，尚未处理：

- `importer/numerics/test_qwen3_vllm.py::test_preserves_function_reuse_and_round_trips`：
  测试期待 Qwen3 importer 生成 `control.for`，当前 importer 没有该节点。
- `passes/tir/bufferize/test_formal_buffer_alignment.py::test_vector_view_of_an_offset_subspan_satisfies_its_formal_abi`
  的两组参数 `[1]`、`[None]`：`BufferPlanner._plan_tensor_subspan` 抛出
  `Tensor subspan does not satisfy consumer alignment`。根因仍待定位。
- `targets/nvidia/test_implementation_model.py::test_sm90_package_grid_excludes_chip_and_node_axes`：
  测试向 `sm90_codegen_package_plan` 传入 `capability=None`，当前接口读取
  `compute_capability` 时抛出 `AttributeError`，需核对 fixture 与接口契约。

347 个变更 Python 文件通过语法检查，`git diff --check` 通过。
本检查点没有把上述失败标记为通过或跳过，也不作为合入验收结论。

## 后续入口

关键回归可独立运行：

```sh
conda run -n flagtree pytest -q \
  python/test/flagmega/codegen/triton/kernels/distributed_boxing/test_partial_reduce_owner_axes.py \
  python/test/flagmega/importer/test_shared_tensor_mapping.py \
  python/test/tle/unit/test_distributed_context_abi.py \
  python/test/tle/unit/test_communication_cleanup.py
```

四卡测试需通过 `CUDA_VISIBLE_DEVICES` 暴露四张空闲 SM90 卡，运行
`python/test/flagmega/codegen/triton/kernels/distributed_boxing/test_cross_chip_runtime.py`。

后续仍需定位完整调用组合中的剩余 spill，验证最新 TMA 选择，再执行
完整四卡多步推理，对比 logits、token、KDA/KV/DSA 状态和性能。
04 教程的公开复现脚本尚未整理完成。私有日志、IR dump、checkpoint、
rdata 与实验产物继续保留在 gitignored 的 `.local/` 中。
