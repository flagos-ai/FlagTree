# Triton TLE MegaMoE 已验证入口

本文只记录这个隔离 MegaMoE operator 的当前入口、依赖、验证证据和手动复现命令。

## 当前 production-shape 候选（2026-09-01）

| 场景 | runner | 定位 | 当前状态 |
|---|---|---|---|
| 单 rank 回归 | `megamoe_operator/production/v25/run.py` | BM64、单 math warp-group、D8 descriptorless TMA1D、wide L2 scatter | NP1 MoE-7 correctness PASS |
| 多 rank 主线 | `megamoe_operator/production/v234/run.py` | BM128、双 math warp-group、双 D8 pull stream、独立 A/SFA 与 B TMA producer、真实 W1 布局 | H100 MoE-7 np8/t512 8/8 PASS |

v234 是 `zhiyuan_megakernel` 当前 manifest 中的 multi-rank mainline，来源提交为
`fdb7f8e`，打包前 immutable kernel SHA256 为
`d57fb2252c63818a0058f936ff3ed46b9e7fadba29858f9f47eadb9526d6464b`。
本目录只调整运行时和 raw helper 的相对路径，不改变 kernel 语义。

每个 rank 只发射一次 Triton/TLE persistent kernel。Triton/TLE 保留 pipe、
WGMMA、scale、SwiGLU、L1/L2 和 combine；raw CUDA 边界仅承载 D6-D9 的
descriptorless TMA1D dispatch/pull 以及 L2 remote scatter。

## 编译器依赖

v25 依赖当前 main 已有的 pipe/warp-specialization 支持。v234 还依赖经过
验证的 PR837 TLE compiler build；算子实际走到的新增能力为：

- `buffered_tensor.subslice`：保持 rank、SMEM layout 和 allocation shape；
- 同一 pipe 上多个纯 TMA writer；
- 多 writer 推导出的 TMA token `full_count` lowering。

v234 会在导入 Triton 前设置 `TLE_MULTI_TMA_WRITERS=1`。未合入上述编译器
能力的 FlagTree 不能编译该入口。

当前已验证的组合是本 PR 中的 v234 加历史 PR837 build：H100 NP2 synthetic
correctness 为 2/2 rank PASS，历史 H100 NP8 真实数据为 8/8 rank PASS。另行在
FlagTree current main 上只叠加 `subslice` 与 multi-writer 两个精简提交的组合，
虽然能够完成编译和 launch，但 NP2 在运行时触发 CUDA illegal instruction；该
组合不能视为已支持，也不能替代完整的配套 compiler 验证。

## 统一 workload

```text
H=4096, I=1536, E=128, topk=8, tokens/rank=512, drop=0, stages=4
```

### 8-rank synthetic correctness

```bash
MEGAMOE_NP=8 W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=0 W_TIMEOUT=900 \
python python/test/tle/integration/megamoe/megamoe_operator/production/v234/run.py
```

通过标准为 8 个 rank 都输出：

```text
partials=4096 scatter_bad=0 errors=0 ... -> PASS
```

### 真实 Qwen3 FP8 数据

设置 `MEGAMOE_SHARED_DATA_DIR` 后，v234 会读取 CUDA/TLE 共用的数据集，并在
host preprocessing 中完成 checkpoint W1 的 SM90 gran-8 gate/up 交织：

```bash
MEGAMOE_SHARED_DATA_DIR=/path/to/qwen3_fp8_shared \
MEGAMOE_NP=8 W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=0 W_TIMEOUT=900 \
python python/test/tle/integration/megamoe/megamoe_operator/production/v234/run.py
```

已归档证据为 H100 np8/t512 8/8 rank PASS、`scatter_bad=0 errors=0`；独立
PyTorch oracle 的 relative L2 为 `0.0032209039`，cosine 为
`0.9999948372`。

### Event benchmark

在 correctness 通过后使用同一 workload：

```bash
MEGAMOE_NP=8 W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=1 W_WARMUP=10 W_ITERS=30 \
W_BENCH_REDUCE=mean W_GPU_START_BARRIER=1 W_TIMEOUT=900 \
python python/test/tle/integration/megamoe/megamoe_operator/production/v234/run.py
```

不同 revision、输入、时钟状态或 rank reduction 得到的 latency 不可直接与
CUDA 数字混算。本 PR 不新增跨实现性能比例声明。

## UserHopper 等价性边界

| CUDA behavior | v234 behavior | 状态 | 仍未闭合项 |
|---|---|---|---|
| SM90 W1 FP8 transform | host 固化 `[all gate | all up]` 到 gran-8 gate/up 交织 | `equivalent`（当前 H100 np8/t512 scope） | 其它模型层和 shape 尚未覆盖 |
| W1/L2 weight SF layout | W1 使用 `[EPR, NL1N, NK1]`，gate/up 六处地址已修复；L2 SF 保持 CUDA contract | `equivalent`（当前测试 scope） | 其它模型层和 shape 尚未覆盖 |
| D8 FP8 payload pull | 双 stream descriptorless TMA1D；activation SF/top-k weight 保持普通 load | `partial` | 完整 D0-D10 stage JSON 和 workspace lifecycle 尚未闭合 |
| TMA/WGMMA/WS performance path | BM128、双 math WG、独立 A/SFA 与 B TMA producer | `partial` | 仍需相同输入和相同计时口径下的 UserHopper matched 复测 |
| combine/cleanup | 单 kernel 内完成当前 combine 与输出校验 | `partial` | UserHopper TMA/SMEM/mbarrier combiner 和完整 epoch/reuse contract 尚未闭合 |
