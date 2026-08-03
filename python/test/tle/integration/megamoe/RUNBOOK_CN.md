# Triton TLE MegaMoE 已跑通 Case

本文档是当前目录唯一说明文档，只记录这个隔离 MegaMoE operator 已跑通的 case 和手动复现命令。

## 当前 production-shape 候选（2026-08-03）

当前单卡与多卡候选已更新为：

| 场景 | runner | 结构 | 当前状态 |
|---|---|---|---|
| 单卡 | `megamoe_operator/production/v25/run.py` | BM64、单 math warp-group、D8 descriptorless raw TMA1D、v23 wide L2 scatter | NP1 MoE-7 correctness PASS |
| 8 rank | `megamoe_operator/production/v33/run.py` | BM128、双 math warp-group、SMEM expert count、并行 NVLink signal、单 D8 TMA1D stream | MoE-7 8/8 correctness PASS |

两者都是一个 rank 一次 Triton/TLE persistent-kernel launch。raw CUDA 边界只承载
D8 TMA1D 和 L2 scatter；routing、TLE pipe、WGMMA、SwiGLU、scale 和 combine
仍在 Triton/TLE kernel 中。

### 编译器基线

这两个 production-shape runner 需要新的 pipe API，以及 warp-specialized
helper 中 `tle.local_pointers` 的安全 inlining。PR837 的第一版实现曾被 PR845
回退；更窄的 TLE-specific 实现随后由 PR859 重新合入 `main`。本分支因此直接
forward-port 到包含 PR859 的 upstream `main`，不携带已回退的 PR837 实现。

### 统一 workload

```text
H=4096, I=1536, E=128, topk=8, tokens/rank=512, drop=0, stages=4
```

单卡 v25 correctness：

```bash
MEGAMOE_NP=1 W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=0 W_D8_TMA1D_LEVEL=2 \
python python/test/tle/integration/megamoe/megamoe_operator/production/v25/run.py
```

8-rank v33 correctness：

```bash
MEGAMOE_NP=8 W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=0 W_D8_TMA1D_LEVEL=2 \
W_SMEM_EXPERT_COUNT=1 W_FAST_NVLINK_BARRIER=1 W_D8_PULL_STREAMS=1 \
python python/test/tle/integration/megamoe/megamoe_operator/production/v33/run.py
```

benchmark 在上述命令中改为：

```text
W_BENCH=1 W_WARMUP=10 W_ITERS=30 W_BENCH_REDUCE=mean
```

本次 forward-port 后复测：

| candidate | correctness | event latency |
|---|---:|---:|
| v25 / NP1 | 1/1 PASS；4,096 partial rows；`scatter_bad=0 errors=0` | 1,272.5 us（PR837-compatible validation build） |
| v33 / NP8 | 8/8 PASS；每 rank 4,096 partial rows；`scatter_bad=0 errors=0` | rank mean 473.54 us，rank std 4.70 us |

参考 CUDA event baseline 为 335.8 us，故本次 v33 复测为 CUDA 的约 70.9%。

### 等价性边界

| CUDA behavior | 当前 TLE behavior | 状态 | 未闭合项 |
|---|---|---|---|
| D8 FP8 payload TMA pull | descriptorless TMA1D；SF/top-k 保持普通 load | `partial` | stage-specific lifecycle evidence 尚未覆盖完整 CUDA contract |
| L1/L2 WGMMA 与 combine | 单 kernel 内完成并通过当前 exact/数值检查 | `partial` | workspace cleanup/reuse 和全部 UserHopper stage parity 未闭合 |
| 双 math warp-group | v33 BM128 两个 math WG | `partial` | 双 D8 pull stream 仍被 layout lowering 阻塞，默认保持 1 stream |
