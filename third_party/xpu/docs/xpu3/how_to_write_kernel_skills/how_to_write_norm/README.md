# how_to_write_norm

归一化类算子（group_norm / layer_norm / instance_norm / batch_norm / rms_norm）在 KL3
上怎么写、怎么定位问题、怎么测量。内容来自 FlagGems 上 `native_group_norm` 迁到
`tle.gpu` 那一轮改动的实测记录，不是设计文档。

## 目录

| 文件 | 用途 |
|---|---|
| [SKILL.md](SKILL.md) | 方法论主干，也是一个可直接安装的 Comate skill |
| [references/narrow_writeback.md](references/narrow_writeback.md) | 窄写回为什么贵：消融、地址对照实验、asm 对比 |
| [references/norm_envelope.md](references/norm_envelope.md) | `tle.gpu` 在 KL3 上做归一化能做什么：scope 规则、`local_ptr` 索引语义、bf16 store 缺口、constexpr scope 的坑，每条附最小复现 |
| [references/measurement.md](references/measurement.md) | 事件计时开关、选卡、DCE 假象、canary 规则 |
| [scripts/probe_narrow_writeback.py](scripts/probe_narrow_writeback.py) | 独立 kernel，把「窄 strip 写回」单独拎出来，扫宽度/dtype/GM 间距/program 数。任何"小结果行很贵"的算子先跑这个 |
| [scripts/sweep_group_norm.py](scripts/sweep_group_norm.py) | 正确性 + 交错 A/B 计时（形状 × dtype） |
| [group_norm_case_study.md](group_norm_case_study.md) | 案例：FlagGems 的 `native_group_norm` 具体怎么改的，带前后数据和走错的路 |

## 三分钟版本

1. **统计量的写回就是这个算子。** 一个 30us 的融合 group_norm，(16,16,128)，
   tile 进出 + reduce + affine 合计 ~6.9us，而 mean/rstd 两行
   `[XBLOCK]` LM strip 的 store+copy 要 **+19.3us**。半 KB 的输出压过 44KB 的 tile 流量。
2. **原因是 8 个 cluster 撞在同一条 64B GM cache line 上。** 和拷贝无关（两个 copy 合计
   ~0.8us）、和 reduce 无关（0.5~1.1us）、和 fence 无关（26 条 mfence 删掉 2 条毫无变化）、
   和指令数无关。把同样的 16 个 f32 改成每个独占一条 64B line，+10.3us 掉到 +2.3us；
   只把**program** 之间拉开 4KB 也能掉到 +1.8us —— 证明冲突在 cluster 之间，不是
   一个 cluster 内的 64 个核。
3. **解法是走 cluster-shared SM。** `scope=tle.gpu.smem` 是全 cluster 共享的，没有 per-core
   归属问题，而且 SM 的 `local_ptr` **认索引**（LM 的会丢掉索引，只给你自己那一片），
   所以每个核丢自己那一道进去，写回由编译器合并成一条连续 `s_sm2gm`。+19.3us → +0.8us，
   三个 shape 的官方 speedup 0.6934 → 0.9687。
4. **这需要工具链改动**：`copy_l2g` 从 `smem` 出来原来是一句 assert。后加了 SM 分支和
   `emitCoalescedSM2GM`（按 64 字节切分，`min(核数, ceil(bytes/64))`
   参与）。
5. **bf16 上不了这条路**：XPU3 没有 addrspace 2 的 16-bit float store，而且 f16 / i16
   两种容器都会被 `OPTIMIZE_O3` 的 InstCombine 折回 `store bfloat`。只能 gate 回 LM 路径，
   并如实说明那 1.6x。
6. **测量**：`XPU_EVENT_KL3_ENABLE=1` 必须开（否则 `do_bench` 静默返回 0.0）；用
   `CUDA_VISIBLE_DEVICES` 选卡（`XPU_VISIBLE_DEVICES` 会在 import 阶段炸）；跑之前探空闲卡
   （有一次带着 26.8GB 残留显存的卡把某个 shape 量成 165ms，正常 18ms）；
   循环里留 aten 作 canary。

## 当成 skill 安装

```bash
d=third_party/xpu/docs/xpu3/how_to_write_kernel_skills/how_to_write_norm
mkdir -p ~/.comate/skills/xpu-tle-gpu-norm
cp -r $d/{SKILL.md,references,scripts} ~/.comate/skills/xpu-tle-gpu-norm/
```

`SKILL.md` 的 frontmatter 里 `name` / `description` 决定触发，正文和 `references/`
按需加载。装好后问「group_norm 在 XPU 上为什么慢」「mean/rstd 的写回怎么这么贵」
「norm 的统计量该走 LM 还是 SM」这类问题会自动带上这套上下文。

## 相关文档

- [how_to_write_reduce/](../how_to_write_reduce/) —— 归约那条路；归一化的前半段就是它，
  mask/清零、tile 形状、DCE 陷阱那几条在这里同样成立
- [how_to_write_copy/](../how_to_write_copy/) —— 搬运类算子（`tle.dsa` / SDNN）
