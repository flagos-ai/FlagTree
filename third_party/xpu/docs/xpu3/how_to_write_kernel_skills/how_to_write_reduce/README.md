# how_to_write_reduce

归约类算子（sum / mean / max / min / prod / norm，以及各种 `*_dim`）在 KL3 上怎么写、
怎么定位问题、怎么测量。内容来自 FlagGems 上 `sum` / `sum_dim` 迁到 `tle.gpu` 那一轮
改动的实测记录，不是设计文档。

## 目录

| 文件 | 用途 |
|---|---|
| [SKILL.md](SKILL.md) | 方法论主干，也是一个可直接安装的 Comate skill |
| [references/tile_sizing.md](references/tile_sizing.md) | 48 点 XBLOCK 扫描、结果写回的开销阶梯、fold 的 tile 与两趟测量 |
| [references/reduce_envelope.md](references/reduce_envelope.md) | `tle.gpu` 能归约什么：轴、mask、dtype、descriptor、pipeline 门槛，每条附最小复现 |
| [references/measurement.md](references/measurement.md) | 事件计时开关、选卡、DCE 假象、compile/load/run 的区分 |
| [scripts/sweep_sum.py](scripts/sweep_sum.py) | 正确性 sweep（形状 × dtype × keepdim × out=），改 `ENTRY` 即可用于别的归约算子 |
| [scripts/sweep_tile.py](scripts/sweep_tile.py) | XBLOCK/YBLOCK 扫描模板，带正确性列 |
| [sum_case_study.md](sum_case_study.md) | 案例：FlagGems 的 `sum` / `sum_dim` 具体怎么改的，带前后数据和走错的路 |

## 三分钟版本

1. **先读 XDNN**。`baidu/xpu/api/src/wrapper_aten/internal/reduce_calc_common.cpp` 把
   任何归约都归一成 `(m, t, n)`，分派 `reduce_mt` / `reduce_tn` / `reduce_mtn`。
   `reduce_mtn`（中间轴保一个 n 宽累加器、把 t 折进循环）就是该抄的算法——它也说明中间
   轴**不能**走 `dim_compress`：1G f32 沿 dim=1 光那次转置拷贝就 5.69ms + 归约 2.50ms，
   而 aten 整个算子 2.39ms。
2. **归约轴只能是最后一轴**。`TLECoreTiling.cpp:383` 明确拒绝其他轴，所以中间轴必须两趟。
3. **tile 形状是算法，不是调参**。XBLOCK 同时决定结果写回的开销（64 行时 ~9us，512 行时
   ~0.7us）、grid（一个 cluster 一个 program 最好）、和 YBLOCK（长行要长 YBLOCK）。
   48 个点里挑错差 1.3~6.4 倍。
4. **短 tile 清零，不要 mask**；但清零只在溢出到**张量之外**时有效。连续轴溢出（读到下一
   行/下一个 batch）用**末块左移**，归约轴只能保整除或退成单趟。
5. **launch 用 `flat_launchers` 绑定一次**（省 ~9us），指针别进 specialization key
   （否则 1G 归约 767ms/call）。宿主端 Python 优化收益是 0，别浪费时间。
6. **这台设备没有真的 f64**（`torch.float64` 就是 f32）。不要写 f64 分支。
7. **测量**：`XPU_EVENT_KL3_ENABLE=1` 必须开（否则计时器返回 0 或恒定 1.25ms）；选空闲卡
   （有负载时 aten 自己能抖 33 倍）；删 kernel 尾部定位耗时会被 DCE 骗。

## 当成 skill 安装

```bash
d=third_party/xpu/docs/xpu3/how_to_write_kernel_skills/how_to_write_reduce
mkdir -p ~/.comate/skills/xpu-tle-gpu-reduce
cp -r $d/{SKILL.md,references,scripts} ~/.comate/skills/xpu-tle-gpu-reduce/
```

`SKILL.md` 的 frontmatter 里 `name` / `description` 决定触发，正文和 `references/`
按需加载。装好后问「KL3 上这个 reduce 为什么慢」「中间轴归约怎么不走 dim_compress」
「XBLOCK 该取多少」这类问题会自动带上这套上下文。

## 相关文档

- [how_to_write_copy/](../how_to_write_copy/) —— 搬运类算子（`tle.dsa` / SDNN）那条路
- [how_to_write_norm/](../how_to_write_norm/) —— 归一化类算子（group_norm / layer_norm …）；
  它的前半段就是本 skill，后半段是统计量的窄写回
- [tle_dsa_user_guide.md](../../tle_dsa_user_guide.md) —— `tle.dsa` / `tle.pipe` 的 API 与已知问题
