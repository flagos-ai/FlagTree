# 案例：FlagGems 的 copy 家族改走 tle.dsa（SDNN）

方法论见 [SKILL.md](SKILL.md)，这里是它的来源：一次具体改动的完整记录，包含前后数据和
走过的弯路。

- 仓库：FlagGems，`src/flag_gems/runtime/backend/_kunlunxin/`
- commit：`eb8304d0b [kunlunxin] move the copy family onto tle.dsa`（另有
  `c5c2c8db6` 清理死代码）；同一份文档也放在该仓 `harness/solution/copy/copy_tle_dsa.md`
- 算子：`copy_` / `copy`、`alias_copy`、`permute_copy`、`unfold_copy`
- 硬件：KL3（arch 3），全部实测在卡 1 / 卡 3

文中提到的文件路径若无前缀，均相对 FlagGems 仓库根目录；XDNN 手写 kernel 在
`baidu/xpu/api/`。

## 1. 基线与问题

`tle_copy` 原来分两条路：两侧连续同 dtype 走 `tle.gpu` 的扁平 TMA tile；其余走
`tle.gpu` 的**逐元素 gather**（每个 block 物化一个 GM 指针向量，靠 lowering 降成
g2l/l2g）。gather 那条是正确的，但慢得离谱：

| case | 老 gather | aten | pointwise |
|---|---|---|---|
| 4096×4096 f32 strided window | **32.8ms** | 157us | 31.7ms |
| 1024×1024 f32 转置 | 2.00ms | 11.2us | 12.1ms |
| 4096×4096 f32 广播行 | 30.9ms | 49.8us | — |

即「tile 路径接不住的所有布局都比什么都不做还差 ~200 倍」。算子层的表现：
`permute_copy` dtype 等权 **0.461**（其中 `[32,64,128] perm=[2,0,1]` 是 **0.012x**），
`unfold_copy` **1.556**（6 个 case 是 0.000x）。

根因不是参数没调好：**一个元素一个 descriptor 不是这个 DMA 引擎做的事**。

## 2. 硬件实际能做的形状

参照 XDNN 手写 kernel（`baidu/xpu/api/src/kernel/kunlun3cpp/kunlun3cpp_aten/`）：

- `memcpy_2d_sdnn.xpu`：`dma_cfg_2d(loop, dst_stride, src_stride)` —— **loop 行 × 行内连续
  len 字节，src/dst 行 stride 各自独立**
- `transpose_021_sdnn_bsp.xpu`：3 核 BSP 流水，cid0 `dmai_2d` 搬进 uni_sram、cid1 用
  **`ds_shuffle_coa_1d` 片上转置**、cid2 `dmao_2d` 搬出，uni_sram 分两半 ping-pong

关键结论：**native 从不带 stride 读 GM**。转置是在片上完成的，两侧 GM 都是「行内连续」的
2D DMA。

## 3. 改成三条路径

| 路径 | 命中条件（collapse 之后） | 形状 |
|---|---|---|
| `_tle_tile_copy_kernel`（tle.gpu） | 两侧连续 + 同 dtype | 扁平 TMA tile，`_tile_launch` 绑定一次复用 |
| `_tle_dsa_row_copy_kernel`（tle.dsa） | 最内维两侧都连续（`s_col == 1`） | 行 × 连续段的 2D DMA，行 stride 各自任意 |
| `_tle_dsa_trans_copy_kernel`（tle.dsa） | 最内维 dst 连续、src 在次内维连续 | 64×64 tile → `tl.trans` → 2D tile 出 |

三条之外 `tle_copy` 返回 False，调用方用自己的 pointwise kernel。

实现要点：

- **维度折叠**：`_collapse` 按 `|dst stride|` 排序并合并，得到最内维（列）+ 次内维（行）+
  最多 3 个外层维；外层维度算成**标量基址**挂在 grid 上，传输本身不受影响
- **尾块用 `sizes`，不能用 mask**：masked load 把 extent 藏在 staging buffer 的 view 里，
  dsa 改向时会丢掉，编译器直接报错。出方向同一个 `sizes` 变成 store mask
- **转置 tile 必须是方的且等于 64×64**：转置后的 tile 复用同一 buffer；128×128 和 32×256
  在 2048² f16 上都是 120~140ms，64×64 是 66us
- **dtype 转换**在两次传输之间的 tensor 侧做，`TO_BOOL` 复现「非零即真」
- **纯搬运按位型走**：1/2/4 字节直接搬，8 字节用 `view(torch.float32)` 拆成两个 4 字节
  （它只在 `stride(-1) == 1` 时允许，正好是行路径要求连续的那一段）

## 4. 三条静默算错的边界（必须挡掉）

这些都在真卡上实测，而且**不报错**，所以只能在 host 侧提前拒绝：

1. **最内段带 stride 的写**（`y[::2]`）：stride 被丢掉，按连续打包写
2. **最内段 stride 0**（从单个元素广播）：读成连续元素
3. **int → float 的 cast**：结果全 0

第 3 条和 dsa 无关——不带 tle 的普通 `tl.load(...).to(tl.float32)` 配 `is_sdnn=True`
一样错，是 SDNN cast 的问题。native 的转换走的是 cluster 上的 `cluster_cast_kl3`（SIMD），
不是 SDNN。

另外 dtype 侧的硬限制：i32 作 DMA 元素类型是 illegal memory access，i64 编译期报
`Unsupported dma dst data type`，bf16 不能作 dsa buffer 的元素类型
（`bufferization.to_tensor` 报 element types do not match，但作为 cast 的**目标**没问题）。
纯搬运按位型走可以绕开前两条。

## 5. `do_not_specialize`：不是编译一次，是每次调用都重新加载

调用方每次都新分配 `out`，XPU caching allocator 给回的指针 divisibility 分类会变，指针
参与 specialization key 就导致**每次调用换一份 kernel**。热缓存下命中一个新 key 大约
10~30ms（不是冷编译的 1~4s），所以现象是「first ≈ median」，很容易被误判成 kernel 慢。

同一组 6 个 strided copy，同一份代码：

| | 整轮墙钟 | f16 1536² 转置 | f32 1024² 转置 |
|---|---|---|---|
| 指针参与 specialization | **482s** | 14.7ms | 25.0ms |
| `do_not_specialize` 全部运行时参数 | **12s** | 334us | 199us |

判据：**first 和 median 接近就是每次都在换 kernel；first 远大于 median 才是编译一次**。

## 6. 尺寸门槛：最后删掉了

一开始按字节设了 `DSA_MIN_BYTES = 32MB`（对照 aten 测的）。后来 fallback 从 aten 换成
Triton pointwise kernel，依据失效，重测：

| 大小 | dsa | pointwise |
|---|---|---|
| 32~64KB | 106~111us | 103~107us |
| 512KB~1MB | 110~115us | 104~105us |
| 2~4MB | 107~109us | 113~120us |
| 8~16MB | 120~137us | 141~179us |
| 32~64MB | 156~216us | 260~410us |

1MB 以下持平（都是 host/launch 地板），以上 dsa 领先到 1.9 倍。小端 3~8% 的劣势换不来一个
门槛 + 一个环境变量 + 一个 per-caller 参数，所以三个一起删掉，所有调用方同一条规则：
**tle 能表达就走 dsa**。

## 7. 结果

转置一类（f16，与 native 对照）：

| 大小 | 片上转置 | pointwise | native |
|---|---|---|---|
| 8KB | 83us | 124us | 71us |
| 128KB | 96us | 211us | 68us |
| 2MB | 90us | 2086us | 66us |
| 8MB | 110us | 8117us | 71us |
| 32MB | **207us** | 31683us | 246us |

算子层（`tools/run_tests.py`，卡 1，dtype 等权）：

| 算子 | 改动前 | 改动后 |
|---|---|---|
| `permute_copy` | 0.461 | **1.114 / 1.117 / 1.122**（三轮） |
| `unfold_copy` | 1.556 | **2.66 / 3.16 / 2.35**（三轮） |
| `copy` | ~0.99 | ~0.99（连续路径未动） |
| `alias_copy` | ~0.98 | ~0.98 |

60 个 case 里没有低于 0.5x 的（原来有 0.012x 和 0.000x）。accuracy：
`test_copy` 42 + `test_alias_copy` 18 + `test_permute_copy` 18 + `test_unfold_copy` 24，
合计 **120 passed / 4 skipped**。

## 8. 顺带修掉的两个既有 bug

回落路径一旦被真正走到，就暴露了两个原先被 gather 掩盖的问题：

1. **这个后端的 `copy_` 不广播**：`(3,)` 拷到 `(2,3)` 只填第一行（`aten.copy_` redispatch
   和普通 `dst.copy_(src)` 都是）。回落必须自己传 `src.expand(dst.shape)`
2. **`flag_gems.ops.unfold_copy` 通用实现对重叠窗口是错的**：`(4,8)` 取 `size=3, step=1`
   错 12/72 个元素。所以 `unfold_copy` 把回落当最后兜底，不当等价物

第 2 条也是为什么 `copy_` 的最后一步现在是 Triton pointwise kernel 而不是 aten：FlagGems
本来就该是 Triton 实现。仍走 aten 的只有语义分支（zerotensor、src/dst alias 的重叠写、
float8_e8m0fnu、`numel > 2^31-1`、非 strided layout、complex 的 warning、空 tensor 的广播
校验）。

## 9. 测量陷阱（这一轮踩了三次）

- **计时退化成固定常数**：benchmark 出现所有 case 延迟都等于两个常数（6.81/13.66ms 或
  6.44/12.88ms，正好 2 倍关系）、torch 侧和 gems 侧都一样时，这轮数据作废，不是性能变化。
  正常量级是 5~12us
  机制（从代码和数值形态推出，未做 profile）：`benchmark/base.py` 的 kernel 模式走
  `triton.testing.do_bench`，它每次迭代前清一块 cache buffer，在这个后端是 **256MB**
  （`python/triton/backends/xpu/driver.py` 的 `get_empty_cache_for_benchmark`）。这个
  memset 的耗时与 shape 无关、正好毫秒级；一旦 event 的 start/end 没把它排除掉，每个
  case 量到的就是「一个或两个 memset」。而且会级联：`do_bench` 用初始估计定迭代数，
  估计变成 6.4ms 后 `n_repeat` 从上千掉到十几，median 也就没有样本可落了
- **基线漂移**：`copy` 的 bf16 曾出现 2.90x，看着是提升，其实是 torch 侧 0.540ms vs 平时
  0.082ms；gems 侧两轮一致。**先看两侧绝对延迟再看比值**
- **冷缓存污染**：每轮 `FLAGGEMS_CACHE_DIR=$(mktemp -d)` 会把编译缓存全扔掉，小 shape 的
  数会被 1~4s 的编译或 10~30ms 的 kernel 加载吃掉。定性阶段用固定缓存目录（本轮
  `/tmp/gems_cache_c1`），只在最终验收时用 fresh cache

## 10. 还缺什么（相比 pointwise_dynamic）

pointwise 是按输出逐元素取址，任意 stride/dtype/rank 都能算，所以缺口全在 tle 这边：

- 8 字节浮点：`float64` 在这个 xpytorch build 上到设备就变 float32，无从验证
- int → float 转换：SDNN cast 全 0，要补得在 cluster（tle.gpu）上另写 cast kernel；而
  `copy_` 现在的回落已经用上 native 同一条 `cluster_cast_kl3`，差距只有 1.1~1.5 倍，
  不成比例
- 目标最内段带 stride、最内段广播：native 也没有通用 strided-write kernel
  （`as_strided_kl3` 输出恒定线性写），这条是对齐而不是缺口
- collapse 后 > 5 维
- 小尺寸转置仍是 native 的 0.64~0.86x：native 有 3 核 BSP + ping-pong 双缓冲，我们一个
  program 一个 buffer，只靠编译器跨 program 流水。大尺寸（≥32MB）我们反而快 1.1~1.2 倍
