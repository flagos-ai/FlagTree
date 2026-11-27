# how_to_write_kernel_skills

搬运类算子（copy / permute / transpose / unfold / as_strided ...）在 KL3 上怎么写、
怎么定位问题、怎么测量。内容全部来自 FlagGems 上 `copy` 家族那一轮改动的实测记录，
不是设计文档。

## 目录

| 文件 | 用途 |
|---|---|
| [SKILL.md](SKILL.md) | 方法论主干，也是一个可直接安装的 Comate skill |
| [references/measurement.md](references/measurement.md) | 选卡、缓存、四种会得出错误结论的测量假象 |
| [scripts/sweep_copy.py](scripts/sweep_copy.py) | 正确性 sweep（布局 × dtype）+ 三方计时模板，改 `ENTRY` 即可用于别的算子 |
| [copy_family_case_study.md](copy_family_case_study.md) | 案例：FlagGems 的 `copy_` / `alias_copy` / `permute_copy` / `unfold_copy` 具体怎么改的，带前后数据 |

## 三分钟版本

1. **先读 XDNN 的手写 kernel**，不要先设计。
   `baidu/xpu/api/src/wrapper_aten/<op>.cpp` 的 `xpu3_wrapper` 告诉你哪些形状有专用
   kernel、哪些被拆开；`baidu/xpu/api/src/kernel/kunlun3cpp/kunlun3cpp_aten/*.xpu`
   是实现。两个必读：`memcpy_2d_sdnn.xpu`（`dma_cfg_2d(loop, dst_stride, src_stride)`
   —— **loop 行 × 行内连续**，这就是 DMA 的原生形状）和 `transpose_021_sdnn_bsp.xpu`
   （2D DMA 进 → 片上 `ds_shuffle_coa_1d` → 2D DMA 出，**native 从不带 stride 读 GM**）。
2. **把布局折叠后对着原生形状分类**，能表达的走 DMA，不能表达的**回落**。回落是设计，
   不是失败。
3. **三条静默算错的边界必须在 host 侧挡掉**：目标最内段带 stride、源最内段 stride 0、
   int → float 的 cast。这三条不报错、不 fault，只是数据错。
4. **尾块用 `sizes`，不要用 mask**。
5. **tile 常量只能实测**：同一个转置 64×64 是 66us，128×128 和 32×256 是 120~140ms。
6. **`first ≈ median` 说明每次调用都在重新加载 kernel**（指针参与 specialization），
   不是 kernel 慢；`first >> median` 才是编译一次。
7. **报数给三个值**：你的路径 / 回落 / native 的绝对延迟。只给比值会同时藏住基线漂移
   和「两边都很慢」。

## 当成 skill 安装

```bash
d=third_party/xpu/docs/xpu3/how_to_write_kernel_skills/how_to_write_copy
mkdir -p ~/.comate/skills/xpu-tle-dsa-copy
cp -r $d/{SKILL.md,references,scripts} ~/.comate/skills/xpu-tle-dsa-copy/
```

`SKILL.md` 的 frontmatter 里 `name` / `description` 决定触发，正文和 `references/`
按需加载。装好后问「KL3 上这个非连续拷贝为什么慢」「tle.dsa 能不能表达这个布局」
这类问题会自动带上这套上下文。

## 相关文档

- [tle_dsa_user_guide.md](../../tle_dsa_user_guide.md) —— `tle.dsa` / `tle.pipe` 的 API 与已知问题
- [how_to_write_reduce/](../how_to_write_reduce/) —— 归约类算子；[how_to_write_norm/](../how_to_write_norm/) —— 归一化类算子
- FlagGems 侧的同一份案例：`harness/solution/copy/copy_tle_dsa.md`（内容相同，便于在该仓内查阅）
