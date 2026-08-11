# Matmul Triton Kernel — 成功运行记录

**日期**: 2026-07-05
**状态**: ✅ 精度验证通过 (Test Passed!)

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 平台 | Linux aarch64 (4.19.90-2102.2.0.0066.ctl2.aarch64) |
| Python | 3.10.20 (conda env: `dlcompiler`) |
| PyTorch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| 设备 | Ascend910B2 (62420MB, 24 cube cores, 48 vector cores) |
| Triton | FlagTree 源码 editable install (`/mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree/python/triton/`) |

---

## 执行命令

```bash
conda run -n dlcompiler python /mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree/test/CommonIR/matmul_triton.py
```

### 输出

```
Test Passed!
```

---

## 测试参数

| 参数 | 值 |
|------|-----|
| M | 1024 |
| N | 1024 |
| K | 1024 |
| BLOCK_M | 128 |
| BLOCK_N | 256 |
| BLOCK_K | 128 |
| NUM_CORES | 24 |
| 数据类型 | float16 (计算 float32 累加) |
| 精度阈值 | rtol=1e-2, atol=1e-2 |

---

## Kernel 流程概述

### 算法

单循环 matmul kernel，使用 `tile_copy` + `tl.dot`：

1. Grid = `(NUM_CORES,)`，每个 core 以 round-robin 方式处理多个 `(M_tile, N_tile)` 输出块
2. 对每个输出块，沿 K 维度迭代：
   - `tile_copy`: 从 Global Memory 拷贝 A/B tile 到 L1
   - `tl.dot`: 执行矩阵乘累加 (在 L0C 上累加)
3. 将 float32 累加结果转换为 float16 写回 Global Memory

### 关键 API 调用

```python
# 片上缓冲区分配
mat_a_l1 = tile_alloc([BLOCK_M, BLOCK_K], mat_a.dtype.element_ty, L1)
mat_b_l1 = tile_alloc([BLOCK_K, BLOCK_N], mat_b.dtype.element_ty, L1)

# 数据搬运: GM → L1
tile_copy(a_block_ptr, mat_a_l1, [BLOCK_M, BLOCK_K])
tile_copy(b_block_ptr, mat_b_l1, [BLOCK_K, BLOCK_N])

# 矩阵乘累加
mat_c_acc = tl.dot(
    tile_to_tensor(mat_a_l1, writable=False),
    tile_to_tensor(mat_b_l1, writable=False),
    mat_c_acc, out_dtype=tl.float32)
```

### 精度验证

```python
ref = torch.matmul(mat_a.float(), mat_b.float()).to(torch.float16)
torch.testing.assert_close(ref, mat_c, rtol=1e-2, atol=1e-2)
```

---

## 编译流程 (Dump 模式)

脚本还支持两种 IR dump 模式（不需要 NPU 设备）：

### TTIR Dump

```bash
conda run -n dlcompiler python test/CommonIR/matmul_triton.py --dump-ttir
```

Pipeline: Python AST → TTIR (Triton IR)

### Linalg Dump

```bash
conda run -n dlcompiler python test/CommonIR/matmul_triton.py --dump-linalg
```

Pipeline:
1. ① `tileir_to_hivm` — tile.* → memref/hivm
2. ①b `erase_linalg_casts` — 消除 unrealized casts
3. ② `structure(r1)` + `discrete_mask_access_conversion`
4. ③ `unstructure` + `hivm` + `hfusion` + `llvm`
5. ④ `bubble_up` + `structure(r2)`
6. ④b `inline` + `canonicalize`
7. ⑤ `triton_to_linalg_incubated`
8. ⑤c `fold_staging_copy`
9. ⑤b `erase_linalg_casts` (post)
10. ⑥ `canonicalize` + `CSE` + `symbol_dce`

---

## 构建方式

使用 `flagtree_build.sh` 构建 (editable install)：

```bash
# 关键环境变量
export PYTHON=/root/miniconda3/envs/dlcompiler/bin/python
export LLVM_SYSPATH=/root/.flagtree/ascend/llvm-a66376b0-ubuntu-aarch64-python311-compat
export FLAGTREE_BACKEND=ascend
export MAX_JOBS=32

# 构建
cd python && $PYTHON -m pip install -e . --no-build-isolation -v
```

---

## 注意事项

1. **必须使用 `dlcompiler` conda 环境** — Python 3.10，包含 torch/torch_npu 且与编译的 `libtriton.so` ABI 兼容
2. 默认 `base` 环境 (Python 3.13) 会触发 `_PyThreadState_UncheckedGet` undefined symbol 错误
3. `flagtree` 环境 (Python 3.12) 缺少 torch 依赖
4. Triton 通过 editable install 指向 FlagTree 源码目录，而非 upstream triton 3.5.0
