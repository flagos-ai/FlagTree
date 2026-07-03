# FA 3-Task Kernel: 32×32 成功编译运行记录

## 概述

使用 `TLE_REPLACE_IR_FILE` 机制，将 arch kernel 的 linalg IR 注入 dummy kernel 的编译流水线，在 Ascend 910B 上成功编译并运行 32×32 tile 的 flash attention 3-task mix-mode kernel。

**结果**: 编译通过，kernel 启动无 crash，输出为 NaN（同步问题待调试）。

---

## 执行过程

### 1. 生成 32×32 linalg IR（arch kernel）

```bash
# fa_triton_arch.py 设置 BLOCK_M=32, BLOCK_N=32, DIM=32
python fa_triton_arch.py --dump-linalg=temp_mlir/tmp_fa_linalg_32.mlir --S 1024
```

输出:
```
[dump_linalg] ① tileir_to_hivm: verify=True
[dump_linalg] ①b erase_linalg_casts: verify=True
[dump_linalg] ② structure(r1)+discrete_mask: verify=True
[dump_linalg] ③ unstructure+hivm+hfusion+llvm: verify=True
[dump_linalg] ④ bubble_up+structure(r2): verify=True
[dump_linalg] ④b inline+canonicalize: verify=True
[dump_linalg] ⑤ triton_to_linalg_incubated: verify=True
[dump_linalg] ⑤c fold_staging_copy: verify=True
[dump_linalg] ⑤b erase_linalg_casts (post): verify=True
[dump_linalg] ⑥ final cleanup: verify=True
[dump_linalg] verify=True; wrote Linalg IR (67927 chars) to temp_mlir/tmp_fa_linalg_32.mlir
```

### 2. IR 特征

```
func.func @flash_attention_fwd_3task_kernel(
  %arg0: memref<?xi8>,          -- syncBlockLock (compiler injected)
  %arg1: memref<?xi8>,          -- workspace (compiler injected)
  %arg2-4: memref<?xf16> {tt.tensor_kind = 0},  -- Q, K, V
  %arg5: memref<?xf16> {tt.tensor_kind = 1},    -- Out
  %arg6-8: memref<?xf16> {tt.tensor_kind = 2},  -- workspace_s/p/pv
  %arg9-10: memref<?xf32> {tt.tensor_kind = 2}, -- workspace_rescale/expsum
  %arg11: f32,                   -- sm_scale
  %arg12-34: i32,               -- Python kernel 的 23 个 i32 参数
  %arg35-37: i32,               -- 死参数 (未使用，编译器注入的 padding)
  %arg38-40: i32                 -- grid dims (program_id X/Y/Z)
) attributes {
  SyncBlockLockArgIdx = 0, WorkspaceArgIdx = 1,
  mix_mode = "mix", parallel_mode = "simd"
}
```

- 总参数: 41 个
- mix_mode = "mix" (cube + vector 双流)
- tensor_kinds = [0, 0, 0, 1, 2, 2, 2, 2, 2]

### 3. Dummy Kernel 关键修改

#### 3.1 添加 3 个 padding 参数（解决 arg layout 不匹配）

```python
def flash_attention_fwd_3task_kernel(
    Q, K, V, Out,
    workspace_s, workspace_p, workspace_pv,
    workspace_rescale, workspace_expsum,
    sm_scale,
    B, Hq, Hkv, S,
    sQb, sQh, sQs, sQd,
    sKb, sKh, sKs, sKd,
    sOb, sOh, sOs, sOd,
    num_seq_blocks, heads_q, gqa_group,
    num_kv_blocks, conbined_block_num, block_num_per_core, rem_block_num,
    _pad0, _pad1, _pad2,     # <-- 死参数，匹配 linalg IR layout (arg35-37)
    CB: tl.constexpr,
    ...
):
```

调用处:
```python
flash_attention_fwd_3task_kernel[grid](
    ...
    num_kv_blocks, conbined_block_num, block_num_per_core, rem_block_num,
    0, 0, 0,  # _pad0, _pad1, _pad2
    CB=CB,
    ...
)
```

#### 3.2 添加 scope 标记（尝试触发 mix-mode，实际效果有限）

```python
with tle.scope(core_mode="cube"):
    pass
with tle.scope(core_mode="vector"):
    pass
```

### 4. 运行命令

```bash
rm -rf ~/.triton/cache
export TLE_REPLACE_IR_FILE=test/CommonIR/temp_mlir/tmp_fa_linalg_32.mlir
python test/CommonIR/fa_triton_dummy.py --D 32 --S 1024 --no-check
```

### 5. 运行输出

```
/mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree/test/CommonIR/fa_triton_dummy.py:512:
  UserWarning: Cannot create tensor with interal format while allow_internel_format=False,
  tensor will be created with base format.
  out = torch.empty_like(q)
[TLE] Replacing linalg IR with external file: test/CommonIR/temp_mlir/tmp_fa_linalg_32.mlir
Reference check skipped.
```

**无 crash，无 MTE DDR out-of-range 错误，kernel 成功执行完毕。**

### 6. 编译产物 metadata (JSON)

```json
{
  "kernel_name": "flash_attention_fwd_3task_kernel",
  "arch": "Ascend910B2",
  "mix_mode": "mix",
  "parallel_mode": "simd",
  "shared_mem_dynamic_size": 221184,
  "tensor_kinds": [0, 0, 0, 1, 2, 2, 2, 2, 2],
  "bs_task_type": 32,
  "shared": 1,
  "name": "flash_attention_fwd_3task_kernel mix",
  "multibuffer": false
}
```

### 7. 正确性验证

```bash
python test/CommonIR/fa_triton_dummy.py --D 32 --S 64
```

```
Mismatched elements: 129924 / 131072 (99.1%)
Greatest absolute difference: nan at index (0, 0, 5, 16)
```

输出包含大量 NaN，少量 -0/0。

---

## 问题分析

### 已解决: MTE DDR Out-of-Range (arg layout 不匹配)

**根因**: Runtime JIT 打包 args 结构为:
```
sync_ptr + ws_ptr + [Python non-constexpr args] + gridX + gridY + gridZ
```

而 IR 的 binary 期望 41 个参数（包含 3 个死参数在 Python args 和 grid 之间）。
没有 padding 时，binary 在 arg38 位置读 program_id，但 runtime 在该位置放的是 gridX 后面的内存（越界），导致 MTE 访问非法地址。

**修复**: 在 dummy kernel 中插入 3 个 `_pad0, _pad1, _pad2` i32 参数，使 runtime struct 与 binary 的 arg 布局完全对齐。

### 待解决: NaN 输出 (mix-mode 同步问题)

可能原因:
1. Mix-mode 下 cube/vector 两个流的 pipe barrier 同步未正确工作
2. Ring buffer 的写入-读取顺序在 bishengir 调度后出错
3. Workspace 初始化问题（ring slot 在首次使用前未被写入）

---

## 关键发现

| 项目 | 值 |
|------|-----|
| 目标硬件 | Ascend 910B2 |
| UB 大小 | 192KB (1,572,864 bits) |
| 32×32 编译 | ✅ 成功 |
| 32×32 运行 | ✅ 无 crash |
| 32×32 正确性 | ❌ NaN |
| 64×64 编译 | ❌ UB overflow (5.3Mbit) |
| 128×128 编译 | ❌ UB overflow (21Mbit) |
| Arg layout fix | 插入 3 个 i32 padding 参数 |
| `_parse_linalg_metadata` | 正确提取 mix_mode/tensor_kinds |

---

## 文件路径

- Arch kernel: `test/CommonIR/fa_triton_arch.py`
- Dummy kernel: `test/CommonIR/fa_triton_dummy.py`
- 生成的 32×32 linalg IR: `test/CommonIR/temp_mlir/tmp_fa_linalg_32.mlir`
- 编译器 IR 替换逻辑: `third_party/ascend/backend/compiler.py` (line ~505, `TLE_REPLACE_IR_FILE`)
- Metadata 解析: `third_party/ascend/backend/compiler.py:249` (`_parse_linalg_metadata`)
- Runtime launch 代码: `python/triton/backends/ascend/driver.py:741` (`_launch` function)
