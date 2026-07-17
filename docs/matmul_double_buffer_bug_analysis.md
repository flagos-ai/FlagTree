# matmul_double_buffer 结果错误根因分析

## 1. 内核逻辑概述

`matmul_double_buffer.py` 实现的是 **copy-first（prefetch-first）双缓冲矩阵乘**。

在标准双缓冲方案中，两个 L1 缓冲槽（buf0、buf1）交替承担"正在被 Cube 单元计算"和"正在由 MTE2 预取下一个 K-tile"的角色，从而让数据搬运与矩阵乘法流水重叠。

### 1.1 内核结构

```python
# 分配 4 个独立 L1 缓冲（buf0/buf1，分别对应 A、B）
mat_a_l1_0 = tile_alloc([BLOCK_M, BLOCK_K], ...)  # buf0_A
mat_a_l1_1 = tile_alloc([BLOCK_M, BLOCK_K], ...)  # buf1_A（与 buf0_A 形状/类型完全相同）
mat_b_l1_0 = tile_alloc([BLOCK_K, BLOCK_N], ...)  # buf0_B
mat_b_l1_1 = tile_alloc([BLOCK_K, BLOCK_N], ...)  # buf1_B

# Prologue：把 k=0 的 tile 搬进 buf0
tile_copy(a_ptr[k=0], mat_a_l1_0, ...)
tile_copy(b_ptr[k=0], mat_b_l1_0, ...)

# Main K-loop（步长 2）
for k_pair in range(0, NUM_K_BLOCKS - 1, 2):
    # 偶步：copy(k+1) -> buf1，barrier，dot(buf0)，barrier
    tile_copy(a_ptr[k+1], mat_a_l1_1, ...)
    tile_copy(b_ptr[k+1], mat_b_l1_1, ...)
    tl.debug_barrier()          # 等 buf1 DMA 完成
    mat_c_acc = tl.dot(to_tensor(mat_a_l1_0), to_tensor(mat_b_l1_0), ...)
    tl.debug_barrier()          # 等 dot(buf0) 完成

    # 奇步：copy(k+2) -> buf0，barrier，dot(buf1)，barrier
    if k_pair + 2 < NUM_K_BLOCKS:
        tile_copy(a_ptr[k+2], mat_a_l1_0, ...)
        tile_copy(b_ptr[k+2], mat_b_l1_0, ...)
    tl.debug_barrier()
    mat_c_acc = tl.dot(to_tensor(mat_a_l1_1), to_tensor(mat_b_l1_1), ...)
    tl.debug_barrier()
```

语义上，每个 K-tile 应依次被搬入 buf0 或 buf1，dot 读取的槽在两轮之间交替，逻辑完全正确。

---

## 2. IR 下降过程

完整 lowering pipeline（`backend/compiler.py: ttir_to_linalg`）分为以下关键步骤：

```
TTIR（triton.jit 产生）
 ↓ make_ttir: inliner + combine + canonicalize + reorder_broadcast + CSE + symbol_dce + loop_unroll
TTIR（优化后）
 ↓ tileir_to_hivm
 ↓ erase_linalg_casts + canonicalize
 ↓ triton_to_structure_incubated + discrete_mask_access_conversion
 ↓ triton_to_unstructure_incubated + triton_to_hivm + triton_to_hfusion + triton_to_llvm
 ↓ bubble_up_operation + triton_to_structure_incubated
 ↓ inliner + canonicalize
 ↓ triton_to_linalg_incubated
 ↓ fold_staging_copy
 ↓ erase_linalg_casts + canonicalize + CSE + symbol_dce
Linalg IR (ttadapter.mlir，送往 bishengir-compile)
```

### 2.1 TTIR（CSE 之前）：4 个独立 tile.alloc

```mlir
%5 = tile.alloc {space = 1} : <[128, 128], f16, l1>   // mat_a_l1_0 / buf0_A
%6 = tile.alloc {space = 1} : <[128, 128], f16, l1>   // mat_a_l1_1 / buf1_A
%7 = tile.alloc {space = 1} : <[128, 256], f16, l1>   // mat_b_l1_0 / buf0_B
%8 = tile.alloc {space = 1} : <[128, 256], f16, l1>   // mat_b_l1_1 / buf1_B

// 内层循环，偶步：
tile.copy %33 -> %6                     // copy k+1 -> buf1_A
tile.copy %34 -> %8                     // copy k+1 -> buf1_B
gpu.barrier
%35 = tile.to_tensor %5                 // read buf0_A（k tile）
%36 = tile.to_tensor %7                 // read buf0_B
%37 = tt.dot %35, %36, %arg8           // dot(buf0)
gpu.barrier

// 奇步：
tile.copy %44 -> %5                     // copy k+2 -> buf0_A
tile.copy %45 -> %7                     // copy k+2 -> buf0_B
gpu.barrier
%41 = tile.to_tensor %6                 // read buf1_A
%42 = tile.to_tensor %8                 // read buf1_B
%43 = tt.dot %41, %42, %37            // dot(buf1)
```

此时 `%5`、`%6`、`%7`、`%8` 是 **4 个独立的** `tile.buf`，双缓冲语义完整。

### 2.2 TTIR（CSE 之后）：2 个 tile.alloc

`make_ttir` 中的 **CSE（公共子表达式消除）** 将形状和类型完全相同的 `tile.alloc` 识别为等价操作并合并：

```mlir
// CSE 之前：4 个 alloc
%5 = tile.alloc : <[128, 128], f16, l1>
%6 = tile.alloc : <[128, 128], f16, l1>   // ← CSE 认为与 %5 等价，合并掉
%7 = tile.alloc : <[128, 256], f16, l1>
%8 = tile.alloc : <[128, 256], f16, l1>   // ← CSE 认为与 %7 等价，合并掉

// CSE 之后：2 个 alloc，所有对 %6/%8 的引用被替换为 %4/%5
%4 = tile.alloc : <[128, 128], f16, l1>   // buf0_A ≡ buf1_A（物理同一块）
%5 = tile.alloc : <[128, 256], f16, l1>   // buf0_B ≡ buf1_B（物理同一块）

// 内层循环，偶步：
tile.copy %16 -> %4    // copy k+1 -> %4（覆写了 "buf0"）
tile.copy %17 -> %5
gpu.barrier
%18 = tile.to_tensor %4   // 读 %4（此时已是 k+1！）
%20 = tt.dot %18, %19, %arg8  // dot 读到 k+1，不是 k

// 奇步：
tile.copy %25 -> %4    // copy k+2 -> %4（再次覆写）
tile.copy %26 -> %5
gpu.barrier
%24 = tt.dot %18, %19, %20   // %18/%19 是旧快照（to_tensor 已被 CSE 复用）
```

CSE 合并后：
- 偶步的 `copy(k+1)->%4` 和随后 `dot(%4)` 读到的是同一个 `%4`，结果等于 dot(k+1) 而非 dot(k)。
- `tile.to_tensor` 的结果也被 CSE 复用，奇步 dot 用的是偶步 to_tensor 的快照，读到的还是 k+1 的数据。
- 整个 K 维循环实际上始终读同一内存地址，计算完全错误。

### 2.3 triton_to_linalg_incubated：2 个 cbuf alloc

由于 TTIR 在 CSE 后只剩 2 个 `tile.alloc`，`triton_to_linalg_incubated` 相应只生成 2 个 cbuf memref：

```mlir
// 降级后，%alloc 同时承担 buf0_A 和 buf1_A 的角色
%alloc   = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>  // 来自 mat_a_l1_0 (line 57)
%alloc_0 = memref.alloc() : memref<128x256xf16, #hivm.address_space<cbuf>>  // 来自 mat_b_l1_0 (line 59)
// mat_a_l1_1 (line 58) 和 mat_b_l1_1 (line 60) 的 alloc 已不存在

// 内层循环同时产生了两组 staging alloc（default space），如：
%alloc_9  = memref.alloc() : memref<128x128xf16>                     // staging for 偶步 A
%alloc_10 = memref.alloc() : memref<128x128xf16, gm>
memref.copy %alloc_10, %alloc  // 偶步：copy k+1 -> %alloc

%alloc_15 = memref.alloc() : memref<128x128xf16>                     // staging for 奇步 A
%alloc_16 = memref.alloc() : memref<128x128xf16, gm>
memref.copy %alloc_16, %alloc  // 奇步：copy k+2 -> 同一个 %alloc！

// 两次 matmul 都读同一对 bufferization.to_tensor(%alloc, %alloc_0)
%28 = bufferization.to_tensor %alloc  restrict
%29 = bufferization.to_tensor %alloc_0 restrict
%30 = linalg.matmul ins(%28, %29 ...)   // 偶步 dot
%34 = linalg.matmul ins(%28, %29 ...)   // 奇步 dot（%28/%29 完全相同）
```

### 2.4 fold_staging_copy：消除中间 staging

`fold_staging_copy` 识别形如

```
%stage = memref.alloc()   // default space
copy %src(GM) -> %stage
copy %stage -> %dst(cbuf)
```

的链，将其折叠为 `copy %src(GM) -> %dst(cbuf)`（直接 GM→cbuf）：

```mlir
// fold 之后（staging alloc 消失，直接 copy GM -> cbuf）：
memref.copy %memspacecast,    %alloc    // 偶步 copy k+1 -> %alloc
memref.copy %memspacecast_10, %alloc    // 奇步 copy k+2 -> %alloc（同一目标！）
```

此步是合理的折叠优化，**不是根因**。根因在 CSE 阶段已经造成，`fold_staging_copy` 只是在已损坏的 IR 上继续工作。

### 2.5 最终 Linalg IR（送往 bishengir）

```mlir
%alloc   = memref.alloc() : memref<128x128xf16, cbuf>   // ← 唯一 A 缓冲
%alloc_0 = memref.alloc() : memref<128x256xf16, cbuf>   // ← 唯一 B 缓冲

for k_pair = 0 to 7 step 2:
  // 偶步：先 copy，再读
  memref.copy gm_A[k+1] -> %alloc
  memref.copy gm_B[k+1] -> %alloc_0
  gpu.barrier
  %28 = bufferization.to_tensor %alloc
  %29 = bufferization.to_tensor %alloc_0
  %30 = linalg.matmul ins(%28, %29)   // 读 k+1，应读 k！

  gpu.barrier
  // 奇步（scf.if）：再次覆写同一 cbuf
  memref.copy gm_A[k+2] -> %alloc
  memref.copy gm_B[k+2] -> %alloc_0
  gpu.barrier
  %34 = linalg.matmul ins(%28, %29)   // %28/%29 是旧快照（k+1），应读 k+1，
                                       // 但 copy 已写入 k+2，行为取决于硬件缓存
```

结论：每次迭代中，dot 读到的数据比预期超前一步；两次 dot 读的是相同的数据快照。

---

## 3. 根因：CSE 错误合并 tile.alloc

### 3.1 直接证据

| Pipeline 阶段 | L1 tile.alloc 数 |
|---|---|
| TTIR 原始（make_ttir 前）| **4**（buf0_A, buf1_A, buf0_B, buf1_B）|
| TTIR 经 CSE（make_ttir 中）| **2**（buf0_A≡buf1_A, buf0_B≡buf1_B）|
| triton_to_linalg 后 | 2 cbuf alloc |
| fold_staging_copy 后 | 2 cbuf alloc（staging 被折叠）|
| bishengir 输入 | 2 cbuf alloc |

环境变量 `COMMONIR_SKIP_CSE=1` 可复现此差异：设置该变量后 TTIR 保留 4 个 `tile.alloc`，最终 Linalg IR 也有 4 个 cbuf alloc，双缓冲语义得以保持。

### 3.2 CSE 为何认为两个 alloc 等价

`make_ttir` 中调用 `passes.common.add_cse(pm)`，其依据是：

> 若两个操作的 opcode 相同、所有属性相同、所有操作数相同，则认为结果等价，保留第一个，用其结果替换第二个的所有引用。

`tile.alloc {space = 1} : <[128, 128], f16, l1>` 没有任何操作数，两次 alloc 的 opcode 和属性完全相同，CSE 将第二个 alloc 替换为第一个的 SSA 值。

这对无副作用的常量计算是正确的。但 `tile.alloc` 分配物理内存，**两次 alloc 语义上是两块独立缓冲区**，它们不应被视为等价。根本原因是 `tile.alloc` 在 TileIR dialect 定义中未被标记为 `Memory Write` / `HasRecursiveMemoryEffects` 或类似的有副作用特性，导致 CSE 将其当作纯函数处理。

### 3.3 为何 compute-first（serial）不受影响

`matmul_double_buffer_serial.py` 采用 compute-first 顺序：dot 先于 copy，内核不含 `tl.debug_barrier()`。

该文件的 `_compile_matmul_module()` 在 `make_ttir` 之前不经过标准的 `ttir_to_linalg` pipeline（它直接走 dump 路径，传入自定义 options），因此 **在其测试路径上 CSE 未被调用**，4 个 tile.alloc 得以保留。

更关键的是，即便最终 cbuf 被合并，compute-first 的 loop 结构（dot 在 copy 之前）会被 bishengir 的 `--enable-auto-multi-buffer` pass 识别为流水线模式，自动插入正确的 set_flag/wait_flag 同步，不依赖两个独立 cbuf 的存在。而 copy-first 的结构不满足该模式的识别条件。

---

## 4. 修复方向

### 方案 A：修复 tile.alloc 的副作用声明（推荐，根本修复）

在 TileIR dialect 的 Op 定义中，为 `tile.alloc` 标记 `MemAlloc` 效果（类似 `memref.alloc` 在 MLIR 中的处理方式）：

```tablegen
// 示例：在 TileIROps.td 中
def TileAllocOp : ... {
  let summary = "allocate an on-chip tile buffer";
  // 标记为有副作用，阻止 CSE 合并
  let effects = [MemoryEffects::EffectInstance<MemAlloc::MemAllocEffect, LocalMemoryResource>];
}
```

这样 CSE 会将 `tile.alloc` 识别为有副作用操作，不再将两个独立 alloc 合并为同一 SSA 值。

### 方案 B：在 make_ttir 中跳过 CSE（现有 workaround）

设置环境变量 `COMMONIR_SKIP_CSE=1`（代码中已有此机制）可跳过 CSE，保留 4 个独立 alloc。代价是整个编译过程可能错过合法的 CSE 优化。

```bash
COMMONIR_SKIP_CSE=1 python matmul_double_buffer.py
```

### 方案 C：改用 compute-first 顺序

将内核重写为 compute-first（dot 先于 copy），与 `matmul_double_buffer_serial.py` 一致。这种顺序可被 bishengir 的自动多缓冲 pass 正确处理，不依赖两个独立 cbuf alloc 的存在。

---

## 5. 总结

```
tile.alloc buf0_A  ──┐
tile.alloc buf1_A  ──┤  ← CSE 认为这两者等价（同 opcode、同属性、同操作数）
                     │    将 buf1_A 的所有引用替换为 buf0_A
                     ↓
只剩 tile.alloc buf0_A（buf0_A ≡ buf1_A）

→ triton_to_linalg 只生成 1 个 128×128 cbuf alloc
→ 偶步和奇步的 copy 都写同一个 cbuf
→ 两次 dot 读的都是同一块内存（内容取决于最后一次 copy 写入了什么）
→ 计算结果错误
```

`fold_staging_copy` 在此过程中是无辜的——它在 CSE 已经损坏 IR 之后运行，只是忠实地折叠了 staging chain，并不引入新的错误。

真正需要修复的是 **`tile.alloc` 在 CSE 面前的副作用语义声明**。
