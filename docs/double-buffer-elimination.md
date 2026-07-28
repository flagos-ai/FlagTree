# 2× L1 缓冲区在 Linalg 降级流水线中被意外消除

## 问题描述

`test/CommonIR/native_matmul_dsa_slice.py` 中，matmul kernel 在 Python 层显式申请了两倍大小的 L1 片上缓冲区：

```python
mat_a_l1 = tle.dsa.alloc([2 * BLOCK_M, BLOCK_K], ...)   # [256, 128] f16
mat_b_l1 = tle.dsa.alloc([BLOCK_K, 2 * BLOCK_N], ...)   # [128, 512] f16
```

设计意图是 ping-pong 双缓冲：K 循环的偶数迭代使用 slot 0，奇数迭代使用 slot 1，两个 slot 交替读写，编译器无法静态证明任一 slot 是死代码，因此理应保留全部 2× 分配。

然而，从 TTIR 一路降级到 Linalg IR 后，输出文件中只剩下一倍大小的临时 staging buffer：

```mlir
%alloc   = memref.alloc() : memref<128x128xf16>   -- 原为 256x128
%alloc_2 = memref.alloc() : memref<128x256xf16>   -- 原为 128x512
```

2× 缓冲区彻底消失。

---

## 编译流水线

降级流水线共 6 个阶段（见 `dump_linalg()` 函数）：

| 步骤 | Pass | 输入方言 |
|------|------|----------|
| ① | `add_inliner` + `add_tileir_to_hivm` | TileIR → memref/hivm |
| ①b | `add_erase_linalg_casts` + `add_canonicalizer` | 消除 unrealized_conversion_cast |
| **②** | **`add_triton_to_structure_incubated`** + `add_discrete_mask_access_conversion` | **← 消除发生在此** |
| ③ | `add_triton_to_unstructure_incubated` + hivm + hfusion + llvm | |
| ④ | `add_bubble_up_operation` + `add_triton_to_structure_incubated` | |
| ④b | `add_inliner` + `add_canonicalizer` | |
| ⑤ | `add_triton_to_linalg_incubated` | tt.load/dot → memref.copy/linalg.matmul |
| ⑤b/c | `add_fold_staging_copy` + `add_erase_linalg_casts` | |
| ⑥ | `add_canonicalizer` + `add_cse` + `add_symbol_dce` | |

逐 pass 追踪 alloc / insert_slice / extract_slice 数量变化：

```
TTIR (baseline)                   alloc: 2  insert_slice: 2  extract_slice: 2
① tileir_to_hivm                  alloc: 4  insert_slice: 2  extract_slice: 2
①b erase_linalg_casts+canonicalize alloc: 4  insert_slice: 2  extract_slice: 2
② structure_incubated              alloc: 0  insert_slice: 0  extract_slice: 0  ← 全部清零
```

---

## step ① 之后的 IR 结构

`tileir_to_hivm` 把 `tile.alloc` 转换为带 `#hivm.address_space<cbuf>` 的 `memref.alloc`，`tile.to_tensor` 转换为 `bufferization.to_tensor`。step ①b 的 canonicalize 将 `unrealized_conversion_cast` 替换为 `bufferization.to_tensor restrict`。此时内层 K 循环体的关键结构如下：

```mlir
%alloc   = memref.alloc() : memref<256x128xf16, #hivm.address_space<cbuf>>   // 在循环体外
%alloc_0 = memref.alloc() : memref<128x512xf16, #hivm.address_space<cbuf>>

scf.for %k = 0 to 8 step 1
    iter_args(%acc = %cst_f32, %ptr_a = %a_block_ptr, %ptr_b = %b_block_ptr) {

  %row_off = arith.select (%k%2==0), 0, 128  : i32   // 运行时值
  %col_off = arith.select (%k%2==0), 0, 256  : i32

  // A: 2× buffer ping-pong
  %full_a   = bufferization.to_tensor %alloc restrict
  %loaded_a = tt.load %ptr_a
  %ins_a    = tensor.insert_slice %loaded_a into %full_a[%row_off, 0] [128,128] [1,1]
  %tile_a   = tensor.extract_slice %ins_a[%row_off, 0] [128,128] [1,1]

  // B: 同样结构
  %full_b   = bufferization.to_tensor %alloc_0 restrict
  %loaded_b = tt.load %ptr_b
  %ins_b    = tensor.insert_slice %loaded_b into %full_b[0, %col_off] [128,256] [1,1]
  %tile_b   = tensor.extract_slice %ins_b[0, %col_off] [128,256] [1,1]

  %acc_new  = tt.dot %tile_a, %tile_b, %acc
  scf.yield %acc_new, tt.advance(%ptr_a,[0,128]), tt.advance(%ptr_b,[128,0])
}
```

此时 `insert_slice`/`extract_slice` 的 offset 是运行时 `arith.select` 结果（`%row_off`、`%col_off`），**无法被静态 fold**。

---

## step ② 消除机制

`add_triton_to_structure_incubated` 内部依次执行：

1. `PromotePointerIterArgsPattern`（canonicalization patterns）
2. `MemOpConverter::LoadConverter` + `StoreConverter`（triton-to-structured patterns）
3. 内置 `createCSEPass` + `createCanonicalizerPass`

消除由 **`PromotePointerIterArgsPattern`** 主导，其 `matchAndRewriteAdvancePtr` 分支处理本例。

### PromotePointerIterArgsPattern 的工作原理

该 pattern 识别 scf.for 中通过 `tt.advance` 更新的 pointer iter args，将其替换为整数 offset iter args：

```
改写前 iter_args:  (%acc: tensor, %ptr_a: !tt.ptr<tensor>, %ptr_b: !tt.ptr<tensor>)
改写后 iter_args:  (%acc: tensor, %offset_a: i32,            %offset_b: i32          )
```

新循环体通过 `cloneInstructionsForAdvancePtr` 重建：

- **跳过** `tt.advance` ops（pointer 更新步骤）
- **clone 所有其他 ops**，使用 IRMapping 将旧值映射到新值

### 死代码链的形成

IRMapping 在 clone 前建立如下映射：

```
旧 %ptr_a  →  tt.advance(%arg0_base, [0, %offset_a])  (新构造的指针)
旧 %ptr_b  →  tt.advance(%arg1_base, [0, %offset_b])
```

clone 循环体时，`tt.load %ptr_a` 被 clone 为使用新指针的 `tt.load`，其结果被映射到一个新的 SSA 值。`tt.dot` 也被 clone，但 **IRMapping 已把 `%tile_a`（extract_slice 结果）直接重映射为新 `tt.load` 的结果**，因为 clone 是按 op 拓扑顺序进行的，`tt.dot` 的输入 mapping 在遇到 `tt.dot` 时已经指向新 `tt.load`。

于是 clone 出来的 `insert_slice` / `extract_slice` / `bufferization.to_tensor` 没有任何 use，形成死代码链：

```
%full_a   = bufferization.to_tensor %alloc  →  无 use（insert_slice 无 use）
%ins_a    = tensor.insert_slice ...         →  无 use（extract_slice 无 use）
%tile_a   = tensor.extract_slice ...        →  无 use（tt.dot 已直连 tt.load）
```

### 最终清理

`PromotePointerIterArgsPattern` 重写完成后，pass 内置的 `createCSEPass` + `createCanonicalizerPass` 清除所有无 use 的 ops。整条链被 DCE 删除，连同循环体外的 `memref.alloc<256x128>` 和 `memref.alloc<128x512>` 一起消失。

step ② 之后 IR 如下，2× 缓冲区已无踪影：

```mlir
scf.for %k = 0 to 8 step 1
    iter_args(%acc = %cst_f32, %offset_a = 0, %offset_b = 0) {

  %ptr_a_cur = tt.advance %a_base, [0,      %offset_a]
  %ptr_b_cur = tt.advance %b_base, [%offset_b, 0     ]
  %loaded_a  = tt.load %ptr_a_cur
  %loaded_b  = tt.load %ptr_b_cur
  %acc_new   = tt.dot %loaded_a, %loaded_b, %acc   // 直连 tt.load，无任何 buffer 中间层
  scf.yield %acc_new, addi(%offset_a,128), addi(%offset_b,128)
}
```

---

## 为什么 insert/extract_slice 无法阻止被 DCE

`PromotePointerIterArgsPattern` 在重建循环体时，`tt.dot` 的输入 mapping 优先于循环体内的数据流路径。具体来说：

- `%tile_a = extract_slice(...)` 进入 mapping 前，pattern 已将 `tt.dot` 的 A 输入 remapped 到新 `tt.load` 的结果
- 因此 clone 出来的 `%tile_a_cloned` 从未被 `tt.dot` 使用

若要防止此类消除，需要让 2× buffer 的读写产生**可观测的副作用**（side-effecting ops），或者使 `tt.dot` 的输入只能通过 buffer 的 extract_slice 获得而非直接来自 `tt.load`。

---

## 影响

| 阶段 | A buffer | B buffer |
|------|----------|----------|
| TTIR（Python 意图） | `memref<256x128xf16, L1>` | `memref<128x512xf16, L1>` |
| Linalg IR（实际输出） | `memref<128x128xf16>` (default space) | `memref<128x256xf16>` (default space) |

- L1 地址空间丢失，降为 default（global memory）
- 内存占用减半，ping-pong 语义完全丢失
- step ⑤ `triton_to_linalg_incubated` 为每次 `tt.load` 重新生成一个一倍大小的 staging buffer（`memref<128x128xf16>` 和 `memref<128x256xf16>`），地址空间为 default 而非 L1
- 后续 TileIR-to-HIVM 降级看不到任何 L1 buffer，无法生成片上 DMA 指令，缺失 double-buffer 带来的访存延迟隐藏

---

## 相关代码位置

| 文件 | 说明 |
|------|------|
| `test/CommonIR/native_matmul_dsa_slice.py` | kernel 定义与降级流水线 |
| `third_party/flir/lib/Conversion/TritonToStructuredIncubated/CannonicalizerConverter.cpp:918` | `cloneInstructionsForAdvancePtr` |
| `third_party/flir/lib/Conversion/TritonToStructuredIncubated/CannonicalizerConverter.cpp:445` | `matchAndRewriteAdvancePtr` |
| `third_party/flir/lib/Conversion/TritonToStructuredIncubated/TritonToStructuredIncubatedPass.cpp:154` | pass 内置 CSE+canonicalize |
| `third_party/ascend/lib/Conversion/TileIRToHIVM/TileIRToHIVM.cpp:172` | tile.alloc → memref.alloc 转换 |
