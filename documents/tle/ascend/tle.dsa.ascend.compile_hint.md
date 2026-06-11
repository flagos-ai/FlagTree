# tle.dsa.ascend.compile_hint 接口文档

# 1. 硬件背景

在昇腾 NPU 上编写 Triton 算子时，编译器自动进行代码生成和优化调度。但在某些场景下，开发者对具体的数据流和处理方式有更明确的预期，此时可以通过 `compile_hint` 向编译器传递优化提示（Hint），指导代码生成和性能调优，使生成的指令序列更符合预期。

`disable_bubble_up` 提示的典型场景是：当 kernel 中存在 Vector 计算 + `extract_slice` 逐行写入模式时，编译器可能会将 Vector 运算"上浮"（bubble up）到 `scf.for` 循环内部，导致循环体内反复执行相同的向量计算。通过该 Hint 可以阻止这种上浮，将 Vector 运算保持在循环外的全 tensor 上完成，循环内仅保留 `extract_slice` 和 store 操作，从而减少冗余计算。

# 2. 接口说明

<table>
  <tr>
    <td>Plain Text<br>def compile_hint(ptr, hint_name, hint_val=None, _builder=None):</td>
  </tr>
</table>

## 2.1 入参

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>ptr</td>
    <td>tensor</td>
    <td>是</td>
    <td>目标张量的指针。</td>
  </tr>
  <tr>
    <td>hint_name</td>
    <td>str</td>
    <td>是</td>
    <td>提示名称，用于指定编译器优化策略。例如：<br><code>"disable_bubble_up"</code> — 阻止编译器将 Vector 运算上浮到循环内部，保持全 tensor 计算。</td>
  </tr>
  <tr>
    <td>hint_val</td>
    <td>多种类型</td>
    <td>否</td>
    <td>提示参数值（可选），为优化提示提供额外参数。</td>
  </tr>
</table>

## 2.2 返回值

无

# 3. 约束说明

- `hint_name` 为预定义的字符串常量，需根据编译器支持的提示类型传入。
- `ptr` 必须为合法的 Triton 张量对象。
- `hint_val` 为可选参数，根据 `hint_name` 决定是否需要传入。

# 4. 示例：disable_bubble_up 提示

以下示例实现了一个 RMS Norm 逐行写入的 kernel，展示 `disable_bubble_up` 的作用。

## 4.1 不加 Hint（参考实现）

<table>
  <tr>
    <td>Python<br>@triton.jit<br>def triton_rms_slice_ref(x_ptr, w_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,<br>                         BLOCK_SIZE: tl.constexpr, eps: tl.constexpr):<br>    pid = tl.program_id(0)<br>    cols = tl.arange(0, BLOCK_SIZE)<br><br>    x = tl.load(x_ptr + pid * N + cols)<br>    var = tl.sum(x * x, axis=0) * (1.0 / N)<br>    rrms = tl.rsqrt(var + eps)<br>    w = tl.load(w_ptr + cols)<br>    y = (x * rrms * w).to(out_ptr.dtype.element_ty)<br><br>    # Extract_slice each row → store (simulating KV cache scatter-write)<br>    for i in tl.static_range(N):<br>        value_reload = tle.dsa.extract_slice(y, (i,), (1,), (1,))<br>        offs = pid * N + i + tl.arange(0, 1)<br>        tl.store(out_ptr + offs, value_reload)</td>
  </tr>
</table>

## 4.2 增加 disable_bubble_up Hint

<table>
  <tr>
    <td>Python<br>@triton.jit<br>def triton_rms_slice_disable_bubble_up(x_ptr, w_ptr, out_ptr, M: tl.constexpr,<br>                                        N: tl.constexpr, BLOCK_SIZE: tl.constexpr,<br>                                        eps: tl.constexpr):<br>    pid = tl.program_id(0)<br>    cols = tl.arange(0, BLOCK_SIZE)<br><br>    x = tl.load(x_ptr + pid * N + cols)<br>    var = tl.sum(x * x, axis=0) * (1.0 / N)<br>    rrms = tl.rsqrt(var + eps)<br>    w = tl.load(w_ptr + cols)<br>    y = (x * rrms * w).to(out_ptr.dtype.element_ty)<br><br>    # Extract_slice each row with disable_bubble_up hint<br>    for i in tl.static_range(N):<br>        value_reload = tle.dsa.extract_slice(y, (i,), (1,), (1,))<br>        tl.compile_hint(value_reload, "disable_bubble_up")<br>        offs = pid * N + i + tl.arange(0, 1)<br>        tl.store(out_ptr + offs, value_reload)</td>
  </tr>
</table>

## 4.3 生成 IR 对比

### 不加 Hint — Vector 运算被上浮到循环内

在生成的 MLIR 中，Vector 运算（`arith.mulf`、`linalg.fill`、`tensor.extract` 等）被上浮到 `scf.for` 循环内部，每行迭代都会重复执行：

<table>
  <tr>
    <td>Plain Text<br>scf.for %arg15 = %c0_i32 to %c64_i32 step %c1_i32 : i32 {<br>  %extracted_slice = tensor.extract_slice %4[%15, 0] [1, 128] [1, 1] {DiscreteMemAccess} : tensor<64x128xf16> to tensor<1x128xf16><br>  %extracted = tensor.extract %10[%15] {DiscreteMemAccess} : tensor<64xf16><br>  %16 = tensor.empty() : tensor<1x128xf16><br>  %17 = linalg.fill ins(%extracted : f16) outs(%16 : tensor<1x128xf16>) -> tensor<1x128xf16><br>  %18 = arith.mulf %extracted_slice, %17 : tensor<1x128xf16><br>  %extracted_slice_6 = tensor.extract_slice %5[%15, 0] [1, 128] [1, 1] {DiscreteMemAccess} : tensor<64x128xf16> to tensor<1x128xf16><br>  %19 = arith.mulf %18, %extracted_slice_6 : tensor<1x128xf16><br>  %reshape = tensor.reshape %19(%cst) : (tensor<1x128xf16>, tensor<1xi64>) -> tensor<128xf16><br>  bufferization.materialize_in_destination %reshape in writable ...<br>}</td>
  </tr>
</table>

### 使用 disable_bubble_up — Vector 运算保持在循环外

添加 Hint 后，Vector 运算在循环外以全 tensor 方式完成，循环体内仅保留 `extract_slice` 和 store：

<table>
  <tr>
    <td>Plain Text<br>  # Vector 运算在循环外以全 tensor 完成<br>  %12 = arith.mulf %4, %broadcasted : tensor<64x128xf16><br>  %13 = arith.mulf %12, %5 : tensor<64x128xf16><br>  bufferization.materialize_in_destination %13 in writable ...<br><br>  # 循环内仅保留 extract_slice 和 store<br>  scf.for %arg15 = %c0_i32 to %c64_i32 step %c1_i32 : i32 {<br>    %extracted_slice = tensor.extract_slice %13[%15, 0] [1, 128] [1, 1] : tensor<64x128xf16> to tensor<1x128xf16><br>    %reshape = tensor.reshape %extracted_slice(%cst) : (tensor<1x128xf16>, tensor<1xi64>) -> tensor<128xf16><br>    bufferization.materialize_in_destination %reshape in writable ...<br>  }</td>
  </tr>
</table>

## 4.4 效果说明

- **不加 Hint**：编译器将 `mul(rrms, w)` 等 Vector 运算上浮到 `scf.for` 循环内部，每行迭代都重复执行相同的向量运算，产生大量冗余计算。
- **使用 `disable_bubble_up`**：Vector 运算在循环外以全 tensor 粒度一次性完成，循环体仅做数据搬移，消除了冗余计算，在行数较多时收益显著。

# 5. 相关hint优化

## 5.1 mayDiscretememaccess

### 用途

避免由于硬件 32B 对齐要求导致的 UB overflow 或非连续内存访问问题。当加载的 tensor 最后一维大小不足 32B 对齐时（如 `<Nx1xf32>`），硬件会自动将其扩展为 `<Nx8xf32>`，导致 UB 空间膨胀 8×，可能引发 UB overflow。

### Hint 值

`"mayDiscretememaccess"` — 提示编译器将 tensor 访问降级为 scalar 级离散访存，避免 32B 对齐扩展。

### 触发条件

`M, K, N = 256, 1, 256`，`BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 1`。当 `K=1` 且 `BLOCK_K=1` 时，加载的 A 块形状为 `(BLOCK_M, 1) x f32`，`1xf32 = 4B < 32B`，触发对齐扩展。

### 使用方式

<table>
  <tr>
    <td>Python<br>a_block = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)<br>b_block = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)<br>tle.dsa.ascend.compile_hint(a_block, "mayDiscretememaccess")<br>tle.dsa.ascend.compile_hint(b_block, "mayDiscretememaccess")<br>acc = tl.dot(a_block, b_block, acc)</td>
  </tr>
</table>

### IR 对比

**不加 Hint**（`matmul_ref.ttadapter.mlir`）— 使用 `memref.copy` 做全量 tensor 拷贝，shape 保持 `<64x1xf16>`：

```
%subview = memref.subview %reinterpret_cast[0, 0] [%10, 1] [1, 1]
    : memref<64x1xf16, strided<[1, 1], offset: ?>>
    to memref<?x1xf16, strided<[1, 1], offset: ?>>
%subview_1 = memref.subview %alloc[0, 0] [%10, 1] [1, 1]
    : memref<64x1xf16> to memref<?x1xf16, strided<[1, 1]>>
memref.copy %subview, %subview_1
    : memref<?x1xf16, strided<[1, 1], offset: ?>>
    to memref<?x1xf16, strided<[1, 1]>>
%12 = bufferization.to_tensor %alloc restrict writable : memref<64x1xf16>
```

**使用 mayDiscretememaccess**（`may_discrete.ttadapter.mlir`）— 降级为 `scf.for` + `memref.load` 逐元素加载，避免对齐扩展：

```
%15 = scf.for %arg11 = %c0 to %14 step %c1 iter_args(%arg12 = %4) -> (tensor<64x1xf16>) {
  %31 = memref.load %reinterpret_cast_4[%c0]
      : memref<1xf16, strided<[1], offset: ?>>
  %inserted = tensor.insert %31 into %arg12[%arg11, %c0]
      : tensor<64x1xf16>
  scf.yield %inserted : tensor<64x1xf16>
} {ExtractedLoadOrStore}
```

同时在 matmul 结果上附加了 `annotation.mark`：

```
annotation.mark %22 {mayDiscretememaccess} : tensor<64x64xf32>
```

### 测试结果

```
[BENCH] mayDiscretememaccess (hint): X.XX us, (no hint): X.XX us
Note: hint downgrades tensor→scalar, avoiding 8× UB expansion from <Nx1xf32> alignment.
```

---

## 5.2 dot_pad_only_k

### 用途

在 MatMul 操作中，当 M/N 维度已经对齐但 K 维度不对齐时，通过该 hint 告知编译器只需在 K 方向做 padding，避免不必要的 M/N 方向 padding 处理。

### Hint 值

`"dot_pad_only_k"` — 提示编译器只对 Dot 操作的 K 维度进行 padding。

### 触发条件

`M = 2048, K = 4, N = 16384`，`BLOCK_K = 4`。当 K 不是对齐粒度的小倍数时，K 维度需要 padding，但 M/N 维度已经对齐，无需额外处理。

### 使用方式

<table>
  <tr>
    <td>Python<br>a_block = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)<br>b_block = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)<br>tle.dsa.ascend.compile_hint(a_block, "dot_pad_only_k")<br>tle.dsa.ascend.compile_hint(b_block, "dot_pad_only_k")<br>acc = tl.dot(a_block, b_block, acc)</td>
  </tr>
</table>

### 优化效果

优化效果有限，主要减少最后一次 tile 时 K 方向的 padding 开销。

---

## 5.3 hivm.tile_mix_cube_num

### 用途

在 Flash Attention 等 cube → vector → cube 流水模式中，编译器默认只对单个 matmul 进行切分需求分析，不感知其他 matmul 的生命周期。当多个 matmul 的生命周期重叠时，可能导致 L1 越界。该 hint 告知编译器对相关 matmul 进行 sub tiling。

### Hint 值

`"hivm.tile_mix_cube_num"` — hint_val 为 `int`，表示 sub tile 数量。例如 `tle.dsa.ascend.compile_hint(o_acc, "hivm.tile_mix_cube_num", 4)`。

### 使用方式

<table>
  <tr>
    <td>Python<br># 第一个 matmul<br>s_acc = tl.dot(q_block, k_block, s_acc)<br>tle.dsa.ascend.compile_hint(s_acc, "hivm.tile_mix_cube_num", 4)<br><br># Vector ops (softmax)<br># ...<br><br># 第二个 matmul<br>o_acc = tl.dot(p_block, v_block, o_acc)<br>tle.dsa.ascend.compile_hint(o_acc, "hivm.tile_mix_cube_num", 4)</td>
  </tr>
</table>

### IR 对比

以 `tile_mix_cube_num.ttadapter.mlir` 为例，在第一个 matmul 的结果上附加了 `annotation.mark`：

```
%55 = linalg.matmul {input_precison = "ieee"}
      ins(%51, %transposed : tensor<64x300xf16>, tensor<300x360xf16>)
      outs(%arg17 : tensor<64x360xf32>) -> tensor<64x360xf32>
annotation.mark %55 {hivm.tile_mix_cube_num = 4 : i32}
      : tensor<64x360xf32>
```

编译器据此对 cube 操作进行 sub tiling，避免多 matmul 生命周期重叠导致的 L1 越界。

### 注意事项

- 主要限制 L1 overflow 场景（L1 ≈ 128KB）
- UB 限制（≈ 192KB）仍是更大瓶颈：sub tile ≥ 96KB 时仍会在 UB 上 overflow
- FA 算子等 CV 类场景可配合以下编译选项使用：
  - `Multibuffer` — 设置是否启用乒乓流水
  - `limit_auto_multi_buffer_of_local_buffer` — 设置乒乓流水在片内(L1, L0, UB)的作用范围：`"no-limit"` 不限范围，`"no-l0c"` 仅 L0 缓存外启用
  - `limit_auto_multi_buffer_only_for_local_buffer` — 是否在 GM workspace 中启用 CV 流水并行，`False` 表示启用
  - `set_workspace_multibuffer` — 设置 CV 并行的并行度，默认 2
  - `tile_mix_vector_loop` — 设置 vector 切分数量
  - `tile_mix_cube_loop` — 设置 cube 切分数量
  - `unit_flag` — 设置 cube 按 block 方式搬出

---

## 5.4 disable_bubble_up

### 用途

阻止编译器将 Vector 运算"上浮"（bubble up）到 `scf.for` 循环内部。当 kernel 中存在 Vector 计算 + `extract_slice` 逐行写入模式时，编译器可能将 `y = x * rrms * w` 等运算推到循环内逐行重建，产生大量冗余计算。

### Hint 值

`"disable_bubble_up"` — 告知编译器不要对当前 OP 做 bubble up 优化。

### 使用方式

<table>
  <tr>
    <td>Python<br>y = (x * rrms * w).to(out_ptr.dtype.element_ty)<br>for i in tl.static_range(N):<br>    value_reload = tle.dsa.extract_slice(y, (i,), (1,), (1,))<br>    tl.compile_hint(value_reload, "disable_bubble_up")<br>    offs = pid * N + i + tl.arange(0, 1)<br>    tl.store(out_ptr + offs, value_reload)</td>
  </tr>
</table>

### IR 对比

**不加 Hint**（`ref_bubble.ttadapter.mlir`）— `extract_slice` 被推过 `y = x * rrms * w` 计算链，`arith.mulf`、`linalg.fill`、`tensor.extract` 等 Vector 运算全部在 `scf.for` 循环内重复执行：

```
scf.for %arg15 = %c0_i32 to %c64_i32 step %c1_i32 : i32 {
  %extracted_slice = tensor.extract_slice %4[%15, 0] [1, 128] [1, 1]
      {DiscreteMemAccess} : tensor<64x128xf16> to tensor<1x128xf16>
  %extracted = tensor.extract %10[%15] {DiscreteMemAccess} : tensor<64xf16>
  %16 = tensor.empty() : tensor<1x128xf16>
  %17 = linalg.fill ins(%extracted : f16) outs(%16 : tensor<1x128xf16>)
      -> tensor<1x128xf16>
  %18 = arith.mulf %extracted_slice, %17 : tensor<1x128xf16>
  %extracted_slice_6 = tensor.extract_slice %5[%15, 0] [1, 128] [1, 1]
      {DiscreteMemAccess} : tensor<64x128xf16> to tensor<1x128xf16>
  %19 = arith.mulf %18, %extracted_slice_6 : tensor<1x128xf16>
  %reshape = tensor.reshape %19(%cst)
      : (tensor<1x128xf16>, tensor<1xi64>) -> tensor<128xf16>
  bufferization.materialize_in_destination %reshape in writable ...
}
```

**使用 disable_bubble_up**（`disbubble.ttadapter.mlir`）— Vector 运算在循环外以全 tensor 完成，循环内仅保留 `extract_slice` 和 store：

```
# Vector 运算在循环外以全 tensor 完成
%12 = arith.mulf %4, %broadcasted : tensor<64x128xf16>
%13 = arith.mulf %12, %5 : tensor<64x128xf16>
bufferization.materialize_in_destination %13 in writable ...

# 循环内仅保留 extract_slice 和 store
scf.for %arg15 = %c0_i32 to %c64_i32 step %c1_i32 : i32 {
  %extracted_slice = tensor.extract_slice %13[%15, 0] [1, 128] [1, 1]
      : tensor<64x128xf16> to tensor<1x128xf16>
  %reshape = tensor.reshape %extracted_slice(%cst)
      : (tensor<1x128xf16>, tensor<1xi64>) -> tensor<128xf16>
  bufferization.materialize_in_destination %reshape in writable ...
}
```

### 适用条件

`disable_bubble_up` 仅在以下条件同时满足时才是正优化：
1. **计算很重** — 如 RMS Norm 中的 `rsqrt(sum(x²)/N)`
2. **N 很大** — 标量化会生成爆炸性数量的指令
3. **store 较简单** — compute 才是瓶颈
4. 需要extract_slice，将整个tile计算下沉到extract的slice分块的计算，
5. 并且会在循环外进行引用操作(例如store).

---

## 5.5 bitwise_mask

### 用途

在昇腾硬件上，布尔类型（i1）的张量在全局内存（GM）中按 i8（1 字节）存储。当 Triton 处理以 i1 张量作为输入的运算（如 `tl.where` 的 condition）时，它会将 i1 视为 i8 搬入，然后进行 `vcast(i8→f16)` → `vcmp(f16, 0)` → `vnot(i1)` → `vsel(i1)` 的转换链，导致不必要的类型转换开销。`bitwise_mask` 旨在跳过此转换链，直接使用原始 i8 bitmask 进行 select。

### Hint 值

`"bitwise_mask"` — 提示编译器将 condition 视为位掩码，跳过 vcast→vcmp→vnot 转换链。

### 使用方式

<table>
  <tr>
    <td>Python<br>cond = tl.load(cond_ptr + xindex, xmask)<br>tl.compile_hint(cond, "bitwise_mask")<br>res = tl.where(cond, in1, in0)<br>tl.store(out_ptr0 + xindex, res, xmask)</td>
  </tr>
</table>
