# tle.dsa.ascend.inplaced_elemop_gm 接口文档

## 1. 硬件背景

昇腾 NPU 的 fixpipe 及数据搬运单元支持一种"原子更新"模式：打开后，**所有写 GM 的操作（含 Cube 结果经 fixpipe 的 L0C→GM 回写、以及 UB→GM 的普通 store）都从"覆写"变为"读旧值-按 kind 计算后-写回"的原子操作**。利用该模式，`C += A@B` 类算子（addmm/split-K 累加等）可以完全省掉对 C 的一次 GM 读；max/min 模式同理。

该开关在下游 HIVM 编译器中对应 `hivm.hir.set_atomic` op（`FlagTree-AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td:101`，kind 支持 add/max/min，dtype 支持 F32/F16/BF16/S8/S16/S32）。本接口在 TTIR 层直接产出该 op，随编译链路透传至 hivmc 消费。

## 2. 接口说明

```python
tle.dsa.ascend.inplaced_elemop_gm(kind, dtype=tl.float32)
```

### 2.1 入参

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| kind | str constexpr | 是 | `'add'`（累加）/ `'max'` / `'min'`（数值选择）打开对应模式；`'none'` 关闭模式并恢复普通覆写 |
| dtype | tl.dtype | 否，默认 `tl.float32` | 被更新数据的元素类型，**必须与紧邻 store 的元素类型一致**（fp32 场景可省略；其他 dtype 务必显式传入，漏传会导致硬件按错误宽度做更新）；支持 fp16/bf16/fp32/int8/int16/int32（对齐 `hivm.hir.set_atomic` 声明的 F32/F16/S16/S32/S8/BF16；S8/S16 未实测）；`'none'` 时忽略（op 仍携带默认 type 属性，语义空转） |

### 2.2 返回值

无。

## 3. 用法示例（addmm：C += A@B，不 load C）

```python
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
out = tl.dot(a, b)                          # 不再 load C、不再 + c
tle.dsa.ascend.inplaced_elemop_gm("add", out.dtype)
tl.store(c_ptrs, out)                       # 硬件上变为 C += out
tle.dsa.ascend.inplaced_elemop_gm("none")
```

生成的 TTIR：

```
%v_1 = tt.load %v_0 : tensor<32x!tt.ptr<f32>>
hivm.hir.set_atomic kind = <add>[type = f32]
tt.store %1, %v_1 : tensor<32x!tt.ptr<f32>>
hivm.hir.set_atomic kind = <none>[type = f32]
```

## 4. 约束说明

- 开/关必须**成对、紧贴目标 store** 使用（`"add"`/`"max"`/`"min"` ↔ `"none"`）：模式打开期间，区间内**所有** GM 写都会被原地更新——夹杂其他 GM 写会污染数据。
