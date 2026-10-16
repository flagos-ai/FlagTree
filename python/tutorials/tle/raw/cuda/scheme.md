# tle_raw.call 升级方案

## 总体目标：实现 `05-sort.py` 和 `05-sort.cu` 中展示的外部函数调用方式

具体而言，在 python 侧：

- 输入 Tensor 时默认自动缓冲完整 Tensor 到一个可寻址平凡布局的 buffer。
- 实现 fragment 参数标记函数，用于标记一个 Tensor 输入，表示不传递完整 Tensor 到每个执行单元而是各执行单元获得自己所有的 Tensor 片段的值。
- 实现 output_buffer 输出描述标记函数，在定义用于传递输出的 buffer 的属性的同时，使 tle_raw.call 产出一份对应的 Tensor 返回值。

在 CUDA 侧：

- 基于 C++ 公共头文件实现用户友好的 Tensor 参数封装类型。
- 默认 buffer 使用无 swizzle 的平凡布局，存储顺序由后端确定。不同维数平凡缓冲的参数封装类型均采用 stride 寻址；wrapper 根据实际 buffer 布局提供各维度的 stride。

## 总体实现框架：

1. 新的 external 装饰器，产生通用性的 ExternalFunction 信息记录类型，我们借此区分新旧接口。
2. python 前端生成通用的新表示 tle.call_extern，由后端 pass 处理参数如何传递和输出缓冲的生成。
3. 对于 output_buffer 及普通 Tensor 参数，在 TTIR->TTGIR Conversion 之后立即插入缓冲分配操作，并改写 tle.call_extern 的 SSA 参数。
4. 对于 fragment 参数，在 TTGIR->LLVM IR Conversion 阶段收集 Tensor 布局信息用于生成 C++ wrapper。
5. 在 TTGIR->LLVM IR Conversion 阶段执行 wrapper 的生成与编译，然后生成 LLVM 调用 ，彻底清除 tle.call_extern 表示。

## 具体实现方案：

### 1. fragment 和 output_buffer 只应该在 tle_raw.call 的参数列表中使用

我们可以修改 `FlagTree/python/triton/language/semantic.py:TritonSemantic:to_tensor()`，在现有的转换路径后新增判断被转换的 object 是否自身具有 to_tensor 的代码。
如果有，我们尝试使用对象自身的方法来执行转换，并判断转换结果是否是 Tensor。
这既增强了 `TritonSemantic:to_tensor()`，同时还能通过为标记函数的返回类型实现产生异常的 `to_tensor()` 来初步禁止不被期望的外部赋值使用。

目前处于原型阶段，先不封堵所有不被期望的使用。

### 2. 通用的前端实现

当前 FlagTree 的 `@dialect` 为不同方言的函数生成不同的 python 前端对象，并通过这些对象进行代码生成。这在当前不需要对参数进行定制化处理的条件下是成立的。
但我们的方案为 Tensor 参数引入了不同的传参模式，不同目标平台对此的处理可能不同，因此我们实现通用的前端，生成通用的新表示 tle.call_extern，再在目标已知的后端 pass 阶段处理参数传递。

具体而言，前端在 tle.call_extern 中产生：

- 调用所在的源代码位置
- 调用所需的已经产生的 SSA
- 调用的返回 SSA
- 调用的参数信息描述表
- `fn_id`，用于从 C++ 侧关联 Python 注册的外部函数描述；具体调用的 `request_id` 在后端分配。

### 3. CUDA wrapper 示例

对于 `05-sort.py` 和 `05-sort.cu`，我们获取输入 Tensor 在单个线程上的元素数 `N`、计算必要的哈希值或 id 标识（假设是1A2B3C）, 然后生成：

```cpp
#include "05-sort.cu"

extern "C" __attribute__((device, used))
void wrapped_RadixHistogramDigits8x2048_1A2B3C(
    uint16_t fragment_1_1, uint16_t fragment_1_2, ..., uint16_t fragment_1_N1,
    bool fragment_2_1, bool fragment_2_2, ..., bool fragment_2_N2,
    uint32_t* output
) {
    uint16_t input_1[N1] = {fragment_1_1, fragment_1_2, ..., fragment_1_N1};
    bool input_2[N2] = {fragment_2_1, fragment_2_2, ..., fragment_2_N2};
    RadixHistogramDigits8x2048(
        tle::TensorFragment<uint16_t>{input_1, N1},
        tle::TensorFragment<bool>{input_2, N2},
        tle::PlainBuffer1D<uint32_t>{output, 1}
    );
}
```

以上代码只是表达大致的实现方式，具体实现细节应有所不同。编译为 LLVM IR 后我们在 LLVM IR 中生成对 `wrapped_RadixHistogramDigits8x2048_1A2B3C` 的调用。
在理想情况下，局部数组和 LLVM 调用均可被消除。

### 4. call_extern op 的具体细节

#### 4.1 op 和参数描述表

| op 字段 | 含义 |
| --- | --- |
| `inputs` | 按参数描述表顺序排列的 SSA operands |
| `outputs` | 尚未缓冲化的 `output_buffer` 对应的 Tensor results |
| `fn_id` | 非负整数，关联进程内的 `ExternalFunction` 描述 |
| `parameters` | `ArrayAttr`，每项为一个外部函数参数的 `DictionaryAttr` |
| `hint` | 可选字符串，保留给后端；具体支持哪些值由后端决定 |
| MLIR location | 保留 Python 调用位置，用于 pass 和外部编译失败时的诊断 |

参数描述表中的位置对应用户函数的参数位置，与 operand/result 的位置分别计数。

| 参数字段 | MLIR 属性类型 | 含义 |
| --- | --- | --- |
| `type` | `StringAttr` | `tensor_input`、`fragment_input`、`output_buffer` 或 `value` |
| `operand_index` | i64 整数属性 | 对应 operand 的位置；没有 operand 时为 -1 |
| `result_index` | i64 整数属性 | 对应 result 的位置；没有 result 时为 -1 |
| `shape` | `DenseI64ArrayAttr` | `tensor_input`、`output_buffer` 的逻辑形状 |
| `dtype` | `TypeAttr` | 所有参数必需；Tensor/MemDesc 参数记录元素类型，标量或指针参数记录值本身的类型，与 `type` 的传参分类含义不同 |
| `layout` | 原生 MLIR encoding attribute | 可选的 buffer 存储布局要求，不是返回 Tensor 的线程分布布局 |
| `bufferized` | `BoolAttr` | 可选参数，表示当前调用参数是否已经被缓冲，省略表示 false |

所有参数均显式携带 `dtype`，verifier 将其与 SSA operand/result 的类型进行一致性校验。`fragment_input` 的 `dtype` 为 Tensor 元素类型；`value` 的 `dtype` 为值本身的类型，指针参数保留完整指针类型，而不是仅记录指向元素的类型。fragment 的最终分布布局在 TTGIR 阶段获取。

`tensor_input` 和 `output_buffer` 参数描述可包含布尔属性 `bufferized`，前端省略，缺省为 `false`。`fragment_input` 和 `value` 不携带该属性。

- 缓冲化前，`tensor_input` 的 operand 为 Tensor；`output_buffer` 的 operand 为可选 initializer，result 为独立 Tensor。
- 缓冲化后，两类参数的 operand 均为 MemDesc，`result_index` 为 -1；输出初始化和结果读取由调用外的 store/load 完成。
- verifier 逐参数检查标记与实际类型、shape、dtype 和显式 layout 一致，允许同一次调用中存在不同缓冲化状态。pass 实现可以一次性缓冲化所有 buffer 参数；LLVM lowering 应要求所有 buffer 参数已完成缓冲化。

前端对每一个实际 operand 分配独立的索引位置。同一个 SSA value 被多次传入时可以在 operand 列表中重复出现，不应直接用 SSA value 去重，否则会破坏参数与位置的对应关系。

当前索引规则为：按参数顺序遍历，所有非 -1 的 `operand_index` 必须连续覆盖所有 operands；所有非 -1 的 `result_index` 必须连续覆盖所有 results。只有未缓冲化的 `output_buffer` 可以拥有 result。

#### 4.2 缓冲化前后的表示

转换时重新构造 operand 列表、result 列表与参数表，不能仅将原始 operand 原位替换：无 initializer 的输出会增加 operand，完成缓冲化的输出会删除 result，后续索引因此发生变化。

如果只缓冲化部分参数，应保留其他参数的状态，并重新编号剩余 results。IR 允许这种中间状态；实际 pass 可以一次完成全部参数。重复运行 pass 时跳过 `bufferized=true` 的参数，避免重复分配。

#### 4.3 前端返回值与校验边界

- 没有 `output_buffer` 时，Python 返回 `None`；一个输出返回 Tensor；多个输出按参数声明顺序返回 `tl.tuple`。
- 当前新接口通过 `output_buffer` 表达输出，不使用 `output_indices`；普通 C++ 返回值暂不映射为 Python 返回值。
- `output_buffer(init_value=...)` 的 shape、dtype 可以从 initializer 推导；显式提供时必须与 initializer 一致。没有 initializer 时两者必须提供。
- initializer 是初始内容，不能据此将输出分配与输入分配合并。调用后原 initializer Tensor 的 SSA 语义不变。
- marker 的 `to_tensor()` 用于阻止常见的错误 Tensor 转换，不能保证禁止所有局部赋值或其他 Python 用法；原型阶段保持现有边界。
- verifier 检查单参数的类型、索引、shape、dtype 和显式 layout 一致性；后端负责检查目标设备是否支持该存储布局、元素类型及执行方式。
