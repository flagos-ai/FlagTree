# 指导：如何新增kernel测试用例
新增kernel测试用例可以分为三大步:
1、准备pt文件
2、在triton-ascend仓中添加kernel算子，完成本地kernel测试
3、将pt文件上传到obs桶中

## 1、准备pt文件

pt 文件用于把 GPU（或参考实现）上的输入与输出作为 golden 数据，后续测试会在 NPU 上运行 Triton kernel 并与之比对。

**三步生成流程**

- **步骤 1 — 构造GPU输入并保存副本预处理成NPU kernel的输入**：根据GPU上kernel或pytorch算子的参数构造 `input_data`（键名须与 kernel 参数一致），把所有 Tensor 克隆到 CPU，形成 `input_data_before`，若GPU上算子的输入和NPU上算子有出入，需要提前预处理使`input_data_before`符合NPU上算子入参的要求。
- **步骤 2 — 运行GPU Kernel获取输出**：在GPU上运行GPU kernel，得到 `gpu_output`，并将 Tensor 转为 CPU。
- **步骤 3 — 打包并保存**：把 `input_data_before`、`grid`、`gpu_output` 封装为字典，通过 `torch.save` 保存为 `{kernel_name}.pt`。如果有多组用例，保存为 list-of-dicts（`[case0, case1]`）。

**精简示例**

```python
import copy
import torch

DEVICE = torch.device("cuda:0")
batch_size = 2
grid = (batch_size,)

input_data = {
	"output_token_ids_ptr": torch.zeros((batch_size, 4), dtype=torch.int32, device=DEVICE),
	"cu_num_draft_tokens_ptr": torch.tensor([2, 1], dtype=torch.int32, device=DEVICE),
	# ... 其它字段
}

# 保存输入副本到 CPU
input_data_before = {
	k: (v.clone().cpu() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
	for k, v in input_data.items()
}
# 预处理 input_data_before 符合 NPU kernel 输入
input_data_before["npu_need_param_key"] = NPU_NEED_PARAMS_VALUE
# 运行 kernel（在 GPU / 参考实现上）并收集输出
triton_kernel[grid](**input_data)
# 这里用 input_data 作为示例，实际应调用对应的 triton/pytorch 函数
gpu_output = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in input_data.items()}

save_obj = {"input_data": input_data_before, "grid": grid, "gpu_output": gpu_output}
torch.save(save_obj, "<kernel_name>.pt")
# 多组用例场景：torch.save([save_obj1, save_obj2], "<kernel_name>.pt")
```

## 2、在triton-ascend新增三方kernel测试用例

- **步骤 1 — 在triton-ascend仓中新增kernel算子** ：本地验证阶段，在 kernels/xxx(例如vllm、sglang) 下新增与算子同名的 Python 文件，内容为Triton kernel函数。
- **步骤 2 — 本地测试** ：将pt文件放在kernels目录下，在项目根目录运行
python -m pytest -v third_party/ascend/unittest/kernels/test_triton_kernel.py

**说明**
- 指定单个 kernel：在项目根目录下执行 python -m pytest -v ascend/test/common/test_triton_kernel.py --kernel={kernel_name}
- pt文件查找策略：优先使用仓库内匹配的本地 pt，若本地不存在则按需从远端 OBS 下载 {kernel_name}.pt文件。
- 本地已存在的pt文件，在执行完测试后不会删除，从obs桶取的文件在跑完测试后会被测试程序直接删除。

## 3、将pt文件上传至obs桶
本地验证通过后，将pt文件统一上传到OBS桶当中，OBS桶链接：https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/test/kernels/{xxx}_pt/{kernel_name}.pt，xxx为vllm或sglang
