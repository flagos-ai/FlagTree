import copy
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


def set_module_names(model):
    """为每个模块设置可读名称"""
    for name, module in model.named_modules():
        module.name = name


def compare_outputs(ref_out, test_out, name, atol=1e-2, rtol=1e-2):
    """比较两个输出并返回误差报告"""
    ref_out = ref_out.detach().float()
    test_out = test_out.detach().float()

    abs_diff = torch.abs(ref_out - test_out)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()

    rel_diff = abs_diff / (torch.abs(ref_out) + 1e-7)
    max_rel_diff = torch.max(rel_diff).item()

    passed = torch.allclose(ref_out, test_out, atol=atol, rtol=rtol)

    report = {
        'name': name, 'passed': passed, 'max_abs_diff': max_abs_diff, 'mean_abs_diff': mean_abs_diff, 'max_rel_diff':
        max_rel_diff
    }
    return report


def test_accuracy_resnet18(dtype, device):
    # 加载模型
    model = torch.hub.load('/home/share/nfs_share/pytorch_vision_v0.10.0', 'resnet18', pretrained=True, source='local')

    # 准备输入
    inputs = torch.randn(1, 3, 224, 224).to(device)

    # 参考模型 (FP32)
    ref_model = copy.deepcopy(model)
    ref_model = ref_model.float().to(device).eval()
    ref_inputs = inputs.clone().float()

    # 测试模型 (目标精度)
    test_model = copy.deepcopy(model)
    test_model = test_model.to(dtype).to(device).eval()
    test_inputs = inputs.clone().to(dtype)

    # 设置模块名称用于标识
    set_module_names(ref_model)
    set_module_names(test_model)

    # 存储各层输出
    ref_outputs = {}
    test_outputs = {}

    # 注册钩子收集输出
    def ref_hook(module, input, output):
        ref_outputs[module.name] = output

    def test_hook(module, input, output):
        test_outputs[module.name] = output

    # 注册钩子
    ref_handles = []
    test_handles = []
    for module in ref_model.modules():
        ref_handles.append(module.register_forward_hook(ref_hook))
    for module in test_model.modules():
        test_handles.append(module.register_forward_hook(test_hook))

    # 运行参考模型
    with torch.no_grad():
        ref_output = ref_model(ref_inputs)

    # 运行测试模型
    with flag_gems.use_gems():
        with torch.no_grad():
            test_output = test_model(test_inputs)

    # 移除钩子
    for handle in ref_handles + test_handles:
        handle.remove()

    # 最终输出比较
    final_score = torch.nn.functional.cosine_similarity(
        ref_output.flatten(),
        test_output.flatten(),
        dim=0,
        eps=1e-6,
    )
    print(f"Final output cosine similarity: {final_score.item():.6f}")

    # 逐层比较
    layer_reports = []
    for name in ref_outputs:
        if name not in test_outputs:
            continue
        report = compare_outputs(ref_outputs[name], test_outputs[name], name)
        layer_reports.append(report)

    # 打印失败层信息
    failed_layers = [r for r in layer_reports if not r['passed']]
    if failed_layers:
        print("\nFAILED LAYERS:")
        for r in failed_layers:
            print(f"{r['name']}: "
                  f"max_abs_diff={r['max_abs_diff']:.4e}, "
                  f"max_rel_diff={r['max_rel_diff']:.4e}")
    else:
        print("\nAll layers passed tolerance check")

    # 最终断言
    assert len(failed_layers) == 0, \
        f"{len(failed_layers)} layers failed accuracy check"
    print("PASS")


if __name__ == "__main__":
    dtype = torch.float32
    device = "cpu"
    test_accuracy_resnet18(dtype, device)
