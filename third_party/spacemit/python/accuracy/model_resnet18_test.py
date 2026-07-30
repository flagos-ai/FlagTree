import copy

import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems
# import torchvision.models as models


def test_accuracy_resnet18(dtype, device):
    # model = models.resnet18(pretrained=False).eval()
    model = torch.hub.load('/home/share/nfs_share/pytorch_vision_v0.10.0', 'resnet18', pretrained=True, source='local')
    inputs = torch.randn(1, 3, 224, 224).to(device)

    ref_model = copy.deepcopy(model)
    ref_model = ref_model.to(torch.float32).to(device).eval()
    ref_inputs = inputs.clone().to(torch.float32)

    with torch.no_grad():
        ref_output = ref_model(ref_inputs)

    res_model = copy.deepcopy(model)
    res_model = res_model.to(dtype).to(device).eval()
    res_inputs = inputs.clone().to(dtype)

    with flag_gems.use_gems():
        with torch.no_grad():
            res_output = res_model(res_inputs)

    score = torch.nn.functional.cosine_similarity(
        ref_output.flatten(),
        res_output.flatten(),
        dim=0,
        eps=1e-6,
    )
    print("score", score)
    assert torch.allclose(ref_output, res_output, atol=1e-2, rtol=1e-2)
    print("PASS")


if __name__ == "__main__":
    dtype = torch.float32
    device = "cpu"
    test_accuracy_resnet18(dtype, device)
