import torch

from triton.flagmega.ir import DType, tensor_type
from triton.flagmega.ir.ops.nn._kda_state import KdaStateConfig, create_kda_state


def test_kda_state_has_layer_object_contract():
    config = KdaStateConfig(
        layer_ids=(0, 4),
        convolution_type=tensor_type(DType.BFLOAT16, (12, 3)),
        recurrent_type=tensor_type(DType.FLOAT32, (1, 2, 2, 2)),
    )
    state = create_kda_state(config)
    assert tuple(state.convolution.shape) == (2, 12, 3)
    assert tuple(state.recurrent.shape) == (2, 1, 2, 2, 2)

    convolution = torch.ones((12, 3), dtype=torch.bfloat16)
    recurrent = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    state.update_convolution_layer(convolution, 4)
    state.update_recurrent_layer(recurrent, 4)
    assert torch.equal(state.convolution_layer(4), convolution)
    assert torch.equal(state.recurrent_layer(4), recurrent)
    assert torch.count_nonzero(state.convolution_layer(0)) == 0
    assert torch.count_nonzero(state.recurrent_layer(0)) == 0
