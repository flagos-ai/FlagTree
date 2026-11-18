import torch
import torch_npu
import triton
import triton.language as tl
import pytest


@triton.jit
def triton_asm_time(output_ptr, ):
    y = tl.inline_asm_elementwise(
        asm="""
            MOV $0, SYS_CNT
        """,
        constraints="=l",
        args=[],
        dtype=(tl.int64),
        is_pure=False,
        pack=1,
    )
    tl.store(output_ptr, y)


@pytest.mark.parametrize(
    "param_list",
    [[
        "int64",
    ]],
)
def test_case(param_list):
    (dtype, ) = param_list
    res_cal = torch.zeros((1, ), dtype=eval("torch." + dtype)).npu()
    triton_asm_time[(1, )](res_cal, )
