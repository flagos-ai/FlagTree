import torch
import triton
import triton.language as tl


@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.load(Y + tl.arange(0, BLOCK))
    tl.device_print("x: ", x, hex=True)
    tl.device_print("constant : ", tl.constexpr(42))
    tl.store(Y + tl.arange(0, BLOCK), x)


def test_print():
    x = torch.arange(16, dtype=torch.int32)
    x.reshape(4, 4)
    y = torch.zeros_like(x)
    kernel_device_print[(1, )](x, y, BLOCK=16)


if __name__ == "__main__":
    # Run the test with pytest
    test_print()
