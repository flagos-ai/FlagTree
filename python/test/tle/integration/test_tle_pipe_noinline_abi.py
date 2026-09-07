# flagtree tle
"""Pipe identity and control-state ABI, independent of any model/codegen."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _send(endpoint, output, value):
    tl.store(output, value)
    endpoint.commit(0)


@triton.jit(noinline=True)
def _receive(endpoint, output, value):
    endpoint.wait(0)
    tl.store(output, value)


@triton.jit(noinline=True)
def _send_pair(first, second, output):
    _send(first, output, tl.program_id(0) + 11)
    _send(second, output + 1, tl.program_id(0) + 22)


@triton.jit(noinline=True)
def _receive_pair(first, second, output):
    _receive(first, output + 2, tl.program_id(0) + 33)
    _receive(second, output + 3, tl.program_id(0) + 44)


@triton.jit
def _two_control_pipes(output, SAME_NAME: tl.constexpr):
    first = tle.pipe(capacity=1, one_shot=True, name="first")
    second = tle.pipe(capacity=1, one_shot=True, name="first" if SAME_NAME else "second")
    tle.gpu.warp_specialize([
        (_send_pair, (first.writer(), second.writer(), output)),
        (_receive_pair, (first.reader(), second.reader(), output)),
    ], [1], [32])


@pytest.mark.require_tle("pipe", "gpu.warp_specialize")
@pytest.mark.parametrize("same_name", [False, True])
def test_distinct_control_pipes_share_nested_noinline_functions(same_name, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires NVIDIA Hopper warp specialization")
    output = torch.zeros(4, dtype=torch.int32, device="cuda")
    kernel = _two_control_pipes[(1,)](output, same_name, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), torch.tensor([11, 22, 33, 44], dtype=torch.int32))
    # Source decorators/TTIR are insufficient: the final device bodies and
    # calls must survive token lowering and LLVM optimization.
    llir = kernel.asm["llir"]
    for symbol in ("_send__", "_receive__", "_send_pair__", "_receive_pair__"):
        definitions = re.findall(r"^define .*@([^ (]*\." + symbol + r"[^ (]*)\(", llir, re.M)
        assert len(definitions) == 1, (symbol, definitions)
        assert re.search(r"\bcall\b[^\n]*@" + re.escape(definitions[0]) + r"\(", llir)
    # Fieldless pipes must also round-trip through MLIR's textual format.
    from triton._C.libtriton import ir
    context = ir.context()
    ir.load_dialects(context)
    from triton.backends.nvidia.compiler import CUDABackend
    CUDABackend(triton.runtime.driver.active.get_current_target()).load_dialects(context)
    ttir_path = tmp_path / "fieldless.ttir"
    ttir_path.write_text(kernel.asm["ttir"])
    ir.parse_mlir_module(str(ttir_path), context)
