"""The selected flags protocol has an explicit hardware/ISA contract."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

DOMAIN = tl.constexpr(tle.device_mesh({'block': [('block_x', 33)]}))


@triton.jit
def target_publication(data, output, SELECTED: tl.constexpr):
    owner = tl.program_id(0)
    tl.store(data + owner, owner)
    if SELECTED:
        tle.distributed_barrier(DOMAIN, protocol='flags32', launch_grid=(33, 1, 1))
    else:
        tle.distributed_barrier(DOMAIN)
    tl.store(output + owner, tl.load(data + (owner + 1) % 33))


def compile_target(sm, ptx, selected=True):
    return triton.compile(
        ASTSource(target_publication, {'data': '*i32', 'output': '*i32'},
                  constexprs={'SELECTED': selected}),
        target=GPUTarget('cuda', sm, 32),
        options={'num_warps': 8, 'ptx_version': ptx})


def test_flags_minimum_supported_target():
    compiled = compile_target(90, 86)
    assert compiled.metadata.global_scratch_size == 256
    assert 'st.release.gpu.u32' in compiled.asm['ptx']
    assert 'fence.acquire.gpu' in compiled.asm['ptx']
    assert 'vote.sync.all.pred' in compiled.asm['ptx']


@pytest.mark.parametrize('sm,ptx', [(80, 87), (89, 87), (90, 85)])
def test_flags_unsupported_target_rejected(sm, ptx, capfd):
    with pytest.raises(RuntimeError):
        compile_target(sm, ptx)
    captured = capfd.readouterr()
    assert 'flags32 requires NVIDIA SM90+ and PTX ISA8.6+' in captured.err


@pytest.mark.parametrize('sm,ptx', [(80, 87), (90, 85)])
def test_default_counter_remains_supported(sm, ptx):
    compiled = compile_target(sm, ptx, selected=False)
    assert compiled.metadata.global_scratch_size == 4
    assert 'atom.add.release.gpu' in compiled.asm['ptx']
    assert 'st.release.gpu.u32' not in compiled.asm['ptx']
