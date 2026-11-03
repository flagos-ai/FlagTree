"""HCU loop hints reach LLVM without dropping independent LICM metadata."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _mm(A, B, C, LICM: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 128)
    k = tl.arange(0, 32)
    acc = tl.zeros((64, 128), tl.float32)
    for i in tl.range(0, 1024, 32, num_stages=2, disable_licm=LICM):
        a = tle.load(A + m[:, None] * 1024 + (i + k)[None, :], is_async=True)
        b = tle.load(B + (i + k)[:, None] * 128 + n[None, :], is_async=True)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * 128 + n[None, :], acc)


class _Captured(Exception):
    pass


@pytest.mark.parametrize('dtype', ['bf16', 'fp16'])
@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('licm', [False, True])
def test_llvm_loop_unroll_metadata(dtype, disable, licm, monkeypatch, tmp_path):
    pytest.importorskip('triton.backends.hcu.compiler_hcu')
    # Find the class actually selected by the backend registry.
    from triton.compiler.compiler import make_backend
    instance = make_backend(GPUTarget('hip', 'gfx936', 64))
    captured = []

    def capture(llir, metadata, options):
        captured.append(llir)
        raise _Captured

    monkeypatch.setattr(knobs.cache, 'dir', str(tmp_path / 'cache'))
    monkeypatch.setattr(knobs.hcu, 'use_buffer_ops', True)
    monkeypatch.setattr(type(instance), 'make_amdgcn', staticmethod(capture))
    source = ASTSource(
        _mm, signature={p: '*' + dtype for p in ['A', 'B', 'C']},
        constexprs={'LICM': licm},
        attrs={(i,): [('tt.divisibility', 16), ('tt.pointer_range', 32)]
               for i in range(3)},
    )
    with pytest.raises(_Captured):
        triton.compile(source, target=GPUTarget('hip', 'gfx936', 64),
                       options={'num_warps': 4, 'num_stages': 2,
                                'disable_llvm_loop_unroll': disable})
    assert len(captured) == 1
    assert ('"llvm.loop.unroll.disable"' in captured[0]) == disable
    assert ('"llvm.licm.disable"' in captured[0]) == licm
    assert instance.parse_options({}).disable_llvm_loop_unroll is False
    assert instance.parse_options({}).hash() != instance.parse_options(
        {'disable_llvm_loop_unroll': True}).hash()


def test_explicit_llvm_unroll_hint_takes_precedence(tmp_path):
    from triton._C.libtriton import hcu, ir
    from triton.compiler.compiler import make_backend
    context = ir.context()
    ir.load_dialects(context)
    make_backend(GPUTarget('hip', 'gfx936', 64)).load_dialects(context)
    source = tmp_path / 'loop.mlir'
    source.write_text('''module {
  tt.func @loop(%end: index) -> i32 {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %value = arith.constant 0 : i32
    %result = scf.for %i = %zero to %end step %one
        iter_args(%acc = %value) -> (i32) {
      scf.yield %acc : i32
    } {llvm.loop_annotation = #llvm.loop_annotation<
        unroll = #llvm.loop_unroll<count = 4 : i32>,
        licm = #llvm.loop_licm<disable = true>>}
    tt.return %result : i32
  }
}
''')
    module = ir.parse_mlir_module(str(source), context)
    module.context = context
    assert module.verify()
    before = module.str()
    hcu.disable_llvm_loop_unroll(module)
    assert module.str() == before
