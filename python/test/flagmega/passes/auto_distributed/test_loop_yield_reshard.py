# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A loop Yield restores its shared iter-argument ABI before the backedge."""
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.ir.region import is_region


def mixed_loop(stop=3, kind='ordinary', second_stop=None):
    b = fm.IRBuilder(dialect='high_level', stage='imported')
    tensor, index = fm.tensor_type('float32', (4096,)), fm.tensor_type('int64', ())
    iv = b.var('iv', index, id='iv')
    carried = b.var('carried', tensor, id='carried')
    capture = b.var('capture', tensor, id='capture')
    with construction_scope(b):
        a = fm.F.math.add(carried, capture, name='fixed_before', metadata={'execution_mesh': (8, 16)})
        c = fm.F.math.add(a, capture, name='dynamic', metadata={'execution_mesh': '*',
            'kernel_program_kind': kind, **({'kernel_max_ctas': 2} if kind == 'persistent' else {})})
        d = fm.F.math.add(c, capture, name='fixed_after', metadata={'execution_mesh': (4, 16)})
    body = b.region('body', (iv, carried, capture), (d,), iter_count=1)
    value = b.var('value', tensor, id='value')
    increment = b.var('increment', tensor, id='increment')
    with construction_scope(b):
        lower = fm.F.builtin.scalar_const(index, 0, name='lower')
        upper = fm.F.builtin.scalar_const(index, stop, name='upper')
        stride = fm.F.builtin.scalar_const(index, 1, name='stride')
        result = fm.F.control.for_loop(lower, upper, stride, value, body=body,
                                      captures=(increment,), name='loop')
        output = fm.F.tensors.get_item(result, 0, name='output')
        if second_stop is not None:
            other_upper = fm.F.builtin.scalar_const(index, second_stop, name='other_upper')
            other = fm.F.control.for_loop(lower, other_upper, stride, output, body=body,
                                         captures=(increment,), name='other_loop')
            output = fm.F.tensors.get_item(other, 0, name='other_output')
    b.function('main', (value, increment), (output,))
    return b.build(entry='main')


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('stop', [0, 1, 3])
def test_mixed_loop_yield_restores_the_exact_iter_argument_type(stop, kind, tmp_path):
    source = mixed_loop(stop, kind)
    module = Compiler().compile(source, stop_after='auto-distributed').module
    region, = (f for f in module.functions if is_region(f))
    yielded, = region.outputs
    argument = region.parameters[1]
    assert module.node_map[yielded].type == module.node_map[argument].type
    assert yielded != 'fixed_after'  # Its original placement is (4, 16).
    assert module.node_map[yielded].op.startswith('distributed.')
    assert module.node_map['loop'].attrs == source.node_map['loop'].attrs
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'loop.py'))
    assert restored.semantic_hash == module.semantic_hash
    fm.verify_module(restored)


def test_reused_body_has_one_yield_adapter_and_preserves_both_loops():
    source = mixed_loop(stop=1, second_stop=2)
    module = Compiler().compile(source, stop_after='auto-distributed').module
    region, = (f for f in module.functions if is_region(f))
    yielded, = region.outputs
    assert module.node_map[yielded].type == module.node_map[region.parameters[1]].type
    assert [n.id for n in module.nodes if n.op == 'control.for'] == ['loop', 'other_loop']
    assert module.node_map['loop'].attrs['callee'] == module.node_map['other_loop'].attrs['callee'] == region.name
    assert sum(n.id == yielded for n in module.nodes) == 1


def test_loop_yield_path_remains_subject_to_target_boundary_policy():
    from triton.flagmega.errors import IRVerificationError
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.passes.auto_distributed.realization import (
        DistributedReshardRealization, DistributedReshardUsageKind,
    )

    class Target(NvidiaSm90Target):
        def distributed_reshard_realization_policy(self):
            delegate = super().distributed_reshard_realization_policy()

            class Policy:
                def classify(self, context):
                    if (context.usage_kind is DistributedReshardUsageKind.FUNCTION_BOUNDARY
                            and context.source_type != context.target_type):
                        return DistributedReshardRealization.UNSUPPORTED
                    return delegate.classify(context)

                def uses_sharded_views_for_constants(self):
                    return delegate.uses_sharded_views_for_constants()
            return Policy()

    compiler = Compiler()
    compiler.target = Target()
    with pytest.raises(IRVerificationError, match='INFEASIBLE'):
        compiler.compile(mixed_loop(), stop_after='auto-distributed')
