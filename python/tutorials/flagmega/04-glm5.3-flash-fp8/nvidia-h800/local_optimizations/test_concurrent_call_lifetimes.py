"""The declared CTA overlap must extend normal SSA input lifetimes before allocation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_optimizations.base_recipe import activate
from triton.flagmega.passes.tir.bufferize import program_lifetimes
from triton.flagmega.passes.tir.bufferize.allocation import BufferLifetime
from triton.flagmega.passes.tir.bufferize.first_fit_allocator import FirstFitBufferAllocator
from triton.flagmega.ir.bufferization import MemorySpace,AllocationStrategy
from triton.flagmega.errors import IRVerificationError

def inputs(groups):
    module=SimpleNamespace(entry='main',stage='selected_microkernels',metadata={'tle.concurrent_call_groups':{'body':groups}})
    function=SimpleNamespace(name='body')
    nodes=[SimpleNamespace(id=name,op=op) for name,op in (('input','builtin.var'),('recurrent','tir.call'),('gb','tir.call'),('result','builtin.var'))]
    return module,function,nodes

def test_declared_overlap_keeps_recurrent_input_live_through_gb_write():
    activate()
    module,function,nodes=inputs((('recurrent','gb'),))
    ends=program_lifetimes.program_use_ends(module,function,nodes)
    space=MemorySpace(name='workspace',kind='global',granularity=256,maximum_bytes=4096,strategy=AllocationStrategy.REUSE)
    lifetimes=(BufferLifetime('recurrent_input',256,256,0,ends.get('recurrent',1)),
               BufferLifetime('gb_output',256,256,2,3))
    plan=FirstFitBufferAllocator().allocate(lifetimes,space)
    a,b=(plan.offset_map[name] for name in ('recurrent_input','gb_output'))
    assert a+256<=b or b+256<=a

def test_invalid_declared_overlap_call_is_rejected():
    activate()
    module,function,nodes=inputs((('recurrent','missing'),))
    with pytest.raises(IRVerificationError):
        program_lifetimes.program_use_ends(module,function,nodes)

def test_unannotated_body_uses_the_original_lifetime_policy():
    activate()
    module,function,nodes=inputs(())
    assert program_lifetimes.program_use_ends(module,function,nodes)=={}
