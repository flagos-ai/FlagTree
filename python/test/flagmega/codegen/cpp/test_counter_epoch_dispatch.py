from dataclasses import replace

import pytest

from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.cpp.renderer import describe_cpp_package, render_cpp_package
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.cpp.test_cpp_package import _elementwise_module


@pytest.fixture(scope='module')
def nonserial_module():
    return Compiler(CompileOptions(target='nvidia-sm120', bufferize_opt_level='fast')).compile(
        _elementwise_module()).module


def test_unrequested_nonserial_package_keeps_legacy_launch(nonserial_module):
    package = describe_cpp_package(nonserial_module)
    assert 'counter_epoch_plan' not in nonserial_module.metadata
    assert 'cpp_programs' not in package and 'cpp_counter_epoch_plan' not in package
    assert package['launch_abi_version'] == 2
    source = render_cpp_package(package, 'counter-epoch-dispatch-control')
    assert 'flagmega_launch' in source


@pytest.mark.parametrize('explicit_plan', [None, {}, {'schema': 'unsupported/v99'}])
def test_actual_nonserial_dispatch_rejects_every_explicit_companion(nonserial_module, explicit_plan):
    requested = replace(nonserial_module, metadata={**nonserial_module.metadata,
                                                   'counter_epoch_plan': explicit_plan})
    # This exercises the actual public dispatcher used by normal codegen, not
    # a mocked descriptor or the serial-only configure helper.
    with pytest.raises(CodegenError, match='serial NVIDIA program package'):
        describe_cpp_package(requested)
