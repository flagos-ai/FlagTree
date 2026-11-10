"""Reviewed t26 ordinary counter policy. No model data is read by this module."""
from dataclasses import replace
from pathlib import Path
import json

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.counter_epoch_plan import (
    CounterEffectContract, CounterEpochPlan, register_counter_effect_contract,
)

_CONTRACT_SHA256 = '6247976660acf599d2df840a5594844a9f773c244d785a708f525d090b5695bc'
_PLAN_SHA256 = {
    'baseline275': '8bc24e869cc23a6dfbd543d3742c125827b60e407f9f0011337302fa5232e004',
    'candidate49': '0a2de14843f1d9465fc5a2407572ace1e30be3dc5efae16cf16623e69709e434',
}


def register():
    contract = CounterEffectContract.from_data(json.loads(Path(__file__).with_name('contract.json').read_text()))
    if contract.contract_sha256 != _CONTRACT_SHA256:
        raise IRVerificationError('Reset49 reviewed contract differs from the frozen implementation extension.')
    register_counter_effect_contract(contract)
    # Install after the inherited workload renderer extension. Its selected
    # source annotations must finish before the ordinary effect hash is checked.
    from triton.flagmega.codegen.cpp import renderer
    from triton.flagmega.codegen.cpp.programs import configure_counter_epoch_descriptor
    if not getattr(renderer.describe_cpp_package, '_counter_epoch_finalizer', False):
        previous = renderer.describe_cpp_package

        def describe(module):
            if 'counter_epoch_plan' not in module.metadata:
                return previous(module)
            base = replace(module, metadata={key: value for key, value in module.metadata.items()
                                             if key != 'counter_epoch_plan'})
            descriptor = previous(base)
            return configure_counter_epoch_descriptor(descriptor, module.metadata['counter_epoch_plan'])

        describe._counter_epoch_finalizer = True
        renderer.describe_cpp_package = describe
    return contract


def apply(module, schedule):
    """Attach the frozen typed companion after physical lowering, before C++ codegen.

    Normal codegen independently reconstructs all entries, invocations and device
    text and checks this companion against the registered effect contract.
    """
    if schedule not in _PLAN_SHA256:
        raise IRVerificationError('Reset49 permits only the two reviewed complete schedules.')
    if 'counter_epoch_plan' in module.metadata:
        raise IRVerificationError('Reset49 cannot silently replace an existing explicit companion.')
    register()
    plan = CounterEpochPlan.from_data(json.loads(Path(__file__).with_name(schedule+'.json').read_text()))
    if plan.plan_sha256 != _PLAN_SHA256[schedule]:
        raise IRVerificationError('Reset49 schedule differs from its frozen identity.')
    return replace(module, metadata={**module.metadata, 'counter_epoch_plan': plan.to_data()})
