# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The tutorial's output-only override requires a replicated activation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from triton.flagmega import ir as fm


def test_output_split_does_not_confuse_exclusive_and_replicated_operands():
    path = Path(__file__).parents[3] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/tp4_distribution.py'
    spec = importlib.util.spec_from_file_location('glm_tp4_projection_override', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mesh = fm.Placement((4,8,16), 'cyx', 'cbb')
    b = fm.SBP.broadcast()
    activation = fm.tensor_type('bfloat16', (1,128))
    output = fm.DistributedType(fm.tensor_type('bfloat16', (1,4096)),
                                (b,fm.SBP.split_contiguous((0,1,2))),mesh)
    candidates = []
    for axes in ((), (1,), (2,), (1,2)):
        value = fm.DistributedType(activation,(b,b),mesh,
                                  exclusive=fm.SBP.exclusive(axes) if axes else None)
        candidates.append(SimpleNamespace(id=str(axes),input_types=(value,),return_type=output))
    bucket = SimpleNamespace(node_id='projection',candidates=tuple(candidates))
    assert module.select_output_split(bucket,(0,1,2)).id == '()'
