# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prepare composed template overrides without evaluating discarded namespaces."""

from pathlib import Path
import json
import subprocess
import sys

from jinja2 import ChoiceLoader, DictLoader
import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError


def test_preparation_composes_context_and_loader_before_template_evaluation(tmp_path):
    (tmp_path / 'module.py.jinja').write_text('unprepared {{ value }}')
    events = []

    class First(TritonTemplateRegistry):
        def prepare(self, name, context):
            events.append('first')
            context['value'] *= 3
            self.environment.loader = ChoiceLoader([
                DictLoader({name: 'prepared {{ value }}'}), self.environment.loader])

    class Second(First):
        def prepare(self, name, context):
            events.append('second-before')
            context['value'] += 2
            super().prepare(name, context)
            events.append('second-after')

    assert Second(tmp_path).render('module.py.jinja', {'value': 5}) == 'prepared 21'
    assert events == ['second-before', 'first', 'second-after']


def test_failed_preparation_does_not_evaluate_a_template(tmp_path):
    (tmp_path / 'module.py.jinja').write_text('{{ missing_value }}')

    class Rejected(TritonTemplateRegistry):
        def prepare(self, name, context):
            raise CodegenError('override ownership proof failed')

    with pytest.raises(CodegenError, match='override ownership proof failed'):
        Rejected(tmp_path).render('module.py.jinja', {})


def test_glm_recipe_renders_two_selected_namespaces_once(tmp_path):
    # Recipe installation changes class callbacks. Isolate it from other tests.
    repo = Path(__file__).resolve().parents[5]
    probe = r'''
from pathlib import Path
import json
import sys
import jinja2
repo = Path.cwd()
sys.path.insert(0, str(repo / 'python/tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'))
from local_optimizations import activate_base
activate_base()
from python.test.flagmega.codegen.triton.test_program_source_cache import compile_pair
from triton.flagmega.codegen.triton import render_triton_package
module = compile_pair('tir.dense_matmul.packed_k_major_mma')
visits = []
original = jinja2.Template.render
def counted(self, *args, **kwargs):
    if self.name == 'module.py.jinja':
        visits.append(self.name)
    return original(self, *args, **kwargs)
jinja2.Template.render = counted
package = render_triton_package(module, Path(sys.argv[1]))
print('preparation-probe:' + json.dumps(dict(namespaces=len(package['program_sources']),
                                          module_renders=len(visits))))
'''
    result = subprocess.run([sys.executable, '-c', probe, str(tmp_path / 'rendered')],
                            cwd=repo, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    line, = [line for line in result.stdout.splitlines() if line.startswith('preparation-probe:')]
    counts = json.loads(line.split(':', 1)[1])
    assert counts == {'namespaces': 2, 'module_renders': 2}, counts
