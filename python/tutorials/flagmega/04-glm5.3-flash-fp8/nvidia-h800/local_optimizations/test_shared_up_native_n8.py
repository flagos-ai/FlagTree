"""Keep all shared GLU K32 updates when removing unused duplicate columns."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from local_optimizations import stage5


KEY = 'kernels/matmul_glu/platforms/nvidia/sm90/cooperative_wgmma.py.jinja'


def template_root():
    return Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton'


def test_actual_recipe_selects_native_n8_shared_up_template(monkeypatch):
    monkeypatch.setattr(Environment, 'get_template', Environment.get_template)
    stage5.activate()
    env = Environment(loader=FileSystemLoader(template_root()))
    env.get_template(KEY)
    changed = env.loader.get_source(env, KEY)[0]
    original = (template_root() / KEY).read_text()
    old = 'tl.broadcast_to(_fm_q[:, None], (128, 16))'
    new = 'tl.broadcast_to(_fm_q[:, None], (128, 8))'
    assert changed.count(new) == 1
    assert old not in changed
    assert changed.replace(new, old) == original
    assert changed.count('out_dtype=tl.float32') == 2
    assert 'for _fm_block in tl.static_range({{ call.k_blocks }})' in changed
