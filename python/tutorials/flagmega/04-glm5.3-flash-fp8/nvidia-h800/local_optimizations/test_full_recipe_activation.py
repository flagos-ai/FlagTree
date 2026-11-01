"""Complete recipe must compose with the newer Native scheduling wrappers."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("recipe", ["base_recipe", "kernel_recipe"])
def test_recipe_keeps_native_wrappers_and_complete_overrides(recipe, tmp_path):
    tutorial = Path(__file__).resolve().parent.parent
    script = f"""
import sys
sys.path.insert(0, {str(tutorial)!r})
from local_optimizations.{recipe} import activate
activate()
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
current = TritonTemplateRegistry.render
modules, sources, seen = set(), [], set()
while callable(current) and id(current) not in seen:
    seen.add(id(current))
    modules.add(current.__module__)
    cells = dict(zip(current.__code__.co_freevars,
                     (cell.cell_contents for cell in current.__closure__ or ())))
    sources.extend(value for value in cells.values() if isinstance(value, str))
    parents = [cells[name] for name in ('original', 'previous_render', 'original_render')
               if name in cells and callable(cells[name])]
    assert len(parents) <= 1
    current = parents[0] if parents else None
for name in ('native_chunk_precision', 'native_dsa_pool_scores',
             'native_latent_attention_schedule', 'native_moe_down_load_schedule'):
    assert 'local_optimizations.' + name in modules, name
if {recipe!r} == 'kernel_recipe':
    assert 'local_optimizations.fences' in modules
    assert any('oproj-head.py.jinja' in source and 'oproj-tail.py.jinja' in source
               for source in sources)
    from triton.flagmega.codegen.triton import function_programs
    assert function_programs.project_function_programs.__module__ == 'local_optimizations.scalar_offsets'
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
