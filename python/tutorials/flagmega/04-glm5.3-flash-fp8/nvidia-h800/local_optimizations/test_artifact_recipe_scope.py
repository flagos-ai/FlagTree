"""Frozen identity guards and renderer state must survive artifact switching."""
from pathlib import Path
import sys

import pytest

from local_optimizations.artifact_recipe import activated_artifact_recipe, _hook_targets


def _recipe(tmp_path, name):
    trial = tmp_path / name
    package = trial / "local_optimizations"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "identity.py").write_text(f"IDENTITY = {name!r}\n")
    (package / "kernel_recipe.py").write_text('''
from local_optimizations.identity import IDENTITY
from triton.flagmega.codegen.triton import tir_package, renderer
from triton.flagmega.codegen.triton.registry import PackageRendererSpec
from triton.flagmega.errors import CodegenError
def activate():
    previous = tir_package.describe_tir_package
    def describe(module):
        if module != IDENTITY:
            raise CodegenError('O-proj override requires its frozen verified complete module')
        return {'identity': IDENTITY}
    tir_package.describe_tir_package = describe
    spec = renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(
        spec.name, spec.kind, spec.matches, describe, spec.render)
''')
    return trial


@pytest.mark.parametrize("reverse", [False, True])
def test_each_artifact_retains_its_own_frozen_identity(tmp_path, reverse):
    from triton.flagmega.codegen.triton import tir_package
    from triton.flagmega.errors import CodegenError

    trials = [_recipe(tmp_path, name) for name in ("first", "second")]
    if reverse:
        trials.reverse()
    baseline = tir_package.describe_tir_package
    baseline_path = list(sys.path)
    baseline_package = sys.modules["local_optimizations"]
    for trial in trials * 2:
        with activated_artifact_recipe(trial):
            assert tir_package.describe_tir_package(trial.name) == {"identity": trial.name}
            with pytest.raises(CodegenError, match="frozen verified complete module"):
                tir_package.describe_tir_package("different")
        assert tir_package.describe_tir_package is baseline
        assert sys.path == baseline_path
        assert sys.modules["local_optimizations"] is baseline_package


def test_hooks_registry_and_captured_source_restored_on_failure(tmp_path, monkeypatch):
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    trial = _recipe(tmp_path, "first")
    def original_render():
        source = "original inner renderer source"
        def render(*args):
            return source
        return render
    monkeypatch.setattr(TritonTemplateRegistry, "render", original_render())
    targets, registry = _hook_targets()
    baseline = [(owner, name, getattr(owner, name)) for owner, name in targets]
    specs = dict(registry._specs)
    with pytest.raises(RuntimeError, match="failed lazy preparation"):
        with activated_artifact_recipe(trial):
            # The O-proj installer makes this same kind of in-place mutation.
            baseline[2][2].__closure__[0].cell_contents = "extended source"
            for owner, name, _ in baseline:
                setattr(owner, name, lambda *args: None)
            registry._specs.clear()
            raise RuntimeError("failed lazy preparation")
    assert all(getattr(owner, name) is value for owner, name, value in baseline)
    assert registry._specs == specs
    assert TritonTemplateRegistry.render() == "original inner renderer source"
