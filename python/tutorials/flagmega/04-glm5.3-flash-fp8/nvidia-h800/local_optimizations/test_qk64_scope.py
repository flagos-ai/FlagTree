"""QK loading overlay must be scoped to its head-sharded latent program."""
import pytest
from jinja2 import DictLoader

from triton.flagmega.codegen.triton import renderer, tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from . import qk64


@pytest.fixture
def registry(monkeypatch):
    transformed = []
    def render(self, name, context):
        return self.environment.get_template(qk64.KEY).render()
    def transform(source):
        transformed.append(source)
        return "selected QK overlay"
    monkeypatch.setattr(TritonTemplateRegistry, "render", render)
    monkeypatch.setattr(tir_package, "describe_tir_package", tir_package.describe_tir_package)
    monkeypatch.setattr(renderer.renderer_registry, "_specs", dict(renderer.renderer_registry._specs))
    monkeypatch.setattr(qk64, "transform", transform)
    qk64.activate()
    registry = TritonTemplateRegistry()
    registry.environment.loader = DictLoader({qk64.KEY: "original source"})
    return registry, transformed


def selected_call():
    return dict(family="dsa_paged_attention", variant="decode", decode_schedule="bf16_mma",
                head_dim=512, num_query_heads=16, head_sharded=True)


def test_unrelated_namespace_keeps_original_renderer(registry):
    registry, transformed = registry
    loader = registry.environment.loader
    assert registry.render("module.py.jinja", dict(render_calls=[dict(family="partitioned_linear")])) == "original source"
    assert transformed == [] and registry.environment.loader is loader


def test_selected_namespace_keeps_overlay(registry):
    registry, transformed = registry
    loader = registry.environment.loader
    assert registry.render("module.py.jinja", dict(render_calls=[selected_call()])) == "selected QK overlay"
    assert transformed == ["original source"] and registry.environment.loader is loader


def test_selected_namespace_keeps_head_ownership_gate(registry):
    registry, transformed = registry
    call = selected_call()
    call["head_sharded"] = False
    with pytest.raises(CodegenError, match="exact selected head-sharded attention"):
        registry.render("module.py.jinja", dict(render_calls=[call]))
    assert transformed == []


def test_multiple_selected_calls_still_reject(registry):
    registry, transformed = registry
    with pytest.raises(CodegenError, match="exact selected head-sharded attention"):
        registry.render("module.py.jinja", dict(render_calls=[selected_call(), selected_call()]))
    assert transformed == []
