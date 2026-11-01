"""The latent column overlay belongs only to its selected program namespace."""
import pytest
from jinja2 import DictLoader

from triton.flagmega.codegen.triton import renderer, tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from .dsa_column_owners import KEY, activate


@pytest.fixture
def registry(monkeypatch):
    def render(self, name, context):
        calls = context["render_calls"]
        if not calls or calls[0]["family"] != "dsa_paged_attention":
            return "unrelated program"
        return self.environment.get_template(KEY).render(call=calls[0])

    monkeypatch.setattr(TritonTemplateRegistry, "render", render)
    monkeypatch.setattr(tir_package, "describe_tir_package", tir_package.describe_tir_package)
    monkeypatch.setattr(renderer.renderer_registry, "_specs", dict(renderer.renderer_registry._specs))
    activate()
    registry = TritonTemplateRegistry()
    registry.environment.loader = DictLoader({KEY:
        "        _fm_dsa_dim = tl.arange(0, {{ call.padded_head_dim }})\n"
        "        acc = tl.zeros((16, {{ call.padded_head_dim }}), tl.float32)\n"})
    return registry


def selected_call(**changes):
    return dict(family="dsa_paged_attention", variant="decode", decode_schedule="bf16_mma",
                parameters={}, padded_head_dim=512, output_column_width=128,
                writer_active="block_index < 4", **changes)


def test_unrelated_program_does_not_receive_latent_overlay(registry):
    loader = registry.environment.loader
    assert registry.render("module.py.jinja", dict(render_calls=[dict(family="partitioned_linear")])) == "unrelated program"
    assert registry.environment.loader is loader


def test_selected_program_keeps_column_overlay_and_restores_loader(registry):
    loader = registry.environment.loader
    source = registry.render("module.py.jinja", dict(render_calls=[selected_call()]))
    assert "block_index * 128 + tl.arange(0, 128)" in source
    assert "tl.zeros((16, 128), tl.float32)" in source
    assert registry.environment.loader is loader


def test_selected_program_still_rejects_missing_owner_proof(registry):
    call = selected_call()
    call["output_column_width"] = 64
    with pytest.raises(CodegenError, match="normally propagated"):
        registry.render("module.py.jinja", dict(render_calls=[call]))


def test_multiple_latent_calls_remain_unsupported(registry):
    with pytest.raises(CodegenError, match="one selected latent implementation"):
        registry.render("module.py.jinja", dict(render_calls=[selected_call(), selected_call()]))
