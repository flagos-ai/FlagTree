"""Head layouts must be installed before cooperative preparation copies loops."""
from jinja2 import ChoiceLoader, DictLoader, Environment

from triton.flagmega.codegen.triton import renderer, tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from . import head_load


def test_layout_is_present_before_preparation_duplicates_head_loop(monkeypatch):
    load = "_fm_mla_w = tl.load({{ call.weight }} + ({{ call.weight_offset }}), mask=_fm_mla_mask, other=0.)"
    source = "{{ entry.kernel_decorator(call) }}\n" + load
    original_get_template = Environment.get_template
    observed = []

    def prepare(self, name, context):
        original = self.environment.loader.get_source(self.environment, head_load.KEY)[0]
        observed.append("__memory_layout" in original)
        header, body = original.split("\n", 1)
        duplicated = header + "\n{% if cooperative %}\n" + body + "\n{% else %}\n" + body + "\n{% endif %}"
        self.environment.loader = ChoiceLoader([DictLoader({head_load.KEY: duplicated}), self.environment.loader])

    def render(self, name, context):
        self.prepare(name, context)
        self.environment.get_template(head_load.KEY)
        return self.environment.loader.get_source(self.environment, head_load.KEY)[0]

    monkeypatch.setattr(TritonTemplateRegistry, "prepare", prepare)
    monkeypatch.setattr(TritonTemplateRegistry, "render", render)
    monkeypatch.setattr(Environment, "get_template", original_get_template)
    monkeypatch.setattr(tir_package, "describe_tir_package", tir_package.describe_tir_package)
    monkeypatch.setattr(renderer.renderer_registry, "_specs", dict(renderer.renderer_registry._specs))
    head_load.activate()
    registry = TritonTemplateRegistry()
    registry.environment.loader = DictLoader({head_load.KEY: source})
    output = registry.render("module.py.jinja", dict(render_calls=[]))
    assert observed == [True]
    assert output.count("tle.gpu.set_layout(tl.load(tle.gpu.set_layout(") == 2
