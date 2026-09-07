from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.egraph import EGraphSession
from triton.flagmega.rules import RewriteResult, RewriteRule


def test_rule_snapshot_retains_hash_consed_helper_ids():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value = builder.var("value", fm.tensor_type("float32", (4,)), id="value")
    existing = builder.call("tensors.cast", (value,), value.type, id="existing", attrs={"dtype": "float32"})
    root = builder.call("math.add", (value, value), value.type, id="root")
    builder.function("main", (value,), (root, existing))
    module = builder.build(entry="main")
    seen = set()

    def rewrite(node, module):
        helper = replace(existing, id="helper_alias")
        return RewriteResult(replace(node, inputs=(helper.id, value.id)), (helper,))

    def inspect(node, module):
        for value_id in node.inputs:
            seen.add(module.node_map[value_id].id)
        return False

    rules = (
        RewriteRule("IntroduceAlias", matches=lambda node, module: node.id == "root" and node.inputs == ("value", "value"),
                    rewrite=rewrite),
        RewriteRule("InspectOperands", matches=inspect, rewrite=lambda node, module: None),
    )
    session = EGraphSession()
    session.construct(module)
    session.apply_rules("aliases", rules)
    assert "helper_alias" in seen
    assert session.graph.class_for_node("helper_alias") == session.graph.class_for_node("existing")
