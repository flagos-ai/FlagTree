from triton.flagmega import ir as fm
from triton.flagmega.passes import (
    FunctionTraversalOrder,
    FunctionalFunctionPass,
    PassManager,
    current_pass_context,
)


def _two_function_module():
    builder = fm.IRBuilder(dialect="high_level", stage="unit")
    value_type = fm.tensor_type("float32", [1])
    main_input = builder.var("main_input", value_type, id="main_input")
    callee_input = builder.var("callee_input", value_type, id="callee_input")
    call = builder.call(
        "builtin.call",
        [main_input],
        value_type,
        id="call",
        attrs={"callee": "callee"},
    )
    builder.function("main", [main_input], [call])
    builder.function("callee", [callee_input], [callee_input])
    return fm.verify_module(builder.build(entry="main"))


def test_function_pass_uses_module_order_and_scoped_context():
    seen = []

    def visit(function, module, context):
        assert context is current_pass_context()
        assert context.module is module
        assert context.function == function.name
        seen.append(function.name)
        return module

    result = PassManager("functions").add(
        FunctionalFunctionPass("visit", visit)
    ).run(_two_function_module())

    assert seen == ["main", "callee"]
    assert result.executed == ("visit",)


def test_function_pass_can_request_callee_first_traversal():
    seen = []

    result = PassManager("functions").add(FunctionalFunctionPass(
        "visit",
        lambda function, module, _context: seen.append(function.name) or module,
        traversal_order=FunctionTraversalOrder.CALLEE_FIRST,
    )).run(_two_function_module())

    assert seen == ["callee", "main"]
    assert result.module.function_map["main"].outputs == ("call",)
