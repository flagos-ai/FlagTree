# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT


def test_public_namespaces_import():
    import triton.flagmega
    import triton.flagmega.diagnostics
    import triton.flagmega.egraph
    import triton.flagmega.evaluator
    import triton.flagmega.importer
    import triton.flagmega.ir
    import triton.flagmega.passes
    import triton.flagmega.pattern_match
    import triton.flagmega.rules
    import triton.flagmega.runtime
    import triton.flagmega.targets

    assert triton.flagmega.__version__.startswith("0.1")
    assert triton.flagmega.load is triton.flagmega.runtime.load
    assert triton.flagmega.DumpFlags.PassIR is triton.flagmega.DumpFlags.PASS_IR
