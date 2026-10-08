"""Guard: the `tle.raw` payload/call-site check compares *types*, nothing else.

`merge_raw_payloads` refuses to hand a module to `llc` when a payload call and
its definition disagree, because `llvm-link` links a mismatched pair and neither
the LLVM parser nor the verifier rejects the result -- `llc` then codegens a call
into a function with a different ABI.

That comparison is textual, so it has to ignore everything that is not part of
the ABI: the operand *values* at the call site (a payload call passes tile sizes
and descriptor addresses as literals or constant expressions) and the parameter
attributes.  Both spellings below are real compiler output:

  * the xpu3 checkin #274 failure -- a call site whose operands are literals
    (`i64 16, i64 64, ...`) and `addrspacecast` constant expressions, which the
    old parser read as part of the parameter types and rejected;
  * the same kernel built with line info on, where a `!dbg` follows the call's
    attribute group -- the old detector required `#N` at end-of-line, so it
    silently skipped the check in every default build.

Run with:  pytest third_party/xpu/python/test/unit/language/test_tle_raw_payload_signature.py
"""

import pytest

from triton.experimental.tle.raw import merge

# `mm_tile_f16` as its SDNN payload declares it: three pointer parameters, each
# followed by its four descriptor fields.
_PAYLOAD_SIG = "ptr, i64, i64, i64, i64, ptr, i64, i64, i64, i64, ptr, i64, i64, i64, i64"

# (16, 64, 64) tile: the descriptor fields arrive as literals, the two derived
# pointers as constant expressions.
_LITERAL_CALL = ("ptr %0, i64 16, i64 64, i64 64, i64 1, "
                 "ptr addrspacecast (ptr addrspace(2) null to ptr), i64 16, i64 64, i64 64, i64 1, "
                 "ptr addrspacecast (ptr addrspace(2) inttoptr (i32 2048 to ptr addrspace(2)) to ptr), "
                 "i64 64, i64 64, i64 64, i64 1")

# `getelementptr` keeps its inner commas: a list split on every comma would cut
# this operand in three and compare the fragments as types.
_EXPR_CALL = ("ptr getelementptr inbounds (i8, ptr @g, i64 4), i64 0, i64 0, i64 0, i64 0, "
              "ptr null, i64 0, i64 0, i64 0, i64 0, ptr null, i64 0, i64 0, i64 0, i64 0")


def _module(call_args, call_tail="#1", payload_sig=_PAYLOAD_SIG):
    return f"""; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:e-p:32:32"

define internal void @mm_tile_f16({payload_sig}) #1 {{
  ret void
}}

define void @mm_raw_kernel(ptr %0) #0 {{
  tail call void @mm_tile_f16({call_args}) {call_tail}
  ret void
}}

attributes #0 = {{ "triton_gpu.num-warps"="1" }}
attributes #1 = {{ alwaysinline }}
"""


@pytest.mark.parametrize("tail", ["#1", "#1, !dbg !33"], ids=["no-line-info", "line-info"])
def test_operand_values_are_not_part_of_the_abi(tail):
    merge._check_raw_calls_are_defined(_module(_LITERAL_CALL, tail))


def test_constant_expression_with_commas_is_one_operand():
    merge._check_raw_calls_are_defined(_module(_EXPR_CALL))


@pytest.mark.parametrize("tail", ["#1", "#1, !dbg !33"], ids=["no-line-info", "line-info"])
def test_a_real_type_mismatch_is_still_rejected(tail):
    """One descriptor field passed as i32 where the payload takes i64."""
    unique = "i64 64, i64 64, i64 1, ptr addrspacecast (ptr addrspace(2) null to ptr)"
    assert _LITERAL_CALL.count(unique) == 1
    mismatched = _LITERAL_CALL.replace(unique, unique.replace("i64 64, i64 64", "i32 64, i64 64", 1))
    with pytest.raises(RuntimeError, match="does not match the call site"):
        merge._check_raw_calls_are_defined(_module(mismatched, tail))
