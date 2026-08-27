import re

from triton.backends.compiler import GPUTarget

# Iluvatar prints SME/TCU layout parameters even at their defaults, specializes on a 4-byte
# alignment granularity instead of 16, and has no custom assembly format for tt.load (its SME
# variant takes an extra operand), so tt.load spells out attributes upstream leaves implicit.
# None of that changes the IR's meaning, so erase it before comparing against upstream text.
COREX_PATS = [
    (re.compile(', isSme = false, smeMask = false, smeWarpsPerCTA = \\[\\]'
                '|, useTcu = false'
                '|, useSme = 0'
                '|, kRotate = 0'), ''),
    (re.compile('(: i\\d+) \\{tt.divisibility = 4 : i32\\}'), '\\1'),
    (re.compile('tt.divisibility = 4 : i32'), 'tt.divisibility = 16 : i32'),
    (re.compile(' \\{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false\\}'), ''),
]


def filecheck_make_ir(src, target, options, codegen_fns, module_map, context):
    return src.make_ir(target, options, codegen_fns, module_map, context)


def spec_get_stub_target() -> GPUTarget:
    return GPUTarget("corex", 71, 64)


def filecheck_default_kwargs(kwargs, target):
    # Backends derive their warp size from the arch they were built for rather than from the
    # target handed to them, so a backend compiling for a foreign target would otherwise
    # stamp its own warp size onto the module and reject that target's layouts.
    if "warp_size" in kwargs:
        return kwargs
    kwargs = dict(kwargs)
    kwargs["warp_size"] = target.warp_size
    return kwargs


def filecheck_anonymize_ir(module_str):
    # NOTE: Must use absolute path import.
    from triton._internal_testing import is_corex
    if not is_corex():
        return module_str
    for pat, repl in COREX_PATS:
        module_str = pat.sub(repl, module_str)
    return module_str
