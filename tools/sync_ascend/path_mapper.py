"""
读取 triton-ascend 的文件路径（相对仓库根），输出 JSON 格式的目标列表。
每条记录：{"repo": "flagtree"|"flir", "path": "<相对目标仓库根的路径>"}
"""
import sys
import os
import json

# ==============================================================
# 忽略规则（静默跳过，记录到 skipped.txt 但不报警告）
#
# SKIP_PREFIXES  : 路径以这些前缀开头时忽略
# SKIP_EXACT     : 路径完全匹配时忽略
# 注：triton-ascend 根目录下的文件（路径中不含 /）单独检测
# ==============================================================
SKIP_PREFIXES = [
    'docker/',
    'docs/',
    'third_party/ascend/AscendNPU-IR/',
]

SKIP_EXACT = {
    'third_party/ascend/.gitignore',
}

# ==============================================================
# 新增文件需人工处理的路径前缀
# 当一个文件是新增（created.txt）且路径匹配以下前缀时，
# 在 created.txt 中标注 (需人工处理)
# ==============================================================
MANUAL_REVIEW_PREFIXES = [
    'bin/',
    'include/',
    'lib/',
    'python/',
    'test/',
]

# ==============================================================
# 前缀映射规则（PATH_PREFIX_RULES）
#
# 每条规则字段：
#   src      : triton-ascend 侧路径前缀（必填）
#   dst      : 目标路径前缀，strip src 后拼接（必填）
#   dst_repo : 目标仓库，'flagtree' 或 'flir'（必填）
#   rename   : basename 重命名表 {old: new}，不填则不做重命名
#
# 规则按顺序匹配，第一条命中即返回结果。
# 需要同时写入多个目标（如 LIB_BOTH）的情形使用下方文件集合配置。
# ==============================================================
PATH_PREFIX_RULES = [
    # ----------------------------------------------------------------
    # third_party/ascend/include/  →  flir/include/
    # ----------------------------------------------------------------
    {
        'src': 'third_party/ascend/include/Dialect/TritonAscend/IR/', 'dst': 'include/npu/Dialect/TritonAscend/IR/',
        'dst_repo': 'flir'
    },
    {
        'src': 'third_party/ascend/include/Dialect/TritonStructured/IR/', 'dst':
        'include/incubated/Dialect/TritonStructuredIncubated/IR/', 'dst_repo': 'flir', 'rename': {
            'TritonStructuredDialect.h': 'TritonStructuredDialectIncubated.h',
            'TritonStructuredDialect.td': 'TritonStructuredDialectIncubated.td',
        }
    },
    {
        'src': 'third_party/ascend/include/DiscreteMaskAccessConversion/', 'dst':
        'include/incubated/Conversion/DiscreteMaskAccessConversion/', 'dst_repo': 'flir'
    },
    {
        'src': 'third_party/ascend/include/TritonToAnnotation/', 'dst':
        'include/incubated/Conversion/TritonToAnnotation/', 'dst_repo': 'flir'
    },
    {
        'src': 'third_party/ascend/include/TritonToLinalg/', 'dst':
        'include/incubated/Conversion/TritonToLinalgIncubated/', 'dst_repo': 'flir', 'rename': {
            'TritonToLinalgPass.h': 'TritonToLinalgIncubatedPass.h',
        }
    },
    {
        'src': 'third_party/ascend/include/TritonToStructured/', 'dst':
        'include/incubated/Conversion/TritonToStructuredIncubated/', 'dst_repo': 'flir', 'rename': {
            'TritonToStructuredPass.h': 'TritonToStructuredIncubatedPass.h',
        }
    },
    {
        'src': 'third_party/ascend/include/TritonToUnstructure/', 'dst':
        'include/incubated/Conversion/TritonToUnstructureIncubated/', 'dst_repo': 'flir'
    },
    {
        'src': 'third_party/ascend/include/Utils/', 'dst': 'include/incubated/Conversion/UtilsIncubated/', 'dst_repo':
        'flir'
    },

    # ----------------------------------------------------------------
    # third_party/ascend/include/  →  flagtree/third_party/ascend/include/
    # ----------------------------------------------------------------
    {
        'src': 'third_party/ascend/include/TritonToHFusion/', 'dst': 'third_party/ascend/include/TritonToHFusion/',
        'dst_repo': 'flagtree'
    },
    {
        'src': 'third_party/ascend/include/TritonToHIVM/', 'dst': 'third_party/ascend/include/TritonToHIVM/',
        'dst_repo': 'flagtree'
    },
    {
        'src': 'third_party/ascend/include/TritonToLLVM/', 'dst': 'third_party/ascend/include/TritonToLLVM/',
        'dst_repo': 'flagtree'
    },

    # ----------------------------------------------------------------
    # third_party/ascend/lib/  →  flir/lib/
    # ----------------------------------------------------------------
    {
        'src': 'third_party/ascend/lib/Dialect/TritonAscend/IR/', 'dst': 'lib/Dialect/TritonAscend/IR/', 'dst_repo':
        'flir'
    },
    {
        'src': 'third_party/ascend/lib/Dialect/TritonStructured/IR/', 'dst':
        'lib/Dialect/TritonStructuredIncubated/IR/', 'dst_repo': 'flir', 'rename': {
            'TritonStructuredDialect.cpp': 'TritonStructuredDialectIncubated.cpp',
            'TritonStructuredOps.cpp': 'TritonStructuredOpsIncubated.cpp',
        }
    },
    {
        'src': 'third_party/ascend/lib/DiscreteMaskAccessConversion/', 'dst':
        'lib/Conversion/DiscreteMaskAccessConversion/', 'dst_repo': 'flir'
    },
    {
        'src': 'third_party/ascend/lib/TritonToAnnotation/', 'dst': 'lib/Conversion/TritonToAnnotation/', 'dst_repo':
        'flir'
    },
    {
        'src': 'third_party/ascend/lib/TritonToLinalg/', 'dst': 'lib/Conversion/TritonToLinalgIncubated/', 'dst_repo':
        'flir', 'rename': {
            'TritonToLinalgPass.cpp': 'TritonToLinalgIncubatedPass.cpp',
        }
    },
    {
        'src': 'third_party/ascend/lib/TritonToStructured/', 'dst': 'lib/Conversion/TritonToStructuredIncubated/',
        'dst_repo': 'flir', 'rename': {
            'TritonToStructuredPass.cpp': 'TritonToStructuredIncubatedPass.cpp',
        }
    },
    {
        'src': 'third_party/ascend/lib/TritonToUnstructure/', 'dst': 'lib/Conversion/TritonToUnstructureIncubated/',
        'dst_repo': 'flir'
    },
    {'src': 'third_party/ascend/lib/Utils/', 'dst': 'lib/UtilsIncubated/', 'dst_repo': 'flir'},

    # ----------------------------------------------------------------
    # third_party/ascend/lib/  →  flagtree/third_party/ascend/lib/Conversion/
    # ----------------------------------------------------------------
    {
        'src': 'third_party/ascend/lib/TritonToHFusion/', 'dst': 'third_party/ascend/lib/Conversion/TritonToHFusion/',
        'dst_repo': 'flagtree'
    },
    {
        'src': 'third_party/ascend/lib/TritonToHIVM/', 'dst': 'third_party/ascend/lib/Conversion/TritonToHIVM/',
        'dst_repo': 'flagtree'
    },
    {
        'src': 'third_party/ascend/lib/TritonToLLVM/', 'dst': 'third_party/ascend/lib/Conversion/TritonToLLVM/',
        'dst_repo': 'flagtree'
    },

    # ----------------------------------------------------------------
    # include/runtime/libentry/  →  spec/include/
    # ----------------------------------------------------------------
    {
        'src': 'include/runtime/libentry/', 'dst': 'third_party/ascend/backend/spec/include/runtime/libentry/',
        'dst_repo': 'flagtree'
    },

    # ----------------------------------------------------------------
    # third_party/ascend/unittest/  →  flagtree 同路径
    # ----------------------------------------------------------------
    {'src': 'third_party/ascend/unittest/', 'dst': 'third_party/ascend/unittest/', 'dst_repo': 'flagtree'},
]

# ==============================================================
# 文件集合配置（用于需要按文件名分流的目录）
# ==============================================================

# include/triton/Dialect/Triton/IR/ 下：
#   仅走 spec 的文件（其余未列出的归主干）
INCLUDE_TRITON_IR_SPEC = {
    'OpInterfaces.h',
    'TritonAttrDefs.td',
    'TritonOpInterfaces.td',
    'TritonOps.td',
    'TritonTypes.td',
}

# lib/ 下：
#   只走 spec（主干无对应）
LIB_SPEC_ONLY = {
    'runtime/libentry/libentry.cpp',
}
#   同时写入主干 + spec
LIB_BOTH = {
    'Dialect/Triton/IR/Dialect.cpp',
    'Dialect/Triton/IR/Ops.cpp',
    'Dialect/Triton/IR/Traits.cpp',
}

# python/triton/ 下：
#   只走 spec（主干标"不使用"）
PYTHON_TRITON_SPEC = {
    'compiler/code_generator.py',
    'compiler/compiler.py',
    'compiler/errors.py',
    'language/_utils.py',
    'language/core.py',
    'language/math.py',
    'language/semantic.py',
    'language/standard.py',
    'runtime/autotuner.py',
    'runtime/code_cache.py',
    'runtime/interpreter.py',
    'runtime/jit.py',
    'runtime/libentry.py',
}
#   只走主干（spec 无对应）
PYTHON_TRITON_MAIN = {
    'compiler/__init__.py',
    'language/__init__.py',
    'runtime/__init__.py',
}

# python/src/ 下：
#   走 spec（third_party/ascend/python/src/）
PYTHON_SRC_SPEC = {'ir.cc', 'ir.h', 'main.cc'}
#   走主干
PYTHON_SRC_MAIN = {'interpreter.cc', 'llvm.cc', 'passes.cc', 'passes.h'}

# ==============================================================
# 辅助函数
# ==============================================================


def _apply_rename(rename_table, path):
    """对路径的 basename 应用重命名表"""
    dn = os.path.dirname(path)
    bn = os.path.basename(path)
    new_bn = rename_table.get(bn, bn)
    return os.path.join(dn, new_bn) if dn else new_bn


def _apply_rule(rule, path):
    """对单条 PATH_PREFIX_RULES 规则求目标路径列表，未命中返回 None"""
    src = rule['src']
    if not path.startswith(src):
        return None
    rest = path[len(src):]
    rename_table = rule.get('rename')
    if rename_table:
        rest = _apply_rename(rename_table, rest)
    return [{'repo': rule['dst_repo'], 'path': rule['dst'] + rest}]


# ==============================================================
# 主映射函数
# ==============================================================


def map_path(triton_path):
    """
    返回 [{"repo": ..., "path": ...}, ...] 列表。
    返回 'skip' 表示静默跳过（记录到 skipped.txt，不报警告）。
    返回 None   表示未知路径（报警告）。
    """
    p = triton_path

    # 根目录文件（路径中不含 /）
    if '/' not in p:
        return 'skip'

    # SKIP_EXACT
    if p in SKIP_EXACT:
        return []

    # SKIP_PREFIXES
    # 同时处理带尾部 '/' 的前缀（目录内文件）和不带 '/' 的整体路径（如子模块引用）
    for pfx in SKIP_PREFIXES:
        if p.startswith(pfx) or p == pfx.rstrip('/'):
            return 'skip'

    # PATH_PREFIX_RULES
    for rule in PATH_PREFIX_RULES:
        result = _apply_rule(rule, p)
        if result is not None:
            return result

    # third_party/ascend/ 其余文件：直接同路径到 flagtree
    # (include/ 和 lib/ 已在上方规则中处理，不会走到这里)
    if p.startswith('third_party/ascend/'):
        return [{'repo': 'flagtree', 'path': p}]

    # include/ 文件
    if p.startswith('include/'):
        rest = p[len('include/'):]

        if rest.startswith('triton/Dialect/Triton/IR/'):
            bn = os.path.basename(rest)
            if bn in INCLUDE_TRITON_IR_SPEC:
                return [{'repo': 'flagtree', 'path': f'third_party/ascend/backend/spec/include/{rest}'}]
            return [{'repo': 'flagtree', 'path': p}]

        return [{'repo': 'flagtree', 'path': p}]

    # lib/ 文件
    if p.startswith('lib/'):
        rest = p[len('lib/'):]
        if rest in LIB_SPEC_ONLY:
            return [{'repo': 'flagtree', 'path': f'third_party/ascend/backend/spec/lib/{rest}'}]
        if rest in LIB_BOTH:
            return [
                {'repo': 'flagtree', 'path': p},
                {'repo': 'flagtree', 'path': f'third_party/ascend/backend/spec/lib/{rest}'},
            ]
        return [{'repo': 'flagtree', 'path': p}]

    # python/triton/ 文件
    if p.startswith('python/triton/'):
        rest = p[len('python/triton/'):]
        if rest in PYTHON_TRITON_SPEC:
            return [{'repo': 'flagtree', 'path': f'third_party/ascend/backend/spec/triton/{rest}'}]
        # compiler/、language/、runtime/ 下的所有文件（含未在 PYTHON_TRITON_SPEC
        # 中显式列出的新增文件）统一走 spec
        if (rest.startswith('compiler/') or rest.startswith('language/') or rest.startswith('runtime/')):
            return [{'repo': 'flagtree', 'path': f'third_party/ascend/backend/spec/triton/{rest}'}]
        return [{'repo': 'flagtree', 'path': p}]

    # python/src/ 文件
    if p.startswith('python/src/'):
        rest = p[len('python/src/'):]
        if rest in PYTHON_SRC_SPEC:
            return [{'repo': 'flagtree', 'path': f'third_party/ascend/python/src/{rest}'}]
        return [{'repo': 'flagtree', 'path': p}]

    # bin/ 文件：直接同路径到 flagtree
    if p.startswith('bin/'):
        return [{'repo': 'flagtree', 'path': p}]

    # 未匹配
    return None


if __name__ == '__main__':
    triton_path = sys.argv[1]
    result = map_path(triton_path)
    manual = isinstance(result, list) and any(t['repo'] == 'flagtree' and t['path'].startswith(pfx)
                                              for t in result
                                              for pfx in MANUAL_REVIEW_PREFIXES)
    if result is None:
        print(json.dumps({'status': 'unknown', 'targets': [], 'manual_review': False}))
    elif result == 'skip':
        print(json.dumps({'status': 'skip', 'targets': [], 'manual_review': False}))
    else:
        print(json.dumps({'status': 'ok', 'targets': result, 'manual_review': manual}))
