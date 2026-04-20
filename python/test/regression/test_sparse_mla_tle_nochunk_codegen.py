import ast
from pathlib import Path

SPARSE_MLA_PATH = (Path(__file__).resolve().parents[2] / "tutorials" / "tle" / "deepseek_v32" / "02-sparse-mla.py")


def _get_function_source(function_name):
    source = SPARSE_MLA_PATH.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"function {function_name!r} not found")


def _get_function_ast(function_name):
    source = _get_function_source(function_name)
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def test_tle_sparse_mla_stages_full_kv_tile_without_static_chunks():
    source = _get_function_source("tle_sparse_mla_fwd")
    function = _get_function_ast("tle_sparse_mla_fwd")

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        assert not (isinstance(func, ast.Attribute) and func.attr
                    == "static_range"), "TLE sparse MLA must not split KV staging through static chunk loops"

    forbidden_fragments = [
        "offs_d_chunk",
        "offs_td_chunk",
        "kv_ptr_chunk",
        "tkv_ptr_chunk",
        "kv_chunk_ptr",
        "tkv_chunk_ptr",
        "p_smem",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source

    assert "tl.store(kv_smem_ptr, kv_blk, mask=kv_msk)" in source
    assert "tl.store(tkv_smem_ptr, tkv_blk, mask=tkv_msk)" in source


def test_tle_sparse_mla_masks_qk_before_dot_accumulation():
    source = _get_function_source("tle_sparse_mla_fwd")

    mask_pos = source.index("qk = tl.where(mask_ids[None, :], qk, float(\"-inf\"))")
    tail_dot_pos = source.index("qk = tl.dot(tq_blk, tl.trans(tkv_blk), qk")
    main_dot_pos = source.index("qk = tl.dot(q_blk, tl.trans(kv_blk), qk")

    assert mask_pos < tail_dot_pos < main_dot_pos
