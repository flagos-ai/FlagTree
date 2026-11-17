# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ast
import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.backends.ascend.runtime.autotuner as ascend_autotuner
from triton.backends.ascend.runtime.tile_generator import KernelMeta
from triton.backends.ascend.runtime.utils import is_valid_axis_name
import triton.language as tl

VALID_AXIS_NAMES = ["x", "y", "z", "w", "v", "t"]
AUTOTUNER_PATH = Path(__file__).resolve().parents[2] / "backend" / "runtime" / "autotuner.py"
VECTOR_AXES_PATH = Path(__file__).resolve().parents[2] / "backend" / "runtime" / "vector_axes.py"


def _load_vector_axes_module():
    spec = importlib.util.spec_from_file_location("vector_axes_test_runtime", VECTOR_AXES_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_autotuner_methods(*method_names):
    source = AUTOTUNER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(AUTOTUNER_PATH))
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AutoTilingTuner")
    selected = [
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name in set(method_names)
    ]
    extracted_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(extracted_module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "valid_axis_names": VALID_AXIS_NAMES,
        "VectorAxes": _load_vector_axes_module().VectorAxes,
        "_InternalNPUOptionInt": ascend_autotuner._InternalNPUOptionInt,
        "_DEFAULT_COMPILE_MODE": ascend_autotuner._DEFAULT_COMPILE_MODE,
        "_multibuffer_mode_to_tuple": ascend_autotuner._multibuffer_mode_to_tuple,
        "_normalize_config_hints": ascend_autotuner._normalize_config_hints,
    }
    exec(compile(extracted_module, str(AUTOTUNER_PATH), "exec"), namespace)
    return namespace


def _normalize_loaded_method(method):
    if isinstance(method, staticmethod):
        return method.__func__
    return method


def _init_axis_state_for_test(
    key,
    *,
    hints_axes,
    split_params,
    tiling_params,
    low_dim_axes,
    reduction_axes,
    enable_vv_parser_v2=False,
    vv_parser_v2_mode="off",
):
    namespace = _load_autotuner_methods(
        "_parse_hints_axes",
        "_get_runtime_arg_names_for_hints_axes",
        "_infer_hints_axes_from_key",
        "_normalize_axis_name_mapping",
        "_rebuild_vector_axes",
        "_get_parser_axis_arg_names",
        "_is_direct_runtime_length_arg_name",
        "_init_axis_params",
    )
    tuner = SimpleNamespace()
    tuner.arg_names = ["x_ptr", "n_elements", "BLOCK_SIZE", "BLOCK_SIZE_SUB"]
    tuner._get_constexpr_candidates = lambda: ["BLOCK_SIZE", "BLOCK_SIZE_SUB"]
    parse_hints_axes = _normalize_loaded_method(namespace.get("_parse_hints_axes"))
    if parse_hints_axes is not None:
        tuner._parse_hints_axes = parse_hints_axes.__get__(tuner, SimpleNamespace)
    get_runtime_arg_names_for_hints_axes = _normalize_loaded_method(
        namespace.get("_get_runtime_arg_names_for_hints_axes"))
    if get_runtime_arg_names_for_hints_axes is not None:
        tuner._get_runtime_arg_names_for_hints_axes = get_runtime_arg_names_for_hints_axes.__get__(
            tuner,
            SimpleNamespace,
        )
    rebuild_vector_axes = _normalize_loaded_method(namespace.get("_rebuild_vector_axes"))
    if rebuild_vector_axes is not None:
        tuner._rebuild_vector_axes = rebuild_vector_axes.__get__(tuner, SimpleNamespace)
    normalize_axis_name_mapping = _normalize_loaded_method(namespace.get("_normalize_axis_name_mapping"))
    if normalize_axis_name_mapping is not None:
        tuner._normalize_axis_name_mapping = normalize_axis_name_mapping.__get__(
            tuner,
            SimpleNamespace,
        )
    infer_hints_axes_from_key = _normalize_loaded_method(namespace.get("_infer_hints_axes_from_key"))
    if infer_hints_axes_from_key is not None:
        tuner._infer_hints_axes_from_key = infer_hints_axes_from_key.__get__(tuner, SimpleNamespace)
    get_parser_axis_arg_names = _normalize_loaded_method(namespace.get("_get_parser_axis_arg_names"))
    if get_parser_axis_arg_names is not None:
        tuner._get_parser_axis_arg_names = get_parser_axis_arg_names.__get__(tuner, SimpleNamespace)
    is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace.get("_is_direct_runtime_length_arg_name"))
    if is_direct_runtime_length_arg_name is not None:
        tuner._is_direct_runtime_length_arg_name = is_direct_runtime_length_arg_name
    tuner.enable_vv_parser_v2 = enable_vv_parser_v2
    tuner.vv_parser_v2_mode = vv_parser_v2_mode
    init_axis_params = _normalize_loaded_method(namespace["_init_axis_params"])
    init_signature = init_axis_params.__code__.co_varnames[:init_axis_params.__code__.co_argcount]
    if "hints_axes" in init_signature:
        init_axis_params(
            tuner,
            key,
            split_params,
            tiling_params,
            low_dim_axes,
            reduction_axes,
            hints_axes,
        )
    else:
        init_axis_params(
            tuner,
            key,
            split_params,
            tiling_params,
            low_dim_axes,
            reduction_axes,
        )
    return tuner


def test_init_axis_params_keeps_key_as_argument_name_list():
    tuner = _init_axis_state_for_test(
        ["n_elements"],
        hints_axes={"x": "n_elements"},
        split_params={"x": "BLOCK_SIZE"},
        tiling_params={"x": "BLOCK_SIZE_SUB"},
        low_dim_axes=["x"],
        reduction_axes=[],
    )

    assert tuner.keys == ["n_elements"]
    assert tuner.hints_axes == {"x": "n_elements"}
    assert tuner.axis_arg_names == {"x": "n_elements"}


def test_init_axis_params_falls_back_to_key_order_when_hints_axes_missing():
    tuner = _init_axis_state_for_test(
        ["n_elements"],
        hints_axes=None,
        split_params=None,
        tiling_params=None,
        low_dim_axes=None,
        reduction_axes=None,
    )

    assert tuner.keys == ["n_elements"]
    assert tuner.hints_axes == {"x": "n_elements"}
    assert tuner.vector_axes.axis_length_exprs == {"x": "n_elements"}
    assert tuner.axis_arg_names == {"x": "n_elements"}


def test_init_axis_params_defers_key_fallback_when_vv_assist_without_hints_axes():
    tuner = _init_axis_state_for_test(
        ["n_elements"],
        hints_axes=None,
        split_params=None,
        tiling_params=None,
        low_dim_axes=None,
        reduction_axes=None,
        enable_vv_parser_v2=True,
        vv_parser_v2_mode="assist",
    )

    assert tuner.keys == ["n_elements"]
    assert tuner.hints_axes == {}
    assert tuner.vector_axes.axis_length_exprs == {}
    assert tuner.axis_arg_names == {}


def test_init_axis_params_normalizes_tuple_key():
    tuner = _init_axis_state_for_test(
        ("n_elements", ),
        hints_axes={"x": "n_elements"},
        split_params={"x": "BLOCK_SIZE"},
        tiling_params={"x": "BLOCK_SIZE_SUB"},
        low_dim_axes=["x"],
        reduction_axes=[],
    )

    assert tuner.keys == ["n_elements"]
    assert tuner.axis_arg_names == {"x": "n_elements"}


def test_init_axis_params_rejects_invalid_hints_axes_name():
    with pytest.raises(ValueError, match="All keys in 'hints.axes' must be valid axis names"):
        _init_axis_state_for_test(
            ["n_elements"],
            hints_axes={"x0": "n_elements"},
            split_params={"x": "BLOCK_SIZE"},
            tiling_params={"x": "BLOCK_SIZE_SUB"},
            low_dim_axes=["x"],
            reduction_axes=[],
        )


def test_init_axis_params_rejects_non_string_hints_axes_value():
    with pytest.raises(ValueError, match="All values in 'hints.axes' must be non-empty argument names"):
        _init_axis_state_for_test(
            ["n_elements"],
            hints_axes={"x": 123},
            split_params=None,
            tiling_params=None,
            low_dim_axes=None,
            reduction_axes=None,
        )


def test_init_axis_params_rejects_unknown_hints_axes_arg_name():
    with pytest.raises(ValueError, match="must reference runtime non-constexpr argument names"):
        _init_axis_state_for_test(
            ["n_elements"],
            hints_axes={"x": "unknown_arg"},
            split_params=None,
            tiling_params=None,
            low_dim_axes=None,
            reduction_axes=None,
        )


def test_init_axis_params_rejects_constexpr_hints_axes_arg_name():
    with pytest.raises(ValueError, match="must reference runtime non-constexpr argument names"):
        _init_axis_state_for_test(
            ["n_elements"],
            hints_axes={"x": "BLOCK_SIZE"},
            split_params=None,
            tiling_params=None,
            low_dim_axes=None,
            reduction_axes=None,
        )


def test_init_axis_params_accepts_literal_hints_axes_value():
    tuner = _init_axis_state_for_test(
        ["n_elements"],
        hints_axes={"x": "128"},
        split_params=None,
        tiling_params=None,
        low_dim_axes=None,
        reduction_axes=None,
    )

    assert tuner.hints_axes == {"x": "128"}
    assert tuner.vector_axes.axis_length_exprs == {"x": "128"}


def test_init_axis_params_rejects_dict_key():
    with pytest.raises(ValueError, match="key must be a list"):
        _init_axis_state_for_test(
            {"x": "n_elements"},
            hints_axes={"x": "n_elements"},
            split_params={"x": "BLOCK_SIZE"},
            tiling_params={"x": "BLOCK_SIZE_SUB"},
            low_dim_axes=["x"],
            reduction_axes=[],
        )


def test_init_axis_params_requires_hints_axes_when_axis_metadata_is_present():
    with pytest.raises(ValueError, match="hints.axes must be provided when axis metadata"):
        _init_axis_state_for_test(
            ["n_elements"],
            hints_axes=None,
            split_params={"x": "BLOCK_SIZE"},
            tiling_params={"x": "BLOCK_SIZE_SUB"},
            low_dim_axes=["x"],
            reduction_axes=[],
        )


def test_init_axis_params_rejects_axis_metadata_axes_missing_from_hints_axes():
    with pytest.raises(ValueError, match="missing from 'hints.axes'"):
        _init_axis_state_for_test(
            ["n_elements"],
            hints_axes={"x": "n_elements"},
            split_params={"y": "BLOCK_SIZE"},
            tiling_params={"x": "BLOCK_SIZE_SUB"},
            low_dim_axes=["x"],
            reduction_axes=[],
        )


def test_resolve_axis_length_arg_name_uses_internal_axis_map_not_self_keys():
    namespace = _load_autotuner_methods(
        "_is_direct_runtime_length_arg_name",
        "_get_parser_axis_arg_names",
        "_resolve_axis_length_arg_name",
    )
    vector_axes_module = _load_vector_axes_module()
    tuner = SimpleNamespace(
        vv_adapter_result_v2=None,
        enable_vv_parser_v2=False,
        parser_mode="vector",
        vector_axes=vector_axes_module.VectorAxes.from_hints_axes({"x": "n_elements"}),
        axis_arg_names={"x": "n_elements"},
        keys=["different_cache_key"],
    )
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)

    result = _normalize_loaded_method(namespace["_resolve_axis_length_arg_name"])(tuner, "x")

    assert result == "n_elements"


def test_resolve_axis_length_arg_name_does_not_fallback_to_key_order():
    namespace = _load_autotuner_methods(
        "_is_direct_runtime_length_arg_name",
        "_get_parser_axis_arg_names",
        "_resolve_axis_length_arg_name",
    )
    tuner = SimpleNamespace(
        vv_adapter_result_v2=None,
        enable_vv_parser_v2=False,
        parser_mode="vector",
        vector_axes=None,
        axis_arg_names={},
        keys=["n_elements"],
    )
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)

    result = _normalize_loaded_method(namespace["_resolve_axis_length_arg_name"])(tuner, "x")

    assert result is None


def test_resolve_axis_length_arg_name_uses_hints_axes_when_vv_enabled():
    namespace = _load_autotuner_methods(
        "_is_direct_runtime_length_arg_name",
        "_get_parser_axis_arg_names",
        "_resolve_axis_length_arg_name",
    )
    vector_axes_module = _load_vector_axes_module()
    tuner = SimpleNamespace(
        vv_adapter_result_v2=SimpleNamespace(axis_length_exprs={}),
        enable_vv_parser_v2=True,
        parser_mode="vector",
        vector_axes=vector_axes_module.VectorAxes.from_hints_axes({"x": "n_elements"}),
        axis_arg_names={"x": "n_elements"},
        keys=["different_cache_key"],
    )
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)

    result = _normalize_loaded_method(namespace["_resolve_axis_length_arg_name"])(tuner, "x")

    assert result == "n_elements"


def test_resolve_axis_length_arg_name_uses_base_vv_axis_expr_for_reduction_axis():
    namespace = _load_autotuner_methods(
        "_is_direct_runtime_length_arg_name",
        "_get_parser_axis_arg_names",
        "_resolve_axis_length_arg_name",
    )
    tuner = SimpleNamespace(
        vv_adapter_result_v2=SimpleNamespace(axis_length_exprs={"x": "n_elements"}),
        enable_vv_parser_v2=True,
        parser_mode="vector",
        vector_axes=None,
        axis_arg_names={},
        keys=[],
    )
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)

    result = _normalize_loaded_method(namespace["_resolve_axis_length_arg_name"])(tuner, "x")

    assert result == "n_elements"


def test_is_valid_axis_name_rejects_reduction_prefix():
    assert is_valid_axis_name("x") is True
    assert is_valid_axis_name("rx") is False


def test_kernel_meta_marks_reduction_axes_from_explicit_list():
    kernel_meta = KernelMeta(
        axis_sizes={"x": 128, "y": 64},
        split_params={"x": "XBLOCK"},
        fixed_split_params={},
        tiling_params={"y": "YBLOCK_SUB"},
        low_dims=["y"],
        reduction_axes=["y"],
        dtype=torch.float16,
        persistent_reduction=True,
        dual_reduction=False,
        num_buffers=1,
        is_simt_mode=False,
    )

    axis_by_name = {axis.name: axis for axis in kernel_meta.axis_info}
    assert axis_by_name["x"].is_reduction is False
    assert axis_by_name["y"].is_reduction is True
    assert [axis.name for axis in kernel_meta.low_dims_axis] == ["y"]


def test_kernel_meta_rejects_prefixed_reduction_axes():
    with pytest.raises(ValueError, match="reduction axis"):
        KernelMeta(
            axis_sizes={"x": 128, "y": 64},
            split_params={"x": "XBLOCK"},
            fixed_split_params={},
            tiling_params={"y": "YBLOCK_SUB"},
            low_dims=["y"],
            reduction_axes=["ry"],
            dtype=torch.float16,
            persistent_reduction=True,
            dual_reduction=False,
            num_buffers=1,
            is_simt_mode=False,
        )


def test_apply_vv_axis_semantic_result_promotes_internal_axis_map_only():
    namespace = _load_autotuner_methods(
        "_normalize_reduction_axis_name",
        "_normalize_reduction_axes",
        "_normalize_vv_reduction_axes",
        "_get_axis_base_name",
        "_get_parser_axis_arg_names",
        "_is_direct_runtime_length_arg_name",
        "_promote_axis_arg_name_to_reduction",
        "_apply_vv_axis_semantic_result",
    )
    vector_axes_module = _load_vector_axes_module()
    tuner = SimpleNamespace(
        vv_adapter_result_v2=SimpleNamespace(
            status="ok",
            reduction_axes=["x"],
            low_dim_axes=[],
            split_params={},
            tiling_params={},
            axis_pid_dims={},
        ),
        reduction_axes=[],
        low_dim_axes=[],
        split_params={},
        tiling_params={},
        axis_pid_dims={},
        vector_axes=vector_axes_module.VectorAxes.from_hints_axes({"x": "n_elements"}),
        axis_arg_names={"x": "n_elements"},
        keys=["n_elements"],
        dual_reduction=False,
    )
    tuner._normalize_reduction_axis_name = _normalize_loaded_method(namespace["_normalize_reduction_axis_name"])
    tuner._normalize_reduction_axes = _normalize_loaded_method(namespace["_normalize_reduction_axes"]).__get__(
        tuner, SimpleNamespace)
    tuner._normalize_vv_reduction_axes = _normalize_loaded_method(namespace["_normalize_vv_reduction_axes"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._promote_axis_arg_name_to_reduction = _normalize_loaded_method(
        namespace["_promote_axis_arg_name_to_reduction"]).__get__(
            tuner,
            SimpleNamespace,
        )

    applied = _normalize_loaded_method(namespace["_apply_vv_axis_semantic_result"])(tuner)

    assert applied is True
    assert tuner.keys == ["n_elements"]
    assert tuner.axis_arg_names == {"x": "n_elements"}
    assert tuner.reduction_axes == ["x"]


def test_promote_reduction_axis_rejects_prefixed_axis_name():
    namespace = _load_autotuner_methods(
        "_get_axis_base_name",
        "_get_parser_axis_arg_names",
        "_is_direct_runtime_length_arg_name",
        "_promote_axis_arg_name_to_reduction",
    )
    vector_axes_module = _load_vector_axes_module()
    vector_axes = vector_axes_module.VectorAxes.from_hints_axes({"y": "r1_numel"})
    tuner = SimpleNamespace(
        vector_axes=vector_axes,
        axis_arg_names={"y": "r1_numel"},
    )
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._promote_axis_arg_name_to_reduction = _normalize_loaded_method(
        namespace["_promote_axis_arg_name_to_reduction"]).__get__(tuner, SimpleNamespace)

    with pytest.raises(ValueError, match="r-prefixed"):
        tuner._promote_axis_arg_name_to_reduction("ry")


def test_generate_key_and_configs_uses_axis_arg_names_for_kv_dict():
    namespace = _load_autotuner_methods(
        "_parse_hints_axes",
        "_get_runtime_arg_names_for_hints_axes",
        "_normalize_axis_name_mapping",
        "_rebuild_vector_axes",
        "_get_axis_base_name",
        "_get_parser_axis_arg_names",
        "_is_direct_runtime_length_arg_name",
        "_promote_axis_arg_name_to_reduction",
        "_init_axis_params",
        "generate_key_and_configs",
    )
    captured = {}

    class FakeArg:
        dtype = "float16"

    def fake_get_byte_per_numel(dtype):
        return 0 if dtype is None else 1

    namespace["get_byte_per_numel"] = fake_get_byte_per_numel
    namespace["_expand_configs_with_hints"] = lambda fn, configs, config_hints: configs

    tuner = SimpleNamespace(
        arg_names=["x_ptr", "n_elements"],
        fn=SimpleNamespace(),
        _get_constexpr_candidates=lambda: [],
        cache={},
        auto_gen_config=True,
        parser_mode="vector",
        config_hints={},
        _raw_config_hints={},
        gen_configs=[],
        user_configs=[],
        is_simt_mode=False,
        user_specified_warps=None,
        user_specified_multibuffer=None,
        _autoparse_axis_params=lambda all_args: None,
        _gen_tile_configs=lambda kv_dict, dtype, all_args: captured.update(
            kv_dict=dict(kv_dict),
            dtype=dtype,
            all_args=dict(all_args),
        ) or setattr(tuner, "gen_configs", [SimpleNamespace(kwargs={"BLOCK_SIZE": 128})]),
    )
    tuner._parse_hints_axes = _normalize_loaded_method(namespace["_parse_hints_axes"]).__get__(tuner, SimpleNamespace)
    tuner._get_runtime_arg_names_for_hints_axes = _normalize_loaded_method(
        namespace["_get_runtime_arg_names_for_hints_axes"]).__get__(tuner, SimpleNamespace)
    tuner._rebuild_vector_axes = _normalize_loaded_method(namespace["_rebuild_vector_axes"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._normalize_axis_name_mapping = _normalize_loaded_method(namespace["_normalize_axis_name_mapping"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._promote_axis_arg_name_to_reduction = _normalize_loaded_method(
        namespace["_promote_axis_arg_name_to_reduction"]).__get__(tuner, SimpleNamespace)

    _normalize_loaded_method(namespace["_init_axis_params"])(
        tuner,
        ["n_elements"],
        None,
        None,
        None,
        None,
        {"x": "n_elements"},
    )

    key = _normalize_loaded_method(namespace["generate_key_and_configs"])(
        tuner,
        FakeArg(),
        17,
    )

    assert key == (17, "float16", ("compile_mode", "simd_simt_template"))
    simt_key = _normalize_loaded_method(namespace["generate_key_and_configs"])(
        tuner,
        FakeArg(),
        17,
        compile_mode="simt_only",
    )
    assert simt_key == (17, "float16", ("compile_mode", "simt_only"))
    assert captured["kv_dict"] == {"x": 17}


def test_generate_key_and_configs_preserves_promoted_reduction_axis_identity():
    namespace = _load_autotuner_methods(
        "_parse_hints_axes",
        "_get_runtime_arg_names_for_hints_axes",
        "_normalize_axis_name_mapping",
        "_rebuild_vector_axes",
        "_get_axis_base_name",
        "_get_parser_axis_arg_names",
        "_is_direct_runtime_length_arg_name",
        "_promote_axis_arg_name_to_reduction",
        "_init_axis_params",
        "generate_key_and_configs",
    )
    captured = {}

    class FakeArg:
        dtype = "float16"

    namespace["get_byte_per_numel"] = lambda dtype: 0 if dtype is None else 1
    namespace["_expand_configs_with_hints"] = lambda fn, configs, config_hints: configs

    tuner = SimpleNamespace(
        arg_names=["x_ptr", "n_elements"],
        fn=SimpleNamespace(),
        _get_constexpr_candidates=lambda: [],
        cache={},
        auto_gen_config=True,
        parser_mode="vector",
        config_hints={},
        _raw_config_hints={},
        gen_configs=[],
        user_configs=[],
        is_simt_mode=False,
        user_specified_warps=None,
        user_specified_multibuffer=None,
        _autoparse_axis_params=lambda all_args: None,
        _gen_tile_configs=lambda kv_dict, dtype, all_args: captured.update(kv_dict=dict(kv_dict)) or setattr(
            tuner, "gen_configs", [SimpleNamespace(kwargs={"BLOCK_SIZE": 128})]),
    )
    tuner._parse_hints_axes = _normalize_loaded_method(namespace["_parse_hints_axes"]).__get__(tuner, SimpleNamespace)
    tuner._get_runtime_arg_names_for_hints_axes = _normalize_loaded_method(
        namespace["_get_runtime_arg_names_for_hints_axes"]).__get__(tuner, SimpleNamespace)
    tuner._rebuild_vector_axes = _normalize_loaded_method(namespace["_rebuild_vector_axes"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._normalize_axis_name_mapping = _normalize_loaded_method(namespace["_normalize_axis_name_mapping"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._promote_axis_arg_name_to_reduction = _normalize_loaded_method(
        namespace["_promote_axis_arg_name_to_reduction"]).__get__(tuner, SimpleNamespace)

    _normalize_loaded_method(namespace["_init_axis_params"])(
        tuner,
        ["n_elements"],
        None,
        None,
        None,
        None,
        {"x": "n_elements"},
    )
    tuner._promote_axis_arg_name_to_reduction("x")

    _normalize_loaded_method(namespace["generate_key_and_configs"])(
        tuner,
        FakeArg(),
        23,
    )

    assert tuner.keys == ["n_elements"]
    assert captured["kv_dict"] == {"x": 23}


def test_refresh_vector_axes_keeps_base_axis_arg_names_without_reduction_aliases():
    namespace = _load_autotuner_methods(
        "_parse_hints_axes",
        "_get_runtime_arg_names_for_hints_axes",
        "_normalize_axis_name_mapping",
        "_rebuild_vector_axes",
        "_get_axis_base_name",
        "_get_parser_axis_arg_names",
        "_is_direct_runtime_length_arg_name",
        "_promote_axis_arg_name_to_reduction",
        "_refresh_vector_axes",
        "_init_axis_params",
        "generate_key_and_configs",
    )
    captured = {}

    class FakeArg:
        dtype = "float16"

    namespace["get_byte_per_numel"] = lambda dtype: 0 if dtype is None else 1
    namespace["_expand_configs_with_hints"] = lambda fn, configs, config_hints: configs

    tuner = SimpleNamespace(
        arg_names=["x_ptr", "n_elements"],
        fn=SimpleNamespace(),
        _get_constexpr_candidates=lambda: [],
        cache={},
        auto_gen_config=True,
        parser_mode="vector",
        config_hints={},
        _raw_config_hints={},
        gen_configs=[],
        user_configs=[],
        is_simt_mode=False,
        user_specified_warps=None,
        user_specified_multibuffer=None,
        _autoparse_axis_params=lambda all_args: None,
        _gen_tile_configs=lambda kv_dict, dtype, all_args: captured.update(kv_dict=dict(kv_dict)) or setattr(
            tuner, "gen_configs", [SimpleNamespace(kwargs={"BLOCK_SIZE": 128})]),
    )
    tuner._parse_hints_axes = _normalize_loaded_method(namespace["_parse_hints_axes"]).__get__(tuner, SimpleNamespace)
    tuner._get_runtime_arg_names_for_hints_axes = _normalize_loaded_method(
        namespace["_get_runtime_arg_names_for_hints_axes"]).__get__(tuner, SimpleNamespace)
    tuner._rebuild_vector_axes = _normalize_loaded_method(namespace["_rebuild_vector_axes"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._normalize_axis_name_mapping = _normalize_loaded_method(namespace["_normalize_axis_name_mapping"]).__get__(
        tuner, SimpleNamespace)
    tuner._get_parser_axis_arg_names = _normalize_loaded_method(namespace["_get_parser_axis_arg_names"]).__get__(
        tuner, SimpleNamespace)
    tuner._is_direct_runtime_length_arg_name = _normalize_loaded_method(namespace["_is_direct_runtime_length_arg_name"])
    tuner._promote_axis_arg_name_to_reduction = _normalize_loaded_method(
        namespace["_promote_axis_arg_name_to_reduction"]).__get__(tuner, SimpleNamespace)
    tuner._refresh_vector_axes = _normalize_loaded_method(namespace["_refresh_vector_axes"]).__get__(
        tuner, SimpleNamespace)

    _normalize_loaded_method(namespace["_init_axis_params"])(
        tuner,
        ["n_elements"],
        None,
        None,
        None,
        None,
        {"x": "n_elements"},
    )

    tuner.reduction_axes = []
    tuner._refresh_vector_axes()

    _normalize_loaded_method(namespace["generate_key_and_configs"])(
        tuner,
        FakeArg(),
        29,
    )

    assert tuner.axis_arg_names == {"x": "n_elements"}
    assert captured["kv_dict"] == {"x": 29}


def test_legacy_low_dim_and_tiling_parsers_consume_base_axis_names():
    namespace = _load_autotuner_methods(
        "_get_axis_base_name",
        "_normalize_axis_name_mapping",
        "_autoparse_tiling_params",
        "_autoparse_low_dim_axes",
    )
    parser_inputs = {}

    class StubTilingAxesParser:

        def __init__(self, func_ast, axis_arg_names, candidates_params):
            parser_inputs["tiling"] = dict(axis_arg_names)

        def parse(self):
            return {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"}

    class StubLowDimsAxesParser:

        def __init__(self, func_ast, axis_arg_names):
            parser_inputs["low_dim"] = dict(axis_arg_names)

        def parse(self):
            return ["y"]

    namespace["TilingAxesParser"] = StubTilingAxesParser
    namespace["LowDimsAxesParser"] = StubLowDimsAxesParser

    refresh_calls = []
    tuner = SimpleNamespace(
        fn=SimpleNamespace(parse=lambda: "fake-func-ast"),
        print_autotuning=False,
        tiling_params={},
        low_dim_axes=[],
        _get_parser_axis_arg_names=lambda: {"x": "x0_numel", "y": "r1_numel"},
        _refresh_vector_axes=lambda: refresh_calls.append(True),
    )
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._normalize_axis_name_mapping = _normalize_loaded_method(namespace["_normalize_axis_name_mapping"]).__get__(
        tuner, SimpleNamespace)

    tiling_params = _normalize_loaded_method(namespace["_autoparse_tiling_params"])(
        tuner,
        ["X0BLOCK_SUB", "R1BLOCK_SUB"],
    )
    low_dim_axes = _normalize_loaded_method(namespace["_autoparse_low_dim_axes"])(tuner)

    assert parser_inputs["tiling"] == {"x": "x0_numel", "y": "r1_numel"}
    assert parser_inputs["low_dim"] == {"x": "x0_numel", "y": "r1_numel"}
    assert tiling_params == {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"}
    assert tuner.tiling_params == {"x": "X0BLOCK_SUB", "y": "R1BLOCK_SUB"}
    assert low_dim_axes == ["y"]
    assert tuner.low_dim_axes == ["y"]
    assert refresh_calls == [True, True]


def test_parse_vv_axis_info_v2_collects_fixed_tiling_expr_for_provided_constexpr():
    from triton.backends.ascend.runtime.dsl_analysis.vv_param_parser_v2 import (
        parse_vv_axis_info_v2, )

    func_ast = ast.parse("""
def silu_like_kernel(
    x,
    y,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]
            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            y_ptrs = y + rows_off[:, None] * stride_row + cols_off[None, :]
            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)
            tl.store(y_ptrs, x_chunk, mask=block_mask)
""").body[0]

    result = parse_vv_axis_info_v2(
        func_ast,
        provided_args={
            "n_rows": 4096,
            "n_cols": 8192,
            "BLOCK_SIZE_N": 2048,
        },
        hints={"vv_parser_v2_mode": "assist"},
    )

    assert result.axis_length_exprs == {"x": "n_rows", "y": "n_cols"}
    assert result.fixed_tiling_exprs == {"y": "BLOCK_SIZE_N"}


def test_parse_vv_axis_info_v2_prefers_extent_resolved_site_for_binned_copy_pattern():
    from triton.backends.ascend.runtime.dsl_analysis.vv_param_parser_v2 import (
        parse_vv_axis_info_v2, )

    func_ast = ast.parse("""
def binned_copy_like_kernel(
    x,
    grad,
    wgrad,
    num_experts,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)
""").body[0]

    result = parse_vv_axis_info_v2(
        func_ast,
        provided_args={"NUM_COLUMNS": 1536, "TOP_K": 4},
        hints={"vv_parser_v2_mode": "assist"},
    )

    assert result.axis_length_exprs == {"x": "NUM_COLUMNS"}
    assert result.tiling_params == {"x": "BLOCK_X"}
    assert result.low_dim_axes == ["x"]


def test_vector_axes_materialize_axis_sizes_prefers_fixed_tiling_expr_for_non_split_axis():
    vector_axes_module = _load_vector_axes_module()
    vector_axes = vector_axes_module.VectorAxes.from_hints_axes({"x": "n_rows", "y": "n_cols"})
    vector_axes.apply_semantic_fields(
        split_params={"x": "BLOCK_SIZE_M"},
        low_dim_axes=["y"],
        fixed_tiling_exprs={"y": "BLOCK_SIZE_N"},
    )

    axis_sizes, diagnostics = vector_axes.materialize_axis_sizes({
        "n_rows": 4096,
        "n_cols": 8192,
        "BLOCK_SIZE_N": 2048,
    })

    assert axis_sizes == {"x": 4096, "y": 2048}
    assert diagnostics["y"]["resolved_by"] == "fixed_tiling_expr_arg"


def test_parse_cv_params_preserves_block_size_n_for_inline_dot_rhs_expression():
    from triton.backends.ascend.runtime.dsl_analysis.cv_param_parser import (
        parse_cv_params, )

    func_ast = ast.parse("""
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    phy_grids,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 4,
):
    offs_am = (0 + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (0 + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for i in range(4):
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K)):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
            mask = 3 << (2 * i)
            b = (b_uint8 & mask) >> (2 * i)
            tensor_full = tl.full((1,), 1, dtype=tl.int8)
            accumulator += tl.dot(a, b.to(tl.int8) - tensor_full, out_dtype=tl.int32)
""").body[0]

    parse_result = parse_cv_params(
        func_ast,
        parser_mode="mix",
        arg_names=[
            "a_ptr",
            "b_ptr",
            "c_ptr",
            "M",
            "N",
            "K",
            "stride_am",
            "stride_ak",
            "stride_bk",
            "stride_bn",
            "stride_cm",
            "stride_cn",
            "phy_grids",
        ],
        provided_args={
            "M": 2,
            "N": 3,
            "K": 128,
            "stride_am": 128,
            "stride_ak": 1,
            "stride_bk": 3,
            "stride_bn": 1,
            "stride_cm": 3,
            "stride_cn": 1,
            "phy_grids": 1,
        },
        explicit_tunable_params=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
    )

    tunable_names = {item.name for item in parse_result.tunable_params}

    assert "BLOCK_SIZE_M" in tunable_names
    assert "BLOCK_SIZE_N" in tunable_names
    assert "BLOCK_SIZE_K" in tunable_names
    assert parse_result.dot_sites
    assert parse_result.dot_sites[0].n.tunable_param == "BLOCK_SIZE_N"


def test_parse_cv_params_resolves_multi_level_k_alias_for_dot_operands():
    from triton.backends.ascend.runtime.dsl_analysis.cv_param_parser import (
        parse_cv_params, )

    func_ast = ast.parse("""
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        current_k = k + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=current_k[None, :] < K,
            other=0.0,
        )
        b = tl.load(
            b_ptr + current_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=current_k[:, None] < K,
            other=0.0,
        )
        acc += tl.dot(a, b)
""").body[0]

    arg_names = [arg.arg for arg in func_ast.args.args]
    parse_result = parse_cv_params(
        func_ast,
        parser_mode="mix",
        arg_names=arg_names,
        provided_args={},
    )

    assert parse_result.dot_sites
    assert parse_result.dot_sites[0].k.axis_symbol == "current_k"
    assert parse_result.dot_sites[0].k.tunable_param == "BLOCK_K"
    assert parse_result.dot_sites[0].k.length_expr == "K"
    assert parse_result.dot_sites[0].k.state == "tunable"


@triton.autotune(
    configs=[], key=["n_elements"], hints={
        "axes": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SIZE_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    })
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE
    loops1 = (BLOCK_SIZE + BLOCK_SIZE_SUB - 1) // BLOCK_SIZE_SUB
    for loop in range(0, loops1):
        x0 = offset + loop * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_SUB)
        mask = x0 < n_elements
        x = tl.load(x_ptr + x0, mask)
        y = tl.load(y_ptr + x0, mask)
        output = x + y
        tl.store(output_ptr + x0, output)


def add_torch(x, y):
    return x + y


def add_autotune(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )](x, y, output, n_elements)
    return output


@pytest.mark.autotune
@pytest.mark.parametrize('size', [
    2048,
])
def test_add(size: int):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output_torch = add_torch(x, y)
    output_triton = add_autotune(x, y)
    assert torch.allclose(output_triton, output_torch)


@pytest.mark.autotune
def test_add_no_reduction_axes():
    with pytest.raises(ValueError, match="reduction_axes must be a list"):

        @triton.autotune(
            configs=[], key=["n_elements"], hints={
                "axes": {"x": "n_elements"},
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
            })
        @triton.jit
        def add_kernel_exception(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            BLOCK_SIZE_SUB: tl.constexpr,
        ):
            pass


@pytest.mark.autotune
def test_add_no_low_dim_axes():
    with pytest.raises(ValueError, match="low_dim_axes must be a list"):

        @triton.autotune(
            configs=[], key=["n_elements"], hints={
                "axes": {"x": "n_elements"},
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            BLOCK_SIZE_SUB: tl.constexpr,
        ):
            pass


@pytest.mark.autotune
def test_add_no_tiling_params():
    with pytest.raises(ValueError, match="tiling_params must be a dict"):

        @triton.autotune(
            configs=[], key=["n_elements"], hints={
                "axes": {"x": "n_elements"},
                "split_params": {"x": "BLOCK_SIZE"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            BLOCK_SIZE_SUB: tl.constexpr,
        ):
            pass


@pytest.mark.autotune
def test_add_no_split_params():
    with pytest.raises(ValueError, match="split_params must be a dict"):

        @triton.autotune(
            configs=[], key=["n_elements"], hints={
                "axes": {"x": "n_elements"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            BLOCK_SIZE_SUB: tl.constexpr,
        ):
            pass


@pytest.mark.autotune
def test_add_invalid_hints_axes_name():
    with pytest.raises(ValueError, match="All keys in 'hints.axes' must be valid axis names"):

        @triton.autotune(
            configs=[], key=["n_elements"], hints={
                "axes": {"x0": "n_elements"},
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception():
            pass


def _materialize_expanded_configs(kernel):
    x = torch.zeros(1, dtype=torch.float32)
    output = torch.empty_like(x)
    kernel.cache = {}
    kernel.generate_key_and_configs(x, output, x.numel())
    return kernel.configs


def test_expand_explicit_configs_with_hints():
    base_configs = [
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
    ]

    @triton.autotune(configs=base_configs, key=["n_elements"], hints={
        "GROUP_M": [2, 4, 8],
    })
    @triton.jit
    def kernel_with_group_hint(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_with_group_hint.configs) == 2
    expanded_configs = _materialize_expanded_configs(kernel_with_group_hint)
    assert len(expanded_configs) == 12
    actual_triplets = {(cfg.kwargs["BLOCK_SIZE"], cfg.kwargs["GROUP_M"], cfg.num_stages) for cfg in expanded_configs}
    expected_triplets = {
        (128, 2, 1),
        (128, 2, 2),
        (128, 4, 1),
        (128, 4, 2),
        (128, 8, 1),
        (128, 8, 2),
        (256, 2, 1),
        (256, 2, 2),
        (256, 4, 1),
        (256, 4, 2),
        (256, 8, 1),
        (256, 8, 2),
    }
    assert actual_triplets == expected_triplets


def test_expand_hints_multibuffer_maps_to_num_stages():

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
        "GROUP_M": [2],
        "multibuffer": [True, False],
    })
    @triton.jit
    def kernel_with_multibuffer_alias(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    expanded_configs = _materialize_expanded_configs(kernel_with_multibuffer_alias)
    assert len(expanded_configs) == 2
    assert {cfg.num_stages for cfg in expanded_configs} == {1, 2}
    assert all("multibuffer" not in cfg.kwargs for cfg in expanded_configs)
    assert kernel_with_multibuffer_alias.config_hints == {
        "GROUP_M": [2],
        "num_stages": [2, 1],
    }


def test_expand_hints_explicit_num_stages_precedes_multibuffer():

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
        "GROUP_M": [2, 4],
        "num_stages": [1],
        "multibuffer": [True, False],
    })
    @triton.jit
    def kernel_num_stages_precedence(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    expanded_configs = _materialize_expanded_configs(kernel_num_stages_precedence)
    assert len(expanded_configs) == 2
    assert {cfg.num_stages for cfg in expanded_configs} == {1}
    assert kernel_num_stages_precedence.config_hints == {
        "GROUP_M": [2, 4],
        "num_stages": [1],
    }


def test_expand_explicit_configs_with_mixed_hints():
    base_configs = [
        triton.Config({"BLOCK_SIZE": 128}),
    ]

    @triton.autotune(configs=base_configs, key=["n_elements"], hints={
        "GROUP_M": [2, 4],
        "num_stages": [1, 2],
        "enable_ubuf_saving": [True, False],
    })
    @triton.jit
    def kernel_with_mixed_hints(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    expanded_configs = _materialize_expanded_configs(kernel_with_mixed_hints)
    assert len(expanded_configs) == 8
    assert {cfg.num_stages for cfg in expanded_configs} == {1, 2}
    assert {cfg.kwargs["GROUP_M"] for cfg in expanded_configs} == {2, 4}
    assert {cfg.kwargs["enable_ubuf_saving"] for cfg in expanded_configs} == {True, False}


def test_expand_hints_coexist_with_axis_hints():
    base_configs = [
        triton.Config({"BLOCK_SIZE": 128, "BLOCK_SIZE_SUB": 32}),
        triton.Config({"BLOCK_SIZE": 256, "BLOCK_SIZE_SUB": 64}),
    ]

    @triton.autotune(
        configs=base_configs, key=["n_elements"], hints={
            "axes": {"x": "n_elements"},
            "split_params": {"x": "BLOCK_SIZE"},
            "tiling_params": {"x": "BLOCK_SIZE_SUB"},
            "low_dim_axes": ["x"],
            "reduction_axes": [],
            "GROUP_M": [2, 4],
        })
    @triton.jit
    def kernel_with_axis_hints(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_SUB: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    expanded_configs = _materialize_expanded_configs(kernel_with_axis_hints)
    assert len(expanded_configs) == 8
    assert {cfg.num_stages for cfg in expanded_configs} == {1, 2}
    assert kernel_with_axis_hints.hints == {
        "axes": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SIZE_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    assert kernel_with_axis_hints.config_hints == {
        "GROUP_M": [2, 4],
        "num_stages": [1, 2],
    }


def test_expand_hints_defer_explicit_configs_until_runtime():

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
        "GROUP_M": [2, 4],
    })
    @triton.jit
    def kernel_defer_explicit_configs(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_defer_explicit_configs.configs) == 1
    assert kernel_defer_explicit_configs.config_hints == {
        "GROUP_M": [2, 4],
        "num_stages": [1, 2],
    }

    expanded_configs = _materialize_expanded_configs(kernel_defer_explicit_configs)
    assert len(expanded_configs) == 4
    assert {(cfg.kwargs["GROUP_M"], cfg.num_stages)
            for cfg in expanded_configs} == {
                (2, 1),
                (2, 2),
                (4, 1),
                (4, 2),
            }


def test_expand_hints_invalid_key():
    with pytest.raises(ValueError, match="Unsupported hints keys"):

        @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
            "INVALID_KEY": [1, 2],
        })
        @triton.jit
        def kernel_invalid_hint_key(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pass


def test_expand_hints_invalid_value_container():

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
        "GROUP_M": 4,
    })
    @triton.jit
    def kernel_invalid_hint_value(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pass

    with pytest.raises(ValueError, match="must be a non-empty list/tuple"):
        _materialize_expanded_configs(kernel_invalid_hint_value)


def test_expand_hints_invalid_multibuffer_values():
    with pytest.raises(ValueError, match="must contain only boolean values"):

        @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
            "multibuffer": [1, 0],
        })
        @triton.jit
        def kernel_invalid_multibuffer_hint(
            x_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pass


def test_expand_hints_invalid_compile_option_value():

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 128})], key=["n_elements"], hints={
        "num_stages": [3],
    })
    @triton.jit
    def kernel_invalid_compile_hint(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pass

    with pytest.raises(ValueError, match="Invalid value for 'num_stages'"):
        _materialize_expanded_configs(kernel_invalid_compile_hint)


def test_expand_hints_defer_when_only_auto_generated_configs_exist():

    @triton.autotune(configs=[], key=["n_elements"], hints={
        "GROUP_M": [2, 4],
    })
    @triton.jit
    def kernel_require_configs(
        x_ptr,
        output_ptr,
        n_elements,
        GROUP_M: tl.constexpr,
    ):
        pass

    assert len(kernel_require_configs.configs) == 1
    assert all("GROUP_M" not in cfg.kwargs for cfg in kernel_require_configs.configs)
    assert kernel_require_configs.config_hints == {
        "GROUP_M": [2, 4],
    }


def test_non_simt_num_stages_candidates_priority():
    tuner = object.__new__(ascend_autotuner.AutoTilingTuner)
    tuner.user_specified_num_stages = None
    tuner.user_specified_multibuffer = None

    assert tuner._get_non_simt_num_stages_candidates() == [1, 2]

    tuner.user_specified_multibuffer = True
    assert tuner._get_non_simt_num_stages_candidates() == [2]

    tuner.user_specified_num_stages = 1
    assert tuner._get_non_simt_num_stages_candidates() == [1]


def test_expand_simt_num_warps_configs_default_candidates():
    tuner = object.__new__(ascend_autotuner.AutoTilingTuner)
    tuner.user_specified_warps = None
    tuner.print_autotuning = False

    expanded_configs = tuner._expand_simt_num_warps_configs([triton.Config({"BLOCK_SIZE": 128})])

    assert len(expanded_configs) == 4
    assert [cfg.num_warps for cfg in expanded_configs] == [8, 16, 32, 64]


@pytest.mark.parametrize(
    "axis",
    [
        "x",
        "y",
        "z",
    ],
)
def test_normalize_reduction_axis_name_accepts_base_axes_only(axis):
    namespace = _load_autotuner_methods("_normalize_reduction_axis_name")

    result = _normalize_loaded_method(namespace["_normalize_reduction_axis_name"])(axis)

    assert result == axis


@pytest.mark.parametrize("axis", ["rx", "ry", "rry"])
def test_normalize_reduction_axis_name_rejects_prefixed_axes(axis):
    namespace = _load_autotuner_methods("_normalize_reduction_axis_name")

    with pytest.raises(ValueError, match="r-prefixed"):
        _normalize_loaded_method(namespace["_normalize_reduction_axis_name"])(axis)


def test_autoparse_reduction_axes_rejects_prefixed_parser_output():
    namespace = _load_autotuner_methods(
        "_normalize_reduction_axis_name",
        "_normalize_reduction_axes",
        "_get_axis_base_name",
        "_autoparse_reduction_axes",
    )

    class StubReductionAxesParser:

        def __init__(self, func_ast, axis_arg_names):
            self.func_ast = func_ast
            self.axis_arg_names = axis_arg_names

        def parse(self):
            return ["ry"]

    namespace["ReductionAxesParser"] = StubReductionAxesParser

    promoted_axes = []
    refresh_calls = []

    tuner = SimpleNamespace(
        fn=SimpleNamespace(parse=lambda: "fake-func-ast"),
        axis_arg_names={"x": "seq_len", "y": "dim"},
        reduction_axes=[],
        print_autotuning=False,
        _get_parser_axis_arg_names=lambda: {"x": "seq_len", "y": "dim"},
        _promote_axis_arg_name_to_reduction=lambda axis: promoted_axes.append(axis),
        _refresh_vector_axes=lambda: refresh_calls.append(True),
    )
    tuner._get_axis_base_name = _normalize_loaded_method(namespace["_get_axis_base_name"])
    tuner._normalize_reduction_axis_name = _normalize_loaded_method(namespace["_normalize_reduction_axis_name"])
    tuner._normalize_reduction_axes = _normalize_loaded_method(namespace["_normalize_reduction_axes"]).__get__(
        tuner, SimpleNamespace)

    with pytest.raises(ValueError, match="r-prefixed"):
        _normalize_loaded_method(namespace["_autoparse_reduction_axes"])(tuner)

    assert promoted_axes == []
    assert tuner.reduction_axes == []
    assert refresh_calls == []


def test_inject_grid_num_tiles_uses_only_static_grid_and_preserves_internal_value():
    namespace = _load_autotuner_methods("_inject_grid_num_tiles")
    inject_grid_num_tiles = _normalize_loaded_method(namespace["_inject_grid_num_tiles"])

    static_grid = {"grid": (2, 16)}
    inject_grid_num_tiles(static_grid)
    assert static_grid["grid_num_tiles"] == 16
    assert isinstance(static_grid["grid_num_tiles"], ascend_autotuner._InternalNPUOptionInt)

    # Only the first three launch dimensions are visible to the compiler; use
    # the outermost one among those, matching the ChunkCoalescing contract.
    four_dim_grid = {"grid": [2, 3, 32, 64]}
    inject_grid_num_tiles(four_dim_grid)
    assert four_dim_grid["grid_num_tiles"] == 32

    callable_grid = {"grid": lambda _: (2, 16)}
    inject_grid_num_tiles(callable_grid)
    assert "grid_num_tiles" not in callable_grid

    internal_hint = {
        "grid": (2, 16),
        "grid_num_tiles": ascend_autotuner._InternalNPUOptionInt(99),
    }
    inject_grid_num_tiles(internal_hint)
    assert internal_hint["grid_num_tiles"] == 99


def test_make_kernel_call_extracts_name_from_jit_run():
    namespace = _load_autotuner_methods("_make_kernel_call")
    _make_kernel_call = _normalize_loaded_method(namespace["_make_kernel_call"])

    @triton.jit
    def test_kernel_jit(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(x_ptr + offsets, x, mask=mask)

    fake_self = SimpleNamespace(fn=test_kernel_jit, pre_hook=lambda full_nargs: None,
                                post_hook=lambda full_nargs, exception=None: None, nargs={})
    fake_config = SimpleNamespace(kwargs={"BLOCK_SIZE": 32}, all_kwargs=lambda: {"BLOCK_SIZE": 32}, pre_hook=None)

    x = torch.zeros(128, dtype=torch.float32, device="npu")
    kernel_call_closure = _make_kernel_call(fake_self, x, x.numel(), config=fake_config, grid=(1, ))
    kernel_call_closure(warmup=False)

    assert kernel_call_closure.target_kernel_name == "test_kernel_jit"


def test_make_kernel_call_extracts_name_from_libentry_tuple():
    namespace = _load_autotuner_methods("_make_kernel_call")
    _make_kernel_call = _normalize_loaded_method(namespace["_make_kernel_call"])

    from triton.runtime.libentry import libentry

    @libentry()
    @triton.jit
    def test_kernel_libentry(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(x_ptr + offsets, x, mask=mask)

    fake_self = SimpleNamespace(fn=test_kernel_libentry, pre_hook=lambda full_nargs: None,
                                post_hook=lambda full_nargs, exception=None: None, nargs={})
    fake_config = SimpleNamespace(kwargs={"BLOCK_SIZE": 32}, all_kwargs=lambda: {"BLOCK_SIZE": 32}, pre_hook=None)

    x = torch.zeros(128, dtype=torch.float32, device="npu")
    kernel_call_closure = _make_kernel_call(fake_self, x, x.numel(), config=fake_config, grid=(1, ))
    kernel_call_closure(warmup=False)

    assert kernel_call_closure.target_kernel_name == "test_kernel_libentry"
