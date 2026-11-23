"""Partial Core ATen must-alias analysis at the model compiler boundary.

只读分析：视图共享缓冲，不代表不同组之间一定不共享。这里不推测分配
容量、byte offset 或设备完成时间，也不授权释放、复用或原地写。
The allowlist supplies *must-view* semantics; schema alias labels alone only
describe possible aliasing and must never authorize arbitrary merges.
"""

from __future__ import annotations

import hashlib
import json
import re

from .lifetime_intervals import union_lifetime_intervals
from .model_ir_adapter import CompilerIRAnalysisError, adapt_model_ir, manifest_identity


VIEW_RULES = {
    "aten.view.default": ("aten::view", ""),
    "aten.slice.Tensor": ("aten::slice", "Tensor"),
    "aten.expand.default": ("aten::expand", ""),
    "aten.permute.default": ("aten::permute", ""),
}
FACT_VERSION = "f08-core-aten-schema-facts/0.1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def collect_operator_schemas(manifest):
    """Read installed dispatcher declarations, without executing any operator.

    Torch is imported only on this explicit enabled-analysis path. No tensor,
    model, PT2 load, CUDA call, storage pointer or device probe is involved.
    Serialized declarations are replayable compiler inputs, not runtime proof.
    """
    snapshot, identity = manifest_identity(manifest)
    import torch

    def argument(item):
        info = item.alias_info
        return {
            "name": item.name, "type": str(item.type),
            "alias": None if info is None else {
                "before": sorted(info.before_set), "after": sorted(info.after_set),
                "is_write": info.is_write,
            },
        }

    records = {}
    targets = sorted({n["target"] for g in snapshot["graphs"] for n in g["nodes"]
                      if n["op"] == "call_function" and isinstance(n.get("target"), str)})
    for target in targets:
        # 不 eval 图里的 target；仅访问明确的 aten namespace/overload。
        if not re.fullmatch(r"aten\.[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*", target):
            records[target] = {"status": "unavailable", "reason": "not_aten_overload"}
            continue
        _, name, overload = target.split(".")
        try:
            schema = getattr(getattr(torch.ops.aten, name), overload)._schema
        except (AttributeError, RuntimeError) as error:
            records[target] = {"status": "unavailable", "reason": type(error).__name__}
            continue
        record = {
            "status": "available", "name": schema.name,
            "overload": schema.overload_name, "text": str(schema),
            "arguments": [argument(x) for x in schema.arguments],
            "returns": [argument(x) for x in schema.returns],
            "is_mutable": schema.is_mutable,
        }
        record["record_sha256"] = _digest(record)
        records[target] = record
    return {
        "schema_version": FACT_VERSION, "manifest_sha256": identity,
        "source": "installed_pytorch_dispatcher_schema", "torch_version": str(torch.__version__),
        "targets": records, "device_execution_observed": False,
    }


def _validate_facts(facts, identity):
    if not isinstance(facts, dict) or facts.get("schema_version") != FACT_VERSION:
        raise CompilerIRAnalysisError("invalid_operator_schema_facts", "schema version required")
    if facts.get("manifest_sha256") != identity:
        raise CompilerIRAnalysisError("operator_schema_manifest_mismatch", "facts belong to another IR")
    if (facts.get("source") != "installed_pytorch_dispatcher_schema"
            or not isinstance(facts.get("torch_version"), str)
            or not isinstance(facts.get("targets"), dict)):
        raise CompilerIRAnalysisError("invalid_operator_schema_facts", "source/version/targets required")
    for target, record in facts["targets"].items():
        if not isinstance(record, dict) or record.get("status") not in {"available", "unavailable"}:
            raise CompilerIRAnalysisError("invalid_operator_schema_record", str(target))
        if record["status"] == "available":
            content = {k: v for k, v in record.items() if k != "record_sha256"}
            if record.get("record_sha256") != _digest(content):
                raise CompilerIRAnalysisError("operator_schema_record_mismatch", str(target))
            if (not isinstance(record.get("arguments"), list) or not isinstance(record.get("returns"), list)
                    or not isinstance(record.get("is_mutable"), bool)):
                raise CompilerIRAnalysisError("invalid_operator_schema_record", str(target))
            for item in record["arguments"] + record["returns"]:
                if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                        or not isinstance(item.get("type"), str)):
                    raise CompilerIRAnalysisError("invalid_operator_schema_record", str(target))
                info = item.get("alias")
                if info is not None and (
                    not isinstance(info, dict) or not isinstance(info.get("is_write"), bool)
                    or any(not isinstance(info.get(key), list)
                           or not all(isinstance(label, str) for label in info[key])
                           for key in ("before", "after"))
                ):
                    raise CompilerIRAnalysisError("invalid_operator_schema_record", str(target))


def _view_schema_matches(target, record):
    if record.get("status") != "available":
        return False
    if (record.get("name"), record.get("overload")) != VIEW_RULES[target]:
        return False
    args, returns = record["arguments"], record["returns"]
    if (record["is_mutable"] or not args or len(returns) != 1
            or args[0].get("name") != "self"
            or args[0].get("type") != "Tensor" or returns[0].get("type") != "Tensor"):
        return False
    # 仅 self 与单个 Tensor 返回值拥有同一、只读、非通配 alias 集。
    left, right = args[0].get("alias"), returns[0].get("alias")
    if not isinstance(left, dict) or not isinstance(right, dict) or left != right:
        return False
    labels = left.get("before")
    return (left.get("is_write") is False and isinstance(labels, list) and len(labels) == 1
            and isinstance(labels[0], str) and labels[0] != "*" and bool(labels[0])
            and labels == left.get("after") and all(x.get("alias") is None for x in args[1:]))


def analyze_compiler_aliases(manifest, operator_schemas):
    """Union proven view/projection relations and their complete IR intervals.

    最后一个 view 的消费者决定该组的最晚引用；单个原 Tensor 已经用完
    不能缩短整个组的生命期。沿用 G1 的闭区间合并函数，事件坐标仍是
    模型 IR 序号，不是最终 kernel 完成顺序。
    """
    snapshot, identity = manifest_identity(manifest)
    ir = adapt_model_ir(snapshot)
    _validate_facts(operator_schemas, identity)
    schema_records = operator_schemas["targets"]
    reports = []
    for graph, ir_graph in zip(snapshot["graphs"], ir["graphs"]):
        values = {v["source_tensor_id"]: v for v in ir_graph["values"]}
        by_node = {}
        for value in values.values():
            by_node.setdefault(value["node"], []).append(value)
        parents = {key: key for key in values}

        def find(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        def join(source, target):
            parents[find(target)] = find(source)

        edges, diagnostics, writes = [], [], []
        for edge in ir_graph["projection_aliases"]:
            join(edge["source"], edge["target"])
            edges.append({**edge, "rule": "validated_tuple_projection", "target_operator": "getitem"})
        for node in graph["nodes"]:
            target = node.get("target")
            record = schema_records.get(target, {})
            if node["op"] == "call_function" and record.get("is_mutable") is True:
                writes.append(node["name"])
            if node["op"] != "call_function" or target not in VIEW_RULES:
                continue
            reason = None
            if not _view_schema_matches(target, record):
                reason = "view_schema_missing_or_unproven"
            args = node["args"]
            source_ref = args[0] if args else node["kwargs"].get("self")
            source_values = (by_node.get(source_ref["node"], [])
                             if isinstance(source_ref, dict) and set(source_ref) == {"node"} else [])
            output_values = by_node.get(node["name"], [])
            if not reason and (len(source_values) != 1 or len(output_values) != 1):
                reason = "view_requires_single_tensor_input_and_output"
            if not reason:
                source, output = source_values[0], output_values[0]
                for key in ("dtype", "device"):
                    if not source["metadata"].get(key) or source["metadata"].get(key) != output["metadata"].get(key):
                        reason = "view_metadata_mismatch"
                # Sparse/nested layouts have separate storage semantics; don't generalize.
                if any(v["metadata"].get("layout") not in {"contiguous", "strided", "symbolic_strided", "torch.strided"}
                       for v in (source, output)):
                    reason = "view_layout_not_supported"
            if reason:
                diagnostics.append({"code": reason, "node": node["name"], "target": target})
                continue
            join(source["source_tensor_id"], output["source_tensor_id"])
            edges.append({
                "source": source["source_tensor_id"], "target": output["source_tensor_id"],
                "rule": "core_aten_must_view_allowlist", "target_operator": target,
                "schema_sha256": record["record_sha256"],
            })
        members = {}
        for key in values:
            members.setdefault(find(key), []).append(values[key])
        groups = []
        for group in members.values():
            start, end = union_lifetime_intervals(
                (v["definition_index"], v["retained_through_index"]) for v in group)
            groups.append({
                "alias_group_id": min(v["source_tensor_id"] for v in group),
                "members": [v["source_tensor_id"] for v in group],
                "lifetime_start": start, "lifetime_end": end,
                "protected_reasons": sorted({r for v in group for r in v["protected_reasons"]}),
                "is_graph_output": any(v["is_graph_output"] for v in group),
                "physical_storage_id": None, "capacity_bytes": None,
                "safe_to_release": False,
            })
        reports.append({
            "graph": graph["name"], "alias_groups": groups, "must_alias_edges": edges,
            "diagnostics": diagnostics, "schema_observed_mutation_nodes": writes,
            # Even a schema without alias annotations doesn't establish purity.
            "nodes_with_unproven_effects": ir_graph["call_nodes_with_unproven_effects"],
            "has_control_flow": ir_graph["has_control_flow"],
        })
    return {
        "report_version": "f08-compiler-alias-lifetimes/0.1", "manifest_sha256": identity,
        "operator_schemas_sha256": _digest(operator_schemas),
        "status": "partial_must_alias_groups_physical_storage_pending", "graphs": reports,
        "supported_view_targets": sorted(VIEW_RULES),
        "order_status": "logical_ir_sequence_not_observed_kernel_schedule",
        "groups_may_alias_each_other": True,
        "physical_storage_analysis_complete": False, "peak_bytes": None,
        "runtime_applied": False, "optimization_applied": False,
        "runtime_release": "not_authorized_by_analysis",
    }
