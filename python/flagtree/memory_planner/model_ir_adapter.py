"""Read-only use/definition analysis of the team's serialized model IR.

This first adapter deliberately does not invent storage IDs.  Core ATen values,
physical storage, and launched kernels are three different objects.  Without a
compiler alias/allocation contract, a last IR reference is NOT permission to
release or overwrite a device buffer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


class CompilerIRAnalysisError(ValueError):
    """Invalid compiler input, carrying a stable diagnostic code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def _json_metadata(value: Any):
    """Preserve valid +/-inf IR constants without emitting non-standard JSON.

    Attention masks legitimately use -inf.  Tagging this literal in the copied
    analysis metadata is not an IR rewrite and never changes compiled inputs.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"kind": "f08_float_literal", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, list):
        return [_json_metadata(x) for x in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _json_metadata(item) for key, item in value.items()}
    raise TypeError("compiler manifest must contain only serialized metadata")


def manifest_identity(manifest: Mapping[str, Any]) -> tuple[dict, str]:
    """Own the identity calculation; no caller-supplied hash can override it.

    JSON round-tripping also ensures we neither retain Tensor objects nor mutate
    a caller's metadata.  There is no Torch import or device access here.
    """
    if not isinstance(manifest, Mapping):
        raise CompilerIRAnalysisError("malformed_manifest", "manifest must be an object")
    try:
        encoded = json.dumps(_json_metadata(manifest), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        snapshot = json.loads(encoded)
        # Hash the original scalar form, not only the tagged copy: a literal
        # -inf and an actual dictionary spelling that tag must remain distinct.
        identity_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=True).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise CompilerIRAnalysisError("malformed_manifest", "metadata must be serializable compiler metadata") from exc
    return snapshot, hashlib.sha256(identity_bytes).hexdigest()


def _references(value: Any):
    """Decode the existing unified_ir._argument_summary node-reference format."""
    if isinstance(value, dict):
        if set(value) == {"node"}:
            if not isinstance(value["node"], str):
                raise CompilerIRAnalysisError("malformed_manifest", "node reference must be a string")
            yield value["node"]
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _tensor_leaves(value: Any, path: str = ""):
    # 与团队 contract.py 保持同一来源 ID 规则：root Tensor 是 value，
    # tuple/list 输出是 [0]/[1]。不能把 max.dim 的两个输出揉成一个 tensor。
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            yield path or "value", value
        else:
            for key in sorted(value):
                yield from _tensor_leaves(value[key], f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _tensor_leaves(item, f"{path}[{i}]")


def _has_opaque_metadata(value: Any) -> bool:
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "tensor":
            return False
        if kind is not None:
            return kind not in {"SymInt", "SymFloat", "SymBool"}
        return any(_has_opaque_metadata(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_opaque_metadata(item) for item in value)
    return False


def _projection(source_value: Any, index: Any) -> tuple[Any, str] | None:
    """Only a literal container projection proves a narrower value reference."""
    if isinstance(index, bool):
        return None
    if isinstance(source_value, list) and isinstance(index, int):
        if not -len(source_value) <= index < len(source_value):
            return None
        i = index % len(source_value)
        return source_value[i], f"[{i}]"
    if (isinstance(source_value, dict) and source_value.get("kind") != "tensor"
            and isinstance(index, str) and index in source_value):
        return source_value[index], index
    return None


def _diagnostic(code: str, graph: str, node: str | None = None) -> dict:
    result = {"code": code, "graph": graph}
    if node is not None:
        result["node"] = node
    return result


def _analyze_graph(graph: dict) -> dict:
    name = graph["name"]
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise CompilerIRAnalysisError("malformed_manifest", f"{name}: nodes must be nonempty")
    if "node_count" in graph and graph["node_count"] != len(nodes):
        raise CompilerIRAnalysisError("malformed_manifest", f"{name}: node_count mismatch")
    by_name, positions = {}, {}
    allowed_ops = {"placeholder", "get_attr", "call_function", "call_method", "call_module", "output"}
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not node["name"]:
            raise CompilerIRAnalysisError("malformed_manifest", f"{name}: invalid node")
        node_name = node["name"]
        if node_name in by_name:
            raise CompilerIRAnalysisError("duplicate_node", f"{name}/{node_name}")
        if node.get("op") not in allowed_ops:
            raise CompilerIRAnalysisError("unsupported_node_kind", f"{name}/{node_name}")
        if not isinstance(node.get("users"), list) or not all(isinstance(x, str) for x in node["users"]):
            raise CompilerIRAnalysisError("malformed_manifest", f"{name}/{node_name}: invalid users")
        if len(set(node["users"])) != len(node["users"]):
            raise CompilerIRAnalysisError("users_mismatch", f"{name}/{node_name}: duplicate users")
        # Strings such as '%x' are not the team's serialized argument format.
        if "args" not in node or not isinstance(node["args"], list) or not isinstance(node.get("kwargs"), dict):
            raise CompilerIRAnalysisError("malformed_manifest", f"{name}/{node_name}: serialized args/kwargs required")
        by_name[node_name], positions[node_name] = node, i

    outputs = [node for node in nodes if node["op"] == "output"]
    if len(outputs) != 1 or outputs[0] is not nodes[-1]:
        raise CompilerIRAnalysisError("malformed_manifest", f"{name}: one final output node required")
    node_inputs, actual_users = {}, {n: set() for n in by_name}
    for node in nodes:
        node_name = node["name"]
        refs = set(_references([node["args"], node["kwargs"]]))
        for ref in refs:
            if ref not in by_name:
                raise CompilerIRAnalysisError("unknown_reference", f"{name}/{node_name} -> {ref}")
            if positions[ref] >= positions[node_name]:
                raise CompilerIRAnalysisError("non_topological_reference", f"{name}/{node_name} -> {ref}")
            actual_users[ref].add(node_name)
        node_inputs[node_name] = refs
    for node in nodes:
        if set(node["users"]) != actual_users[node["name"]]:
            raise CompilerIRAnalysisError("users_mismatch", f"{name}/{node['name']}: args and users disagree")

    values, node_values, diagnostics = [], {}, []
    call_nodes = []
    for node in nodes:
        node_name = node["name"]
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            raise CompilerIRAnalysisError("malformed_manifest", f"{name}/{node_name}: invalid metadata")
        node_values[node_name] = []
        if node["op"] == "output":
            continue
        if node["op"].startswith("call_"):
            call_nodes.append(node_name)
        if "value" not in metadata:
            diagnostics.append(_diagnostic("missing_value_metadata", name, node_name))
            continue
        if _has_opaque_metadata(metadata["value"]):
            diagnostics.append(_diagnostic("incomplete_opaque_value_metadata", name, node_name))
        for path, tensor in _tensor_leaves(metadata["value"]):
            reasons = []
            if node["op"] in {"placeholder", "get_attr"}:
                reasons.append("graph_input_or_state")
            if tensor.get("requires_grad") is not False:
                reasons.append("gradient_boundary_not_excluded")
            required = {"dtype", "shape", "stride", "device", "requires_grad"}
            if (not required <= set(tensor) or not isinstance(tensor.get("requires_grad"), bool)
                    or any(not isinstance(tensor.get(key), str) or not tensor.get(key) for key in ("dtype", "device"))):
                diagnostics.append(_diagnostic("incomplete_tensor_metadata", name, node_name))
            if (not isinstance(tensor.get("shape"), list) or not isinstance(tensor.get("stride"), list)
                    or len(tensor.get("shape", [])) != len(tensor.get("stride", []))):
                diagnostics.append(_diagnostic("incomplete_tensor_layout", name, node_name))
            record = {
                "source_tensor_id": f"{name}/{node_name}:{path}",
                "node": node_name, "leaf_path": path, "metadata": tensor,
                "definition_index": positions[node_name], "consumer_nodes": [],
                "last_reference_index": positions[node_name],
                "retained_through_index": positions[node_name],
                "is_graph_output": False, "protected_reasons": reasons,
                "safe_to_release": False, "storage_id": None,
            }
            values.append(record)
            node_values[node_name].append(record)
    if len({v["source_tensor_id"] for v in values}) != len(values):
        raise CompilerIRAnalysisError("ambiguous_tensor_id", f"{name}: ambiguous output leaf paths")

    # 普通 tuple 引用消费全部叶子；只有结构和元数据都匹配的 getitem 才缩小。
    # 这是 Python/IR 输出投影关系，不是所有 view 的物理 storage 别名分析。
    projections, selected_inputs = [], {}
    for node in nodes:
        node_name = node["name"]
        target = str(node.get("target", ""))
        args = node["args"]
        if (node["op"] == "call_function" and target in {"operator.getitem", "builtins.getitem", "_operator.getitem"}
                and len(args) == 2 and isinstance(args[0], dict) and set(args[0]) == {"node"}
                and not node["kwargs"]):
            source = args[0]["node"]
            source_meta = by_name[source].get("metadata", {}).get("value")
            selected = _projection(source_meta, args[1])
            if selected is not None:
                subtree, prefix = selected
                destination_meta = node.get("metadata", {}).get("value")
                if subtree == destination_meta:
                    source_leaves = node_values[source]
                    selected_inputs[node_name] = {
                        v["source_tensor_id"] for v in source_leaves
                        if v["leaf_path"] == prefix or v["leaf_path"].startswith(prefix + "[")
                        or v["leaf_path"].startswith(prefix + ".")
                    }
                    source_by_path = {path: v for path, v in _tensor_leaves(subtree)}
                    actual_by_path = {v["leaf_path"]: v for v in node_values[node_name]}
                    for child_path in source_by_path:
                        # 字典真的可能有一个 key 叫 value；不能把这个路径误当
                        # 根 Tensor 的 fallback 路径，否则会漏掉嵌套投影保护。
                        root_tensor = isinstance(subtree, dict) and subtree.get("kind") == "tensor"
                        parent_path = prefix if root_tensor else prefix + (child_path if child_path.startswith("[") else "." + child_path)
                        parent = next((v for v in source_leaves if v["leaf_path"] == parent_path), None)
                        child = actual_by_path.get(child_path)
                        if parent is not None and child is not None:
                            projections.append({"source": parent["source_tensor_id"], "target": child["source_tensor_id"]})
                else:
                    diagnostics.append(_diagnostic("projection_metadata_mismatch", name, node_name))

    for node in nodes:
        node_name = node["name"]
        for ref in sorted(node_inputs[node_name], key=positions.__getitem__):
            for value in node_values[ref]:
                if node_name in selected_inputs and value["source_tensor_id"] not in selected_inputs[node_name]:
                    continue
                value["consumer_nodes"].append(node_name)
                value["last_reference_index"] = positions[node_name]
                if node["op"] == "output":
                    value["is_graph_output"] = True
        # tuple 容器还活着时，不能因某叶子的最后直接投影较早就声称可释放。
    for node in nodes:
        node_name = node["name"]
        raw = node.get("metadata", {}).get("value")
        container = isinstance(raw, list) or (isinstance(raw, dict) and raw.get("kind") != "tensor")
        container_end = max([positions[x] for x in actual_users[node_name]], default=positions[node_name])
        for value in node_values[node_name]:
            value["retained_through_index"] = max(value["last_reference_index"], container_end if container else value["last_reference_index"])

    # 投影返回的是同一 IR Tensor 引用；传播保护/存续信息，但绝不授权物理缓冲释放。
    by_id = {v["source_tensor_id"]: v for v in values}
    groups = {key: key for key in by_id}
    def find(key):
        while groups[key] != key:
            groups[key] = groups[groups[key]]
            key = groups[key]
        return key
    for edge in projections:
        groups[find(edge["target"])] = find(edge["source"])
    members = {}
    for key, value in by_id.items():
        members.setdefault(find(key), []).append(value)
    for group in members.values():
        end = max(v["retained_through_index"] for v in group)
        protected = sorted({x for v in group for x in v["protected_reasons"]})
        output = any(v["is_graph_output"] for v in group)
        if output:
            protected.append("graph_output")
        for value in group:
            value["retained_through_index"] = end
            value["protected_reasons"] = sorted(set(protected))

    control_flow = bool(graph.get("control_flow")) or any(
        node["op"] == "call_function" and any(marker in str(node.get("target", "")) for marker in ("higher_order", "while_loop", "map_impl"))
        for node in nodes
    )
    if control_flow:
        diagnostics.append(_diagnostic("control_flow_requires_region_and_capture_contract", name))
    return {
        "graph": name, "node_order": [node["name"] for node in nodes],
        "values": values, "projection_aliases": projections,
        "diagnostics": diagnostics, "call_nodes_with_unproven_effects": call_nodes,
        "use_def_scope": "local_graph_only", "has_control_flow": control_flow,
        "value_metadata_complete": not any(x["code"].startswith(("missing_", "incomplete_")) for x in diagnostics),
    }


def adapt_model_ir(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the existing serialized IR without evaluating model/operator code."""
    snapshot, identity = manifest_identity(manifest)
    graphs = snapshot.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise CompilerIRAnalysisError("malformed_manifest", "nonempty graphs required")
    names = set()
    reports = []
    for graph in graphs:
        if not isinstance(graph, dict) or not isinstance(graph.get("name"), str) or not graph["name"]:
            raise CompilerIRAnalysisError("malformed_manifest", "graph name required")
        if graph["name"] in names:
            raise CompilerIRAnalysisError("duplicate_graph", graph["name"])
        names.add(graph["name"])
        reports.append(_analyze_graph(graph))
    all_ids = [v["source_tensor_id"] for report in reports for v in report["values"]]
    if len(set(all_ids)) != len(all_ids):
        raise CompilerIRAnalysisError("ambiguous_tensor_id", "graph/node/leaf source IDs collide")
    diagnostics = [d for report in reports for d in report["diagnostics"]]
    if len(reports) != 1:
        diagnostics.append({"code": "cross_graph_capture_and_control_flow_not_analyzed"})
    return {
        "manifest_sha256": identity, "analysis_stage": "model_ir",
        "order_status": "logical_ir_sequence_not_observed_kernel_schedule",
        "graphs": reports, "diagnostics": diagnostics,
        "storage_peak_bytes": None, "runtime_applied": False,
        "physical_storage_alias_analysis_complete": False,
    }
