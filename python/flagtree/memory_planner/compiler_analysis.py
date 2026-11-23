"""Compiler-stage entry for F08 model IR analysis (disabled by default).

The hook consumes serialized IR metadata, not live tensors.  Stage 1 provides
use-def/lifetime information; physical storage analysis remains blocked until
F02/allocation facts have an agreed, validated compiler contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model_ir_adapter import CompilerIRAnalysisError, adapt_model_ir


def analyze_compiler_manifest(
    manifest: Mapping[str, Any], storage_facts=None, *, enabled: bool = False,
    operator_schemas=None,
) -> dict[str, Any] | None:
    """Analyze at compile time; never change the payload graph or runtime.

    No work at all is performed when disabled.  The unused storage_facts input
    is reserved explicitly: accepting arbitrary historical pointer records here
    would turn an unproved contract into a false alias/storage proof.
    """
    if not isinstance(enabled, bool):
        raise CompilerIRAnalysisError("invalid_enabled_flag", "enabled must be Boolean")
    if not enabled:
        return None
    if storage_facts is not None:
        raise CompilerIRAnalysisError(
            "storage_facts_not_supported_in_stage1",
            "compiler storage/alias fact schema requires F02/allocation interface agreement",
        )
    ir = adapt_model_ir(manifest)
    report = {
        "report_version": "f08-compiler-memory-analysis/0.1",
        "analysis_stage": "model_ir", "enabled": True,
        "manifest_sha256": ir["manifest_sha256"],
        "status": "ir_value_analysis_only_storage_pending",
        "runtime_applied": False, "optimization_applied": False,
        "runtime_release": "not_authorized_by_analysis",
        "ir_analysis": ir,
        "storage_analysis": {
            "status": "blocked_missing_compiler_storage_facts",
            "peak_bytes": None,
            "missing": ["allocation_ownership", "storage_alias_and_offset", "storage_capacity"],
        },
        "reuse_planning": {
            "status": "blocked", "runtime_applied": False,
            "blockers": ["compiler_storage_alias_contract_missing", "operator_effect_contract_missing",
                         "final_schedule_and_lowering_map_missing", "compiled_allocation_consumer_unconfirmed"],
        },
        "inplace_legality": {"status": "not_run_missing_alias_and_effects"},
        "evidence_scope": {
            "ir_node_is_kernel": False, "device_execution_observed": False,
            "metadata_is_runtime_layout_proof": False,
            "measured_memory_saving_bytes": None,
        },
    }
    if operator_schemas is not None:
        from .compiler_aliases import analyze_compiler_aliases

        report["report_version"] = "f08-compiler-memory-analysis/0.2"
        report["alias_analysis"] = analyze_compiler_aliases(manifest, operator_schemas)
        report["status"] = "ir_alias_lifetimes_only_storage_pending"
    return report
