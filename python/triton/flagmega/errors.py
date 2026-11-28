# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable error types shared by the FlagMega compiler and CLI."""

from __future__ import annotations


class FlagMegaError(Exception):
    """Base class for errors with a stable machine-readable code."""

    code = "flagmega_error"

    def __init__(self, message: str, *, stage: str | None = None, node_id: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.node_id = node_id

    def to_data(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "stage": self.stage,
            "node_id": self.node_id,
        }


class IRSchemaError(FlagMegaError):
    code = "ir_schema_error"


class IRVerificationError(FlagMegaError):
    code = "ir_verification_error"


class StageError(FlagMegaError):
    code = "stage_error"


class UnsupportedSelectionError(FlagMegaError):
    code = "unsupported_selection"


class ReviewRequired(FlagMegaError):
    code = "review_required"


class CheckpointError(FlagMegaError):
    code = "checkpoint_error"


class ArtifactError(FlagMegaError):
    code = "artifact_error"


class CodegenError(FlagMegaError):
    code = "codegen_error"


class RuntimeContractError(FlagMegaError):
    code = "runtime_contract_error"


class ImporterError(FlagMegaError):
    code = "importer_error"


class EvaluationError(FlagMegaError):
    code = "evaluation_error"


class NumpyMaterializationUnsupported(EvaluationError):
    """One constant recipe requires semantic arithmetic outside raw storage."""

    code = "numpy_materialization_unsupported"
