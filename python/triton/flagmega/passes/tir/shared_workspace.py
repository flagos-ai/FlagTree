# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable buffer-plan identities for target-private TIR shared workspaces."""

from __future__ import annotations


def shared_workspace_buffer_id(function_name: str, workspace_name: str) -> str:
    if not function_name or not workspace_name:
        raise ValueError("Shared workspace identity requires function and workspace names.")
    return f"@{function_name}.shared.{workspace_name}"


__all__ = ["shared_workspace_buffer_id"]
