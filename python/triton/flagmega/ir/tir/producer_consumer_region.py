# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit block-local producer/consumer execution region."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRStmt, tir_node
from triton.flagmega.ir.tir.pipeline_drain import PipelineDrain
from triton.flagmega.ir.tir.pipeline_handoff import PipelineHandoff
from triton.flagmega.ir.tir.pipeline_stage import PipelineStage
from triton.flagmega.ir.tir.sequential import Sequential


@tir_node("producer_consumer_region")
@dataclass(frozen=True)
class ProducerConsumerRegion(TIRStmt):
    """Two task bodies with an identical ordered set of pipeline stages."""

    produce_body: Sequential
    consume_body: Sequential

    def __post_init__(self) -> None:
        if not isinstance(self.produce_body, Sequential) or not isinstance(
            self.consume_body, Sequential
        ):
            raise IRSchemaError(
                "ProducerConsumerRegion requires Sequential task bodies."
            )
        produce = _collect_structure(self.produce_body, "producer")
        consume = _collect_structure(self.consume_body, "consumer")
        if not produce.stage_ids or not consume.stage_ids:
            raise IRSchemaError(
                "Producer/consumer regions must contain at least one pipeline stage."
            )
        if produce.stage_ids != consume.stage_ids:
            raise IRSchemaError(
                "Producer and consumer pipeline stages must have identical IDs and order."
            )
        if produce.drain_ids != consume.drain_ids:
            raise IRSchemaError(
                "Producer and consumer pipeline drain sets must be identical."
            )
        if produce.handoff_ids != consume.handoff_ids:
            raise IRSchemaError(
                "Producer and consumer handoffs must have identical IDs."
            )


@dataclass(frozen=True)
class _RegionStructure:
    stage_ids: tuple[str, ...]
    drain_ids: frozenset[str]
    handoff_ids: frozenset[str]


def _collect_structure(body: Sequential, role: str) -> _RegionStructure:
    # Imports stay local so these structural nodes remain independently
    # importable and registration cannot form a cycle.
    from triton.flagmega.ir.tir.block import Block
    from triton.flagmega.ir.tir.for_loop import For
    from triton.flagmega.ir.tir.if_then_else import IfThenElse
    from triton.flagmega.ir.tir.let import Let

    stage_ids: list[str] = []
    seen_stages: set[str] = set()
    drains: set[str] = set()
    handoffs: set[str] = set()

    def visit(statement: TIRStmt) -> None:
        if isinstance(statement, PipelineStage):
            if statement.stage_id in seen_stages:
                raise IRSchemaError(
                    f"{role} body contains duplicate pipeline stage "
                    f"{statement.stage_id!r}."
                )
            seen_stages.add(statement.stage_id)
            stage_ids.append(statement.stage_id)
            return
        if isinstance(statement, PipelineDrain):
            if statement.stage_id not in seen_stages:
                raise IRSchemaError(
                    f"{role} body drains pipeline stage {statement.stage_id!r} "
                    "before executing it."
                )
            if statement.stage_id in drains:
                raise IRSchemaError(
                    f"{role} body drains pipeline stage {statement.stage_id!r} "
                    "more than once."
                )
            drains.add(statement.stage_id)
            return
        if isinstance(statement, PipelineHandoff):
            if statement.handoff_id in handoffs:
                raise IRSchemaError(
                    f"{role} body contains duplicate pipeline handoff "
                    f"{statement.handoff_id!r}."
                )
            handoffs.add(statement.handoff_id)
            return
        if isinstance(statement, Sequential):
            for field in statement.fields:
                visit(field)
            return
        if isinstance(statement, IfThenElse):
            visit(statement.then_body)
            visit(statement.else_body)
            return
        if isinstance(statement, For):
            visit(statement.body)
            return
        if isinstance(statement, Let):
            visit(statement.body)
            return
        if isinstance(statement, Block):
            visit(statement.init_body)
            visit(statement.body)
            return
        if isinstance(statement, ProducerConsumerRegion):
            raise IRSchemaError("Producer/consumer regions cannot be nested.")

    visit(body)
    return _RegionStructure(tuple(stage_ids), frozenset(drains), frozenset(handoffs))


__all__ = ["ProducerConsumerRegion"]
