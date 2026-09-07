# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable human/agent command-line interface for FlagMega."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from triton.flagmega.artifacts import load_artifact, write_artifact
from triton.flagmega.codegen.triton.function_schedule import (
    describe_function_schedule,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags, dump_flags_text, parse_dump_flags
from triton.flagmega.errors import (
    ArtifactError,
    CheckpointError,
    CodegenError,
    FlagMegaError,
    IRSchemaError,
    IRVerificationError,
    ReviewRequired,
    RuntimeContractError,
    UnsupportedSelectionError,
)
from triton.flagmega.ir import IRModule, companion_suffix, emit_module, load_module, verify_module
from triton.flagmega.importer import DirectoryCheckpoint, import_model, import_model_layer
from triton.flagmega.options import CompileOptions
from triton.flagmega.selection import emit_plan, load_plan, override_plan
from triton.flagmega.stages import stage_names
from triton.flagmega.targets import target_names
from triton.flagmega.runtime import load as load_runtime


CLI_SCHEMA = "flagmega.cli/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flagmega", description="Editable staged compiler for FlagTree")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Load and verify a Python IR checkpoint")
    verify.add_argument("input")
    verify.add_argument("--expect-stage")
    verify.add_argument("--expect-dialect")
    _json_option(verify)

    inspect = subparsers.add_parser("inspect", help="Inspect a Python IR checkpoint")
    inspect.add_argument("input")
    _json_option(inspect)

    schedule = subparsers.add_parser(
        "schedule",
        help="Inspect the physical call schedule of bufferized Python IR",
    )
    schedule.add_argument("input")
    schedule.add_argument("--function")
    _json_option(schedule)

    diff = subparsers.add_parser("diff", help="Compare two Python IR checkpoints")
    diff.add_argument("lhs")
    diff.add_argument("rhs")
    _json_option(diff)

    candidates = subparsers.add_parser("candidates", help="List compiler selection points")
    candidates.add_argument("input")
    _json_option(candidates)

    select = subparsers.add_parser("select", help="Write an agent/user selection override plan")
    select.add_argument("input")
    select.add_argument("--point", required=True)
    select.add_argument("--candidate", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--rationale", default="Explicit CLI override.")
    _json_option(select)

    import_command = subparsers.add_parser("import", help="Import a supported model layer into editable Python IR")
    import_command.add_argument("--model", required=True)
    import_command.add_argument("--revision")
    import_command.add_argument("--numerical-profile", default="nncase", help="Explicit importer numerical contract")
    import_command.add_argument("--layer", type=int, choices=(0, ), default=0)
    import_command.add_argument("--full-model", action="store_true")
    import_command.add_argument("--mode", choices=("decode-1", ), default="decode-1")
    import_command.add_argument("--output", required=True)
    _json_option(import_command)

    stage = subparsers.add_parser("stage", help="Run exactly one named compiler stage")
    stage.add_argument("stage", choices=stage_names())
    stage.add_argument("--input", required=True)
    stage.add_argument("--output", required=True)
    stage.add_argument("--decisions")
    _compiler_options(stage)

    compile_command = subparsers.add_parser("compile", help="Compile an imported checkpoint end to end")
    compile_source = compile_command.add_mutually_exclusive_group(required=True)
    compile_source.add_argument("--input")
    compile_source.add_argument("--model")
    compile_command.add_argument("--revision")
    compile_command.add_argument("--numerical-profile", help="Importer contract; with --input it must match the saved IR")
    compile_command.add_argument("--layer", type=int, choices=(0, ), default=0)
    compile_command.add_argument("--full-model", action="store_true")
    compile_command.add_argument("--mode", choices=("decode-1", ), default="decode-1")
    compile_command.add_argument(
        "--checkpoint",
        help="Checkpoint root used to materialize weights when compiling an edited --input IR",
    )
    compile_command.add_argument("--output", required=True, help="Artifact output directory")
    compile_command.add_argument("--stop-after")
    compile_command.add_argument("--emit-executable", action="store_true")
    compile_rdata_cache = compile_command.add_mutually_exclusive_group()
    compile_rdata_cache.add_argument("--rdata-cache-dir")
    compile_rdata_cache.add_argument("--no-rdata-cache", action="store_true")
    _compiler_options(compile_command)

    for name in ("resume", "replay"):
        command = subparsers.add_parser(name, help=f"{name.capitalize()} compilation from a Python checkpoint")
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--stop-after")
        command.add_argument("--checkpoint", help="Checkpoint root used to materialize executable rdata")
        command.add_argument("--emit-executable", action="store_true")
        rdata_cache = command.add_mutually_exclusive_group()
        rdata_cache.add_argument("--rdata-cache-dir")
        rdata_cache.add_argument("--no-rdata-cache", action="store_true")
        _compiler_options(command)

    artifact = subparsers.add_parser("artifact", help="Artifact operations")
    artifact_subparsers = artifact.add_subparsers(dest="artifact_command", required=True)
    artifact_verify = artifact_subparsers.add_parser("verify")
    artifact_verify.add_argument("path")
    _json_option(artifact_verify)
    artifact_run = artifact_subparsers.add_parser("run")
    artifact_run.add_argument("path")
    artifact_run.add_argument("--input", action="append", required=True, dest="inputs")
    artifact_run.add_argument("--output", required=True)
    artifact_run.add_argument("--device", default="cuda:0")
    artifact_run.add_argument("--steps", type=int, default=1)
    _json_option(artifact_run)
    return parser


def _json_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="json_output")


def _compiler_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", choices=target_names(), default="nvidia-sm90")
    parser.add_argument("--work-dir")
    parser.add_argument(
        "--dump-flags",
        type=parse_dump_flags,
        default=DumpFlags.NONE,
        metavar="FLAGS",
        help="Comma-separated dump categories (pass-ir,compile,rewrite,egraph-cost,evaluator,schedule,codegen or all)",
    )
    parser.add_argument(
        "--emit-every-stage",
        action="store_true",
        help="Compatibility alias that adds the compile dump flag",
    )
    parser.add_argument("--require-review", action="store_true")
    _json_option(parser)


def _options(args: argparse.Namespace) -> CompileOptions:
    return CompileOptions(
        target=args.target,
        emit_every_stage=bool(args.emit_every_stage),
        require_review=bool(args.require_review),
        work_dir=None if args.work_dir is None else Path(args.work_dir),
        dump_flags=DumpFlags(args.dump_flags),
    )


def _summary(module: IRModule) -> dict[str, object]:
    return {
        "ir_version": module.ir_version,
        "dialect": module.dialect,
        "stage": module.stage,
        "semantic_hash": module.semantic_hash,
        "entry": module.entry,
        "nodes": len(module.nodes),
        "functions": len(module.functions),
        "constant_phase": str(module.metadata.get("constant_phase", "open")),
        "constant_recipes": len(module.constant_recipes),
        "selection_points": len(module.selection_points),
        "selections": len(module.selections),
    }


def _selection_points(module: IRModule) -> list[dict[str, object]]:
    selected = module.selection_map
    return [
        {
            "id": point.id,
            "kind": point.kind,
            "owner": point.owner,
            "default": point.default_candidate,
            "selected": selected.get(point.id).candidate_id if point.id in selected else None,
            "alternatives": [candidate.to_data() for candidate in point.candidates],
        } for point in module.selection_points
    ]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "verify":
        module = load_module(args.input, expected_stage=args.expect_stage, expected_dialect=args.expect_dialect)
        return _ok(args.command, input_semantic_hash=module.semantic_hash, module=_summary(module))
    if args.command == "inspect":
        module = load_module(args.input)
        return _ok(
            args.command,
            input_semantic_hash=module.semantic_hash,
            module=_summary(module),
            metadata=dict(module.metadata),
            constant_recipes=[{
                "id": recipe.id,
                "fingerprint": recipe.fingerprint,
                "nodes": len(recipe.nodes),
                "outputs": list(recipe.outputs),
            } for recipe in module.constant_recipes],
            selection_points=_selection_points(module),
        )
    if args.command == "schedule":
        module = load_module(
            args.input,
            expected_stage="bufferized_tir",
            expected_dialect="bufferized_tir",
        )
        schedule = describe_function_schedule(
            module,
            function_name=args.function,
        )
        runtime_binding = describe_function_runtime_binding(
            module,
            function_name=args.function,
        )
        return _ok(
            args.command,
            input_semantic_hash=module.semantic_hash,
            schedule=schedule,
            runtime_binding=runtime_binding,
        )
    if args.command == "diff":
        lhs = load_module(args.lhs)
        rhs = load_module(args.rhs)
        lhs_nodes = {node.id: node.to_data() for node in lhs.nodes}
        rhs_nodes = {node.id: node.to_data() for node in rhs.nodes}
        changed = sorted(
            node_id for node_id in lhs_nodes.keys() & rhs_nodes.keys()
            if lhs_nodes[node_id] != rhs_nodes[node_id]
        )
        lhs_recipes = {recipe.id: recipe.to_data() for recipe in lhs.constant_recipes}
        rhs_recipes = {recipe.id: recipe.to_data() for recipe in rhs.constant_recipes}
        changed_recipes = sorted(
            recipe_id for recipe_id in lhs_recipes.keys() & rhs_recipes.keys()
            if lhs_recipes[recipe_id] != rhs_recipes[recipe_id]
        )
        return _ok(
            args.command,
            input_semantic_hash=lhs.semantic_hash,
            output_semantic_hash=rhs.semantic_hash,
            equal=lhs.semantic_hash == rhs.semantic_hash,
            added_nodes=sorted(rhs_nodes.keys() - lhs_nodes.keys()),
            removed_nodes=sorted(lhs_nodes.keys() - rhs_nodes.keys()),
            changed_nodes=changed,
            added_constant_recipes=sorted(rhs_recipes.keys() - lhs_recipes.keys()),
            removed_constant_recipes=sorted(lhs_recipes.keys() - rhs_recipes.keys()),
            changed_constant_recipes=changed_recipes,
        )
    if args.command == "candidates":
        module = load_module(args.input)
        return _ok(
            args.command,
            input_semantic_hash=module.semantic_hash,
            selection_points=_selection_points(module),
            next_actions=[{"argv": ["flagmega", "select", args.input, "--point", "POINT", "--candidate", "CANDIDATE"]}],
        )
    if args.command == "select":
        module = load_module(args.input)
        plan = override_plan(module, ((args.point, args.candidate), ), rationale=args.rationale)
        output = emit_plan(plan, args.output)
        return _ok(
            args.command,
            input_semantic_hash=module.semantic_hash,
            outputs=[str(output)],
            selection_records=[record.to_data() for record in plan.records],
        )
    if args.command == "import":
        module = (
            import_model(args.model, revision=args.revision, numerical_profile=args.numerical_profile)
            if args.full_model
            else import_model_layer(args.model, layer=args.layer, revision=args.revision,
                                    numerical_profile=args.numerical_profile)
        )
        output = emit_module(module, args.output)
        return _ok(
            args.command,
            output_semantic_hash=module.semantic_hash,
            outputs=[str(output), str(output.with_suffix(companion_suffix(module)))],
            module=_summary(module),
        )
    if args.command == "stage":
        module = load_module(args.input)
        plan = None if args.decisions is None else load_plan(args.decisions)
        compiler = Compiler(_options(args))
        result = compiler.run_stage(module, args.stage, plan=plan, output=args.output)
        return _ok(
            args.command,
            input_semantic_hash=module.semantic_hash,
            output_semantic_hash=result.module.semantic_hash,
            outputs=[
                str(result.checkpoint or args.output),
                str(Path(result.checkpoint or args.output).with_suffix(companion_suffix(result.module))),
            ],
            module=_summary(result.module),
            selection_points=_selection_points(result.module),
            dump_flags=dump_flags_text(compiler.options.effective_dump_flags),
        )
    if args.command in {"compile", "resume", "replay"}:
        source = (
            (
                import_model(args.model, revision=args.revision, numerical_profile=args.numerical_profile or "nncase")
                if args.full_model
                else import_model_layer(args.model, layer=args.layer, revision=args.revision,
                                        numerical_profile=args.numerical_profile or "nncase")
            )
            if args.command == "compile" and args.model is not None
            else load_module(args.input)
        )
        compiler = Compiler(_options(args))
        if (args.command == "compile" and args.input is not None and args.numerical_profile is not None
                and args.numerical_profile != source.metadata.get("numerical_contract", "nncase")):
            raise IRSchemaError("--numerical-profile does not match the saved IR; select the contract at import.")
        result = compiler.compile(source, stop_after=args.stop_after)
        checkpoint_path = (
            args.model
            if args.command == "compile" and args.model is not None
            else getattr(args, "checkpoint", None)
        )
        checkpoint = None if checkpoint_path is None else DirectoryCheckpoint(checkpoint_path)
        rdata_cache_dir = _rdata_cache_dir(args, checkpoint)
        artifact = write_artifact(
            result.module,
            args.output,
            target=args.target,
            checkpoint=checkpoint,
            emit_executable=bool(getattr(args, "emit_executable", False)),
            dumper=compiler.diagnostics.dumper,
            rdata_cache_dir=rdata_cache_dir,
        )
        return _ok(
            args.command,
            input_semantic_hash=source.semantic_hash,
            output_semantic_hash=result.module.semantic_hash,
            outputs=[str(artifact)],
            module=_summary(result.module),
            reports=[report.to_data() for report in result.reports],
            dump_flags=dump_flags_text(compiler.options.effective_dump_flags),
            rdata_cache_dir=(
                None if rdata_cache_dir is None else str(rdata_cache_dir)
            ),
        )
    if args.command == "artifact" and args.artifact_command == "verify":
        manifest, module = load_artifact(args.path)
        return _ok(
            "artifact.verify",
            input_semantic_hash=module.semantic_hash,
            artifact=manifest,
            module=_summary(module),
        )
    if args.command == "artifact" and args.artifact_command == "run":
        try:
            import torch
        except ImportError as error:
            raise RuntimeContractError("Artifact CLI run requires PyTorch tensor serialization.") from error
        inputs = tuple(torch.load(path, map_location=args.device, weights_only=True) for path in args.inputs)
        if args.steps <= 0:
            raise RuntimeContractError("Artifact run --steps must be positive.")
        runtime = load_runtime(args.path, device=args.device)
        if hasattr(runtime, "create_state"):
            if len(inputs) not in {1, args.steps}:
                raise RuntimeContractError(
                    "Stateful artifact expects one reusable input tensor or exactly "
                    f"--steps tensors, got {len(inputs)} inputs for {args.steps} steps.")
            step_inputs = inputs * args.steps if len(inputs) == 1 else inputs
            state = runtime.create_state()
            result_kind = getattr(runtime, "result_kind", None)
            if result_kind == "logits_token_state":
                output, next_token = runtime.create_outputs()
                runtime.prepare(
                    step_inputs[0], state, logits=output, next_token=next_token)
                for input_ids in step_inputs:
                    runtime.run_into(output, next_token, input_ids, state)
            elif result_kind == "tensor_state":
                output = runtime.create_outputs()
                runtime.prepare(step_inputs[0], state, output=output)
                for input_ids in step_inputs:
                    runtime.run_into(output, input_ids, state)
            else:
                raise RuntimeContractError(
                    f"Stateful runtime has unsupported result contract {result_kind!r}."
                )
            torch.cuda.synchronize(torch.device(args.device))
            if hasattr(state, "kv_caches"):
                serialized_output = {
                    (
                        "logits"
                        if result_kind == "logits_token_state"
                        else "hidden"
                    ): output.detach().cpu(),
                    "kv_caches": state.kv_caches.detach().cpu(),
                    "query_start_loc": state.query_start_loc.detach().cpu(),
                    "seq_lens": state.seq_lens.detach().cpu(),
                    "slot_mapping": state.slot_mapping.detach().cpu(),
                    "block_table": state.block_table.detach().cpu(),
                }
                if result_kind == "logits_token_state":
                    serialized_output["next_token"] = next_token.detach().cpu()
            elif hasattr(state, "convolution") and hasattr(state, "recurrent"):
                serialized_output = {
                    "hidden": output.detach().cpu(),
                    "convolution_state": state.convolution.detach().cpu(),
                    "recurrent_state": state.recurrent.detach().cpu(),
                }
            else:
                raise RuntimeContractError(
                    "Stateful runtime returned an unsupported state ABI."
                )
        else:
            if args.steps != 1:
                raise RuntimeContractError("Stateless artifacts only support --steps 1.")
            runtime.prepare(*inputs)
            serialized_output = runtime.run(*inputs).detach().cpu()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(serialized_output, output_path)
        return _ok(
            "artifact.run",
            input_semantic_hash=runtime.ir_module.semantic_hash,
            outputs=[str(output_path)],
            device=args.device,
            steps=args.steps,
            prepare_count=runtime.prepare_count,
            resources=runtime.resource_report,
        )
    raise AssertionError(f"Unhandled command {args.command!r}.")


def _rdata_cache_dir(
    args: argparse.Namespace,
    checkpoint: DirectoryCheckpoint | None,
) -> Path | None:
    """Keep CLI cache entries beside artifacts so Linux FICLONE can apply."""

    if checkpoint is None or bool(getattr(args, "no_rdata_cache", False)):
        return None
    requested = getattr(args, "rdata_cache_dir", None)
    if requested is not None:
        return Path(requested)
    return Path(args.output).parent / ".flagmega-rdata-cache"


def _ok(command: str, **fields: Any) -> dict[str, Any]:
    return {"schema": CLI_SCHEMA, "ok": True, "command": command, **fields}


def _json_value(value: Any) -> Any:
    """Convert frozen IR values to the stable, plain CLI JSON boundary."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    to_data = getattr(value, "to_data", None)
    if callable(to_data):
        return _json_value(to_data())
    return value


def _exit_code(error: BaseException) -> int:
    if isinstance(error, (IRSchemaError, IRVerificationError, CheckpointError)):
        return 10
    if isinstance(error, UnsupportedSelectionError):
        return 20
    if isinstance(error, ReviewRequired):
        return 30
    if isinstance(error, ArtifactError):
        return 50
    if isinstance(error, CodegenError):
        return 40
    if isinstance(error, RuntimeContractError):
        return 60
    if isinstance(error, FlagMegaError):
        return 1
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except Exception as error:
        if isinstance(error, FlagMegaError):
            error_data = error.to_data()
        else:
            error_data = {"code": "internal_error", "message": str(error), "stage": None, "node_id": None}
        result = {"schema": CLI_SCHEMA, "ok": False, "command": args.command, "error": error_data}
        if getattr(args, "json_output", False):
            print(json.dumps(_json_value(result), sort_keys=True))
        else:
            print(f"flagmega: {error_data['code']}: {error_data['message']}", file=sys.stderr)
        return _exit_code(error)
    if getattr(args, "json_output", False):
        print(json.dumps(_json_value(result), sort_keys=True))
    else:
        print(json.dumps(_json_value(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
