# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Flag-controlled evaluator argument/result dumps with tensor byte payloads."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

from triton.flagmega.diagnostics import DumpFlags, DumpScope
from triton.flagmega.diagnostics.paths import encode_file_component
from triton.flagmega.ir import IRModule, Node, get_definition


class EvaluatorDumpWriter:
    """One evaluator-run dump scope; incomplete calls retain argument dumps."""

    def __init__(self, module: IRModule) -> None:
        root = DumpScope.current()
        self.enabled = root.is_enabled(DumpFlags.EVALUATOR) and root.directory is not None
        self.module = module
        self.index = 0
        if not self.enabled:
            self.dumper = root.create_sub_dumper("Evaluate")
            return
        prefix = (root.relative_directory / "Evaluate").as_posix().strip("/")
        prefix = f"{prefix}/" if prefix else ""
        runs = {
            record.relative_path[len(prefix):].split("/", 1)[0]
            for record in root.manager.artifacts
            if record.relative_path.startswith(f"{prefix}Run")
        }
        run_index = len(runs)
        self.dumper = root.create_sub_dumper(f"Evaluate/Run{run_index:04d}")

    def before(self, function: str, node: Node, arguments: tuple[Any, ...]) -> int:
        token = self.index
        self.index += 1
        if not self.enabled:
            return token
        directory = f"{token:04d}_{_file_name(node.id)}"
        definition = get_definition(node.op)
        names: list[str] = []
        for parameter in definition.input_parameters:
            if parameter.variadic:
                names.extend(
                    f"{parameter.name}[{index}]"
                    for index in range(len(names), len(arguments))
                )
            else:
                names.append(parameter.name)
        payload = {
            "schema": "flagmega.evaluator-call/v1",
            "phase": "before",
            "sequence": token,
            "function": function,
            "node": node.id,
            "op": node.op,
            "module_semantic_hash": self.module.semantic_hash,
            "arguments": [
                {
                    "name": names[index] if index < len(names) else f"arg{index}",
                    "value": self._value(directory, f"arg{index}", value),
                }
                for index, value in enumerate(arguments)
            ],
        }
        self._json(f"{directory}/arguments.json", payload)
        return token

    def after(self, token: int, function: str, node: Node, value: Any) -> None:
        if not self.enabled:
            return
        directory = f"{token:04d}_{_file_name(node.id)}"
        payload = {
            "schema": "flagmega.evaluator-call/v1",
            "phase": "after",
            "sequence": token,
            "function": function,
            "node": node.id,
            "op": node.op,
            "module_semantic_hash": self.module.semantic_hash,
            "result": self._value(directory, "result", value),
        }
        self._json(f"{directory}/result.json", payload)

    def _value(self, directory: str, name: str, value: Any, *, parents: tuple[str, ...] = ()) -> object:
        path = (*parents, name)
        if hasattr(value, "detach") and hasattr(value, "shape") and hasattr(value, "dtype"):
            tensor = value.detach().contiguous().cpu()
            payload_name = "/".join(encode_file_component(part) for part in path) + ".bin"
            relative = f"{directory}/{payload_name}"
            # Viewing as uint8 preserves BF16/FP8 bit patterns without numpy
            # dtype support. ``bytes(list)`` would be prohibitively slow.
            # dtype-view requires rank > 0 when element widths differ. Keep
            # scalar shape in metadata, flatten only the raw byte view.
            byte_tensor = tensor.reshape(-1).view(_torch().uint8)
            payload = byte_tensor.numpy().tobytes()
            with self.dumper.open_artifact(
                relative,
                "wb",
                category=DumpFlags.EVALUATOR,
                kind="tensor-bytes",
                producer="TorchEvaluator",
                source_semantic_hash=self.module.semantic_hash,
            ) as stream:
                stream.write(payload)
            return {
                "kind": "tensor",
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "device": str(value.device),
                "nbytes": len(payload),
                "payload": payload_name,
            }
        if isinstance(value, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "fields": [
                    self._value(directory, f"item_{index}", item, parents=path)
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "fields": {
                    str(key): self._value(directory, f"field_{key}", item, parents=path)
                    for key, item in sorted(value.items(), key=lambda item: str(item[0]))
                },
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "kind": "dataclass",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    field.name: self._value(directory, f"field_{field.name}", getattr(value, field.name), parents=path)
                    for field in fields(value)
                },
            }
        if isinstance(value, Enum):
            return {"kind": "enum", "type": type(value).__qualname__, "value": value.value}
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"kind": "scalar", "value": value}
        return {"kind": "opaque", "type": f"{type(value).__module__}.{type(value).__qualname__}"}

    def _json(self, relative: str, payload: object) -> None:
        with self.dumper.open_artifact(
            relative,
            category=DumpFlags.EVALUATOR,
            kind="evaluator-manifest",
            producer="TorchEvaluator",
            source_semantic_hash=self.module.semantic_hash,
            encoding="utf-8",
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")


def _file_name(name: str) -> str:
    return encode_file_component(str(name))


def _torch():
    import torch

    return torch


__all__ = ["EvaluatorDumpWriter"]
