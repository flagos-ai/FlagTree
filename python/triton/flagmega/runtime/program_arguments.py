# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""No-copy per-invocation projection into a shared GPU function ABI."""

from triton.flagmega.errors import RuntimeContractError


def project_arguments(bindings, roots):
    result = []
    for binding in bindings:
        kind = binding["kind"]
        if kind == "immediate":
            value = binding["value"]
        else:
            value = roots[binding["source"]]
            if kind == "pointer":
                import torch

                dtype = getattr(torch, binding["dtype"])
                offset = binding["byte_offset"]
                # Offset/scope units are bytes. The CTA displacement remains
                # in the wrapper, so this is the unselected parent frame.
                byte_view = value.view(torch.uint8).reshape(-1)
                remaining = byte_view.numel() - offset
                if offset < 0 or remaining < 0 or offset % dtype.itemsize:
                    raise RuntimeContractError("Shared GPU argument is outside its scalar-aligned root allocation.")
                value = byte_view.narrow(0, offset, remaining // dtype.itemsize * dtype.itemsize).view(dtype)
            elif kind == "scalar_argument":
                # An identity binding must preserve bool and signed zero.
                # Only rebased descriptor origins carry a nonzero addend.
                if binding["add"]:
                    value += binding["add"]
            elif kind != "argument":
                raise RuntimeContractError("Unknown shared GPU entry argument binding.")
        result.append(value)
    return tuple(result)


class BoundProgram:
    """Own scratch per invocation while sharing the compiled specialization."""

    def __init__(self, prepared, bindings, dynamic_indices, root_dynamic_indices):
        self.prepared = prepared
        self._bindings = tuple(bindings[i] for i in dynamic_indices)
        self._root_dynamic = tuple(root_dynamic_indices)

    def __getattr__(self, name):
        return getattr(self.prepared, name)

    def launch(self, *arguments, stream=None):
        if len(arguments) != len(self._root_dynamic):
            raise RuntimeContractError("Shared GPU program requires the complete dynamic root ABI.")
        roots = dict(zip(self._root_dynamic, arguments, strict=True))
        self.prepared.launch(*project_arguments(self._bindings, roots), stream=stream)
