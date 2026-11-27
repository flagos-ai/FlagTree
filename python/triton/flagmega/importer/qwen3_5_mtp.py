# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One Qwen3.5 NextN step; uses the dense full-attention decoder dataflow.

The returned hidden state follows llama.cpp qwen35's post-head-norm h_nextn
contract. It and the independently owned MTP KV state feed the next draft step.
This semantic importer does not add batched verification to the target model.
"""
from triton.flagmega.importer.qwen3_5_dense import Qwen35DenseImporter


class Qwen35MTPImporter(Qwen35DenseImporter):
    def __init__(self, checkpoint, *, revision=None, block_size=256, num_blocks=16, hidden_lanes=1):
        super().__init__(checkpoint, revision=revision, block_size=block_size,
                         num_blocks=num_blocks, _mtp=True, _mtp_hidden_lanes=hidden_lanes)


def import_qwen35_mtp(checkpoint, **options):
    return Qwen35MTPImporter(checkpoint, **options).import_module()
