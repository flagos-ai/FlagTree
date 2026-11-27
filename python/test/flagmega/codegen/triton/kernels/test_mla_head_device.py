# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from math import prod
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
from python.test.flagmega.ir.ops.nn._mla_head_fixtures import policies
from python.test.flagmega.codegen.triton.kernels.test_mla_head_codegen import compile_projection


@pytest.mark.parametrize(
    "direction,fp8,cyclic,staged,scale_sharded,reusable,lanes,output_lanes",
    [
        ("absorb_query", False, False, False, False, False, (2, 4), (4,)),
        ("expand_value", False, True, False, False, False, (8,), (2, 4)),
        ("absorb_query", True, True, False, False, True, (), (2, 4)),
        ("expand_value", True, False, True, True, True, (2, 2, 2), ()),
        ("absorb_query", True, False, False, True, False, (1,), (8,)),
        ("expand_value", True, True, False, True, False, (2, 4), (4,)),
    ],
)
def test_device_head_ownership_preserves_all_outputs(
    tmp_path, direction, fp8, cyclic, staged, scale_sharded, reusable, lanes, output_lanes
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    mesh = fm.Placement((2, 2), "yx", "bb")
    heads = 7 if cyclic else 8
    hp, rp = policies(mesh, heads=heads, cyclic=cyclic, staged=staged)
    runtimes = {}
    arguments = {}
    outputs = {}
    shared = {}
    guards = []

    def allocate(kind):
        t = fm.logical_type(kind)
        shape = tuple(d.fixed_value for d in t.shape) + tuple(getattr(t.dtype, "lanes", ()))
        dtype = getattr(torch, getattr(t.dtype, "elem_type", t.dtype).value)
        storage = torch.full((prod(shape) + 32,), 64, dtype=torch.float32, device="cuda").to(dtype)
        guards.append((storage, storage[:16].view(torch.uint8).clone(), storage[-16:].view(torch.uint8).clone()))
        return storage[16:-16].reshape(shape)

    try:
        for split in (False, True):
            m, p = compile_projection(
                tmp_path / str(split),
                direction=direction,
                fp8=fp8,
                mesh=mesh,
                heads=heads,
                head_policy=hp if split else None,
                row_policy=rp if split else None,
                scale_sharded=scale_sharded if split else False,
                reusable=reusable,
                lanes=lanes,
                output_lanes=output_lanes,
            )
            r = load(p, device="cuda:0")
            runtimes[split] = r
            values = {}
            for spec in r.external_arguments:
                name = spec["value"]
                if spec["role"] == "result":
                    outputs[split] = allocate(m.node_map[name].type)
                    values[name] = outputs[split]
                else:
                    if name not in shared:
                        shared[name] = allocate(m.node_map[name].type)
                    values[name] = shared[name]
            arguments[split] = tuple(values[s["value"]] for s in r.external_arguments)
            Run.prepare(r, *arguments[split])
            assert r.resource_report["spill_bytes"] == 0
        g = torch.Generator().manual_seed(28173)
        prefix = "outer_" if reusable else ""
        width = 16 if direction == "absorb_query" else 32
        for mutation in range(6):
            x = (torch.randint(-16, 17, (1, heads, width), generator=g).float() / 16).bfloat16()
            w = (torch.randint(-16, 17, (heads * 32, 32), generator=g).float() / 16).bfloat16()
            if mutation == 5:
                x = torch.randn((1, heads, width), generator=g).bfloat16()
                w = torch.randn((heads * 32, 32), generator=g).bfloat16()
            shared[prefix + "x"].copy_(x.reshape(shared[prefix + "x"].shape))
            if fp8:
                w = w.to(torch.float8_e4m3fn)
                scale = (2.0 ** torch.randint(-3, 3, (heads * 4, 4), generator=g)).float()
                shared[prefix + "scale"].copy_(scale)
                expected_w = (w.float() * scale.repeat_interleave(8, 0).repeat_interleave(8, 1)).bfloat16()
            else:
                expected_w = w
            shared[prefix + "w"].copy_(w)
            for split, r in runtimes.items():
                outputs[split].fill_(-17)
                Run.run_into(r, *arguments[split])
            torch.cuda.synchronize()
            assert torch.equal(outputs[True].view(torch.uint8), outputs[False].view(torch.uint8))
            if mutation < 5:
                selected = expected_w.reshape(heads, 32, 32)
                selected = selected[:, :16] if direction == "absorb_query" else selected[:, 16:].transpose(1, 2)
                expected = torch.bmm(x.transpose(0, 1).float(), selected.float()).bfloat16().transpose(0, 1)
                assert torch.equal(outputs[True].cpu().reshape(expected.shape), expected)
            for storage, before, after in guards:
                assert torch.equal(storage[:16].view(torch.uint8), before) and torch.equal(
                    storage[-16:].view(torch.uint8), after
                )
    finally:
        for r in runtimes.values():
            r.close()
