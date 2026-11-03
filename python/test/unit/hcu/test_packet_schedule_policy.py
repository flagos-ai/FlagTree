"""Finite scheduling combinations, preparation order, and unchanged-source paths."""
from itertools import product
from types import SimpleNamespace

import pytest

schedule = pytest.importorskip("triton.backends.hcu.packet_schedule")
compiler = pytest.importorskip("triton.backends.hcu.compiler")


def options(**overrides):
    values = dict(arch="gfx936", mma_phase_schedule=False, packet_wait_rotation=False,
                  packet_post_mma_wait=False, packet_operand_overlap_diagnostic=False,
                  packet_reuse_wait="none", packet_redundant_lgkm_cleanup=False,
                  coalesce_lgkm_waits=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("phase,rotation,post,overlap,reuse", product(
    (False, True), (False, True), (False, True), (False, True), ("none", "precopy")))
def test_wait_strategy_combinations(phase, rotation, post, overlap, reuse):
    value = options(mma_phase_schedule=phase, packet_wait_rotation=rotation,
                    packet_post_mma_wait=post, packet_operand_overlap_diagnostic=overlap,
                    packet_reuse_wait=reuse)
    valid = sum((rotation, post, overlap)) <= 1 and (reuse == "none" or overlap)
    if not valid:
        with pytest.raises(ValueError):
            schedule.PacketSchedulePolicy.from_options(value)
    else:
        policy = schedule.PacketSchedulePolicy.from_options(value)
        assert policy.wait == ("rotated" if rotation else "post_mma" if post else
                               "overlap_diagnostic" if overlap else "full")
        assert policy.phase is phase and policy.reuse == reuse


@pytest.mark.parametrize("name", ["mma_phase_schedule", "packet_wait_rotation", "packet_post_mma_wait",
                                 "packet_operand_overlap_diagnostic", "packet_redundant_lgkm_cleanup",
                                 "coalesce_lgkm_waits"])
@pytest.mark.parametrize("invalid", [0, 1, 2, None, "false"])
def test_policy_types(name, invalid):
    with pytest.raises(ValueError, match="must be boolean"):
        schedule.PacketSchedulePolicy.from_options(options(**{name: invalid}))


@pytest.mark.parametrize("name", ["packet_operand_overlap_diagnostic", "packet_redundant_lgkm_cleanup",
                                 "coalesce_lgkm_waits"])
def test_arch_contract(name):
    with pytest.raises(ValueError, match="requires gfx936"):
        schedule.PacketSchedulePolicy.from_options(options(arch="gfx942", **{name: True}))


@pytest.mark.parametrize("wait,phase,loops", product(
    ("full", "rotated", "post_mma", "overlap_diagnostic"), (False, True), (0, 2)))
def test_preparation_order_and_zero_loop_source(wait, phase, loops):
    calls = []

    def binding(name, overlap=False):
        def prepare(source, arch, threads, shared):
            calls.append((name, source, arch, threads, shared))
            result = (source + ":wait", loops)
            return (*result, 3, "reason") if overlap else (*result, "reason")
        return prepare

    def group(source, arch, warps):
        calls.append(("phase", source, arch, warps))
        return source + ":phase", loops

    llvm = SimpleNamespace(
        prepare_hcu_packet_wait_rotation=binding("rotated"),
        prepare_hcu_packet_post_mma_wait=binding("post_mma"),
        prepare_hcu_packet_operand_overlap_diagnostic=binding("overlap_diagnostic", True),
        prepare_hcu_mma_phase_loops=group)
    policy = schedule.PacketSchedulePolicy(wait, phase, "none", False, False)
    metadata = {"shared": 32768}
    source, prepared = schedule.prepare_packet_schedule("original", metadata, policy, llvm, "gfx936", 4)
    expected = "original" + (":wait" if wait != "full" and loops else "")
    if wait != "full":
        assert calls[0] == (wait, "original", "gfx936", 256, 32768)
        prefix = {"rotated": "hcu_packet_wait_rotation", "post_mma": "hcu_packet_post_mma_wait",
                  "overlap_diagnostic": "hcu_packet_operand_overlap"}[wait]
        assert metadata[prefix + "_loops"] == loops
        assert metadata[prefix + "_result"] == "reason"
        if wait == "overlap_diagnostic":
            assert metadata[prefix + "_groups"] == metadata[prefix + "_removed_waits"] == 3
            assert metadata[prefix + "_rotation_loops"] == loops
    if phase:
        assert calls[-1] == ("phase", expected, "gfx936", 4)
        if loops:
            expected += ":phase"
            assert metadata["hcu_mma_phase_loops"] == loops
        else:
            assert "hcu_mma_phase_loops" not in metadata
    assert source == expected
    assert prepared == bool(loops and (wait != "full" or phase))
    if wait == "full" and not phase:
        assert calls == [] and metadata == {"shared": 32768}


@pytest.mark.parametrize("value", [0, 1])
def test_sql_boolean_normalization(value):
    from triton.backends.compiler import GPUTarget
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    names = ("mma_phase_schedule", "packet_wait_rotation", "packet_redundant_lgkm_cleanup",
             "coalesce_lgkm_waits")
    parsed = backend.parse_options(dict.fromkeys(names, value))
    assert all(getattr(parsed, name) is bool(value) for name in names)


def test_conflict_rejected_before_compilation():
    with pytest.raises(ValueError, match="Choose one packet wait strategy"):
        compiler.HIPOptions(arch="gfx936", packet_wait_rotation=True, packet_post_mma_wait=True)


def test_only_selected_binding_is_required():
    llvm = SimpleNamespace(prepare_hcu_packet_post_mma_wait=lambda *args: ("prepared", 1, "ok"))
    policy = schedule.PacketSchedulePolicy.from_options(options(packet_post_mma_wait=True))
    assert schedule.prepare_packet_schedule("source", {}, policy, llvm, "gfx936", 4) == ("prepared", True)
