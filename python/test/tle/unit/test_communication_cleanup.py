# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A communicator can be closed before its symmetric window is created."""

from types import SimpleNamespace

import pytest

from triton.experimental.tle.language import communication


@pytest.mark.parametrize("window_created", [False, True])
def test_cleanup_frees_only_created_resources_and_preserves_caller_group(monkeypatch, window_created):
    calls = []

    class FlagCX:
        def __getattr__(self, name):
            return lambda *args: calls.append((name, args))

    comm, group = object(), object()
    mem, dev_comm, win = (object(), object(), object()) if window_created else (None, None, None)
    for name, value in {
        "_init_communicator_": True, "_owns_process_group": False, "_sync_group": group,
        "comm": comm, "rank": 0, "dev_mem": mem, "dev_comm": dev_comm, "win": win, "flagcx": FlagCX(),
    }.items():
        monkeypatch.setattr(communication, name, value, raising=False)
    monkeypatch.setattr(communication, "_distributed_barrier", lambda: None)
    monkeypatch.setattr(communication, "dist", SimpleNamespace(
        is_initialized=lambda: True,
        destroy_process_group=lambda *args: calls.append(("destroy_group", args)),
    ))

    communication.cleanup_communicator()
    expected = ([] if not window_created else [
        ("flagcxDevMemFreeDevicePtr", (mem,)), ("flagcxDevCommFreeDevicePtr", (dev_comm,)),
        ("flagcxDevMemDestroy", (comm, mem)), ("flagcxDevCommDestroy", (comm, dev_comm)),
        ("flagcxCommWindowDeregister", (comm, win)),
    ]) + [("flagcxCommDestroy", (comm,)), ("destroy_group", (group,))]
    assert calls == expected
    communication.cleanup_communicator()
    assert calls == expected
