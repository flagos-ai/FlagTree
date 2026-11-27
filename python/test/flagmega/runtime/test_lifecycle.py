# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import RuntimeModule, RuntimeState


class _Module(RuntimeModule):
    def __init__(self):
        super().__init__()
        self.release_count = 0

    def load(self, device="cpu"):
        self._mark_loaded(device)
        return self

    def prepare(self):
        self._mark_prepared()
        return self

    def run(self):
        self._require_prepared()
        return "ran"

    def _release_resources(self):
        self.release_count += 1


def test_runtime_lifecycle_is_explicit_and_close_is_idempotent():
    module = _Module()
    assert module.runtime_state is RuntimeState.CREATED
    with pytest.raises(RuntimeContractError, match="must be prepared"):
        module.run()

    module.load().prepare()
    assert module.runtime_state is RuntimeState.PREPARED
    assert module.run() == "ran"
    module.close()
    module.close()
    assert module.runtime_state is RuntimeState.CLOSED
    assert module.device is None
    assert module.release_count == 1


def test_runtime_module_cannot_reload_or_reopen_after_close():
    module = _Module().load()
    with pytest.raises(RuntimeContractError, match="only load from created"):
        module.load()
    module.close()
    with pytest.raises(RuntimeContractError, match="only load from created"):
        module.load()
    with pytest.raises(RuntimeContractError, match="cannot be re-entered"):
        module.__enter__()


def test_runtime_context_manager_uninitializes_resources():
    module = _Module().load()
    with module as active:
        assert active is module
        active.prepare()
    assert module.runtime_state is RuntimeState.CLOSED
    assert module.release_count == 1
