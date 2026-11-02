import errno
import importlib.util
import logging
from pathlib import Path
import uuid
from unittest.mock import patch

import pytest


@pytest.fixture
def logger_module():
    source = Path(__file__).resolve().parents[1] / "backend" / "logger_config.py"
    spec = importlib.util.spec_from_file_location("tsingmicro_logger_config", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def logger_name():
    name = "tsingmicro-test-" + uuid.uuid4().hex
    yield name
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize("error_code", [errno.EROFS, errno.EACCES])
def test_unwritable_log_keeps_console(logger_module, logger_name, error_code, capsys, monkeypatch):
    monkeypatch.setenv("TX_LAUNCH_LOG_LEVEL", "info")
    with patch.object(logging, "FileHandler", side_effect=OSError(error_code, "read-only log")) as open_log:
        logger = logger_module.setup_logger(logger_name)
        logger.error("backend is available")
        assert logger_module.setup_logger(logger_name) is logger
    assert len(logger.handlers) == 1
    assert open_log.call_count == 1
    output = capsys.readouterr().out
    assert "Cannot open log file" in output
    assert "backend is available" in output


def test_writable_log_keeps_file_and_console(logger_module, logger_name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    logger = logger_module.setup_logger(logger_name)
    logger.error("file and console message")
    assert logger_module.setup_logger(logger_name) is logger
    assert len(logger.handlers) == 2
    assert "file and console message" in (tmp_path / f"{logger_name}.log").read_text(encoding="utf-8")
    assert "file and console message" in capsys.readouterr().out
