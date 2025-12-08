# content of conftest.py
import os
import pytest
import tempfile


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='gcu')


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            yield tmpdir
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = terminalreporter.stats.get("skipped", [])
    if skipped:
        terminalreporter.write_sep("=", "detailed skipped tests")
        for report in skipped:
            node_id = report.nodeid
            skip_info = ""

            if isinstance(report.longrepr, tuple):
                skip_info = report.longrepr[2] if len(report.longrepr) > 2 else str(report.longrepr)
            else:
                skip_info = str(report.longrepr).split("\n")[-1]

            terminalreporter.write_line(f"{node_id} - {skip_info}")
