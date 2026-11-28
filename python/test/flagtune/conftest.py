def pytest_configure(config):
    config.addinivalue_line("markers", "remote_integration: external service tests requiring explicit opt-in")
