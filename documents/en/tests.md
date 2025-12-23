[中文](../zh-cn/tests.md)

## Running tests

After installation, you can run tests. For NVIDIA, the command is:

```shell
cd python/test/unit
python3 -m pytest -s
```

For other backends, the command is:

```shell
cd third_party/<backend>/python/test/unit
python3 -m pytest -s
```
