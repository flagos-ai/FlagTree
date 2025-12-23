[English](../en/tests.md)

## 运行测试

安装完成后可以在设备支持的环境下运行测试。对于 NVIDIA 后端，可以执行：

```shell
cd python/test/unit
python3 -m pytest -s
```

对于其他后端，可以执行：

```shell
cd third_party/<backend>/python/test/unit
python3 -m pytest -s
```
