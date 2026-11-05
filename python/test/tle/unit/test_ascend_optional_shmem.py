"""Test Ascend host communication without requiring a device or a SHMEM install."""
import builtins
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[3] / "triton/experimental/tle/language/dsa/ascend/communication.py"
REAL_IMPORT = builtins.__import__


class Tensor:
    dtype = "float32"

    def numel(self):
        return 12


class CommunicationTests(unittest.TestCase):

    def setUp(self):
        self.torch = types.ModuleType("torch")
        self.torch.__path__ = []
        self.torch.Tensor = Tensor
        self.torch.npu = types.SimpleNamespace(set_device=Mock(), synchronize=Mock())
        self.dist = types.ModuleType("torch.distributed")
        for name in ("init_process_group", "get_rank", "get_world_size", "barrier"):
            setattr(self.dist, name, Mock())
        self.dist.get_rank.return_value = 0
        self.dist.get_world_size.return_value = 2
        self.torch.distributed = self.dist
        self.modules = patch.dict(sys.modules, {
            "torch": self.torch,
            "torch.distributed": self.dist,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.imports = []
        self.shmem_error = ModuleNotFoundError("No module named 'shmem'", name="shmem")
        self.ash = types.SimpleNamespace(
            InitAttr=lambda: types.SimpleNamespace(option_attr=types.SimpleNamespace()),
            OpEngineType=types.SimpleNamespace(MTE="MTE"),
            set_conf_store_tls=Mock(),
            aclshmem_init=Mock(return_value=0),
            aclshmem_create_tensor=Mock(side_effect=["buffer0", "buffer1"]),
            aclshmem_free_tensor=Mock(),
            aclshmem_finalize=Mock(),
        )

        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "shmem":
                self.imports.append(name)
                if self.shmem_error is not None:
                    raise self.shmem_error
                return self.ash
            return REAL_IMPORT(name, globals, locals, fromlist, level)

        self.guard = patch("builtins.__import__", side_effect=controlled_import)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def load(self):
        spec = importlib.util.spec_from_file_location("communication_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def assert_no_device_side_effects(self):
        self.torch.npu.set_device.assert_not_called()
        self.torch.npu.synchronize.assert_not_called()
        self.dist.init_process_group.assert_not_called()
        self.dist.barrier.assert_not_called()
        self.ash.aclshmem_init.assert_not_called()
        self.ash.aclshmem_create_tensor.assert_not_called()
        self.ash.aclshmem_free_tensor.assert_not_called()

    def test_import_without_shmem(self):
        module = self.load()
        self.assertEqual(self.imports, [])
        self.assertTrue(callable(module.init_communicator))
        self.assertTrue(callable(module.create_dist_tensor))
        self.assertTrue(callable(module.cleanup_communicator))
        self.assert_no_device_side_effects()

    def test_missing_shmem_fails_before_side_effects(self):
        for api, args in [
            ("init_communicator", ()),
            ("create_dist_tensor", (Tensor(), )),
            ("cleanup_communicator", ("buffer", )),
        ]:
            with self.subTest(api=api):
                module = self.load()
                with self.assertRaises(ModuleNotFoundError) as ctx:
                    getattr(module, api)(*args)
                self.assertEqual(ctx.exception.name, "shmem")
                self.assertIn("Ascend TLE distributed communication", str(ctx.exception))
                self.assertIs(ctx.exception.__cause__, self.shmem_error)
                self.assertEqual(module._g_mem_count, 0)
                self.assertIsNone(module._g_pe)
                self.assert_no_device_side_effects()

    def test_transitive_missing_dependency_preserved(self):
        module = self.load()
        self.shmem_error = ModuleNotFoundError("No module named 'vendor_runtime'", name="vendor_runtime")
        with self.assertRaises(ModuleNotFoundError) as ctx:
            module.init_communicator()
        self.assertIs(ctx.exception, self.shmem_error)
        self.assert_no_device_side_effects()

    def test_import_error_preserved(self):
        module = self.load()
        self.shmem_error = ImportError("libshmem.so: undefined symbol")
        with self.assertRaises(ImportError) as ctx:
            module.init_communicator()
        self.assertIs(ctx.exception, self.shmem_error)
        self.assert_no_device_side_effects()

    def test_os_error_preserved(self):
        module = self.load()
        self.shmem_error = OSError("vendor library cannot be loaded")
        with self.assertRaises(OSError) as ctx:
            module.init_communicator()
        self.assertIs(ctx.exception, self.shmem_error)
        self.assert_no_device_side_effects()

    def test_installed_dependency_is_still_lazy(self):
        self.shmem_error = None
        self.load()
        self.assertEqual(self.imports, [])

    def test_dependency_can_become_available_after_failed_call(self):
        module = self.load()
        with self.assertRaises(ModuleNotFoundError):
            module.init_communicator()
        self.shmem_error = None
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            module.init_communicator()
        self.ash.aclshmem_init.assert_called_once()

    def test_default_init_contract(self):
        self.shmem_error = None
        module = self.load()
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=True):
            module.init_communicator()
        self.torch.npu.set_device.assert_called_once_with(0)
        self.dist.init_process_group.assert_called_once_with(backend="hccl", rank=0)
        self.dist.barrier.assert_called_once_with()
        self.ash.set_conf_store_tls.assert_called_once_with(False, "")
        attr = self.ash.aclshmem_init.call_args.args[0]
        self.assertEqual(attr.my_rank, 0)
        self.assertEqual(attr.n_ranks, 2)
        self.assertEqual(attr.local_mem_size, 1024**3)
        self.assertEqual(attr.ip_port, "tcp://127.0.0.1:8666")
        self.assertEqual(attr.option_attr.data_op_engine_type, "MTE")
        self.assertEqual(module._g_mem_count, 0)

    def test_explicit_init_options(self):
        self.shmem_error = None
        module = self.load()
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            module.init_communicator("tcp://test-host:1234", 4096, "custom-engine")
        attr = self.ash.aclshmem_init.call_args.args[0]
        self.assertEqual(attr.ip_port, "tcp://test-host:1234")
        self.assertEqual(attr.local_mem_size, 4096)
        self.assertEqual(attr.option_attr.data_op_engine_type, "custom-engine")

    def test_environment_address(self):
        module = self.load()
        with patch.dict(os.environ, {"ASH_MASTER_ADDR": "test-host", "ASH_MASTER_PORT": "1234"}):
            self.assertEqual(module._get_ash_ip_port(), "tcp://test-host:1234")
        self.assertEqual(self.imports, [])

    def test_initialization_failure_still_raises(self):
        self.shmem_error = None
        self.ash.aclshmem_init.return_value = 1
        module = self.load()
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            with self.assertRaisesRegex(ValueError, "aclshmem_init failed"):
                module.init_communicator()

    def test_two_allocations_finalize_after_last_free(self):
        self.shmem_error = None
        module = self.load()
        module._g_pe = 0
        first = module.create_dist_tensor(Tensor())
        second = module.create_dist_tensor(Tensor())
        self.assertEqual(module._g_mem_count, 2)
        self.ash.aclshmem_create_tensor.assert_called_with([12], dtype="float32", device_id=0)
        module.cleanup_communicator(first)
        self.assertEqual(module._g_mem_count, 1)
        self.ash.aclshmem_finalize.assert_not_called()
        module.cleanup_communicator(second)
        self.assertEqual(module._g_mem_count, 0)
        self.ash.aclshmem_finalize.assert_called_once_with()
        self.assertEqual(self.torch.npu.synchronize.call_count, 2)
        self.assertEqual(self.dist.barrier.call_count, 2)

    def test_failed_allocation_does_not_increment_count(self):
        self.shmem_error = None
        self.ash.aclshmem_create_tensor.side_effect = RuntimeError("allocation failed")
        module = self.load()
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            module.create_dist_tensor(Tensor())
        self.assertEqual(module._g_mem_count, 0)

    def test_invalid_tensor_still_rejected(self):
        self.shmem_error = None
        module = self.load()
        with self.assertRaisesRegex(ValueError, "buf_tensor must be a tensor"):
            module.create_dist_tensor(object())
        self.ash.aclshmem_create_tensor.assert_not_called()
        self.assertEqual(module._g_mem_count, 0)

    def test_double_cleanup_still_rejected(self):
        self.shmem_error = None
        module = self.load()
        module._g_mem_count = 0
        with self.assertRaisesRegex(ValueError, "more times than"):
            module.cleanup_communicator("buffer")
        self.ash.aclshmem_finalize.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
