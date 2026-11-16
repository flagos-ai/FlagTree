"""CPU metadata fixtures replay actual Torch UUID text; no native/GPU objects.

FlagTree Torch2.10/cu128 raw9ee5d074 and SG Torch2.11/cu130 rawb0d4f1e7
captured torch._C._CUuuid.__str__ as canonical UUID text without GPU-prefix.
CapturedCUuuid models that provider protocol explicitly; it is not a CUDA object.
"""
import importlib.abc
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','triton','vllm','sglang','pynvml','cuda','cupy'}:
            raise AssertionError('CPU UUID test attempted GPU/model import: '+fullname)
sys.meta_path.insert(0,NoGPUImports())
def no_driver_load(event,args):
    if event=='ctypes.dlopen':raise AssertionError('CPU UUID test attempted library load')
sys.addaudithook(no_driver_load)
import device_identity as policy

CAPTURED = [
    ('FlagTree2.10_GPU3','5b5e235a-09a2-fe8b-7754-916b7eb85976',3,'00000000:44:00.0','0000:44:00.0','nvml-index','3'),
    ('SGLang2.11_GPU0','73c1a37e-d3c0-dc24-b81d-6a5563ebe677',0,'00000000:0F:00.0','0000:0F:00.0','uuid','GPU-73c1a37e-d3c0-dc24-b81d-6a5563ebe677'),
]
class CapturedCUuuid:
    def __init__(self,text):self.text=text
    def __str__(self):return self.text

def provider(value,registered=True):
    calls=[]
    def properties(index):calls.append(index);return SimpleNamespace(uuid=value)
    cuda=SimpleNamespace(device_count=lambda:1,current_device=lambda:0,get_device_properties=properties)
    result=SimpleNamespace(cuda=cuda,calls=calls)
    if registered:result._C=SimpleNamespace(_CUuuid=CapturedCUuuid)
    return result

def metadata(row):
    name,text,index,nvml_pci,cuda_pci,kind,selector=row
    full='GPU-'+text
    return {'backend':'vllm' if kind=='nvml-index' else 'sglang','physical_gpu_uuid':full,
        'nvml_index':index,'pci_bus_id':nvml_pci,'selector_kind':kind,'cuda_visible_device':selector,
        'CUDA_DEVICE_ORDER':'PCI_BUS_ID','cuda':{'visible_count':1,'logical_device':0,'compute_capability':[9,0],
        'uuid':full,'pci_bus_id':cuda_pci}}

def verify(value,row=CAPTURED[0],registered=True):
    identity=metadata(row);torch=provider(value,registered)
    with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=identity['cuda_visible_device'],CUDA_DEVICE_ORDER='PCI_BUS_ID'):
        return policy.verify_torch_identity(torch,identity)

class TorchUUIDIdentity(unittest.TestCase):
    def test_actual_captured_trusted_carriers_match_physical_uuid(self):
        for row in CAPTURED:
            with self.subTest(capture=row[0]):
                result=verify(CapturedCUuuid(row[1]),row)
                self.assertEqual(result['uuid'],'GPU-'+row[1])
                self.assertEqual(uuid.UUID(result['uuid'][4:]).bytes,uuid.UUID(row[1]).bytes)
                self.assertEqual((result['visible_count'],result['current_device']),(1,0))

    def test_old_GPU_prefixed_text_and_bytes_still_match_without_native_type(self):
        for row in CAPTURED:
            for value in ('GPU-'+row[1],('GPU-'+row[1]).encode()):
                self.assertEqual(verify(value,row,registered=False)['uuid'],'GPU-'+row[1])

    def test_trusted_carrier_canonical_case_and_all_128_bits(self):
        torch=provider(None)
        values=[CAPTURED[0][1],CAPTURED[1][1],'00000000-0000-0000-0000-000000000001',
            '80000000-0000-0000-0000-000000000000','ffffffff-ffff-ffff-ffff-ffffffffffff']
        for text in values:
            for rendering in (text,text.upper(),'GPU-'+text):
                out=policy.normalize_torch_uuid(CapturedCUuuid(rendering),torch)
                self.assertEqual(uuid.UUID(out[4:]).int,uuid.UUID(text).int)
                self.assertEqual(out,'GPU-'+str(uuid.UUID(text)))

    def test_wrong_actual_GPU_and_single_bit_changes_are_rejected(self):
        for row in CAPTURED:
            for bit in (0,63,127):
                different=str(uuid.UUID(int=uuid.UUID(row[1]).int^(1<<bit)))
                with self.assertRaisesRegex(ValueError,'Torch CUDA0 UUID differs'):
                    verify(CapturedCUuuid(different),row)

    def test_MIG_alias_partial_or_malformed_trusted_carrier_rejected(self):
        for text in ('MIG-GPU-'+CAPTURED[0][1]+'/1/0','GPU-0','0','cuda:0','GPU-'+CAPTURED[0][1]+'extra',
                     CAPTURED[0][1].replace('-',''),' '+CAPTURED[0][1],CAPTURED[0][1]+'\n','', 'z'*36):
            with self.subTest(value=text),self.assertRaises(ValueError):verify(CapturedCUuuid(text))

    def test_unsupported_or_unregistered_carriers_and_bare_text_rejected(self):
        class Foreign:
            def __str__(self):return CAPTURED[0][1]
        class Subtype(CapturedCUuuid):pass
        for value in (None,True,3,uuid.UUID(CAPTURED[0][1]),Foreign(),Subtype(CAPTURED[0][1]),
                      CAPTURED[0][1],CAPTURED[0][1].encode()):
            with self.subTest(value=type(value).__name__),self.assertRaises(ValueError):verify(value)
        with self.assertRaises(ValueError):verify(CapturedCUuuid(CAPTURED[0][1]),registered=False)

    def test_type_name_spoof_is_not_trusted_class_identity(self):
        Spoof=type('_CUuuid',(),{'__module__':'torch._C','__str__':lambda self:CAPTURED[0][1]})
        with self.assertRaises(ValueError):verify(Spoof())

    def test_global_NVML_and_cli_parsers_remain_strict(self):
        for row in CAPTURED:
            self.assertEqual(policy.normalize_uuid('GPU-'+row[1]),'GPU-'+row[1])
            for text in (row[1],'MIG-GPU-'+row[1]+'/1/0','GPU-0','cuda:0'):
                with self.assertRaises(ValueError):policy.normalize_uuid(text)
            with self.assertRaises(ValueError):policy.selector_policy('sglang','GPU-'+row[1],'uuid',row[1])
        with self.assertRaises(ValueError):policy.selector_policy('vllm','GPU-'+CAPTURED[0][1],'uuid','GPU-'+CAPTURED[0][1])

    def test_driver_identity_mismatch_is_rejected_before_Torch_read(self):
        row=CAPTURED[0];torch=provider(CapturedCUuuid(row[1]));identity=metadata(row)
        identity['cuda']['uuid']='GPU-'+CAPTURED[1][1]
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID'):
            with self.assertRaisesRegex(ValueError,'logical CUDA0 UUID differs'):policy.verify_torch_identity(torch,identity)
        self.assertEqual(torch.calls,[])

    def test_pci_visibility_and_logical_device_guards_stay_active(self):
        self.assertEqual(policy.normalize_pci(CAPTURED[0][3]),policy.normalize_pci(CAPTURED[0][4]))
        for case in ('PCI','count','logical','SM','environment'):
            identity=metadata(CAPTURED[0]);torch=provider(CapturedCUuuid(CAPTURED[0][1]))
            if case=='PCI':identity['cuda']['pci_bus_id']='0000:01:00.0'
            if case=='count':identity['cuda']['visible_count']=2
            if case=='logical':identity['cuda']['logical_device']=1
            if case=='SM':identity['cuda']['compute_capability']=[8,0]
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='2' if case=='environment' else '3',CUDA_DEVICE_ORDER='PCI_BUS_ID'):
                with self.assertRaises(ValueError):policy.verify_torch_identity(torch,identity)
            self.assertEqual(torch.calls,[])

if __name__=='__main__':unittest.main()
