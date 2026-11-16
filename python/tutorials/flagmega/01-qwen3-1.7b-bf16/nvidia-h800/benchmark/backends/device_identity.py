"""Benchmark selector policy and GPU-phase identity checks. Import is CPU-only.

CUDA/NVML identity queries are separate from policy validation. No query runs at
import, source preflight or planning. Numeric visibility is a pinned vLLM493
interface; the leased physical UUID remains the semantic identity for all modes.
"""
from __future__ import annotations
import ast
import inspect
import json
import os
from pathlib import Path
import re
import uuid as uuid_module


def normalize_uuid(value):
    value = value.decode() if isinstance(value, bytes) else str(value)
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        raise ValueError('Require one full GPU UUID; MIG/aliases are outside this contract')
    return 'GPU-'+str(uuid_module.UUID(value[4:]))


def normalize_torch_uuid(value, torch):
    """Decode the trusted Torch metadata carrier without relaxing NVML text.

    Both pinned Torch versions export _CUuuid whose __str__ returns a complete
    bare UUID. Type identity is checked against that live provider; arbitrary
    objects or bare config strings do not acquire physical GPU identity here.
    """
    if isinstance(value, (str, bytes)):
        return normalize_uuid(value)  # Preserve existing full GPU-prefixed text.
    carrier_type = getattr(getattr(torch, '_C', None), '_CUuuid', None)
    if not isinstance(carrier_type, type) or type(value) is not carrier_type:
        raise ValueError('Require trusted Torch UUID carrier or full GPU UUID text')
    text = str(value)
    if text.startswith('GPU-'):
        return normalize_uuid(text)
    if not re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', text):
        raise ValueError('Require one complete canonical Torch UUID')
    return normalize_uuid('GPU-' + text)


def normalize_pci(value):
    value = value.decode() if isinstance(value, bytes) else str(value)
    if not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', value):
        raise ValueError('Invalid actual PCI bus identity')
    domain, bus, device = value.split(':')
    return f'{int(domain,16):08x}:{int(bus,16):02x}:{device.lower()}'


def selector_policy(backend, physical_uuid, kind, selector):
    expected = 'nvml-index' if backend == 'vllm' else 'uuid'
    if kind != expected:
        raise ValueError(f'{backend} requires predeclared {expected} selector policy')
    if physical_uuid is not None: physical_uuid = normalize_uuid(physical_uuid)
    if selector is not None:
        if kind == 'uuid':
            if physical_uuid is None or normalize_uuid(selector) != physical_uuid:
                raise ValueError('UUID selector disagrees with leased physical UUID')
        elif not re.fullmatch(r'0|[1-9][0-9]*', selector):
            raise ValueError('Pinned native numeric selector must be one nonnegative decimal index')
    return {'backend': backend, 'physical_gpu_uuid': physical_uuid,
            'selector_kind': kind, 'cuda_visible_device': selector,
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'actual_identity': 'pending GPU-phase queries'}


def validate_visibility(identity, environment=None):
    env = os.environ if environment is None else environment
    backend = identity.get('backend', 'vllm' if identity['selector_kind'] == 'nvml-index' else 'sglang')
    policy = selector_policy(backend, identity['physical_gpu_uuid'], identity['selector_kind'],
                             identity['cuda_visible_device'])
    if policy['physical_gpu_uuid'] is None or policy['cuda_visible_device'] is None:
        raise ValueError('Runtime identity requires both leased UUID and explicit selector')
    if env.get('CUDA_VISIBLE_DEVICES') != policy['cuda_visible_device'] or env.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('Actual process CUDA visibility/order differs from declared selector')
    if identity['selector_kind'] == 'nvml-index' and int(identity['cuda_visible_device']) != identity['nvml_index']:
        raise ValueError('Declared numeric selector differs from leased NVML index')
    if not isinstance(identity['nvml_index'], int) or identity['nvml_index'] < 0:
        raise ValueError('Missing actual nonnegative NVML index')
    normalize_pci(identity['pci_bus_id'])
    return policy


def validate_cuda_identity(identity, cuda):
    if (cuda['visible_count'], cuda['logical_device'], cuda['compute_capability']) != (1, 0, [9, 0]):
        raise ValueError('Require one logical CUDA0 SM90 full GPU')
    if normalize_uuid(cuda['uuid']) != normalize_uuid(identity['physical_gpu_uuid']):
        raise ValueError('Actual logical CUDA0 UUID differs from leased NVML UUID')
    if normalize_pci(cuda['pci_bus_id']) != normalize_pci(identity['pci_bus_id']):
        raise ValueError('Actual logical CUDA0 PCI differs from leased NVML PCI')


def query_cuda_identity():
    """GPU-phase cuInit/device queries only; never create/retain a CUDA context."""
    import ctypes
    import ctypes.util
    library = ctypes.util.find_library('cuda') or 'libcuda.so.1'
    driver = ctypes.CDLL(library)
    codes = {}
    def call(name, argtypes, *args):
        fn = getattr(driver, name); fn.argtypes = argtypes; fn.restype = ctypes.c_int
        result = fn(*args); codes[name] = result
        if result: raise RuntimeError(f'CUDA identity API {name} returned {result}')
    call('cuInit', [ctypes.c_uint], 0)
    count, device, major, minor, version = (ctypes.c_int() for _ in range(5))
    call('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if count.value != 1: raise ValueError('Identity query requires exactly one visible GPU')
    call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
    raw_uuid = (ctypes.c_ubyte*16)(); pci = ctypes.create_string_buffer(32)
    call('cuDeviceGetUuid_v2', [ctypes.c_void_p, ctypes.c_int], raw_uuid, device.value)
    call('cuDeviceGetPCIBusId', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], pci, 32, device.value)
    call('cuDeviceComputeCapability', [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int],
         ctypes.byref(major), ctypes.byref(minor), device.value)
    call('cuDriverGetVersion', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(version))
    maps = [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if 'libcuda.so' in line and '/' in line]
    return {'uuid': 'GPU-'+str(uuid_module.UUID(bytes=bytes(raw_uuid))),
            'pci_bus_id': pci.value.decode(), 'logical_device': 0, 'visible_count': count.value,
            'compute_capability': [major.value, minor.value], 'driver_version': version.value,
            'requested_library': library, 'loaded_library_paths': sorted(set(maps)),
            'api_return_codes': codes, 'context_created': False}


def prepare_identity(args, backend):
    """Invoked after CPU preflight returns, before backend/Torch import."""
    kind = args.gpu_selector_kind
    selector = args.cuda_visible_device or (args.physical_gpu_uuid if kind == 'uuid' else None)
    policy = selector_policy(backend, args.physical_gpu_uuid, kind, selector)
    path = args.device_identity_proof or os.environ.get('FLAGMEGA_DEVICE_IDENTITY_PROOF')
    if not path: raise ValueError('GPU runtime requires actual launcher NVML identity proof')
    identity = json.loads(Path(path).read_text())
    if identity.get('schema') != 'flagmega.device-identity/v3' or identity.get('evidence_kind') != 'actual-NVML-query':
        raise ValueError('GPU identity proof must be an actual versioned NVML query receipt')
    for key in ('physical_gpu_uuid', 'selector_kind', 'cuda_visible_device', 'backend'):
        if identity.get(key) != policy[key]: raise ValueError('Launcher/child identity disagreement: '+key)
    validate_visibility(identity)
    cuda = query_cuda_identity(); validate_cuda_identity(identity, cuda)
    import hashlib
    return {**identity, 'cuda': cuda, 'proof_path': str(Path(path).resolve()),
            'proof_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def verify_torch_identity(torch, identity):
    validate_visibility(identity); validate_cuda_identity(identity, identity['cuda'])
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise ValueError('Torch must retain the sole logical CUDA0 device')
    props = torch.cuda.get_device_properties(0)
    torch_uuid = normalize_torch_uuid(props.uuid, torch)
    if torch_uuid != normalize_uuid(identity['physical_gpu_uuid']):
        raise ValueError('Actual Torch CUDA0 UUID differs from leased GPU')
    return {'visible_count': 1, 'current_device': 0, 'uuid': torch_uuid}


def verify_completion_identity(torch, identity):
    import hashlib
    if hashlib.sha256(Path(identity['proof_path']).read_bytes()).hexdigest() != identity['proof_sha256']:
        raise ValueError('Actual launcher identity proof changed during the cohort')
    result = verify_torch_identity(torch, identity)
    cuda = query_cuda_identity(); validate_cuda_identity(identity, cuda)
    return {'torch': result, 'cuda': cuda}


def verify_tensor_devices(tensors):
    if not tensors: raise ValueError('No actual model/KV tensors to verify')
    devices = [str(t.device) for t in tensors]
    if any(d != 'cuda:0' for d in devices): raise ValueError('Native model/KV tensors must reside on CUDA0')
    return {'tensor_count': len(devices), 'devices': sorted(set(devices))}


def source_contract(path, class_name, methods=(), attributes=(), calls=(), fields=()):
    """Link required adapter protocol to actual parsed native class/method AST."""
    tree = ast.parse(Path(path).read_text()); cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None: raise ValueError('Native class missing: '+class_name)
    selected = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    if {n.name for n in selected} != set(methods): raise ValueError('Native methods missing: '+class_name)
    nodes = [n for node in (selected or [cls]) for n in ast.walk(node)]
    attrs = {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    invoked = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
               for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
    annotations = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    if not set(attributes) <= attrs or not set(calls) <= invoked or not set(fields) <= annotations:
        raise ValueError('Native AST attribute/call/returned-field contract changed: '+class_name)
    return {'class': class_name, 'methods': list(methods), 'attributes': list(attributes), 'calls': list(calls), 'fields': list(fields)}


def loaded_origin(cls, binding):
    path = Path(inspect.getfile(cls)).resolve()
    matches = [p for p in binding['files'] if Path(p['path']).resolve() == path]
    if len(matches) != 1: raise ValueError('Loaded native class is outside frozen CPU source manifest: '+str(path))
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != matches[0]['sha256']:
        raise ValueError('Loaded native class source changed: '+str(path))
    return {'class': cls.__module__+'.'+cls.__qualname__, 'path': str(path), 'sha256': matches[0]['sha256']}
