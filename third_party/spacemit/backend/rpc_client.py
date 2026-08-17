"""Spine-Triton RPC Client for communicating with RISC-V RPC Server."""
import socket
import struct

PROTOCOL_MAGIC = 0x53545250
PROTOCOL_VERSION = 0x01

MSG_LOAD_KERNEL = 0x01
MSG_LOAD_ACK = 0x02
MSG_EXECUTE_KERNEL = 0x10
MSG_EXECUTE_RESULT = 0x11
MSG_ALLOC_MEMORY = 0x20
MSG_ALLOC_ACK = 0x21
MSG_FREE_MEMORY = 0x22
MSG_FREE_ACK = 0x23
MSG_WRITE_MEMORY = 0x24
MSG_WRITE_ACK = 0x25
MSG_READ_MEMORY = 0x26
MSG_READ_RESULT = 0x27
MSG_PING = 0xF0
MSG_PONG = 0xF1
MSG_ERROR = 0xFF

ARG_TYPE_I32 = 0x01
ARG_TYPE_I64 = 0x02
ARG_TYPE_F32 = 0x03
ARG_TYPE_F64 = 0x04
ARG_TYPE_PTR = 0x10

# KernelArg.flags (only meaningful for ARG_TYPE_PTR). ARG_FLAG_OUTPUT marks a
# pointer buffer whose contents must be read back from the kernel executable
# and written into device memory (required by the isolated-execution runtime).
ARG_FLAG_OUTPUT = 0x01

HEADER_FMT = '<IBBHIII'
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class SpineTritonRPCClient:

    def __init__(self, host: str, port: int, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.seq_id = 0
        # Set when the connection desyncs (timeout / bad magic / seq mismatch /
        # peer close). A broken connection carries unconsumed bytes, so it must
        # be discarded and rebuilt before the next request instead of reused
        # (reusing it cascades into bad-magic/seq-mismatch on every later op).
        self._broken = False

    def connect(self):
        self._close_socket()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Each connection is an independent stream; the server echoes the
        # request seq_id, so resetting keeps client and server in step.
        self.seq_id = 0
        self._broken = False

    def _close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def disconnect(self):
        self._close_socket()

    def _ensure_connected(self):
        """Rebuild the connection if it was never opened or got desynced."""
        if self.sock is None or self._broken:
            self.connect()

    def _mark_broken(self):
        self._broken = True
        self._close_socket()

    def _next_seq(self):
        self.seq_id += 1
        return self.seq_id

    def _send(self, msg_type: int, payload: bytes) -> int:
        seq_id = self._next_seq()
        hdr = struct.pack(HEADER_FMT, PROTOCOL_MAGIC, PROTOCOL_VERSION, msg_type, 0, seq_id, len(payload), 0)
        try:
            self.sock.sendall(hdr + payload)
        except OSError:
            self._mark_broken()
            raise
        return seq_id

    def _recv(self, expected_seq: int):
        try:
            hdr_data = self._recv_exact(HEADER_SIZE)
            magic, ver, msg_type, flags, seq_id, payload_size, checksum = \
                struct.unpack(HEADER_FMT, hdr_data)
            if magic != PROTOCOL_MAGIC:
                raise RuntimeError(f"bad magic: 0x{magic:08X}")
            if seq_id != expected_seq:
                raise RuntimeError(f"seq mismatch: got {seq_id}, want {expected_seq}")
            payload = self._recv_exact(payload_size) if payload_size > 0 else b''
        except (OSError, RuntimeError):
            # Socket error, timeout, or framing desync: the stream is no longer
            # trustworthy. Discard the connection so the next op reconnects.
            self._mark_broken()
            raise
        # A server-side MSG_ERROR is a well-framed, in-sync response; the
        # connection stays healthy, so surface it without marking broken.
        if msg_type == MSG_ERROR:
            raise RuntimeError(f"server error: {payload.decode('utf-8', errors='replace')}")
        return msg_type, payload

    def _recv_exact(self, size: int) -> bytes:
        data = b''
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise RuntimeError("connection closed")
            data += chunk
        return data

    def ping(self) -> bool:
        self._ensure_connected()
        seq = self._send(MSG_PING, b'')
        msg_type, _ = self._recv(seq)
        return msg_type == MSG_PONG

    def load_kernel(self, name: str, binary: bytes) -> int:
        self._ensure_connected()
        name_bytes = name.encode('utf-8')[:63].ljust(64, b'\x00')
        payload = name_bytes + struct.pack('<I', len(binary)) + binary
        seq = self._send(MSG_LOAD_KERNEL, payload)
        _, resp = self._recv(seq)
        handle, status = struct.unpack('<QI', resp[:12])
        if status != 0:
            raise RuntimeError("load_kernel failed")
        return handle

    def execute_kernel(self, handle: int, grid: tuple, args: list) -> int:
        self._ensure_connected()
        grid_x, grid_y, grid_z = grid
        args_data = b''
        for arg in args:
            if isinstance(arg, tuple):
                tag, value = arg[0], arg[1]
                # Optional 3rd element carries KernelArg.flags (e.g. output mark)
                flags = arg[2] if len(arg) > 2 else 0
                if tag == 'ptr':
                    args_data += struct.pack('<BBxxQ', ARG_TYPE_PTR, flags, int(value))
                elif tag == 'f32':
                    args_data += struct.pack('<Bxxxd', ARG_TYPE_F32, float(value))
                elif tag == 'f64':
                    args_data += struct.pack('<Bxxxd', ARG_TYPE_F64, float(value))
                elif tag == 'i64':
                    args_data += struct.pack('<Bxxxq', ARG_TYPE_I64, int(value))
                else:
                    args_data += struct.pack('<Bxxxq', ARG_TYPE_I32, int(value))
            elif isinstance(arg, float):
                args_data += struct.pack('<Bxxxd', ARG_TYPE_F64, arg)
            elif isinstance(arg, int):
                if arg > 0xFFFFFFFF or arg < 0:
                    args_data += struct.pack('<Bxxxq', ARG_TYPE_I64, arg)
                else:
                    args_data += struct.pack('<Bxxxq', ARG_TYPE_I32, arg)
            else:
                args_data += struct.pack('<BxxxQ', ARG_TYPE_PTR, int(arg))

        payload = struct.pack('<QIIII', handle, grid_x, grid_y, grid_z, len(args))
        payload += args_data
        seq = self._send(MSG_EXECUTE_KERNEL, payload)
        _, resp = self._recv(seq)
        status, exec_time = struct.unpack('<IQ', resp[:12])
        if status != 0:
            raise RuntimeError("execute_kernel failed")
        return exec_time

    def alloc_memory(self, size: int, alignment: int = 64) -> int:
        self._ensure_connected()
        payload = struct.pack('<QI', size, alignment)
        seq = self._send(MSG_ALLOC_MEMORY, payload)
        _, resp = self._recv(seq)
        addr, status = struct.unpack('<QI', resp[:12])
        if status != 0:
            raise RuntimeError("alloc_memory failed")
        return addr

    def free_memory(self, address: int):
        self._ensure_connected()
        payload = struct.pack('<Q', address)
        seq = self._send(MSG_FREE_MEMORY, payload)
        self._recv(seq)

    def write_memory(self, address: int, data: bytes):
        self._ensure_connected()
        payload = struct.pack('<QI', address, len(data)) + data
        seq = self._send(MSG_WRITE_MEMORY, payload)
        _, resp = self._recv(seq)
        status = struct.unpack('<I', resp[:4])[0]
        if status != 0:
            raise RuntimeError("write_memory failed")

    def read_memory(self, address: int, size: int) -> bytes:
        self._ensure_connected()
        payload = struct.pack('<QI', address, size)
        seq = self._send(MSG_READ_MEMORY, payload)
        _, resp = self._recv(seq)
        actual_size = struct.unpack('<I', resp[:4])[0]
        return resp[4:4 + actual_size]
