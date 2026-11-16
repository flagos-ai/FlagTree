"""Birth-bound Linux process-session ownership; no CUDA or NVML imports.

The scan and pidfd revalidation follow the frozen whole-SID controller45665844.
The spawning monitor additionally adopts orphan descendants so it can reap them
after a payload leader exits. No foreign session or recycled PID is signalled.
"""
from pathlib import Path
import os
import signal
import time


def pidfd_library():
    # This Python build omits os.pidfd_open. Use the proven glibc interface,
    # retaining the same kernel pidfd safety; there is no PID-only fallback.
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, 'pidfd_open') or not hasattr(library, 'pidfd_send_signal'):
        raise RuntimeError('Linux/glibc pidfd API is required')
    library.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    library.pidfd_open.restype = ctypes.c_int
    library.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    library.pidfd_send_signal.restype = ctypes.c_int
    return library


def pidfd_open(pid):
    import ctypes
    result = pidfd_library().pidfd_open(pid, 0)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def pidfd_signal(descriptor, number):
    import ctypes
    if pidfd_library().pidfd_send_signal(descriptor, number, None, 0) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'pgid': int(fields[2]), 'sid': int(fields[3]), 'start_ticks': int(fields[19])}


def scan(sid):
    members = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            row = identity(int(entry.name))
            if row is not None and row['sid'] == sid:
                members.append(row)
    return sorted(members, key=lambda row: row['pid'])


class Subreaper:
    """Use libc process control only; restore the caller's previous setting."""
    def __enter__(self):
        import ctypes
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.prctl.restype = ctypes.c_int
        self.previous = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(self.previous), 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER failed')
        if self.libc.prctl(36, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
        self.entered = True
        return self

    def __exit__(self, *args):
        if getattr(self, 'entered', False):
            self.entered = False
            if self.libc.prctl(36, self.previous.value, 0, 0, 0) < 0:
                import ctypes
                raise OSError(ctypes.get_errno(), 'restore child subreaper failed')


class OwnedSession:
    def __init__(self, child):
        self.child = child
        self.anchor = identity(child.pid)
        if (self.anchor is None or self.anchor['sid'] != child.pid or
                self.anchor['pgid'] != child.pid or self.anchor['sid'] == os.getsid(0)):
            raise ValueError('new child session birth could not be anchored')
        self.observations = []
        self.signals = []
        self.reaped = []
        self.empty_scans = []
        self.cleanup_errors = []
        self.payload_exit_code = None

    def members(self):
        leader = identity(self.anchor['pid'])
        if leader is not None and any(leader[key] != self.anchor[key] for key in ('pid', 'sid', 'start_ticks')):
            raise ValueError('owned leader birth/SID changed; refuse reused PID')
        rows = scan(self.anchor['sid'])
        if any(row['start_ticks'] < self.anchor['start_ticks'] for row in rows):
            raise ValueError('session member predates anchored leader birth')
        return rows

    def observe(self):
        # Popen owns its direct child's wait status; never reap it independently.
        code = self.child.poll()
        if code is not None:
            self.payload_exit_code = code
        rows = self.members()
        for row in rows:
            if row['pid'] == self.child.pid or row['state'] != 'Z':
                continue
            try:
                pid, status = os.waitpid(row['pid'], os.WNOHANG)
            except ChildProcessError:
                continue  # A live owned parent still owns this zombie's wait.
            if pid:
                self.reaped.append({'birth': row, 'wait_status': status})
        rows = self.members()
        event = {'monotonic_ns': time.monotonic_ns(), 'members': rows,
                 'payload_exit_code': self.payload_exit_code}
        self.observations.append(event)
        if rows:
            self.empty_scans.clear()
        elif self.payload_exit_code is not None:
            self.empty_scans.append(event)
        return rows

    @property
    def terminal_qualified(self):
        return self.payload_exit_code is not None and len(self.empty_scans) >= 2

    def send_owned(self, row, number):
        if row['sid'] != self.anchor['sid'] or row['start_ticks'] < self.anchor['start_ticks']:
            raise ValueError('foreign SID/birth rejected before signal')
        try:
            descriptor = pidfd_open(row['pid'])
        except ProcessLookupError:
            return None
        try:
            current = identity(row['pid'])
            if current is None:
                return None
            if any(current[key] != row[key] for key in ('pid', 'sid', 'pgid', 'start_ticks')):
                raise ValueError('candidate birth/SID/PGID changed; refuse signal')
            if current['state'] == 'Z':
                return None
            pidfd_signal(descriptor, number)
            event = {'target_birth': current, 'signal': number,
                     'monotonic_ns': time.monotonic_ns(), 'pidfd_used': True}
            self.signals.append(event)
            return event
        except (FileNotFoundError, ProcessLookupError):
            return None
        finally:
            os.close(descriptor)

    def cleanup(self, timeout_seconds=8, poll_seconds=.05):
        """Repeated birth-safe waves, preserving parents briefly to reap workers."""
        start = time.monotonic()
        while time.monotonic() - start <= timeout_seconds:
            try:
                rows = self.observe()
                if self.terminal_qualified:
                    return self.report()
                elapsed = time.monotonic() - start
                separate = [row for row in rows if row['pgid'] != self.anchor['sid']]
                candidates = separate if elapsed < .5 and separate else rows
                number = signal.SIGTERM if elapsed < 2 else signal.SIGKILL
                for row in reversed(candidates):
                    self.send_owned(row, number)
            except (OSError, ValueError) as error:
                self.cleanup_errors.append(f'{type(error).__name__}: {error}')
                break  # A changed ownership boundary never permits a signal fallback.
            time.sleep(poll_seconds)
        return self.report()

    def report(self):
        try:
            remaining = self.members()
        except (OSError, ValueError) as error:
            remaining = None
            self.cleanup_errors.append(f'{type(error).__name__}: {error}')
        return {'anchor': self.anchor, 'payload_exit_code': self.payload_exit_code,
                'terminal_qualified': self.terminal_qualified and remaining == [] and not self.cleanup_errors,
                'all_owned_SID_members_absent': remaining == [], 'remaining_owned_members': remaining,
                'observations': self.observations, 'final_empty_scans': self.empty_scans[-2:],
                'signals': self.signals, 'reaped_descendants': self.reaped,
                'cleanup_errors': self.cleanup_errors}
