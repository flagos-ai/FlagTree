# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional
from triton._C.libproton import proton as libproton  # type: ignore
import json as json
from .flags import flags


def get(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
    Returns:
        str: The profiling data in JSON format.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return json.loads(libproton.get_data(session, phase))


def get_msgpack(session: Optional[int] = 0, phase: int = 0):
    """
    Retrieves profiling data for a given session encoded with MessagePack.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        bytes: The profiling data encoded with MessagePack.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be retrieved when running from the command line.")
    return libproton.get_data_msgpack(session, phase)


def advance_phase(session: Optional[int] = 0) -> Optional[int]:
    """
    Advances the profiling phase for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.

    Returns:
        Optional[int]: The next phase number after advancing.
    """
    if session is None:
        return None
    if flags.command_line and session != 0:
        raise ValueError("Only one session can advance phase when running from the command line.")
    return libproton.advance_data_phase(session)


def is_phase_complete(session: Optional[int] = 0, phase: int = 0) -> bool:
    """
    Checks if the profiling data for a given session and phase is complete.

    A "complete" phase is safe to read/clear because all device-side records for
    the phase have been flushed to the host and the phase will no longer receive
    new records.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to check. Defaults to 0.

    Returns:
        bool: True if the phase data is complete, False otherwise.
    """
    if session is None:
        return False
    if flags.command_line and session != 0:
        raise ValueError("Only one session can check phase completion status when running from the command line.")
    return libproton.is_data_phase_complete(session, phase)


def clear(
    session: Optional[int] = 0,
    phase: int = 0,
    clear_up_to_phase: bool = False,
) -> None:
    """
    Clears profiling data for a given session.

    Args:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is inactive.
        phase (int): The phase number to clear. Defaults to 0.
        clear_up_to_phase (bool): If True, clear all phases up to and including `phase`.
    """
    if session is None:
        return
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be cleared when running from the command line.")
    libproton.clear_data(session, phase, clear_up_to_phase)
