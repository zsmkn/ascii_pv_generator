"""Windows console setup: UTF-8 code page and VT (ANSI escape) processing.

Windows Terminal already has VT on; the classic conhost needs it switched on per process. Both need
UTF-8 output or box/block glyphs turn into '?' under code pages such as cp950.
"""

from __future__ import annotations

import sys

STD_OUTPUT_HANDLE = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


def setup_console() -> bool:
    """Make stdout UTF-8 and ANSI-capable. Returns True if VT processing is on."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetConsoleOutputCP(65001)
    k32.SetConsoleCP(65001)
    handle = k32.GetStdHandle(STD_OUTPUT_HANDLE)
    mode = wintypes.DWORD()
    if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False  # not a console (redirected to a file or pipe)
    if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
        return True
    return bool(k32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
