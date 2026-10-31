"""Prove Linux TCP listener ownership before trusting local model endpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

ListenerOwner = Literal["free", "owned", "foreign"]
_SOCKET_LINK = re.compile(r"socket:\[(\d+)\]")


def _listeners(port: int, proc_root: Path) -> set[str]:
    """Read both TCP tables; unknown table contents must never imply a free port."""
    inodes = set()
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except FileNotFoundError:
            # IPv6 may be disabled. The IPv4 table is required on Bouchet.
            if name == "tcp6":
                continue
            raise
        if not lines or "local_address" not in lines[0]:
            raise ValueError("Malformed Linux TCP table")
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError("Malformed Linux TCP table entry")
            address, separator, encoded_port = fields[1].rpartition(":")
            if not separator or not address:
                raise ValueError("Malformed Linux TCP listener address")
            local_port = int(encoded_port, 16)
            if local_port == port and fields[3] == "0A":
                if not fields[9].isdecimal() or int(fields[9]) <= 0:
                    raise ValueError("TCP listener has no inspectable socket inode")
                # Conservatively include every local address on this port. An IPv6
                # wildcard can accept IPv4 loopback connections as well.
                inodes.add(fields[9])
    return inodes


def _process_group(stat: str) -> int:
    """Linux comm may contain spaces and ')'; numeric fields follow its last ')' ."""
    closing = stat.rfind(")")
    if closing < 0 or "(" not in stat[:closing]:
        raise ValueError("Malformed Linux process stat")
    fields = stat[closing + 1 :].split()
    if len(fields) < 3:
        raise ValueError("Incomplete Linux process stat")
    return int(fields[2])  # state, parent PID, process-group ID


def _group_sockets(pgid: int, proc_root: Path, wanted: set[str]) -> set[str]:
    owned = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if _process_group((entry / "stat").read_text(encoding="utf-8")) != pgid:
                continue
            for descriptor in (entry / "fd").iterdir():
                try:
                    link = str(descriptor.readlink())
                except OSError:
                    # Descriptors and processes can disappear between proc reads.
                    continue
                match = _SOCKET_LINK.fullmatch(link)
                if match and match.group(1) in wanted:
                    owned.add(match.group(1))
        except (OSError, ValueError, UnicodeError):
            # An inaccessible candidate cannot establish ownership. It may also
            # have exited normally; the final TCP-table read handles that race.
            continue
        if wanted <= owned:
            break
    return owned


def listener_owner(port: int, pgid: int, proc_root: Path = Path("/proc")) -> ListenerOwner:
    """Return ``free``, ``owned``, or ``foreign`` for a local TCP port.

    ``owned`` requires every listening socket on the port to be held by a
    process in ``pgid`` (including descendants in that process group).
    ``foreign`` also covers permission errors or malformed proc data when
    ownership cannot be proved. No endpoint request should be sent in that case.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535")
    if type(pgid) is not int or pgid <= 0:
        raise ValueError("Process group must be a positive integer")
    proc_root = Path(proc_root)
    try:
        initial = _listeners(port, proc_root)
        if not initial:
            return "free"
        owned = _group_sockets(pgid, proc_root, initial)
        current = _listeners(port, proc_root)
        if not current:
            return "free"
        return "owned" if current <= owned else "foreign"
    except (OSError, ValueError, UnicodeError):
        return "foreign"
