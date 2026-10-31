"""Linux endpoint ownership tests using synthetic proc trees and one local socket."""

import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxmgr.harness.clm_vllm_process import _process_group, listener_owner

HEADER = (
    "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
)


def tcp_row(port, inode, *, state="0A", ipv6=False):
    address = "00000000000000000000000000000000" if ipv6 else "0100007F"
    return (
        f"  0: {address}:{port:04X} {address}:0000 {state} 00000000:00000000 "
        f"00:00000000 00000000 1000 0 {inode} 1 0000000000000000 100 0 0 10 0\n"
    )


class ListenerOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proc = Path(self.temp.name)
        (self.proc / "net").mkdir()
        (self.proc / "net/tcp").write_text(HEADER)
        (self.proc / "net/tcp6").write_text(HEADER)

    def process(self, pid, pgid, inode, *, comm="worker", parent=1):
        root = self.proc / str(pid)
        (root / "fd").mkdir(parents=True)
        (root / "stat").write_text(f"{pid} ({comm}) S {parent} {pgid} 1 0 -1 0 0\n")
        (root / "fd/8").symlink_to(f"socket:[{inode}]")
        return root

    def test_empty_tables_and_non_listening_connections_are_free(self):
        self.assertEqual(listener_owner(8123, 100, self.proc), "free")
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 10, state="01"))
        self.assertEqual(listener_owner(8123, 100, self.proc), "free")

    def test_descendant_in_same_process_group_owns_listener(self):
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 70))
        self.process(901, 100, 70, parent=100, comm="vllm child ) with spaces ) tail")
        self.assertEqual(listener_owner(8123, 100, self.proc), "owned")

    def test_other_process_group_or_orphan_listener_is_foreign(self):
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 70))
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")
        self.process(901, 200, 70)
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")

    def test_every_ipv4_and_ipv6_listener_must_have_proven_ownership(self):
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 70))
        (self.proc / "net/tcp6").write_text(HEADER + tcp_row(8123, 80, ipv6=True))
        self.process(901, 100, 70)
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")
        self.process(902, 100, 80)
        self.assertEqual(listener_owner(8123, 100, self.proc), "owned")

    def test_disabled_ipv6_is_supported_but_missing_ipv4_is_not_trusted(self):
        (self.proc / "net/tcp6").unlink()
        self.assertEqual(listener_owner(8123, 100, self.proc), "free")
        (self.proc / "net/tcp").unlink()
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")

    def test_malformed_process_or_tcp_table_fails_closed(self):
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 70))
        process = self.process(901, 100, 70)
        (process / "stat").write_text("incomplete")
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")
        (self.proc / "net/tcp").write_text("malformed TCP table")
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")

    def test_permission_errors_do_not_masquerade_as_a_free_or_owned_port(self):
        original_read = Path.read_text

        def denied_table(path, *args, **kwargs):
            if path == self.proc / "net/tcp":
                raise PermissionError("synthetic denied table")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", denied_table):
            self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 70))
        process = self.process(901, 100, 70)
        original_iterdir = Path.iterdir

        def denied_descriptors(path):
            if path == process / "fd":
                raise PermissionError("synthetic denied descriptors")
            return original_iterdir(path)

        with patch.object(Path, "iterdir", denied_descriptors):
            self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")

    def test_vanished_process_and_descriptor_are_handled(self):
        (self.proc / "net/tcp").write_text(HEADER + tcp_row(8123, 70))
        (self.proc / "900").mkdir()
        process = self.process(901, 100, 70)
        (process / "fd/8").unlink()
        (process / "fd/8").write_text("not a socket link")
        self.assertEqual(listener_owner(8123, 100, self.proc), "foreign")
        (self.proc / "net/tcp").write_text(HEADER)
        self.assertEqual(listener_owner(8123, 100, self.proc), "free")

    @unittest.skipUnless(
        sys.platform == "linux" and Path("/proc/net/tcp").exists(), "Requires Linux proc"
    )
    def test_actual_current_process_listener(self):
        # Some test containers expose host IDs in proc but virtualize getpgrp().
        # Read the same proc namespace the ownership checker actually inspects.
        process_group = _process_group(Path("/proc/self/stat").read_text())
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            self.assertEqual(listener_owner(port, process_group), "owned")
            self.assertEqual(listener_owner(port, process_group + 100000000), "foreign")
        self.assertEqual(listener_owner(port, process_group), "free")


if __name__ == "__main__":
    unittest.main()
