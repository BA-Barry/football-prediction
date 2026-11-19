"""Two short real process checks; no model data, training, or dependencies."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(r"G:\Codex\football-prediction-2026-10-22")
RUNNER = ROOT / "work" / "run_bounded_experiment.py"
LOG_ROOT = ROOT / "work" / "bounded-experiment-checks"


def process_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: PID absent.
            return False
        raise OSError(ctypes.get_last_error(), f"cannot inspect process {pid}")
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


class BoundedExperimentChecks(unittest.TestCase):
    def run_command(self, name, seconds, script, *arguments):
        result = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--name", name,
             "--max-seconds", str(seconds), "--log-dir", str(LOG_ROOT),
             "--", sys.executable, "-B", str(script), *arguments],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        message = json.loads(result.stdout.strip().splitlines()[-1])
        record = json.loads(Path(message["result"]).read_text(encoding="utf-8"))
        print(json.dumps({"check": name, "wrapper_exit": result.returncode,
                          "status": record["status"], "stop_reason": record["stop_reason"],
                          "elapsed_seconds": record["elapsed_seconds"], "result": message["result"]}))
        return result, record

    def test_short_normal_exit(self):
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="success-input-", dir=LOG_ROOT) as temporary:
            script = Path(temporary) / "success.py"
            script.write_text("import os\nprint('completed')\nprint(os.environ['TEMP'])\nprint(os.environ['PYTHONDONTWRITEBYTECODE'])\n", encoding="utf-8")
            result, record = self.run_command("short-normal-exit", 5, script)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["returncode"], 0)
            output = Path(record["stdout"]).read_text(encoding="utf-8")
            self.assertIn("completed", output)
            self.assertIn(str(ROOT), output)
            self.assertTrue(output.strip().endswith("1"))
            self.assertEqual(record["status_counts"]["success"], 1)

    def test_timeout_stops_child_and_grandchild(self):
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="timeout-input-", dir=LOG_ROOT) as temporary:
            directory = Path(temporary)
            script = directory / "tree.py"
            script.write_text(
                "import os, pathlib, subprocess, sys, time\n"
                "root = pathlib.Path(sys.argv[1])\n"
                "depth = int(sys.argv[2])\n"
                "(root / ('pid-' + str(depth))).write_text(str(os.getpid()))\n"
                "if depth < 2:\n"
                "    subprocess.Popen([sys.executable, '-B', __file__, str(root), str(depth + 1)])\n"
                "while True:\n"
                "    (root / ('heartbeat-' + str(depth))).write_text(str(time.monotonic()))\n"
                "    time.sleep(0.05)\n", encoding="utf-8",
            )
            result, record = self.run_command("short-timeout-tree", 1.5, script, str(directory), "0")
            pids = [int(path.read_text()) for path in sorted(directory.glob("pid-*"))]
            try:
                self.assertEqual(result.returncode, 124, result.stderr)
                self.assertEqual(record["status"], "time_budget_exhausted")
                self.assertEqual(record["stop_reason"], "maximum_wall_time_reached")
                self.assertEqual(len(pids), 3, "root, child and grandchild must actually start")
                self.assertEqual(record["status_counts"]["time_budget_exhausted"], 1)
                self.assertNotIn("error", record["cleanup"], record["cleanup"])
                self.assertTrue(all(not process_alive(pid) for pid in pids), f"remaining processes: {pids}")
                before = {path.name: path.read_text() for path in directory.glob("heartbeat-*")}
                time.sleep(0.2)
                after = {path.name: path.read_text() for path in directory.glob("heartbeat-*")}
                self.assertEqual(before, after, "a descendant continued writing after timeout")
                self.assertLess(record["elapsed_seconds"], 12)
            finally:
                # Cleanup only these known test PIDs if a regression leaves one alive.
                for pid in pids:
                    if process_alive(pid):
                        if os.name == "nt":
                            subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"),
                                            "/PID", str(pid), "/T", "/F"], capture_output=True,
                                           timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            os.kill(pid, 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
