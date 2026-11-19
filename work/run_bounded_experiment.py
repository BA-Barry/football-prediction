"""Run one explicit command with a required time budget; never retry it.

Example (run from the project directory):
  python -B work/run_bounded_experiment.py --name smoke --max-seconds 30 \
    --log-dir work/bounded-logs \
    -- python -B work/test_bounded_experiment.py
Exit codes: 0 success, 1 failure, 124 budget exhausted, 130 interrupted.
The time budget ends execution; recorded elapsed time also includes bounded
process cleanup. Windows taskkill targets the live root's process tree. A root
that exits early or deliberately detaches descendants cannot be fully tracked
by this minimal wrapper; use foreground commands that wait for their workers.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re
import shutil
import signal
import subprocess
import sys
import time


PROJECT_ROOT = REPOSITORY_ROOT.resolve()
STATUSES = ("success", "failure", "time_budget_exhausted", "interrupted")


def positive_seconds(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-seconds must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("max-seconds must be finite and positive")
    return number


def project_log_dir(value):
    candidate = Path(value).expanduser().resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"log-dir must be within {PROJECT_ROOT}") from exc
    return candidate


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def explicit_command(command):
    executable = shutil.which(command[0])
    if not executable:
        raise ValueError(f"executable was not found: {command[0]}")
    resolved = [str(Path(executable).resolve()), *command[1:]]
    # Make Python input scripts unambiguous in both execution and the record.
    if Path(executable).stem.lower().startswith("python"):
        index = 1
        while index < len(resolved):
            argument = resolved[index]
            if argument in ("-c", "-m"):
                if index + 1 >= len(resolved):
                    raise ValueError(f"{argument} requires an argument")
                break
            if argument in ("-W", "-X"):
                index += 2
                continue
            if argument == "--":
                index += 1
                if index >= len(resolved):
                    raise ValueError("Python input script is missing")
                argument = resolved[index]
            elif argument.startswith("-"):
                index += 1
                continue
            script = Path(argument).resolve()
            if not script.is_file():
                raise ValueError(f"Python input script does not exist: {script}")
            resolved[index] = str(script)
            break
        else:
            raise ValueError("provide an explicit Python script, -c code, or -m module")
    return resolved


def child_environment(run_dir):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    locations = {
        "TEMP": "temp", "TMP": "temp", "TMPDIR": "temp",
        "NUMBA_CACHE_DIR": "numba", "PIP_CACHE_DIR": "pip",
        "PYTHONPYCACHEPREFIX": "pycache", "MPLCONFIGDIR": "matplotlib",
        "XDG_CACHE_HOME": "cache", "JOBLIB_TEMP_FOLDER": "joblib",
        "HF_HOME": "huggingface", "TORCH_HOME": "torch",
    }
    for variable, directory in locations.items():
        path = run_dir / "runtime" / directory
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    return environment


def terminate_tree(process):
    """Target only this live child's tree; do not enumerate other Python jobs."""
    details = {"method": "taskkill /PID /T /F" if os.name == "nt" else "killpg"}
    if process.poll() is not None:
        details["error"] = "child exited before tree cleanup; descendants cannot be confirmed"
        return details
    try:
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
            cleanup = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            details.update(returncode=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
            if cleanup.returncode:
                details["error"] = "taskkill did not confirm tree termination"
        else:
            os.killpg(process.pid, signal.SIGKILL)
            details["returncode"] = 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        details["error"] = str(exc)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        details["error"] = "tree cleanup failed; only the immediate child was stopped"
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-seconds", required=True, type=positive_seconds)
    parser.add_argument("--log-dir", required=True, type=project_log_dir)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] != ["--"]:
        parser.error("use -- before the executable and its arguments")
    command = args.command[1:]
    if not command:
        parser.error("provide -- followed by an explicit executable and arguments")
    if not args.name.strip():
        parser.error("name must not be empty")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.name).strip(".-")[:80] or "run"
    run_dir = args.log_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {
        "name": args.name, "started": utc_now(), "finished": None,
        "time_budget_seconds": args.max_seconds, "elapsed_seconds": None,
        "command": command, "cwd": str(Path.cwd()), "pid": None,
        "returncode": None, "status": "failure", "stop_reason": "launch_failed",
        "stdout": str(run_dir / "stdout.log"), "stderr": str(run_dir / "stderr.log"),
    }
    process = None
    try:
        record["command"] = explicit_command(command)
        environment = child_environment(run_dir)
        with open(record["stdout"], "wb") as stdout, open(record["stderr"], "wb") as stderr:
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                record.update(status="time_budget_exhausted", stop_reason="budget_exhausted_before_launch")
            else:
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
                process = subprocess.Popen(
                    record["command"], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    env=environment, shell=False, **options,
                )
                record["pid"] = process.pid
                try:
                    remaining = max(0, args.max_seconds - (time.monotonic() - started))
                    process.wait(timeout=remaining)
                    record.update(
                        status="success" if process.returncode == 0 else "failure",
                        stop_reason="command_exited_zero" if process.returncode == 0 else "command_exited_nonzero",
                    )
                except subprocess.TimeoutExpired:
                    record.update(status="time_budget_exhausted", stop_reason="maximum_wall_time_reached")
                    record["cleanup"] = terminate_tree(process)
                except KeyboardInterrupt:
                    record.update(status="interrupted", stop_reason="keyboard_interrupt")
                    record["cleanup"] = terminate_tree(process)
    except KeyboardInterrupt:
        record.update(status="interrupted", stop_reason="keyboard_interrupt")
        if process is not None and process.poll() is None:
            record["cleanup"] = terminate_tree(process)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc)
        if process is not None:
            record["stop_reason"] = "execution_error"
        if process is not None and process.poll() is None:
            record["cleanup"] = terminate_tree(process)
    finally:
        record["returncode"] = process.returncode if process is not None else None
        record["finished"] = utc_now()
        record["elapsed_seconds"] = round(time.monotonic() - started, 6)
        record["status_counts"] = {status: int(record["status"] == status) for status in STATUSES}
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"result": str(result_path), "status": record["status"],
                          "stop_reason": record["stop_reason"], "elapsed_seconds": record["elapsed_seconds"]}))
    return {"success": 0, "failure": 1, "time_budget_exhausted": 124, "interrupted": 130}[record["status"]]


if __name__ == "__main__":
    sys.exit(main())
