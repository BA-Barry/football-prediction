"""Create a project-local virtual environment on Windows, macOS or Linux."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv",
                        help="virtual-environment directory (default: <repo>/.venv)")
    parser.add_argument("--extras", default="test",
                        help="comma-separated optional dependency groups")
    args = parser.parse_args()
    environment = args.venv.expanduser().resolve()
    if not environment.exists():
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment_python(environment)
    if not python.is_file():
        raise SystemExit(f"Virtual-environment Python was not created: {python}")
    extras = f"[{args.extras}]" if args.extras.strip() else ""
    subprocess.run([str(python), "-m", "pip", "install", "-e", f"{ROOT}{extras}"],
                   check=True, cwd=ROOT)
    print(f"Environment ready: {environment}")
    print(f"Demo: {python} -m football_model.demo")
    print(f"Tests: {python} -m pytest")


if __name__ == "__main__":
    main()
