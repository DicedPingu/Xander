#!/usr/bin/env python3
"""Compatibility launcher for the packaged Xander interface."""

import os
import sys
from pathlib import Path

from xander_agent.cli import entrypoint


def main() -> None:
    """Retain the callable used by ASKAR's existing ``bin/xander`` launcher."""

    project_launcher = Path(__file__).resolve().parent / ".venv" / "bin" / "xander"
    project_python = project_launcher.parent / "python"
    if project_launcher.exists() and Path(sys.executable).resolve() != project_python.resolve():
        os.execv(str(project_launcher), [str(project_launcher), *sys.argv[1:]])
    entrypoint()


if __name__ == "__main__":
    main()
