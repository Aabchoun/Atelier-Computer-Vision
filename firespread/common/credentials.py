"""Personal credentials and ids (FIRMS MAP_KEY, Earth Engine project).

Read from environment variables first, then from the `.env` file at the project root.
`.env` is ignored by Git: never put a key in code, configs, notebooks, screenshots or commits.
Each team member has their own `.env` (template: `.env.example`).
"""

from __future__ import annotations

import os
from pathlib import Path

from firespread.common.config import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"


class MissingCredentialError(RuntimeError):
    pass


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse KEY=VALUE lines (comments with #, optional quotes). Missing file -> {}."""
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: Notepad may add a BOM
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_credential(name: str, path: Path = ENV_FILE) -> str:
    value = os.environ.get(name) or read_env_file(path).get(name, "")
    if not value:
        raise MissingCredentialError(f"{name} is not set: add the line {name}=... to {path} (template: .env.example)")
    return value
