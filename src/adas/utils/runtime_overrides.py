from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, TypeVar

T = TypeVar("T")


def get_runtime_overrides_path(project_root: str | None = None) -> str:
    if project_root:
        return os.path.join(project_root, "data", "processed", "runtime_overrides.json")
    env_path = os.environ.get("ADAS_RUNTIME_OVERRIDES_PATH")
    if env_path:
        return env_path
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "data" / "processed" / "runtime_overrides.json")


def load_runtime_overrides(project_root: str | None = None) -> Dict[str, Any]:
    path = get_runtime_overrides_path(project_root)
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def apply_dataclass_overrides(default_obj: T, section: str, project_root: str | None = None) -> T:
    try:
        overrides = load_runtime_overrides(project_root)
        values = overrides.get(section)
        if not isinstance(values, dict):
            return default_obj
        allowed = {f.name for f in dataclasses.fields(default_obj)}
        filtered = {k: v for k, v in values.items() if k in allowed}
        if not filtered:
            return default_obj
        return dataclasses.replace(default_obj, **filtered)
    except Exception:
        return default_obj
