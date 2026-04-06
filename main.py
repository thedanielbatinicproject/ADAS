#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.ui.master_panel import run_master_dashboard  # noqa: E402


if __name__ == "__main__":
    run_master_dashboard(PROJECT_ROOT)
