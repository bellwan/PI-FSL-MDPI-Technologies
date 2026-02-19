from __future__ import annotations
import sys
from pathlib import Path

def add_project_roots_to_syspath(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cwru_dir = root / "cwru"
    if cwru_dir.exists() and str(cwru_dir) not in sys.path:
        sys.path.insert(0, str(cwru_dir))

    bosch_dir = root / "bosch"
    if bosch_dir.exists() and str(bosch_dir) not in sys.path:
        sys.path.insert(0, str(bosch_dir))

    return root
