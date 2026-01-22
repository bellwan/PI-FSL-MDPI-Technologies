from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional, Tuple


def _ensure_src_package(src_path: Path) -> None:
    if "src" in sys.modules:
        return
    pkg = types.ModuleType("src")
    pkg.__path__ = [str(src_path)]  
    sys.modules["src"] = pkg


def locate_code_roots(project_root: Path) -> Tuple[Path, Path]:
    src_root = project_root / "src"
    if not src_root.exists():
        raise FileNotFoundError(f"Expected source folder: {src_root}")
    return src_root, src_root


def bootstrap(project_root: Optional[str] = None) -> Path:
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[3]
    )

    src_root, _ = locate_code_roots(root)

    _ensure_src_package(src_root)

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root
