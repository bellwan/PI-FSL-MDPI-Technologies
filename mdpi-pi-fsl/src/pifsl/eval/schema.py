from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class ResultRecord:
    # identity
    timestamp_utc: str
    dataset: str
    scenario: str
    method: str
    variant: str            
    result_source: str  

    # protocol
    source_domain: str
    target_domain: str
    n_way: int
    k_shot: int
    q_query: int
    train_episodes: int
    eval_episodes: int
    seed: int

    # preprocessing 
    fs: float
    window_samples: int
    window_seconds: float
    overlap_ratio: float
    normalization: str
    input_representation: str  # raw_1d / scalogram_64x64

    # metrics (mean over eval episodes)
    acc_mean: float
    acc_ci95: Optional[float]
    bacc_mean: float
    bacc_ci95: Optional[float]
    macro_f1_mean: float
    macro_f1_ci95: Optional[float]

    # audit
    notes: str = ""
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["extra"] = d.get("extra") or {}
        return d


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as fb:
            fb.seek(-1, 2)
            last = fb.read(1)
        if last not in (b"\n", b"\r"):
            with path.open("a", encoding="utf-8") as f:
                f.write("\n")

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

