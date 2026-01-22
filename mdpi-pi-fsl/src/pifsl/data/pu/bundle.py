from __future__ import annotations
from typing import Tuple
from pifsl.core.base import DatasetBundle

class PUAdapter:
    name = "pu"

    def load_windows(self, *args, **kwargs) -> Tuple[DatasetBundle, DatasetBundle]:
        raise NotImplementedError(
        )
