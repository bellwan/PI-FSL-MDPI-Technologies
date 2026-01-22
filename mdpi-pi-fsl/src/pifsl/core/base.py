from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Tuple
import numpy as np

@dataclass
class DatasetBundle:
    X: List[np.ndarray] 
    y: List[int]  
    domain: List[str] 
    file_id: List[str] 
    fs: float    

class DatasetAdapter(Protocol):
    name: str
    def load_windows(self, *args, **kwargs) -> Tuple[DatasetBundle, DatasetBundle]:
        ...
