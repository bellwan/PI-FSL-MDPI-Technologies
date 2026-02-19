import numpy as np
from typing import List, Tuple, Dict, Any

from pifsl.data.bosch_drilling.raw_processing.gating import quick_psd_embed
from pifsl.data.bosch_drilling.raw_processing.datasets import FewShotScalogramSet
from pifsl.core.sampling.psd_sampling import PSDGuidedSampler
from pifsl.data.bosch_drilling import config as cfg
import torch


class DualRoleFeatureStrategy:
    def __init__(self, sampling_config: Dict[str, Any] = None):
        self.sampling_config = sampling_config or {}
        self.psd_sampler = PSDGuidedSampler(**self.sampling_config)
        
    def prepare_training_data(self, X: List[np.ndarray], y: List[int], 
                            strategy: str = "psd-guided") -> Tuple[List[np.ndarray], List[int]]:
        if strategy != "psd-guided":
            raise ValueError(
                f"Unsupported sampling strategy '{strategy}'. "
                "Only 'psd-guided' is supported in the current PI-FSL setup."
            )

        return self.psd_sampler.balance_dataset(X, y)

    def _random_balanced_subset(self, X: List[np.ndarray], y: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        y_np = np.array(y)
        healthy_indices = np.where(y_np == 0)[0]
        worn_indices = np.where(y_np == 1)[0]
        
        n_samples = min(len(healthy_indices), len(worn_indices))
        
        healthy_selected = np.random.choice(healthy_indices, n_samples, replace=False)
        worn_selected = np.random.choice(worn_indices, n_samples, replace=False)
        
        selected_indices = np.concatenate([healthy_selected, worn_selected])
        np.random.shuffle(selected_indices)
        
        return [X[i] for i in selected_indices], [y[i] for i in selected_indices]
    
    def create_dataset(self, X: List[np.ndarray], y: List[int], 
                      features: str = "cwt") -> torch.utils.data.Dataset:
        if features == "cwt":
            return FewShotScalogramSet(X, y, cfg.FS)
        else:
            raise ValueError(f"Unknown features: {features}")
    
    def extract_dual_features(self, X: List[np.ndarray]) -> Dict[str, np.ndarray]:
        psd_features = np.vstack([quick_psd_embed(x, cfg.FS) for x in X])
        cwt_features = []  
        
        return {
            'psd': psd_features,
            'cwt': cwt_features
        }