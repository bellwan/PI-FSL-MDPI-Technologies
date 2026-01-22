import numpy as np
from typing import List, Tuple, Optional

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from pifsl.data.bosch_drilling.raw_processing.gating import quick_psd_embed
from pifsl.data.bosch_drilling import config as cfg

class PSDGuidedSampler:
    def __init__(self, target_imbalance_ratio: float = 0.3, 
                 n_clusters: Optional[int] = None,
                 clustering_method: str = 'kmeans',
                 random_state: int = 42):
        self.target_ratio = target_imbalance_ratio
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.random_state = random_state
        np.random.seed(random_state)
    
    def balance_dataset(self, X: List[np.ndarray], y: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        y_np = np.array(y)
        
        # Separate classes
        healthy_indices = np.where(y_np == 0)[0]
        worn_indices = np.where(y_np == 1)[0]
        
        if len(healthy_indices) == 0 or len(worn_indices) == 0:
            return X, y  
        
        X_healthy = [X[i] for i in healthy_indices]
        X_worn = [X[i] for i in worn_indices]
        
        # Apply sparse sampling to majority class 
        if len(X_healthy) > len(X_worn):
            X_healthy_sampled, y_healthy_sampled = self.sparse_sample_healthy(X_healthy, [0]*len(X_healthy))
            X_worn_kept, y_worn_kept = X_worn, [1]*len(X_worn)
        else:
            X_worn_sampled, y_worn_sampled = self.sparse_sample_healthy(X_worn, [1]*len(X_worn))
            X_healthy_kept, y_healthy_kept = X_healthy, [0]*len(X_healthy)
            return self._combine_and_shuffle(X_healthy_kept, y_healthy_kept, X_worn_sampled, y_worn_sampled)
        
        return self._combine_and_shuffle(X_healthy_sampled, y_healthy_sampled, X_worn_kept, y_worn_kept)
    
    def sparse_sample_healthy(self, X_healthy: List[np.ndarray], 
                            y_healthy: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        if not X_healthy:
            return [], []
            
        # Extract PSD features
        psd_features = np.vstack([quick_psd_embed(x, cfg.FS) for x in X_healthy])
        
        # Determine number of clusters
        if self.n_clusters is None:
            self.n_clusters = max(2, int(len(X_healthy) * self.target_ratio))
        
        if len(X_healthy) <= self.n_clusters:
            return X_healthy, y_healthy  
            
        try:
            # Cluster in PSD space
            if self.clustering_method == 'kmeans':
                cluster_labels = self._kmeans_clustering(psd_features)
            else:
                raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
            
            # Select representative samples from each cluster
            selected_indices = self._select_representative_samples(psd_features, cluster_labels)
            
            X_selected = [X_healthy[i] for i in selected_indices]
            y_selected = [y_healthy[i] for i in selected_indices]
            
            return X_selected, y_selected
            
        except Exception as e:
            print(f"PSD-guided sampling failed: {e}")
            return self._random_fallback(X_healthy, y_healthy)
    
    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=self.n_clusters, 
                       random_state=self.random_state,
                       n_init=10)
        return kmeans.fit_predict(features)
    
    def _select_representative_samples(self, features: np.ndarray, 
                                     cluster_labels: np.ndarray) -> List[int]:
        selected_indices = []
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Select sample closest to cluster centroid
                cluster_features = features[cluster_indices]
                centroid = cluster_features.mean(axis=0)
                
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                best_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(best_idx)
        
        return selected_indices
    
    def _random_fallback(self, X: List[np.ndarray], y: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        n_samples = max(1, int(len(X) * self.target_ratio))
        indices = np.random.choice(len(X), n_samples, replace=False)
        return [X[i] for i in indices], [y[i] for i in indices]
    
    def _combine_and_shuffle(self, X1: List[np.ndarray], y1: List[int],
                           X2: List[np.ndarray], y2: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        X_combined = X1 + X2
        y_combined = y1 + y2
        
        # Shuffle
        indices = np.random.permutation(len(X_combined))
        return [X_combined[i] for i in indices], [y_combined[i] for i in indices]
    
    def balance_dataset_with_indices(self, X, y):
        def _as_1d_signal(w):
            if isinstance(w, dict):
                # Bosch style: {"vibration":..., "current":...}
                if "vibration" in w:
                    return np.asarray(w["vibration"], dtype=np.float32).reshape(-1)
                # fallback: first array-like value
                for v in w.values():
                    try:
                        return np.asarray(v, dtype=np.float32).reshape(-1)
                    except Exception:
                        pass
                raise ValueError("No numeric signal found in dict window for PSD embedding.")
            return np.asarray(w, dtype=np.float32).reshape(-1)

        y_np = np.array(y)

        healthy_indices = np.where(y_np == 0)[0]
        worn_indices = np.where(y_np == 1)[0]

        if len(healthy_indices) == 0 or len(worn_indices) == 0:
            sel = np.arange(len(X), dtype=np.int64)
            return X, y, sel

        if len(healthy_indices) >= len(worn_indices):
            maj_idx = healthy_indices
            min_idx = worn_indices
        else:
            maj_idx = worn_indices
            min_idx = healthy_indices

        X_maj = [X[i] for i in maj_idx]
        n_maj = len(X_maj)
        if n_maj == 0:
            sel = np.arange(len(X), dtype=np.int64)
            return X, y, sel

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = max(2, int(n_maj * self.target_ratio))
            n_clusters = min(n_clusters, n_maj)

        try:
            psd_features = np.vstack([quick_psd_embed(_as_1d_signal(x), cfg.FS) for x in X_maj])

            if self.clustering_method != "kmeans":
                raise ValueError(f"Unsupported clustering method: {self.clustering_method}")

            kmeans = KMeans(
                n_clusters=int(n_clusters),
                random_state=self.random_state,
                n_init=10,
            )
            cluster_labels = kmeans.fit_predict(psd_features)

            selected_rel = []
            for cluster_id in range(int(n_clusters)):
                cluster_rel = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_rel) == 0:
                    continue
                cluster_feats = psd_features[cluster_rel]
                centroid = cluster_feats.mean(axis=0)
                d = np.linalg.norm(cluster_feats - centroid, axis=1)
                best_rel = cluster_rel[np.argmin(d)]
                selected_rel.append(int(best_rel))

            if len(selected_rel) == 0:
                n_samples = max(1, int(n_maj * self.target_ratio))
                rng = np.random.RandomState(self.random_state)
                selected_rel = rng.choice(n_maj, n_samples, replace=False).tolist()

            sel_maj = maj_idx[np.array(selected_rel, dtype=np.int64)]

        except Exception as e:
            print(f"PSD-guided sampling failed (with indices): {e}")
            n_samples = max(1, int(n_maj * self.target_ratio))
            rng = np.random.RandomState(self.random_state)
            sel_maj = maj_idx[rng.choice(n_maj, n_samples, replace=False)]

        combined = np.concatenate([sel_maj, min_idx]).astype(np.int64)
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(combined)

        X_bal = [X[i] for i in combined]
        y_bal = [int(y[i]) for i in combined]
        return X_bal, y_bal, combined



