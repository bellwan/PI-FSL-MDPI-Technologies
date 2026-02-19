import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import wasserstein_distance, energy_distance
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from scipy.stats import energy_distance

class DomainShiftAnalyzer:
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.metrics_history = []
        np.random.seed(random_state)

    def compute_comprehensive_shift(self,
                                  source_features: np.ndarray,
                                  target_features: np.ndarray,
                                  source_labels: Optional[np.ndarray] = None,
                                  target_labels: Optional[np.ndarray] = None,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        metrics = {}

        # Basic distribution distance metrics
        metrics.update(self._compute_distribution_distances(source_features, target_features))

        # Geometric and structural metrics
        metrics.update(self._compute_geometric_metrics(source_features, target_features))

        # Class-aware metrics (if labels available)
        if source_labels is not None and target_labels is not None:
            metrics.update(self._compute_class_aware_metrics(
                source_features, source_labels, target_features, target_labels
            ))

        # Feature-level shift analysis
        metrics.update(self._compute_feature_level_shifts(
            source_features, target_features, feature_names
        ))

        # Store metrics with timestamp
        self.metrics_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics.copy()
        })

        return metrics

    def _compute_distribution_distances(self, source: np.ndarray,
                                      target: np.ndarray) -> Dict[str, float]:
        metrics = {}

        # Wasserstein distance (approximated)
        metrics['wasserstein_distance'] = self._compute_sliced_wasserstein(source, target)

        # Maximum Mean Discrepancy (MMD)
        metrics['mmd'] = self._compute_mmd(source, target)

        # Energy distance
        metrics['energy_distance'] = self._compute_energy_distance(source, target)

        # Jensen-Shannon divergence (approximated)
        metrics['js_divergence'] = self._compute_js_divergence(source, target)

        return metrics

    def _compute_geometric_metrics(self, source: np.ndarray,
                                 target: np.ndarray) -> Dict[str, float]:
        metrics = {}

        # Centroid shift
        metrics['centroid_shift'] = self._compute_centroid_shift(source, target)

        # Distribution overlap
        metrics['distribution_overlap'] = self._compute_distribution_overlap(source, target)

        # Local neighborhood preservation
        metrics['neighborhood_preservation'] = self._compute_neighborhood_preservation(source, target)

        # Variance ratio
        metrics['variance_ratio'] = self._compute_variance_ratio(source, target)

        return metrics

    def _compute_class_aware_metrics(self, source_features: np.ndarray,
                                   source_labels: np.ndarray,
                                   target_features: np.ndarray,
                                   target_labels: np.ndarray) -> Dict[str, float]:
        metrics = {}

        unique_classes = np.unique(np.concatenate([source_labels, target_labels]))
        class_shifts = []
        conditional_shifts = []

        for cls in unique_classes:
            source_cls_mask = source_labels == cls
            target_cls_mask = target_labels == cls

            if np.sum(source_cls_mask) > 1 and np.sum(target_cls_mask) > 1:
                source_cls_features = source_features[source_cls_mask]
                target_cls_features = target_features[target_cls_mask]

                # Class-conditional distribution shift
                class_shift = self._compute_sliced_wasserstein(
                    source_cls_features, target_cls_features
                )
                class_shifts.append(class_shift)

                # Conditional distribution alignment
                conditional_shift = self._compute_conditional_alignment(
                    source_cls_features, target_cls_features
                )
                conditional_shifts.append(conditional_shift)

        if class_shifts:
            metrics['mean_class_shift'] = float(np.mean(class_shifts))
            metrics['max_class_shift'] = float(np.max(class_shifts))
            metrics['std_class_shift'] = float(np.std(class_shifts))

        if conditional_shifts:
            metrics['mean_conditional_shift'] = float(np.mean(conditional_shifts))

        return metrics

    def _compute_feature_level_shifts(self, source: np.ndarray,
                                    target: np.ndarray,
                                    feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        metrics = {}

        # Feature importance shift
        metrics['feature_importance_shift'] = self._compute_feature_importance_shift(source, target)

        # Correlation structure shift
        metrics['correlation_shift'] = self._compute_correlation_shift(source, target)

        # Dimensionality mismatch
        metrics['intrinsic_dimensionality_ratio'] = self._compute_intrinsic_dimensionality_ratio(source, target)

        return metrics

    def _compute_sliced_wasserstein(self, source: np.ndarray,
                                  target: np.ndarray,
                                  n_projections: int = 100) -> float:
        if len(source) == 0 or len(target) == 0:
            return float('inf')

        n_features = source.shape[1]
        projections = np.random.randn(n_projections, n_features)
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)

        wasserstein_dists = []
        for proj in projections:
            source_proj = source @ proj
            target_proj = target @ proj
            w_dist = wasserstein_distance(source_proj, target_proj)
            wasserstein_dists.append(w_dist)

        return float(np.mean(wasserstein_dists))

    def _compute_mmd(self, source: np.ndarray, target: np.ndarray,
                    gamma: Optional[float] = None) -> float:
        if len(source) == 0 or len(target) == 0:
            return float('inf')

        n_source = len(source)
        n_target = len(target)

        if gamma is None:
            # Use median heuristic for bandwidth
            pairwise_dists = pdist(np.vstack([source, target]))
            gamma = 1.0 / (2.0 * (np.median(pairwise_dists) ** 2 + 1e-8))

        # Compute kernel matrices
        XX = np.exp(-gamma * cdist(source, source) ** 2)
        YY = np.exp(-gamma * cdist(target, target) ** 2)
        XY = np.exp(-gamma * cdist(source, target) ** 2)

        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return float(np.sqrt(max(mmd, 0)))

    def _compute_energy_distance(self, source, target):
        source = np.asarray(source)
        target = np.asarray(target)

        if source.size == 0 or target.size == 0:
            return float("inf")

        # Reshape to (n_samples, n_features)
        source_2d = source.reshape(source.shape[0], -1)
        target_2d = target.reshape(target.shape[0], -1)

        n_features = source_2d.shape[1]
        if n_features == 0:
            return 0.0


        distances = []
        for j in range(n_features):
            u = source_2d[:, j]
            v = target_2d[:, j]
            # energy_distance works on 1D arrays
            d = energy_distance(u, v)
            distances.append(d)

        return float(np.mean(distances))

    def _compute_js_divergence(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return float('inf')

        # Combine samples
        combined = np.vstack([source, target])
        n_source = len(source)
        n_total = len(combined)

        # Use kNN to estimate probability ratios
        k = min(10, n_total // 10)
        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(combined)

        distances, indices = knn.kneighbors(combined)

        # Estimate probability density ratios
        source_mask = np.zeros(n_total, dtype=bool)
        source_mask[:n_source] = True

        js_div = 0.0
        for i in range(n_total):
            neighbors = indices[i, 1:]  # Exclude self
            source_neighbors = np.sum(source_mask[neighbors])
            p_source = source_neighbors / k
            p_target = 1 - p_source

            if source_mask[i]:
                p = p_source
                q = p_target
            else:
                p = p_target
                q = p_source

            if p > 0 and q > 0:
                m = 0.5 * (p + q)
                js_div += 0.5 * (p * np.log(p / m) + q * np.log(q / m))

        return float(js_div / n_total)

    def _compute_centroid_shift(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return float('inf')

        source_centroid = source.mean(axis=0)
        target_centroid = target.mean(axis=0)

        return float(np.linalg.norm(source_centroid - target_centroid))

    def _compute_distribution_overlap(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return 0.0

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(target)

        distances, _ = knn.kneighbors(source)
        median_distance = np.median(distances)

        # Count samples within adaptive threshold
        threshold = median_distance * 1.5
        overlap_count = np.sum(distances < threshold)
        overlap_ratio = overlap_count / len(source)

        return float(overlap_ratio)

    def _compute_neighborhood_preservation(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return 0.0

        # Use kNN to find neighbors in both domains
        k = min(10, len(source) // 10, len(target) // 10)

        knn_source = NearestNeighbors(n_neighbors=k + 1)
        knn_source.fit(source)
        source_neighbors = knn_source.kneighbors(source, return_distance=False)

        knn_target = NearestNeighbors(n_neighbors=k + 1)
        knn_target.fit(target)
        target_neighbors = knn_target.kneighbors(target, return_distance=False)

        # Compute neighborhood overlap
        overlap_scores = []
        # Compare neighborhoods for as many paired indices as possible
        n_pairs = min(len(source_neighbors), len(target_neighbors))
        preservation_scores = []

        for i in range(n_pairs):
            # Exclude self (index 0 is the point itself)
            source_neighbor_set = set(source_neighbors[i, 1:])
            target_neighbor_set = set(target_neighbors[i, 1:])

            intersection = len(source_neighbor_set & target_neighbor_set)
            union = len(source_neighbor_set | target_neighbor_set)

            if union > 0:
                preservation_scores.append(intersection / union)

        if not preservation_scores:
            return 0.0

        return float(np.mean(preservation_scores))


    def _compute_variance_ratio(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return 1.0

        source_variance = np.mean(np.var(source, axis=0))
        target_variance = np.mean(np.var(target, axis=0))

        if target_variance == 0:
            return float('inf')

        return float(source_variance / target_variance)

    def _compute_conditional_alignment(self, source_cls: np.ndarray,
                                     target_cls: np.ndarray) -> float:
        return self._compute_sliced_wasserstein(source_cls, target_cls)

    def _compute_feature_importance_shift(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return float('inf')

        # Use variance as proxy for feature importance
        source_importance = np.std(source, axis=0)
        target_importance = np.std(target, axis=0)

        # Normalize
        source_importance = source_importance / (np.sum(source_importance) + 1e-12)
        target_importance = target_importance / (np.sum(target_importance) + 1e-12)

        # Compute importance shift
        importance_shift = wasserstein_distance(source_importance, target_importance)
        return float(importance_shift)

    def _compute_correlation_shift(self, source: np.ndarray, target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return float('inf')

        source_corr = np.corrcoef(source.T)
        target_corr = np.corrcoef(target.T)

        # Handle potential NaN values
        source_corr = np.nan_to_num(source_corr, nan=0.0)
        target_corr = np.nan_to_num(target_corr, nan=0.0)

        # Flatten correlation matrices and compute distance
        source_corr_flat = source_corr[np.triu_indices_from(source_corr, k=1)]
        target_corr_flat = target_corr[np.triu_indices_from(target_corr, k=1)]

        correlation_shift = np.linalg.norm(source_corr_flat - target_corr_flat)
        return float(correlation_shift)

    def _compute_intrinsic_dimensionality_ratio(self, source: np.ndarray,
                                              target: np.ndarray) -> float:
        if len(source) == 0 or len(target) == 0:
            return 1.0

        # Use PCA to estimate intrinsic dimensionality
        pca_source = PCA()
        pca_source.fit(source)
        source_variance_ratio = pca_source.explained_variance_ratio_

        pca_target = PCA()
        pca_target.fit(target)
        target_variance_ratio = pca_target.explained_variance_ratio_

        # Find number of components explaining 95% variance
        source_cumulative = np.cumsum(source_variance_ratio)
        target_cumulative = np.cumsum(target_variance_ratio)

        source_dims = np.argmax(source_cumulative >= 0.95) + 1
        target_dims = np.argmax(target_cumulative >= 0.95) + 1

        if target_dims == 0:
            return float('inf')

        return float(source_dims / target_dims)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        if not self.metrics_history:
            return pd.DataFrame()

        records = []
        for entry in self.metrics_history:
            record = entry['metrics'].copy()
            record['timestamp'] = entry['timestamp']
            records.append(record)

        return pd.DataFrame(records)
