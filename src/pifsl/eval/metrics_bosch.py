import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, average_precision_score, brier_score_loss,
    matthews_corrcoef, confusion_matrix, cohen_kappa_score
)
from scipy import stats
import pandas as pd

class ComprehensiveMetrics:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_basic_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._compute_probability_metrics(y_true, y_proba))
        
        # Statistical metrics
        metrics.update(self._compute_statistical_metrics(y_true, y_pred))
        
        # Confidence intervals
        metrics.update(self._compute_confidence_intervals(metrics, len(y_true)))
        
        return metrics
    
    def _compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
            'kappa': float(cohen_kappa_score(y_true, y_pred)),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def _compute_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        if len(np.unique(y_true)) < 2:
            return {
                'roc_auc': np.nan,
                'pr_auc': np.nan,
                'brier_score': np.nan
            }
        
        return {
            'roc_auc': float(roc_auc_score(y_true, y_proba)),
            'pr_auc': float(average_precision_score(y_true, y_proba)),
            'brier_score': float(brier_score_loss(y_true, y_proba))
        }
    
    def _compute_statistical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # Youden's J statistic
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._compute_specificity(y_true, y_pred)
        youdens_j = sensitivity + specificity - 1
        
        # F-beta scores
        f2_score = f1_score(y_true, y_pred, beta=2, zero_division=0)
        f0_5_score = f1_score(y_true, y_pred, beta=0.5, zero_division=0)
        
        return {
            'youdens_j': float(youdens_j),
            'f2_score': float(f2_score),
            'f0_5_score': float(f0_5_score)
        }
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    def _compute_confidence_intervals(self, metrics: Dict[str, float], 
                                    n_samples: int) -> Dict[str, float]:
        ci_metrics = {}
        
        # For accuracy (binomial proportion confidence interval)
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            se = np.sqrt(acc * (1 - acc) / n_samples)
            z = stats.norm.ppf((1 + self.confidence_level) / 2)
            ci_metrics['accuracy_ci'] = float(z * se)
        
        # For F1 score (approximate)
        if 'f1' in metrics:
            # Simplified CI for F1 - in practice would use bootstrap
            ci_metrics['f1_ci'] = 1.96 / np.sqrt(n_samples)  # Conservative estimate
        
        return ci_metrics
    
    def statistical_significance_test(self, metrics_a: Dict[str, float],
                                    metrics_b: Dict[str, float],
                                    n_samples_a: int, n_samples_b: int,
                                    test_type: str = 'mcnemar') -> Dict[str, float]:
        
        p_values = {}
        
        # For accuracy (assuming independent samples)
        if 'accuracy' in metrics_a and 'accuracy' in metrics_b:
            # Z-test for proportions
            p1, p2 = metrics_a['accuracy'], metrics_b['accuracy']
            n1, n2 = n_samples_a, n_samples_b
            
            pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
            z_score = (p1 - p2) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            p_values['accuracy_p_value'] = float(p_value)
        
        return p_values
    
    def compute_effect_sizes(self, metrics_a: Dict[str, float],
                           metrics_b: Dict[str, float]) -> Dict[str, float]:
        effect_sizes = {}
        
        for metric in ['accuracy', 'f1', 'roc_auc']:
            if metric in metrics_a and metric in metrics_b:
                # Cohen's d-like effect size
                diff = metrics_a[metric] - metrics_b[metric]
                # Simplified pooled standard deviation
                pooled_std = 0.1  # Conservative estimate for binary metrics
                effect_sizes[f'{metric}_effect_size'] = diff / pooled_std
        
        return effect_sizes