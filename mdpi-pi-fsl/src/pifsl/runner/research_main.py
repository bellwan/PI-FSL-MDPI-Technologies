from __future__ import annotations

import os
import traceback

# Limit MKL threads to avoid KMeans memory leak warning on Windows
os.environ["OMP_NUM_THREADS"] = "2"

import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats

import re

import json

sys.path.append(str(Path(__file__).parent.parent))

from pifsl.data.bosch_drilling import config as cfg

from pifsl.data.bosch_drilling.raw_processing.io import read_h5_tri
from pifsl.data.common.inventory import build_inventory
from pifsl.data.bosch_drilling.raw_processing.gating import collect_windows, quick_psd_embed
from pifsl.core.sampling.sampling import undersample_ok_safe
from pifsl.data.bosch_drilling.raw_processing import FewShotScalogramSet
from pifsl.core.physics_regularization import RelationNet, ConvEmbedding
from pifsl.runner.train_eval_bosch import episodic_train, evaluate_relation

# New modular components
from pifsl.runner.config_loader import ConfigLoader, ExperimentConfig
from pifsl.core.physics_regularization import PhysicsInformedRegularizer
from tools.bosch_shift_quantification import DomainShiftAnalyzer
from pifsl.core.sampling.psd_sampling import PSDGuidedSampler
from pifsl.core.sampling.dual_role_features import DualRoleFeatureStrategy
from pifsl.eval.metrics_bosch import ComprehensiveMetrics
from pifsl.viz.bosch.domain_analysis import DomainShiftVisualizer
from pifsl.viz.bosch.results_plots import ResultsVisualizer

# Pre-model / data-level diagnostics
from pifsl.viz.bosch.pre_model_plots import (
    plot_inventory_overview,
    plot_domain_shift_analysis,
    plot_feature_importance_analysis,
    plot_temporal_analysis,
    plot_psd_gating_scatter,
)

# Post-model / meta-analysis (per-scenario)
from pifsl.viz.bosch.post_model_plots import (
    plot_method_tradeoffs_radar,
    plot_statistical_significance,
    plot_fewshot_episode_preview,
    plot_probability_hist,
    plot_calibration_analysis,
    plot_computational_efficiency,
)

class PhysicsInformedFewShotExperiment:
    def __init__(self, config_dir: str = "config", output_dir: Optional[str] = None):
        self.config_loader = ConfigLoader(config_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(cfg.OUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.domain_analyzer = DomainShiftAnalyzer()
        self.metrics_calculator = ComprehensiveMetrics()

        # Experiment state
        self.current_experiment: Optional[ExperimentConfig] = None
        self.results: List[Dict[str, Any]] = []
        self.domain_shift_metrics: List[Dict[str, Any]] = []

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(cfg.DEVICE)
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def run_scenario(self, scenario_name: str, inventory: pd.DataFrame) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"Running Scenario: {scenario_name}")
        print(f"{'=' * 60}")

        # Load scenario configuration
        try:
            experiment_config = self.config_loader.load_experiment_config(scenario_name)
            self.current_experiment = experiment_config
        except Exception as e:
            print(f"Error loading configuration for {scenario_name}: {e}")
            return {}

        scenario_dir = self.output_dir

        scenario_results: Dict[str, Any] = {
            "scenario_name": scenario_name,
            "config": experiment_config,
            "runs": [],
            "domain_shift_analysis": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Run multiple seeds for statistical significance (currently 1 for speed)
        total_runs = 1
        for seed_idx, seed in enumerate(experiment_config.seeds[:total_runs]):
            print(f"\n--- Run {seed_idx + 1}/{total_runs} (Seed: {seed}) ---")
            run_result = self._run_single_execution(
                experiment_config, inventory, seed, scenario_dir
            )
            scenario_results["runs"].append(run_result)

        # Aggregate and analyze results (scenario name is used in file names)
        scenario_results = self._analyze_scenario_results(scenario_results, scenario_dir)
        self.results.append(scenario_results)

        print(f"\nCompleted scenario: {scenario_name}")
        return scenario_results

    def _run_single_execution(
        self,
        experiment_config: ExperimentConfig,
        inventory: pd.DataFrame,
        seed: int,
        output_dir: Path,
    ) -> Dict[str, Any]:
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        run_result: Dict[str, Any] = {
            "seed": seed,
            "training_performance": {},
            "testing_performance": {},
            "domain_shift_metrics": {},
            "physics_regularization_losses": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Prepare source domain data
        source_inv = self._prepare_domain_data(inventory, experiment_config.source_domains)
        if source_inv.empty:
            print("No source domain data found")
            return run_result

        X_source, y_source = collect_windows(source_inv)
        print(f"Source domain: {len(X_source)} windows, {np.sum(y_source)} worn samples")

        # Initialize feature strategy (used later by each baseline)
        feature_strategy = DualRoleFeatureStrategy()

        # Test each target domain
        for target_domain in experiment_config.target_domains:
            target_name = f"{target_domain.machine}_{target_domain.operation}"
            print(f"  Testing target: {target_name}")

            # Prepare target domain data
            target_inv = self._prepare_domain_data(inventory, [target_domain])
            if target_inv.empty:
                print(f"    No data for target domain {target_name}")
                continue

            X_target, y_target = collect_windows(target_inv)
            print(
                f"    Target domain: {len(X_target)} windows, "
                f"{np.sum(y_target)} worn samples"
            )

            # Pre-model diagnostics for this target (PSD bands, temporal stats, gating PCA)
            try:
                plot_feature_importance_analysis(
                    X_target,
                    y_target,
                    fs=cfg.FS,
                    exp_name="BOSCH_DRILLING",
                    split_tag=target_name,
                    train_name=target_name,
                )

                plot_temporal_analysis(
                    X_target,
                    y_target,
                    fs=cfg.FS,
                    exp_name="BOSCH_DRILLING",
                    split_tag=target_name,
                    train_name=target_name,
                )

                plot_psd_gating_scatter(
                    X_target,
                    y_target,
                    fs=cfg.FS,
                    exp_name="BOSCH_DRILLING",
                    split_tag=target_name,
                )
            except Exception as e:
                print(f"[WARN] extra per-target plots failed for {target_name}: {e}")

            # Quantify domain shift between source and this target
            shift_metrics = self._analyze_domain_shift(
                X_source,
                y_source,
                X_target,
                y_target,
                experiment_config.source_domains[0],
                target_domain,
            )
            run_result["domain_shift_metrics"][target_name] = shift_metrics

            # Run baseline comparisons
            baseline_results = self._run_baseline_comparisons(
                experiment_config,
                X_source,
                y_source,
                X_target,
                y_target,
                seed,
                target_name,
                output_dir,
            )
            run_result["testing_performance"][target_name] = baseline_results

        return run_result

    def _prepare_domain_data(
        self, inventory: pd.DataFrame, domains: List[Any]
    ) -> pd.DataFrame:
        if inventory.empty:
            return pd.DataFrame()

        def _norm_period_tag(tag: str) -> str:
            s = str(tag).strip()
            m = re.match(r"([A-Za-z]{3})[-_](\d{2,4})", s)
            if not m:
                return s
            mon = m.group(1).title()
            yr = m.group(2)
            if len(yr) == 2:
                yr_full = f"20{yr}"
            else:
                yr_full = yr
            return f"{mon}_{yr_full}"

        has_date_col = "Date" in inventory.columns

        if has_date_col:
            inv_date_norm = inventory["Date"].astype(str).map(_norm_period_tag)
        else:
            inv_date_norm = None

        domain_filters = []
        for domain in domains:
            machine_filter = inventory["Machine"] == domain.machine
            operation_filter = inventory["ProcessName"] == domain.operation

            if hasattr(domain, "periods") and domain.periods and has_date_col:
                norm_periods = [_norm_period_tag(p) for p in domain.periods]
                period_filter = inv_date_norm.isin(norm_periods)
                domain_filter = machine_filter & operation_filter & period_filter
            else:
                domain_filter = machine_filter & operation_filter

            domain_filters.append(domain_filter)

        if not domain_filters:
            return pd.DataFrame()

        combined = domain_filters[0]
        for flt in domain_filters[1:]:
            combined = combined | flt

        return inventory[combined].reset_index(drop=True)

    def _analyze_domain_shift(
        self,
        X_source: List[np.ndarray],
        y_source: List[int],
        X_target: List[np.ndarray],
        y_target: List[int],
        source_domain: Any,
        target_domain: Any,
    ) -> Dict[str, float]:
        if not X_source or not X_target:
            return {}

        # Extract PSD features for shift analysis
        source_psd = np.vstack([quick_psd_embed(x, cfg.FS, bins=64) for x in X_source])
        target_psd = np.vstack([quick_psd_embed(x, cfg.FS, bins=64) for x in X_target])

        source_labels = np.array(y_source)
        target_labels = np.array(y_target)

        # Compute comprehensive shift metrics
        shift_metrics = self.domain_analyzer.compute_comprehensive_shift(
            source_psd, target_psd, source_labels, target_labels
        )

        # Extra visualization of domain shift (PCA + min-distance histogram)
        try:
            train_name = f"{source_domain.machine}_{source_domain.operation}"
            test_name = f"{target_domain.machine}_{target_domain.operation}"
            plot_domain_shift_analysis(
                X_train=X_source,
                y_train=y_source,
                train_domain=train_name,
                X_test=X_target,
                y_test=y_target,
                test_domain=test_name,
                fs=cfg.FS,
                exp_name="BOSCH_DRILLING",
            )
        except Exception as e:
            print(f"[WARN] plot_domain_shift_analysis failed: {e}")

        # Store for overall analysis
        self.domain_shift_metrics.append(
            {
                "source": f"{source_domain.machine}_{source_domain.operation}",
                "target": f"{target_domain.machine}_{target_domain.operation}",
                "metrics": shift_metrics,
            }
        )

        return shift_metrics

    def _run_baseline_comparisons(
        self,
        experiment_config: ExperimentConfig,
        X_source: List[np.ndarray],
        y_source: List[int],
        X_target: List[np.ndarray],
        y_target: List[int],
        seed: int,
        target_name: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        baseline_results: Dict[str, Any] = {}

        for baseline_config in experiment_config.baselines:
            if not baseline_config.get("enabled", True):
                continue
            method_name = baseline_config["name"]
            print(f"    Method: {method_name}")

            try:
                # Train and evaluate model
                method_results = self._train_and_evaluate_method(
                    baseline_config,
                    X_source,
                    y_source,
                    X_target,
                    y_target,
                    seed,
                    experiment_config,
                )

                baseline_results[method_name] = method_results

                # Generate per-method visualizations
                if method_results.get("success", False):
                    self._generate_method_visualizations(
                        method_name, method_results, target_name, output_dir
                    )

            except Exception as e:

                print(f"      Method {method_name} failed with exception:")
                traceback.print_exc()
                baseline_results[method_name] = {
                    "success": False,
                    "error": repr(e),
                    "metrics": {},
                }

        return baseline_results

    def _train_and_evaluate_method(
        self,
        method_config: Dict[str, Any],
        X_train: List[np.ndarray],
        y_train: List[int],
        X_test: List[np.ndarray],
        y_test: List[int],
        seed: int,
        experiment_config: ExperimentConfig,
    ) -> Dict[str, Any]:
        # Initialize feature strategy
        feature_strategy = DualRoleFeatureStrategy()

        # Prepare training data
        strategy = method_config.get("strategy", "psd-guided")
        features = method_config.get("features", "cwt")

        X_balanced, y_balanced = feature_strategy.prepare_training_data(
            X_train, y_train, strategy
        )

        train_dataset = feature_strategy.create_dataset(
            X_balanced, y_balanced, features
        )

        use_transfer = bool(method_config.get("transfer_finetune", False))
        adapt_dataset = None  # default: no target adaptation

        if use_transfer:
            # Window-level split of target domain
            target_adapt_ratio = float(
                experiment_config.training_config.get("target_adapt_ratio", 0.4)
            )

            y_test_np = np.asarray(y_test, dtype=int)
            pos_idx = np.where(y_test_np == 1)[0]
            neg_idx = np.where(y_test_np == 0)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                X_eval, y_eval = X_test, y_test
            else:
                rng = np.random.RandomState(seed)
                rng.shuffle(pos_idx)
                rng.shuffle(neg_idx)

                def _split_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    n = len(indices)
                    n_adapt = max(1, int(round(target_adapt_ratio * n)))
                    # leave at least one sample for eval if possible
                    if n - n_adapt == 0 and n > 1:
                        n_adapt -= 1
                    adapt_i = indices[:n_adapt]
                    eval_i = indices[n_adapt:]
                    return adapt_i, eval_i

                pos_adapt, pos_eval = _split_indices(pos_idx)
                neg_adapt, neg_eval = _split_indices(neg_idx)

                # If eval loses a class, fallback to no split to avoid invalid metrics
                if len(pos_eval) == 0 or len(neg_eval) == 0:
                    X_eval, y_eval = X_test, y_test
                else:
                    adapt_indices = np.concatenate([pos_adapt, neg_adapt])
                    eval_indices = np.concatenate([pos_eval, neg_eval])

                    rng.shuffle(adapt_indices)
                    rng.shuffle(eval_indices)

                    X_adapt = [X_test[i] for i in adapt_indices]
                    y_adapt = [y_test[i] for i in adapt_indices]
                    X_eval = [X_test[i] for i in eval_indices]
                    y_eval = [y_test[i] for i in eval_indices]

                    adapt_dataset = feature_strategy.create_dataset(
                        X_adapt, y_adapt, features
                    )

            test_dataset = feature_strategy.create_dataset(X_eval, y_eval, features)
        else:
            # No target fine-tuning: evaluate on all target windows
            test_dataset = feature_strategy.create_dataset(X_test, y_test, features)


        # few-shot episode preview (on test split)
        method_name = method_config.get("name", "method")
        try:
            fs_cfg = experiment_config.few_shot
            plot_fewshot_episode_preview(
                ds=test_dataset,
                exp_name="BOSCH_DRILLING",
                K=fs_cfg.k_shot[0],
                Q=fs_cfg.query_size,
                seed=seed,
                split_tag=f"{experiment_config.scenario_name}__{method_name}",
            )
        except Exception as e:
            print(f"[WARN] few-shot preview failed for {method_name}: {e}")

        # Initialize model
        model = self._initialize_model(experiment_config.model_config)
        model.to(self.device)

        # Initialize physics regularizer if enabled
        physics_regularizer = None
        if method_config.get("physics_regularization", False):
            physics_regularizer = PhysicsInformedRegularizer(
                experiment_config.physics_config
            )

        # Train model on source-domain episodes
        training_metrics = self._train_model_with_physics(
            model,
            train_dataset,
            physics_regularizer,
            X_balanced,
            y_balanced,
            experiment_config,
            seed,
        )

        # transfer-learning fine-tuning on *target* domain
        if use_transfer and adapt_dataset is not None:
            print(
                "    [Transfer] Running target-domain fine-tuning for method "
                f"{method_config.get('name', '')} (window-level adapt split)"
            )
            transfer_info = self._transfer_finetune_on_target(
                model,
                adapt_dataset,            # <-- use adaptation subset ONLY
                experiment_config,
                seed,
            )
            if transfer_info:
                training_metrics.update(transfer_info)
        elif use_transfer and adapt_dataset is None:
            print(
                "    [Transfer] Skipping target-domain fine-tuning for method "
                f"{method_config.get('name', '')}: no valid adaptation subset."
            )


        # Evaluate model after (possible) transfer adaptation
        evaluation_out = self._evaluate_model(model, test_dataset, experiment_config)

        return {
            "success": True,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_out["metrics"],
            "probabilities": evaluation_out.get("probs"),
            "labels": evaluation_out.get("labels"),
            "model_info": {
                "strategy": strategy,
                "features": features,
                "physics_regularization": physics_regularizer is not None,
            },
        }

    def _transfer_finetune_on_target(
        self,
        model: nn.Module,
        target_dataset: torch.utils.data.Dataset,
        experiment_config: ExperimentConfig,
        seed: int,
    ) -> Dict[str, Any]:
        transfer_cfg = experiment_config.training_config.get("transfer_learning", {})
        if not transfer_cfg or not transfer_cfg.get("enabled", False):
            print(
                "      [Transfer] transfer_learning config disabled or missing "
                "– skipping fine-tuning."
            )
            return {}

        few_shot_cfg = experiment_config.few_shot

        # Hyperparameters with sensible fallbacks
        K = int(transfer_cfg.get("k_shot", few_shot_cfg.k_shot[0]))
        Q = int(transfer_cfg.get("query_size", few_shot_cfg.query_size))
        episodes = int(transfer_cfg.get("episodes", 100))

        base_lr = experiment_config.training_config.get("learning_rate", 1e-3)
        lr = float(transfer_cfg.get("learning_rate", base_lr * 0.1))
        weight_decay = float(
            transfer_cfg.get(
                "weight_decay",
                experiment_config.training_config.get("weight_decay", 0.0),
            )
        )
        step_size = int(
            transfer_cfg.get(
                "step_size",
                experiment_config.training_config.get("step_size", 100),
            )
        )
        gamma = float(
            transfer_cfg.get(
                "gamma",
                experiment_config.training_config.get("gamma", 0.5),
            )
        )

        print(
            f"      [Transfer] Starting target fine-tuning: "
            f"episodes={episodes}, K={K}, Q={Q}, lr={lr}"
        )

        model.to(self.device)
        model.train()

        torch.manual_seed(seed)
        np.random.seed(seed)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        transfer_losses: List[float] = []

        for episode in range(episodes):
            try:
                Sx, Sy, Qx, Qy = target_dataset.sample_episode(
                    K, Q, seed=seed + episode
                )

                Sx = Sx.to(self.device)
                Sy = Sy.to(self.device)
                Qx = Qx.to(self.device)
                Qy = Qy.to(self.device)

                optimizer.zero_grad()

                scores, classes = model.forward_episode(Sx, Sy, Qx)

                cls_sorted, _ = classes.sort()
                target = torch.zeros_like(Qy)
                for i, lab in enumerate(cls_sorted):
                    target[Qy == lab] = i

                loss = torch.nn.functional.cross_entropy(scores, target)
                loss.backward()
                optimizer.step()

                transfer_losses.append(loss.item())

                if episode % 50 == 0:
                    print(
                        f"        [Transfer] Episode {episode}: "
                        f"loss={loss.item():.4f}"
                    )

                scheduler.step()

            except Exception as e:
                print(f"        [Transfer] Episode {episode} failed: {e}")
                continue

        return {
            "transfer_episodes_completed": len(transfer_losses),
            "final_transfer_loss": float(
                np.mean(transfer_losses[-50:]) if transfer_losses else 0.0
            ),
        }

    def _initialize_model(self, model_config: Dict[str, Any]) -> nn.Module:
        model_type = model_config.get("name", "RelationNet")

        if model_type == "RelationNet":
            channels = model_config.get("channels", [32, 32, 64, 64])
            return RelationNet(channels=channels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _train_model_with_physics(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        physics_regularizer: Optional[PhysicsInformedRegularizer],
        X_raw: List[np.ndarray],
        y_raw: List[int],
        experiment_config: ExperimentConfig,
        seed: int,
    ) -> Dict[str, Any]:
        model.train()

        # Training configuration
        training_config = experiment_config.training_config
        few_shot_config = experiment_config.few_shot

        # Get raw values from config (could be str like "1e-3" or float like 0.001)
        lr_raw = training_config.get("learning_rate", cfg.LR)
        wd_raw = training_config.get("weight_decay", cfg.WEIGHT_DECAY)

        # Force them to floats so torch.optim.Adam is happy
        lr = float(lr_raw)
        weight_decay = float(wd_raw)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        episodes = few_shot_config.episodes
        K = few_shot_config.k_shot[0]  # Use first k-shot setting for training
        Q = few_shot_config.query_size

        training_losses: List[float] = []
        physics_losses: List[float] = []

        for episode in range(episodes):
            try:
                # Sample episode
                Sx, Sy, Qx, Qy = train_dataset.sample_episode(K, Q, seed=seed + episode)
                Sx, Sy, Qx, Qy = (
                    Sx.to(self.device),
                    Sy.to(self.device),
                    Qx.to(self.device),
                    Qy.to(self.device),
                )

                # Forward pass
                scores, classes = model.forward_episode(Sx, Sy, Qx)

                # Classification loss
                cls_sorted, _ = classes.sort()
                target = torch.zeros_like(Qy)
                for i, lab in enumerate(cls_sorted):
                    target[Qy == lab] = i
                classification_loss = torch.nn.functional.cross_entropy(scores, target)

                # Physics regularization loss
                physics_loss = torch.tensor(0.0, device=self.device)
                if physics_regularizer is not None:
                    with torch.no_grad():
                        if hasattr(model, "emb"):
                            feature_maps = model.emb(Sx)
                        else:
                            feature_maps = None

                    physics_loss = physics_regularizer(
                        (scores, classes), X_raw, y_raw, feature_maps, cfg.FS
                    )

                # Combined loss
                total_loss = classification_loss + physics_loss

                # Optimization step
                optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping if specified
                if training_config.get("gradient_clipping"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training_config["gradient_clipping"]
                    )

                optimizer.step()

                # Record losses
                training_losses.append(classification_loss.item())
                physics_losses.append(physics_loss.item())

                if episode % 100 == 0:
                    print(
                        f"      Episode {episode}: "
                        f"Class Loss = {classification_loss.item():.4f}, "
                        f"Physics Loss = {physics_loss.item():.4f}"
                    )

            except Exception as e:
                print(f"      Training episode {episode} failed: {e}")
                continue

        return {
            "final_classification_loss": float(
                np.mean(training_losses[-100:]) if training_losses else 0.0
            ),
            "final_physics_loss": float(
                np.mean(physics_losses[-100:]) if physics_losses else 0.0
            ),
            "training_episodes_completed": len(training_losses),
        }

    def _evaluate_model(
        self,
        model: nn.Module,
        test_dataset: torch.utils.data.Dataset,
        experiment_config: ExperimentConfig,
    ) -> Dict[str, Any]:
        model.eval()

        few_shot_config = experiment_config.few_shot
        K = few_shot_config.k_shot[0]  # Use first k-shot setting for evaluation
        Q = few_shot_config.query_size
        episodes = experiment_config.evaluation_config.get("episodes", 100)

        evaluation = evaluate_relation(
            model,
            test_dataset,
            K=K,
            Q=Q,
            episodes=episodes,
            seed=experiment_config.seeds[0],
            device=str(self.device),
            thresh=experiment_config.evaluation_config.get("threshold", cfg.THRESH),
        )

        # Backwards compatible:
        metrics = evaluation
        probs = None
        labels = None

        if isinstance(evaluation, dict):
            # Allow evaluate_relation to optionally include raw outputs
            probs = evaluation.get("probs")
            labels = evaluation.get("labels")
            if "metrics" in evaluation:
                metrics = evaluation["metrics"]

        return {
            "metrics": metrics,
            "probs": probs,
            "labels": labels,
        }

    def _generate_method_visualizations(
        self,
        method_name: str,
        method_results: Dict[str, Any],
        target_name: str,
        output_dir: Path,
    ):
        try:
            domain_visualizer = DomainShiftVisualizer()
            results_visualizer = ResultsVisualizer()

            base = f"{target_name}_{method_name}".replace(" ", "_")

            # Domain shift visualization (if available)
            if hasattr(self, "domain_shift_metrics") and self.domain_shift_metrics:
                domain_visualizer.plot_comprehensive_shift_analysis(
                    self.domain_shift_metrics,
                    output_dir / f"method_{base}_domain_shift.png",
                )

            # Performance visualization for this method
            if method_results.get("evaluation_metrics"):
                results_visualizer.plot_method_comparison(
                    {method_name: method_results["evaluation_metrics"]},
                    output_dir / f"method_{base}_performance.png",
                )

            # Probability histogram & calibration (if raw outputs are available)
            probs = method_results.get("probabilities")
            labels = method_results.get("labels")
            if probs is not None and labels is not None:
                try:
                    if self.current_experiment and self.current_experiment.source_domains:
                        src = self.current_experiment.source_domains[0]
                        train_name = f"{src.machine}_{src.operation}"
                    else:
                        train_name = "source"

                    test_name = target_name
                    exp_name = (
                        self.current_experiment.scenario_name
                        if self.current_experiment is not None
                        else "experiment"
                    )

                    plot_probability_hist(
                        probs=probs,
                        labels=labels,
                        exp_name=exp_name,
                        train_name=train_name,
                        test_name=test_name,
                    )

                    plot_calibration_analysis(
                        all_probs=probs,
                        all_labels=labels,
                        exp_name=exp_name,
                        train_name=train_name,
                        test_name=test_name,
                    )
                except Exception as e:
                    print(f"      Probability / calibration plots failed: {e}")

        except Exception as e:
            print(f"      Visualization generation failed: {e}")

    def _analyze_scenario_results(
        self, scenario_results: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Any]:
        runs = scenario_results["runs"]
        if not runs:
            return scenario_results

        # Aggregate performance across seeds
        performance_aggregates: Dict[str, Dict[str, Any]] = {}

        for run in runs:
            for target_domain, methods in run["testing_performance"].items():
                if target_domain not in performance_aggregates:
                    performance_aggregates[target_domain] = {}

                for method_name, method_results in methods.items():
                    if method_name not in performance_aggregates[target_domain]:
                        performance_aggregates[target_domain][method_name] = {
                            "f1_scores": [],
                            "accuracy_scores": [],
                            "success_count": 0,
                        }

                    if method_results.get("success", False):
                        metrics = method_results.get("evaluation_metrics", {})
                        performance_aggregates[target_domain][method_name][
                            "f1_scores"
                        ].append(metrics.get("f1", 0.0))
                        performance_aggregates[target_domain][method_name][
                            "accuracy_scores"
                        ].append(metrics.get("acc", 0.0))
                        performance_aggregates[target_domain][method_name][
                            "success_count"
                        ] += 1

        # Build per-run results DataFrame for post-model meta-analysis
        rows = []
        for target_domain, methods in performance_aggregates.items():
            for method_name, aggregates in methods.items():
                for f1 in aggregates["f1_scores"]:
                    rows.append(
                        {
                            "target": target_domain,
                            "method": method_name,
                            "f1": float(f1),
                        }
                    )

        if rows:
            try:
                results_df = pd.DataFrame(rows)
                # Scenario-level meta plots (radar & F1 distribution)
                plot_method_tradeoffs_radar(results_df)
                plot_statistical_significance(results_df)
            except Exception as e:
                print(f"[WARN] scenario-level post-model plots failed: {e}")

        # Compute statistics (for JSON/CSV + summary)
        scenario_results["performance_statistics"] = {}
        for target_domain, methods in performance_aggregates.items():
            scenario_results["performance_statistics"][target_domain] = {}

            for method_name, aggregates in methods.items():
                if aggregates["success_count"] > 0:
                    f1_scores = aggregates["f1_scores"]
                    accuracy_scores = aggregates["accuracy_scores"]

                    scenario_results["performance_statistics"][target_domain][
                        method_name
                    ] = {
                        "f1_mean": float(np.mean(f1_scores)),
                        "f1_std": float(np.std(f1_scores)),
                        "f1_ci": float(
                            1.96 * np.std(f1_scores) / np.sqrt(len(f1_scores))
                        ),
                        "accuracy_mean": float(np.mean(accuracy_scores)),
                        "accuracy_std": float(np.std(accuracy_scores)),
                        "n_successful_runs": aggregates["success_count"],
                    }

        # Generate comprehensive scenario-level visualizations
        self._generate_scenario_visualizations(scenario_results, output_dir)

        return scenario_results

    def _generate_scenario_visualizations(
        self, scenario_results: Dict[str, Any], output_dir: Path
    ):
        try:
            visualizer = ResultsVisualizer()
            scenario_name = scenario_results.get("scenario_name", "scenario")
            scenario_tag = scenario_name.replace(" ", "_")

            # Performance comparison across methods
            if "performance_statistics" in scenario_results:
                visualizer.plot_scenario_performance_comparison(
                    scenario_results,
                    output_dir / f"scenario_{scenario_tag}_performance_comparison.png",
                )

            # Domain shift analysis vs performance
            if hasattr(self, "domain_shift_metrics") and self.domain_shift_metrics:
                domain_visualizer = DomainShiftVisualizer()
                domain_visualizer.plot_domain_shift_correlation(
                    self.domain_shift_metrics,
                    output_dir / f"scenario_{scenario_tag}_domain_shift_correlation.png",
                )

        except Exception as e:
            print(f"Scenario visualization generation failed: {e}")


    def run_ablation_studies(self, scenario_name: str, inventory: pd.DataFrame):
        print(f"\n🔬 Running Ablation Studies for {scenario_name}")

        # Load scenario configuration
        try:
            experiment_config = self.config_loader.load_experiment_config(scenario_name)
        except Exception as e:
            print(f"Error loading configuration for ablation study: {e}")
            return

        ablation_results: Dict[str, Any] = {}

        for ablation_config in experiment_config.ablation_studies:
            component = ablation_config["component"]
            variations = ablation_config["variations"]
            description = ablation_config.get("description", "")

            print(f"\nAblating: {component} - {description}")
            ablation_results[component] = {}

            for variation in variations:
                print(f"  Variation: {variation}")

                # Modify configuration for ablation (simulation only for now)
                modified_config = self._modify_config_for_ablation(
                    experiment_config, component, variation
                )

                ablation_results[component][str(variation)] = {
                    "description": f"{component} = {variation}",
                    "status": "simulated",
                }

        return ablation_results

    def _modify_config_for_ablation(
        self, config: ExperimentConfig, component: str, variation: Any
    ) -> ExperimentConfig:
        modified_config = ExperimentConfig(
            scenario_name=config.scenario_name + f"_ablation_{component}_{variation}",
            description=config.description,
            seeds=config.seeds,
            source_domains=config.source_domains,
            target_domains=config.target_domains,
            few_shot=config.few_shot,
            model_config=config.model_config.copy(),
            training_config=config.training_config.copy(),
            physics_config=config.physics_config,
            evaluation_config=config.evaluation_config.copy(),
            baselines=config.baselines.copy(),
            ablation_studies=config.ablation_studies,
        )

        if component == "physics_regularization":
            modified_config.physics_config.enabled = variation
        elif component == "psd_sampling":
            for baseline in modified_config.baselines:
                if baseline.get("strategy") == "psd-guided":
                    baseline["enabled"] = variation

        return modified_config

    def save_results(self, filename: Optional[str] = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}"

        results_file = self.output_dir / f"{filename}.json"

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "scenario_name": result["scenario_name"],
                "timestamp": result["timestamp"],
                "performance_statistics": result.get("performance_statistics", {}),
            }
            serializable_results.append(serializable_result)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        # Save domain shift metrics (optional)
        if self.domain_shift_metrics:
            shift_file = self.output_dir / f"{filename}_domain_shifts.csv"
            shift_data = []
            for metric in self.domain_shift_metrics:
                record = {
                    "source": metric["source"],
                    "target": metric["target"],
                }
                record.update(metric["metrics"])
                shift_data.append(record)

            pd.DataFrame(shift_data).to_csv(shift_file, index=False)

        print(f"\nResults saved to: {results_file}")

    def generate_final_report(self):
        print(f"\nGenerating Final Report")

        report_dir = self.output_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate summary statistics
        self._generate_summary_statistics(report_dir)

        # Generate comparative analysis
        self._generate_comparative_analysis(report_dir)

        # Generate domain shift analysis
        self._generate_domain_shift_report(report_dir)

        print(f"Final report generated in: {report_dir}")

    def _generate_summary_statistics(self, report_dir: Path):
        summary_data = []

        for scenario_result in self.results:
            scenario_name = scenario_result["scenario_name"]
            stats_dict = scenario_result.get("performance_statistics", {})

            for target_domain, methods in stats_dict.items():
                for method_name, method_stats in methods.items():
                    summary_data.append(
                        {
                            "scenario": scenario_name,
                            "target_domain": target_domain,
                            "method": method_name,
                            "f1_mean": method_stats["f1_mean"],
                            "f1_std": method_stats["f1_std"],
                            "accuracy_mean": method_stats["accuracy_mean"],
                            "n_runs": method_stats["n_successful_runs"],
                        }
                    )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = report_dir / "final_summary_statistics.csv"
            summary_df.to_csv(summary_file, index=False)

    def _generate_comparative_analysis(self, report_dir: Path) -> None:
        records = []

        for scenario_result in self.results:
            scenario_name = scenario_result.get("scenario_name", "unknown")

            for run in scenario_result.get("runs", []):
                seed = run.get("seed")
                testing_perf = run.get("testing_performance", {})

                for target_domain, methods in testing_perf.items():
                    for method_name, method_result in methods.items():
                        if not method_result.get("success", False):
                            continue

                        metrics = method_result.get("evaluation_metrics", {}) or {}

                        records.append(
                            {
                                "scenario": scenario_name,
                                "target_domain": target_domain,
                                "method": method_name,
                                "seed": seed,
                                # core classification metrics
                                "f1": metrics.get("f1", np.nan),
                                "accuracy": metrics.get("acc", np.nan),
                                "balanced_accuracy": metrics.get("bacc", np.nan),
                                "precision": metrics.get("prec", np.nan),
                                "recall": metrics.get("rec", np.nan),
                                "roc_auc": metrics.get("roc_auc", np.nan),
                                "pr_auc": metrics.get("pr_auc", np.nan),
                                "brier": metrics.get("brier", np.nan),
                                "mcc": metrics.get("mcc", np.nan),
                                # confusion statistics
                                "tp": metrics.get("tp", np.nan),
                                "fp": metrics.get("fp", np.nan),
                                "tn": metrics.get("tn", np.nan),
                                "fn": metrics.get("fn", np.nan),
                                "n_samples": metrics.get("n", np.nan),
                            }
                        )

        if not records:
            return

        per_run_df = pd.DataFrame(records)
        per_run_file = report_dir / "final_per_run_metrics.csv"
        per_run_df.to_csv(per_run_file, index=False)

        agg_df = (
            per_run_df.groupby(["scenario", "method"])
            .agg(
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                n=("f1", "count"),
                tp_sum=("tp", "sum"),
                fp_sum=("fp", "sum"),
                tn_sum=("tn", "sum"),
                fn_sum=("fn", "sum"),
            )
            .reset_index()
        )
        agg_file = report_dir / "final_comparative_performance.csv"
        agg_df.to_csv(agg_file, index=False)

        methods = sorted(per_run_df["method"].dropna().unique())
        if len(methods) >= 2:
            comparisons = []
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    m1, m2 = methods[i], methods[j]
                    df1 = per_run_df[per_run_df["method"] == m1]
                    df2 = per_run_df[per_run_df["method"] == m2]

                    merged = df1.merge(
                        df2,
                        on=["scenario", "target_domain", "seed"],
                        suffixes=("_" + m1, "_" + m2),
                    )
                    if merged.empty:
                        continue

                    f1_1 = merged["f1_" + m1].values
                    f1_2 = merged["f1_" + m2].values

                    t_stat, p_val = stats.ttest_rel(f1_1, f1_2)
                    comparisons.append(
                        {
                            "method_1": m1,
                            "method_2": m2,
                            "n_pairs": int(len(f1_1)),
                            "t_stat": float(t_stat),
                            "p_value": float(p_val),
                        }
                    )

            if comparisons:
                comp_df = pd.DataFrame(comparisons)
                comp_file = report_dir / "final_method_comparisons_ttest.csv"
                comp_df.to_csv(comp_file, index=False)

        try:
            # Average across scenarios → one bar per method
            global_agg = (
                per_run_df.groupby("method")
                .agg(
                    f1_mean=("f1", "mean"),
                    f1_std=("f1", "std"),
                    n=("f1", "count"),
                )
                .reset_index()
            )

            order = global_agg.sort_values("f1_mean", ascending=False)

            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                order["method"],
                order["f1_mean"],
                yerr=order["f1_std"],
                capsize=4,
                label="Mean F1-score ± 1σ",
            )

            plt.xlabel("Method", fontsize=12)
            plt.ylabel("F1-score (dimensionless)", fontsize=12)
            plt.title(
                "Comparative Tool-Condition Classification Performance",
                fontsize=14,
            )
            plt.ylim(0.0, 1.05)
            plt.xticks(rotation=45, ha="right", fontsize=11)
            plt.yticks(fontsize=11)
            plt.grid(axis="y", linestyle="--", alpha=0.4)
            plt.legend(fontsize=11)

            # Annotate bars with numeric F1
            for bar, mean in zip(bars, order["f1_mean"]):
                h = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.02,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.tight_layout()
            plot_file = report_dir / "final_comparative_f1_barplot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Comparative analysis plotting failed: {e}")


    def _generate_domain_shift_report(self, report_dir: Path):
        metrics_list = getattr(self, "domain_shift_metrics", None)
        if not metrics_list:
            return

        records = []
        for entry in metrics_list:
            metrics = entry.get("metrics", {})
            record = {
                "source": entry.get("source"),
                "target": entry.get("target"),
            }
            record.update(metrics)
            records.append(record)

        if not records:
            return

        shift_df = pd.DataFrame(records)
        shift_file = report_dir / "final_domain_shift_analysis.csv"
        shift_df.to_csv(shift_file, index=False)


def main():
    print("Physics-Informed Few-Shot Learning Experiment Framework")

    full_out_dir = Path(cfg.OUT_DIR) / "experiment"
    full_out_dir.mkdir(parents=True, exist_ok=True)

    experiment_runner = PhysicsInformedFewShotExperiment(
        config_dir="config",
        output_dir=str(full_out_dir),
    )

    # Load inventory data
    print("\nLoading inventory data...")
    inventory = build_inventory(cfg.DATASET_DIR, cfg.OPS_IN_SCOPE)

    if inventory.empty:
        print("No inventory data found!")
        return

    print(f"Loaded inventory with {len(inventory)} records")

    # Extra dataset-level overview plots (one-time, pre-model)
    try:
        plot_inventory_overview(inventory)
    except Exception as e:
        print(f"[WARN] plot_inventory_overview failed: {e}")

    # Define scenarios to run
    scenarios = [
        "E1_cross_machine",
        "E2_cross_operation",
        "E3_cross_machine_operation",
        "E4_multi_source_to_target",
    ]

    # Run experiments for each scenario
    for scenario in scenarios:
        try:
            experiment_runner.run_scenario(scenario, inventory)
            print(f"Completed scenario: {scenario}")
        except Exception as e:
            print(f"Scenario {scenario} failed: {e}")

     # Save all results
    experiment_runner.save_results()

    # Generate final report
    experiment_runner.generate_final_report()

    try:
        final_dir = full_out_dir  # Path(cfg.OUT_DIR) / "experiment"
        per_run_path = final_dir / "final_per_run_metrics.csv"

        if per_run_path.exists():
            per_run = pd.read_csv(per_run_path)

            # Only proceed if timing columns exist
            if {"method", "train_time_s", "eval_time_s"}.issubset(per_run.columns):
                timing_dict = {}
                for method, g in per_run.groupby("method"):
                    timing_dict[method] = {
                        "train": float(g["train_time_s"].mean()),
                        "infer": float(g["eval_time_s"].mean()),
                    }

                plot_computational_efficiency(timing_dict)
            else:
                print(
                    "[WARN] final_per_run_metrics.csv has no train_time_s / eval_time_s; "
                    "skipping postmodel_computational_efficiency.png"
                )
        else:
            print(f"[WARN] {per_run_path} not found; skipping computational efficiency plot.")
    except Exception as e:
        print(f"[WARN] Failed to generate computational efficiency plot: {e}")

    print(f"\nExperiment Framework Completed Successfully!")
    print(f"Results available in: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()
