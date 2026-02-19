import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    machine: str
    operation: str
    periods: List[str]
    description: str = ""


@dataclass
class FewShotConfig:
    n_way: int = 2
    k_shot: Optional[List[int]] = None
    query_size: int = 15
    episodes: int = 1000
    validation_episodes: int = 100

    def __post_init__(self):
        if self.k_shot is None:
            self.k_shot = [1, 3, 5]


@dataclass
class PhysicsRegularizationConfig:
    # Backwards-compatible defaults 
    enabled: bool = True
    lambda_energy: float = 0.1
    lambda_spectral: float = 0.1
    lambda_envelope: float = 0.05
    spectral_bands: Optional[Dict[str, List[float]]] = None

    # motor-current loss (MCSA-style spectral consistency)
    motor_current_enabled: bool = False
    lambda_current: float = 0.1
    current_key: str = "motor_current"
    current_spectral_bands: Optional[Dict[str, List[float]]] = None

    def __post_init__(self):
        # Default vibration bands (safe generic defaults)
        if self.spectral_bands is None:
            self.spectral_bands = {
                "low": [0.0, 50.0],
                "mid": [50.0, 200.0],
                "high": [200.0, 800.0],
            }

        if self.current_spectral_bands is None:
            self.current_spectral_bands = dict(self.spectral_bands)


@dataclass
class ExperimentConfig:
    scenario_name: str
    description: str
    seeds: List[int]
    source_domains: List[DomainConfig]
    target_domains: List[DomainConfig]
    few_shot: FewShotConfig
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    physics_config: PhysicsRegularizationConfig = field(default_factory=PhysicsRegularizationConfig)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    baselines: List[Dict[str, Any]] = field(default_factory=list)
    ablation_studies: List[Dict[str, Any]] = field(default_factory=list)


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)

    def load_experiment_config(self, scenario: str) -> ExperimentConfig:
        scenario_yaml = f"{scenario}.yaml"

        project_root = Path(__file__).resolve().parents[2]
        this_dir = Path(__file__).resolve().parent

        candidates = [
            # Preferred: yaml files live alongside this module in bosch/experiments/
            this_dir / scenario_yaml,

            # Common layout: config/experiments/<scenario>.yaml
            project_root / "config" / "experiments" / scenario_yaml,
            self.config_dir / "experiments" / scenario_yaml,
        ]

        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError(
                "Configuration file not found. Tried:\n" + "\n".join(str(p) for p in candidates)
            )

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        return self._parse_config(raw_config, scenario)

    def _parse_config(self, raw_config: Dict[str, Any], scenario: str) -> ExperimentConfig:
        # Validate required fields
        required_fields = ["description", "seeds", "source_domains", "target_domains"]
        for field in required_fields:
            if field not in raw_config:
                raise ValueError(f"Missing required field: {field} in {scenario}")

        # Parse domains
        source_domains = [DomainConfig(**d) for d in raw_config["source_domains"]]
        target_domains = [DomainConfig(**d) for d in raw_config["target_domains"]]

        # Few-shot config
        few_shot_raw = raw_config.get("few_shot_setting", {})
        few_shot_config = FewShotConfig(**few_shot_raw)

        # Physics config
        physics_raw = raw_config.get("physics_regularization", {})
        physics_config = PhysicsRegularizationConfig(**physics_raw)

        return ExperimentConfig(
            scenario_name=scenario,
            description=raw_config["description"],
            seeds=raw_config["seeds"],
            source_domains=source_domains,
            target_domains=target_domains,
            few_shot=few_shot_config,
            model_config=raw_config.get("model_config", {}),
            training_config=raw_config.get("training_config", {}),
            physics_config=physics_config,
            evaluation_config=raw_config.get("evaluation_config", {}),
            baselines=raw_config.get("baselines", []),
            ablation_studies=raw_config.get("ablation_studies", []),
        )

    def save_experiment_config(self, config: ExperimentConfig, scenario: str):
        """Save an ExperimentConfig back to YAML under <config_dir>/experiments/."""
        exp_dir = self.config_dir / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        config_path = exp_dir / f"{scenario}.yaml"

        config_dict = {
            "description": config.description,
            "seeds": config.seeds,
            "source_domains": [vars(d) for d in config.source_domains],
            "target_domains": [vars(d) for d in config.target_domains],
            "few_shot_setting": vars(config.few_shot),
            "model_config": config.model_config,
            "training_config": config.training_config,
            "physics_regularization": vars(config.physics_config),
            "evaluation_config": config.evaluation_config,
            "baselines": config.baselines,
            "ablation_studies": config.ablation_studies,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
