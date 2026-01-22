from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class PhysicsRegularizationConfig:
    enabled: bool = True

    # Main (vibration) losses
    lambda_energy: float = 0.1
    lambda_spectral: float = 0.1
    lambda_envelope: float = 0.05
    spectral_bands: Optional[Dict[str, List[float]]] = None

    # Optional motor-current loss (MCSA-style spectral consistency)
    motor_current_enabled: bool = False
    lambda_current: float = 0.1
    current_key: str = "motor_current"
    current_spectral_bands: Optional[Dict[str, List[float]]] = None

    def __post_init__(self):
        if self.spectral_bands is None:
            # Generic defaults (safe; datasets can override via YAML)
            self.spectral_bands = {
                "low": [0.0, 75.0],
                "mid": [75.0, 300.0],
                "high": [300.0, 1000.0],
            }
        if self.current_spectral_bands is None:
            self.current_spectral_bands = dict(self.spectral_bands)


RawSignal = Union[np.ndarray, Dict[str, np.ndarray]]


class PhysicsInformedRegularizer(nn.Module):
    def __init__(self, cfg: PhysicsRegularizationConfig):
        super().__init__()
        self.cfg = cfg

        self.lambda_energy = float(cfg.lambda_energy)
        self.lambda_spectral = float(cfg.lambda_spectral)
        self.lambda_envelope = float(cfg.lambda_envelope)

        self.motor_current_enabled = bool(getattr(cfg, 'motor_current_enabled', False))
        self.lambda_current = float(getattr(cfg, 'lambda_current', 0.0))
        self.current_key = str(getattr(cfg, 'current_key', 'motor_current'))

        self.register_buffer("energy_loss_accum", torch.tensor(0.0))
        self.register_buffer("spectral_loss_accum", torch.tensor(0.0))
        self.register_buffer("envelope_loss_accum", torch.tensor(0.0))
        self.register_buffer("current_loss_accum", torch.tensor(0.0))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))

    # ------------------------- utilities -------------------------

    @staticmethod
    def _to_1d(x: Any) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        return a

    @staticmethod
    def _extract(signal: RawSignal, key: str, fallback_first: bool = True) -> Optional[np.ndarray]:
        if isinstance(signal, dict):
            if key in signal:
                return PhysicsInformedRegularizer._to_1d(signal[key])
            if fallback_first and len(signal) > 0:
                # take first available modality
                return PhysicsInformedRegularizer._to_1d(next(iter(signal.values())))
            return None
        return PhysicsInformedRegularizer._to_1d(signal)

    @staticmethod
    def _fft_band_energies(x: np.ndarray, fs: float, bands: Dict[str, List[float]]) -> np.ndarray:
        # Guard
        if x.size < 8 or fs <= 0:
            return np.zeros((len(bands),), dtype=np.float32)

        # Remove DC
        x = x - float(np.mean(x))
        # FFT
        X = np.fft.rfft(x)
        P = (np.abs(X) ** 2).astype(np.float64)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / float(fs))

        energies = []
        for _, (f0, f1) in bands.items():
            f0 = float(f0); f1 = float(f1)
            mask = (freqs >= f0) & (freqs < f1)
            e = float(P[mask].sum()) if np.any(mask) else 0.0
            energies.append(e)
        arr = np.asarray(energies, dtype=np.float32)
        s = float(arr.sum()) + 1e-12
        return (arr / s).astype(np.float32)

    # ------------------------- loss terms -------------------------

    def _within_class_variance(self, feats: torch.Tensor, labels: List[int]) -> torch.Tensor:
        device = feats.device
        y = torch.tensor(labels, dtype=torch.long, device=device)
        if y.numel() != feats.size(0):
            y = y[: feats.size(0)]
        classes = torch.unique(y)
        if classes.numel() <= 1:
            return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        cnt = 0
        for c in classes:
            m = (y == c)
            if int(m.sum().item()) < 2:
                continue
            fc = feats[m]
            loss = loss + fc.var(dim=0, unbiased=False).mean()
            cnt += 1
        return loss / (cnt if cnt > 0 else 1.0)

    def _energy_distribution_constraint(self, feature_maps: torch.Tensor, labels: List[int]) -> torch.Tensor:
        # feature_maps: (N, C, H, W)
        device = feature_maps.device
        N = feature_maps.size(0)
        if N < 2:
            return torch.tensor(0.0, device=device)

        # Per-sample energy per channel (N, C)
        e = (feature_maps ** 2).mean(dim=(2, 3))
        return self._within_class_variance(e, labels)

    def _spectral_band_ratio_constraint(
        self,
        raw_signals: List[RawSignal],
        labels: List[int],
        fs: float,
        bands: Dict[str, List[float]],
        key: str = "vibration",
    ) -> torch.Tensor:
        device = self.energy_loss_accum.device
        vecs = []
        for s in raw_signals:
            x = self._extract(s, key=key, fallback_first=True)
            if x is None:
                continue
            vecs.append(self._fft_band_energies(x, fs, bands))
        if len(vecs) < 2:
            return torch.tensor(0.0, device=device)

        feats = torch.tensor(np.stack(vecs, axis=0), device=device, dtype=torch.float32)
        # Align labels to available vecs (best-effort)
        labels2 = labels[: feats.size(0)] if labels else [0] * feats.size(0)
        return self._within_class_variance(feats, labels2)

    def _envelope_consistency_constraint(self, raw_signals: List[RawSignal], fs: float, key: str = "vibration") -> torch.Tensor:
        # Simple smoothness prior on the analytic-signal envelope
        try:
            from scipy.signal import hilbert
        except Exception:

            return torch.tensor(0.0, device=self.energy_loss_accum.device)

        device = self.energy_loss_accum.device
        vals = []
        for s in raw_signals:
            x = self._extract(s, key=key, fallback_first=True)
            if x is None or x.size < 16:
                continue
            env = np.abs(hilbert(x))
            d = np.diff(env)
            vals.append(float(np.mean(d * d)))
        if not vals:
            return torch.tensor(0.0, device=device)
        return torch.tensor(float(np.mean(vals)), device=device, dtype=torch.float32)

    # ------------------------- forward -------------------------

    def forward(
        self,
        pred_outputs: Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, ...],
        raw_signals: List[RawSignal],
        labels: List[int],
        feature_maps: Optional[torch.Tensor],
        fs: float,
    ) -> torch.Tensor:
        if not self.cfg.enabled:
            return torch.tensor(0.0, device=pred_outputs[0].device if pred_outputs else "cpu")

        fs = float(fs)
        device = pred_outputs[0].device if pred_outputs else self.energy_loss_accum.device

        total = torch.tensor(0.0, device=device)

        if self.lambda_energy > 0 and feature_maps is not None:
            el = self._energy_distribution_constraint(feature_maps, labels)
            total = total + self.lambda_energy * el
            self.energy_loss_accum += el.detach()

        if self.lambda_spectral > 0:
            sl = self._spectral_band_ratio_constraint(
                raw_signals=raw_signals,
                labels=labels,
                fs=fs,
                bands=self.cfg.spectral_bands or {},
                key="vibration",
            )
            total = total + self.lambda_spectral * sl
            self.spectral_loss_accum += sl.detach()

        if self.lambda_envelope > 0:
            al = self._envelope_consistency_constraint(raw_signals, fs=fs, key="vibration")
            total = total + self.lambda_envelope * al
            self.envelope_loss_accum += al.detach()

        if self.motor_current_enabled and self.lambda_current > 0:
            cfg = getattr(self, "cfg", None)
            bands = {}
            if cfg is not None:
                bands = getattr(cfg, "current_spectral_bands", None) or {}

            if bands:
                cl = self._spectral_band_ratio_constraint(
                    raw_signals=raw_signals,
                    labels=labels,
                    fs=fs,
                    bands=bands,
                    key=self.current_key,
                )
                total = total + self.lambda_current * cl
                self.current_loss_accum += cl.detach()

        self.steps += 1
        return total

    def metrics(self) -> Dict[str, float]:
        denom = float(self.steps.item()) if int(self.steps.item()) > 0 else 1.0
        return {
            "phys_energy": float(self.energy_loss_accum.item() / denom),
            "phys_spectral": float(self.spectral_loss_accum.item() / denom),
            "phys_envelope": float(self.envelope_loss_accum.item() / denom),
            "phys_current": float(self.current_loss_accum.item() / denom),
        }

    def reset(self):
        self.energy_loss_accum.zero_()
        self.spectral_loss_accum.zero_()
        self.envelope_loss_accum.zero_()
        self.current_loss_accum.zero_()
        self.steps.zero_()
