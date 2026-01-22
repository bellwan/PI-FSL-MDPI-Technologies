
from __future__ import annotations
import os
import torch
import numpy as np
from pathlib import Path

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = os.environ.get("PI_FSL_DATASET_DIR", str(_REPO_ROOT / "data"))
OUT_DIR     = os.environ.get("PI_FSL_OUT_DIR",     str(_REPO_ROOT / "outputs"))

os.makedirs(OUT_DIR, exist_ok=True)


# IO / dataset
ACC_KEY       = "vibration_data"
FS            = 2000.0
OPS_IN_SCOPE  = ["OP05","OP07"]
MACHINES      = ["M01","M02","M03"]

# Windowing / gating
WINDOW_SPECS   = [(1000, 500), (512, 256)]
ENV_LOWPASS_HZ = 10.0
TRI_thrE_lo_q  = 0.40
TRI_thrR_q     = 0.90

# CWT image
CWT_WAVELET  = "morl"
CWT_SCALES   = np.linspace(2, 128, 64)
IMG_H, IMG_W = 64, 64

# Undersampling (Condensed NN) + PSD embed
U_CNN_MAX_ITERS = 20
U_CNN_K         = 1
U_CNN_SEED      = 42
PSD_BINS        = 64

# Training / eval (RelationNet)
SEED             = 123
EPISODES_TRAIN   = 1200
K_SHOTS          = 3
Q_PER_CLASS      = 8
EMBED_CHANNELS   = [32,32,64,64]
LR               = 1e-3
STEP_LR_GAMMA    = 0.5
STEP_LR_EVERY    = 400
WEIGHT_DECAY     = 0.0
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
THRESH           = 0.5
RANDOM_STATE     = 42

# Spindle/order norm (few-shot part)
OP_SPEED_HZ = {"OP05":200, "OP07":200}
ORDER_BASE_HZ = 200.0
ORDER_NORM = {op: (ORDER_BASE_HZ / OP_SPEED_HZ[op]) for op in OP_SPEED_HZ}

# ========================== CPD / CYCLE SCRIPT CONSTANTS ==========================
BASE_PATH   = DATASET_DIR
SAVE_DIR    = os.path.join(OUT_DIR, "plots_cpd_cycle")
RESULTS_DIR = OUT_DIR
WIN_S       = 2000
PSD_MAX_HZ  = 1000.0
SCALOGRAM_SIZE = 64
CPD_SEG_S   = 20
ROLL_STEP_S = 20
N_SEGMENTS_PER_STATUS = 10
SAVE_DPI   = 160
RNG_SEED   = 42

# richer OP_SPEED_HZ for CPD (extend)
OP_SPEED_HZ.update({
    "OP00": 250, "OP01": 250, "OP02": 200, "OP03": 250, "OP04": 250,
    "OP06": 250, "OP08": 250, "OP09": 250,
    "OP10": 250, "OP11": 250, "OP12": 250, "OP13": 75,  "OP14": 250,
})

DOMAIN_FILTER = {
    "M01_OP05", "M01_OP07", "M02_OP05", "M02_OP07", "M03_OP07"
}
GEN_TIER1 = True
GEN_TIER2 = True
GEN_CPD   = True
GEN_CYCLE = True

# ========================== PLOTS (phase 3.0) ==========================
GENERATE_PLOTS = True
PLOTS_DIR      = os.path.join(OUT_DIR, "plots_phase42")
FEWSHOT_K      = K_SHOTS
FEWSHOT_Q      = Q_PER_CLASS
COMBOS_TO_PLOT = [
    ("M01","OP05"), ("M01","OP07"),
    ("M02","OP05"), ("M02","OP07"),
    ("M03","OP07")
]
