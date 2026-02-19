## Repository layout

- `src/pifsl/`  
  PI-FSL, baseline methods, dataset loaders, runners, and evaluation utilities.

- `configs/`  
  Dataset/output paths and experiment definitions.

- `scripts/`  
  Reproduction entrypoints for the manuscript experiments.

- `tools/`  
  utility script: `bosch_shift_quantification.py`.

## Setup

### 1) Create an isolated environment (recommended)

From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 2) Install dependencies

Install Python packages (from repository root):

```bash
pip install -r mdpi-pi-fsl/requirements.txt
pip install -e mdpi-pi-fsl
```

Notes:
- Python 3.10+ is recommended.
- GPU is optional. If using CUDA, install a compatible PyTorch build for the system.
- If `learn2learn` fails to install on a platform, use a matching PyTorch/Python combination or install build tools for the OS.

### 3) Configure dataset and output paths

Edit `mdpi-pi-fsl/configs/paths.yaml`:
- Set dataset roots under `datasets:`
- (Optional) set output locations under `artifacts:`

Quick sanity check (verifies the package can import):

```bash
python -c "import pifsl; print('pifsl import ok')"
```
## Reproducing manuscript experiments

All scripts print the executed command lines to the console.
### Running scripts reliably

All manuscript scripts live in `mdpi-pi-fsl/scripts/`. Run them from the repository root so relative paths work:

```bash
python mdpi-pi-fsl/scripts/<script_name>.py
```

If a script depends on the configured dataset paths, confirm `mdpi-pi-fsl/configs/paths.yaml` is correct first.

Re-running:
- If an output directory already exists, some scripts may skip computation or append results depending on the script.
- If results look stale, delete the corresponding artifact folder under the configured `artifacts:` paths and rerun.


### Phase A: Bosch drilling benchmark (E1 to E4)

Runs cross-machine, cross-operation, combined shift, and multi-source to target experiments.

Run:
`python scripts/10_phase_a_run_bosch_e1_e4.py --paths configs/paths.yaml --method pi_fsl --variant full`

Output JSONL: `artifacts/results/jsonl/phase_a_bosch_e1_e4.jsonl`

### Phase B: Swap diagnostic (Bosch and CWRU)

Runs the swap diagnostic experiment set defined in `configs/paths.yaml` (default: `E9`).

Run:
`python scripts/20_phase_b_run_swap_bosch_cwru.py --paths configs/paths.yaml`

Output JSONL: `artifacts/results/jsonl/phase_b_swap.jsonl`

### Phase C: Support-suite (5 methods across multiple datasets)

Runs the unified 2-way 5-shot protocol comparing RelationNet, ProtoNet, MatchingNet, MAML, and PI-FSL.

Run:
`python scripts/30_phase_c_run_support_suite_5x4.py --paths configs/paths.yaml --make_tables`

Output JSONL: `artifacts/results/jsonl/support_suite_5x4.jsonl`  
Tables: `artifacts/results/tables/support_suite_5x4/`

### Convert JSONL results to tables

Run:
`python scripts/80_make_tables.py --results_jsonl artifacts/results/jsonl/phase_a_bosch_e1_e4.jsonl --out_dir artifacts/results/tables/phase_a`

## Notes

- Bosch drilling experiment definitions are in `configs/experiments/bosch_drilling/`.
- The pipeline writes results to JSONL, and the table generator converts JSONL into summary tables.

## Troubleshooting

- **`FileNotFoundError` for datasets**: verify `mdpi-pi-fsl/configs/paths.yaml` points to the correct local dataset roots.
- **CUDA device errors**: install a PyTorch build compatible with the GPU driver/CUDA version, or run on CPU by setting the script/config device to `"cpu"` where applicable.
- **Permission errors writing outputs**: change `artifacts:` locations in `configs/paths.yaml` to a writable directory.
- **`learn2learn` install issues**: confirm Python/PyTorch versions are supported, then reinstall in a clean environment.
