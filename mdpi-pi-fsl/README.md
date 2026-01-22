## Repository layout

- `src/pifsl/`  
  PI-FSL, baseline methods, dataset loaders, runners, and evaluation utilities.

- `configs/`  
  Dataset/output paths and experiment definitions.

- `scripts/`  
  Reproduction entrypoints for the manuscript experiments.

- `tools/`  
  Optional utility script: `bosch_shift_quantification.py`.

## Setup

### 1) Install dependencies

Python 3.10+ is recommended.

Run (from repository root):
`pip install -r requirements.txt`  
`pip install -e .`

### 2) Configure dataset and output paths

Edit: `configs/paths.yaml`  
Update dataset roots under `datasets:` and (optionally) change output locations under `artifacts:`.

## Reproducing manuscript experiments

All scripts print the executed command lines to the console.

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
