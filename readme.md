# ISRJ Parameter Estimation (Reproducible Pipeline)

This repository implements a full pipeline:

1. Generate dataset (`single ISRJ + AWGN`)
2. Train ME-MVNet
3. Evaluate on split
4. Export figures and paper tables

## 1. Install

```bash
python -m pip install -r requirements.txt
```

## 2. Data

```bash
python scripts/00_make_folders.py
python scripts/01_generate_dataset.py --config configs/data.yaml
python scripts/05_sanity_check_dataset.py --config configs/data.yaml
```

## 3. Train

Smoke run:

```bash
python scripts/02_train.py --mode smoke --exp-name exp_001_smoke
```

Formal run:

```bash
python scripts/02_train.py --mode formal --exp-name exp_002_formal
```

## 4. Evaluate

```bash
python scripts/03_eval.py --ckpt runs/exp_002_formal/checkpoints/best.pt --split test
```

## 5. Export paper figures and tables

```bash
python scripts/04_export_plots.py --run-dir runs/exp_002_formal --split test
```

Outputs:

- Run artifacts: `runs/exp_xxx/`
- Paper assets: `paper/figures/`, `paper/tables/`
