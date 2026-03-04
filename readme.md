# ME-MVSepPE Pipeline (Composite ISRJ Separation + Per-Source Estimation)

Main workflow in this repo now follows `need.md` (route B):

1. Generate composite dataset (`2~3 ISRJ + echo + AWGN`)
2. Stage-1 train SepNet (separation only)
3. Stage-2 joint train (SepNet + PE-Net)
4. Evaluate with Wang-style metrics (`A_Tl/A_Ts/A_NF/A_total`)
5. Export figures/tables to `paper/`

## 1. Install

```bash
python -m pip install -r requirements.txt
```

## 2. Generate Composite Dataset

```bash
python scripts/10_generate_dataset_composite.py --config configs/data_composite.yaml
python scripts/11_sanity_check_composite.py --config configs/data_composite.yaml
```

## 3. Stage-1 Train SepNet

Smoke:

```bash
python scripts/12_train_sepnet.py --mode smoke --exp-name exp_sep_smoke
```

Formal:

```bash
python scripts/12_train_sepnet.py --mode formal --exp-name exp_sep_formal
```

## 4. Stage-2 Joint Train

Smoke:

```bash
python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_smoke/checkpoints/best.pt --mode smoke --exp-name exp_joint_smoke
```

Formal:

```bash
python scripts/13_train_seppe_joint.py --sep-ckpt runs/exp_sep_formal/checkpoints/best.pt --mode formal --exp-name exp_joint_formal
```

## 5. Evaluate

```bash
python scripts/14_eval_seppe.py --ckpt runs/exp_joint_formal/checkpoints/best.pt --split test --run-dir runs/exp_joint_formal
```

Outputs:

- `runs/exp_joint_formal/tables/test_metrics.json`
- `runs/exp_joint_formal/tables/test_metrics_by_jnr.csv`
- `runs/exp_joint_formal/tables/test_metrics_by_kactive.csv`

## 6. Export Plots/Tables

```bash
python scripts/15_export_plots_seppe.py --run-dir runs/exp_joint_formal --split test
```

Exports to:

- `runs/exp_joint_formal/figures/`
- `paper/figures/`
- `paper/tables/`

## Legacy Single-Source Pipeline

Older single-source scripts (`scripts/01~05`) remain in repo for reference, but are no longer the default route.
