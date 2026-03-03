```
isrj_param_estimation/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── eval.yaml
├── scripts/
│   ├── 00_make_folders.py
│   ├── 01_generate_dataset.py
│   ├── 02_train.py
│   ├── 03_eval.py
│   ├── 04_export_plots.py
│   └── 05_sanity_check_dataset.py
├── data/
│   ├── raw/
│   │   └── isrj_single_dataset_v1/
│   │       ├── train.npz
│   │       ├── val.npz
│   │       ├── test.npz
│   │       └── meta.json
│   ├── processed/
│   │   └── isrj_single_dataset_v1/
│   │       ├── train_cache.npz
│   │       ├── val_cache.npz
│   │       └── test_cache.npz
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── seed.py
│   │   ├── io.py
│   │   ├── meters.py
│   │   ├── logging.py
│   │   └── plot.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── isrj_generator.py
│   │   ├── dataset_npz.py
│   │   ├── transforms.py
│   │   └── stft.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── memvnet.py
│   │   ├── blocks_1d.py
│   │   ├── blocks_2d.py
│   │   ├── fusion.py
│   │   └── losses.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optim.py
│   │   └── scheduler.py
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py
│       └── calibration.py
├── runs/
│   ├── exp_001/
│   │   ├── config_dump.yaml
│   │   ├── checkpoints/
│   │   │   ├── last.pt
│   │   │   └── best.pt
│   │   ├── logs/
│   │   │   ├── train.log
│   │   │   └── metrics.jsonl
│   │   ├── predictions/
│   │   │   ├── val_pred.npz
│   │   │   └── test_pred.npz
│   │   └── figures/
│   │       ├── scatter_Tl.png
│   │       ├── scatter_Tf.png
│   │       ├── cm_NF.png
│   │       └── jnr_sweep.png
│   └── README.md
├── paper/
│   ├── figures/            # 最终论文用图（从 runs 导出）
│   ├── tables/             # 最终论文用表
│   └── notes.md            # 方法/实验记录（可选）
└── tests/
    ├── test_generator.py
    ├── test_dataset_loading.py
    └── test_shapes.py
```

