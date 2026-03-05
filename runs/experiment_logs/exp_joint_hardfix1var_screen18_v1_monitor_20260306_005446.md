# Joint Monitor Run (20260306_005446)

- mode: `formal`
- attempts: `3`
- sep_ckpt: `runs/exp_sep_formal_datafix_v1/checkpoints/best.pt`
- bad_check_epoch: `18`
- min_best_a_total: `0.2`
- plateau_patience: `50`
- accept_a_total: `0.2`

## Live
- [attempt 01] epoch=1 val_A_total=0.0148 best=0.0148@1
- [attempt 01] epoch=2 val_A_total=0.0703 best=0.0703@2
- [attempt 01] epoch=3 val_A_total=0.0571 best=0.0703@2
- [attempt 01] epoch=4 val_A_total=0.0928 best=0.0928@4
- [attempt 01] epoch=5 val_A_total=0.1840 best=0.1840@5
- [attempt 01] epoch=6 val_A_total=0.1374 best=0.1840@5
- [attempt 01] epoch=7 val_A_total=0.1692 best=0.1840@5
- [attempt 01] epoch=8 val_A_total=0.1913 best=0.1913@8
- [attempt 01] epoch=9 val_A_total=0.1736 best=0.1913@8
- [attempt 01] epoch=10 val_A_total=0.2139 best=0.2139@10
- [attempt 01] epoch=11 val_A_total=0.3279 best=0.3279@11
- [attempt 01] epoch=12 val_A_total=0.3139 best=0.3279@11
- [attempt 01] epoch=13 val_A_total=0.2473 best=0.3279@11
- [attempt 01] epoch=14 val_A_total=0.2529 best=0.3279@11
- [attempt 01] epoch=15 val_A_total=0.2868 best=0.3279@11
- [attempt 01] epoch=16 val_A_total=0.2852 best=0.3279@11
- [attempt 01] epoch=17 val_A_total=0.3385 best=0.3385@17

- attempt=01 seed=20260304 status=completed best_A_total=0.3385@17 last_A_total=0.3354@18 reason=completed log=runs\experiment_logs\exp_joint_hardfix1var_screen18_v1_a01.log

## Final Summary
| attempt | seed | status | best_A_total | best_epoch | last_A_total | last_epoch | reason |
|---|---:|---|---:|---:|---:|---:|---|
| 01 | 20260304 | completed | 0.3385 | 17 | 0.3354 | 18 | completed |

- success: `True`
