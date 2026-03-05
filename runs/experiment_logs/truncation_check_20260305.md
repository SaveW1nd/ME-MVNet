# Truncation Check (2026-03-05)

## Scope
- Dataset: `data/raw/cisrj_seppe_v1/{train,val,test}.npz`
- Goal: check whether active jammer forwarding is clipped by range limit.

## Split composition (K_active)
- train: `{2: 2500, 3: 2500}`
- val: `{2: 500, 3: 500}`
- test: `{2: 500, 3: 500}`

## Gate-run statistics
- train:
  - active sources: `12500`
  - runs: `19457`
  - short runs (len < round(Tl*fs)): `0`
  - edge runs (touch n-1): `6`
  - edge short runs: `0`
  - zero-energy active sources: `0`
- val:
  - active sources: `2500`
  - runs: `3908`
  - short runs: `0`
  - edge runs: `2`
  - edge short runs: `0`
  - zero-energy active sources: `0`
- test:
  - active sources: `2500`
  - runs: `3964`
  - short runs: `0`
  - edge runs: `0`
  - edge short runs: `0`
  - zero-energy active sources: `0`

## Conclusion
- No evidence of out-of-range clipping that shortens active forwarding segments.
- No active source with zero gate energy.
- Current generator's safe-delay strategy is working for this issue.
