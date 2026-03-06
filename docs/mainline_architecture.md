# Mainline Architecture

## Canonical paper model

The paper mainline is only:

1. Separator: `SepNet(periodicctx)`
2. Parameter reader: `GateFormer`
3. Ts constraint: `Ts = (NF + 1) * Tl`

No extra branches are part of the canonical architecture:
- no slot refiner
- no slot-set decoder
- no periodic anchor refiner
- no sigmoid-mask branch in mainline
- no mask-hint branch in mainline
- no edge auxiliary / sepgate auxiliary in mainline
- no iterative extractor in mainline

## Current canonical config

- Separator config: `configs/model_sep_sf28_periodicctx.yaml`
- PE config: `configs/model_pe.yaml`
- Joint train config: `configs/train_joint.yaml`
- Data config: `configs/data_composite.yaml`

## Architecture summary

### 1. Separator

`X (IQ, 2x4000)`
-> Conv encoder
-> TCN stack
-> PeriodicContextAggregator
-> mask decoder
-> 3 jammer slots + 1 background

Purpose:
- separate repeated forwarding structures
- keep the separator simple and physically aligned with ISRJ periodicity

### 2. Parameter reader

For each separated jammer slot:
- raw IQ branch
- TF branch
- mechanism-statistics branch
- GateFormer

Outputs:
- `NF`
- `Tl`
- `Ts`

Constraint:
- mainline uses structural `Ts = (NF + 1) * Tl`

## Experiment policy

All later ideas must satisfy one of these two rules:

1. Replace one core block.
   Example: replace `PeriodicContextAggregator` with a better separator block.

2. Add one isolated branch for a single hypothesis test.
   The branch must stay out of the canonical paper architecture until it beats the mainline clearly.

Not allowed:
- stacking multiple weak tricks into the paper model
- reporting a mixed architecture whose contribution cannot be explained clearly
- modifying the mainline definition after every experiment

## Reporting policy

Paper comparisons should always use the same mainline unless a new model clearly replaces it.
Current best accepted mainline result:
- all: `0.5860`
- dual: `0.7090`
- multi: `0.5040`

## Cleanup policy

- `runs/tmp_monitor/` and temporary PE/SEP configs are experiment-only.
- Experimental hooks in code are not paper claims.
- Before writing the paper method section, only the canonical mainline will be described.
