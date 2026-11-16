# Cross-instrument runs

Everything the cross-instrument section of the manuscript reports is recomputed
from this directory. Checkpoints are not included; nothing in the manuscript
needs them.

## Layout

```
runs_p3/<arm>/<backbone>/seed<k>/
    fold<j>.npz           per-slice probabilities, labels and recording ids for fold j
    fold<j>_log.csv       per-epoch training log (loss, accuracy, learning rate)
    fold<j>_result.json   fold-level pooled metrics
    fold<j>_probe.json    instrument-identity probe output (arms A0 and A1 only)
    pooled.json           metrics pooled over the three folds
```

* arms: `A0` (random slice-level folds), `A1` (instrument-withheld folds),
  `A1r` (recording-grouped folds with instrument overlap), `A2`
  (instrument-withheld folds plus the adversarial instrument head);
* backbones: `alexnet`, `swin_t`;
* seeds: `seed0`, `seed1` (classifier-training seeds);
* folds: `fold1`, `fold2`, `fold3`.

178 files, 8.8 MB: 48 prediction archives, 48 training logs, 48 fold results,
18 probe files and 16 pooled summaries.

## Recomputing the reported numbers

```bash
python code/analyze_p3.py       # pooled tables from the fold results
python code/p3_probe.py         # the probe table from fold<j>_probe.json
```

`data/results/p3/p3_probe.csv`, `data/results/p3/p3_metrics.csv`,
`data/results/p3b_leak/p3_delta.csv` and `data/results/p3b_instr/p3_delta.csv`
are the tables the manuscript prints; they are the output of these two scripts
over this directory.

The 7.8 GB of `.pt` checkpoints stay out of the repository. They are needed only
to refit the probe on new features, not to reproduce a printed number.
