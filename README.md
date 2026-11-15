# StrokePerturb

**Transcriptomic perturbation prediction across animals and targets**

StrokePerturb predicts RNA changes for animal–target populations and evaluates whether low prediction error coincides with recovery of the measured response. It separates a low-rank response from composition-associated and residual RNA changes, then calibrates their contributions inside the training folds. The mouse brain benchmark uses an in vivo Perturb-seq atlas. Separately measured stroke profiles identify tissue regions, sampling times and intervention contrasts for validation.

![StrokePerturb architecture. A matched non-targeting control subset and a target descriptor enter the response heads. Inner training folds select the component weights. Fixed predictions are scored against an RNA contrast defined with a disjoint control subset.](docs/strokeperturb-overview.svg)

[Model specification](docs/model.md) · [Benchmark results](reproduce/results/README.md) · [Reproduction guide](reproduce/README.md)

## Installation

Python 3.11 or later. The core and reproduction scripts run on CPU.

```bash
git clone https://github.com/shumao469/StrokePerturb.git
cd StrokePerturb
python -m pip install -e .
```

The joint-holdout diagnostics use NumPy. Install the public replay dependencies with `python -m pip install -e ".[reproduce]"`.

## Model input

The command-line interface accepts `.npz` files containing aligned response components. Each row represents one animal–target population.

| Key | Content |
| --- | --- |
| `source`, `target` | Unicode animal and target identifiers, one pair per row |
| `genes` | Ordered Unicode gene identifiers, shared across all component arrays |
| `background`, `U`, `C`, `W` | Population × gene matrices for the reference, low-rank response, composition-associated change and residual RNA change |
| `truth` | Observed RNA change on the same axes. Supply it for calibration and scoring. The prediction input omits this key. |

In the original assay, separate halves of matched non-targeting controls serve the query and evaluation stages. Full replay of the original mouse checkpoint requires its scope-specific upstream inputs and audit workflow. See the [model specification](docs/model.md) for data shapes, control construction, and endpoint definitions.

## Features

- Joint holdout by animal and perturbation target.
- C/W training-label decomposition, fixed linear residual kernels and rank-16 response projection.
- Inner-fold calibration of `B + (1 − λ)aU + λ(bC + cW)` and comparison with the strongest single response family selected on those folds.
- Target-macro MSE for downstream and within-animal-centred responses, with the perturbed gene masked.

## Calibrate, predict and score

Build the inner component bundle from out-of-fold predictions restricted to outer training data. Freeze the calibration before applying it to the outer query. The scorer rebuilds the fixed forecast from the same outer components and calibration, then evaluates it against the supplied truth.

```bash
strokeperturb calibrate inner_oof.npz calibration.json
strokeperturb predict outer_components.npz forecasts.npz --calibration calibration.json
strokeperturb score outer_with_truth.npz scores.json --calibration calibration.json
```

## Python example

This synthetic example holds out one animal and two targets. The arrays illustrate the API using precomputed components.

```python
import numpy as np
from strokeperturb import apply_calibration, calibrate_inner, score_outer

genes = ["A", "B", "C", "D", "marker"]
zero = np.zeros((2, len(genes)))
w = zero.copy()
w[:, -1] = [-1, 1]

inner = {"background": zero, "U": zero, "C": zero, "W": w}
frozen = calibrate_inner(0.5 * w, inner, ["train"] * 2, ["A", "B"], genes)

outer = {**inner, "W": 2 * w}
axes = (["test"] * 2, ["C", "D"], genes)
forecasts = apply_calibration(outer, *axes, frozen)
report, _ = score_outer(w, outer, *axes, frozen)

print(forecasts["candidate"].shape)
print(report["scores"]["candidate"])
```

## Benchmarks

Run `python -m strokeperturb demo` for a CPU example. The [aggregate benchmark tables](reproduce/results/README.md) give the reported scores and the [reproduction guide](reproduce/README.md) gives input requirements and commands.

| Benchmark | Main comparison |
| --- | --- |
| Mouse brain, out-of-fold development | All 12 comparisons improved on the matched no-change reference. Two narrowly improved on the strongest prespecified reference, and 0/12 reached the prespecified 1% gain. |
| Mouse brain, nested joint holdout | On 56 populations from two mice and 40 targets, the selected forecast matched the same-input full-gene residual. The median predicted-to-measured RMS amplitude ratio was 0.103% and the median gene-wise cosine was −0.0012. |
| Norman K562 gene pairs | A fitted linear correction lowered MSE by 29.8% versus summing the measured constituent responses. |
| ComboSciPlex A549 drug pairs | With measured constituent responses, the fitted linear reference gave the lowest mean MSE across seven held-out pairs. |

The nested block shares populations with the mouse development benchmark. The [joint-holdout diagnostics](reproduce/joint_holdout/README.md) use the frozen predictions and their paired measurements to check response magnitude, direction, ranking and cross-mouse agreement.

## Public data replays

| Dataset | Workflow |
| --- | --- |
| Norman K562 · GSE133344 | Whole-pair genetic perturbation prediction with measured singles and no-change, additive and linear baselines |
| ComboSciPlex A549 · GSE206741 | Held-out drug-pair prediction with same-input fitted linear and C/W comparisons |
| Lrg1 · GSE245386 | Measured animal-level KO–WT expression contrast after stroke |
| ISP · GSE168934 | Measured animal-level post-stroke ISP versus GEO-labelled control expression contrast |
| H9 · GSE75748 | Endpoint-only interpolation of group mean expression at intermediate time points |

The cell-line workflows fit within their own assays. The Lrg1 and ISP workflows summarize observed expression, and the H9 workflow evaluates a fixed interpolation rule.

## Citation and contact

When referring to the software, cite the [StrokePerturb repository](https://github.com/shumao469/StrokePerturb) and the commit used. Cite the original studies for any data or derived results using the references in the [reproduction guide](reproduce/README.md).

Copyright, Fudan University, Xulab. shumaoxu@fudan.edu.cn

Use [GitHub Issues](https://github.com/shumao469/StrokePerturb/issues) for questions and reproducibility reports.

[Project illustration](https://github.com/user-attachments/assets/e9ae24e1-55e5-4297-8762-18919caefb94)
