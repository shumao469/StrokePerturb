# Benchmark results

CSV tables contain aggregate manuscript results. Positive relative gain denotes lower mean squared error (MSE). The unchanged forecast is zero on the matched control-referenced expression-change axis.

| Table | Evaluation |
| --- | --- |
| [Mouse development MSE](mouse_development_mse.csv) | Extended Data Fig. 2. Animal, target and joint out-of-fold development comparisons across two endpoints and two selectors. |
| [Mouse nested joint-holdout MSE](mouse_joint_holdout_mse.csv) | Extended Data Fig. 4. Pooled target-macro results for both endpoints on one outer block. |
| [Mouse shared-target summary](mouse_shared_target_summary.csv) | Extended Data Fig. 4. Agreement, response scale and permutation summaries on 16 targets common to both outer mice. |
| [Norman K562 pair MSE](norman_k562_pair_mse.csv) | Fig. 4. Equal-pair scores across 131 held-out gene pairs with measured constituent singles. |
| [ComboSciPlex A549 pair MSE](combosciplex_a549_pair_mse.csv) | Fig. 5. Equal-pair scores across seven held-out drug pairs with measured constituent singles. |

## Mouse brain evaluation

In the canonical comparisons, the candidate reduced error against no change by 0.00065–0.149%. Against the strongest prespecified references, relative gains ranged from −0.288% to +0.00022%, and **0 of 12** crossed the prespecified 1% threshold. Two positive gains came from the same within-animal-centred target-holdout endpoint under the two selectors.

The nested block spans 56 populations, two mice and 40 targets. Its 56 populations also appear in the canonical joint development set. Calibration selected 0.5 × W, which matched the same-input full-gene residual. Relative to no change, within-animal-centred MSE fell by 0.0000134% and uncentred MSE rose by 0.000332%. The median predicted-to-measured RMS amplitude ratio was 0.103% and the median gene-wise cosine was −0.0012. On the 16 common targets, the median measured cross-mouse cosine was 0.0067 after re-centring both mice on the shared target set.

## Combination assays

In K562, fitted ridge reduced MSE by 29.8% versus adding the two measured single-gene effects. The rank-16 adaptation selected the ridge prediction in all five outer folds. In A549, fitted linear regression gave the lowest mean error of the evaluated predictors. Both assays fit and score within their own data.

See the [reproduction guide](../README.md) for commands and [mouse diagnostic input schemas](../joint_holdout/README.md) for the scope-qualified replay.
