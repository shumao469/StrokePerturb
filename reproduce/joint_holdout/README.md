# Mouse brain joint-holdout diagnostics

These CPU scripts reproduce Figure 3f, Extended Data Figures 3e and 4, and their companion diagnostic tables from the frozen joint evaluation. The block has 56 animal–target populations, two mice, 40 targets, and 19,070 ordered genes. The arrays must match the original score masks and row order. Use scope-qualified, already-scored arrays for numerical replay.

Run from the repository root.

```bash
python reproduce/joint_holdout/rank_cosine_diagnostics.py centred_scored_arrays.npz res/joint/ranks
python reproduce/joint_holdout/replication_diagnostics.py raw_scored_arrays.npz res/joint/replication
```

| NPZ key | Shape and use |
| --- | --- |
| `mice`, `targets` | Unicode arrays of length 56, in identical row order for all matrices. The two mouse blocks include 23 and 33 populations and share 16 targets. |
| `genes` | Unicode array of 19,070 genes in score order. |
| `mask` | Boolean 56 × 19,070 matrix. True marks a retained entry. |
| `observed_centered`, `predicted_centered` | Observed and frozen forecast matrices for the rank/cosine script. Centre each separately per mouse and gene over the retained entries. |
| `observed_raw`, `predicted_raw` | Uncentred KO-minus-score-control measurement and frozen uncentred 0.5 × W prediction for the repeatability script. |

The rank/cosine script evaluates top and bottom gene overlap under target-label permutations that preserve the animal structure and shared identities. For maximum cosine retrieval, it compares equally sized same-target and different-target candidate sets using pairwise intersections of the masks.

The repeatability script computes exact-mask no-change, calibrated low-rank, same-input full-gene and unit-residual comparisons for both MSE endpoints. It also re-centres each mouse on its 16 shared targets and quantifies agreement between measured responses. The generated CSV files include paired losses, summary statistics and the permutation-null histogram.

See the [aggregate CSV tables](../results/README.md) for results that can be inspected directly from the repository.
