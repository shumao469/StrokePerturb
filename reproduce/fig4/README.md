# Figure 4 · A549 ComboSciPlex

The replay uses the public A549 cell-line experiment from [GEO GSE206741](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741) and the matching [processed archive](https://zenodo.org/records/14638780). It refits a composition and within-state predictor on the same leave-one-pair-out folds as four comparators. The archived mouse model is outside this analysis.

## Inputs

| File | Source | Expected SHA-256 |
| --- | --- | --- |
| `sciplex3_comb.h5ad.gz` | [Zenodo 14638780](https://zenodo.org/records/14638780/files/sciplex3_comb.h5ad.gz?download=1) | `c022b1f4a708ba293fe372b782690e04e9ee370b5dbda1c99c3ee1b27d271a1e` |
| `sciplex3_comb.h5ad` | Decompressed Zenodo file | `43d4197bf50340b3a14c602f33ddf0274f82877c3af396370f2f8c62a8db5640` |
| `GSE206741_cell_metadata.tsv.gz` | [NCBI GEO supplementary file](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206741/suppl/GSE206741_cell_metadata.tsv.gz) | `67593f3785759bf9f2089cd369ab6f3d5aca76555fa1a31fd7ba781a79b20765` |

For example, from the repository root on a Unix-like shell:

```bash
mkdir -p data/combosciplex
mkdir -p res
curl -fL 'https://zenodo.org/records/14638780/files/sciplex3_comb.h5ad.gz?download=1' -o data/combosciplex/sciplex3_comb.h5ad.gz
curl -fL https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206741/suppl/GSE206741_cell_metadata.tsv.gz -o data/combosciplex/GSE206741_cell_metadata.tsv.gz
gzip -dc data/combosciplex/sciplex3_comb.h5ad.gz > data/combosciplex/sciplex3_comb.h5ad
python reproduce/fig4/combosciplex.py --h5ad data/combosciplex/sciplex3_comb.h5ad --archive data/combosciplex/sciplex3_comb.h5ad.gz --metadata data/combosciplex/GSE206741_cell_metadata.tsv.gz --output res/fig4_combo.json
```

The script verifies all three checksums before analysis. It writes fold weights, well-level scores, and display data for the measured and predicted responses. Add `--forecasts res/fig4_forecasts.npz` to save the full 5,000-gene predictions and measured pair means locally.

The replay was checked with NumPy 1.26.4, SciPy 1.16.3, h5py 3.15.1 and scikit-learn 1.7.2. Prediction hashes record each run's arrays. Use numerical tolerances when comparing arrays across BLAS implementations.

## Reference results

The unit is a well pseudobulk response, computed as `log1p(CP10K)` relative to the mean of three vehicle wells. Genes and wells are averaged within each pair. The final score weights each of the seven pairs equally.

| Method | Pair-macro MSE |
| --- | ---: |
| No change | 0.21932975 |
| Additive | 0.04737949 |
| Shrunk additive | 0.04359701 |
| Fitted linear | 0.02560762 |
| Composition and within-state adaptation | 0.04071376 |

Four transcriptional states are fitted using the shared vehicle cells, with 128 variable genes for that fit. Each held-out pair has measured constituent single-agent responses and three scoring wells. The fitted weights use the other six pair means. The [analysis specification](design.json) describes the vehicle reference and comparison. The processed 5,000-gene axis comes from the public archive. [Primary study](https://doi.org/10.15252/msb.202211517).
