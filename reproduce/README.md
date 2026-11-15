# Benchmark reproduction

Run the commands from the repository root after `python -m pip install -r requirements.txt`. Create `data/` for inputs and `res/` for outputs. Leave downloaded archives in their original format for checksum verification. Output schemas and scientific roles are specific to each replay.

| Figure | Public source | Replay |
| --- | --- | --- |
| 4a–e | [Norman et al.](https://doi.org/10.1126/science.aax4438), [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) | Grouped whole-pair prediction in K562 with measured constituent single perturbations |
| 4f | [GSE245386](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245386) | Observed Lrg1 genotype contrast across three KO and two WT post-MCAO/R mice |
| 5a–f | [GSE206741](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741), [Zenodo 14638780](https://zenodo.org/records/14638780) | Whole-pair drug-response prediction in A549 |
| 5h | [GSE168934](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168934) | Observed day-14 ISP versus GEO-labelled control contrast in mouse peri-infarct cortex bulk RNA |
| 6 | [GSE75748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748), [Chu et al.](https://doi.org/10.1186/s13059-016-1033-x) | Endpoint-only aggregate interpolation in an in vitro H9 differentiation time course |

## Mouse brain perturbation benchmarks

The [result tables](results/README.md) include the 12 canonical out-of-fold development comparisons and the pooled nested joint-holdout baselines. The [diagnostic scripts](joint_holdout/README.md) reproduce the rank permutation, matched-count maximum cosine, same-input baselines and shared-target analyses from the scored arrays for the nested block. The scripts require the original scoped arrays for manuscript-level numerical replay. They run on CPU with NumPy.

The remaining sections give the public assay commands with current manuscript figure numbers. Script directory names and `plot_public.py` selectors retain their original identifiers.

## Figure 4a–e · Norman whole-pair holdout

Use the version-pinned [published processed condition means](https://github.com/aarnavbhat23/perturbation-composition-lab/tree/main/data). Save [pseudobulk.npz](https://raw.githubusercontent.com/aarnavbhat23/perturbation-composition-lab/main/data/pseudobulk.npz) and [pseudobulk.manifest.json](https://raw.githubusercontent.com/aarnavbhat23/perturbation-composition-lab/main/data/pseudobulk.manifest.json) in `data/`. The loader verifies the binary SHA-256 and Git blob identity. The original assay identifies K562. The processed archive contains an inconsistent upstream cell-type field. Rows are grouped using the original perturbation condition labels.

```bash
python reproduce/fig3/baselines.py --input data/pseudobulk.npz --manifest data/pseudobulk.manifest.json --output res/norman_baselines.json
python reproduce/fig3/rank16.py --input data/pseudobulk.npz --baselines res/norman_baselines.json --output res/norman_rank16.json
python reproduce/plot_public.py fig3 --input res/norman_rank16.json --output res/Fig04_K562.pdf
```

Equal-pair MSE over 131 held-out combinations and 5,045 genes is 0.00759661 for no change, 0.00178510 for the sum of singles, and 0.00125345 for fitted linear ridge. All five outer folds selected zero projection strength from training data, so the rank-16 adaptation reproduced the ridge predictions exactly in this analysis.

## Figure 4f · Lrg1

Download the 15 `features.tsv.gz`, `barcodes.tsv.gz`, and `matrix.mtx.gz` files listed in [the input manifest](fig3/gse245386_inputs.json) into a single directory, preserving their names (about 792 MB total). The script applies the fixed cell thresholds, sums counts per animal and calculates the KO–WT contrast on the uniquely matched gene axis.

```sh
python reproduce/fig3/gse245386_lrg1.py --input-dir data/GSE245386 --out-dir res/GSE245386
```

The default uses the 32,284 uniquely identified expression genes common to the five deposited feature tables after excluding Lrg1. Its zero-reference MSE is 0.132356. To reproduce the Figure 4f MSE of 0.186594, add `--axis path/to/ordered_gene_axis.csv` using the original 19,070-row project axis. The script pins its SHA-256 and scores the 19,059 matched genes after removing Lrg1. The original ordered axis is a separately supplied project input. The five public matrices alone suffice for the default descriptive analysis and the fixed-marker values.

## Figure 5a–f · ComboSciPlex

Follow the [input and analysis instructions](fig4/README.md). The replay re-fits each comparison using the six training pairs, then scores all three wells of the remaining pair. Generate a summary with `python reproduce/plot_public.py fig4 --input res/fig4_combo.json --output res/Fig05_A549.pdf`.

## Figure 5h · ISP

Download the [GEO count table](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE168nnn/GSE168934/suppl/GSE168934_read_counts_matrix.csv.gz). The script uses four GEO-labelled control and four ISP libraries and retains genes with CPM ≥ 1 in at least four libraries. The output contains 14,305 qualified genes, mean ISP-to-control squared distance of 0.131177 and mean control leave-one-out distance of 0.082129. The released code uses the deposited vehicle label as the control-group key. The source study describes the comparator as scrambled peptide in Results and vehicle in RNA-seq Methods.

```sh
python reproduce/fig4/gse168934_isp.py --counts data/GSE168934_read_counts_matrix.csv.gz --out-dir res/GSE168934
```

The Lrg1 and ISP replays quantify measured expression differences with animals as the biological unit. E-TABM-612 was assessed for input qualification. A scored comparison requires verified contemporaneous controls, biological pools, and a common numeric probe scale.

## Figure 6 · H9 time course

Download [GSE75748_sc_time_course_ec.csv.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75748/suppl/GSE75748_sc_time_course_ec.csv.gz) to `data/`. Selection of 256 genes and the coordinate basis uses the 0 h and 96 h samples. Observations at 12, 24, 36 and 72 h score the group mean forecasts. Time points receive equal weight.

```bash
python reproduce/fig5/h9_interpolation.py --input data/GSE75748_sc_time_course_ec.csv.gz --endpoint-freeze res/h9_freeze.json --output res/h9.json
python reproduce/plot_public.py fig5 --input res/h9.json --output res/Fig06_H9.pdf
```

The intermediate-time MAE is 1.45539 for linear interpolation, 2.32368 for carrying forward the 0 h mean, and 2.14849 for using the 96 h mean. This is a fixed aggregate interpolation comparison on a separate public assay.
