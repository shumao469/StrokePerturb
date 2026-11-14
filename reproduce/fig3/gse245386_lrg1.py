"""Replay the observed Lrg1 KO–WT contrast from five public GEO libraries."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread
from threadpoolctl import threadpool_limits

MANIFEST = Path(__file__).with_name("gse245386_inputs.json")
SELF = "ENSMUSG00000037095"
MARKERS = ("Lrg1", "Cldn11", "Itgb5", "Pcdh9", "Anxa2", "Pecam1", "Aif1", "Rbfox3", "S100a8", "Mbp", "Gfap", "Cx3cr1")
ORIGINAL_AXIS_SHA256 = "5e12e7d835bc8ddcfcf55fda049899df3bdf33be8a450cd8d51936f4432084d5"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_features(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        features = list(csv.reader(stream, delimiter="\t"))
    if not features or any(len(row) != 3 for row in features):
        raise ValueError(f"Expected a three-column GEO 10x feature file: {path.name}")
    ids = [row[0].split(".", 1)[0] for row in features]
    count = Counter(ids)
    mapping = {gene: i for i, gene in enumerate(ids) if count[gene] == 1 and features[i][2] == "Gene Expression"}
    return features, mapping


def get_axis(axis_path, feature_tables, mappings):
    common = set.intersection(*(set(mapping) for mapping in mappings))
    if axis_path is None:
        first = mappings[0]
        rows = [(gene, feature_tables[0][index][1]) for gene, index in first.items() if gene in common]
    else:
        if sha256(axis_path) != ORIGINAL_AXIS_SHA256:
            raise ValueError("The supplied ordered gene-axis file does not match the frozen axis SHA-256")
        with axis_path.open(encoding="utf-8", newline="") as stream:
            original = list(csv.DictReader(stream))
        if len(original) != 19070 or any(int(row["expression_column"]) != i for i, row in enumerate(original)):
            raise ValueError("Original ordered gene axis mismatch")
        ids = [row["gene_ids"].split(".", 1)[0] for row in original]
        count = Counter(ids)
        rows = [(gene, row["gene_name"]) for gene, row in zip(ids, original) if gene in common and count[gene] == 1]
        if len(rows) != 19060:
            raise ValueError("Expected exactly 19,060 matched genes in these five libraries")
    if SELF not in {gene for gene, _ in rows}:
        raise ValueError("Expected Lrg1 on the matched axis")
    return rows


def pseudobulk(paths, features, mapping, axis):
    eligible = np.array([row[2] == "Gene Expression" for row in features], dtype=bool)
    mito = np.array([row[1].casefold().startswith("mt-") and e for row, e in zip(features, eligible)], dtype=bool)
    if not mito.any():
        raise ValueError("Expected deposited mitochondrial feature annotations")
    with gzip.open(paths["barcodes.tsv"], "rt", encoding="utf-8") as stream:
        barcodes = [row.strip() for row in stream]
    if len(set(barcodes)) != len(barcodes) or "" in barcodes:
        raise ValueError("Expected unique deposited barcodes within a sample")
    with gzip.open(paths["matrix.mtx"], "rb") as stream, threadpool_limits(limits=1):
        counts = mmread(stream)
    if counts.shape != (len(features), len(barcodes)) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts.data < 0):
        raise ValueError("Matrix dimensions or values are incompatible with the accompanying GEO files")
    counts = counts.tocsc()
    counts.eliminate_zeros()
    gene_counts = counts[eligible].tocsc()
    totals = np.asarray(gene_counts.sum(axis=0)).ravel()
    detected = np.diff(gene_counts.indptr)
    mitochondrial = np.asarray(counts[mito].sum(axis=0)).ravel()
    fraction = np.divide(mitochondrial, totals, out=np.ones(totals.shape), where=totals > 0)
    keep = (detected > 500) & (totals > 1000) & (fraction < 0.25)
    if not keep.any():
        raise ValueError("No deposited cells pass the stated QC thresholds")
    summed = np.asarray(counts[:, keep].sum(axis=1)).ravel()
    denom = int(summed[eligible].sum())
    if denom != int(totals[keep].sum()):
        raise ArithmeticError("Cell and pseudobulk read-count totals disagree")
    expression = np.log1p(summed.astype(float) * 1e6 / denom)
    symbols = Counter(row[1] for row in features)
    unique_symbol = {row[1]: i for i, row in enumerate(features) if symbols[row[1]] == 1 and eligible[i]}
    markers = {name: float(expression[unique_symbol[name]]) if name in unique_symbol else None for name in MARKERS}
    return expression[[mapping[gene] for gene, _ in axis]], int(keep.sum()), markers


def run(input_dir, output_dir, axis_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    files = [{key: input_dir / pin["filename"] for key, pin in s["files"].items()} for s in samples]
    for sample, paths in zip(samples, files):
        for key, file in paths.items():
            if not file.is_file():
                raise FileNotFoundError(f"Download {sample['files'][key]['url']} to the input directory")
            meta = sample["files"][key]
            if file.stat().st_size != meta["bytes"] or sha256(file) != meta["sha256"]:
                raise ValueError(f"Downloaded file does not match the pinned public bytes: {file.name}")
    feat, maps = zip(*(read_features(paths["features.tsv"]) for paths in files))
    axis = get_axis(axis_path, feat, maps)
    values = [pseudobulk(paths, rows, index, axis) for paths, rows, index in zip(files, feat, maps)]
    profiles = np.stack([v[0] for v in values])
    ko = np.array([s["group"] == "KO" for s in samples])
    wt = ~ko
    baseline = profiles[wt].mean(axis=0)
    delta = profiles[ko].mean(axis=0) - baseline
    score = np.array([gene != SELF for gene, _ in axis])
    wt_profiles = profiles[wt][:, score]
    mse = float(np.mean(delta[score] ** 2))
    result = {
        "source": manifest["source"], "evidence_role": "Observed animal-level post-MCAO/R genotype contrast",
        "axis": "frozen_original_intersection" if axis_path is not None else "public_shared_feature_axis",
        "matched_genes": len(axis), "scored_genes_excluding_lrg1": int(score.sum()),
        "zero_reference_mse": mse, "zero_reference_rms": float(np.sqrt(mse)),
        "ko_vs_two_wt_mean_mse": np.mean((profiles[ko][:, score] - baseline[score]) ** 2, axis=1).tolist(),
        "wt_pair_mse": float(np.mean((wt_profiles[0] - wt_profiles[1]) ** 2)),
        "samples": [{"sample": s["sample"], "gsm": s["gsm"], "group": s["group"], "retained_cells": v[1],
                     "marker_expression_log1p_cpm": v[2]} for s, v in zip(samples, values)],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gse245386_lrg1.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "gse245386_lrg1_genes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ensembl", "gene", "mean_ko_minus_wt_log1p_cpm", "in_mse", *[s["sample"] for s in samples]])
        writer.writerows((gene, name, diff, int(eligible), *exp) for (gene, name), diff, eligible, exp in zip(axis, delta, score, profiles.T))
    print(json.dumps({key: result[key] for key in ("axis", "matched_genes", "scored_genes_excluding_lrg1", "zero_reference_mse")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with the 15 pinned GEO .gz files")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--axis", type=Path, help="Optional original ordered gene-axis CSV to reproduce the v8 MSE exactly")
    args = parser.parse_args()
    run(args.input_dir, args.out_dir, args.axis)
