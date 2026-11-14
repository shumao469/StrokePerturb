"""Replay the observed ISP–vehicle bulk RNA response in public GSE168934."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE168nnn/GSE168934/suppl/GSE168934_read_counts_matrix.csv.gz"
SHA256 = "4c867128482f334503257bca4bde8dc006301429a75be7dec0e14779eb5c72b5"
GEO_IDS = {
    "Con19": "GSM5172591", "Con21": "GSM5172592", "Con36": "GSM5172593", "Con38": "GSM5172594",
    "ISP_L12": "GSM5172595", "ISP_L14": "GSM5172596", "ISP_L16": "GSM5172597", "ISP_L18": "GSM5172598",
    "VehL10": "GSM5172599", "VehL2": "GSM5172600", "VehL6": "GSM5172601", "VehL8": "GSM5172602",
}
MARKERS = ("Ptprs", "Lrg1", "Igfn1", "Penk", "Rasgef1c", "Dact2", "Grm2", "Cited4", "Sag", "Tpbg", "Nr4a2", "Bcl2", "Sema4d")


def run(counts, output_dir):
    if not counts.is_file():
        raise FileNotFoundError(f"Download {URL} before running the replay")
    source_hash = hashlib.sha256(counts.read_bytes()).hexdigest()
    if source_hash != SHA256:
        raise ValueError("The supplied archive differs from the pinned public GEO count archive")
    with gzip.open(counts, "rt", encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows)
        records = [(row[0], [int(cell) for cell in row[1:]]) for row in rows]
    if header[0] != "genes" or len(header) != len(GEO_IDS) + 1 or set(header[1:]) != set(GEO_IDS):
        raise ValueError("Unexpected GEO count header")
    sample_ids = header[1:]
    genes = np.array([g for g, _ in records])
    data = np.asarray([v for _, v in records], dtype=np.float64).T
    if len(set(genes)) != len(genes) or np.any(data < 0) or not np.isfinite(data).all():
        raise ValueError("Expected nonnegative counts on a unique gene symbol axis")
    vehicle = [i for i, name in enumerate(sample_ids) if name.startswith("Veh")]
    treatment = [i for i, name in enumerate(sample_ids) if name.startswith("ISP")]
    order = vehicle + treatment
    if len(vehicle) != 4 or len(treatment) != 4 or np.any(data[order].sum(axis=1) <= 0):
        raise ValueError("Expected four independent libraries per scored group")
    cpm = 1e6 * data[order] / data[order].sum(axis=1)[:, None]
    keep = np.sum(cpm >= 1, axis=0) >= 4
    names = genes[keep]
    x = np.log2(1.0 + cpm[:, keep])
    ref = x[:4].mean(axis=0)
    centred = x - ref
    contrast = centred[4:].mean(axis=0)
    obs_mse = float(np.mean(contrast ** 2))
    isp_mse = np.mean(centred[4:] ** 2, axis=1)
    vehicle_mse = np.array([np.mean((x[i] - np.delete(x[:4], i, axis=0).mean(axis=0)) ** 2) for i in range(4)])
    pca_x = x - x.mean(axis=0)
    _, sv, basis = np.linalg.svd(pca_x, full_matrices=False)
    pcs = pca_x @ basis[:2].T
    for k in range(2):
        if basis[k, np.argmax(np.abs(basis[k]))] < 0:
            pcs[:, k] *= -1
    label_sets = list(itertools.combinations(range(8), 4))
    permutations = [np.mean((x[list(ts)].mean(axis=0) - x[[j for j in range(8) if j not in ts]].mean(axis=0)) ** 2)
                    for ts in label_sets]
    marker_indices = {gene: i for i, gene in enumerate(names) if gene in MARKERS}
    hist, edges = np.histogram(contrast, bins=40)
    result = {
        "data_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168934",
        "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9677607/",
        "source_sha256": source_hash,
        "evidence_role": "Observed mouse peri-infarct cortex bulk RNA contrast at day 14 after stroke",
        "control_label": "Vehicle follows the GEO labels",
        "total_provided_genes": len(genes), "qualified_genes": int(keep.sum()),
        "reference_mse": {
            "isp_vs_all_vehicle_mean": isp_mse.tolist(), "mean_isp_vs_all_vehicle": float(np.mean(isp_mse)),
            "leave_one_vehicle_out": vehicle_mse.tolist(), "mean_leave_one_vehicle_out": float(np.mean(vehicle_mse)),
            "mean_isp_group_contrast": obs_mse,
        },
        "samples": [
            {"sample": sample_ids[old], "gsm": GEO_IDS[sample_ids[old]], "group": "Vehicle" if new < 4 else "ISP",
             "pc1": float(pcs[new, 0]), "pc2": float(pcs[new, 1]),
             "reference_mse": float(vehicle_mse[new] if new < 4 else isp_mse[new - 4])}
            for new, old in enumerate(order)
        ],
        "pca_variance_explained": (sv[:2] ** 2 / np.sum(sv ** 2)).tolist(),
        "display_genes": [{"gene": gene, "mouse_delta_from_vehicle": centred[:, marker_indices[gene]].tolist()} for gene in MARKERS if gene in marker_indices],
        "group_contrast_histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
        "exploratory_label_permutation": {"n_relabelings": len(label_sets),
                                          "tail_fraction": float(np.mean(np.array(permutations) >= obs_mse - 1e-15))},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gse168934_isp.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "gse168934_isp_genes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["gene", "mean_isp_minus_vehicle_log2cpm", *[sample_ids[i] for i in order]])
        writer.writerows((gene, value, *profile) for gene, value, profile in zip(names, contrast, x.T))
    print(json.dumps({key: result[key] for key in ("qualified_genes", "reference_mse")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True, help="Original GEO read-count matrix .csv.gz")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.counts, args.out_dir)
