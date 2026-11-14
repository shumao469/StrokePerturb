"""Cross-fit fixed linear forecasts on published Norman condition means."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SOURCE_URL = "https://github.com/aarnavbhat23/perturbation-composition-lab/blob/main/data/pseudobulk.npz"
MANIFEST_URL = "https://github.com/aarnavbhat23/perturbation-composition-lab/blob/main/data/pseudobulk.manifest.json"
PRIMARY = "https://doi.org/10.1126/science.aax4438"
SHA256 = "b05525fb6190ab570d3fbbe9792149480df6ea1b82a47ecaf2a26f644f005afb"
BLOB_SHA = "736a93aac9327c07322f8f44b1beb757476ce640"
DATASET = "https://dataverse.harvard.edu/api/access/datafile/6154020"
METHODS = ("no_change", "additive", "ridge")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
SALT = "StrokePerturb_norman_public_pair_holdout_v1"


def assign(indices, labels, folds, salt):
    ordered = sorted(indices, key=lambda i: hashlib.sha256(f"{salt}|{labels[i]}".encode()).digest())
    return [np.asarray(ordered[k::folds], dtype=int) for k in range(folds)]


def ridge_basis(train, test, baseline, observed):
    x = baseline[train]
    xc = x - x.mean(axis=0)
    residual = observed[train] - x
    intercept = residual.mean(axis=0)
    gram = xc @ xc.T / xc.shape[1]
    gram_mean_diagonal = np.trace(gram) / len(train)
    transformed = (baseline[test] - x.mean(axis=0)) @ xc.T / xc.shape[1]
    eig, rotation = np.linalg.eigh(gram)
    return transformed @ rotation, rotation.T @ (residual - intercept), np.maximum(eig, 0), gram_mean_diagonal, intercept


def ridge_predict(test, baseline, components, alpha):
    design, coeff, eig, scale, intercept = components
    return baseline[test] + intercept + (design / (eig + alpha * scale)) @ coeff


def main(args):
    timer = time.perf_counter()
    raw = args.input.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    blob_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if file_hash != SHA256 or blob_hash != BLOB_SHA:
        raise ValueError("Compact published condition means failed pinned integrity checks")
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if source_manifest["source"] != DATASET:
        raise ValueError("Unrecognized processed expression provenance")
    with np.load(args.input, allow_pickle=False) as z:
        conditions = z["conditions"].astype(str)
        genes = z["genes"].astype(str)
        counts = z["counts"]
        data = z["means"].astype(np.float64)
    if len(set(conditions)) != len(conditions) or len(set(genes)) != len(genes) or not np.isfinite(data).all():
        raise ValueError("Expression or identity axes are ambiguous")
    index = {name: i for i, name in enumerate(conditions)}
    control = index["ctrl"] if "ctrl" in index else index["control"]
    singles = {name: i for name, i in index.items() if "+" not in name and i != control}
    groups = sorted((tuple(sorted(name.split("+"))), i) for name, i in index.items() if "+" in name)
    eligible = [(genes_ab, i) for genes_ab, i in groups if len(genes_ab) == 2 and genes_ab[0] != genes_ab[1] and set(genes_ab) <= set(singles)]
    if len(set(g for g, _ in eligible)) != len(eligible):
        raise ValueError("Duplicate canonical condition groups")
    pairs = ["+".join(g) for g, _ in eligible]
    paired_indices = np.asarray([i for _, i in eligible])
    control_vec = data[control]
    a = np.array([data[singles[g[0]]] - control_vec for g, _ in eligible])
    b = np.array([data[singles[g[1]]] - control_vec for g, _ in eligible])
    observed = data[paired_indices] - control_vec
    additive = a + b
    ridge = np.zeros_like(additive)
    folds = np.full(len(pairs), -1, dtype=int)
    audit = []
    for i, test in enumerate(assign(range(len(pairs)), pairs, 5, SALT)):
        train = np.setdiff1d(np.arange(len(pairs)), test)
        if np.intersect1d(train, test).size or (folds[test] >= 0).any():
            raise ValueError("Duplicate or overlapping outer groups")
        losses = np.zeros(len(ALPHAS))
        inner_validations = assign(train, pairs, 4, f"{SALT}|inner|{i}")
        for inner_test in inner_validations:
            inner_train = np.setdiff1d(train, inner_test)
            components = ridge_basis(inner_train, inner_test, additive, observed)
            for j, alpha in enumerate(ALPHAS):
                forecast = ridge_predict(inner_test, additive, components, alpha)
                losses[j] += np.square(observed[inner_test] - forecast).mean(axis=1).sum() / len(train)
        alpha = ALPHAS[int(np.argmin(losses))]
        ridge[test] = ridge_predict(test, additive, ridge_basis(train, test, additive, observed), alpha)
        folds[test] = i
        audit.append(dict(fold=i+1, training_pairs=len(train), test_pairs=len(test), alpha=alpha,
                          inner_mse=dict(zip(map(str, ALPHAS), losses.tolist()))))
    if (folds < 0).any() or not np.isfinite(ridge).all():
        raise ValueError("Incomplete OOF forecast")

    methods = (np.zeros_like(observed), additive, ridge)
    mse = np.asarray([np.square(observed - pred).mean(axis=1) for pred in methods])
    cosine = np.asarray([np.divide((observed*pred).sum(axis=1), np.linalg.norm(observed, axis=1)*np.linalg.norm(pred, axis=1),
                                  out=np.full(len(pairs), np.nan), where=np.linalg.norm(pred, axis=1)>0) for pred in methods])
    primary_gene_count = len(genes)
    summaries = {
        k: dict(mse=float(mse[i].mean()), median_pair_mse=float(np.median(mse[i])),
                median_cosine=float(np.nanmedian(cosine[i])) if i else None,
                wins_vs_additive=int((mse[i] < mse[1]).sum()),
                gain_vs_additive_pct=100*float(1-mse[i].mean()/mse[1].mean()))
        for i, k in enumerate(METHODS)
    }
    shown = float(np.ceil(20*np.quantile(np.abs(np.stack([observed, additive, ridge])), .995))/20)
    edges = np.linspace(-shown, shown, 43)
    bins = {
        k: np.histogram2d(observed.ravel(), pred.ravel(), bins=edges)[0].astype(int).tolist()
        for k, pred in zip(METHODS, methods)
    }
    example_indices = [min(pairs.index(p) for p in pairs), len(pairs)//2, len(pairs)-1]
    example = []
    for i in example_indices:
        # Select the display genes on the outer-training pair axis alone.
        train = np.flatnonzero(folds != folds[i])
        supported = np.argsort(np.sqrt(np.mean(np.square(observed[train]), axis=0)))[-8:][::-1]
        example.append(dict(pair=pairs[i], fold=int(folds[i])+1, cells=int(counts[paired_indices[i]]),
                            genes=genes[supported].tolist(), measured=observed[i,supported].tolist(),
                            additive=additive[i,supported].tolist(), ridge=ridge[i,supported].tolist(),
                            single_a=a[i,supported].tolist(), single_b=b[i,supported].tolist()))
    record = dict(
        schema="strokeperturb_public_norman_dual_gene_v1", passed=True,
        evidence_role="Separate retrospective public K562 CRISPRa dual-perturbation benchmark",
        source=dict(primary=PRIMARY, geo="GSE133344", processed_archive=DATASET, published_condition_means=SOURCE_URL,
                    condition_means_manifest=MANIFEST_URL, sha256=file_hash, git_blob_sha1=blob_hash, upstream=source_manifest),
        data=dict(cells=int(counts.sum()), control_cells=int(counts[control]), genes=primary_gene_count,
                  singles=len(singles), available_doubles=len(groups), eligible_doubles=len(eligible)),
        protocol=dict(outer_folds=5, inner_folds=4, grouping="Unordered target pair, whole measured condition",
                      group_assignment="Deterministic SHA-256 order, round-robin assignment",
                      target_seen="Both constituent single-target conditions are measured and used for every tested pair",
                      expression="Published means of processed GEARS expression, each minus the common control mean",
                      loss="Squared error averaged over the fixed published gene axis and then equally over tested pairs",
                      ridge="Multioutput linear ridge on train residual Y−(A+B), intercept and linear kernel fit inside each fold",
                      alpha_candidates=ALPHAS, alpha_scaling="Mean diagonal of each centered training Gram matrix",
                      uncertainty="Per-condition results are descriptive; conditions share constituents and the control"),
        methods=list(METHODS), summary=summaries, folds=audit,
        pairs=[dict(pair=name, fold=int(folds[i])+1, cells=int(counts[paired_indices[i]]),
                    mse={m: float(mse[j,i]) for j,m in enumerate(METHODS)},
                    cosine={m: float(cosine[j,i]) if np.isfinite(cosine[j,i]) else None for j,m in enumerate(METHODS)})
               for i,name in enumerate(pairs)],
        density=dict(edges=edges.tolist(), symmetric_limit=shown, count_total=int(observed.size),
                     plots={m: dict(counts=grid, count_visible=sum(map(sum, grid))) for m,grid in bins.items()}),
        examples=example, elapsed_seconds=time.perf_counter()-timer,
        established_external_test_access=False, stroke_model_fitted=False,
    )
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"data":record["data"],"summary":summaries,"folds":audit,"examples":[e["pair"] for e in example],
                      "elapsed":record["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
