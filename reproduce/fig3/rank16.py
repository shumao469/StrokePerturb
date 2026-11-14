"""Public combination extension using StrokePerturb's native response projection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import baselines as v6
from strokeperturb import response_subspace as subspace

PRIMITIVE = Path(subspace.__file__)
GAMMAS = (0.0, 0.5, 1.0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels_and_arrays(source):
    with np.load(source, allow_pickle=False) as z:
        conditions = z["conditions"].astype(str)
        genes = z["genes"].astype(str)
        counts = z["counts"]
        data = z["means"].astype(np.float64)
    names = {n: i for i, n in enumerate(conditions)}
    control = names["ctrl"] if "ctrl" in names else names["control"]
    singles = {n: i for n, i in names.items() if i != control and "+" not in n}
    pairs = sorted((tuple(sorted(n.split("+"))), i) for n, i in names.items() if "+" in n)
    eligible = [(g, i) for g, i in pairs if len(g) == 2 and g[0] != g[1] and set(g) <= set(singles)]
    base = data[control]
    target_idx = np.array([i for _, i in eligible])
    a = np.array([data[singles[g[0]]] - base for g, _ in eligible])
    b = np.array([data[singles[g[1]]] - base for g, _ in eligible])
    return genes, ["+".join(g) for g, _ in eligible], counts[target_idx], data[target_idx] - base, a, b


def projected_forecasts(train, test, x, y, names, alphas):
    ordered = np.asarray(names)[train]
    weights = np.full(len(train), 1. / len(train))
    residual = y[train] - x[train]
    context = dict(parent_labels=residual, pair_targets=ordered, names=ordered)
    proof = subspace.prepare_subspace(context, weights, gene_count=y.shape[1])
    if proof["rank"] != 16 or proof["query_inputs_read"]:
        raise ValueError("Full train-only response subspace must be available")
    basis = v6.ridge_basis(train, test, x, y)
    intercept = basis[-1]
    # Only the fitted residual slope is projected. Measured single effects
    # and the fold-local residual intercept retain their full native axes.
    for alpha in alphas:
        raw = v6.ridge_predict(test, x, basis, alpha)
        correction = raw - x[test]
        for gamma in GAMMAS:
            pred = raw if gamma == 0 else x[test] + subspace.apply_subspace(correction, intercept, proof, gamma)
            yield alpha, gamma, pred


def main(args):
    timer = time.perf_counter()
    source = args.input
    receipt_file = args.baselines
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if sha(source) != v6.SHA256:
        raise ValueError("Public processed input checksum mismatch")
    genes, names, counts, y, a, b = labels_and_arrays(source)
    if len(genes) != 5045 or names != [r["pair"] for r in receipt["pairs"]] or counts.tolist() != [r["cells"] for r in receipt["pairs"]]:
        raise ValueError("Public axis or original grouped design changed")
    original_gene_constant = subspace.GENES
    if original_gene_constant != 19070 or subspace.FIXED_RANK != 16:
        raise ValueError("Unexpected archived component dimensions")
    x = a + b
    no_change = np.square(y).mean(axis=1)
    additive = np.square(y-x).mean(axis=1)
    preserved = {}
    for method, scores in (("no_change", no_change), ("additive", additive)):
        prior = np.array([r["mse"][method] for r in receipt["pairs"]])
        np.testing.assert_array_equal(scores, prior)
        preserved[method] = float(np.max(np.abs(scores-prior)))
    established = np.zeros_like(y)
    candidate = np.zeros_like(y)
    folds = np.full(len(names), -1)
    ledger = []
    alpha_candidates = tuple(v6.ALPHAS)
    alpha_gamma = [(a, g) for a in alpha_candidates for g in GAMMAS]
    for k, test in enumerate(v6.assign(range(len(names)), names, 5, v6.SALT)):
        train = np.setdiff1d(np.arange(len(names)), test)
        if not np.array_equal(np.sort(test), np.flatnonzero(np.asarray([r["fold"] for r in receipt["pairs"]]) == k+1)):
            raise ValueError("Original v6 outer folds changed")
        losses = {p: 0. for p in alpha_gamma}
        for inner_test in v6.assign(train, names, 4, f"{v6.SALT}|inner|{k}"):
            inner_train = np.setdiff1d(train, inner_test)
            for alpha, gamma, pred in projected_forecasts(inner_train, inner_test, x, y, names, alpha_candidates):
                losses[alpha,gamma] += np.square(y[inner_test] - pred).mean(axis=1).sum() / len(train)
        alpha_old = min(alpha_candidates, key=lambda a: losses[a,0])
        if alpha_old != receipt["folds"][k]["alpha"]:
            raise ValueError("Pure ridge changed inner choice")
        alpha_new, gamma_new = min(alpha_gamma, key=lambda p: (losses[p], p[1], p[0]))
        forecasts = {(a,g): p for a,g,p in projected_forecasts(train, test, x, y, names, (alpha_old, alpha_new))}
        candidate[test] = forecasts[alpha_new,gamma_new]
        established[test] = forecasts[alpha_old,0]
        folds[test] = k+1
        ledger.append(dict(fold=k+1, train_pairs=len(train), test_pairs=len(test), ridge_alpha=alpha_old,
                           component_alpha=alpha_new, gamma=gamma_new, inner_mse={f"{a}:{g}":v for (a,g),v in losses.items()}))
    old = np.square(y-established).mean(axis=1)
    new = np.square(y-candidate).mean(axis=1)
    saved = np.array([r["mse"]["ridge"] for r in receipt["pairs"]])
    np.testing.assert_allclose(old, saved, atol=5e-14, rtol=0)
    preserved["ridge"] = float(np.max(np.abs(old-saved)))
    cos = np.divide((y*candidate).sum(axis=1), np.linalg.norm(y,axis=1)*np.linalg.norm(candidate,axis=1))
    extra = dict(mse=float(new.mean()), median_pair_mse=float(np.median(new)), median_cosine=float(np.median(cos)),
                 wins_vs_additive=int((new < np.asarray([r["mse"]["additive"] for r in receipt["pairs"]])).sum()),
                 wins_vs_ridge=int((new<old).sum()), gain_vs_additive_pct=100*float(1-new.mean()/receipt["summary"]["additive"]["mse"]),
                 gain_vs_ridge_pct=100*float(1-new.mean()/old.mean()))
    edges = receipt["density"]["edges"]
    hist = np.histogram2d(y.ravel(), candidate.ravel(), bins=edges)[0].astype(int)
    receipt["methods"].append("stroke_component")
    receipt["summary"]["stroke_component"] = extra
    receipt["density"]["plots"]["stroke_component"] = dict(counts=hist.tolist(),count_visible=int(hist.sum()))
    for i,row in enumerate(receipt["pairs"]):
        row["mse"]["stroke_component"] = float(new[i])
        row["cosine"]["stroke_component"] = float(cos[i])
    idx = {name:i for i,name in enumerate(genes)}
    for row in receipt["examples"]:
        i = names.index(row["pair"])
        row["stroke_component"] = candidate[i,[idx[g] for g in row["genes"]]].tolist()
    receipt["schema"] = "strokeperturb_public_norman_dual_component_v1"
    forecast_hashes = {name: hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
                       for name,p in (("ridge", established), ("stroke_component", candidate))}
    receipt["component"] = dict(
        name="StrokePerturb R398 rank-16 response-subspace adaptation",
        primary_meaning="Fold-refitted public baseline augmented with the exact archived response-subspace operator",
        primitive_module="strokeperturb.response_subspace", primitive_sha256=sha(PRIMITIVE),
        original_gene_axis=original_gene_constant, applied_gene_axis=len(genes), fixed_rank=16,
        original_frozen_model_run=False, new_public_component_fit=True, source_scope="Public Norman condition means only",
        residual_slope_projection="Existing R398 apply_subspace on calibrated residual slopes; additive singles and intercept untouched",
        calibration="V6 deterministic five outer and four inner grouping; alpha and gamma chosen exclusively from each outer TRAIN",
        gamma_candidates=list(GAMMAS), v6_replay_sha256=sha(receipt_file), zero_gamma_parity_max_abs=preserved["ridge"],
        baseline_parity_max_abs=preserved, forecast_sha256=forecast_hashes,
        fold_ledger=ledger, exact_forecast_match_ridge=forecast_hashes["ridge"] == forecast_hashes["stroke_component"],
        limitations=["No mouse NTC half controls", "No control-subclass and composition sufficient statistics",
                     "Different 5,045 human processed gene axis", "Pairs held out while constituent singles are available"],
        elapsed_seconds=time.perf_counter()-timer,
    )
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"summary":extra,"component":receipt["component"]},ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--baselines", type=Path, required=True, help="JSON output of fig3/baselines.py")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
