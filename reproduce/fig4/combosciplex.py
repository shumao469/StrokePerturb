"""Replay the A549 ComboSciPlex leave-one-pair-out experiment for Figure 4."""
import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

VEHICLE = ("dmso", "dmso")
METHODS = ("no_change", "additive", "shrinkage", "fitted_linear", "cw_adaptation")
DESIGN = Path(__file__).with_name("design.json")
INPUT_CHECKSUMS = {
    "archive": {
        "md5": "e0414e642556288fc9e9a296cfab02bf",
        "sha256": "c022b1f4a708ba293fe372b782690e04e9ee370b5dbda1c99c3ee1b27d271a1e",
    },
    "h5ad": {"sha256": "43d4197bf50340b3a14c602f33ddf0274f82877c3af396370f2f8c62a8db5640"},
    "metadata": {"sha256": "67593f3785759bf9f2089cd369ab6f3d5aca76555fa1a31fd7ba781a79b20765"},
}


def hashes(file, *types):
    objects = [hashlib.new(t) for t in types]
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for obj in objects:
                obj.update(chunk)
    return {t: obj.hexdigest() for t, obj in zip(types, objects)}


def obs_column(node):
    if isinstance(node, h5py.Group):
        codes = node["codes"][:]
        if np.any(codes < 0):
            raise ValueError(f"Missing categorical values: {node.name}")
        return node["categories"][:].astype(str)[codes]
    return node[:].astype(str)


def clean(value):
    return str(value).strip().casefold()


def combination(first, second):
    return tuple(sorted((clean(first), clean(second))))


def read_public_data(h5ad_path, archive_path, metadata_path):
    actual = {
        "archive": hashes(archive_path, "md5", "sha256"),
        "h5ad": hashes(h5ad_path, "sha256"),
        "metadata": hashes(metadata_path, "sha256"),
    }
    for name, expected in INPUT_CHECKSUMS.items():
        if actual[name] != expected:
            raise ValueError(f"Checksum mismatch for {name}: {actual[name]}")
    with gzip.open(metadata_path, "rt", encoding="utf-8-sig") as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    registry = {row["Cell"]: row for row in metadata}
    if len(registry) != len(metadata):
        raise ValueError("Original public barcode axis contains duplicates")
    wells = defaultdict(list)
    checked_aliases = Counter()
    with h5py.File(h5ad_path, "r") as archive:
        obs, var = archive["obs"], archive["var"]
        obs_ids = obs_column(obs[obs.attrs["_index"]])
        primary_ids = [re.sub(r"-1$", "", value) for value in obs_ids]
        if len(set(primary_ids)) != len(primary_ids) or not set(primary_ids) <= set(registry):
            raise ValueError("Processed observations do not map one-to-one to the public metadata")
        if set(obs_column(obs["cell_line"])) != {"A549"}:
            raise ValueError("Unexpected context outside A549")
        observed_first = obs_column(obs["Drug1"])
        observed_second = obs_column(obs["Drug2"])
        for i, barcode in enumerate(primary_ids):
            row = registry[barcode]
            original = combination(row["Drug1"], row["Drug2"])
            processed = combination(observed_first[i], observed_second[i])
            # Processed metadata uses harmonized aliases. Grouping follows original GEO.
            if processed != original:
                checked_aliases[(original, processed)] += 1
            wells[(original, row["Well"])].append(i)
        if len(wells) != 96 or len({c for c, _ in wells}) != 32:
            raise ValueError("Public processed wells or treatment combinations changed")
        matrix = archive["layers"]["counts"]
        shape = tuple(map(int, matrix.attrs["shape"]))
        if shape != (len(obs_ids), 5000):
            raise ValueError("Unexpected processed gene dimension")
        payload = matrix["data"][:]
        if np.any(payload < 0) or not np.all(payload == np.floor(payload)):
            raise ValueError("Selected layer is not a nonnegative count matrix")
        counts = csr_matrix((payload, matrix["indices"][:], matrix["indptr"][:]), shape=shape)
        stable_ids = obs_column(var[var.attrs["_index"]])
        genes = obs_column(var["symbol-1"])
        if len(set(stable_ids)) != len(stable_ids):
            raise ValueError("Gene stable-ID axis contains duplicates")
    return wells, counts, stable_ids, genes, {
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206741",
        "paper_url": "https://doi.org/10.15252/msb.202211517",
        "processed_archive_url": "https://zenodo.org/records/14638780",
        "processed_h5ad_gzip": actual["archive"],
        "processed_h5ad_sha256": actual["h5ad"]["sha256"],
        "original_metadata_sha256": actual["metadata"]["sha256"],
        "metadata_cell_count": len(registry),
        "matched_processed_cell_count": len(obs_ids),
        "label_aliases_retained_as_original": [
            {"geo": pair[0], "processed": pair[1], "n_cells": n} for pair, n in sorted(checked_aliases.items())
        ],
    }


def bulk_response(wells, counts):
    groups = defaultdict(list)
    for (condition, well), cell_indices in sorted(wells.items()):
        pooled = np.asarray(counts[cell_indices].sum(axis=0), dtype=np.float64).ravel()
        if pooled.sum() <= 0:
            raise ValueError("An assayed well has no expression counts")
        value = np.log1p(1e4 * pooled / pooled.sum())
        groups[condition].append((well, len(cell_indices), value))
    if any(len(g) != 3 for g in groups.values()):
        raise ValueError("The documented triplicate-well design has changed")
    reference = np.mean([v[2] for v in groups[VEHICLE]], axis=0)
    response = {condition: np.stack([v[2]-reference for v in group])
                for condition, group in groups.items()}
    return groups, response


def mse(target, predicted):
    return np.mean(np.square(target - predicted), axis=1)


def vector_cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a,b) / den) if den > 0 else None


def score_baselines(groups, responses, ids, genes):
    singles = {next(d for d in key if d != "dmso"): val.mean(axis=0)
               for key, val in responses.items() if key != VEHICLE and "dmso" in key}
    total_pairs = sorted(key for key in responses if "dmso" not in key)
    pair_keys = [key for key in total_pairs if all(drug in singles for drug in key)]
    if len(singles) != 6 or len(total_pairs) != 25 or len(pair_keys) != 7:
        raise ValueError("Matched single-agent support changed")
    all_add = np.stack([sum(singles[drug] for drug in key) for key in pair_keys])
    observed = np.stack([responses[key] for key in pair_keys])
    means = observed.mean(axis=1)
    scalars = []
    predictions = []
    pairs = []
    for i, key in enumerate(pair_keys):
        train = np.arange(len(pair_keys)) != i
        factor = float(np.clip(np.sum(all_add[train] * means[train]) / np.sum(all_add[train] ** 2), 0, 1))
        forecast = factor * all_add[i]
        predictions.append(forecast)
        scalars.append(factor)
        benchmark = mse(observed[i], 0.0)
        additive = mse(observed[i], all_add[i])
        shrunk = mse(observed[i], forecast)
        pair = {
            "compound_ids": key, "calibration_pair_ids": [k for j, k in enumerate(pair_keys) if j != i],
            "fitted_shrinkage": factor,
            "wells": [r[0] for r in groups[key]],
            "n_cells_per_well": [r[1] for r in groups[key]],
            "mse_well": {"no_change": benchmark.tolist(), "additive": additive.tolist(), "shrinkage": shrunk.tolist()},
            "mse": {"no_change": float(np.mean(benchmark)), "additive": float(np.mean(additive)), "shrinkage": float(np.mean(shrunk))},
            "amplitude": {"observed": float(np.sqrt(np.mean(means[i]**2))),
                          "additive": float(np.sqrt(np.mean(all_add[i]**2))),
                          "shrinkage": float(np.sqrt(np.mean(forecast**2)))},
            "cosine_to_observed_mean": {"additive": vector_cosine(means[i], all_add[i]),
                                        "shrinkage": vector_cosine(means[i], forecast)},
        }
        pairs.append(pair)
    predictions = np.stack(predictions)
    gene_order = np.argsort(-np.mean(np.stack(list(singles.values()))**2, axis=0), kind="stable")
    gene_idx = gene_order[np.array([0,1,2,4,7,12,18,26,39,57,85,127])]
    display_observed, display_forecast = means[:,gene_idx], predictions[:,gene_idx]
    macro = {m: float(np.mean([row["mse"][m] for row in pairs])) for m in ("no_change","additive","shrinkage")}
    relative = {m: 100*(macro["no_change"]-macro[m])/macro["no_change"] for m in ("additive","shrinkage")}
    support = [{"condition": key, "n_wells": len(rows), "n_cells": sum(v[1] for v in rows)}
               for key, rows in sorted(groups.items())]
    return {
        "assay_context": "A549 human lung adenocarcinoma cell line",
        "condition_support": support, "condition_count": len(groups), "well_count": sum(map(len, groups.values())),
        "pair_count": len(total_pairs), "supported_pair_count": len(pair_keys),
        "matched_single_agents": sorted(singles), "shared_vehicle_wells": [r[0] for r in groups[VEHICLE]],
        "shared_vehicle_cells": sum(r[1] for r in groups[VEHICLE]),
        "pairs": pairs, "macro_MSE": macro, "macro_gain_vs_no_change_percent": relative,
        "fit": {"kind": "leave-one-pair-out additive shrinkage",
                "calibration_uses": "all genes in the processed panel and measured mean vectors of six other pairs",
                "training_objective": "minimize squared error for one common scalar in [0,1]",
                "available_at_prediction": "both same-cohort measured single-agent responses and shared vehicle wells",
                "dose_note": "one recorded condition per agent or pair in this dataset"},
        "gene_display": {"method": "12 fixed ranks (0-based) of decreasing mean square response across the six single agents",
                         "genes": [{"id": ids[j], "symbol": genes[j], "rank": int(k)}
                                   for j,k in zip(gene_idx,[0,1,2,4,7,12,18,26,39,57,85,127])],
                         "observed": display_observed.tolist(), "predicted": display_forecast.tolist(),
                         "range": float(np.max(np.abs([display_observed, display_forecast])))},
    }


def vehicle_states(wells, counts, ids):
    vehicle = np.concatenate([idx for (condition, _), idx in sorted(wells.items()) if condition == VEHICLE])
    controls = counts[vehicle].astype(np.float64)
    library = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    if not np.all(library > 0):
        raise ValueError("Zero library size on the 5,000-gene public axis")
    normalized = np.log1p(1e4 * controls.toarray() / library[vehicle, None])
    ix = np.lexsort((ids, -normalized.var(axis=0)))[:128]
    mean = normalized[:, ix].mean(axis=0)
    scale = normalized[:, ix].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Nonvariable genes in the control-state input")
    fit = KMeans(n_clusters=4, random_state=20261113, n_init=20, algorithm="lloyd")
    fit.fit((normalized[:, ix] - mean) / scale)
    data = np.log1p(1e4 * counts[:, ix].toarray().astype(np.float64) / library[:, None])
    labels = fit.predict((data-mean)/scale)
    # Deterministic display ordering by descending vehicle-cell frequency.
    original = np.bincount(labels[vehicle], minlength=4)
    if np.min(original) <= 0:
        raise ValueError("A fitted state has no vehicle support")
    order = np.lexsort((np.arange(4), -original))
    remap = np.empty(4,dtype=int)
    remap[order] = np.arange(4)
    labels = remap[labels]
    p = []
    for state in range(4):
        state_counts = np.asarray(counts[vehicle[labels[vehicle] == state]].sum(axis=0),dtype=np.float64).ravel()
        p.append(state_counts/state_counts.sum())
    return labels, np.stack(p), ix, {
        "names": [f"Vehicle state {i+1}" for i in range(4)],
        "vehicle_cells_per_state": np.bincount(labels[vehicle],minlength=4).tolist(),
        "vehicle_cell_count": len(vehicle),
        "input_feature_count": len(ix),
        "inertia": float(fit.inertia_),
        "control_only_fit": True,
    }


def components(wells, counts, labels, reference_states):
    by_cond = defaultdict(list)
    for (condition, well), ix in sorted(wells.items()):
        count_vectors = [
            np.asarray(counts[np.asarray(ix)[labels[ix]==state]].sum(axis=0),dtype=np.float64).ravel()
            for state in range(len(reference_states))
        ]
        mass = np.sum(count_vectors,axis=1)
        raw = np.sum(count_vectors,axis=0)
        fractions = mass/mass.sum()
        theoretical = fractions@reference_states
        observed = np.log1p(1e4 * raw/raw.sum())
        composition = np.log1p(1e4 * theoretical)
        within = observed-composition
        by_cond[condition].append({"well":well,"n_cells":len(ix),
                                   "fractions":fractions,"measured":observed,"composition":composition,"within":within,
                                   "cells_by_state":np.bincount(labels[ix],minlength=len(reference_states))})
    mean = {key:np.mean([r[key] for r in by_cond[VEHICLE]],axis=0)
            for key in ("measured","composition","within")}
    for group in by_cond.values():
        for r in group:
            for key in mean:
                r[key] -= mean[key]
    exact = max(float(np.max(np.abs(r["measured"]-r["composition"]-r["within"])))
                for group in by_cond.values() for r in group)
    if exact>1e-12:
        raise ValueError("Composition/within-state additivity failed")
    return by_cond,exact


def fit_two(features, truth):
    x = np.moveaxis(features,1,-1).reshape(-1,2)
    result = lsq_linear(x,truth.ravel(),bounds=(0.0,1.0),method="bvls",tol=1e-12)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError("Two-component fit did not qualify")
    return result.x


def hash_arrays(preds):
    obj = hashlib.sha256()
    for x in preds:
        obj.update(x.astype("<f8",copy=False).tobytes())
    return obj.hexdigest()


def train_and_score(groups, baseline, ids):
    singles = {next(d for d in cond if d!="dmso"):{k:np.mean([r[k] for r in g],axis=0)
               for k in ("measured","composition","within")}
               for cond,g in groups.items() if cond!=VEHICLE and "dmso" in cond}
    pairs = [tuple(row["compound_ids"]) for row in baseline["pairs"]]
    measured = np.stack([[r["measured"] for r in groups[pair]] for pair in pairs])
    means = measured.mean(axis=1)
    additive = np.stack([singles[a]["measured"]+singles[b]["measured"] for a,b in pairs])
    c_w = np.stack([[singles[a][k]+singles[b][k] for k in ("composition","within")] for a,b in pairs])
    if not np.allclose(c_w.sum(axis=1),additive,atol=1e-12,rtol=0):
        raise ValueError("Summed C/W branches must reproduce the uncalibrated additive forecast")
    ordered = [sorted(pair,key=lambda name:(-np.linalg.norm(singles[name]["measured"]),name)) for pair in pairs]
    linear = np.stack([[singles[a]["measured"],singles[b]["measured"]] for a,b in ordered])
    predicted, fold = [],[]
    # Predictions are formed solely from singles/vehicle and six other pair means.
    for i,pair in enumerate(pairs):
        train = np.arange(len(pairs))!=i
        training_truth = means[train]
        shrinkage = np.clip(np.sum(additive[train]*training_truth)/np.sum(additive[train]**2),0,1)
        linear_w = fit_two(linear[train],training_truth)
        cw_w = fit_two(c_w[train],training_truth)
        pred = np.stack([np.zeros_like(additive[i]),additive[i],shrinkage*additive[i],
                         linear_w@linear[i],cw_w@c_w[i]])
        predicted.append(pred)
        fold.append({"compound_ids":pair,"calibration_pair_ids":[x for j,x in enumerate(pairs) if j!=i],
                     "shrinkage":float(shrinkage),"linear_agents_in_coefficient_order":ordered[i],
                     "linear_weights":linear_w.tolist(),"cw_weights":cw_w.tolist(),
                     "prediction_sha256":hash_arrays(pred),
                     "cw_component_RMS":{"C":float(np.linalg.norm(c_w[i,0])/np.sqrt(len(ids))),
                                         "W":float(np.linalg.norm(c_w[i,1])/np.sqrt(len(ids)))}})
    predicted = np.stack(predicted)
    result = []
    for i,pair in enumerate(pairs):
        p,g = predicted[i],groups[pair]
        error = np.mean((p[:,None,:]-measured[i][None,:,:])**2,axis=2)
        for j,name in enumerate(METHODS[:3]):
            if not np.allclose(error[j],baseline["pairs"][i]["mse_well"][name],atol=2e-12,rtol=0):
                raise ValueError("Independent baseline well-score replay failed")
        result.append(dict(fold[i],wells=[x["well"] for x in g],n_cells_per_well=[x["n_cells"] for x in g],
                           cells_by_state=[x["cells_by_state"].tolist() for x in g],
                           library_mass_fractions_by_state=[x["fractions"].tolist() for x in g],
                           mse_well={name:error[j].tolist() for j,name in enumerate(METHODS)},
                           mse={name:float(error[j].mean()) for j,name in enumerate(METHODS)}))
    macro = {name:float(np.mean([row["mse"][name] for row in result])) for name in METHODS}
    vs_add = {name:100*(macro["additive"]-value)/macro["additive"] for name,value in macro.items()}
    vs_shr = {name:100*(macro["shrinkage"]-value)/macro["shrinkage"] for name,value in macro.items()}
    vs_linear = {name:100*(macro["fitted_linear"]-value)/macro["fitted_linear"] for name,value in macro.items()}
    gene_ids = baseline["gene_display"]["genes"]
    lookup = {name:i for i,name in enumerate(ids)}
    idx = [lookup[row["id"]] for row in gene_ids]
    d = {"genes":gene_ids,"observed":means[:,idx].tolist(),
         **{name:predicted[:,j,idx].tolist() for j,name in enumerate(METHODS)},
         "range":float(max(np.max(np.abs(means[:,idx])),np.max(np.abs(predicted[:,:,idx])))),
         "selection":baseline["gene_display"]["method"]}
    quantile = np.linspace(.025,.975,20)
    density = []
    for i in range(len(pairs)):
        # Same shared axis for every method, retaining full-bin counts and means.
        bin_order = np.argsort(means[i],kind="stable")
        bins = np.array_split(bin_order,20)
        density.append({"n_genes":len(ids),"n_bins":len(bins),"bin_quantile_centres":quantile.tolist(),
                        "observed":[float(np.mean(means[i,z])) for z in bins],
                        **{name:[float(np.mean(predicted[i,j,z])) for z in bins] for j,name in enumerate(METHODS)}})
    return {
        "condition_count":baseline["condition_count"],"well_count":baseline["well_count"],
        "pair_count":baseline["pair_count"],"supported_pair_count":len(pairs),
        "shared_vehicle_wells":baseline["shared_vehicle_wells"],
        "methods":METHODS,"pairs":result,"macro_MSE":macro,
        "gain_vs_additive_percent":vs_add,"gain_vs_shrinkage_percent":vs_shr,
        "gain_vs_fitted_linear_percent":vs_linear,"gene_display":d,
        "paired_binned_scatter":density,
        "baseline_replayed_all_63_well_method_scores": True,
        "forecast_hash_sha256":hash_arrays(predicted),
    },predicted,means

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True, help="Decompressed sciplex3_comb.h5ad")
    parser.add_argument("--archive", type=Path, required=True, help="Original sciplex3_comb.h5ad.gz for verification")
    parser.add_argument("--metadata", type=Path, required=True, help="Original GEO cell metadata TSV gzip")
    parser.add_argument("--output", type=Path, required=True, help="Summary and plot-ready JSON output")
    parser.add_argument("--forecasts", type=Path, help="Optional full prediction NPZ output")
    args = parser.parse_args()
    inputs = {args.h5ad.resolve(), args.archive.resolve(), args.metadata.resolve()}
    outputs = [p.resolve() for p in (args.output, args.forecasts) if p is not None]
    if set(outputs) & inputs or len(set(outputs)) != len(outputs):
        parser.error("Output paths must be distinct from one another and from inputs")

    began = time.perf_counter()
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    wells, counts, ids, genes, source = read_public_data(args.h5ad, args.archive, args.metadata)
    baseline_groups, responses = bulk_response(wells, counts)
    baseline = score_baselines(baseline_groups, responses, ids, genes)
    labels, references, features, states = vehicle_states(wells, counts, ids)
    groups, algebra_error = components(wells, counts, labels, references)
    analysis, forecasts, observed = train_and_score(groups, baseline, ids)
    result = {
        "schema": "strokeperturb_a549_fold_refit_composition_within_state_v1",
        "label": "StrokePerturb-inspired composition and within-state adaptation (A549)",
        "design": design,
        "design_sha256": hashes(DESIGN, "sha256")["sha256"],
        "source": source,
        "gene_panel_size": len(ids),
        "endpoint": design["endpoint"],
        "unit": "Well pseudobulk, with equal weight for each of the seven eligible pairs",
        "state": dict(states, feature_stable_ids=ids[features].tolist(), decomposition_max_abs_error=algebra_error),
        "analysis": analysis,
        "archived_strokeperturb_checkpoint_applied": False,
        "external_stroke_outcomes_opened": False,
        "elapsed_seconds": time.perf_counter() - began,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.forecasts is not None:
        args.forecasts.parent.mkdir(parents=True, exist_ok=True)
        with args.forecasts.open("wb") as stream:
            np.savez_compressed(
                stream,
                forecast=forecasts,
                observed_mean=observed,
                genes=ids,
                methods=np.asarray(METHODS),
                pairs=np.asarray([r["compound_ids"] for r in analysis["pairs"]]),
            )
    print(json.dumps({"supported_pairs": analysis["supported_pair_count"], "macro_MSE": analysis["macro_MSE"]}))


if __name__ == "__main__":
    main()
