"""Compute matched-rank and cosine diagnostics on centred scored arrays."""

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np

KS = (10, 25, 50, 100)
SEED = 20261115
REPS = 10000


def overlap(mice, genes, truth, pred, masks):
    result = np.zeros((len(mice), len(mice), 2, len(KS)))
    sort = lambda values: [np.lexsort((genes, row * sign)) for sign in (-1, 1) for row in values]
    observed, predicted = np.reshape(sort(truth), (2, len(mice), -1)), np.reshape(sort(pred), (2, len(mice), -1))
    for i in range(len(mice)):
        for j in np.flatnonzero(mice == mice[i]):
            shared = masks[i] & masks[j]
            for sign in range(2):
                a, b = observed[sign, i], predicted[sign, j]
                a, b = a[shared[a]][:max(KS)], b[shared[b]][:max(KS)]
                for k_idx, k in enumerate(KS):
                    result[i, j, sign, k_idx] = np.intersect1d(a[:k], b[:k], assume_unique=True).size / k
    return result


def rank_test(kit):
    mice, targets, genes, truth, pred, masks = kit
    pairs = overlap(mice, genes, truth, pred, masks)
    groups = {}
    for target in np.unique(targets):
        rows = np.flatnonzero(targets == target)
        groups.setdefault(tuple(mice[rows]), []).append(rows)
    blocks = [np.asarray(rows) for rows in groups.values()]
    assert sorted(map(len, blocks)) == [7, 16, 17]
    rng = np.random.default_rng(SEED)
    pos = np.arange(len(mice))
    genuine = pairs[pos, pos].mean(axis=0)
    null = np.empty((REPS, 2, len(KS)))
    for rep in range(REPS):
        scrambled = pos.copy()
        for block in blocks:
            scrambled[block] = block[rng.permutation(len(block))]
        null[rep] = pairs[pos, scrambled].mean(axis=0)
    p = (1 + np.count_nonzero(null >= genuine - 1e-14, axis=0)) / (REPS + 1)
    order = np.argsort(p.ravel(), kind='stable')
    adjusted = np.empty(p.size)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p.ravel()[order] * np.arange(p.size, 0, -1)))
    return [dict(rank_tail=sign, K=k, observed_overlap=genuine[s, j], permutation_mean=null[:, s, j].mean(),
                 permutation_q025=np.quantile(null[:, s, j], .025), permutation_q975=np.quantile(null[:, s, j], .975),
                 p_one_sided=p[s, j], p_holm=adjusted.reshape(p.shape)[s, j], permutations=REPS, seed=SEED)
            for s, sign in enumerate(('upper', 'lower')) for j, k in enumerate(KS)]


def equal_candidate_control(kit):
    mice, targets, _, truth, pred, masks = kit
    rows = []
    for i, target in enumerate(targets):
        cosines = np.empty(len(targets))
        for j in range(len(targets)):
            shared = masks[i] & masks[j]
            a, b = pred[i, shared], truth[j, shared]
            cosines[j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        same, other = cosines[targets == target], cosines[targets != target]
        value = np.max(same)
        n, k = len(other), len(same)
        less, equal = np.count_nonzero(other < value), np.count_nonzero(other == value)
        total = comb(n, k)
        wins = comb(int(less), k) / total
        ties = (comb(int(less + equal), k) - comb(int(less), k)) / total
        rows.append(dict(animal=mice[i], target=target, matching_candidate_count=k,
                         other_candidates_available=n, same_target_max_cosine=value,
                         strict_win_probability=wins, tie_probability=ties))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scored_arrays', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    with np.load(args.scored_arrays, allow_pickle=False) as z:
        names = ('mice', 'targets', 'genes', 'observed_centered', 'predicted_centered', 'mask')
        mice, targets, genes, truth, pred, masks = kit = tuple(z[name] for name in names)
    if (mice.shape != (56,) or targets.shape != (56,) or genes.shape != (19070,)
            or len(np.unique(genes)) != 19070 or len(np.unique(targets)) != 40
            or len(set(zip(mice, targets))) != 56
            or truth.shape != (56, 19070) or pred.shape != truth.shape
            or masks.dtype != bool or masks.shape != truth.shape
            or not (np.isfinite(truth[masks]).all() and np.isfinite(pred[masks]).all())):
        raise ValueError('Expected the aligned scored arrays of the 56-query joint block.')
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_directory / 'Fig3f_rank_target_permutation.csv', rank_test(kit))
    write_csv(args.output_directory / 'Extended_Data_Fig3e_equal_candidate_cosine.csv', equal_candidate_control(kit))


if __name__ == '__main__':
    main()
