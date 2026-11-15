"""Generate paired baseline and repeatability results from the scored joint block."""

import argparse
import csv
from pathlib import Path

import numpy as np


SEED = 20261116
N_PERM = 10_000


def median_iqr(values):
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    return {'n': int(np.size(values)), 'mean': float(np.mean(values)), 'median': float(median),
            'q1': float(q1), 'q3': float(q3)}


def mask_centre(values, valid):
    data = np.where(valid, values, 0)
    counts = valid.sum(axis=0)
    means = np.divide(data.sum(axis=0), counts, out=np.zeros(data.shape[1]), where=counts > 0)
    out = np.where(valid, data - means, 0)
    np.testing.assert_allclose(out.sum(axis=0), 0, atol=1e-12)
    return out


def pair_cosines(first, second, valid_first, valid_second):
    matrix = np.empty((len(first), len(second)))
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            common = valid_first[i] & valid_second[j]
            a, b = left[common], right[common]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            assert denom > 0
            matrix[i, j] = np.clip(a @ b / denom, -1, 1)
    return matrix


def no_change_and_candidate(observed, candidate, valid):
    ref_loss = np.sum(np.square(observed), axis=-1) / valid.sum(axis=-1)
    candidate_loss = np.sum(np.square(observed - candidate), axis=-1) / valid.sum(axis=-1)
    return ref_loss, candidate_loss


def metrics(bundle):
    mice, targets, genes, observed, forecast, masks = bundle
    assert len(genes) == 19_070 and len(mice) == 56
    m1, m2 = np.unique(mice)
    by_mouse = [{str(targets[row]): row for row in np.flatnonzero(mice == mouse)} for mouse in (m1, m2)]
    shared = sorted(by_mouse[0].keys() & by_mouse[1].keys())
    assert len(shared) == 16 and sorted(map(len, by_mouse)) == [23, 33]
    indices = np.array([[which[t] for t in shared] for which in by_mouse])
    valid = masks[indices]
    np.testing.assert_array_equal(*valid)
    observed_panel = np.array([mask_centre(observed[ix], good) for ix, good in zip(indices, valid)])
    forecast_panel = np.array([mask_centre(forecast[ix], good) for ix, good in zip(indices, valid)])

    measured_matrix = pair_cosines(*observed_panel, *valid)
    measured_same = measured_matrix.diagonal()
    forecast_same = pair_cosines(*forecast_panel, *valid).diagonal()
    paired = np.stack([pair_cosines(o, f, m, m).diagonal()
                       for o, f, m in zip(observed_panel, forecast_panel, valid)])
    o_rms = np.sqrt(np.square(observed_panel).sum(axis=-1) / valid.sum(axis=-1))
    f_rms = np.sqrt(np.square(forecast_panel).sum(axis=-1) / valid.sum(axis=-1))
    ratio = f_rms / o_rms
    ref_loss, candidate_loss = no_change_and_candidate(observed_panel, forecast_panel, valid)

    rng = np.random.default_rng(SEED)
    n = len(shared)
    original_sum = measured_same.sum()
    null_sum = np.array([measured_matrix[np.arange(n), rng.permutation(n)].sum() for _ in range(N_PERM)])
    null_stats = np.quantile(null_sum / n, [.025, .5, .975])
    tolerance = 64 * np.finfo(float).eps * max(1, abs(original_sum))
    p = (1 + np.sum(null_sum >= original_sum - tolerance)) / (N_PERM + 1)
    counts, edges = np.histogram(null_sum / n, bins=21)
    assert counts.sum() == N_PERM
    null_bins = [{'left_edge': float(left), 'right_edge': float(right), 'count': int(count)}
                 for left, right, count in zip(edges[:-1], edges[1:], counts)]

    names = ('observed_cross_mouse_cosine', 'forecast_cross_mouse_cosine',
             'observed_forecast_mouse_A_cosine', 'observed_forecast_mouse_B_cosine',
             'observed_mouse_A_RMS', 'observed_mouse_B_RMS',
             'forecast_mouse_A_RMS', 'forecast_mouse_B_RMS',
             'forecast_to_observed_mouse_A_RMS_ratio', 'forecast_to_observed_mouse_B_RMS_ratio',
             'no_change_mouse_A_MSE', 'no_change_mouse_B_MSE', 'same_frozen_forecast_mouse_A_MSE',
             'same_frozen_forecast_mouse_B_MSE')
    values = zip(measured_same, forecast_same, *paired, *o_rms, *f_rms, *ratio, *ref_loss, *candidate_loss)
    rows = [{'target': target, 'valid_genes': int(valid[0, i].sum()),
             **{key: float(x) for key, x in zip(names, row)}}
            for i, (target, row) in enumerate(zip(shared, values))]

    return rows, {
        'analysis': 'Exploratory common-target panel diagnostic within the already scored joint block',
        'n_unique_animals': 2,
        'n_shared_targets': n,
        'n_animal_target_rows': int(indices.size),
        'genes_on_input_axis': len(genes),
        'centre': 'Per-gene, within mouse, same 16-target panel for both mice. Original own-target masks retained; pairwise cosine uses intersection. Re-centring a previously centred valid profile is equivalent to direct centring on this panel.',
        'observed_cross_mouse_cosine': median_iqr(measured_same),
        'observed_cross_mouse_cosine_positive_pairs': int((measured_same > 0).sum()),
        'forecast_cross_mouse_cosine': median_iqr(forecast_same),
        'cross_mouse_forecast_exactly_equal': bool(np.array_equal(*forecast_panel)),
        'cross_mouse_forecast_max_absolute_difference': float(np.max(np.abs(forecast_panel[0] - forecast_panel[1]))),
        'same_mouse_observed_forecast_cosine': median_iqr(paired),
        'same_mouse_observed_forecast_amplitude_ratio': median_iqr(ratio),
        'panel_diagnostic_MSE': {'zero': float(ref_loss.mean()), 'sealed_forecast': float(candidate_loss.mean()),
                                 'relative_reduction_percent': float(100 * (1 - candidate_loss.mean() / ref_loss.mean())),
                                 'warning': 'Uses common-panel centring; neither comparison changes the registered endpoint or gate.'},
        'observed_repeatability_permutation': {
            'statistic': 'Mean signed cosine across the 16 cross-mouse pairs',
            'seed': SEED, 'draws': N_PERM,
            'null': 'Permute the target-labelled observed profiles of mouse B across all 16 shared targets; a full permutation preserves the paired design and allows fixed labels. Profiles are not re-centred per draw.',
            'q025': float(null_stats[0]), 'median': float(null_stats[1]), 'q975': float(null_stats[2]),
            'mean': float(null_sum.mean() / n), 'p_one_sided': float(p),
            'min': float(edges[0]), 'max': float(edges[-1]),
            'scope': 'Target-label test conditional on a single mouse pair and the shared target panel. Targets do not provide independent mouse replicates.'},
        'original_gate': '0/12 at the original 1% requirement',
        'external_outcomes_opened': False,
    }, null_bins


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(summary):
    overview = []
    for label in ('observed_cross_mouse_cosine', 'forecast_cross_mouse_cosine',
                  'same_mouse_observed_forecast_cosine', 'same_mouse_observed_forecast_amplitude_ratio'):
        for statistic, value in summary[label].items():
            overview.append({'diagnostic': label, 'statistic': statistic, 'value': value})
    for label in ('cross_mouse_forecast_exactly_equal', 'cross_mouse_forecast_max_absolute_difference'):
        overview.append({'diagnostic': 'paired_forecast_identity', 'statistic': label.removeprefix('cross_mouse_forecast_'),
                         'value': summary[label]})
    for statistic in ('seed', 'draws', 'min', 'max', 'q025', 'median', 'q975', 'mean', 'p_one_sided'):
        overview.append({'diagnostic': 'observed_repeatability_target_permutation', 'statistic': statistic,
                         'value': summary['observed_repeatability_permutation'][statistic]})
    for statistic in ('zero', 'sealed_forecast', 'relative_reduction_percent'):
        overview.append({'diagnostic': 'common_panel_exploratory_MSE', 'statistic': statistic,
                         'value': summary['panel_diagnostic_MSE'][statistic]})
    return overview

def summarize(rows, endpoints, shape):
    target = np.array([row['target'] for row in rows])
    mouse = np.array([row['animal'] for row in rows])
    unique = np.unique(target)
    full_weights = np.array([1 / (len(unique) * np.count_nonzero(target == t)) for t in target])
    result = []
    for scope in ['All', *sorted(set(mouse))]:
        keep = np.ones(len(rows), bool) if scope == 'All' else mouse == scope
        weights = full_weights if scope == 'All' else keep / keep.sum()
        for endpoint in endpoints:
            suffix = 'centred' if endpoint.startswith('target_specific') else 'downstream'
            baseline = np.array([row[f'no_change_{suffix}'] for row in rows])
            for method, coeff in shape:
                loss = np.array([row[f'{method}_{suffix}'] for row in rows])
                # Paired differences are computed before summation.
                base = weights @ baseline
                delta = weights @ (baseline - loss)
                result.append(dict(scope=scope, populations=int(keep.sum()), targets=len(set(target[keep])),
                                   aggregation='Target macro' if scope == 'All' else 'Within-animal target mean',
                                   endpoint=endpoint, method=method, coefficient_on_W=coeff,
                                   MSE=float(weights @ loss), no_change_MSE=float(base),
                                   relative_gain_over_no_change_percent=float(100 * delta / base)))
    return result



def paired_baselines(mice, targets, measured, half, mask):
    prediction = {'no_change': np.zeros_like(half), 'calibrated_low_rank': np.zeros_like(half),
                  'calibrated_full_gene': half, 'selected_mixture': half, 'unit_residual': 2 * half}
    endpoints = ('downstream_MSE', 'target_specific_downstream_MSE')
    y = np.zeros_like(measured)
    w = np.zeros_like(half)
    for mouse in np.unique(mice):
        i = mice == mouse
        y[i] = mask_centre(measured[i], mask[i])
        w[i] = mask_centre(half[i], mask[i])
    count = mask.sum(axis=1)
    obs_rms = np.sqrt(np.sum(y*y,axis=1)/count)
    pred_rms = np.sqrt(np.sum(w*w,axis=1)/count)
    dots = np.sum(y*w,axis=1)/(obs_rms*pred_rms*count)
    rows = [dict(animal=str(a), target=str(t), gene_count=int(count[j]),
                 observed_centred_RMS=float(obs_rms[j]), half_residual_centred_RMS=float(pred_rms[j]),
                 half_residual_to_observed_RMS=float(pred_rms[j]/obs_rms[j]), half_residual_centred_cosine=float(dots[j]))
            for j, (a,t) in enumerate(zip(mice, targets))]
    for key, value in prediction.items():
        raw = np.where(mask, measured - value, 0.)
        centred = raw.copy()
        for mouse in np.unique(mice):
            i=mice==mouse
            centred[i]=mask_centre(raw[i],mask[i])
        raw_loss=np.sum(raw**2,axis=1)/count
        centre_loss=np.sum(centred**2,axis=1)/count
        for row, a,b in zip(rows,raw_loss,centre_loss):
            row[key+'_downstream']=a
            row[key+'_centred']=b
    method_scales=list(zip(prediction, (0,0,.5,.5,1)))
    return rows, summarize(rows,endpoints,method_scales),y,w


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scored_arrays',type=Path)
    p.add_argument('output_directory',type=Path)
    args=p.parse_args()
    with np.load(args.scored_arrays,allow_pickle=False) as z:
        mice, targets, genes, observed, half, mask=[z[k] for k in ('mice','targets','genes','observed_raw','predicted_raw','mask')]
    if (mice.shape != targets.shape or mice.shape != (56,) or len(np.unique(mice)) != 2
            or len(np.unique(targets)) != 40 or genes.shape != (19070,) or len(np.unique(genes)) != 19070
            or len(set(zip(mice,targets))) != 56 or observed.shape != (56,19070)
            or half.shape != observed.shape or mask.shape != observed.shape or mask.dtype != bool
            or not (np.isfinite(observed[mask]).all() and np.isfinite(half[mask]).all())):
        raise ValueError('Expected aligned arrays of the scored 56-query joint block.')
    rows, baseline, obs_c, pred_c = paired_baselines(mice,targets,observed.astype(float),half.astype(float),mask)
    common, panel, bins = metrics((mice,targets,genes,obs_c,pred_c,mask))
    out=args.output_directory
    out.mkdir(parents=True,exist_ok=True)
    write_rows(out/'query_baselines.csv',rows)
    write_rows(out/'baseline_summary.csv',baseline)
    write_rows(out/'common_panel_queries.csv',common)
    write_rows(out/'common_panel_summary.csv',summary_rows(panel))
    write_rows(out/'common_panel_permutation_bins.csv',bins)


if __name__=='__main__':
    main()
