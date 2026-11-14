"""StrokePerturb component calibration and paired loss replay.

The numerical routines implement the fixed-grid nested pilot calculation.
Callers supply aligned, precomputed component predictions and eligible outcomes.
Calibrate on complete inner out-of-fold rows before applying the coefficients
to any outer query. Within-animal centring occurs after inner concatenation.
"""

import hashlib
import itertools
import json

import numpy as np


GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ENDPOINTS = ('downstream_MSE', 'target_specific_downstream_MSE')
PRIMARY = ENDPOINTS[1]
HEADS = ('background', 'U', 'C', 'W')
RTOL, ATOL = 1e-8, 1e-10
CALIBRATION_SCHEMA = 'nested_pilot_inner_calibration_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def axis(value, name):
    a = np.asarray(value)
    require(a.ndim == 1 and len(a) and a.dtype.kind == 'U' and np.all(a != ''),
            'Nonempty Unicode identity vector required: ' + name)
    return a


def matrix(value, shape, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fiu' and np.isfinite(a).all(),
            'Aligned finite numeric matrix required: ' + name)
    return a.astype(np.float64, copy=False)


def validate_axes(source, target, genes):
    source, target, genes = axis(source, 'source'), axis(target, 'target'), axis(genes, 'genes')
    require(len(source) == len(target) and len(genes) >= 2 and len(set(genes)) == len(genes),
            'Aligned source/target and unique gene axes required')
    require(len(set(zip(source, target))) == len(source), 'Duplicate biological mouse-target rows')
    return source, target, genes


def validate_heads(heads, source, target, genes):
    source, target, genes = validate_axes(source, target, genes)
    require(set(heads) == set(HEADS), 'Exact background/U/C/W schema required')
    return {name: matrix(heads[name], (len(source), len(genes)), name) for name in HEADS}


def axes_receipt(source, target, genes):
    source, target, genes = validate_axes(source, target, genes)
    pairs = list(zip(source.tolist(), target.tolist()))
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return dict(rows=len(source), output_genes=len(genes),
                ordered_pairs_sha256=hashlib.sha256(encoded).hexdigest(),
                ordered_genes_sha256=hashlib.sha256('\n'.join(genes.tolist()).encode('utf-8')).hexdigest())


def own_mask(target, genes):
    lookup = {g: i for i, g in enumerate(genes)}
    own = np.array([lookup.get(t, -1) for t in target], dtype=np.int64)
    included = np.ones((len(target), len(genes)), dtype=bool)
    known = np.flatnonzero(own >= 0)
    included[known, own[known]] = False
    return own, included


def _row_terms(arrays, included):
    error, *directions = arrays
    d = included.sum(axis=1)
    dot = lambda a, b: np.einsum('ij,ij->i', a, b) / d
    gram = np.empty((len(error), 3, 3), dtype=np.float64)
    for i, j in itertools.combinations_with_replacement(range(3), 2):
        gram[:, i, j] = gram[:, j, i] = dot(directions[i], directions[j])
    return dot(error, error), np.column_stack([-2.0 * dot(error, x) for x in directions]), gram


def _polynomial_rows(truth, heads, source, target, genes):
    """E^2, -2 E*D, D^T D rows with D=(U,C,W); factor 2 comes from both Gram entries."""
    _, include = own_mask(target, genes)
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        raw = [truth - heads['background'], heads['U'], heads['C'], heads['W']]
        masked = [np.where(include, x, 0.0) for x in raw]
        downstream = _row_terms(masked, include)
        specific = tuple(np.empty_like(x) for x in downstream)
        informative = np.zeros(len(source), dtype=bool)
        for mouse in np.unique(source):
            ix = np.flatnonzero(source == mouse)
            valid = include[ix]
            support = valid.sum(axis=0)
            informative[ix] = np.any(valid & (support > 1), axis=1)
            residuals = []
            for block in masked:
                values = block[ix]
                means = np.divide(values.sum(axis=0), support, where=support > 0,
                                  out=np.zeros(len(genes), dtype=np.float64))
                residuals.append(np.where(valid, values - means, 0.0))
            for result, value in zip(specific, _row_terms(residuals, valid)):
                result[ix] = value
    return dict(zip(ENDPOINTS, (downstream, specific))), informative


def _eval_terms(terms, coefficients):
    a, b, g = terms
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        x = a + b @ coefficients + np.einsum('nij,i,j->n', g, coefficients, coefficients)
    require(np.isfinite(x).all(), 'Nonfinite polynomial loss')
    return x


def target_macro(values, target):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == target.shape and np.isfinite(values).all(), 'Bad target-macro loss rows')
    return float(np.mean([np.mean(values[target == t], dtype=np.float64) for t in np.unique(target)]))


def _score_grid(terms, coefficients, target):
    return {name: target_macro(_eval_terms(terms[name], np.asarray(coefficients, dtype=np.float64)), target)
            for name in ENDPOINTS}


def direct_rows(truth, forecast, source, target, genes):
    """Independent direct error calculation, not the quadratic grid expansion."""
    source, target, genes = validate_axes(source, target, genes)
    shape = (len(source), len(genes))
    truth, forecast = matrix(truth, shape, 'truth'), matrix(forecast, shape, 'forecast')
    lookup = dict(zip(genes, range(len(genes))))
    downstream, specific = np.empty(len(source)), np.empty(len(source))
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        for m in sorted(set(source)):
            rows = np.flatnonzero(source == m)
            error = np.subtract(truth[rows], forecast[rows])
            included = np.full(error.shape, True, dtype=bool)
            for i, row in enumerate(rows):
                own = lookup.get(target[row])
                if own is not None:
                    included[i, own] = False
            clean = np.copy(error)
            clean[~included] = 0.0
            total = np.add.reduce(clean, axis=0)
            support = np.count_nonzero(included, axis=0)
            centre = total / np.maximum(1, support)
            deviation = np.subtract(error, centre)
            deviation[~included] = 0.0
            n = np.count_nonzero(included, axis=1)
            downstream[rows] = np.add.reduce(np.square(clean), axis=1) / n
            specific[rows] = np.add.reduce(np.square(deviation), axis=1) / n
    require(np.isfinite(downstream).all() and np.isfinite(specific).all(), 'Nonfinite direct loss')
    return dict(zip(ENDPOINTS, (downstream, specific)))


def _blend_coeff(a, b, c, lam):
    return ((1.0-lam)*a, lam*b, lam*c)


def _check_coeff(x):
    require(type(x) in (float, int) and x in GRID, 'Coefficient must belong to the original fixed grid')
    return float(x)


def _validate_calibration(calibration, genes):
    require(calibration.get('schema') == CALIBRATION_SCHEMA and calibration.get('grid') == list(GRID)
            and calibration.get('objective') == PRIMARY and calibration.get('outer_outcomes_used') is False
            and calibration.get('independent_replay', {}).get('passed') is True, 'Frozen inner calibration required')
    gene_hash = hashlib.sha256('\n'.join(axis(genes, 'genes').tolist()).encode('utf-8')).hexdigest()
    require(calibration.get('axes', {}).get('ordered_genes_sha256') == gene_hash,
            'Outer gene axis differs from the frozen inner gene axis')
    winner = calibration['strongest_family']
    require(winner in ('low_rank', 'full_RNA'), 'Unknown calibrated family')
    a = _check_coeff(calibration['low_rank']['a'])
    b = _check_coeff(calibration['full_RNA']['b'])
    c = _check_coeff(calibration['full_RNA']['c'])
    lam = _check_coeff(calibration['mixture']['lambda_full_RNA'])
    return a, b, c, lam, winner


def _apply(heads, a, b, c, lam, winner):
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        low = heads['background'] + a * heads['U']
        full = heads['background'] + b * heads['C'] + c * heads['W']
        candidate = (1.0-lam)*low + lam*full
        # Independent fused reconstruction of the same convex mix. A seal may
        # store the explicit blend, which must match the direct residual formula.
        ka, kb, kc = _blend_coeff(a, b, c, lam)
        replay = heads['background'] + ka*heads['U'] + kb*heads['C'] + kc*heads['W']
        np.testing.assert_allclose(candidate, replay, rtol=RTOL, atol=ATOL, equal_nan=False)
    return dict(candidate=candidate, comparator=low if winner == 'low_rank' else full,
                low_rank=low, full_RNA=full)


def apply_calibration(heads, source, target, genes, frozen_calibration):
    """Compose outer candidate and comparator predictions; requires no outcomes."""
    heads = validate_heads(heads, source, target, genes)
    return _apply(heads, *_validate_calibration(frozen_calibration, genes))


def calibrate_inner(truth, heads, source, target, genes):
    """Calibrate component weights on the complete inner out-of-fold axis."""
    source, target, genes = validate_axes(source, target, genes)
    truth = matrix(truth, (len(source), len(genes)), 'inner truth')
    heads = validate_heads(heads, source, target, genes)
    terms, informative = _polynomial_rows(truth, heads, source, target, genes)
    require(informative.any(), 'Inner target-specific objective has no supported rows')
    lows = [dict(a=a, scores=_score_grid(terms, (a,0.0,0.0), target)) for a in GRID]
    fulls = [dict(b=b, c=c, scores=_score_grid(terms, (0.0,b,c), target))
             for b,c in itertools.product(GRID, repeat=2)]
    low = min(lows, key=lambda x: (x['scores'][PRIMARY], x['a']))
    full = min(fulls, key=lambda x: (x['scores'][PRIMARY], x['b']+x['c'], x['b'], x['c']))
    a, b, c = low['a'], full['b'], full['c']
    winner = 'low_rank' if low['scores'][PRIMARY] <= full['scores'][PRIMARY] else 'full_RNA'
    mixes = [dict(lambda_full_RNA=l, scores=_score_grid(terms, _blend_coeff(a,b,c,l), target)) for l in GRID]
    comparator_endpoint = 0.0 if winner == 'low_rank' else 1.0
    chosen = min(mixes, key=lambda x: (x['scores'][PRIMARY],
                x['lambda_full_RNA'] != comparator_endpoint, x['lambda_full_RNA']))
    lam = chosen['lambda_full_RNA']
    pred = _apply(heads, a, b, c, lam, winner)
    independent = []
    for name, coeff in [('low_rank', (a,0.,0.)), ('full_RNA', (0.,b,c)),
                         ('candidate', _blend_coeff(a,b,c,lam))]:
        ref = direct_rows(truth, pred[name], source, target, genes)
        for endpoint in ENDPOINTS:
            computed = _eval_terms(terms[endpoint], np.asarray(coeff))
            np.testing.assert_allclose(computed, ref[endpoint], rtol=RTOL, atol=ATOL, equal_nan=False)
            independent.append(dict(formula=name, endpoint=endpoint, rows=len(source),
                max_absolute_error=float(np.max(np.abs(computed-ref[endpoint])))))
    return dict(schema=CALIBRATION_SCHEMA, objective=PRIMARY, grid=list(GRID),
        aggregation='target_macro_original_rows', centering='joint_outer_TRAIN_axis_after_inner_concatenation',
        ties='exact-score ties: lowest a; lowest b+c then b then c; low_rank family; comparator endpoint lambda then lowest lambda',
        axes=axes_receipt(source,target,genes), rows_with_specific_support=int(informative.sum()),
        low_rank=low, full_RNA=full, strongest_family=winner, mixture=chosen,
        grids=dict(low_rank=lows, full_RNA=fulls, mixture=mixes),
        independent_replay=dict(passed=True, rtol=RTOL, atol=ATOL, comparisons=independent),
        outer_outcomes_used=False)


def score_outer(truth, heads, source, target, genes, frozen_calibration):
    """Evaluate fixed outer formula(s); no fitting, grid, or outer model selection."""
    source, target, genes = validate_axes(source, target, genes)
    truth = matrix(truth, (len(source), len(genes)), 'outer truth')
    heads = validate_heads(heads, source, target, genes)
    a,b,c,lam,winner = _validate_calibration(frozen_calibration, genes)
    forecasts = _apply(heads, a,b,c,lam,winner)
    terms, informative = _polynomial_rows(truth, heads, source, target, genes)
    low, full = (a,0.,0.), (0.,b,c)
    chosen = dict(candidate=_blend_coeff(a,b,c,lam), comparator=low if winner=='low_rank' else full,
                  low_rank=low, full_RNA=full)
    scores, rows, independent = {}, {}, []
    for name, coefs in chosen.items():
        rows[name] = direct_rows(truth, forecasts[name], source, target, genes)
        scores[name] = {e:target_macro(rows[name][e], target) for e in ENDPOINTS}
        for endpoint in ENDPOINTS:
            computed = _eval_terms(terms[endpoint], np.asarray(coefs))
            np.testing.assert_allclose(computed, rows[name][endpoint], rtol=RTOL, atol=ATOL, equal_nan=False)
            independent.append(dict(formula=name, endpoint=endpoint, rows=len(source),
                max_absolute_error=float(np.max(np.abs(computed-rows[name][endpoint])))))
    gains = {e:None if scores['comparator'][e] == 0 else 1.0-scores['candidate'][e]/scores['comparator'][e]
             for e in ENDPOINTS}
    return dict(axes=axes_receipt(source,target,genes), scores=scores, paired_relative_gain=gains,
        strongest_family_frozen_from_inner=winner, coefficients=dict(a=a,b=b,c=c,lambda_full_RNA=lam),
        rows_with_specific_support=int(informative.sum()),
        independent_replay=dict(passed=True, rtol=RTOL, atol=ATOL, comparisons=independent),
        outer_tuning_performed=False), rows
