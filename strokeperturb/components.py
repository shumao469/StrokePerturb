"""Training-only component labels and benchmark reference adjustment."""

import numpy as np

from .calibration import axis, matrix, own_mask, require


def _counts(value, ndim, name):
    result = np.asarray(value, dtype=float)
    require(result.ndim == ndim and np.isfinite(result).all() and (result >= 0).all()
            and np.equal(result, np.floor(result)).all(), f'Valid integer counts required: {name}')
    return result


def decompose_training_bulk(bulk_means, sources, subclass_counts, subclasses,
                            control_means, control_sources, control_subclasses, control_counts):
    """Decompose observed TRAIN bulk ΔRNA into composition and within-class terms.

    Each row of ``subclass_counts`` must follow the order of ``bulk_means``
    and each count column must follow ``subclasses``. Control rows may repeat
    a subclass within a source. The count-weighted same-source pooled control
    mean supplies the fallback for unobserved classes; their fraction is
    returned for every treated population. All RNA means share the output axis.
    """
    sources, subclasses = axis(sources, 'sources'), axis(subclasses, 'subclasses')
    control_sources = axis(control_sources, 'control_sources')
    control_subclasses = axis(control_subclasses, 'control_subclasses')
    bulk = np.asarray(bulk_means, dtype=float)
    require(bulk.ndim == 2 and bulk.shape[0] == len(sources) and bulk.shape[1] >= 2
            and np.isfinite(bulk).all(), 'Aligned finite TRAIN bulk RNA means required')
    ctrl = matrix(control_means, (len(control_sources), bulk.shape[1]), 'control_means')
    require(len(set(subclasses)) == len(subclasses), 'Subclass names must be unique')
    require(len(control_sources) == len(control_subclasses), 'Unaligned control identities')
    require(set(control_sources) == set(sources), 'Control sources must match the TRAIN sources')
    counts = _counts(subclass_counts, 2, 'subclass_counts')
    require(counts.shape == (len(sources), len(subclasses)) and (counts.sum(axis=1) > 0).all(),
            'Each TRAIN population needs a nonempty count profile on the stated subclass axis')
    ctrl_counts = _counts(control_counts, 1, 'control_counts')
    require(ctrl_counts.shape == (len(control_sources),) and (ctrl_counts > 0).all(),
            'Each control row needs positive counts')

    response, composition, within, ntc_mean = [np.zeros_like(bulk) for _ in range(4)]
    missing_mass = np.zeros(len(bulk))
    for source in np.unique(sources):
        rows = np.flatnonzero(sources == source)
        selected = control_sources == source
        means, cells, names = ctrl[selected], ctrl_counts[selected], control_subclasses[selected]
        pooled = np.average(means, axis=0, weights=cells)
        prototypes = np.empty((len(subclasses), bulk.shape[1]))
        absent = np.zeros(len(subclasses), dtype=bool)
        for j, subclass in enumerate(subclasses):
            supported = names == subclass
            absent[j] = not supported.any()
            prototypes[j] = np.average(means[supported], axis=0, weights=cells[supported]) if supported.any() else pooled
        weights = counts[rows] / counts[rows].sum(axis=1, keepdims=True)
        matched_mean = weights @ prototypes
        response[rows] = bulk[rows] - pooled
        ntc_mean[rows] = pooled
        composition[rows] = matched_mean - pooled
        within[rows] = bulk[rows] - matched_mean
        missing_mass[rows] = weights[:, absent].sum(axis=1)
    return dict(response=response, composition=composition, within_subclass=within,
                pooled_training_NTC_mean=ntc_mean, missing_control_target_cell_fraction=missing_mass,
                targeted_cells=counts.sum(axis=1).astype(int),
                reconstruction_max_absolute_error=float(np.max(np.abs(response - composition - within))))


def benchmark_component_bridge(components, reference_half_mean):
    """Apply the TRAIN scoring-reference adjustment to the within-class term."""
    pooled = np.asarray(components['pooled_training_NTC_mean'], dtype=float)
    reference = matrix(reference_half_mean, pooled.shape, 'reference_half_mean')
    adjustment = pooled - reference
    return dict(composition=components['composition'].copy(),
                within_subclass=components['within_subclass'] + adjustment,
                training_NTC_reference_adjustment=adjustment,
                benchmark_response=components['response'] + adjustment)


def split_training_residual(total_residual, composition, composition_nuisance, targets, genes):
    """Return C and W from aligned labels and supplied cross-fitted nuisance.

    ``total_residual`` is the original benchmark TRAIN residual. ``composition``
    is the pooled-control composition component aligned to that residual's
    rows, including the original technical repetitions when applicable.
    ``composition_nuisance`` must be held-out-animal TRAIN predictions from the
    nuisance model; this function does not fit that model.
    """
    targets, genes = axis(targets, 'targets'), axis(genes, 'genes')
    require(len(genes) >= 2 and len(set(genes)) == len(genes), 'Unique ordered output genes required')
    shape = (len(targets), len(genes))
    total = matrix(total_residual, shape, 'total_residual')
    raw = matrix(composition, shape, 'composition') - matrix(composition_nuisance, shape, 'composition_nuisance')
    own, _ = own_mask(targets, genes)
    masked = own >= 0
    raw[np.flatnonzero(masked), own[masked]] = 0.0
    names, counts = np.unique(targets, return_counts=True)
    count = dict(zip(names, counts))
    weights = np.array([1.0 / (len(names) * count[t]) for t in targets])
    residual_c = raw - weights @ raw
    residual_w = total - residual_c
    np.testing.assert_allclose(residual_c + residual_w, total, atol=1e-12, rtol=1e-12)
    return dict(C=residual_c, W=residual_w)
