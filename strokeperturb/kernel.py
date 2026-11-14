"""The fixed linear, unpooled residual-kernel primitive used by the pilot."""

import numpy as np

from .calibration import axis, matrix, own_mask, require


def predict_linear_residual(residual, sources, targets, genes, descriptors, query_descriptors,
                            reliability, *, label_basis=None, output_basis=None):
    """Fit one residual head on supplied TRAIN arrays and predict aligned queries.

    ``descriptors`` follows sorted unique TRAIN targets and represents the
    primary descriptor view. ``query_descriptors`` follows query population
    order on the same feature axis. Supply a mapping of TRAIN target names to
    reliability weights. Control alignment, nuisance fitting, descriptor and
    archetype construction happen upstream.

    For C or W, omit both bases to retain the complete output RNA axis. For U,
    supply the TRAIN-derived label projection and output decoding basis. They
    share a row count and may be different arrays. The pilot uses 16 rows.
    No query expression, target identity pooling or control-state interaction
    enters this fixed linear configuration. The pilot ridge penalty is 1.
    """
    sources, targets, genes = axis(sources, 'sources'), axis(targets, 'targets'), axis(genes, 'genes')
    require(len(sources) == len(targets) and len(genes) >= 2 and len(set(genes)) == len(genes),
            'Aligned TRAIN rows and a unique gene axis required')
    labels = matrix(residual, (len(sources), len(genes)), 'residual').copy()
    own, _ = own_mask(targets, genes)
    valid = own >= 0
    labels[np.flatnonzero(valid), own[valid]] = 0.0
    require((label_basis is None) == (output_basis is None), 'Supply both U bases or neither')
    projected = label_basis is not None
    if projected:
        basis = np.asarray(label_basis, dtype=float)
        require(basis.ndim == 2 and basis.shape[1] == len(genes) and len(basis) > 0
                and np.isfinite(basis).all(), 'Invalid label basis')
        output_basis = matrix(output_basis, basis.shape, 'output_basis')
        labels = labels @ basis.T

    keys = sorted(set(zip(sources.tolist(), targets.tolist())))
    labels = np.stack([labels[(sources == source) & (targets == target)].mean(axis=0)
                       for source, target in keys])
    pair_targets = np.array([t for _, t in keys])
    names, counts = np.unique(pair_targets, return_counts=True)
    index = np.searchsorted(names, pair_targets)
    required = set(names)
    require(required <= set(reliability), 'Missing TRAIN reliability weights')
    weights = np.array([reliability[t] for t in names], dtype=float)
    require(weights.shape == names.shape and np.isfinite(weights).all() and (weights >= 0).all(),
            'TRAIN target reliabilities must be finite nonnegative scalars')
    weights = np.maximum(weights, .05)[index] / counts[index]
    weights /= weights.sum()

    descriptor = np.asarray(descriptors, dtype=float)
    require(descriptor.ndim == 2 and len(descriptor) == len(names) and np.isfinite(descriptor).all(),
            'One finite descriptor row per sorted TRAIN target required')
    query = np.asarray(query_descriptors, dtype=float)
    require(query.ndim == 2 and len(query) > 0 and query.shape[1] == descriptor.shape[1]
            and np.isfinite(query).all(), 'Finite query descriptors must share the TRAIN feature axis')
    centre = descriptor.mean(axis=0)
    scale = np.maximum(descriptor.std(axis=0), .1)
    training = (descriptor - centre) / scale
    dimension = max(float(np.mean(np.sum(training ** 2, axis=1))), 1e-12)
    target_kernel = training @ training.T / dimension
    descriptor_mean = target_kernel.mean(axis=0)
    descriptor_grand = float(target_kernel.mean())

    def centred_descriptor_kernel(standardised):
        raw = standardised @ training.T / dimension
        return raw - raw.mean(axis=1)[:, None] - descriptor_mean[None, :] + descriptor_grand

    raw = centred_descriptor_kernel(training[index])[:, index]
    mean = raw @ weights
    grand = float(weights @ mean)
    centred = raw - mean[:, None] - mean[None, :] + grand
    root = np.sqrt(weights)
    covariance = root[:, None] * centred * root[None, :]
    values, axes = np.linalg.eigh((covariance + covariance.T) / 2)
    values = np.maximum(values, 0.0)
    query_kernel = centred_descriptor_kernel((query - centre) / scale)[:, index]
    query_kernel -= (query_kernel @ weights)[:, None] + mean[None, :] - grand
    centred_labels = labels - weights @ labels
    root_axes = root[:, None] * axes
    if projected:
        right = axes.T @ (root[:, None] * centred_labels)
        prediction = (query_kernel @ ((root_axes / (values + 1.0)[None, :]) @ right)) @ output_basis
    else:
        dual = (root_axes / (values + 1.0)[None, :]) @ root_axes.T
        prediction = (query_kernel @ dual) @ centred_labels
    require(np.isfinite(prediction).all(), 'Nonfinite residual prediction')
    return prediction
