"""Fixed-rank TRAIN response covariance; every original output gene is retained."""
import numpy as np

GENES = 19070
FIXED_RANK = 16
NULL_EIGENVALUE = 1e-12


def prepare_subspace(context, weights, *, gene_count=GENES):
    """Fit an output basis from weighted TRAIN target means, never query inputs.

    A = sqrt(target_weight) * (target_mean - weighted_grand_mean).
    The nonzero eigenvectors of A A.T give the right singular vectors of A.
    At most the prospectively fixed 16 leading directions are retained; this
    changes response rank without selecting any output gene or TRAIN row.
    """
    y = np.asarray(context['parent_labels'], dtype=np.float64)
    targets = np.asarray(context['pair_targets']).astype(str)
    names = np.asarray(context['names']).astype(str)
    w = np.asarray(weights, dtype=np.float64)
    if (names.ndim != 1 or not len(names) or len(set(names)) != len(names)
            or targets.ndim != 1 or set(targets) != set(names)):
        raise ValueError('Complete unique TRAIN target and pair axes required')
    if (y.shape != (len(targets), gene_count) or w.shape != targets.shape
            or not np.isfinite(y).all() or not np.isfinite(w).all()
            or (w <= 0).any() or abs(float(w.sum()) - 1.) > 1e-12):
        raise ValueError('Every finite TRAIN pair, the complete gene axis and normalized positive weights required')
    target_weights = np.empty(len(names), dtype=np.float64)
    target_means = np.empty((len(names), gene_count), dtype=np.float64)
    for i, name in enumerate(names):
        use = targets == name
        target_weights[i] = w[use].sum()
        target_means[i] = (w[use, None] * y[use]).sum(axis=0) / target_weights[i]
    intercept = target_weights @ target_means
    a = np.sqrt(target_weights)[:, None] * (target_means - intercept)
    gram = a @ a.T
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2.)
    # The sample Gram can have roundoff eigenvalues above an absolute floor
    # when the full-gene response norm is large (including exact rank-one
    # inputs). This fixed machine-precision rule handles that numerical null
    # spectrum; it is never selected using scores or query outcomes.
    null_threshold = max(NULL_EIGENVALUE, np.finfo(np.float64).eps * max(a.shape)
                         * max(float(eigenvalues.max(initial=0.)), 0.))
    if float(eigenvalues.min(initial=0.)) < -max(1e-9, null_threshold):
        raise ArithmeticError('TRAIN response covariance must be positive semidefinite')
    positive = np.flatnonzero(eigenvalues > null_threshold)
    keep = positive[-FIXED_RANK:][::-1]
    values = eigenvalues[keep]
    basis = (a.T @ vectors[:, keep]) / np.sqrt(values)[None, :]
    # Resolve arbitrary eigenvector signs without changing the projection.
    if len(keep):
        pivot = np.argmax(np.abs(basis), axis=0)
        signs = np.sign(basis[pivot, np.arange(len(keep))])
        basis *= signs[None, :]
        if not np.allclose(basis.T @ basis, np.eye(len(keep)), rtol=1e-9, atol=1e-9):
            raise ArithmeticError('Orthonormal TRAIN response directions required')
    if basis.shape != (gene_count, len(keep)) or not np.isfinite(basis).all():
        raise ArithmeticError('Finite dense full-gene TRAIN response basis required')
    return dict(basis=basis, rank=int(len(keep)), fixed_rank=FIXED_RANK,
                positive_covariance_rank=int(len(positive)), eigenvalues=values,
                numerical_null_eigenvalue=null_threshold,
                numerical_null_formula='max(1e-12, eps_float64 * max(TRAIN_target_count, gene_count) * largest_eigenvalue)',
                TRAIN_pair_count=int(len(targets)), TRAIN_target_count=int(len(names)),
                output_genes=gene_count, target_weights=target_weights,
                TRAIN_grand_mean=intercept, query_inputs_read=False,
                output_genes_removed=False, TRAIN_pairs_removed=False)


def apply_subspace(predictions, intercept, prepared, gamma):
    """Apply ((1-gamma) I + gamma P) only to the predicted centered slope.

    The caller supplies the actual (possibly shrunk) intercept used by the
    mean predictor. That intercept and the separate contrast component stay
    outside this operator. A zero gamma returns the original float64 object.
    """
    if gamma not in (0., .5, 1.):
        raise ValueError('Prospectively frozen subspace strength required')
    if not isinstance(predictions, np.ndarray) or predictions.dtype != np.float64:
        raise ValueError('Original float64 full-gene prediction array required')
    mu = np.asarray(intercept, dtype=np.float64)
    basis = np.asarray(prepared['basis'], dtype=np.float64)
    gene_count = prepared['output_genes']
    if (predictions.ndim != 2 or predictions.shape[1] != gene_count
            or mu.shape != (gene_count,) or basis.ndim != 2 or basis.shape[0] != gene_count
            or basis.shape[1] > FIXED_RANK or not np.isfinite(predictions).all()
            or not np.isfinite(mu).all() or not np.isfinite(basis).all()):
        raise ValueError('Complete finite aligned predictions, intercept and TRAIN basis required')
    if gamma == 0.:
        return predictions
    slope = predictions - mu
    projected = (slope @ basis) @ basis.T
    out = mu + (1. - gamma) * slope + gamma * projected
    if not np.isfinite(out).all():
        raise ArithmeticError('Finite full-gene response subspace predictions required')
    return out
