"""Identity-only planning for joint animal and perturbation holdout."""

from dataclasses import dataclass

import numpy as np

from .calibration import axis


@dataclass(frozen=True)
class JointHoldout:
    """Indices into the original biological population axis, in input order."""

    train_rows: np.ndarray
    query_rows: np.ndarray
    excluded_rows: np.ndarray


def plan_joint_holdout(source, target, heldout_sources, heldout_targets):
    """Query the intersection and exclude the union from training.

    Identity vectors refer to treated populations. The caller independently
    supplies and checks same-animal control halves and data admission. In a
    nested evaluation, call this function on outer-training identities only
    for each inner block and map the result back to the outer training axis.
    """
    source, target = axis(source, 'source'), axis(target, 'target')
    if source.shape != target.shape or len(set(zip(source, target))) != len(source):
        raise ValueError('Each input row must identify a unique animal and target pair')
    heldout_sources = axis(heldout_sources, 'heldout_sources')
    heldout_targets = axis(heldout_targets, 'heldout_targets')
    if not np.isin(heldout_sources, source).all() or not np.isin(heldout_targets, target).all():
        raise ValueError('Held-out identities must appear on the input axis')
    animal = np.isin(source, heldout_sources)
    perturbation = np.isin(target, heldout_targets)
    train = np.flatnonzero(~animal & ~perturbation)
    query = np.flatnonzero(animal & perturbation)
    if not train.size or not query.size:
        raise ValueError('Joint holdout must have both training and query rows')
    return JointHoldout(train, query, np.flatnonzero(animal ^ perturbation))
