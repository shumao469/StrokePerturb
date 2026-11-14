"""Component calibration and scoring for animal and target holdouts."""

from .calibration import apply_calibration, calibrate_inner, direct_rows, score_outer
from .components import benchmark_component_bridge, decompose_training_bulk, split_training_residual
from .kernel import predict_linear_residual
from .splits import JointHoldout, plan_joint_holdout

__all__ = [
    'JointHoldout',
    'apply_calibration',
    'benchmark_component_bridge',
    'calibrate_inner',
    'decompose_training_bulk',
    'direct_rows',
    'plan_joint_holdout',
    'predict_linear_residual',
    'score_outer',
    'split_training_residual',
]
