"""Command-line entry point for precomputed component arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import apply_calibration, calibrate_inner, plan_joint_holdout, score_outer
from .calibration import HEADS


def _demo():
    rng = np.random.default_rng(42)
    source = np.repeat(['demo_animal_' + str(i) for i in range(5)], 6)
    target = np.tile(['demo_gene_' + str(i) for i in range(6)], 5)
    genes = np.array(sorted(set(target)) + ['demo_marker'])
    split = plan_joint_holdout(source, target, ['demo_animal_3', 'demo_animal_4'],
                              ['demo_gene_3', 'demo_gene_4', 'demo_gene_5'])
    heads = {k: rng.normal(scale=.1, size=(len(source), len(genes))) for k in HEADS}
    truth = heads['background'] + .5 * heads['C'] + .75 * heads['W']
    truth += rng.normal(scale=.01, size=truth.shape)
    inner, outer = split.train_rows, split.query_rows
    calibration = calibrate_inner(truth[inner], {k: x[inner] for k, x in heads.items()},
                                  source[inner], target[inner], genes)
    scores, _ = score_outer(truth[outer], {k: x[outer] for k, x in heads.items()},
                           source[outer], target[outer], genes, calibration)
    return dict(dataset='synthetic precomputed components', inner_rows=len(inner),
                outer_rows=len(outer), excluded_rows=len(split.excluded_rows),
                selected_coefficients=scores['coefficients'],
                comparator=scores['strongest_family_frozen_from_inner'], scores=scores['scores'],
                independent_replay=scores['independent_replay']['passed'])


def _components(path, needs_truth):
    axes = ('source', 'target', 'genes')
    expected = set(axes) | set(HEADS)
    if needs_truth:
        expected.add('truth')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != expected:
            raise ValueError(f'Expected exactly these NPZ keys: {", ".join(sorted(expected))}')
        return ({k: data[k] for k in HEADS}, [data[k] for k in axes],
                data['truth'] if needs_truth else None)


def _json_output(value, path):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path is None:
        print(payload, end='')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command')
    commands.add_parser('demo', help='Run the deterministic synthetic example')
    calibrate = commands.add_parser('calibrate', help='Select weights from complete inner out-of-fold predictions')
    predict = commands.add_parser('predict', help='Apply frozen weights to query components')
    score = commands.add_parser('score', help='Score fixed forecasts on supplied outer outcomes')
    for command in (calibrate, predict, score):
        command.add_argument('components', type=Path, help='NPZ with source, target, genes and background, U, C, W')
        command.add_argument('output', type=Path, help='Output file (NPZ for predict, JSON otherwise)')
    for command in (predict, score):
        command.add_argument('--calibration', required=True, type=Path, help='Frozen calibration JSON')
    args = parser.parse_args()
    if args.command in (None, 'demo'):
        _json_output(_demo(), None)
        return
    heads, axes, truth = _components(args.components, needs_truth=args.command != 'predict')
    if args.command == 'calibrate':
        _json_output(calibrate_inner(truth, heads, *axes), args.output)
        return
    frozen = json.loads(args.calibration.read_text(encoding='utf-8'))
    if args.command == 'predict':
        source, target, genes = axes
        forecasts = apply_calibration(heads, *axes, frozen)
        with args.output.open('xb') as stream:
            np.savez_compressed(stream, **forecasts, source=source, target=target, genes=genes)
    else:
        report, _ = score_outer(truth, heads, *axes, frozen)
        _json_output(report, args.output)


if __name__ == '__main__':
    main()
