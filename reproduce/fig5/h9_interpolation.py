"""Deterministic endpoint-only mean interpolation for the public GEO time course."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SHA256 = '057e015ed1b921f46c0abe486bbb2e865a6aa31d6fbcb4bde434707cbec5e063'
URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75748/suppl/GSE75748_sc_time_course_ec.csv.gz'
HOURS = (0, 12, 24, 36, 72, 96)
COUNTS = {0: 92, 12: 102, 24: 66, 36: 172, 72: 138, 96: 188}
ENDPOINT_HOURS = (0, 96)
MARKERS = ('POU5F1', 'NANOG', 'T', 'SOX17')


def load_columns(input_path, columns):
    admitted = {'Unnamed: 0', *columns}
    table = pd.read_csv(input_path, index_col=0, usecols=admitted, dtype={name: np.float64 for name in columns})
    array = table.to_numpy()
    assert np.isfinite(array).all() and (array >= 0).all()
    assert table.index.is_unique and table.columns.is_unique
    return table


def select_endpoints(input_path, names, times):
    cols = [name for name, t in zip(names, times) if t in ENDPOINT_HOURS]
    table = load_columns(input_path, cols)
    endpoint_log = np.log2(table.to_numpy() + 1)
    genes = np.asarray(table.index, dtype=str)
    variances = endpoint_log.var(axis=1)
    endogenous = np.flatnonzero(~np.char.startswith(genes, 'ERCC-'))
    rank = np.lexsort((genes[endogenous], -variances[endogenous]))
    selection = endogenous[rank[:256]]
    gene_ids = genes[selection]
    t = np.asarray([int(re.match(r'H9\.(\d+)h', name).group(1)) for name in table.columns])
    selected = endpoint_log[selection].T
    center = selected.mean(axis=0)
    _, singular, vt = np.linalg.svd(selected - center, full_matrices=False)
    loadings = vt[:2].copy()
    for row in loadings:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    explained = singular[:2] ** 2 / np.sum(singular ** 2)
    means = {h: selected[t == h].mean(axis=0) for h in ENDPOINT_HOURS}
    fitted = (selected - center) @ loadings.T
    return {
        'genes': gene_ids.tolist(),
        'variance': variances[selection].tolist(),
        'pca_center': center.tolist(),
        'pca_loadings': loadings.tolist(),
        'pca_explained': explained.tolist(),
        'means': {str(h): a.tolist() for h, a in means.items()},
        'prediction': {str(h): ((1 - h / 96) * means[0] + h / 96 * means[96]).tolist() for h in HOURS},
        'endpoint_cell_pca': {str(h): fitted[t == h].tolist() for h in ENDPOINT_HOURS},
    }


def summarize(input_path, frozen, names, times):
    table = load_columns(input_path, names)
    basis = np.asarray(frozen['pca_loadings'])
    offset = np.asarray(frozen['pca_center'])
    genes = frozen['genes']
    selected = np.log2(table.loc[genes].to_numpy().T + 1)
    counts = np.asarray(times)
    assert list(table.columns) == names and len(genes) == 256
    projected = (selected - offset) @ basis.T
    endpoints = {h: np.asarray(frozen['means'][str(h)]) for h in ENDPOINT_HOURS}
    conditions = []
    for h in HOURS:
        measured = selected[counts == h].mean(axis=0)
        pred = np.asarray(frozen['prediction'][str(h)])
        is_endpoint = h in ENDPOINT_HOURS
        if is_endpoint:
            np.testing.assert_array_equal(measured, endpoints[h])
            np.testing.assert_array_equal(measured, pred)
            np.testing.assert_allclose(projected[counts == h], frozen['endpoint_cell_pca'][str(h)], atol=1e-11)
        mae = {
            label: float(np.mean(np.abs(prediction - measured)))
            for label, prediction in (('interpolate', pred), ('carry_0h', endpoints[0]), ('carry_96h', endpoints[96]))
        } if not is_endpoint else None
        conditions.append({
            'hours': h, 'cells': COUNTS[h], 'endpoint': is_endpoint,
            'mean_measured': measured.tolist(), 'mean_interpolated': pred.tolist(),
            'pca': projected[counts == h].tolist(),
            'centroid_measured': ((measured - offset) @ basis.T).tolist(),
            'centroid_interpolated': ((pred - offset) @ basis.T).tolist(),
            'mae': mae,
        })
    m = table.loc[list(MARKERS)].to_numpy()
    marker_stats = []
    for i, symbol in enumerate(MARKERS):
        observed = np.log2(m[i] + 1)
        marker_means = {h: float(np.mean(observed[counts == h])) for h in HOURS}
        fractions = {h: float(np.mean(m[i, counts == h] > 0)) for h in HOURS}
        marker_stats.append({
            'symbol': symbol, 'mean': marker_means, 'fraction_positive': fractions,
            'interpolated': {h: (1 - h / 96) * marker_means[0] + h / 96 * marker_means[96] for h in HOURS},
        })
    return conditions, marker_stats, table.shape[0]


def main(args):
    with args.input.open('rb') as raw:
        if hashlib.file_digest(raw, 'sha256').hexdigest() != SHA256:
            raise ValueError('The GEO expression archive did not match the registered SHA-256')
    with gzip.open(args.input, 'rt', newline='') as handle:
        names = next(csv.reader(handle))[1:]
    times = [int(re.match(r'H9\.(\d+)h', name).group(1)) for name in names]
    assert Counter(times) == COUNTS
    # Finalize the selected genes, PCA model and predictions before reading
    # any intermediate values into the scoring array.
    frozen = select_endpoints(args.input, names, times)
    frozen_bytes = json.dumps(frozen, sort_keys=True, separators=(',', ':')).encode()
    args.endpoint_freeze.write_bytes(frozen_bytes)
    conditions, markers, total_genes = summarize(args.input, frozen, names, times)
    intermediates = [c for c in conditions if not c['endpoint']]
    equal_time_mae = {
        method: float(np.mean([c['mae'][method] for c in intermediates]))
        for method in ('interpolate', 'carry_0h', 'carry_96h')
    }
    out = {
        'source': {'accession': 'GSE75748', 'url': URL, 'archive_bytes': args.input.stat().st_size,
                   'sha256': SHA256, 'doi': '10.1186/s13059-016-1033-x',
                   'title': 'Single-cell RNA-seq reveals novel regulators of human embryonic stem cell differentiation to definitive endoderm'},
        'evidence_role': 'Public H9 in vitro differentiation time course; endpoint-only aggregate interpolation baseline, separate from StrokePerturb.',
        'frozen_rule': {'endpoint_hours': ENDPOINT_HOURS, 'intermediate_hours': HOURS[1:-1],
                        'transform': 'log2(1 + archived expected count)',
                        'feature_rule': '256 endogenous genes with highest population variances pooled across 0 h and 96 h cells; ERCC controls excluded by identifier; gene-symbol ascending tie-break',
                        'interpolation': 'For elapsed hour t, (1-t/96)*mean_0h + (t/96)*mean_96h',
                        'score': 'MAE across the frozen 256 genes in group means; summary equally weights the four intermediate times',
                        'endpoint_freeze_sha256': hashlib.sha256(frozen_bytes).hexdigest(),
                        'display_markers': MARKERS,
                        'pca': 'center and loadings fitted on 280 endpoint cells only; project the other 478 cells'},
        'total_genes': total_genes, 'all_cells': len(times), 'genes': frozen['genes'],
        'pca_explained': frozen['pca_explained'], 'conditions': conditions,
        'marker_stats': markers, 'equal_time_mae': equal_time_mae,
    }
    dst = args.output
    dst.write_text(json.dumps(out, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps({'genes': total_genes, 'counts': COUNTS, 'per_time_mae': {
        c['hours']: c['mae'] for c in intermediates}, 'equal_time_mae': equal_time_mae,
        'freeze_sha256': out['frozen_rule']['endpoint_freeze_sha256'],
        'output_sha256': hashlib.file_digest(dst.open('rb'), 'sha256').hexdigest()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--endpoint-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
