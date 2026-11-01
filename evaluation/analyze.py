"""Grouped metrics and paired bootstrap comparisons from evaluator output."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import binomtest

from evaluation.evaluate import summarize
from intent2cad.datasets import REPRESENTATIONS


def paired(records):
    by_rep = {}
    for row in records:
        group = by_rep.setdefault(row['representation'], {})
        if row['id'] in group:
            raise ValueError('Duplicate ID within a representation.')
        group[row['id']] = row
    results = []
    for a, b in itertools.combinations([r for r in REPRESENTATIONS if r in by_rep], 2):
        if by_rep[a].keys() != by_rep[b].keys():
            raise ValueError('Paired comparisons require identical benchmark IDs.')
        for metric in ('invalid', 'cd', 'ar', 'shared_recovery', 'derived_recovery',
                       'edit_success', 'strict_edit_success', 'target_localized', 'pes'):
            ids = [i for i in by_rep[a] if by_rep[a][i].get(metric) is not None and by_rep[b][i].get(metric) is not None]
            if not ids:
                continue
            va = np.array([by_rep[a][i][metric] for i in ids], dtype=float)
            vb = np.array([by_rep[b][i][metric] for i in ids], dtype=float)
            delta = vb - va
            rng = np.random.default_rng(20260727)
            # Bounded memory for the full generation test set.
            boot = np.concatenate([delta[rng.integers(0, len(ids), size=(100, len(ids)))].mean(axis=1) for _ in range(100)])
            row = {'metric': metric, 'comparison': f'{b} minus {a}', 'paired_n': len(ids),
                   'mean_delta': float(delta.mean()), 'ci95': np.quantile(boot, [0.025, 0.975]).tolist()}
            if metric in ('invalid', 'edit_success', 'strict_edit_success', 'target_localized'):
                a_only = int(((va == 1) & (vb == 0)).sum())
                b_only = int(((va == 0) & (vb == 1)).sum())
                row['mcnemar_exact_p'] = float(binomtest(min(a_only, b_only), a_only + b_only, .5).pvalue) if a_only + b_only else 1.0
            results.append(row)
    return results


def analyze(records):
    result = {'paired': paired(records), 'groups': {}}
    for row in records:
        if 'touched_loops' in row:
            count = len([x for x in row['touched_loops'].split(',') if x])
            row['coordination_load'] = str(count) if count < 3 else '3+'
            op = row.get('op_type', '')
            row['feature_edit_group'] = 'parameter' if op.startswith('param:') else 'replacement' if op.startswith('replace:') else ''
        if 'n_loops' in row:
            row['loop_complexity'] = '2' if row['n_loops'] <= 2 else '3-4' if row['n_loops'] <= 4 else '5+'
    for field in ('category', 'relation_subtype', 'feature_edit_group', 'coordination_load', 'loop_complexity'):
        groups = sorted({row[field] for row in records if row.get(field)})
        if groups:
            result['groups'][field] = {
                group: {rep: summarize([r for r in records if r.get(field) == group and r['representation'] == rep])
                        for rep in REPRESENTATIONS if any(r['representation'] == rep for r in records)}
                for group in groups}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='per_sample.jsonl written by evaluate')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('No evaluation records.')
    report = analyze(records)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
