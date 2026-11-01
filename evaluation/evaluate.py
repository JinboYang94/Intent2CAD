"""Score saved generation, editing or visual-editing predictions."""
from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing as mp
import os
import statistics
import tempfile
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent2cad.datasets import DATA_ROOT, REPRESENTATIONS, metadata_rows, relative_id


def load_samples(root, task, representation):
    root = Path(root)
    if task == 'generation':
        identifiers = (root / task / 'splits' / 'test.txt').read_text().splitlines()
        for identifier in identifiers:
            yield {'id': identifier, 'target': (root / task / representation / 'test' / 'code' / f'{relative_id(identifier)}.py').read_text()}
    else:
        metadata = metadata_rows(root / 'metadata' / f'{task}.tsv')
        index = [json.loads(line) for line in (root / task / 'index.jsonl').read_text().splitlines() if line.strip()]
        if len({row['id'] for row in index}) != len(index) or {row['id'] for row in index} != set(metadata):
            raise ValueError('Benchmark index and metadata must contain the same unique IDs.')
        for row in index:
            identifier = row['id']
            rel = relative_id(identifier)
            base = root / task / representation / 'test'
            yield {**row, **metadata[identifier], 'source': (base / 'src_code' / f'{rel}.py').read_text(),
                   'target': (base / 'tgt_code' / f'{rel}.py').read_text()}


def score_sample(sample, prediction, task, progress=None):
    """Use the original metric definitions and eligible-sample denominators."""
    from evaluation.metrics import generation as gen, editing as edit
    from evaluation.metrics.loops import parse_loop_reprs, reprs_equal
    prediction = gen.strip_pred_file_artifacts(prediction)
    if progress:
        progress('prediction', None)
    ok, result, error = gen.execute_cadquery(prediction)
    record = {'id': sample['id'], 'invalid': not ok, 'syntax_valid': gen.check_syntax(prediction)[0],
              'cd': None, 'execution_error': error}
    def finish():
        if ok:
            if progress:
                progress('reference', record)
            gt_ok, gt, gt_error = gen.execute_cadquery(sample['target'])
            if progress:
                progress('distance', record)
            distance = gen.compute_cd(gt, result) if gt_ok else {'cd': None, 'error': gt_error}
            record.update(cd=distance['cd'], cd_error=distance['error'])
        return record
    if task == 'generation':
        for label, function, key in [('ar', gen.compute_api_recovery_f1, 'api_recovery_f1'),
                                     ('shared_recovery', gen.compute_spf, 'spf'),
                                     ('derived_recovery', gen.compute_dpf, 'dpf')]:
            metric = function(sample['target'], prediction) if ok else {key: None, 'detail': 'skipped_ir_invalid'}
            record[label] = metric[key]
            record[f'{label}_detail'] = metric
        return finish()
    record.update({key: sample[key] for key in ('category', 'op_type', 'relation_subtype', 'touched_loops')})
    edit_result = edit.compute_edit_success(prediction, sample['target'], sample, ok)
    record['edit_success'] = bool(edit_result['success'])
    record['edit_error'] = edit_result.get('error', '')
    source, src_error = parse_loop_reprs(sample['source'])
    target, tgt_error = parse_loop_reprs(sample['target'])
    predicted, pred_error = parse_loop_reprs(prediction)
    comparable = (not src_error and not tgt_error and set(source) == set(target)
                  and all(source[n] is not None and target[n] is not None for n in source))
    affected = {n for n in source if not reprs_equal(source[n], target[n])} if comparable else set(edit.touched_loops_from_metadata(sample))
    untouched = set(source) - affected
    preserved = not pred_error and untouched <= predicted.keys() and all(reprs_equal(source[n], predicted[n]) for n in untouched)
    affected_match = not pred_error and affected <= predicted.keys() and affected <= target.keys() and all(reprs_equal(predicted[n], target[n]) for n in affected)
    loop_set_preserved = not pred_error and set(predicted) == set(target)
    record.update(benchmark_comparable=bool(comparable), untouched_preserved=bool(preserved),
                  strict_edit_success=bool(comparable and ok and preserved and affected_match and loop_set_preserved),
                  loop_set_preserved=bool(loop_set_preserved))
    if task == 'visual_editing':
        changed = {n for n in set(source) | set(predicted) if n not in source or n not in predicted or not reprs_equal(source.get(n), predicted.get(n))} if not src_error and not pred_error else set()
        record.update(target_localized=bool(prediction.strip()) and sample['hidden_target_loop'] in changed, n_loops=int(sample['n_loops']),
                      hidden_target_loop=sample['hidden_target_loop'], changed_loops=sorted(changed))
    return finish()


def _worker(sample, prediction, task, connection):
    try:
        with tempfile.TemporaryDirectory(prefix='intent2cad-eval-') as directory:
            os.chdir(directory)
            def progress(stage, partial):
                connection.send({'stage': stage, 'partial': partial})
            connection.send({'result': score_sample(sample, prediction, task, progress)})
    except BaseException as error:
        connection.send({'result': {'worker_error': f'{type(error).__name__}: {error}'}})
    finally:
        connection.close()


def score_isolated(sample, prediction, task):
    context = mp.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sample, prediction, task, send))
    process.start()
    send.close()
    stage, partial = 'startup', None
    try:
        while receive.poll(30):
            try:
                message = receive.recv()
            except EOFError:
                result = {'worker_error': 'CAD worker exited without a result'}
                break
            if 'result' in message:
                result = message['result']
                break
            stage, partial = message['stage'], message['partial']
        else:
            # A reference/CD timeout must not turn an executable prediction
            # into an IR failure. Each CAD execution has its own deadline.
            result = ({**partial, 'cd_error': f'{stage} timed out after 30 seconds'} if partial is not None
                      else {'worker_error': f'{stage} timed out after 30 seconds'})
    finally:
        process.join(.2)
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        receive.close()
    if 'worker_error' in result:
        # Infrastructure/metric errors must not masquerade as model failures.
        if 'timed out' not in result['worker_error'] and 'without a result' not in result['worker_error']:
            raise RuntimeError(result['worker_error'])
        result.update(id=sample['id'], invalid=True, cd=None, execution_error=result['worker_error'])
        try:
            ast.parse(prediction)
            result['syntax_valid'] = True
        except SyntaxError:
            result['syntax_valid'] = False
        if task == 'generation':
            result.update(ar=None, shared_recovery=None, derived_recovery=None)
        else:
            result.update({key: sample[key] for key in ('category', 'op_type', 'relation_subtype', 'touched_loops')})
            result.update(edit_success=False, untouched_preserved=False, strict_edit_success=False)
            if task == 'visual_editing':
                result.update(target_localized=False, n_loops=int(sample['n_loops']))
    return result


def summarize(records):
    result = {'n': len(records)}
    for key in ('invalid', 'syntax_valid', 'cd', 'ar', 'shared_recovery', 'derived_recovery',
                'edit_success', 'untouched_preserved', 'strict_edit_success', 'target_localized', 'pes'):
        values = [float(row[key]) for row in records if row.get(key) is not None and math.isfinite(float(row[key]))]
        if any(key in row for row in records):
            result[key] = {'mean': statistics.fmean(values) if values else None, 'count': len(values)}
            if key in ('cd', 'pes'):
                result[key]['median'] = statistics.median(values) if values else None
    return result


def read_pes(path):
    result = {}
    if path is None:
        return result
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rep = row['representation'].replace('semantic_parametric', 'semparam')
        identifier = row.get('sample_id', row.get('id'))
        key = (rep, identifier)
        score = float(row['score'])
        if rep not in REPRESENTATIONS or key in result or not 0 <= score <= 10:
            raise ValueError(f'Invalid or duplicate PES score: {key}')
        result[key] = score
    return result


def evaluate(root, task, predictions, output, representations=REPRESENTATIONS, pes_path=None):
    output, predictions = Path(output).resolve(), Path(predictions).resolve()
    root = Path(root).resolve()
    if output == root or root in output.parents or output == predictions or predictions in output.parents:
        raise ValueError('Evaluation output must be separate from the dataset and predictions.')
    output.mkdir(parents=True, exist_ok=False)
    pes = read_pes(pes_path)
    summaries = {}
    with (output / 'per_sample.jsonl').open('x') as handle:
        for rep in representations:
            records = []
            for sample in load_samples(root, task, rep):
                path = predictions / rep / f"{relative_id(sample['id'])}.py"
                missing = not path.is_file()
                record = score_isolated(sample, '' if missing else path.read_text(), task)
                record.update(representation=rep, prediction_missing=missing)
                if task == 'visual_editing':
                    record['pes'] = pes.get((rep, sample['id']))
                handle.write(json.dumps(record, allow_nan=False) + '\n')
                handle.flush()
                records.append(record)
            if not records:
                raise ValueError(f'No benchmark samples for {task}/{rep}.')
            summaries[rep] = {'overall': summarize(records)}
            for field in ('category', 'relation_subtype'):
                groups = sorted({row[field] for row in records if row.get(field)})
                if groups:
                    summaries[rep][f'by_{field}'] = {g: summarize([r for r in records if r.get(field) == g]) for g in groups}
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('generation', 'editing', 'visual_editing'))
    parser.add_argument('predictions', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--data', type=Path, default=DATA_ROOT)
    parser.add_argument('--representation', choices=REPRESENTATIONS)
    parser.add_argument('--pes', type=Path, help='Independent VLM judge scores (JSONL); no model is called by this evaluator.')
    args = parser.parse_args()
    reps = (args.representation,) if args.representation else REPRESENTATIONS
    print(json.dumps(evaluate(args.data, args.task, args.predictions, args.output, reps, args.pes), indent=2))


if __name__ == '__main__':
    main()
