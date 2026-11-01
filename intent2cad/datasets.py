"""Download Intent2CAD and restore the file layout used by synthesis/evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

REPRESENTATIONS = ('primitive', 'semantic', 'semparam')
SPLITS = ('train', 'validation', 'test')
DATA_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'benchmark'
REPOSITORY = 'scottyjb/Intent2CAD'


def relative_id(identifier):
    if not re.fullmatch(r'\d{4}[/_]\d{8}', identifier):
        raise ValueError(f'Invalid sample ID: {identifier!r}')
    return identifier.replace('_', '/')


def rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(value.encode('utf-8') if isinstance(value, str) else value)


def metadata_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        records = list(csv.DictReader(handle, delimiter='\t'))
    result = {row['id']: row for row in records}
    if len(result) != len(records):
        raise ValueError(f'Duplicate metadata IDs: {path}')
    return result


def export_snapshot(snapshot: Path, destination: Path):
    """Export into an empty staging directory; preserve bytes, IDs and order."""
    seen = set()
    counts = {}
    for split in SPLITS:
        ids = []
        for row in rows(snapshot / 'generation' / f'{split}.parquet'):
            identifier = relative_id(row['id'])
            if identifier in seen:
                raise ValueError(f'Duplicate generation ID: {identifier}')
            seen.add(identifier)
            ids.append(identifier)
            for rep in REPRESENTATIONS:
                write(destination / 'generation' / rep / split / 'code' / f'{identifier}.py', row[f'code_{rep}'])
            write(destination / 'generation' / 'text' / split / f'{identifier}.txt', row['text'])
        write(destination / 'generation' / 'splits' / f'{split}.txt', ''.join(i + '\n' for i in ids))
        counts[f'generation/{split}'] = len(ids)
    generation_test = set(ids)
    edit_ids = set()
    for task in ('editing', 'visual_editing'):
        meta_path = snapshot / 'metadata' / f'{task}.tsv'
        if task == 'visual_editing' and not (snapshot / task / 'test.parquet').exists():
            if meta_path.exists():
                raise ValueError('Visual metadata exists without the visual benchmark.')
            continue
        metadata = metadata_rows(meta_path)
        seen_task, index, inputs = set(), [], []
        for row in rows(snapshot / task / 'test.parquet'):
            identifier = row['id']
            rel = relative_id(identifier)
            if identifier in seen_task or identifier not in metadata:
                raise ValueError(f'Duplicate sample or missing metadata: {task}/{identifier}')
            if rel not in (generation_test if task == 'editing' else edit_ids):
                raise ValueError(f'Sample absent from parent benchmark: {task}/{identifier}')
            seen_task.add(identifier)
            if task == 'editing':
                edit_ids.add(rel)
            entry = {'id': identifier, 'category': row['category']}
            index.append(entry)
            for rep in REPRESENTATIONS:
                for field, folder in [('source', 'src_code'), ('target', 'tgt_code')]:
                    write(destination / task / rep / 'test' / folder / f'{rel}.py', row[f'{field}_{rep}'])
                write(destination / task / rep / 'test' / 'text' / f'{rel}.txt', row['instruction'])
            if task == 'visual_editing':
                image = row['image']['bytes']
                if not image:
                    raise ValueError(f'Missing embedded image: {identifier}')
                image_path = f'images/{identifier}.png'
                write(destination / task / image_path, image)
                inputs.append({'id': identifier, 'instruction': row['instruction'], 'image': image_path,
                               **{f'{rep}_code_path': f'{rep}/test/src_code/{rel}.py' for rep in REPRESENTATIONS}})
        if set(metadata) != seen_task:
            raise ValueError(f'Metadata does not match {task} samples.')
        write(destination / task / 'index.jsonl', ''.join(json.dumps(row) + '\n' for row in index))
        write(destination / 'metadata' / f'{task}.tsv', meta_path.read_bytes())
        if inputs:
            write(destination / task / 'vlm_inputs.jsonl', ''.join(json.dumps(row) + '\n' for row in inputs))
        counts[f'{task}/test'] = len(index)
    return counts


def prepare(destination=DATA_ROOT):
    from huggingface_hub import HfApi, snapshot_download
    destination = Path(destination).resolve()
    # Never merge with or overwrite an existing data directory.
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Use the prepared data or choose a new directory.')
    revision = HfApi().dataset_info(REPOSITORY).sha
    snapshot = Path(snapshot_download(REPOSITORY, repo_type='dataset', revision=revision,
                                     allow_patterns=['generation/*.parquet', 'editing/*.parquet',
                                                     'visual_editing/*.parquet', 'metadata/*.tsv']))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.intent2cad-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'benchmark'
        counts = export_snapshot(snapshot, staged)
        write(staged / 'dataset.json', json.dumps({'repository': REPOSITORY, 'revision': revision, 'counts': counts}, indent=2) + '\n')
        # Reserve the destination atomically. A concurrent preparation must
        # fail here rather than merge its files into another completed export.
        destination.mkdir(exist_ok=False)
        staged.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    print(prepare(args.output))


if __name__ == '__main__':
    main()
