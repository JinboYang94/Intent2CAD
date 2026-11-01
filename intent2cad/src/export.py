"""Export aligned variants using the frozen sample IDs and split assignment."""
from __future__ import annotations

import ast
from pathlib import Path


def read_splits(generation_root: Path) -> dict[str, list[str]]:
    splits, seen = {}, set()
    for split in ("train", "validation", "test"):
        ids = (generation_root / "splits" / f"{split}.txt").read_text().splitlines()
        if len(ids) != len(set(ids)) or seen & set(ids):
            raise ValueError("Duplicate IDs within or across data splits.")
        for identifier in ids:
            path = Path(identifier)
            if path.is_absolute() or len(path.parts) != 2 or not all(part.isdigit() for part in path.parts):
                raise ValueError("Invalid sample ID in the split list.")
        seen.update(ids)
        splits[split] = ids
    return splits


def write_sample(root: Path, split: str, identifier: str, programs: dict[str, str], text: str):
    if set(programs) != {"primitive", "semantic", "semparam"}:
        raise ValueError("All three representations are required.")
    for code in programs.values():
        ast.parse(code)
    paths = [(root / representation / split / "code" / f"{identifier}.py", code)
             for representation, code in programs.items()]
    paths.append((root / "text" / split / f"{identifier}.txt", text))
    if any(path.exists() for path, _ in paths):
        raise FileExistsError("The output sample already exists.")
    for path, content in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_splits(root: Path, splits: dict[str, list[str]]):
    target = root / "splits"
    target.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (target / f"{split}.txt").write_text("".join(sid + "\n" for sid in ids))
