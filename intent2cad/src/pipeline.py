"""One entry point for reconstruction, two-stage lifting, text, and edit synthesis."""
from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

from .cleaning import clean_file
from .constraints import extract_for_sample
from .editing import build_edit_dataset
from .export import read_splits, write_sample, write_splits
from .llm import LLM, MODEL
from .parametric import lift_relations
from .reconstruction import process_json_file
from .relations import summarize
from .semantic import lift_program
from .text_generation import describe_program
from .validation import execute_code, strip_exports

log = logging.getLogger(__name__)


def clean_target(code: str, working_root: Path, primitive: bool = False) -> str:
    import ast

    path = working_root / "to_clean.py"
    path.write_text(strip_exports(code))
    cleaned = clean_file(path)
    if primitive:
        class StripTags(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "tag":
                    return node.func.value
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != "tag"]
                return node
        tree = ast.fix_missing_locations(StripTags().visit(ast.parse(cleaned)))
        cleaned = ast.unparse(tree) + "\n"
    ast.parse(cleaned)
    return cleaned


def index_constraints(root: Path) -> dict[str, list[Path]]:
    index = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.yml', '.yaml', '.json'}:
            identifier = path.stem.split('_', 1)[0]
            if len(identifier) == 8 and identifier.isdigit():
                index.setdefault(identifier, []).append(path)
    return index


def synthesize(raw_root: Path, constraint_root: Path, release_root: Path, llm) -> Path:
    """Process retained IDs; no new deduplication or random data split is performed."""
    raw_root, constraint_root, release_root = (Path(p).resolve() for p in (raw_root, constraint_root, release_root))
    for path in (raw_root, constraint_root):
        if not path.is_dir():
            raise ValueError(f"Input directory does not exist: {path}")
    output = release_root / "data" / "synthesized"
    frozen = release_root / "data" / "generation"
    if not frozen.is_dir():
        frozen = release_root / "data" / "benchmark" / "generation"
    if not frozen.is_dir():
        raise ValueError('Prepare the released splits first: python scripts/prepare_data.py')
    splits = read_splits(frozen)
    if output.exists():
        raise FileExistsError("data/synthesized already exists; choose a fresh release copy to rebuild.")
    for input_root in (raw_root, constraint_root, frozen):
        if output == input_root or output in input_root.parents or input_root in output.parents:
            raise ValueError("Synthesis output must be separate from every input directory.")
    constraint_index = index_constraints(constraint_root)
    if not constraint_index:
        raise ValueError("CONSTRAINT_ROOT contains no feature exports named with an eight-digit sample ID.")
    output.mkdir(parents=True)
    working_root = output / "work"
    generation = output / "generation"
    summary_root = working_root / "relation_summaries"
    kept = {split: [] for split in splits}
    with (output / "status.jsonl").open("w") as statuses, (output / "synthesis.log").open("w") as detail:
        for split, identifiers in splits.items():
            for identifier in identifiers:
                status = {"id": identifier, "split": split, "model": MODEL}
                sample_root = working_root / identifier
                sample_root.mkdir(parents=True, exist_ok=True)
                try:
                    raw_path = raw_root / f"{identifier}.json"
                    if not raw_path.is_file():
                        raise FileNotFoundError("Raw CAD JSON is missing.")
                    features = constraint_index.get(Path(identifier).name, [])
                    if len(features) != 1:
                        raise ValueError("Missing or ambiguous local feature export.")
                    with contextlib.redirect_stdout(detail):
                        prefix = sample_root / "primitive"
                        if not process_json_file(raw_path, str(prefix)):
                            raise ValueError("Primitive reconstruction failed.")
                        primitive_path = prefix.with_suffix(".py")
                        primitive = strip_exports(primitive_path.read_text())
                        primitive_path.write_text(primitive)
                        constraints = extract_for_sample(raw_path, features[0], primitive_path)
                        semantic, explicit, stage_i = lift_program(primitive, constraints, sample_root)
                    parametric, stage_ii = lift_relations(explicit, constraints, llm, sample_root, stage_i['baseline'], source_code=primitive)
                    summary = summarize(parametric)
                    text = describe_program(parametric, summary, llm)
                    programs = {name: clean_target(code, sample_root, name == "primitive")
                                for name, code in [("primitive", primitive), ("semantic", semantic), ("semparam", parametric)]}
                    write_sample(generation, split, identifier, programs, text)
                    summary_path = summary_root / f"{identifier}.json"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    # Recompute after variable renumbering so edit metadata matches exported code.
                    summary_path.write_text(json.dumps(summarize(programs['semparam']), indent=2) + "\n")
                    (sample_root / "constraints.json").write_text(json.dumps(constraints, indent=2) + "\n")
                    (sample_root / "lifting.json").write_text(json.dumps({'stage_i': stage_i, 'stage_ii': stage_ii}, indent=2) + "\n")
                    kept[split].append(identifier)
                    status['status'] = 'exported'
                except Exception as error:
                    status.update(status='rejected', reason=f'{type(error).__name__}: {error}')
                    log.warning('%s: %s', identifier, status['reason'])
                statuses.write(json.dumps(status) + "\n")
                statuses.flush()
                detail.flush()
        write_splits(generation, kept)
    if not any(kept.values()):
        raise RuntimeError("No samples were exported; inspect data/synthesized/status.jsonl.")
    log.info("Generation synthesis complete; constructing deterministic editing triples.")
    build_edit_dataset(generation, summary_root, output / "editing")
    return output


def main():
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    missing = [name for name in ('CAD_JSON_ROOT', 'CONSTRAINT_ROOT', 'OPENAI_API_KEY')
               if not os.environ.get(name, '').strip()]
    if missing:
        raise SystemExit("Fill these entries in .env: " + ", ".join(missing))
    try:
        result = synthesize(Path(os.environ['CAD_JSON_ROOT']), Path(os.environ['CONSTRAINT_ROOT']), root, LLM())
    except (ValueError, FileExistsError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
    log.info("Synthesis saved under %s", result.relative_to(root))
