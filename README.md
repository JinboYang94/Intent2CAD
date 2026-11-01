# Intent2CAD

**How Semantic–Parametric Supervision Shapes Text-to-CAD Generation**

Code and datasets for studying how CAD program representations affect text-to-CAD generation and intent-preserving editing.

[Dataset](https://huggingface.co/datasets/scottyjb/Intent2CAD) · [Installation](#installation) · [Data Synthesis](#data-synthesis) · [Evaluation](#evaluation)

## Overview

Intent2CAD constructs three aligned representations of the same CAD model: **primitive**, **semantic**, and **semantic–parametric** programs. The construction pipeline has two stages:

- **Semantic lifting:** recover high-level features from primitive geometry and source constraints.
- **Parametric lifting:** recover shared parameters and derived relations through local LLM-proposed rewrites, followed by execution and geometry checks.

![Intent2CAD construction pipeline](assets/framework.png)

*Figure 3 from the paper: constraint-grounded lifting, parameter rewriting, validation, and text generation.*

## Installation

Python 3.12 and CadQuery 2.7.0.

```bash
conda create -n intent2cad python=3.12
conda activate intent2cad
pip install -r requirements.txt
```

## Dataset

The [Intent2CAD dataset](https://huggingface.co/datasets/scottyjb/Intent2CAD) includes aligned programs and two benchmarks:

| Configuration | Splits | Contents |
| --- | --- | --- |
| `generation` | train / validation / test | Text descriptions and three program representations |
| `editing` | test | Source programs, edit instructions, target programs, and evaluation annotations |

Editing covers Feature Edit (FE), Consistency Edit (CE), and Derived-Relation Edit (DE).

```bash
hf auth login
python scripts/prepare_data.py
```

The dataset is currently private and requires an authorized Hugging Face account. The script prepares `data/benchmark/` with the published IDs and splits. It will not overwrite an existing directory.

## Data Synthesis

```bash
cp .env.example .env
```

Set these values in `.env`:

| Variable | Value |
| --- | --- |
| `OPENAI_API_KEY` | Your API key; both LLM tasks use GPT-5.4 |
| `CAD_JSON_ROOT` | DeepCAD construction records in JSON format, containing sketch geometry and modeling operations, stored as `<group>/<id>.json` |
| `CONSTRAINT_ROOT` | Sketch constraints collected through the Onshape API, stored as `<id>_*_featurescript_*.yml` or `<id>.json`. Subdirectories are searched recursively. |

```bash
python scripts/synthesize.py
```

Prepare the benchmark first to obtain the fixed split lists. The pipeline reconstructs programs, performs both lifting stages, generates descriptions, and builds editing triples. Outputs are written to a new `data/synthesized/` directory. Raw CAD records and constraint exports must be provided separately.

## Evaluation

Save predictions as `<prediction_directory>/<representation>/<group>/<id>.py`, where representation is `primitive`, `semantic`, or `semparam`.

```bash
python evaluation/evaluate.py generation predictions/generation results/generation
python evaluation/evaluate.py editing predictions/editing results/editing
python evaluation/analyze.py results/editing/per_sample.jsonl results/editing/analysis.json
```

Generation evaluation reports IR, CD, AR, ERS, and PRS. Editing evaluation reports edit success and unchanged-loop preservation, with CD as a geometric comparison. Each run writes per-sample results and a summary. Use `--representation semparam` to evaluate one representation; output directories must be new.

## Code Structure

```text
├── scripts/             # Data preparation and synthesis entry points
├── intent2cad/
│   ├── datasets.py      # Hugging Face download and local data layout
│   └── src/             # Reconstruction, lifting, validation, and edit synthesis
├── evaluation/
│   ├── evaluate.py      # Score saved predictions
│   ├── analyze.py       # Grouped results and paired comparisons
│   └── metrics/         # Generation, edit, and loop-preservation metrics
├── assets/              # Paper figures
└── data/                # Downloaded benchmarks and synthesis outputs (not tracked)
```

## Citation

If you use Intent2CAD in your research, please cite:

```bibtex
@misc{yang2026intent2cad,
  title  = {Intent2CAD: How Semantic--Parametric Supervision Shapes Text-to-CAD Generation},
  author = {Yang, Jinbo and Jing, Shikai and Yuan, Mingyue},
  year   = {2026},
  url    = {https://github.com/JinboYang94/Intent2CAD}
}
```

## Acknowledgements

CAD construction sequences are derived from [DeepCAD](https://github.com/rundiwu/DeepCAD). Programs use [CadQuery](https://github.com/CadQuery/cadquery).

## License

Code and data are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Third-party materials retain their original licenses.
