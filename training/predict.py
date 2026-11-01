"""Generate programs from an adapter for the existing evaluation entry point.

Usage:
    python training/predict.py generation semparam
    python training/predict.py editing semparam
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from intent2cad.datasets import DATA_ROOT, REPRESENTATIONS, relative_id
from training.train import SYSTEM_PROMPT

EDIT_SYSTEM = (
    "You are a CadQuery code editor. "
    "Given an edit instruction and a source CadQuery program, output the full edited Python program only. "
    "Modify all and only the code needed to satisfy the instruction. "
)


def samples(data, task, representation):
    if task == "generation":
        for identifier in (data / task / "splits/test.txt").read_text().splitlines():
            rel = relative_id(identifier)
            text = (data / task / "text/test" / f"{rel}.txt").read_text().strip()
            yield rel, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    else:
        for line in (data / task / "index.jsonl").read_text().splitlines():
            rel = relative_id(json.loads(line)['id'])
            base = data / task / representation / "test"
            instruction = (base / "text" / f"{rel}.txt").read_text().strip()
            source = (base / "src_code" / f"{rel}.py").read_text().strip()
            prompt = (
                "Apply the requested edit intent to the source CadQuery program.\n"
                "Make only the necessary changes, and keep the code structure as original.\n"
                "Reason briefly only to locate the requested change and required dependent updates.\n"
                "Return only the full edited code, without explanations or markdown.\n\n"
                f"Edit instruction:\n{instruction}\n\nSource program:\n```python\n{source}\n```"
            )
            yield rel, [{"role": "system", "content": EDIT_SYSTEM}, {"role": "user", "content": prompt}]


def extract_code(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("generation", "editing"))
    parser.add_argument("representation", choices=REPRESENTATIONS)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint or ROOT / "checkpoints" / args.representation / "final"
    provenance = json.loads((checkpoint / "intent2cad.json").read_text())
    if provenance['representation'] != args.representation:
        raise ValueError("Adapter representation does not match the requested representation.")
    output = (args.output or ROOT / "predictions" / args.task) / args.representation
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        provenance['model'], dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, checkpoint).eval()
    output.mkdir(parents=True, exist_ok=False)
    for identifier, messages in samples(args.data, args.task, args.representation):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=4096, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        response = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        path = output / f"{identifier}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(extract_code(response))
        path.with_suffix('.response.txt').write_text(response)


if __name__ == '__main__':
    main()
