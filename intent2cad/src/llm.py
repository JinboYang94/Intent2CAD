"""The two synthesis tasks share one explicit GPT-5.4 API client."""
from __future__ import annotations

import copy
import os

MODEL = "gpt-5.4"


def strict_schema(schema: dict) -> dict:
    """OpenAI strict objects declare all properties and reject extra keys."""
    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


class LLM:
    def __init__(self):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("Set OPENAI_API_KEY in .env before synthesizing data.")
        options = {"api_key": key, "timeout": 180, "max_retries": 2}
        if os.environ.get("OPENAI_BASE_URL", "").strip():
            options["base_url"] = os.environ["OPENAI_BASE_URL"].strip()
        self.client = OpenAI(**options)

    def generate(self, system: str, user: str, schema: dict | None = None) -> str:
        options = {}
        if schema is not None:
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "parameter_relations", "strict": True,
                    "schema": strict_schema(schema),
                },
            }
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            reasoning_effort="none", max_completion_tokens=4096, **options,
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop" or choice.message.refusal:
            raise ValueError("The model did not return a complete accepted response.")
        content = choice.message.content
        if not content or not content.strip():
            raise ValueError("The model returned an empty response.")
        return content.strip()
