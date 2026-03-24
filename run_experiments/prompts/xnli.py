"""
prompts/xnli.py — Prompt template and response_format for the XNLI NLI task.

Asks the model to classify premise/hypothesis as entailment/neutral/contradiction
and return a JSON object ``{"label": "..."}``..
"""

XNLI_PROMPT_TEMPLATE = """Analyze the premise and hypothesis. \
Determine if the relationship is 'entailment', 'neutral', or 'contradiction'. \
You MUST output a valid JSON object with a single key 'label'.

Premise: {premise}
Hypothesis: {hypothesis}"""

XNLI_RESPONSE_FORMAT = {"type": "json_object"}


def build_xnli_prompt(premise: str, hypothesis: str) -> str:
    return XNLI_PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
