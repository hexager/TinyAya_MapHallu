

import pandas as pd
from datasets import load_dataset
import gzip
import urllib.request
import json

# Languages written out in full for the instruction wrapper
LANG_NAMES = {
    "en": "English",
    "ar": "Arabic",
    "bn": "Bengali",
    "fr": "French",
    "zh": "Chinese",
    "hi": "Hindi",
    "sw": "Swahili",
    "de": "German",
    "tr": "Turkish",
    "ru": "Russian",
}


def load_mkqa(languages: list[str], n_samples: int = 100) -> pd.DataFrame:
    """
    Loads MKQA directly from Apple's GitHub release — bypasses the broken
    HuggingFace loading script entirely.
    """
    url = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
    print("[mkqa] Downloading from Apple GitHub...")

    rows = []
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as f:
            for line in f:
                item = json.loads(line.decode("utf-8"))
                row = {
                    "prompt_id": str(item["example_id"]),
                    "source":    "mkqa",
                }
                valid = True
                for lang in languages:
                    query = item["queries"].get(lang)
                    if not query:
                        valid = False
                        break
                    row[f"prompt_{lang}"] = query

                    answers = item["answers"].get(lang) or {}
                    aliases = answers.get("aliases") or []
                    row[f"answer_{lang}"] = aliases[0] if aliases else None

                if valid:
                    rows.append(row)
                if len(rows) >= n_samples:
                    break

    df = pd.DataFrame(rows)
    print(f"[mkqa] Loaded {len(df)} prompts with full coverage for: {languages}")
    return df


def load_hallomtbench(n_samples: int = 100) -> pd.DataFrame:
    """
    HalloMTBench: translation hallucination benchmark across 11 language directions.
    Ground truth is the reference translation — hallucination = faithfulness failure.

    Note: this dataset is English-source only. Cross-lingual prompts for other
    languages will need to be derived from the English source via load_mkqa,
    or by extending this loader later if HalloMTBench adds more source languages.

    TODO: confirm the HuggingFace dataset path before running
    """
    dataset = load_dataset("hallomtbench/hallomtbench", split="test")  # confirm path

    rows = []
    for i, item in enumerate(dataset.select(range(min(n_samples, len(dataset))))):
        rows.append({
            "prompt_id":           f"hmtb_{i}",
            "source":              "hallomtbench",
            "prompt_en":           item.get("source"),
            "answer_en":           item.get("reference"),
            "source_lang":         item.get("src_lang"),
            "target_lang":         item.get("tgt_lang"),
            "hallucination_label": item.get("label"),
        })

    df = pd.DataFrame(rows)
    print(f"[hallomtbench] Loaded {len(df)} prompts")
    return df


def build_prompt_for_language(base_prompt: str, lang: str) -> str:
    """
    Wraps a base prompt with a language instruction.

    The instruction is always in English. This is intentional — we want
    to isolate whether the model's *response* language affects hallucination,
    not whether it can understand instructions in different languages.
    Those are two different questions and mixing them would muddy the signal.
    """
    lang_name = LANG_NAMES.get(lang, lang)
    return f"Please answer the following question in {lang_name}:\n\n{base_prompt}"