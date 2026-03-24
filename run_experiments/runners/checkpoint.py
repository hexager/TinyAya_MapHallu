"""
runners/checkpoint.py — Two-level checkpoint: checkpoint.json + JSONL scan.
"""

import json
from pathlib import Path
from typing import Any


def _ck_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoint.json"


def load_checkpoint(output_dir: str | Path) -> dict:
    """Load existing checkpoint or return an empty one."""
    path = _ck_path(output_dir)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "in_progress": None}


def save_checkpoint(output_dir: str | Path, ck: dict) -> None:
    path = _ck_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ck, f, indent=2, ensure_ascii=False)


def _unit_key(dataset: str, model: str, language: str, experiment: str) -> dict:
    return {"dataset": dataset, "model": model, "language": language, "experiment": experiment}


def is_unit_completed(ck: dict, dataset: str, model: str, language: str, experiment: str) -> bool:
    key = _unit_key(dataset, model, language, experiment)
    for entry in ck.get("completed", []):
        if all(entry.get(k) == v for k, v in key.items()):
            return True
    return False


def mark_unit_completed(ck: dict, dataset: str, model: str, language: str, experiment: str, n: int) -> None:
    key = _unit_key(dataset, model, language, experiment)
    key["n_completed"] = n
    # Remove from completed if already there (shouldn't happen, but be safe)
    ck["completed"] = [
        e for e in ck.get("completed", [])
        if not all(e.get(k) == v for k, v in _unit_key(dataset, model, language, experiment).items())
    ]
    ck["completed"].append(key)
    ck["in_progress"] = None


def mark_in_progress(ck: dict, dataset: str, model: str, language: str, experiment: str, n: int) -> None:
    ck["in_progress"] = {**_unit_key(dataset, model, language, experiment), "n_completed": n}


def scan_completed_sample_ids(jsonl_path: Path) -> set[int]:
    """Scan an existing JSONL file and return the set of sample_ids already written."""
    done: set[int] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add(record["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def scan_completed_base_keys(jsonl_path: Path) -> set[str]:
    """Scan base JSONL and return set of 'sample_id|rep' keys already written."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rep = rec.get("rep", 0)
                done.add(f"{rec['sample_id']}|{rep}")
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def scan_completed_pss_keys(jsonl_path: Path) -> set[str]:
    """Scan PSS JSONL and return set of 'sample_id|variant_type|rep' keys already written."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rep = rec.get("rep", 0)
                done.add(f"{rec['sample_id']}|{rec['variant_type']}|{rep}")
            except (json.JSONDecodeError, KeyError):
                continue
    return done
