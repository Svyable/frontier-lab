"""Teacher data capture and leakage-resistant student SFT export."""

import hashlib
import json
from pathlib import Path

from .schema import SchemaError, pseudo_labels, validate_request, validate_response

SYSTEM_PROMPT = (
    "Return only compact JSON with a single 'decisions' object. "
    "For 'choice' questions return exactly one allowed option ID; "
    "for 'noul' questions return a JSON boolean. Do not add explanation."
)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(obj, dict):
            raise SchemaError(f"{path}:{line_no}: expected object")
        records.append(obj)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for obj in records:
            file.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def collect_cases(cases: list[dict], teacher, *, revision: str) -> list[dict]:
    """Collect real teacher responses; abort if any case or response is invalid."""
    if not revision.strip():
        raise ValueError("provide the downloaded Clef model revision for provenance")
    collected = []
    used = set()
    for case in cases:
        case_id = case.get("id")
        group_id = case.get("group_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in used:
            raise SchemaError("case IDs must be nonempty and unique")
        if not isinstance(group_id, str) or not group_id.strip():
            raise SchemaError(f"case {case_id} must have a nonempty group_id")
        request = validate_request(case.get("request"))
        response = teacher.predict(request)
        validate_response(request, response)
        collected.append({
            "id": case_id,
            "group_id": group_id,
            "source": case.get("source", "synthetic_seed"),
            "teacher_model": "mlx-community/clef-flash-4bit",
            "teacher_revision": revision,
            "request": request,
            "response": response,
            "pseudo_labels": pseudo_labels(request, response),
        })
        used.add(case_id)
    return collected


def to_student_record(row: dict) -> dict:
    request = validate_request(row["request"])
    response = validate_response(request, row["response"])
    labels = pseudo_labels(request, response)
    # Do not train the student on internal teacher metadata or evaluation gold.
    task = {"state": request["state"], "questions": request["questions"]}
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(task, ensure_ascii=False, sort_keys=True)},
        {"role": "assistant", "content": json.dumps({"decisions": labels}, ensure_ascii=False, sort_keys=True)},
    ]}


def _rank(group_id: str, seed: str) -> str:
    return hashlib.sha256((seed + "\x00" + group_id).encode()).hexdigest()


def split_rows(rows: list[dict], seed: str = "orinth-clef-v1") -> dict[str, list[dict]]:
    """Hold out whole scenario groups, deterministic across source ordering."""
    if not rows:
        raise SchemaError("no teacher samples")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise SchemaError("sample IDs must be non-empty and unique")
    groups = {row.get("group_id") for row in rows}
    if any(not isinstance(g, str) or not g for g in groups) or len(groups) < 3:
        raise SchemaError("need at least three independent nonempty scenario groups")
    ordered = sorted(groups, key=lambda group: (_rank(group, seed), group))
    held_out = max(1, len(groups) // 10)
    test = set(ordered[:held_out])
    valid = set(ordered[held_out:2 * held_out])
    out = {"train": [], "valid": [], "test": []}
    for row in sorted(rows, key=lambda record: record["id"]):
        subset = "test" if row["group_id"] in test else "valid" if row["group_id"] in valid else "train"
        out[subset].append(to_student_record(row))
    return out


def export_student_data(rows: list[dict], output: Path, seed: str = "orinth-clef-v1") -> dict:
    partitions = split_rows(rows, seed)
    for split, samples in partitions.items():
        write_jsonl(output / f"{split}.jsonl", samples)
    manifest = {
        "status": "teacher_pseudolabels_not_human_gold",
        "dataset_kind": "decision_label_sft_not_probability_distillation",
        "seed": seed,
        "counts": {split: len(values) for split, values in partitions.items()},
        "source_teacher_revisions": sorted({row["teacher_revision"] for row in rows}),
        "sample_count": len(rows),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
