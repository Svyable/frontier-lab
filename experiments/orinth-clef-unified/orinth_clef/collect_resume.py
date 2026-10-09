"""Incremental teacher collection with strict provenance and resume checks.

The output is a local JSONL log. Completed cases are validated against the
input before they may be skipped; newly acquired cases are flushed per row.
"""

import argparse
import json
import os
from pathlib import Path

from .data import collect_cases, read_jsonl
from .schema import SchemaError, validate_request, validate_response
from .teacher import LocalTeacher, TeacherError


def collect_incremental(cases: list[dict], teacher, *, revision: str, output: Path, max_new: int | None = None) -> dict:
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise SchemaError("expected full 40-character lowercase Hugging Face revision")
    case_by_id = {}
    for case in cases:
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in case_by_id:
            raise SchemaError("all input case IDs must be distinct nonempty strings")
        if not isinstance(case.get("group_id"), str) or not case["group_id"]:
            raise SchemaError("every case must have a group_id")
        validate_request(case.get("request"))
        case_by_id[case["id"]] = case

    existing = read_jsonl(output) if output.exists() else []
    completed = set()
    for row in existing:
        cid = row.get("id")
        if cid not in case_by_id or cid in completed:
            raise SchemaError(f"unknown or duplicate previously collected case: {cid}")
        case = case_by_id[cid]
        if (
            row.get("group_id") != case["group_id"]
            or row.get("request") != validate_request(case["request"])
            or row.get("teacher_revision") != revision
            or row.get("teacher_model") != "mlx-community/clef-flash-4bit"
        ):
            raise SchemaError(f"existing case {cid} differs from input, model or revision")
        validate_response(case["request"], row.get("response"))
        completed.add(cid)

    if max_new is not None and max_new < 1:
        raise ValueError("max_new must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    # Append rather than replacing a nonempty file; fail closed before writing.
    with output.open("a", encoding="utf-8") as stream:
        for case in cases:
            if case["id"] in completed:
                continue
            if max_new is not None and appended >= max_new:
                break
            row = collect_cases([case], teacher, revision=revision)[0]
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            appended += 1
    return {"existing": len(completed), "new": appended, "total": len(completed) + appended, "requested": len(cases)}


def main() -> int:
    p = argparse.ArgumentParser(description="Resume a pinned local Clef teacher data run")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--max-new", type=int)
    args = p.parse_args()
    try:
        results = collect_incremental(
            read_jsonl(args.cases), LocalTeacher(args.url),
            revision=args.revision, output=args.output, max_new=args.max_new,
        )
    except (SchemaError, TeacherError, OSError, ValueError, KeyError) as exc:
        p.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
