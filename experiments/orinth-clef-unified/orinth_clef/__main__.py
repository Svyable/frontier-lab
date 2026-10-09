"""No-third-party-dependency CLI for the teacher capture milestone."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .data import collect_cases, export_student_data, read_jsonl, write_jsonl
from .schema import SchemaError
from .teacher import LocalTeacher, TeacherError


def _doctor() -> int:
    machine = platform.machine()
    system = platform.system()
    details = {"os": system, "architecture": machine, "apple_silicon": system == "Darwin" and machine == "arm64"}
    if system == "Darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
            details["unified_memory_gb"] = round(int(output.strip()) / (1024 ** 3), 1)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            details["unified_memory_gb"] = "unknown"
    print(json.dumps(details, indent=2))
    return 0 if details["apple_silicon"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Orinth/Clef local teacher bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Apple Silicon and available unified memory")
    probe = sub.add_parser("probe", help="Run Clef health check and one known request")
    collect = sub.add_parser("collect", help="Capture real teacher predictions as JSONL")
    export = sub.add_parser("export", help="Export teacher pseudo-labels to MLX SFT chat JSONL")
    for command in (probe, collect):
        command.add_argument("--url", default="http://127.0.0.1:8001")
    collect.add_argument("--cases", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--revision", required=True, help="Exact Hugging Face snapshot git SHA")
    export.add_argument("--raw", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--seed", default="orinth-clef-v1")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "probe":
            teacher = LocalTeacher(args.url)
            print("Health:", json.dumps(teacher.health()))
            result = teacher.predict({
                "model": "clef-flash",
                "state": "Checkout errors are blocking all purchases.",
                "questions": {
                    "team": {"type": "choice", "criteria": {"billing": "Payments and invoices", "technical": "Software outages"}},
                    "urgent": {"type": "noul", "instructions": "Is immediate triage needed?"},
                },
            })
            print("Decision:", json.dumps(result, indent=2))
            return 0
        if args.command == "collect":
            cases = read_jsonl(args.cases)
            rows = collect_cases(cases, LocalTeacher(args.url), revision=args.revision)
            write_jsonl(args.output, rows)
            print(f"Saved {len(rows)} genuine teacher responses to {args.output}")
            return 0
        if args.command == "export":
            manifest = export_student_data(read_jsonl(args.raw), args.output, args.seed)
            print(json.dumps(manifest, indent=2))
            return 0
    except (SchemaError, TeacherError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
