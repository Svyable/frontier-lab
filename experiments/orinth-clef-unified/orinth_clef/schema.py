"""Narrow, intentionally text-only SystemOne contract used by the first experiment.

Only choice and noul are supported in this phase. Unsupported request shapes fail
before any network call rather than silently mutating or truncating data.
"""

import json
import math
from collections.abc import Mapping


class SchemaError(ValueError):
    """A request or a teacher response violates the experiment contract."""


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SchemaError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise SchemaError(f"{name} must be finite and in [0,1]")
    return number


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def validate_request(request: object) -> dict:
    if not isinstance(request, dict):
        raise SchemaError("request must be an object")
    if any(key in request for key in ("images", "videos", "media_kwargs")):
        raise SchemaError("phase 1 is text-only; media inputs are not supported")
    if request.get("model", "clef-flash") != "clef-flash":
        raise SchemaError("phase 1 expects the clef-flash teacher model")
    if "state" not in request or request["state"] is None:
        raise SchemaError("state is required")
    if isinstance(request["state"], str) and not request["state"].strip():
        raise SchemaError("state cannot be blank")
    try:
        json.dumps(request["state"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError("state must be finite JSON-serializable data") from exc
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise SchemaError("questions must contain 1–32 named questions")
    for qid, q in questions.items():
        _nonblank(qid, "question ID")
        if not isinstance(q, dict):
            raise SchemaError(f"question {qid} must be an object")
        kind = q.get("type")
        if kind not in ("choice", "noul"):
            raise SchemaError(f"question {qid}: unsupported type {kind!r}; use choice or noul")
        if "instructions" in q:
            _nonblank(q["instructions"], f"question {qid} instructions")
        if kind == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 50:
                raise SchemaError(f"question {qid} requires 2–50 choice criteria")
            for option, description in criteria.items():
                _nonblank(option, f"{qid} choice ID")
                _nonblank(description, f"{qid}/{option} choice description")
        elif "criteria" in q:
            criteria = q["criteria"]
            if not isinstance(criteria, dict) or not set(criteria).issubset({"true", "false"}):
                raise SchemaError(f"question {qid}: noul criteria must describe true/false")
            for label in criteria.values():
                _nonblank(label, f"question {qid} noul criterion")
    return {**request, "model": "clef-flash"}


def validate_response(request: dict, response: object) -> dict:
    """Validate the teacher response without fabricating missing probabilities."""
    request = validate_request(request)
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise SchemaError("teacher response must have an answers mapping")
    answers = response["answers"]
    if set(answers) != set(request["questions"]):
        raise SchemaError("response question IDs differ from request")
    for qid, q in request["questions"].items():
        ans = answers[qid]
        if not isinstance(ans, Mapping) or ans.get("type") != q["type"]:
            raise SchemaError(f"{qid}: answer type must match question type")
        if q["type"] == "noul":
            _probability(ans.get("noul"), f"{qid} true probability")
            continue
        probabilities = ans.get("probabilities")
        expected = set(q["criteria"])
        if not isinstance(probabilities, Mapping) or set(probabilities) != expected:
            raise SchemaError(f"{qid}: probability keys must equal choice IDs")
        p = {key: _probability(val, f"{qid}/{key}") for key, val in probabilities.items()}
        if abs(sum(p.values()) - 1.0) > 0.015:
            raise SchemaError(f"{qid}: probabilities must sum to 1")
        if ans.get("choice") not in expected:
            raise SchemaError(f"{qid}: selected choice must match an option ID")
        if "confidence" in ans:
            _probability(ans["confidence"], f"{qid} confidence")
    return response


def pseudo_labels(request: dict, response: dict) -> dict:
    """Teacher predictions, NOT ground truth or calibrated student outputs."""
    validate_response(request, response)
    result = {}
    for qid, q in request["questions"].items():
        answer = response["answers"][qid]
        if q["type"] == "choice":
            # Preserve the teacher's selected option even when tied.
            result[qid] = answer["choice"]
        else:
            result[qid] = answer["noul"] >= 0.5
    return result
