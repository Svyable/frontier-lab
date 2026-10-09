"""Experimental reversible compact-label encoding for typed decision tasks.

Keep the original option ID in the semantic criterion description; score a
single-token code in the JSON answer and decode it back to the original ID.
This is an inference-time transform, not training or model compression.
"""
import copy
import string

from .decision_lens import decide
from .schema import SchemaError, validate_request


def compact_task(task, tokenizer):
    validate_request({"model": "clef-flash", **task})
    codes = string.ascii_uppercase + string.ascii_lowercase + string.digits
    # Avoid selecting aliases that collapse onto the same token.
    available, seen = [], set()
    for code in codes:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in seen:
            available.append(code)
            seen.add(ids[0])
    transformed = copy.deepcopy(task)
    mapping = {}
    for qid, question in transformed["questions"].items():
        if question["type"] != "choice":
            continue
        originals = sorted(question["criteria"])
        if len(originals) > len(available):
            raise SchemaError("insufficient distinct single-token aliases")
        aliases = available[:len(originals)]
        mapping[qid] = dict(zip(aliases, originals))
        question["criteria"] = {
            alias: f"Original option ID: {original!r}. {question['criteria'][original]}"
            for alias, original in zip(aliases, originals)
        }
        question["instructions"] = (
            question.get("instructions", "Select the correct option.") +
            " Choose by meaning of the original option ID and criterion; return its single-letter code."
        )
    return transformed, mapping


def decide_compact(model, tokenizer, task):
    compact, mapping = compact_task(task, tokenizer)
    raw, distributions = decide(model, tokenizer, compact, sequence_scorer="reference")
    decoded, decoded_scores = {}, {}
    for qid, value in raw.items():
        if qid in mapping:
            decoded[qid] = mapping[qid][value]
            decoded_scores[qid] = {
                mapping[qid][key]: score for key, score in distributions[qid].items()
            }
        else:
            decoded[qid] = value
            decoded_scores[qid] = distributions[qid]
    return decoded, decoded_scores
