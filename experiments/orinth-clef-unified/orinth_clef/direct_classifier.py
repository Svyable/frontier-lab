"""Experimental question-local one-prefill classification without JSON generation.

A new prompt is evaluated for each question; compare against DecisionLens
on untouched future tasks before any default switch.
"""
import json
import string

from .decision_lens import select_from_logits
from .schema import SchemaError, validate_request


def direct_decide(model, tokenizer, task):
    import mlx.core as mx

    validate_request({"model": "clef-flash", **task})
    state = json.dumps(task["state"], sort_keys=True, ensure_ascii=False)
    answers, scores = {}, {}
    for qid, q in sorted(task["questions"].items()):
        options = sorted(q["criteria"]) if q["type"] == "choice" else [False, True]
        codes = string.ascii_uppercase + string.ascii_lowercase + string.digits
        if len(options) > len(codes):
            raise SchemaError("too many options for compact direct classification")
        labels = codes[:len(options)]
        encoded = [tokenizer.encode(c, add_special_tokens=False) for c in labels]
        if any(len(x) != 1 for x in encoded) or len({x[0] for x in encoded}) != len(encoded):
            raise SchemaError("aliases are not distinct single tokens")
        criteria = q.get("criteria", {})
        option_text = []
        for label, value in zip(labels, options):
            if isinstance(value, bool):
                description = criteria.get(str(value).lower(), str(value).lower())
            else:
                description = criteria[value]
            option_text.append(f"{label}: {value} — {description}")
        user = (
            f"STATE: {state}\nQUESTION {qid}: {q.get('instructions', 'Decide based on state')}\n"
            "OPTIONS:\n" + "\n".join(option_text) +
            "\nReply with exactly one option letter. No explanation."
        )
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a precise decision classifier."},
            {"role": "user", "content": user}
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        logits = model(mx.array([ids]))[0, -1, :]
        mx.eval(logits)
        raw = [float(logits[x[0]].item()) for x in encoded]
        label, distribution = select_from_logits(dict(zip(labels, options)), raw)
        answers[qid] = options[labels.index(label)]
        scores[qid] = {str(value): distribution[code] for code, value in zip(labels, options)}
    return answers, scores
