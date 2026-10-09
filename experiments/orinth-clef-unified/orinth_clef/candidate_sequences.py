"""Teacher-forced candidate sequence scoring with a shared MLX prompt cache.

This supports multi-token choice IDs without unconstrained JSON generation.
Sequence *mean* log-probability is a heuristic; not a calibrated event probability.
"""
import copy
import math

from .decision_lens import candidate_values, select_from_logits
from .schema import SchemaError


def candidate_sequences(tokenizer, question):
    result = {}
    for value in candidate_values(question):
        spelling = value if isinstance(value, str) else ("true" if value else "false")
        ids = tokenizer.encode(spelling, add_special_tokens=False)
        if not ids:
            raise SchemaError(f"empty candidate tokenization: {spelling!r}")
        result[value] = ids
    return result


def score_sequences(model, cache, next_logits, candidates):
    """Score all allowed sequences, branching from the same frozen prefix.

    The caller must have evaluated next_logits, so that cache contains the
    entire prefix. We never mutate that shared cache during branch scoring.
    """
    import mlx.core as mx

    if not candidates:
        raise SchemaError("no candidate sequences")
    mx.eval(next_logits)
    scores = []
    for ids in candidates.values():
        if not ids:
            raise SchemaError("empty candidate sequence")
        first = next_logits[ids[0]] - mx.logsumexp(next_logits)
        if len(ids) == 1:
            score = first
        else:
            # Deep-copy the populated MLX KV cache before every speculative
            # continuation. Each candidate receives the identical prefix.
            branch_cache = copy.deepcopy(cache)
            branch_logits = model(mx.array([ids[:-1]]), cache=branch_cache)[0]
            mx.eval(branch_logits)
            score = first
            for index, token_id in enumerate(ids[1:]):
                logits = branch_logits[index]
                score = score + logits[token_id] - mx.logsumexp(logits)
            score = score / len(ids)
        scores.append(float(score.item()))
    if not all(math.isfinite(x) for x in scores):
        raise SchemaError("nonfinite candidate log likelihood")
    return select_from_logits(candidates, scores)
