"""Experimental in-place KV-cache rollback scorer for Apple MLX.

Eliminates per-branch KV-array deepcopy by rewinding cache offsets after
fully evaluating each speculative branch. Requires mutable mlx_lm KVCache
with overwrite-on-append semantics; NOT thread-safe or generally portable.
"""
import math

from .decision_lens import select_from_logits
from .prefix_trie import build_trie
from .schema import SchemaError


def snapshot_offsets(cache):
    if not isinstance(cache, list) or not cache:
        raise SchemaError("rollback requires a nonempty list of MLX KVCache layers")
    if not all(type(getattr(layer, "offset", None)) is int and
               hasattr(layer, "keys") and hasattr(layer, "values")
               for layer in cache):
        raise SchemaError("rollback requires mutable offset/keys/values KVCache")
    return tuple(layer.offset for layer in cache)


def restore_offsets(cache, offsets):
    if len(cache) != len(offsets):
        raise SchemaError("cache layer count changed")
    for layer, offset in zip(cache, offsets):
        layer.offset = offset


def score_sequences_rollback(model, cache, next_logits, candidates):
    """Score a candidate trie by speculative in-place forward and rollback.

    All branch logits are materialized before rewinding. The original
    prefix offsets are restored even on exceptions. This implementation
    must be isolated per request: never share the cache across threads.
    """
    import mlx.core as mx

    root = build_trie(candidates)
    original_offsets = snapshot_offsets(cache)
    mx.eval(next_logits)
    scored = {}

    def visit(node, prefix_logits, accumulated, depth):
        for label in node["labels"]:
            scored[label] = accumulated / depth
        for token, child in node["children"].items():
            segment = [token]
            endpoint = child
            while not endpoint["labels"] and len(endpoint["children"]) == 1:
                next_token, endpoint = next(iter(endpoint["children"].items()))
                segment.append(next_token)
            needs_next = bool(endpoint["children"])
            forward_tokens = segment if needs_next else segment[:-1]
            checkpoint = snapshot_offsets(cache)
            try:
                output = None
                if forward_tokens:
                    output = model(mx.array([forward_tokens]), cache=cache)[0]
                    mx.eval(output)
                total = accumulated
                logits = prefix_logits
                for i, token_id in enumerate(segment):
                    total += float((logits[token_id] - mx.logsumexp(logits)).item())
                    if i + 1 < len(segment):
                        logits = output[i]
                if needs_next:
                    visit(endpoint, output[-1], total, depth + len(segment))
                else:
                    for label in endpoint["labels"]:
                        scored[label] = total / (depth + len(segment))
            finally:
                restore_offsets(cache, checkpoint)

    try:
        visit(root, next_logits, 0.0, 0)
    finally:
        restore_offsets(cache, original_offsets)
    if set(scored) != set(candidates) or not all(math.isfinite(x) for x in scored.values()):
        raise SchemaError("invalid rollback candidate scores")
    return select_from_logits(candidates, [scored[label] for label in candidates])
